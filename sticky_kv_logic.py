import torch
from torch import nn
import math
from transformers.models.llama.modeling_llama import rotate_half


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(bsz, tgt_len, past_len, dtype, device):
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_len > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_len, dtype=dtype, device=device), mask], dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_len)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
    sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class STICKYKVCache_LayerWise(nn.Module):
    def __init__(
        self,
        p_ratio,
        r_ratio,
        start_idx,
        num_heads,
        layer_idx,
        config=None,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        super().__init__()

        self.total_cache_ratio = r_ratio
        self.local_cache_ratio = p_ratio
        self.k_windows = 3
        self.start_idx = start_idx
        
        from sticky_config import OMEGA, SINK_TOKENS

        self.omega = OMEGA
        self.sink_tokens = SINK_TOKENS
            
        # Force observation window to always equal OMEGA chunk size
        self.tokens_since_last_review = 0
            
        # Support either percentage or fixed local token count
        try:
            from sticky_config import LOCAL_NUM_TOKENS
            self.local_num_tokens = LOCAL_NUM_TOKENS
            self.use_fixed_local_tokens = True
        except ImportError:
            # Fallback if config is missing or not supplied
            if config is not None and hasattr(config, "local_num_tokens"):
                self.local_num_tokens = config.local_num_tokens
                self.use_fixed_local_tokens = True
            else:
                self.local_num_tokens = 0
                self.use_fixed_local_tokens = False
            
        self.local_num = 0
        self.k_seq_dim, self.v_seq_dim = k_seq_dim, v_seq_dim
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.gen_step = 0
        self.num_of_tokens_without_eviction = 0
        self.prompt_boundary = [-1 for _ in range(self.num_heads)]
        self._prefill_done = False  # Tracks whether initial prefill has completed
        try:
            from sticky_config import tracking_flag
            self.tracking_flag = (tracking_flag == 1)
        except ImportError:
            self.tracking_flag = getattr(config, "tracking_flag", 1) == 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config is not None and hasattr(config, "max_position_embeddings"):
            max_context = config.max_position_embeddings
            max_windows = (
                ((max_context - self.sink_tokens) // self.omega) + 1 if max_context > self.sink_tokens else 1
            )
            max_windows = max(max_windows, 100)
        else:
            max_context = 8192
            max_windows = 10000

        window_ids = torch.arange(max_windows, device=device)
        token_map = (window_ids.unsqueeze(1) * self.omega + self.sink_tokens) + torch.arange(
            self.omega, device=device
        )

        self.register_buffer("window_to_token_map", token_map)
        self.register_buffer("sink_indices", torch.arange(0, self.sink_tokens, device=device))
        # Channels: [0]=CumMag, [1]=HitCount, [2]=Logical_ID, [3]=FinalScore
        self.register_buffer(
            "window_scores",
            torch.full(
                (self.num_heads, max_windows, 4),
                float("nan"),
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "head_indices", torch.arange(self.num_heads, device=device)
        )
        
        # Accumulates 1D attention votes from generated tokens over OMEGA steps
        # Max context size is enough to track physically alive tokens
        self.register_buffer(
            "running_attention_votes",
            torch.zeros((self.num_heads, max_context), dtype=torch.float32, device=device)
        )

        # Tracks arrival order for the entire context
        self.register_buffer("global_token_counter", torch.tensor(0, dtype=torch.long, device=device))

        # The Ledger: [Global_ID, Layer_ID, Phys_id_Head0, Phys_id_Head1, ..., Score_Head0, Score_Head1, ...]
        # Size: 2 + num_heads (for physical indices) + num_heads (for scores) = 2 + 2 * num_heads
        if self.tracking_flag:
            self.register_buffer(
                "token_ledger",
                torch.full(
                    (max_context, 2 + 2 * self.num_heads),
                    -1.0,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        else:
            self.token_ledger = None

        # Local History Buffer: preserves per-window (cum_mag, hit_count) for windows
        # that were protected by the local bubble and later re-enter eviction contention.
        self.register_buffer(
            "local_history",
            torch.zeros((self.num_heads, max_windows, 2), dtype=torch.float32, device=device),
        )

        # Optional: High-resolution 2D history for research (Global_ID x Heads)
        if self.tracking_flag:
            self.register_buffer(
                "global_score_history",
                torch.full((max_context, num_heads), -1.0, dtype=torch.float32, device=device),
            )
        else:
            self.global_score_history = None

        # Optional: Full NxN prefill matrix for rigorous research comparison
        # We initialize it as empty and allocate on demand to save memory if not used
        self.prefill_attention_matrix = None

        self.cache_size = int(
            self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens 
        )

    def _compute_window_metrics(self, attn_slice, current_seq_len, full_query_mode=False):
        """
        Computes Temporal Hit Count (THC) signals for a set of windows.
        
        Prefill (full_query_mode=True):
            attn_slice:  [num_heads, N, candidate_tokens] — full 2D attention matrix
            Returns: (cum_mag, hit_count)
        
        Generation (full_query_mode=False):
            attn_slice:  [num_heads, candidate_tokens] — accumulated 1D votes
            Returns: (cum_mag, hit_count)
        """
        target_device = self.window_scores.device
        num_windows = attn_slice.shape[-1] // self.omega
        
        # Threshold: uniform random chance of attending to this window
        threshold = self.omega / max(1.0, float(current_seq_len))

        if full_query_mode:
            # --- 2D path: prefill with full N×N matrix ---
            N = attn_slice.shape[1]
            attn_2d = attn_slice.view(self.num_heads, N, num_windows, self.omega)
            
            # [heads, N, num_windows]
            attn_per_q_win = attn_2d.sum(dim=3)
            
            # Cumulative Magnitude: sum over all queries
            cum_mag = attn_per_q_win.sum(dim=1).to(device=target_device, dtype=torch.float32)
            
            # Hit Count: how many queries attended to this window above threshold?
            hit_count = (attn_per_q_win > threshold).sum(dim=1).to(device=target_device, dtype=torch.float32)

        else:
            # --- 1D path: generation with accumulated votes ---
            attn_1d = attn_slice.view(self.num_heads, num_windows, self.omega)
            
            # Cumulative Magnitude: sum of accumulated votes for this window
            cum_mag = attn_1d.sum(dim=2).to(device=target_device, dtype=torch.float32)
            
            # Hit Count: count how many individual token positions within this
            # window received above-threshold accumulated attention over OMEGA steps.
            # Per-token expected attention under uniform = (1/seq_len) * OMEGA = threshold,
            # so we reuse the same threshold for individual token positions.
            hit_count = (attn_1d > threshold).sum(dim=2).to(device=target_device, dtype=torch.float32)

        return cum_mag, hit_count

        

    def __call__(self, past_key_values, attn_score_cache, full_attn_scores=None):
        bsz, q_heads, q_len, kv_seq_len = attn_score_cache.shape
        
        num_new_tokens = q_len

        # FIX: Define seq_len immediately to avoid ReferenceError in arrival loop
        seq_len = past_key_values[0].size(self.k_seq_dim) if past_key_values is not None else 0
        # === Inside __call__ ===
        global_start = self.global_token_counter.item()
        
        # --- LEDGER REGISTRATION ---
        # After prefill, HF's prepare_inputs_for_generation may re-send old tokens
        # because the evicted cache is smaller than total tokens seen.
        # Only the LAST token in each batch is truly new during generation.
        if not self._prefill_done:
            # Initial prefill: all tokens are genuinely new
            num_new = q_len
            if self.tracking_flag:
                for i in range(num_new):
                    g_id = global_start + i
                    if g_id < self.token_ledger.shape[0]:
                        self.token_ledger[g_id, 0] = float(g_id)
                        self.token_ledger[g_id, 1] = float(self.layer_idx)
                        # All heads start with the same initial chronological physical index
                        phys_idx = float((seq_len if past_key_values else 0) + i)
                        self.token_ledger[g_id, 2:2+self.num_heads] = phys_idx
            self.global_token_counter += num_new
        else:

            # Generation phase: only 1 truly new token per step
            g_id = global_start
            if self.tracking_flag and g_id < self.token_ledger.shape[0]:
                self.token_ledger[g_id, 0] = float(g_id)
                self.token_ledger[g_id, 1] = float(self.layer_idx)
                # The new token is appended at the end of the sequence for all heads
                phys_idx = float(seq_len - 1)
                self.token_ledger[g_id, 2:2+self.num_heads] = phys_idx
            self.global_token_counter += 1
        
        if past_key_values is None:
            return past_key_values

        seq_len = past_key_values[0].size(self.k_seq_dim)
        num_new_tokens = q_len

        if num_new_tokens > 1:
            self._update_k_win_and_local_num(num_new_tokens, 64)
            self.cache_size = (
                self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens
            )
            self.num_of_tokens_without_eviction += seq_len
            for h in range(self.num_heads):
                self.prompt_boundary[h] = seq_len - 1
        else:
            self.num_of_tokens_without_eviction += 1
            self.gen_step += 1

        # Early return bypassed to maintain OMEGA synchronization

        if num_new_tokens > 1:  # Prompt Stage
            # Determine local token count precisely
            local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else (self.local_num * self.omega)
            
            # Application boundary: exclude sinks and the local bubble from eviction candidates
            score_end = max(self.sink_tokens, seq_len - local_tokens_count)
            num_windows = max(0, (score_end - self.sink_tokens) // self.omega)

            # Initialize local-history cache from the full prefill attention.
            # This is later used to seed history for windows that transition from
            # the local protected region into eviction contention.
            self.local_history.zero_()
            total_prompt_windows = max(0, (seq_len - self.omega - self.sink_tokens) // self.omega)
            if total_prompt_windows > 0:
                full_review_end = self.sink_tokens + total_prompt_windows * self.omega
                actual_full_review = min(full_review_end, attn_score_cache.shape[3])
                total_prompt_windows = (actual_full_review - self.sink_tokens) // self.omega
                actual_full_review = self.sink_tokens + total_prompt_windows * self.omega

                if total_prompt_windows > 0:
                    full_scores_slice = attn_score_cache[0, :, :seq_len, self.sink_tokens:actual_full_review]
                    full_mag, full_hits = self._compute_window_metrics(
                        full_scores_slice, seq_len, full_query_mode=True
                    )
                    idx_full = (
                        torch.arange(total_prompt_windows, device=self.local_history.device)
                        .unsqueeze(0)
                        .expand(self.num_heads, -1)
                    )

                    self.local_history[self.head_indices.unsqueeze(1), idx_full, 0] = full_mag
                    self.local_history[self.head_indices.unsqueeze(1), idx_full, 1] = full_hits

            # THC prompt-stage scoring over the eligible region only
            self.window_scores.fill_(float("nan"))
            if num_windows > 0:
                review_end = self.sink_tokens + num_windows * self.omega
                actual_review_end = min(review_end, attn_score_cache.shape[3])
                # Re-calculate exact number of windows based on what's ACTUALLY available
                num_windows = (actual_review_end - self.sink_tokens) // self.omega
                actual_review_end = self.sink_tokens + num_windows * self.omega

                if num_windows > 0:
                    full_scores_slice = attn_score_cache[0, :, :seq_len, self.sink_tokens:actual_review_end]
                    full_mag, full_hits = self._compute_window_metrics(
                        full_scores_slice, seq_len, full_query_mode=True
                    )
                    idx = torch.arange(num_windows, device=self.window_scores.device).unsqueeze(0).expand(self.num_heads, -1)
                    self.window_scores[self.head_indices.unsqueeze(1), idx, 0] = full_mag  # CumMag
                    self.window_scores[self.head_indices.unsqueeze(1), idx, 1] = full_hits  # HitCount
                    self.window_scores[self.head_indices.unsqueeze(1), idx, 2] = idx.float()  # Logical ID

            self._evict_from_window_scores()

            # --- CAPTURE SCORES BEFORE EVICTION ---
            if self.tracking_flag:
                # FIX: Sum across ALL prefill queries for accurate ledger tracking
                importance_map = attn_score_cache[0, :, :seq_len, :].sum(dim=1)
                # --- CAPTURE FULL PREFILL MATRIX (DEFERRED) ---
                # We will capture it AFTER eviction to see what's actually kept.
                full_scores_ref = full_attn_scores if full_attn_scores is not None else attn_score_cache

                # Map scores to Global IDs using their PRE-EVICTION positions
                active_mask = (self.token_ledger[:, 2] >= 0) & (self.token_ledger[:, 0] >= 0)
                active_g_ids = torch.where(active_mask)[0]
                for g_id in active_g_ids:
                    pre_eviction_phys_idx = self.token_ledger[g_id, 2].long()
                    if pre_eviction_phys_idx < importance_map.shape[1]:
                        self.token_ledger[g_id, 2+self.num_heads:2+2*self.num_heads] = importance_map[:, pre_eviction_phys_idx]
                        self.global_score_history[g_id, :] = importance_map[:, pre_eviction_phys_idx]

            # --- 1. GET SURVIVOR MAP ---
            updated_kv, survivor_ids = self._create_mask_and_evict_from_kv_cache_prompt_stage(
                past_key_values, attn_score_cache
            )

            if self.tracking_flag:
                # --- CAPTURE POST-EVICTION FULL MATRIX ---
                raw_matrix = full_scores_ref[0].detach().cpu()
                num_q_heads_total = raw_matrix.shape[0]
                group_size = num_q_heads_total // self.num_heads
                sparse_matrix = torch.zeros_like(raw_matrix)

                try:
                    for kv_h in range(self.num_heads):
                        kv_survivors = survivor_ids[kv_h].cpu().long()
                        kv_survivors = torch.unique(torch.clamp(kv_survivors, 0, seq_len - 1))
                        q_start = kv_h * group_size
                        q_end = q_start + group_size
                        sparse_matrix[q_start:q_end, :, kv_survivors] = raw_matrix[q_start:q_end, :, kv_survivors]
                except IndexError as e:
                    print(f"IndexError details ---> raw_matrix: {raw_matrix.shape}, sparse_matrix: {sparse_matrix.shape}")
                    print(f"survivor_ids shape: {survivor_ids.shape}")
                    raise e

                self.prefill_attention_matrix = sparse_matrix

                full_importance = attn_score_cache[0, :, :seq_len, :].sum(dim=1)
                self.token_ledger[:, 2:2+self.num_heads] = -1.0
                for head_idx in range(self.num_heads):
                    clean_survivors = survivor_ids[head_idx].to(torch.long)
                    for phys_idx, g_id in enumerate(clean_survivors):
                        if g_id >= 0 and g_id < self.token_ledger.shape[0]:
                            self.token_ledger[g_id, 2 + head_idx] = float(phys_idx)
                    for g_id in clean_survivors:
                        if g_id >= 0 and g_id < self.token_ledger.shape[0] and g_id < full_importance.shape[1]:
                            self.token_ledger[g_id, 2 + self.num_heads + head_idx] = full_importance[head_idx, g_id]

            self._prefill_done = True  # Mark prefill as complete
            return updated_kv

        else:  # Generation Stage
            device = self.window_scores.device
            
            # 1. ACCUMULATE VOTES
            self.running_attention_votes[:, :seq_len] += attn_score_cache[0, :, 0, :seq_len]
            self.tokens_since_last_review += 1
            
            # --- LEDGER: Update Scores for Live Tokens ---
            if self.tracking_flag:
                live_mask = self.token_ledger[:, 2] >= 0
                live_g_ids = torch.where(live_mask)[0]
                for head_idx in range(self.num_heads):
                    phys_indices = self.token_ledger[live_g_ids, 2 + head_idx].long()
                    valid_mask = phys_indices >= 0
                    valid_phys = phys_indices[valid_mask]
                    valid_g_ids = live_g_ids[valid_mask]
                    if len(valid_phys) > 0 and valid_phys.max() < seq_len:
                        head_scores = attn_score_cache[0, head_idx, 0, valid_phys]
                        self.token_ledger[valid_g_ids, 2 + self.num_heads + head_idx] = head_scores.float()
                        self.global_score_history[valid_g_ids, head_idx] = head_scores.float()
            
            # 2. PERIODIC EVALUATION
            if self.tokens_since_last_review == self.omega:
                local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else (self.local_num * self.omega)
                num_competing_windows = max(1, (seq_len - self.sink_tokens - local_tokens_count) // self.omega)
                review_end = self.sink_tokens + num_competing_windows * self.omega
                
                actual_review_end = min(review_end, self.running_attention_votes.shape[1])
                num_competing_windows = (actual_review_end - self.sink_tokens) // self.omega
                actual_review_end = self.sink_tokens + num_competing_windows * self.omega
                
                # --- THC: Compute fresh signals for ALL competing windows ---
                if num_competing_windows > 0:
                    # Accumulated attention votes (1D)
                    scores_slice = self.running_attention_votes[:, self.sink_tokens:actual_review_end]
                    
                    fresh_mag, fresh_hits = self._compute_window_metrics(scores_slice, seq_len, full_query_mode=False)
                
                # Retrieve logical ids (using min bound because Vanilla has extra k_windows space)
                num_old_windows = num_competing_windows - 1
                valid_old_windows = min(self.k_windows, num_old_windows)
                
                old_ids = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 2], nan=0.0)
                
                # The logically dropping window is perfectly offset by the observation sequence and local bubble
                last_id_val = (self.num_of_tokens_without_eviction - 2 * self.omega - self.sink_tokens - local_tokens_count) // self.omega
                last_id_tensor = torch.full((self.num_heads, 1), float(max(0, last_id_val)), device=device, dtype=torch.float32)
                
                # Build competing_ids to exactly match num_competing_windows
                competing_ids = torch.cat([old_ids, last_id_tensor], dim=1) # [heads, valid_old + 1]
                
                # Build historical signals for the old windows
                hist_mag = torch.zeros((self.num_heads, num_competing_windows), device=device, dtype=torch.float32)
                hist_hits = torch.zeros((self.num_heads, num_competing_windows), device=device, dtype=torch.float32)
                
                if valid_old_windows > 0:
                    hist_mag[:, :valid_old_windows] = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 0], nan=0.0)
                    hist_hits[:, :valid_old_windows] = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 1], nan=0.0)

                # Local history injection:
                # When one new local window transitions into contention, seed its
                # historical (cum_mag, hit_count) from the accumulated local_history.
                if 0 <= last_id_val < self.local_history.shape[1] and num_competing_windows > valid_old_windows:
                    hist_mag[:, valid_old_windows] = self.local_history[:, last_id_val, 0]
                    hist_hits[:, valid_old_windows] = self.local_history[:, last_id_val, 1]
                    # Consume transferred history so buffer tracks only currently local windows.
                    self.local_history[:, last_id_val, :] = 0.0
                
                # Align to tracked windows
                num_with_ids = competing_ids.shape[1]
                if num_with_ids < num_competing_windows:
                    fresh_mag = fresh_mag[:, :num_with_ids]
                    fresh_hits = fresh_hits[:, :num_with_ids]
                    hist_mag = hist_mag[:, :num_with_ids]
                    hist_hits = hist_hits[:, :num_with_ids]
                    num_competing_windows = num_with_ids
                
                # --- Temporal Hit Count Fusion ---
                total_mag = fresh_mag + hist_mag
                total_hits = fresh_hits + hist_hits
                
                # Final Score = M_cum * log2(1.0 + H)
                thc_scores = total_mag * torch.log2(1.0 + total_hits)
                
                # Determine survivors (Top-K by THC)
                curr_k = min(self.k_windows, num_competing_windows)
                _, top_i = torch.topk(thc_scores, curr_k, dim=1, largest=True)
                
                surviving_ids = torch.gather(competing_ids, 1, top_i)
                sort_idx = torch.argsort(surviving_ids, dim=1)
                
                # Gather the accumulated signals for survivors
                surv_mag = torch.gather(total_mag, 1, top_i)
                surv_hits = torch.gather(total_hits, 1, top_i)
                surv_thc = torch.gather(thc_scores, 1, top_i)
                final_ids = torch.gather(surviving_ids, 1, sort_idx)
                
                self.window_scores.fill_(float("nan"))
                self.window_scores[:, :curr_k, 0] = torch.gather(surv_mag, 1, sort_idx)
                self.window_scores[:, :curr_k, 1] = torch.gather(surv_hits, 1, sort_idx)
                self.window_scores[:, :curr_k, 2] = final_ids
                self.window_scores[:, :curr_k, 3] = torch.gather(surv_thc, 1, sort_idx)

                # Update local history with omega-token scores for windows that
                # remain inside the local protected bubble during this review.
                local_start = self.sink_tokens + num_competing_windows * self.omega
                local_tokens_eff = seq_len - local_start
                local_windows = local_tokens_eff // self.omega
                if local_windows > 0:
                    local_window_start = local_start
                    local_slice = self.running_attention_votes[
                        :, local_window_start : local_window_start + local_windows * self.omega
                    ]
                    fresh_local_mag, fresh_local_hits = self._compute_window_metrics(
                        local_slice, seq_len, full_query_mode=False
                    )
                    local_id_start = (local_window_start - self.sink_tokens) // self.omega
                    ids = torch.arange(
                        local_id_start, local_id_start + local_windows, device=device, dtype=torch.long
                    )
                    valid = (ids >= 0) & (ids < self.local_history.shape[1])
                    if valid.any():
                        ids_valid = ids[valid]
                        self.local_history[:, ids_valid, 0] += fresh_local_mag[:, valid]
                        self.local_history[:, ids_valid, 1] += fresh_local_hits[:, valid]
                
                # If r_ratio is 100, we skip physical eviction and ledger shifting
                if self.total_cache_ratio == 100:
                    self.running_attention_votes.zero_()
                    self.tokens_since_last_review = 0
                    return past_key_values
                
                # PHYSICAL EVICTION
                head_win_tokens = []
                for b in range(self.omega):
                    head_win_tokens.append(self.sink_tokens + top_i * self.omega + b)
                # Stack to [heads, curr_k, omega] -> reshape to [heads, curr_k * omega]
                head_win_tokens = torch.stack(head_win_tokens, dim=-1).view(self.num_heads, -1)
                
                # We sort the physical tokens so they stay chronological
                head_win_tokens, _ = torch.sort(head_win_tokens, dim=1)
                
                sinks = self.sink_indices.unsqueeze(0).expand(self.num_heads, -1)
                local_start = self.sink_tokens + num_competing_windows * self.omega
                local_tokens = torch.arange(local_start, seq_len, device=device).unsqueeze(0).expand(self.num_heads, -1)
                
                all_indices = torch.cat([sinks, head_win_tokens, local_tokens], dim=1)
                
                # Gather physical KV cache
                head_dim = past_key_values[0].shape[-1]
                gather_idx = torch.clamp(all_indices, 0, seq_len - 1).unsqueeze(-1).expand(-1, -1, head_dim)
                k_kept = torch.gather(past_key_values[0][0], 1, gather_idx).unsqueeze(0)
                v_kept = torch.gather(past_key_values[1][0], 1, gather_idx).unsqueeze(0)
                updated_kv = (k_kept, v_kept)
                
                # Universal ledger shift for generic tools - Shift correctly per head
                if self.tracking_flag:
                    for head_idx in range(self.num_heads):
                        phys_col = 2 + head_idx
                        mask_evict = (self.token_ledger[:, phys_col] >= local_start - self.omega) & (self.token_ledger[:, phys_col] < local_start)
                        mask_shift = self.token_ledger[:, phys_col] >= local_start
                        self.token_ledger[mask_evict, phys_col] = -1.0
                        self.token_ledger[mask_shift, phys_col] -= self.omega

                # Reset accumulator
                self.running_attention_votes.zero_()
                self.tokens_since_last_review = 0
                
                return updated_kv
            else:
                return past_key_values
            
    def get_ledger_data(self):
        """
        Retrieves the tri-axial tracking data for research analysis.
        Returns a dictionary containing only the processed tokens.
        """
        # 1. Identify tokens that have entered the system
        total_processed = self.global_token_counter.item()
        
        if not self.tracking_flag:
            empty_long = torch.empty((0,), dtype=torch.long)
            empty_pos = torch.empty((0, self.num_heads), dtype=torch.long)
            empty_scores = torch.empty((0, self.num_heads), dtype=torch.float32)
            return {
                "global_id": empty_long,
                "layer_id": empty_long,
                "physical_id": empty_pos,
                "attention_score": empty_scores,
            }

        # 2. Slice the ledger to exclude unused pre-allocated buffer space
        active_ledger = self.token_ledger[:total_processed].detach().cpu()
        
        # 3. Extract the columns for clarity
        global_ids = active_ledger[:, 0].long()
        layer_ids = active_ledger[:, 1].long()
        physical_positions = active_ledger[:, 2:2+self.num_heads].long()
        attention_scores = active_ledger[:, 2+self.num_heads:2+2*self.num_heads]
        
        return {
            "global_id": global_ids,
            "layer_id": layer_ids,
            "physical_id": physical_positions, # [num_tokens, num_heads]
            "attention_score": attention_scores # [num_tokens, num_heads]
        }

    def _update_window_scores_generation_vectorized(self, attn_scores, local_id, orig_id):
        pass

    def _evict_from_window_scores(self):
        """Evict lowest-THC windows down to k_windows survivors."""
        valid_mask = ~torch.isnan(self.window_scores[:, :, 2])  # Check logical_id channel
        
        # Provide clean 0.0s to avoid NaN explosions
        mag_vals = torch.nan_to_num(self.window_scores[:, :, 0], nan=0.0)
        hit_vals = torch.nan_to_num(self.window_scores[:, :, 1], nan=0.0)
        ids = self.window_scores[:, :, 2]
        
        curr_k = min(self.k_windows, int(valid_mask.sum(dim=1).max().item()))
        
        # Compute THC scores cleanly
        raw_thc = mag_vals * torch.log2(1.0 + hit_vals)
        
        # Manually force invalid padding windows to -inf so they unambiguously lose
        neg_inf = torch.tensor(float("-inf"), device=self.window_scores.device)
        thc_scores = torch.where(valid_mask, raw_thc, neg_inf)
        
        _, top_i = torch.topk(thc_scores, curr_k, dim=1, largest=True)
        
        kept_mag = torch.gather(mag_vals, 1, top_i)
        kept_hits = torch.gather(hit_vals, 1, top_i)
        kept_ids = torch.gather(ids, 1, top_i)
        kept_thc = torch.gather(thc_scores, 1, top_i)
        
        sort_idx = torch.argsort(kept_ids, dim=1)
        self.window_scores.fill_(float("nan"))
        self.window_scores[:, :curr_k, 0] = torch.gather(kept_mag, 1, sort_idx)
        self.window_scores[:, :curr_k, 1] = torch.gather(kept_hits, 1, sort_idx)
        self.window_scores[:, :curr_k, 2] = torch.gather(kept_ids, 1, sort_idx)
        self.window_scores[:, :curr_k, 3] = torch.gather(kept_thc, 1, sort_idx)
        return []

    def _create_mask_and_evict_from_kv_cache_generation_stage(self, past_key_values, evicted_windows):
        seq_len, head_dim = (
            past_key_values[0].size(self.k_seq_dim),
            past_key_values[0].shape[-1],
        )
        
        # Calculate exactly which relative physical indices to keep.
        # Format: Sinks (5) -> Sticky (k_windows * omega) -> Local + Trailing
        # The oldest local window is immediately after the sticky windows.
        start_drop = self.sink_tokens + self.k_windows * self.omega
        end_drop = start_drop + self.omega
        
        keep_sinks_and_sticky = torch.arange(
            0, start_drop, device=past_key_values[0].device
        )
        keep_recent_local = torch.arange(
            end_drop, seq_len, device=past_key_values[0].device
        )
        
        keep_indices = torch.cat([keep_sinks_and_sticky, keep_recent_local])
        
        # Use index_select to preserve the tensor shape across all other dims
        k_kept = torch.index_select(past_key_values[0], self.k_seq_dim, keep_indices)
        v_kept = torch.index_select(past_key_values[1], self.k_seq_dim, keep_indices)
        
        # Keep the logging ledger physically mapped to the compacted space (per-head):
        if self.tracking_flag:
            for head_idx in range(self.num_heads):
                phys_col = 2 + head_idx
                mask_evict = (self.token_ledger[:, phys_col] >= start_drop) & (self.token_ledger[:, phys_col] < end_drop)
                mask_shift = self.token_ledger[:, phys_col] >= end_drop
                self.token_ledger[mask_evict, phys_col] = -1.0
                self.token_ledger[mask_shift, phys_col] -= self.omega
        
        return (k_kept, v_kept), keep_indices.unsqueeze(0)

    def _create_mask_and_evict_from_kv_cache_prompt_stage(self, past_key_values, attn_scores):
        seq_len, head_dim = (
            past_key_values[0].size(self.k_seq_dim),
            past_key_values[0].shape[-1],
        )
        
        num_w, remainder = (seq_len - self.sink_tokens) // self.omega, (seq_len - self.sink_tokens) % self.omega
        sticky_w = torch.nan_to_num(
            self.window_scores[:, : self.k_windows, 2], nan=0.0
        ).long()
        
        local_w = (
            torch.arange(
                max(0, num_w - self.local_num),
                max(max(0, num_w - self.local_num), num_w - 1),
                device=self.window_scores.device,
            )
            .unsqueeze(0)
            .expand(self.num_heads, -1)
            if self.local_num > 1
            else torch.empty(
                (self.num_heads, 0), dtype=torch.long, device=self.window_scores.device
            )
        )
        
        all_w = torch.cat(
            [
                sticky_w,
                local_w,
                torch.full(
                    (self.num_heads, 1),
                    num_w - 1,
                    device=self.window_scores.device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
        window_tokens = self.window_to_token_map[all_w].view(self.num_heads, -1)
        sinks = self.sink_indices.unsqueeze(0).expand(self.num_heads, -1)
        
        if remainder > 0:
            all_indices = torch.cat(
                [
                    sinks,
                    window_tokens,
                    torch.arange(seq_len - remainder, seq_len, device=self.window_scores.device)
                    .unsqueeze(0)
                    .expand(self.num_heads, -1),
                ],
                dim=1,
            )
        else:
            all_indices = torch.cat([sinks, window_tokens], dim=1)
            
        # Optional: ensure unique and properly ordered (not strictly required here depending on generation assumptions, but safe metric)
        # Using simple clamping for gather safety
        all_indices_clamped = torch.clamp(all_indices, 0, seq_len - 1)
        
        # WE MUST SORT AND DEDUPLICATE PER HEAD so the physical KV cache stays chronological
        # and so that the ledger mapping correctly aligns with the physical tensor dimensions.
        sorted_indices = []
        for h in range(self.num_heads):
            # deduplicate and sort
            unique = torch.unique(all_indices_clamped[h])
            sorted_indices.append(unique)
            
        # Due to dynamic deduplication, some heads might have 1 more or less token depending 
        # on overlap between sinks/local/alpha. We pad to the max length across heads for tensor compat.
        max_len = max(len(u) for u in sorted_indices)
        padded_indices = []
        for h in range(self.num_heads):
            u = sorted_indices[h]
            if len(u) < max_len:
                # pad by repeating the last valid token
                pad = u[-1:].expand(max_len - len(u))
                u = torch.cat([u, pad])
            padded_indices.append(u)
            
        final_indices = torch.stack(padded_indices, dim=0) # [heads, max_len]
        
        gather_idx = (
            final_indices
            .unsqueeze(-1)
            .expand(-1, -1, head_dim)
        )
        return (
            torch.gather(past_key_values[0][0], 1, gather_idx).unsqueeze(0),
            torch.gather(past_key_values[1][0], 1, gather_idx).unsqueeze(0),
        ), final_indices
    def _update_k_win_and_local_num(self, new_tokens, max_tokens):
        total_w = (
            (new_tokens + max_tokens) * self.total_cache_ratio // 100
        ) // self.omega
        
        if self.use_fixed_local_tokens:
            target_local = math.ceil(self.local_num_tokens / self.omega)
        else:
            target_local = (total_w * self.local_cache_ratio) // 100
            
        if total_w <= target_local:
            self.local_num = total_w
            self.k_windows = 0 
        else:
            self.local_num = target_local
            self.k_windows = total_w - self.local_num
        self.k_windows = max(0, self.k_windows)

    def _clean_scores(self):
        self.gen_step = self.num_of_tokens_without_eviction = 0
        self.tokens_since_last_review = 0
        if hasattr(self, "running_attention_votes"):
            self.running_attention_votes.zero_()
        self.window_scores.fill_(float("nan"))