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
        self.alpha = self.omega
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
        token_map = (window_ids.unsqueeze(1) * self.omega + 5) + torch.arange(
            self.omega, device=device
        )

        self.register_buffer("window_to_token_map", token_map)
        self.register_buffer("sink_indices", torch.arange(0, 5, device=device))
        self.register_buffer(
            "window_scores",
            torch.full(
                (self.num_heads, max_windows, 3),
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
        self.register_buffer("global_token_counter", torch.tensor(0, dtype=torch.long))

        # The Ledger: [Global_ID, Layer_ID, Phys_id_Head0, Phys_id_Head1, ..., Score_Head0, Score_Head1, ...]
        # Size: 2 + num_heads (for physical indices) + num_heads (for scores) = 2 + 2 * num_heads
        self.register_buffer("token_ledger", 
                            torch.full((max_context, 2 + 2 * self.num_heads), -1.0, dtype=torch.float32))

        # Optional: High-resolution 2D history for research (Global_ID x Heads)
        self.register_buffer("global_score_history", 
                            torch.full((max_context, num_heads), -1.0, dtype=torch.float32))

        # Optional: Full NxN prefill matrix for rigorous research comparison
        # We initialize it as empty and allocate on demand to save memory if not used
        self.prefill_attention_matrix = None

        self.cache_size = int(
            self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + 5 
        )
        

    def __call__(self, past_key_values, attn_score_cache, full_attn_scores=None):
        bsz, q_heads, q_len, kv_seq_len = attn_score_cache.shape
        
        num_new_tokens = q_len

        # FIX: Define seq_len immediately to avoid ReferenceError in arrival loop
        seq_len = past_key_values[0].size(self.k_seq_dim) if past_key_values is not None else 0
        # === Inside __call__ ===
        global_start = self.global_token_counter.item()
        
        # --- LEDGER REGISTRATION ---
        if not self._prefill_done:
            num_new = q_len
            self.global_token_counter += num_new
            if self.tracking_flag:
                for i in range(num_new):
                    g_id = global_start + i
                    if g_id < self.token_ledger.shape[0]:
                        self.token_ledger[g_id, 0] = float(g_id)
                        self.token_ledger[g_id, 1] = float(self.layer_idx)
                        phys_idx = float((seq_len if past_key_values else 0) + i)
                        self.token_ledger[g_id, 2:2+self.num_heads] = phys_idx
        else:
            self.global_token_counter += 1
            if self.tracking_flag:
                g_id = global_start
                if g_id < self.token_ledger.shape[0]:
                    self.token_ledger[g_id, 0] = float(g_id)
                    self.token_ledger[g_id, 1] = float(self.layer_idx)
                    phys_idx = float(seq_len - 1)
                    self.token_ledger[g_id, 2:2+self.num_heads] = phys_idx
        
        if past_key_values is None:
            return past_key_values

        seq_len = past_key_values[0].size(self.k_seq_dim)
        num_new_tokens = q_len

        if num_new_tokens > 1:
            self._update_k_win_and_local_num(num_new_tokens, 64)
            self.cache_size = (
                self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + 5
            )
            self.num_of_tokens_without_eviction += seq_len
            for h in range(self.num_heads):
                self.prompt_boundary[h] = seq_len - 1
        else:
            self.num_of_tokens_without_eviction += 1
            self.gen_step += 1

        # Early return bypassed to maintain OMEGA synchronization

        if num_new_tokens > 1:  # Prompt Stage
            # --- OMEGA OBSERVATION WINDOW SCORING ---
            # Observation Window: The final OMEGA tokens of the prefill sequence strictly exclude Sinks
            ob_start = max(5, seq_len - self.omega)
            ob_end = seq_len - 1
            
            # Determine local token count precisely
            local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else (self.local_num * self.omega)
            
            # Application boundary: Exclude Sinks [0:5] and Local+Observation windows
            score_end = max(5, seq_len - local_tokens_count - self.omega)
            num_windows = max(0, (score_end - self.sink_tokens) // self.omega)
            
            if num_windows > 0:
                review_end = self.sink_tokens + num_windows * self.omega
                # Ensure we don't slice past the actual dimension bounds
                actual_review_end = min(review_end, attn_score_cache.shape[3])
                # Re-calculate exact number of windows based on what's ACTUALLY available
                num_windows = (actual_review_end - self.sink_tokens) // self.omega
                actual_review_end = self.sink_tokens + num_windows * self.omega
                
                if num_windows > 0:
                    # FIX: Use full NxN prefill attention (all seq_len queries) instead of only last OMEGA
                    scores_slice = attn_score_cache[0, :, :seq_len, self.sink_tokens:actual_review_end]
                    obs_sum = scores_slice.sum(dim=1)
                    win_scores = obs_sum.view(self.num_heads, num_windows, self.omega).sum(dim=2).to(dtype=torch.float32)

                    idx = torch.arange(num_windows, device=self.window_scores.device).unsqueeze(0).expand(self.num_heads, -1)
                    self.window_scores[self.head_indices.unsqueeze(1), idx, 0] = win_scores
                    self.window_scores[self.head_indices.unsqueeze(1), idx, 1] = idx.float()
                    self.window_scores[self.head_indices.unsqueeze(1), idx, 2] = idx.float()

            self._evict_from_window_scores()

            # --- CAPTURE SCORES BEFORE EVICTION ---
            if self.tracking_flag:
                importance_map = attn_score_cache[0, :, :seq_len, :].sum(dim=1)
                active_mask = (self.token_ledger[:, 2] >= 0) & (self.token_ledger[:, 0] >= 0)
                active_g_ids = torch.where(active_mask)[0]
                for g_id in active_g_ids:
                    pre_eviction_phys_idx = self.token_ledger[g_id, 2].long()
                    if pre_eviction_phys_idx < importance_map.shape[1]:
                        self.token_ledger[g_id, 2+self.num_heads:2+2*self.num_heads] = importance_map[:, pre_eviction_phys_idx]
                        self.global_score_history[g_id, :] = importance_map[:, pre_eviction_phys_idx]

            # --- 1. GET SURVIVOR MAP ---
            updated_kv, survivor_ids = self._create_mask_and_evict_from_kv_cache_prompt_stage(
                past_key_values, ob_start, ob_end, attn_score_cache
            )

            if self.tracking_flag:
                full_scores_ref = full_attn_scores if full_attn_scores is not None else attn_score_cache
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
                
                if num_competing_windows > 0:
                    scores_slice = self.running_attention_votes[:, self.sink_tokens:actual_review_end]
                    win_scores = scores_slice.view(self.num_heads, num_competing_windows, self.omega).sum(dim=2).to(dtype=torch.float32)
                
                # Retrieve logical ids (using min bound because Vanilla has extra k_windows space)
                num_old_windows = num_competing_windows - 1
                valid_old_windows = min(self.k_windows, num_old_windows)
                
                old_ids = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 1], nan=0.0)
                last_id_val = (self.num_of_tokens_without_eviction - 2 * self.omega - self.sink_tokens - local_tokens_count) // self.omega
                last_id_tensor = torch.full((self.num_heads, 1), float(max(0, last_id_val)), device=device, dtype=torch.float32)
                
                # Build competing_ids to exactly match num_competing_windows
                competing_ids = torch.cat([old_ids, last_id_tensor], dim=1) # [heads, valid_old + 1]
                
                # Build competing_hist to exactly match num_competing_windows
                # Start with zeros for ALL competing windows
                competing_hist = torch.zeros((self.num_heads, num_competing_windows), device=device, dtype=torch.float32)
                # Place old scores at the correct positions (first valid_old_windows entries)
                if valid_old_windows > 0:
                    old_scores = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 0], nan=0.0)
                    competing_hist[:, :valid_old_windows] = old_scores
                # New window (last entry in competing_ids) gets 0 history — already zero
                
                # win_scores covers ALL num_competing_windows from the slice
                # competing_hist also has num_competing_windows entries
                # But competing_ids only has valid_old_windows + 1 entries
                # We need to align: take only the windows that have IDs
                num_with_ids = competing_ids.shape[1]
                if num_with_ids < num_competing_windows:
                    # More windows in the slice than we track — take only the tracked subset
                    win_scores = win_scores[:, :num_with_ids]
                    competing_hist = competing_hist[:, :num_with_ids]
                    num_competing_windows = num_with_ids
                
                total_win_scores = win_scores + competing_hist # [heads, num_competing_windows]
                
                # Determine survivors (Top-K)
                curr_k = min(self.k_windows, num_competing_windows)
                top_v, top_i = torch.topk(total_win_scores, curr_k, dim=1, largest=True)
                
                surviving_ids = torch.gather(competing_ids, 1, top_i)
                sort_idx = torch.argsort(surviving_ids, dim=1)
                
                final_v = torch.gather(top_v, 1, sort_idx)
                final_ids = torch.gather(surviving_ids, 1, sort_idx)
                
                self.window_scores.fill_(float("nan"))
                self.window_scores[:, :curr_k, 0] = final_v
                self.window_scores[:, :curr_k, 1] = final_ids
                self.window_scores[:, :curr_k, 2] = final_ids
                
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
                
                if self.tracking_flag:
                    # Universal ledger shift for generic tools - Shift correctly per head
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
        device = self.window_scores.device
        w_start, w_end = int(local_id * self.omega + 5), int(
            local_id * self.omega + 4 + self.omega
        )
        new_scores = (
            attn_scores[0, :, 0, w_start : w_end + 1]
            .sum(dim=-1)
            .to(self.window_scores.dtype)
        )
        current_ids = self.window_scores[:, :, 1]
        matches = current_ids == local_id
        has_match = matches.any(dim=1)
        if has_match.any():
            match_indices = matches[has_match].float().argmax(dim=1)
            matched_heads = has_match.nonzero().squeeze(-1)
            self.window_scores[matched_heads, match_indices, 0] += new_scores[has_match]
        if (~has_match).any():
            no_match_heads = (~has_match).nonzero().squeeze(-1)
            valid_mask = ~torch.isnan(self.window_scores[:, :, 0])
            counts = valid_mask[no_match_heads].sum(dim=1)
            counts = torch.clamp(counts, 0, self.window_scores.shape[1] - 1)
            self.window_scores[no_match_heads, counts, 0] = new_scores[no_match_heads]
            self.window_scores[no_match_heads, counts, 1] = float(local_id)
            self.window_scores[no_match_heads, counts, 2] = float(orig_id)

    def _evict_from_window_scores(self):
        valid_mask = ~torch.isnan(self.window_scores[:, :, 1])
        scores = torch.where(
            valid_mask,
            self.window_scores[:, :, 0],
            torch.tensor(float("-inf"), device=self.window_scores.device),
        )
        ids, orig_ids = self.window_scores[:, :, 1], self.window_scores[:, :, 2]
        curr_k = min(self.k_windows, int(valid_mask.sum(dim=1).max().item()))
        top_v, top_i = torch.topk(scores, curr_k, dim=1, largest=True)
        kept_ids, kept_orig = torch.gather(ids, 1, top_i), torch.gather(
            orig_ids, 1, top_i
        )
        sort_idx = torch.argsort(kept_ids, dim=1)
        self.window_scores.fill_(float("nan"))
        self.window_scores[:, :curr_k, 0] = torch.gather(top_v, 1, sort_idx)
        self.window_scores[:, :curr_k, 1] = torch.gather(kept_ids, 1, sort_idx)
        self.window_scores[:, :curr_k, 2] = torch.gather(kept_orig, 1, sort_idx)
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
        
        if self.tracking_flag:
            # Keep the logging ledger physically mapped to the compacted space (per-head):
            for head_idx in range(self.num_heads):
                phys_col = 2 + head_idx
                mask_evict = (self.token_ledger[:, phys_col] >= start_drop) & (self.token_ledger[:, phys_col] < end_drop)
                mask_shift = self.token_ledger[:, phys_col] >= end_drop
                self.token_ledger[mask_evict, phys_col] = -1.0
                self.token_ledger[mask_shift, phys_col] -= self.omega
        
        return (k_kept, v_kept), keep_indices.unsqueeze(0)

    def _create_mask_and_evict_from_kv_cache_prompt_stage(
        self, past_key_values, ob_start, ob_end, attn_scores
    ):
        seq_len, head_dim = (
            past_key_values[0].size(self.k_seq_dim),
            past_key_values[0].shape[-1],
        )
        
        num_w, remainder = (seq_len - 5) // self.omega, (seq_len - 5) % self.omega
        sticky_w = torch.nan_to_num(
            self.window_scores[:, : self.k_windows, 1], nan=0.0
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
        
        # We ensure anything from the ob_start sequence that *isn't* already captured in local_w is also added
        # To maintain exact topology for generation, the block from `local_window_start` to `seq_len` must be preserved perfectly
        local_window_start = seq_len - (self.local_num * self.omega) - remainder
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

        # Merge in any alpha tokens that proceed the local block (if alpha is exceptionally large)
        if ob_start < local_window_start:
            missing_alpha = torch.arange(ob_start, local_window_start, device=self.window_scores.device).unsqueeze(0).expand(self.num_heads, -1)
            all_indices = torch.cat([all_indices, missing_alpha], dim=1)
            
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
