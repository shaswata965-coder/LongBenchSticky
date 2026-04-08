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


def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin)


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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config is not None and hasattr(config, "max_position_embeddings"):
            max_context = config.max_position_embeddings
            max_windows = (
                ((max_context - self.sink_tokens) // self.omega) + 1 if max_context > self.sink_tokens else 1
            )
            max_windows = max(max_windows, 100)
        else:
            max_context = 131072
            max_windows = 30000

        window_ids = torch.arange(max_windows, device=device)
        token_map = (window_ids.unsqueeze(1) * self.omega + self.sink_tokens) + torch.arange(
            self.omega, device=device
        )

        self.register_buffer("window_to_token_map", token_map)
        self.register_buffer("sink_indices", torch.arange(0, self.sink_tokens, device=device) if self.sink_tokens > 0 else torch.zeros(0, dtype=torch.long, device=device))
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
        
        self.register_buffer("global_token_counter", torch.tensor(0, dtype=torch.long))
        
        # Accumulates 1D attention votes from generated tokens over OMEGA steps
        # Max context size is enough to track physically alive tokens
        self.register_buffer(
            "running_attention_votes",
            torch.zeros((self.num_heads, max_context), dtype=torch.float32, device=device)
        )

        self.register_buffer(
            "local_history",
            torch.zeros((self.num_heads, max_windows), dtype=torch.float32, device=device),
        )


        self.cache_size = int(
            self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens 
        )
        

    def __call__(self, past_key_values, attn_score_cache, full_attn_scores=None, q_len=None):
        bsz, q_heads, q_len_cache, kv_seq_len = attn_score_cache.shape
        
        q_len = q_len if q_len is not None else q_len_cache
        num_new_tokens = q_len

        # FIX: Define seq_len immediately to avoid ReferenceError in arrival loop
        seq_len = past_key_values[0].size(self.k_seq_dim) if past_key_values is not None else 0
        # === Inside __call__ ===
        
        if not self._prefill_done:
            self.global_token_counter += q_len
        else:
            self.global_token_counter += 1
        
        if past_key_values is None:
            return past_key_values

        seq_len = past_key_values[0].size(self.k_seq_dim)
        num_new_tokens = q_len

        if num_new_tokens > 1:
            import sticky_config as config_module
            self._update_k_win_and_local_num(num_new_tokens, config_module.GENERATION_CONFIG.get("max_new_tokens", 512))
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
            
            # Application boundary: Exclude Sinks, Local windows, and Observation windows
            score_end = max(self.sink_tokens, seq_len - local_tokens_count)
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

            # Seed local_history from full prefill attention (cumulative scores so far).
            self.local_history.zero_()
            total_prompt_windows = max(0, (seq_len - self.omega - self.sink_tokens) // self.omega)
            if total_prompt_windows > 0:
                full_review_end = self.sink_tokens + total_prompt_windows * self.omega
                actual_full_review = min(full_review_end, attn_score_cache.shape[3])
                total_prompt_windows = (actual_full_review - self.sink_tokens) // self.omega
                actual_full_review = self.sink_tokens + total_prompt_windows * self.omega

                if total_prompt_windows > 0:
                    full_scores_slice = attn_score_cache[0, :, :seq_len, self.sink_tokens:actual_full_review]
                    full_obs_sum = full_scores_slice.sum(dim=1)
                    full_win_scores = (
                        full_obs_sum.view(self.num_heads, total_prompt_windows, self.omega)
                        .sum(dim=2)
                        .to(dtype=torch.float32)
                    )
                    idx_full = torch.arange(
                        total_prompt_windows, device=self.local_history.device, dtype=torch.long
                    )
                    self.local_history[:, idx_full] = full_win_scores

            self._evict_from_window_scores()

            # Tracking logic removed per Fast Attention v2 instructions.

            # --- 1. GET SURVIVOR MAP ---
            updated_kv, survivor_ids = self._create_mask_and_evict_from_kv_cache_prompt_stage(
                past_key_values, attn_score_cache
            )

            self._prefill_done = True  # Mark prefill as complete
            return updated_kv

        else:  # Generation Stage
            device = self.window_scores.device
            
            # 1. ACCUMULATE VOTES
            self.running_attention_votes[:, :seq_len] += attn_score_cache[0, :, 0, :seq_len]
            self.tokens_since_last_review += 1
            
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
                
                num_old_windows = num_competing_windows - 1
                valid_old_windows = min(self.k_windows, num_old_windows)
                
                old_ids = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 1], nan=0.0)
                last_id_val = (self.num_of_tokens_without_eviction - 2 * self.omega - self.sink_tokens - local_tokens_count) // self.omega
                last_id_tensor = torch.full((self.num_heads, 1), float(max(0, last_id_val)), device=device, dtype=torch.float32)
                
                # Build competing_ids to exactly match num_competing_windows
                competing_ids = torch.cat([old_ids, last_id_tensor], dim=1) # [heads, valid_old + 1]
                
                # Build competing_hist to exactly match num_competing_windows
                competing_hist = torch.zeros((self.num_heads, num_competing_windows), device=device, dtype=torch.float32)
                if valid_old_windows > 0:
                    old_scores = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 0], nan=0.0)
                    competing_hist[:, :valid_old_windows] = old_scores

                # Local history injection for the newly entering window (from local protected bubble)
                if 0 <= last_id_val < self.local_history.shape[1] and num_competing_windows > valid_old_windows:
                    competing_hist[:, valid_old_windows] = self.local_history[:, last_id_val]
                    # Consume transferred history so buffer tracks only currently local windows.
                    self.local_history[:, last_id_val] = 0.0
                
                # win_scores covers ALL num_competing_windows from the slice
                # We need to align: take only the windows that have IDs
                num_with_ids = competing_ids.shape[1]
                if num_with_ids < num_competing_windows:
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

                # Update local_history with omega-token scores for windows that remain
                # inside the local protected bubble during this review.
                local_start = self.sink_tokens + num_competing_windows * self.omega
                local_tokens_eff = seq_len - local_start
                local_windows = local_tokens_eff // self.omega
                if local_windows > 0:
                    local_slice = self.running_attention_votes[
                        :, local_start : local_start + local_windows * self.omega
                    ]
                    local_scores = (
                        local_slice.view(self.num_heads, local_windows, self.omega)
                        .sum(dim=2)
                        .to(dtype=torch.float32)
                    )
                    local_id_start = (local_start - self.sink_tokens) // self.omega
                    ids = torch.arange(
                        local_id_start, local_id_start + local_windows, device=device, dtype=torch.long
                    )
                    valid = (ids >= 0) & (ids < self.local_history.shape[1])
                    if valid.any():
                        ids_valid = ids[valid]
                        self.local_history[:, ids_valid] += local_scores[:, valid]
                
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

                # Reset accumulator
                self.running_attention_votes.zero_()
                self.tokens_since_last_review = 0
                
                return updated_kv
            else:
                return past_key_values
            
    def get_ledger_data(self):
        """
        Retrieves the tracking data for research analysis.
        (Deprecated in Fast Attention v2.0 - returns empty dict)
        """
        return {}

    def _update_window_scores_generation_vectorized(self, attn_scores, local_id, orig_id):
        device = self.window_scores.device
        w_start, w_end = int(local_id * self.omega + self.sink_tokens), int(
            local_id * self.omega + (self.sink_tokens - 1) + self.omega
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



    def _create_mask_and_evict_from_kv_cache_prompt_stage(self, past_key_values, attn_scores):
        seq_len, head_dim = (
            past_key_values[0].size(self.k_seq_dim),
            past_key_values[0].shape[-1],
        )
        
        num_w, remainder = (seq_len - self.sink_tokens) // self.omega, (seq_len - self.sink_tokens) % self.omega
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
        self.global_token_counter.zero_()
