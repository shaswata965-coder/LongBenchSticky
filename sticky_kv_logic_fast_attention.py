import torch
from torch import nn
import math
from transformers.models.llama.modeling_llama import rotate_half

'''
Duplicates the KV heads n_rep times using memory-efficient 
expansion to match the number of query heads for GQA/MQA.
'''
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

'''
Generates a broadcastable causal mask (bsz, 1, tgt_len, tgt_len + past_len) to enforce autoregressive generation.
It uses minimum float values (-inf) to block future token attention via lower-triangular unmasking (0s), 
and prepends `past_len` zeros to permit full attention to all historical KV cache tokens during decoding.
'''
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
'''
Extracts the specific cos/sin frequencies for the current position_ids and broadcasts them.
It then applies the standard RoPE transformation, rotating the query vectors in latent space 
to explicitly encode the absolute sequence position into the token representations.
'''
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

        # Total cache allocated out of the max sequence length (percentage)
        self.total_cache_ratio = r_ratio
        
        # Percentage of total cache that is allocated to the local sliding window
        self.local_cache_ratio = p_ratio
        
        # Initialize default tracker for how many sticky windows we maintain
        self.k_windows = 3
        
        # Unused padding/offset tracker
        self.start_idx = start_idx
        
        from sticky_config import OMEGA, SINK_TOKENS
        # OMEGA specifies chunk sizes; eviction and review happens in blocks of OMEGA
        self.omega = OMEGA
        
        # SINK_TOKENS specifies permanent non-evictable anchor tokens at start of sequence
        self.sink_tokens = SINK_TOKENS
        
        try:
            from sticky_config import P_RATIO
            self.local_cache_ratio = P_RATIO
        except ImportError:
            self.local_cache_ratio = p_ratio
            
        # Force observation window to always equal OMEGA chunk size to keep chunks uniform
        self.alpha = self.omega
        
        # Initialize counter for tracking tokens generated since the last eviction cycle
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
            
        # Initialize token block tracker for local zone (updated during generation)
        self.local_num = 0
        
        # Pytorch layout trackers for Keys and Values
        self.k_seq_dim, self.v_seq_dim = k_seq_dim, v_seq_dim
        
        # Model layer identifier
        self.layer_idx = layer_idx
        
        # Total KV heads in this configuration
        self.num_heads = num_heads
        
        # Counter for amount of individual generations (tokens outputted)
        self.gen_step = 0
        
        # Counter used to assess total context depth internally
        self.num_of_tokens_without_eviction = 0
        
        # Boundary indicating where prefill ends and generating starts per head
        self.prompt_boundary = [-1 for _ in range(self.num_heads)]
        self._prefill_done = False  # Tracks whether initial prefill has completed

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get context window size dynamically to size trackers perfectly
        if config is not None and hasattr(config, "max_position_embeddings"):
            max_context = config.max_position_embeddings
            # Setup how many possible blocks of OMEGA size could exist
            max_windows = (
                ((max_context - self.sink_tokens) // self.omega) + 1 if max_context > self.sink_tokens else 1
            )
            # Apply safety buffer
            max_windows = max(max_windows, 100)
        else:
            # Fallbacks for extreme limits
            max_context = 131072
            max_windows = 30000

        # Create positional mapping from logical window blocks to physical cache token indices
        window_ids = torch.arange(max_windows, device=device)
        token_map = (window_ids.unsqueeze(1) * self.omega + self.sink_tokens) + torch.arange(
            self.omega, device=device
        )

        self.register_buffer("window_to_token_map", token_map)
        
        # Buffer: permanently protected token indices (0 to sink_tokens - 1)
        self.register_buffer("sink_indices", torch.arange(0, self.sink_tokens, device=device) if self.sink_tokens > 0 else torch.zeros(0, dtype=torch.long, device=device))
        
        # Buffer: stores sticky window scores! Shape is [heads, windows, 3]
        # Dim 2 is [Cumulative Score, Logical Window ID, Logical Window ID]
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
        
        # Tracks total global tokens passed (arrival order) for the entire context
        self.register_buffer("global_token_counter", torch.tensor(0, dtype=torch.long))
        
        # Accumulates 1D attention votes from generated tokens over OMEGA steps
        # Max context size is enough to track physically alive tokens
        self.register_buffer(
            "running_attention_votes",
            torch.zeros((self.num_heads, max_context), dtype=torch.float32, device=device)
        )

        # Local History Buffer: cumulative window score per logical window id.
        # Used when a window transitions from the local protected region into eviction contention.
        self.register_buffer(
            "local_history",
            torch.zeros((self.num_heads, max_windows), dtype=torch.float32, device=device),
        )


        # Defines physical cache sizes using parameters initialized
        self.cache_size = int(
            self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens 
        )
        
        # FIX: Will hold the strictly logical mapping of the physical cache
        self.logical_id_map = None

    def __call__(self, past_key_values, attn_score_cache, full_attn_scores=None, q_len=None):
        bsz, q_heads, q_len_cache, kv_seq_len = attn_score_cache.shape
        
        q_len = q_len if q_len is not None else q_len_cache
        num_new_tokens = q_len

        # Defines full present context length directly from KV cache
        seq_len = past_key_values[0].size(self.k_seq_dim) if past_key_values is not None else 0
        # === Inside __call__ ===
        
        # Keeps counter consistent globally 
        if not self._prefill_done:
            self.global_token_counter += q_len
        else:
            self.global_token_counter += 1
        
        if past_key_values is None:
            return past_key_values

        seq_len = past_key_values[0].size(self.k_seq_dim)
        num_new_tokens = q_len

        # Initial Configuration Update phase (Prefill Setup)
        if num_new_tokens > 1:
            import sticky_config as config_module
            # Distribute local limits vs sticky window constraints securely before calculating score windows
            self._update_k_win_and_local_num(num_new_tokens, config_module.GENERATION_CONFIG.get("max_new_tokens", 512))
            
            self.cache_size = (
                self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens
            )
            # Ensure counters are loaded up accurately considering full prompt is passed
            self.num_of_tokens_without_eviction += seq_len
            # Update boundary tracking parameter defining where generations start natively
            for h in range(self.num_heads):
                self.prompt_boundary[h] = seq_len - 1
        else:
            # Generations only generate 1 token at a time so step++
            self.num_of_tokens_without_eviction += 1
            self.gen_step += 1

        # Early return bypassed to maintain OMEGA synchronization


        if num_new_tokens > 1:  # Prompt Stage Active
            # Determine local token count precisely either fixed constraint or dynamic ratio calculation
            # FIX: Use sequence length and local_cache_ratio instead of local_num (which is 0 at init)
            # Unconditionally trust the CFO Allocator's token math
            local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else self.local_num
            
            # Application boundary: Exclude Sinks, Local windows, and Observation windows
            # Define logical point where eviction boundaries MUST end and local bubble starts
            score_end = max(self.sink_tokens, seq_len - local_tokens_count)
            
            # Find complete blocks capable of being tracked and potentially evicted
            num_windows = max(0, (score_end - self.sink_tokens) // self.omega)
            
            # --- REMAINDER LEAKAGE FIX ---
            # Snap boundary to OMEGA chunks
            score_end = self.sink_tokens + (num_windows * self.omega)
            score_end = min(score_end, attn_score_cache.shape[3])
            
            num_windows = (score_end - self.sink_tokens) // self.omega
            score_end = self.sink_tokens + (num_windows * self.omega)
            
            # Dynamically absorb remainder tokens into the local sliding window
            local_tokens_count = seq_len - score_end
            # -----------------------------
            
            if num_windows > 0:
                # FIX: Use full NxN prefill attention mapped onto the aligned eviction space
                scores_slice = attn_score_cache[0, :, :seq_len, self.sink_tokens:score_end]
                # Collapse query attentions, creating a 1D tensor describing KV importance per-head
                obs_sum = scores_slice.sum(dim=1)
                # Sum within blocks of size 'omega' to generate holistic chunk scores
                win_scores = obs_sum.view(self.num_heads, num_windows, self.omega).sum(dim=2).to(dtype=torch.float32)

                # Update tensor memory allocating values into initial ranking sets
                idx = torch.arange(num_windows, device=self.window_scores.device).unsqueeze(0).expand(self.num_heads, -1)
                self.window_scores[self.head_indices.unsqueeze(1), idx, 0] = win_scores
                self.window_scores[self.head_indices.unsqueeze(1), idx, 1] = idx.float()
                self.window_scores[self.head_indices.unsqueeze(1), idx, 2] = idx.float()

            # Seed local_history from full prefill attention (cumulative scores so far).
            # Everything currently in the "Local Zone" will at some point roll over to the "Evictable Zone"
            # It inherently needs the mass of the prompt to be fair for eviction decisions.
            self.local_history.zero_()
            total_prompt_windows = max(0, (seq_len - self.sink_tokens) // self.omega)
            if total_prompt_windows > 0:
                full_review_end = self.sink_tokens + total_prompt_windows * self.omega
                actual_full_review = min(full_review_end, attn_score_cache.shape[3])
                total_prompt_windows = (actual_full_review - self.sink_tokens) // self.omega
                actual_full_review = self.sink_tokens + total_prompt_windows * self.omega

                if total_prompt_windows > 0:
                    # Accumulate prompt metric again but applied up through the entire context mapping directly to local history
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
                    # Injection into buffer tracker
                    self.local_history[:, idx_full] = full_win_scores

            # Compress scores tensor directly down to Top 'K' specified winners initially via evict
            self._evict_from_window_scores()

            # Tracking logic removed per Fast Attention v2 instructions.

            updated_kv, survivor_ids = self._create_mask_and_evict_from_kv_cache_prompt_stage(
                past_key_values, attn_score_cache, score_end
            )
            
            # THE FIX: Store explicitly mapped Logical IDs, completely eliminating coordinate math later
            self.logical_id_map = torch.where(
                survivor_ids >= self.sink_tokens,
                (survivor_ids - self.sink_tokens) // self.omega,
                torch.tensor(-1, device=self.window_scores.device) # Sinks get -1
            )

            self._prefill_done = True  
            return updated_kv

        else:  # Generation Stage
            device = self.window_scores.device
            
            # 1. ACCUMULATE VOTES
            # Takes the latest generation attention 1D slices across the cache tensor and adds dynamically continuously
            self.running_attention_votes[:, :seq_len] += attn_score_cache[0, :, 0, :seq_len]
            self.tokens_since_last_review += 1
            
            # 2. PERIODIC EVALUATION
            # Process eviction constraints specifically when the tracker hits OMEGA boundary thresholds
            if self.tokens_since_last_review == self.omega:
                local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else self.local_num
                
                # ---------------------------------------------------------
                # FIX 4: DIRECT LOGICAL LOOKUP & SCATTER MAP
                # ---------------------------------------------------------
                
                # Retrieve compressed space properties
                compressed_len = self.logical_id_map.shape[1]
                compressed_votes = self.running_attention_votes[:, :compressed_len]
                
                # Look up Logical IDs directly from our map! No math required.
                logical_ids = self.logical_id_map
                
                is_chunk_token = logical_ids >= 0
                routed_votes = torch.where(is_chunk_token, compressed_votes, torch.zeros_like(compressed_votes))
                
                # Zero out negative IDs (sinks) so scatter_add doesn't throw OutOfBounds errors
                safe_logical_ids = torch.where(is_chunk_token, logical_ids, torch.zeros_like(logical_ids)).long()
                
                scoreboard = torch.zeros((self.num_heads, self.window_scores.shape[1]), device=device)
                scoreboard.scatter_add_(1, safe_logical_ids, routed_votes)
                
                # Determine current valid old competitors
                valid_mask = ~torch.isnan(self.window_scores[:, :, 1])
                valid_old_windows = min(self.k_windows, int(valid_mask.sum(dim=1).max().item()))

                old_ids = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 1], nan=0.0)
                old_scores_hist = torch.nan_to_num(self.window_scores[:, :valid_old_windows, 0], nan=0.0)

                # Collect new mass generated strictly mapped dynamically 
                old_w_gen_scores = torch.gather(scoreboard, 1, old_ids.long()) if valid_old_windows > 0 else torch.zeros_like(old_scores_hist)
                old_scores = old_scores_hist + old_w_gen_scores

                # ---------------------------------------------------------
                # SCORE THE CHALLENGER (The Emerging Window)
                # ---------------------------------------------------------
                # Calculate the Logical ID of the window that just fell out of the local bubble
                last_id_val = max(0, (self.num_of_tokens_without_eviction - self.sink_tokens - local_tokens_count) // self.omega - 1)
                last_id_tensor = torch.full((self.num_heads, 1), float(last_id_val), device=device, dtype=torch.float32)

                # Because the Emerging Window was physically inside the compressed cache, 
                # its votes were already perfectly routed into our dynamic scoreboard!
                if last_id_val < scoreboard.shape[1]:
                    new_w_gen_scores = scoreboard[:, int(last_id_val)]
                else:
                    new_w_gen_scores = torch.zeros(self.num_heads, dtype=torch.float32, device=device)

                # Safely map prefill history
                if 0 <= last_id_val < self.local_history.shape[1]:
                    last_id_hist_scores = self.local_history[:, last_id_val].clone()
                    self.local_history[:, last_id_val] = 0.0
                else:
                    last_id_hist_scores = torch.zeros(self.num_heads, dtype=torch.float32, device=device)

                new_w_total_scores = new_w_gen_scores + last_id_hist_scores

                # Compete directly integrating tracking frameworks
                competing_ids = torch.cat([old_ids, last_id_tensor], dim=1)
                competing_scores = torch.cat([old_scores, new_w_total_scores.unsqueeze(1)], dim=1)

                curr_k = min(self.k_windows, competing_scores.shape[1])
                top_v, top_i = torch.topk(competing_scores, curr_k, dim=1, largest=True)

                surviving_ids = torch.gather(competing_ids, 1, top_i)
                sort_idx = torch.argsort(surviving_ids, dim=1)

                final_v = torch.gather(top_v, 1, sort_idx)
                final_ids = torch.gather(surviving_ids, 1, sort_idx)

                self.window_scores.fill_(float("nan"))
                self.window_scores[:, :curr_k, 0] = final_v
                self.window_scores[:, :curr_k, 1] = final_ids
                self.window_scores[:, :curr_k, 2] = final_ids

                # ---------------------------------------------------------
                # 4. UPDATE LOCAL HISTORY
                # ---------------------------------------------------------
                local_tokens_eff = seq_len - (compressed_len + self.omega)
                local_windows = local_tokens_eff // self.omega

                if local_windows > 0:
                    local_zone_start_phys = compressed_len + self.omega
                    valid_local_len = local_windows * self.omega
                    local_slice = self.running_attention_votes[:, local_zone_start_phys : local_zone_start_phys + valid_local_len]
                    local_scores = local_slice.view(self.num_heads, local_windows, self.omega).sum(dim=2).to(dtype=torch.float32)

                    local_id_start = last_id_val + 1
                    ids = torch.arange(local_id_start, local_id_start + local_windows, device=device, dtype=torch.long)

                    valid = (ids >= 0) & (ids < self.local_history.shape[1])
                    if valid.any():
                        ids_valid = ids[valid]
                        self.local_history[:, ids_valid] += local_scores[:, valid]

                # If r_ratio is 100, skip physical eviction
                if self.total_cache_ratio == 100:
                    self.running_attention_votes.zero_()
                    self.tokens_since_last_review = 0
                    return past_key_values
                
                # ---------------------------------------------------------
                # 5. PHYSICAL EVICTION (Purely Relative Slicing)
                # ---------------------------------------------------------
                # Step A: Filter the OLD compressed cache using relative indices
                is_survivor = torch.zeros_like(self.logical_id_map, dtype=torch.bool)
                for h in range(self.num_heads):
                    is_survivor[h] = torch.isin(self.logical_id_map[h], final_ids[h])
                
                relative_indices = torch.arange(compressed_len, device=device).unsqueeze(0).expand(self.num_heads, -1)
                
                # Assign a massively high index to losers so they sort to the end and get chopped off
                kept_old_relative = torch.where(is_survivor, relative_indices, torch.tensor(seq_len + 999, device=device))

                # Step B: Sinks and Local Zone are defined strictly by their current relative positions
                sinks_relative = torch.arange(self.sink_tokens, device=device).unsqueeze(0).expand(self.num_heads, -1)
                local_relative = torch.arange(compressed_len + self.omega, seq_len, device=device).unsqueeze(0).expand(self.num_heads, -1)
                
                all_relative = torch.cat([sinks_relative, kept_old_relative, local_relative], dim=1)
                
                # Step C: Deduplicate, Sort, and remove Losers
                sorted_relative = []
                for h in range(self.num_heads):
                    unique = torch.unique(all_relative[h])
                    unique = unique[unique < seq_len] # Drop the +999 losers
                    sorted_relative.append(unique)
                    
                max_len = max(len(u) for u in sorted_relative)
                padded_indices = []
                for h in range(self.num_heads):
                    u = sorted_relative[h]
                    if len(u) < max_len:
                        pad = u[-1:].expand(max_len - len(u)) if len(u) > 0 else torch.tensor([0], device=device).expand(max_len)
                        u = torch.cat([u, pad])
                    padded_indices.append(u)
                    
                final_relative_indices = torch.stack(padded_indices, dim=0)
                
                # ---------------------------------------------------------
                # 6. REBUILD THE LOGICAL MAP FOR THE NEXT CYCLE
                # ---------------------------------------------------------
                new_logical_id_map = torch.zeros_like(final_relative_indices, dtype=torch.float32)
                
                for h in range(self.num_heads):
                    for i in range(final_relative_indices.shape[1]):
                        rel_idx = final_relative_indices[h, i]
                        if rel_idx < compressed_len:
                            # If it came from the old cache, inherit its exact Logical ID
                            new_logical_id_map[h, i] = self.logical_id_map[h, rel_idx]
                        else:
                            # If it's a newly generated local token, calculate its new Logical ID
                            offset = rel_idx - (compressed_len + self.omega)
                            new_logical_id_map[h, i] = (last_id_val + 1) + (offset // self.omega)
                            
                self.logical_id_map = new_logical_id_map 
                
                # Gather physical KV cache using relative arrays
                head_dim = past_key_values[0].shape[-1]
                gather_idx = torch.clamp(final_relative_indices, 0, seq_len - 1).unsqueeze(-1).expand(-1, -1, head_dim)
                k_kept = torch.gather(past_key_values[0][0], 1, gather_idx).unsqueeze(0)
                v_kept = torch.gather(past_key_values[1][0], 1, gather_idx).unsqueeze(0)
                updated_kv = (k_kept, v_kept)

                # Reset accumulator vectors entirely explicitly
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
        # Maps boundaries using chunk constraints logically targeting tensor blocks efficiently.
        device = self.window_scores.device
        w_start, w_end = int(local_id * self.omega + self.sink_tokens), int(
            local_id * self.omega + (self.sink_tokens - 1) + self.omega
        )
        new_scores = (
            attn_scores[0, :, 0, w_start : w_end + 1]
            .sum(dim=-1)
            .to(self.window_scores.dtype)
        )
        
        # Validate elements identically locating target array
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
        # Checks purely valid values bypassing NaN allocations explicitly generated earlier
        valid_mask = ~torch.isnan(self.window_scores[:, :, 1])
        scores = torch.where(
            valid_mask,
            self.window_scores[:, :, 0],
            torch.tensor(float("-inf"), device=self.window_scores.device),
        )
        ids, orig_ids = self.window_scores[:, :, 1], self.window_scores[:, :, 2]
        
        # Calculates dynamic upperbound constraining max values properly
        curr_k = min(self.k_windows, int(valid_mask.sum(dim=1).max().item()))
        
        # Pull best-performing IDs out corresponding safely with limits established
        top_v, top_i = torch.topk(scores, curr_k, dim=1, largest=True)
        kept_ids, kept_orig = torch.gather(ids, 1, top_i), torch.gather(
            orig_ids, 1, top_i
        )
        
        # Assure positional integrity maintaining accurate sequential extraction chronological
        sort_idx = torch.argsort(kept_ids, dim=1)
        self.window_scores.fill_(float("nan"))
        self.window_scores[:, :curr_k, 0] = torch.gather(top_v, 1, sort_idx)
        self.window_scores[:, :curr_k, 1] = torch.gather(kept_ids, 1, sort_idx)
        self.window_scores[:, :curr_k, 2] = torch.gather(kept_orig, 1, sort_idx)
        return []



    def _create_mask_and_evict_from_kv_cache_prompt_stage(self, past_key_values, attn_scores, local_start_idx):
        seq_len, head_dim = (
            past_key_values[0].size(self.k_seq_dim),
            past_key_values[0].shape[-1],
        )
        
        device = self.window_scores.device
        
        # 1. Take out sink tokens and assign array
        sinks = self.sink_indices.unsqueeze(0).expand(self.num_heads, -1)
        
        # [Steps 2 & 3 Removed: Redundant math deleted to enforce boundary synchronization]
        
        sticky_w = torch.nan_to_num(
            self.window_scores[:, : self.k_windows, 1], nan=0.0
        ).long()
        window_tokens = self.window_to_token_map[sticky_w].view(self.num_heads, -1)
        
        # 4. Final local zone calculations building indices directly out to sequence end exclusively 
        # FIX: Directly inject the exact boundary passed from the caller
        local_start = local_start_idx
        
        if local_start < seq_len:
            local_zone = torch.arange(local_start, seq_len, device=device).unsqueeze(0).expand(self.num_heads, -1)
            all_indices = torch.cat([sinks, window_tokens, local_zone], dim=1)
        else:
            all_indices = torch.cat([sinks, window_tokens], dim=1)
            
        # Optional: ensure unique and properly ordered (not strictly required here depending on generation assumptions, but safe metric)
        # Using simple clamping for gather safety preventing OutOfBounds indexing exceptions natively
        all_indices_clamped = torch.clamp(all_indices, 0, seq_len - 1)
        
        # WE MUST SORT AND DEDUPLICATE PER HEAD so the physical KV cache stays chronological
        # and so that the ledger mapping correctly aligns with the physical tensor dimensions.
        sorted_indices = []
        for h in range(self.num_heads):
            # deduplicate and sort specifically preventing index mismatching errors
            unique = torch.unique(all_indices_clamped[h])
            sorted_indices.append(unique)
            
        # Due to dynamic deduplication, some heads might have 1 more or less token depending 
        # on overlap between sinks/local/alpha. We pad to the max length across heads for tensor compat.
        max_len = max(len(u) for u in sorted_indices)
        padded_indices = []
        for h in range(self.num_heads):
            u = sorted_indices[h]
            if len(u) < max_len:
                # pad by repeating the last valid token maintaining sequential uniform lengths internally
                pad = u[-1:].expand(max_len - len(u))
                u = torch.cat([u, pad])
            padded_indices.append(u)
            
        final_indices = torch.stack(padded_indices, dim=0) # [heads, max_len]
        
        # Shape indices exactly natively indexing vector space across head dimension arrays matching constraints
        gather_idx = (
            final_indices
            .unsqueeze(-1)
            .expand(-1, -1, head_dim)
        )
        
        # Returns physically compressed parameters correctly matching gathered index
        return (
            torch.gather(past_key_values[0][0], 1, gather_idx).unsqueeze(0),
            torch.gather(past_key_values[1][0], 1, gather_idx).unsqueeze(0),
        ), final_indices

    def _update_k_win_and_local_num(self, new_tokens, max_tokens):
        # 1. Calculate the absolute global token budget
        total_token_budget = (new_tokens + max_tokens) * self.total_cache_ratio // 100
        
        # 2. Calculate target local TOKENS natively
        if self.use_fixed_local_tokens:
            target_local_tokens = self.local_num_tokens
        else:
            # Apply percentage directly to the raw token budget
            target_local_tokens = (total_token_budget * self.local_cache_ratio) // 100
            
        # 3. Starvation Protection: Ensure local tokens never exceed the absolute budget
        self.local_num = min(target_local_tokens, total_token_budget)
        
        # 4. Calculate exact remaining tokens available for sticky history
        # We mathematically subtract the protected local zone and the permanent sinks
        available_sticky_tokens = total_token_budget - self.local_num - self.sink_tokens
        
        # 5. Convert leftover tokens directly into complete sticky chunks
        self.k_windows = max(0, available_sticky_tokens // self.omega)

    def _clean_scores(self):
        # Hard resets dynamically tracking metrics inside class resetting variables unconditionally 
        self.gen_step = self.num_of_tokens_without_eviction = 0
        self.tokens_since_last_review = 0
        if hasattr(self, "running_attention_votes"):
            self.running_attention_votes.zero_()
        self.window_scores.fill_(float("nan"))
        self.global_token_counter.zero_()
