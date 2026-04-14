import torch
from torch import nn
import math
from transformers.models.llama.modeling_llama import rotate_half


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Get dimensions from the hidden states (kv-cache)
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # If the number of query heads exactly matches the number of KV heads, no repetition needed
    if n_rep == 1:
        return hidden_states
    
    # Add a dimension for repeats and expand it across that dimension
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    
    # Flatten the repeated head dimension into the original head dimension
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(bsz, tgt_len, past_len, dtype, device):
    # Initialize a mask with the minimum possible float value (negative infinity equivalent)
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    
    # Create the condition for what to mask (causal, upper triangular masked)
    mask_cond = torch.arange(mask.size(-1), device=device)
    
    # Fill the lower triangular part with 0s (allowed attention part)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    
    # Convert mask to the expected data type
    mask = mask.to(dtype)
    
    # Connect past cache length to the causal mask by prefixing zeros
    if past_len > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_len, dtype=dtype, device=device), mask], dim=-1
        )
        
    # Expand to match batch size and number of heads
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_len)


def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    # Select the rotary embeddings specifically for the position IDs of the given queries
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    # Apply standard RoPE (Rotary Position Embedding) formula
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
        
        # Tracks whether initial prefill matrix was extracted
        self._prefill_done = False
        
        try:
            from sticky_config import tracking_flag
            self.tracking_flag = (tracking_flag == 1)
        except ImportError:
            self.tracking_flag = getattr(config, "tracking_flag", 1) == 1

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

        # Buffer: logical ID to physical index map
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
        
        # Accumulates 1D attention votes from generated tokens over OMEGA steps
        self.register_buffer(
            "running_attention_votes",
            torch.zeros((self.num_heads, max_context), dtype=torch.float32, device=device)
        )

        # Tracks total global tokens passed (arrival order) for the entire context
        self.register_buffer("global_token_counter", torch.tensor(0, dtype=torch.long))

        # The Ledger: [Global_ID, Layer_ID, Phys_id_Head0, Phys_id_Head1, ..., Score_Head0, Score_Head1, ...]
        self.register_buffer("token_ledger", 
                            torch.full((max_context, 2 + 2 * self.num_heads), -1.0, dtype=torch.float32))

        # Local History Buffer: cumulative window score per logical window id.
        self.register_buffer(
            "local_history",
            torch.zeros((self.num_heads, max_windows), dtype=torch.float32, device=device),
        )

        # Optional: High-resolution 2D history for research (Global_ID x Heads)
        self.register_buffer("global_score_history", 
                            torch.full((max_context, num_heads), -1.0, dtype=torch.float32))

        # Optional: Full NxN prefill matrix for rigorous research comparison
        self.prefill_attention_matrix = None

        # Defines physical cache sizes using parameters initialized
        self.cache_size = int(
            self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens 
        )
        
        # FIX: Will hold the strictly logical mapping of the physical cache
        self.logical_id_map = None
        

    def __call__(self, past_key_values, attn_score_cache, full_attn_scores=None):
        # Extract sizes
        bsz, q_heads, q_len, kv_seq_len = attn_score_cache.shape
        
        # Amount of tokens being put in the cache currently
        num_new_tokens = q_len

        # Defines full present context length directly from KV cache
        seq_len = past_key_values[0].size(self.k_seq_dim) if past_key_values is not None else 0
        
        # Start count
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
                        phys_idx = float(seq_len - q_len + i)
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
                    self.token_ledger[g_id, 2+self.num_heads:2+2*self.num_heads] = 0.0
                    self.global_score_history[g_id, :] = 0.0
        
        if past_key_values is None:
            return past_key_values

        seq_len = past_key_values[0].size(self.k_seq_dim)
        num_new_tokens = q_len

        # Initial Configuration Update phase (Prefill Setup)
        if num_new_tokens > 1:
            import sticky_config as config_module
            # FIX 1: CFO Allocator runs securely using token-native boundaries
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

        if num_new_tokens > 1:  # Prompt Stage Active
            # FIX 2: PREFILL BOUNDARY ALIGNMENT & REMAINDER LEAKAGE FIX
            local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else self.local_num
            
            score_end = max(self.sink_tokens, seq_len - local_tokens_count)
            num_windows = max(0, (score_end - self.sink_tokens) // self.omega)
            
            # Snap boundary to OMEGA chunks mathematically preventing out of bounds tensor slices
            score_end = self.sink_tokens + (num_windows * self.omega)
            score_end = min(score_end, attn_score_cache.shape[3])
            
            num_windows = (score_end - self.sink_tokens) // self.omega
            score_end = self.sink_tokens + (num_windows * self.omega)
            
            # Dynamically absorb remainder tokens perfectly protecting the trailing prompt sequence
            local_tokens_count = seq_len - score_end
            
            if num_windows > 0:
                # Direct chunking utilizing the aligned boundary
                scores_slice = attn_score_cache[0, :, :seq_len, self.sink_tokens:score_end]
                obs_sum = scores_slice.sum(dim=1)
                win_scores = obs_sum.view(self.num_heads, num_windows, self.omega).sum(dim=2).to(dtype=torch.float32)

                idx = torch.arange(num_windows, device=self.window_scores.device).unsqueeze(0).expand(self.num_heads, -1)
                self.window_scores[self.head_indices.unsqueeze(1), idx, 0] = win_scores
                self.window_scores[self.head_indices.unsqueeze(1), idx, 1] = idx.float()
                self.window_scores[self.head_indices.unsqueeze(1), idx, 2] = idx.float()

            self.local_history.zero_()
            total_prompt_windows = max(0, (seq_len - self.sink_tokens) // self.omega)
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
            # FIX 3: Inject score_end boundary directly avoiding calculation collisions
            updated_kv, survivor_ids = self._create_mask_and_evict_from_kv_cache_prompt_stage(
                past_key_values, attn_score_cache, score_end
            )
            
            # THE FIX: Store explicitly mapped Logical IDs, completely eliminating coordinate math later
            self.logical_id_map = torch.where(
                survivor_ids >= self.sink_tokens,
                (survivor_ids - self.sink_tokens) // self.omega,
                torch.tensor(-1, device=self.window_scores.device) # Sinks get -1
            )

            if self.tracking_flag:
                full_scores_ref = full_attn_scores if full_attn_scores is not None else attn_score_cache
                raw_matrix = full_scores_ref[0].detach().cpu()
                num_q_heads_total = raw_matrix.shape[0]
                group_size = num_q_heads_total // self.num_heads  
                
                self.prefill_attention_matrix = raw_matrix
                
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

            self._prefill_done = True  
            return updated_kv

        else:  # Generation Stage
            device = self.window_scores.device
            
            self.running_attention_votes[:, :seq_len] += attn_score_cache[0, :, 0, :seq_len]
            self.tokens_since_last_review += 1
            
            if self.tracking_flag:
                live_mask = self.token_ledger[:, 2] >= 0
                live_g_ids = torch.where(live_mask)[0]
                
                for head_idx in range(self.num_heads):
                    phys_indices = self.token_ledger[live_g_ids, 2 + head_idx].long()
                    valid_mask = phys_indices >= 0
                    valid_phys = phys_indices[valid_mask]
                    valid_g_ids = live_g_ids[valid_mask]
                    
                    if len(valid_phys) > 0 and valid_phys.max() < attn_score_cache.size(-1):
                        head_scores = attn_score_cache[0, head_idx, 0, valid_phys]
                        self.token_ledger[valid_g_ids, 2 + self.num_heads + head_idx] += head_scores.float()
                        self.global_score_history[valid_g_ids, head_idx] += head_scores.float()
            
            # 2. PERIODIC EVALUATION
            if self.tokens_since_last_review == self.omega:
                
                # FIX 4: GENERATION SCATTER MAP AND DYNAMIC SCOREBOARD RECONSTRUCTION
                local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else self.local_num
                
                # ---------------------------------------------------------
                # FIX 4: DIRECT LOGICAL LOOKUP & SCATTER MAP
                # ---------------------------------------------------------
                local_tokens_count = self.local_num_tokens if self.use_fixed_local_tokens else self.local_num
                
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

                # Update Local History strictly relying on appending properties preventing geometric crashes
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
                
                # Gather physical KV cache using flawless relative arrays
                head_dim = past_key_values[0].shape[-1]
                gather_idx = torch.clamp(final_relative_indices, 0, seq_len - 1).unsqueeze(-1).expand(-1, -1, head_dim)
                k_kept = torch.gather(past_key_values[0][0], 1, gather_idx).unsqueeze(0)
                v_kept = torch.gather(past_key_values[1][0], 1, gather_idx).unsqueeze(0)
                updated_kv = (k_kept, v_kept)
                
                if self.tracking_flag:
                    for head_idx in range(self.num_heads):
                        phys_col = 2 + head_idx
                        
                        live_mask = self.token_ledger[:, 2] >= 0
                        g_ids = torch.where(live_mask)[0]
                        old_phys = self.token_ledger[g_ids, phys_col].long()
                        
                        # THE FIX: Use the unpadded `sorted_relative` array to build the map!
                        # This prevents padded duplicates (like 0s or trailing tokens) 
                        # from overwriting the correct indices of legitimate tokens.
                        kept_phys = sorted_relative[head_idx]
                        
                        mapping = torch.full((seq_len,), -1.0, device=device)
                        mapping[kept_phys] = torch.arange(len(kept_phys), device=device, dtype=torch.float32)
                        
                        valid_old = (old_phys >= 0) & (old_phys < seq_len)
                        valid_old_phys = old_phys[valid_old]
                        valid_g_ids = g_ids[valid_old]
                        
                        # Collapse matrix properly tracking new shifted coordinate frame
                        new_phys = mapping[valid_old_phys]
                        self.token_ledger[valid_g_ids, phys_col] = new_phys

                self.running_attention_votes.zero_()
                self.tokens_since_last_review = 0
                
                return updated_kv
            else:
                return past_key_values
            
    def get_ledger_data(self):
        total_processed = self.global_token_counter.item()
        active_ledger = self.token_ledger[:total_processed].detach().cpu()
        
        global_ids = active_ledger[:, 0].long()
        layer_ids = active_ledger[:, 1].long()
        physical_positions = active_ledger[:, 2:2+self.num_heads].long()
        attention_scores = active_ledger[:, 2+self.num_heads:2+2*self.num_heads]
        
        return {
            "global_id": global_ids,
            "layer_id": layer_ids,
            "physical_id": physical_positions,
            "attention_score": attention_scores 
        }

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

    def _create_mask_and_evict_from_kv_cache_prompt_stage(self, past_key_values, attn_scores, local_start_idx):
        seq_len, head_dim = (
            past_key_values[0].size(self.k_seq_dim),
            past_key_values[0].shape[-1],
        )
        
        device = self.window_scores.device
        
        sinks = self.sink_indices.unsqueeze(0).expand(self.num_heads, -1)
        
        sticky_w = torch.nan_to_num(
            self.window_scores[:, : self.k_windows, 1], nan=0.0
        ).long()
        window_tokens = self.window_to_token_map[sticky_w].view(self.num_heads, -1)
        
        local_start = local_start_idx
        if local_start < seq_len:
            local_zone = torch.arange(local_start, seq_len, device=device).unsqueeze(0).expand(self.num_heads, -1)
            all_indices = torch.cat([sinks, window_tokens, local_zone], dim=1)
        else:
            all_indices = torch.cat([sinks, window_tokens], dim=1)
            
        all_indices_clamped = torch.clamp(all_indices, 0, seq_len - 1)
        
        sorted_indices = []
        for h in range(self.num_heads):
            unique = torch.unique(all_indices_clamped[h])
            sorted_indices.append(unique)
            
        max_len = max(len(u) for u in sorted_indices)
        padded_indices = []
        for h in range(self.num_heads):
            u = sorted_indices[h]
            if len(u) < max_len:
                pad = u[-1:].expand(max_len - len(u))
                u = torch.cat([u, pad])
            padded_indices.append(u)
            
        final_indices = torch.stack(padded_indices, dim=0) 
        
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
        # FIX 5: CFO Allocator evaluates strictly natively across token budget. 
        total_token_budget = (new_tokens + max_tokens) * self.total_cache_ratio // 100
        
        if self.use_fixed_local_tokens:
            target_local_tokens = self.local_num_tokens
        else:
            target_local_tokens = (total_token_budget * self.local_cache_ratio) // 100
            
        self.local_num = min(target_local_tokens, total_token_budget)
        available_sticky_tokens = total_token_budget - self.local_num - self.sink_tokens
        self.k_windows = max(0, available_sticky_tokens // self.omega)

    def _clean_scores(self):
        self.gen_step = self.num_of_tokens_without_eviction = 0
        self.tokens_since_last_review = 0
        if hasattr(self, "running_attention_votes"):
            self.running_attention_votes.zero_()
        self.window_scores.fill_(float("nan"))