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
        
        # Q_RATIO: percentage of total cache budget reserved for int8-quantized evicted tokens
        try:
            from sticky_config import Q_RATIO
            self.q_ratio = Q_RATIO
        except ImportError:
            self.q_ratio = 0
        
        # Tracks the number of token slots allocated for the quantized cache (computed at prefill)
        self.q_num = 0
        self.q_windows_count = 0
        
        # head_dim needed for int8 compression ratio calculation
        if config is not None and hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
            self.head_dim = config.hidden_size // config.num_attention_heads
        else:
            self.head_dim = 64  # Llama 3.2 1B default
        
        # INT8 quantized side-cache — per-WINDOW quantization (lazy-initialized at prefill)
        # Layout: [num_heads, q_windows_count, omega, head_dim]
        self.q_cache_k_int8 = None        # [H, W, omega, D] uint8
        self.q_cache_v_int8 = None        # [H, W, omega, D] uint8
        # K scale/zp: per-channel per-window, RoPE-paired → [H, W, 1, D]
        self.q_cache_k_scale = None       # [H, W, 1, D] float16
        self.q_cache_k_zp = None          # [H, W, 1, D] float16
        # V scale/zp: per-token per-window → [H, W, omega, 1]
        self.q_cache_v_scale = None       # [H, W, omega, 1] float16
        self.q_cache_v_zp = None          # [H, W, omega, 1] float16
        self.q_cache_ids = None           # [num_heads, q_windows_count] logical window IDs
        self.q_cache_scores = None        # [num_heads, q_windows_count] cumulative scores
        
        # Retired metadata store — scale/zp from promoted windows (never deleted)
        # Each entry: {window_id, head, k_scale[1,1,D], k_zp[1,1,D], v_scale[1,omega,1], v_zp[1,omega,1]}
        self.q_retired_meta = {}  # keyed by (window_id_float, head_int) → O(1) lookup
        
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
        
        # Persisted dynamic local count (absorbs remainder from prefill alignment)
        self._dynamic_local_count = 0
        
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
        

    def __call__(self, past_key_values, attn_score_cache, full_attn_scores=None, q_attn_scores=None):
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
        # FIX (Audit Bug 2): Gate on _prefill_done instead of num_new_tokens > 1.
        # Using num_new_tokens > 1 would misfire under speculative decoding where
        # q_len > 1 can occur during generation, destroying all eviction state.
        if not self._prefill_done:
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

        if not self._prefill_done:  # Prompt Stage Active
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
            
            # Persist the dynamic local count so generation eviction uses the same boundary
            self._dynamic_local_count = local_tokens_count
            
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

            q_loser_ids, q_loser_scores = self._evict_from_window_scores()

            # --- Q-CACHE: Capture and quantize loser KV data ---
            if q_loser_ids is not None:
                q_count = q_loser_ids.shape[1]
                q_phys_indices = self.window_to_token_map[q_loser_ids.long()]  # [H, q_count, omega]
                q_phys_flat = q_phys_indices.reshape(self.num_heads, -1)  # [H, q_count * omega]
                q_phys_flat = torch.clamp(q_phys_flat, 0, seq_len - 1)
                hd = past_key_values[0].shape[-1]
                gather_q = q_phys_flat.unsqueeze(-1).expand(-1, -1, hd)
                q_k_data = torch.gather(past_key_values[0][0], 1, gather_q)  # [H, q_count*omega, D]
                q_v_data = torch.gather(past_key_values[1][0], 1, gather_q)
                # Reshape to per-window layout [H, W, omega, D] for independent quantization
                q_k_data = q_k_data.view(self.num_heads, q_count, self.omega, hd)
                q_v_data = q_v_data.view(self.num_heads, q_count, self.omega, hd)
                self.q_cache_k_int8, self.q_cache_k_scale, self.q_cache_k_zp = self._quantize_k_per_window(q_k_data)
                self.q_cache_v_int8, self.q_cache_v_scale, self.q_cache_v_zp = self._quantize_v_per_window(q_v_data)
                self.q_cache_ids = q_loser_ids.float()
                self.q_cache_scores = q_loser_scores

            if self.tracking_flag:
                importance_map = attn_score_cache[0, :, :seq_len, :].sum(dim=1)
                active_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1) & (self.token_ledger[:, 0] >= 0)
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
                torch.full_like(survivor_ids, -1)  # Sinks get -1
            ).to(torch.long)

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
                    # FIX (B3): survivor_ids contains physical cache indices, not global IDs.
                    # Use global_start offset to map physical positions to ledger rows.
                    for phys_idx, phys_id in enumerate(clean_survivors):
                        g_id = global_start + phys_id.item()
                        if 0 <= g_id < self.token_ledger.shape[0]:
                            self.token_ledger[g_id, 2 + head_idx] = float(phys_idx)
                            
                    for phys_id in clean_survivors:
                        g_id = global_start + phys_id.item()
                        if 0 <= g_id < self.token_ledger.shape[0] and phys_id < full_importance.shape[1]:
                            self.token_ledger[g_id, 2 + self.num_heads + head_idx] = full_importance[head_idx, phys_id]

            self._prefill_done = True  
            return updated_kv

        else:  # Generation Stage
            device = self.window_scores.device
            
            # FIX (B4): Guard against generation being called before prefill constructed logical_id_map
            if self.logical_id_map is None:
                return past_key_values
            
            self.running_attention_votes[:, :seq_len] += attn_score_cache[0, :, 0, :seq_len]
            self.tokens_since_last_review += 1
            
            # Accumulate q-cache attention scores from joint softmax
            if q_attn_scores is not None and self.q_cache_scores is not None:
                q_per_token = q_attn_scores[0, :, 0, :]  # [kv_heads, q_tokens]
                q_tokens_total = self.q_cache_ids.shape[1] * self.omega
                if q_per_token.shape[1] >= q_tokens_total:
                    q_per_token = q_per_token[:, :q_tokens_total]
                    q_per_window = q_per_token.view(self.num_heads, self.q_cache_ids.shape[1], self.omega).sum(dim=2)
                    self.q_cache_scores = self.q_cache_scores + q_per_window.to(self.q_cache_scores.dtype)
            
            if self.tracking_flag:
                live_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1)
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
                # FIX (Bug 1): Use the persisted dynamic local count that absorbed
                # the prefill remainder, instead of re-reading the static config value.
                local_tokens_count = self._dynamic_local_count
                
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
                
                scoreboard = torch.zeros((self.num_heads, self.window_scores.shape[1]), device=device, dtype=torch.float32)
                scoreboard.scatter_add_(1, safe_logical_ids, routed_votes)
                
                # FIX (Bug 1): Route votes for the omega new tokens not yet in logical_id_map.
                # These tokens live at physical indices [compressed_len, seq_len) and have
                # accumulated votes in running_attention_votes, but scatter_add_ above skipped
                # them because logical_id_map doesn't include them yet. Without this, their
                # votes are permanently lost when running_attention_votes is zeroed at cycle end.
                if seq_len > compressed_len:
                    n_new = seq_len - compressed_len
                    new_tok_votes = self.running_attention_votes[:, compressed_len:seq_len]
                    js = torch.arange(n_new, device=device, dtype=torch.long)
                    new_lids = ((self.num_of_tokens_without_eviction - self.omega + js - self.sink_tokens) // self.omega).clamp(min=0)
                    valid_new = new_lids < scoreboard.shape[1]
                    if valid_new.any():
                        scoreboard.scatter_add_(1, new_lids[valid_new].unsqueeze(0).expand(self.num_heads, -1), new_tok_votes[:, valid_new])
                
                # Determine current valid old competitors
                valid_mask = ~torch.isnan(self.window_scores[:, :, 1])
                valid_old_windows = min(self.k_windows, int(valid_mask.sum(dim=1).max().item()))

                raw_ids = self.window_scores[:, :valid_old_windows, 1]
                raw_scores = self.window_scores[:, :valid_old_windows, 0]
                # FIX (B2): Track which slots are genuinely registered vs NaN-padded empty slots
                is_valid_slot = ~torch.isnan(raw_ids)
                
                old_ids = torch.nan_to_num(raw_ids, nan=0.0)
                old_scores_hist = torch.nan_to_num(raw_scores, nan=0.0)

                # Collect new mass generated strictly mapped dynamically 
                old_w_gen_scores = torch.gather(scoreboard, 1, old_ids.long()) if valid_old_windows > 0 else torch.zeros_like(old_scores_hist)
                # FIX (B2): Zero out phantom Window 0 scores gathered from NaN→0 converted empty slots
                old_w_gen_scores = torch.where(is_valid_slot, old_w_gen_scores, torch.zeros_like(old_w_gen_scores))
                old_scores = old_scores_hist + old_w_gen_scores

                # ---------------------------------------------------------
                # SCORE THE CHALLENGER (The Emerging Window)
                # ---------------------------------------------------------
                # Calculate the Logical ID of the window that just fell out of the local bubble
                raw_last_id_val = (self.num_of_tokens_without_eviction - self.sink_tokens - local_tokens_count) // self.omega - 1
                has_challenger = raw_last_id_val >= 0
                last_id_val = raw_last_id_val

                if has_challenger:
                    last_id_tensor = torch.full((self.num_heads, 1), float(last_id_val), device=device, dtype=torch.float32)

                    # FIX (Bug 5): Per-head guard against double-counting.
                    # Each head independently checks if the challenger is already tracked.
                    # Heads where it's tracked get -inf score so topk naturally excludes the duplicate.
                    already_tracked_per_head = (old_ids.long() == last_id_val).any(dim=1)  # [num_heads]

                    # Because the Emerging Window was physically inside the compressed cache, 
                    # its votes were already perfectly routed into our dynamic scoreboard!
                    if last_id_val < scoreboard.shape[1]:
                        new_w_gen_scores = scoreboard[:, int(last_id_val)]
                    else:
                        new_w_gen_scores = torch.zeros(self.num_heads, dtype=torch.float32, device=device)

                    # Safely map prefill history
                    if last_id_val < self.local_history.shape[1]:
                        last_id_hist_scores = self.local_history[:, last_id_val].clone()
                        # Only zero out history for heads where the window is entering competition
                        self.local_history[:, last_id_val] = torch.where(
                            already_tracked_per_head,
                            self.local_history[:, last_id_val],
                            torch.zeros_like(self.local_history[:, last_id_val])
                        )
                    else:
                        last_id_hist_scores = torch.zeros(self.num_heads, dtype=torch.float32, device=device)

                    new_w_total_scores = new_w_gen_scores + last_id_hist_scores

                    # Mask to -inf for heads where challenger is already tracked — topk excludes it
                    new_w_total_scores = torch.where(
                        already_tracked_per_head,
                        torch.full_like(new_w_total_scores, float('-inf')),
                        new_w_total_scores
                    )

                    # Concatenate challenger with existing windows
                    competing_ids = torch.cat([old_ids, last_id_tensor], dim=1)
                    competing_scores = torch.cat([old_scores, new_w_total_scores.unsqueeze(1)], dim=1)
                else:
                    # No valid challenger yet — existing windows compete among themselves
                    competing_ids = old_ids
                    competing_scores = old_scores

                # --- Q-CACHE MERGE: Include q-cache windows in competition ---
                if self.q_windows_count > 0 and self.q_cache_ids is not None:
                    competing_ids = torch.cat([competing_ids, self.q_cache_ids], dim=1)
                    competing_scores = torch.cat([competing_scores, self.q_cache_scores], dim=1)

                curr_k = min(self.k_windows, competing_scores.shape[1])
                top_v, top_i = torch.topk(competing_scores, curr_k, dim=1, largest=True)

                surviving_ids = torch.gather(competing_ids, 1, top_i)
                sort_idx = torch.argsort(surviving_ids, dim=1)

                final_v = torch.gather(top_v, 1, sort_idx)
                final_ids = torch.gather(surviving_ids, 1, sort_idx)

                # --- Q-CACHE: Determine new q-cache from remaining losers ---
                new_q_loser_ids = None
                new_q_loser_scores = None
                if self.q_windows_count > 0:
                    remaining_scores = competing_scores.clone()
                    remaining_scores.scatter_(1, top_i, float("-inf"))
                    num_remaining = int((remaining_scores > float("-inf")).sum(dim=1).min().item())
                    if num_remaining > 0:
                        q_count = min(self.q_windows_count, num_remaining)
                        q_top_v, q_top_i = torch.topk(remaining_scores, q_count, dim=1, largest=True)
                        new_q_loser_ids = torch.gather(competing_ids, 1, q_top_i)
                        new_q_loser_scores = q_top_v

                # --- Q-CACHE: Handle promotions (q-cache → main cache) ---
                # q-cache layout is now [H, W, omega, D] with per-window scale/zp
                promoted_q_data_k = {}
                promoted_q_data_v = {}
                if self.q_cache_ids is not None:
                    for h in range(self.num_heads):
                        promoted_q_data_k[h] = []
                        promoted_q_data_v[h] = []
                        for qi in range(self.q_cache_ids.shape[1]):
                            q_wid = self.q_cache_ids[h, qi]
                            if torch.isin(q_wid, final_ids[h]).item():
                                # Dequantize using per-window index qi directly
                                k_deq = self._dequantize_from_int8(
                                    self.q_cache_k_int8[h:h+1, qi:qi+1],   # [1,1,omega,D]
                                    self.q_cache_k_scale[h:h+1, qi:qi+1],  # [1,1,1,D]
                                    self.q_cache_k_zp[h:h+1, qi:qi+1]
                                )  # → [1,1,omega,D]
                                v_deq = self._dequantize_from_int8(
                                    self.q_cache_v_int8[h:h+1, qi:qi+1],
                                    self.q_cache_v_scale[h:h+1, qi:qi+1],
                                    self.q_cache_v_zp[h:h+1, qi:qi+1]
                                )
                                # Squeeze to [omega, D] for insertion
                                promoted_q_data_k[h].append((q_wid.item(), k_deq.squeeze(0).squeeze(0)))
                                promoted_q_data_v[h].append((q_wid.item(), v_deq.squeeze(0).squeeze(0)))
                                # Archive per-window scale/zp (never deleted)
                                self.q_retired_meta[(q_wid.item(), h)] = {
                                    'k_scale': self.q_cache_k_scale[h, qi].detach().clone(),
                                    'k_zp': self.q_cache_k_zp[h, qi].detach().clone(),
                                    'v_scale': self.q_cache_v_scale[h, qi].detach().clone(),
                                    'v_zp': self.q_cache_v_zp[h, qi].detach().clone(),
                                }

                self.window_scores.fill_(float("nan"))
                self.window_scores[:, :curr_k, 0] = final_v
                self.window_scores[:, :curr_k, 1] = final_ids
                self.window_scores[:, :curr_k, 2] = final_ids

                # Update Local History
                # FIX (Bug 1+2): The old formula `seq_len - (compressed_len + omega)`
                # always evaluates to 0 because compressed_len = logical_id_map.shape[1]
                # which already includes the local zone from the previous cycle.
                # The local zone always has exactly `local_tokens_count` tokens.
                # The scoreboard already routes local-zone votes correctly via
                # logical_id_map, so we read from it directly.
                # FIX (Bug 2): Use ceiling division so partial-window votes are
                # preserved in local_history instead of being wiped by zero_().
                local_windows = (local_tokens_count + self.omega - 1) // self.omega

                if local_windows > 0 and has_challenger:
                    local_id_start = last_id_val + 1
                    ids = torch.arange(local_id_start, local_id_start + local_windows, device=device, dtype=torch.long)

                    valid = (ids >= 0) & (ids < self.local_history.shape[1])
                    if valid.any():
                        ids_valid = ids[valid]
                        # Read votes directly from scoreboard — local zone votes are
                        # already correctly routed here via scatter_add_ + logical_id_map
                        self.local_history[:, ids_valid] += scoreboard[:, ids_valid]

                if self.total_cache_ratio == 100:
                    self.running_attention_votes.zero_()
                    self.tokens_since_last_review = 0
                    return past_key_values
                
                # --- Q-CACHE: Rebuild with ZERO-DEGRADATION routing ---
                # Three paths per window to prevent quantization error accumulation:
                # A) RETAINED: was in q-cache, stays in q-cache → copy raw int8+scale directly
                # B) DEMOTED with archived meta: use archived scale/zp from q_retired_meta
                # C) FRESH: truly new loser → fresh per-window quantization
                if new_q_loser_ids is not None and self.q_windows_count > 0:
                    new_q_count = new_q_loser_ids.shape[1]
                    head_dim = past_key_values[0].shape[-1]
                    dtype_fp = past_key_values[0].dtype
                    
                    # Pre-allocate per-window tensors
                    new_k_int8 = torch.zeros(self.num_heads, new_q_count, self.omega, head_dim, device=device, dtype=torch.uint8)
                    new_v_int8 = torch.zeros(self.num_heads, new_q_count, self.omega, head_dim, device=device, dtype=torch.uint8)
                    new_k_scale = torch.zeros(self.num_heads, new_q_count, 1, head_dim, device=device, dtype=dtype_fp)
                    new_k_zp = torch.zeros(self.num_heads, new_q_count, 1, head_dim, device=device, dtype=dtype_fp)
                    new_v_scale = torch.zeros(self.num_heads, new_q_count, self.omega, 1, device=device, dtype=dtype_fp)
                    new_v_zp = torch.zeros(self.num_heads, new_q_count, self.omega, 1, device=device, dtype=dtype_fp)
                    
                    for qi in range(new_q_count):
                        for h in range(self.num_heads):
                            wid = new_q_loser_ids[h, qi]
                            
                            # Path A: RETAINED — was already in q-cache, copy raw int8
                            retained = False
                            if self.q_cache_ids is not None:
                                q_match = (self.q_cache_ids[h] == wid).nonzero(as_tuple=True)[0]
                                if len(q_match) > 0:
                                    old_qi = q_match[0].item()
                                    new_k_int8[h, qi] = self.q_cache_k_int8[h, old_qi]
                                    new_v_int8[h, qi] = self.q_cache_v_int8[h, old_qi]
                                    new_k_scale[h, qi] = self.q_cache_k_scale[h, old_qi]
                                    new_k_zp[h, qi] = self.q_cache_k_zp[h, old_qi]
                                    new_v_scale[h, qi] = self.q_cache_v_scale[h, old_qi]
                                    new_v_zp[h, qi] = self.q_cache_v_zp[h, old_qi]
                                    retained = True
                            
                            if not retained:
                                # Gather fp16 KV from main cache
                                phys_mask = (self.logical_id_map[h] == wid.item())
                                phys_positions = phys_mask.nonzero(as_tuple=True)[0]
                                if len(phys_positions) >= self.omega:
                                    phys_positions = phys_positions[:self.omega]
                                    phys_positions = torch.clamp(phys_positions, 0, seq_len - 1)
                                    k_fp = past_key_values[0][0, h, phys_positions]  # [omega, D]
                                    v_fp = past_key_values[1][0, h, phys_positions]
                                else:
                                    k_fp = torch.zeros(self.omega, head_dim, device=device, dtype=dtype_fp)
                                    v_fp = torch.zeros(self.omega, head_dim, device=device, dtype=dtype_fp)
                                
                                # Path B: O(1) dict lookup for archived scales
                                archived = False
                                meta = self.q_retired_meta.get((wid.item(), h))
                                if meta is not None:
                                    ks = meta['k_scale'].to(device)
                                    kz = meta['k_zp'].to(device)
                                    k_q = torch.round((k_fp.unsqueeze(0) - kz) / ks).clamp(0, 255).to(torch.uint8)
                                    new_k_int8[h, qi] = k_q.squeeze(0)
                                    new_k_scale[h, qi, 0] = ks.squeeze(0)
                                    new_k_zp[h, qi, 0] = kz.squeeze(0)
                                    vs = meta['v_scale'].to(device)
                                    vz = meta['v_zp'].to(device)
                                    v_q = torch.round((v_fp.unsqueeze(0) - vz) / vs).clamp(0, 255).to(torch.uint8)
                                    new_v_int8[h, qi] = v_q.squeeze(0)
                                    new_v_scale[h, qi] = vs.squeeze(0)
                                    new_v_zp[h, qi] = vz.squeeze(0)
                                    archived = True
                                
                                if not archived:
                                    # Path C: FRESH — compute new per-window quantization
                                    k_4d = k_fp.unsqueeze(0).unsqueeze(0)  # [1,1,omega,D]
                                    v_4d = v_fp.unsqueeze(0).unsqueeze(0)
                                    kq, ks, kz = self._quantize_k_per_window(k_4d)
                                    vq, vs, vz = self._quantize_v_per_window(v_4d)
                                    new_k_int8[h, qi] = kq[0, 0]
                                    new_v_int8[h, qi] = vq[0, 0]
                                    new_k_scale[h, qi] = ks[0, 0]
                                    new_k_zp[h, qi] = kz[0, 0]
                                    new_v_scale[h, qi] = vs[0, 0]
                                    new_v_zp[h, qi] = vz[0, 0]
                    
                    self.q_cache_k_int8 = new_k_int8
                    self.q_cache_v_int8 = new_v_int8
                    self.q_cache_k_scale = new_k_scale
                    self.q_cache_k_zp = new_k_zp
                    self.q_cache_v_scale = new_v_scale
                    self.q_cache_v_zp = new_v_zp
                    self.q_cache_ids = new_q_loser_ids.float()
                    self.q_cache_scores = new_q_loser_scores
                elif self.q_windows_count > 0:
                    self.q_cache_k_int8 = None
                    self.q_cache_v_int8 = None
                    self.q_cache_k_scale = None
                    self.q_cache_k_zp = None
                    self.q_cache_v_scale = None
                    self.q_cache_v_zp = None
                    self.q_cache_ids = None
                    self.q_cache_scores = None

                # ---------------------------------------------------------
                # 5. PHYSICAL EVICTION (Explicit Construction)
                # ---------------------------------------------------------
                head_dim = past_key_values[0].shape[-1]
                dtype_fp = past_key_values[0].dtype
                
                # FIX (Bug 1): DO NOT recompute local_tokens_count here.
                # The value from the top of the block is correct (local_num_tokens or local_num).
                # The old formula `seq_len - (compressed_len + omega)` always gave 0
                # because compressed_len includes the local zone.
                new_compressed_len = self.sink_tokens + curr_k * self.omega
                new_seq_len = new_compressed_len + local_tokens_count
                
                new_k = torch.zeros(1, self.num_heads, new_seq_len, head_dim, device=device, dtype=dtype_fp)
                new_v = torch.zeros(1, self.num_heads, new_seq_len, head_dim, device=device, dtype=dtype_fp)
                new_logical_id_map = torch.zeros(self.num_heads, new_seq_len, device=device, dtype=torch.long)

                # To support token ledger:
                mapping = torch.full((self.num_heads, seq_len), -1.0, device=device, dtype=torch.float32)

                # Pre-build O(1) lookup: (head, wid_float) → first physical position
                # Replicates the old (logical_id_map[h] == wid).nonzero() scan but built once.
                _lid_list = self.logical_id_map.cpu().tolist()
                _phys_first = {}
                for _h in range(self.num_heads):
                    _counts = {}
                    _first_pos = {}
                    for _p, _w in enumerate(_lid_list[_h]):
                        if _w >= 0:
                            if _w not in _counts:
                                _counts[_w] = 0
                                _first_pos[_w] = _p
                            _counts[_w] += 1
                    for _w, _cnt in _counts.items():
                        if _cnt >= self.omega:
                            _phys_first[(_h, _w)] = _first_pos[_w]

                # Pre-build promoted data lookup: (head, wid_float) → tensor
                _prom_k = {(_h, int(w)): k for _h in range(self.num_heads) for w, k in promoted_q_data_k[_h]}
                _prom_v = {(_h, int(w)): v for _h in range(self.num_heads) for w, v in promoted_q_data_v[_h]}

                # Precompute local logical IDs vector (same for all heads)
                _local_lids = None
                if local_tokens_count > 0 and has_challenger:
                    _offsets = torch.arange(local_tokens_count, device=device, dtype=torch.long)
                    _local_lids = (last_id_val + 1) + (_offsets // self.omega)

                final_ids_list = final_ids.tolist()

                for h in range(self.num_heads):
                    # 1. Sinks
                    new_k[0, h, :self.sink_tokens] = past_key_values[0][0, h, :self.sink_tokens]
                    new_v[0, h, :self.sink_tokens] = past_key_values[1][0, h, :self.sink_tokens]
                    new_logical_id_map[h, :self.sink_tokens] = self.logical_id_map[h, :self.sink_tokens]
                    mapping[h, :self.sink_tokens] = torch.arange(self.sink_tokens, device=device, dtype=torch.float32)
                    
                    # 2. Sticky Zone — O(1) dict lookup replaces per-window nonzero scan
                    for i in range(curr_k):
                        wid = int(final_ids_list[h][i])
                        new_pos = self.sink_tokens + i * self.omega

                        old_pos = _phys_first.get((h, wid))
                        if old_pos is not None:
                            # From main cache
                            new_k[0, h, new_pos:new_pos+self.omega] = past_key_values[0][0, h, old_pos:old_pos+self.omega]
                            new_v[0, h, new_pos:new_pos+self.omega] = past_key_values[1][0, h, old_pos:old_pos+self.omega]
                            new_logical_id_map[h, new_pos:new_pos+self.omega] = wid
                            mapping[h, old_pos:old_pos+self.omega] = torch.arange(new_pos, new_pos+self.omega, device=device, dtype=torch.float32)
                        else:
                            # From q_cache (promoted) — O(1) dict lookup
                            p_k = _prom_k.get((h, wid))
                            p_v = _prom_v.get((h, wid))
                            if p_k is not None:
                                new_k[0, h, new_pos:new_pos+self.omega] = p_k
                                new_v[0, h, new_pos:new_pos+self.omega] = p_v
                            new_logical_id_map[h, new_pos:new_pos+self.omega] = wid
                            # Promoted tokens didn't have an old_phys position in the main cache, so no mapping needed
                    
                    # 3. Local Zone
                    if local_tokens_count > 0:
                        # FIX (Bug 1): The local zone always occupies the TAIL of the
                        # cache. old_local_start must point to the most-recent
                        # local_tokens_count tokens (inclusive of any new gen tokens
                        # appended since the last eviction).
                        # The old formula `compressed_len + omega` pointed past the
                        # end of the cache (out-of-bounds), silently copying nothing.
                        old_local_start = seq_len - local_tokens_count
                        new_local_start = new_compressed_len
                        new_k[0, h, new_local_start:] = past_key_values[0][0, h, old_local_start:old_local_start+local_tokens_count]
                        new_v[0, h, new_local_start:] = past_key_values[1][0, h, old_local_start:old_local_start+local_tokens_count]
                        
                        if _local_lids is not None:
                            new_logical_id_map[h, new_local_start:new_local_start + local_tokens_count] = _local_lids
                            
                        mapping[h, old_local_start:old_local_start+local_tokens_count] = torch.arange(new_local_start, new_local_start+local_tokens_count, device=device, dtype=torch.float32)
                
                self.logical_id_map = new_logical_id_map
                updated_kv = (new_k, new_v)
                
                if self.tracking_flag:
                    for head_idx in range(self.num_heads):
                        phys_col = 2 + head_idx
                        live_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1)
                        g_ids = torch.where(live_mask)[0]
                        old_phys = self.token_ledger[g_ids, phys_col].long()
                        
                        valid_old = (old_phys >= 0) & (old_phys < seq_len)
                        valid_old_phys = old_phys[valid_old]
                        valid_g_ids = g_ids[valid_old]
                        
                        # FIX: Record votes for ALL tokens that were valid in this cycle BEFORE eviction.
                        # This ensures dying tokens get their final 'omega' steps of attention recorded.
                        if len(valid_g_ids) > 0:
                            head_scores_acc = self.running_attention_votes[head_idx, valid_old_phys]
                            self.token_ledger[valid_g_ids, 2 + self.num_heads + head_idx] += head_scores_acc.float()
                            self.global_score_history[valid_g_ids, head_idx] += head_scores_acc.float()

                        # Now update the physical indices for the survivors
                        new_phys = mapping[head_idx, valid_old_phys]
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

    # REMOVED (Audit Bug 5): _update_window_scores_generation_vectorized was a dead method
    # never called by any code path. The active pipeline uses scatter_add_ via scoreboard.

    @staticmethod
    def _quantize_k_per_window(tensor):
        """Quantize K cache: per-channel per-window, with RoPE-paired dimension tying.
        Args:
            tensor: [H, W, omega, D] fp16 — already has RoPE applied
        Returns: (uint8_tensor[H,W,omega,D], scale[H,W,1,D], zp[H,W,1,D])
        """
        # Per-channel: reduce across token dim (omega) → [H, W, 1, D]
        t_min = tensor.amin(dim=2, keepdim=True)
        t_max = tensor.amax(dim=2, keepdim=True)
        
        # RoPE preservation: tie scales for paired dimensions (i, i+D//2)
        # LLaMA RoPE rotates pairs (x_i, x_{i+D/2}) in a 2D plane.
        # Using independent scales distorts the rotation angle.
        # We unify scale/zp for each pair so the plane is uniformly scaled.
        half_d = tensor.shape[-1] // 2
        t_min_h1, t_min_h2 = t_min[..., :half_d], t_min[..., half_d:]
        t_max_h1, t_max_h2 = t_max[..., :half_d], t_max[..., half_d:]
        t_min_tied = torch.min(t_min_h1, t_min_h2)
        t_max_tied = torch.max(t_max_h1, t_max_h2)
        t_min = torch.cat([t_min_tied, t_min_tied], dim=-1)
        t_max = torch.cat([t_max_tied, t_max_tied], dim=-1)
        
        scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
        quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
        return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)

    @staticmethod
    def _quantize_v_per_window(tensor):
        """Quantize V cache: per-token per-window.
        Args:
            tensor: [H, W, omega, D] fp16
        Returns: (uint8_tensor[H,W,omega,D], scale[H,W,omega,1], zp[H,W,omega,1])
        """
        # Per-token: reduce across head_dim (D) → [H, W, omega, 1]
        t_min = tensor.amin(dim=3, keepdim=True)
        t_max = tensor.amax(dim=3, keepdim=True)
        scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
        quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
        return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)

    @staticmethod
    def _dequantize_from_int8(int8_tensor, scale, zero_point):
        """Dequantize int8 tensor back to fp16.
        Works for any shape — scale/zp broadcast automatically.
        """
        return int8_tensor.to(scale.dtype) * scale + zero_point

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
        
        # Capture top-q losers BEFORE overwriting window_scores
        q_loser_ids = None
        q_loser_scores = None
        if self.q_windows_count > 0:
            total_valid = int(valid_mask.sum(dim=1).min().item())
            num_losers = total_valid - curr_k
            if num_losers > 0:
                loser_scores = scores.clone()
                loser_scores.scatter_(1, top_i, float("-inf"))
                q_count = min(self.q_windows_count, num_losers)
                q_top_v, q_top_i = torch.topk(loser_scores, q_count, dim=1, largest=True)
                q_loser_ids = torch.gather(ids, 1, q_top_i)
                q_loser_scores = q_top_v
        
        sort_idx = torch.argsort(kept_ids, dim=1)
        self.window_scores.fill_(float("nan"))
        self.window_scores[:, :curr_k, 0] = torch.gather(top_v, 1, sort_idx)
        self.window_scores[:, :curr_k, 1] = torch.gather(kept_ids, 1, sort_idx)
        self.window_scores[:, :curr_k, 2] = torch.gather(kept_orig, 1, sort_idx)
        return q_loser_ids, q_loser_scores

    def _create_mask_and_evict_from_kv_cache_prompt_stage(self, past_key_values, attn_scores, local_start_idx):
        seq_len, head_dim = (
            past_key_values[0].size(self.k_seq_dim),
            past_key_values[0].shape[-1],
        )
        
        device = self.window_scores.device
        
        sinks = self.sink_indices.unsqueeze(0).expand(self.num_heads, -1)
        
        # FIX (Bug 1): Only map genuinely valid (non-NaN) window score entries.
        # NaN→0 conversion previously resurrected evicted window-0 tokens.
        raw_w = self.window_scores[:, :self.k_windows, 1]
        valid_w_mask = ~torch.isnan(raw_w)
        sticky_w = torch.where(valid_w_mask, raw_w.long(), torch.zeros_like(raw_w, dtype=torch.long))
        all_window_tokens = self.window_to_token_map[sticky_w]  # [H, k_windows, omega]
        # Zero out tokens from invalid (NaN) slots so they don't enter the survivor set
        all_window_tokens = all_window_tokens * valid_w_mask.unsqueeze(-1).long()
        window_tokens = all_window_tokens.view(self.num_heads, -1)
        
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
            
        # FIX (Bug A): Use min-len truncation instead of max-len padding.
        # Padding duplicated the last KV entry, corrupting model attention.
        # Truncation drops only the highest-index local tail tokens, which
        # safely re-enter at the next eviction cycle.
        safe_len = min(len(u) for u in sorted_indices)
        final_indices = torch.stack(
            [u[:safe_len] for u in sorted_indices], dim=0
        )
        
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
        
        # Reserve a portion of the total budget for int8-quantized evicted tokens
        self.q_num = (total_token_budget * self.q_ratio) // 100
        
        # Int8 compression: ~2x more windows fit
        # FIX (Bug 3): omega is a token-count, not a byte overhead.
        # INT8 uses 1 byte per element vs bf16's 2 bytes per element.
        bf16_bytes = 2 * self.head_dim
        int8_bytes = self.head_dim  # 1 byte per element for INT8
        compression_ratio = bf16_bytes / int8_bytes
        effective_q_tokens = int(self.q_num * compression_ratio)
        self.q_windows_count = effective_q_tokens // self.omega
        
        available_sticky_tokens = total_token_budget - self.local_num - self.sink_tokens - self.q_num
        self.k_windows = max(0, available_sticky_tokens // self.omega)

    def _clean_scores(self):
        # Hard resets for cross-document isolation
        self.gen_step = self.num_of_tokens_without_eviction = 0
        self.tokens_since_last_review = 0
        if hasattr(self, "running_attention_votes"):
            self.running_attention_votes.zero_()
        self.window_scores.fill_(float("nan"))
        self.global_token_counter.zero_()
        self.local_history.zero_()
        self.token_ledger.fill_(-1.0)
        self.global_score_history.fill_(-1.0)
        self._prefill_done = False
        self.logical_id_map = None
        self._dynamic_local_count = 0
        self.prompt_boundary = [-1 for _ in range(self.num_heads)]
        self.prefill_attention_matrix = None  # FIX (Audit Bug 4): prevent CPU RAM leak across documents
        # Reset q-cache state
        self.q_cache_k_int8 = None
        self.q_cache_v_int8 = None
        self.q_cache_k_scale = None
        self.q_cache_k_zp = None
        self.q_cache_v_scale = None
        self.q_cache_v_zp = None
        self.q_cache_ids = None
        self.q_cache_scores = None
        self.q_windows_count = 0
        self.q_num = 0
        # FIX (Bug 3): MUST clear q_retired_meta on each document reset.
        # window_id values restart from 0 for every new document, so stale
        # entries from previous documents will falsely match new windows with
        # the same ID, applying incorrect float16 scale/zp to their quantization.
        self.q_retired_meta = {}