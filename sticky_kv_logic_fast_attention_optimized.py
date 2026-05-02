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
        self._prefill_done = False  # Tracks whether initial prefill has completed

        # Q_RATIO: percentage of total cache budget reserved for quantized evicted tokens
        try:
            from sticky_config import Q_RATIO
            self.q_ratio = Q_RATIO
        except ImportError:
            self.q_ratio = 0

        # Quantization bit-width: 8 (int8) or 4 (packed int4). Defaults to 8.
        try:
            from sticky_config import QUANTIZATION_BIT_WIDTH
            self.quant_bit_width = QUANTIZATION_BIT_WIDTH
        except ImportError:
            self.quant_bit_width = 8
        
        self.q_num = 0
        self.q_windows_count = 0
        
        # head_dim needed for compression ratio calculation
        if config is not None and hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
            self.head_dim = config.hidden_size // config.num_attention_heads
        else:
            self.head_dim = 64  # Llama 3.2 1B default
        
        # Quantized side-cache — per-WINDOW quantization (lazy-initialized at prefill).
        # INT8: q_cache_k_quant shape is [H, W, omega, D] uint8.
        # INT4: q_cache_k_quant shape is [H, W, omega, D//2] uint8 (two nibbles packed per byte).
        self.q_cache_k_quant = None
        self.q_cache_v_quant = None
        self.q_cache_k_scale = None       # [H, W, 1, D] float16
        self.q_cache_k_zp = None          # [H, W, 1, D] float16
        self.q_cache_v_scale = None       # [H, W, omega, 1] float16
        self.q_cache_v_zp = None          # [H, W, omega, 1] float16
        self.q_cache_ids = None
        self.q_cache_scores = None
        self.q_retired_meta = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get context window size dynamically to size trackers perfectly
        if config is not None and hasattr(config, "max_position_embeddings"):
            max_context = config.max_position_embeddings
            # Setup how many possible blocks of OMEGA size could exist
            max_windows = (
                (max_context - self.sink_tokens) // self.omega if max_context > self.sink_tokens else 1
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
        # Dim 2 is [Cumulative Score, Current Window ID, Original Window ID]
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
        
        # FIX (L5): Initialize to prevent AttributeError if accessed before _clean_scores
        self.prefill_attention_matrix = None
        
        # Precomputed quant byte width (also re-set by _update_k_win_and_local_num and _clean_scores)
        self._quant_bytes_len = self.head_dim if self.quant_bit_width == 8 else (self.head_dim // 2)

    def _find_logical_window_span(self, h, wid_val, seq_len):
        positions = (self.logical_id_map[h] == int(wid_val)).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            return None

        start = int(positions.min().item())
        end = int(positions.max().item()) + 1

        expected = torch.arange(start, end, device=positions.device)
        if positions.numel() != (end - start) or not torch.equal(positions, expected):
            raise RuntimeError(f"Logical window {wid_val} has non-contiguous physical positions")

        if end - start != self.omega:
            return None  # defer partial windows; do not promote them into sticky/q-cache

        if end > seq_len:
            return None
        return start, end

    def _gather_window_from_current_kv(self, past_key_values, h, wid_val, *, seq_len):
        span = self._find_logical_window_span(h, wid_val, seq_len)
        if span is None:
            return None
        start, end = span
        return (
            past_key_values[0][0, h, start:end],
            past_key_values[1][0, h, start:end],
        )

    def __call__(self, past_key_values, attn_score_cache, full_attn_scores=None, q_len=None, q_attn_scores=None):
        bsz, q_heads, q_len_cache, kv_seq_len = attn_score_cache.shape
        
        q_len = q_len if q_len is not None else q_len_cache
        num_new_tokens = q_len

        # Defines full present context length directly from KV cache
        seq_len = past_key_values[0].size(self.k_seq_dim) if past_key_values is not None else 0
        # === Inside __call__ ===
        
        # Keeps counter consistent globally 
        if not self._prefill_done:
            # Defensive: ensure counter is zeroed at the start of a new document's prefill
            # in case _clean_cache() was missed between documents.
            self.global_token_counter.zero_()
            self.global_token_counter += q_len
        else:
            self.global_token_counter += 1
        
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


        if not self._prefill_done:  # Prompt Stage Active
            # Use the clamped CFO allocator value — already handles both fixed and ratio modes.
            # self.local_num = min(target_local_tokens, remaining) is set by _update_k_win_and_local_num.
            local_tokens_count = self.local_num
            
            # Application boundary: Exclude Sinks, Local windows, and Observation windows
            # Define logical point where eviction boundaries MUST end and local bubble starts
            score_end = max(self.sink_tokens, seq_len - local_tokens_count)
            
            # Find complete blocks capable of being tracked and potentially evicted
            num_windows = max(0, (score_end - self.sink_tokens) // self.omega)
            
            # --- REMAINDER LEAKAGE FIX ---
            # OPT-4: Combine first snap + min; keep the omega-aligned recalculation
            # because attn_score_cache.shape[3] may not be omega-aligned.
            score_end = min(
                self.sink_tokens + (num_windows * self.omega),
                attn_score_cache.shape[3]
            )
            num_windows = (score_end - self.sink_tokens) // self.omega
            score_end = self.sink_tokens + (num_windows * self.omega)
            
            # Ceiling division in _update_k_win_and_local_num already absorbs the partial
            # window into sticky/q-cache slots. Keep the local zone at the CFO-allocated value.
            # (score_end is still used below to slice attn_score_cache — do not remove that.)
            self._dynamic_local_count = self.local_num
            local_tokens_count = self.local_num
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
            # local_history is already zeroed by _clean_scores between documents
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

            q_loser_ids, q_loser_scores = self._evict_from_window_scores()

            # --- Q-CACHE: Capture and quantize loser KV data ---
            if q_loser_ids is not None:
                q_count = q_loser_ids.shape[1]
                q_phys_indices = self.window_to_token_map[q_loser_ids.long()]
                q_phys_flat = q_phys_indices.reshape(self.num_heads, -1)
                q_phys_flat = torch.clamp(q_phys_flat, 0, seq_len - 1)
                hd = past_key_values[0].shape[-1]
                gather_q = q_phys_flat.unsqueeze(-1).expand(-1, -1, hd)
                q_k_data = torch.gather(past_key_values[0][0], 1, gather_q)
                q_v_data = torch.gather(past_key_values[1][0], 1, gather_q)
                # Reshape to per-window layout [H, W, omega, D]
                q_k_data = q_k_data.view(self.num_heads, q_count, self.omega, hd)
                q_v_data = q_v_data.view(self.num_heads, q_count, self.omega, hd)
                self.q_cache_k_quant, self.q_cache_k_scale, self.q_cache_k_zp = self._quantize_k_per_window(q_k_data, self.quant_bit_width)
                self.q_cache_v_quant, self.q_cache_v_scale, self.q_cache_v_zp = self._quantize_v_per_window(q_v_data, self.quant_bit_width)
                self.q_cache_ids = q_loser_ids.float()
                self.q_cache_scores = q_loser_scores

            # Tracking logic removed per Fast Attention v2 instructions.

            updated_kv, survivor_ids = self._create_mask_and_evict_from_kv_cache_prompt_stage(
                past_key_values, attn_score_cache, score_end
            )
            
            # THE FIX: Store explicitly mapped Logical IDs, completely eliminating coordinate math later
            self.logical_id_map = torch.where(
                survivor_ids >= self.sink_tokens,
                (survivor_ids - self.sink_tokens) // self.omega,
                torch.full_like(survivor_ids, -1)  # Sinks get -1
            ).to(torch.long)

            self._prefill_done = True  
            return updated_kv

        else:  # Generation Stage
            device = self.window_scores.device
            
            # FIX (B4): Guard against generation being called before prefill constructed logical_id_map
            if self.logical_id_map is None:
                return past_key_values
            
            # 1. ACCUMULATE VOTES
            # Takes the latest generation attention 1D slices across the cache tensor and adds dynamically continuously
            self.running_attention_votes[:, :seq_len] += attn_score_cache[0, :, 0, :seq_len]
            self.tokens_since_last_review += 1
            
            # Accumulate q-cache attention scores from joint softmax
            if q_attn_scores is not None and self.q_cache_scores is not None and self.q_cache_ids is not None:
                q_per_token = q_attn_scores[0, :, 0, :]
                q_tokens_total = self.q_cache_ids.shape[1] * self.omega
                if q_per_token.shape[1] >= q_tokens_total:
                    q_per_token = q_per_token[:, :q_tokens_total]
                    q_per_window = q_per_token.view(self.num_heads, self.q_cache_ids.shape[1], self.omega).sum(dim=2)
                    self.q_cache_scores = self.q_cache_scores + q_per_window.to(self.q_cache_scores.dtype)
                else:
                    print(f"WARNING [Layer {self.layer_idx}]: q-cache score shape mismatch: got {q_per_token.shape[1]}, expected >= {q_tokens_total}. Skipping q-cache score accumulation.")
            
            # 2. PERIODIC EVALUATION
            # Process eviction constraints specifically when the tracker hits OMEGA boundary thresholds
            if self.tokens_since_last_review == self.omega:
                # FIX (Issue 5): Removed unconditional _prof block and syncs
                # to eliminate latency spikes and standard error pollution.

                # FIX (Bug 1): Use the persisted dynamic local count that absorbed
                # the prefill remainder, instead of re-reading the static config value.
                local_tokens_count = self._dynamic_local_count
                
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
                    # FIX (L3): Remove .clamp(min=0) and instead filter out negative IDs
                    # to prevent artificially inflating Window-0 scores in early generation.
                    raw_new_lids = (self.num_of_tokens_without_eviction - self.omega + js - self.sink_tokens) // self.omega
                    valid_new = (raw_new_lids >= 0) & (raw_new_lids < scoreboard.shape[1])
                    if valid_new.any():
                        scoreboard.scatter_add_(1, raw_new_lids[valid_new].unsqueeze(0).expand(self.num_heads, -1), new_tok_votes[:, valid_new])
                
                # Determine current valid old competitors
                valid_mask = ~torch.isnan(self.window_scores[:, :, 1])
                valid_old_windows = min(self.k_windows, int(valid_mask.sum(dim=1).min().item()))
                valid_old_windows = min(self.k_windows, int(valid_mask.sum(dim=1).min().item()))

                raw_ids = self.window_scores[:, :valid_old_windows, 1]
                raw_scores = self.window_scores[:, :valid_old_windows, 0]
                # FIX (B2): Track which slots are genuinely registered vs NaN-padded empty slots
                is_valid_slot = ~torch.isnan(raw_ids)
                
                # OPT-5: Use nan_to_num once for safe gather index, then in-place masked_fill_
                # to avoid allocating a second [H, k_windows] tensor via torch.where.
                old_ids = raw_ids.nan_to_num(nan=0.0)
                safe_ids = old_ids.long()
                old_scores_hist = torch.nan_to_num(raw_scores, nan=0.0)

                # Collect new mass generated strictly mapped dynamically 
                if valid_old_windows > 0:
                    old_w_gen_scores = scoreboard.gather(1, safe_ids)
                    # FIX (B2): Zero out phantom Window 0 scores gathered from NaN→0 converted empty slots
                    old_w_gen_scores.masked_fill_(~is_valid_slot, 0.0)  # in-place, no extra allocation
                else:
                    old_w_gen_scores = old_scores_hist.new_zeros(old_scores_hist.shape)
                old_scores = old_scores_hist + old_w_gen_scores

                # ---------------------------------------------------------
                # SCORE THE CHALLENGER (The Emerging Window)
                # ---------------------------------------------------------
                # Calculate the Logical ID of the window that just fell out of the local bubble
                raw_last_id_val = (self.num_of_tokens_without_eviction - self.sink_tokens - local_tokens_count) // self.omega - 1
                # FIX (BUG-4): Clamp challenger ID to valid window range
                max_valid_wid = self.window_scores.shape[1] - 1
                raw_last_id_val = min(raw_last_id_val, max_valid_wid)
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
                    # OPT-A: scatter (non-inplace) avoids an explicit clone()+scatter_() pair.
                    # Keeps correct cross-head min-count to guard against pre-existing -inf slots.
                    loser_scores = competing_scores.scatter(1, top_i, float("-inf"))
                    num_remaining = int((loser_scores > float("-inf")).sum(dim=1).min().item())
                    if num_remaining > 0:
                        q_count = min(self.q_windows_count, num_remaining)
                        q_top_v, q_top_i = torch.topk(loser_scores, q_count, dim=1, largest=True)
                        new_q_loser_ids = torch.gather(competing_ids, 1, q_top_i)
                        new_q_loser_scores = q_top_v

                # --- Q-CACHE: Handle promotions (q-cache → main cache) ---
                promoted_q_data_k = {h: [] for h in range(self.num_heads)}
                promoted_q_data_v = {h: [] for h in range(self.num_heads)}
                if self.q_cache_ids is not None:
                    # Strictly Per-head membership: [H, q_windows, 1] == [H, 1, curr_k] -> [H, q_windows]
                    promo_mask = (
                        self.q_cache_ids.long().unsqueeze(2) == final_ids.long().unsqueeze(1)
                    ).any(dim=2)
                    
                    # OPT-P1: Batch-dequantize ALL q-cache windows at once (2 kernel
                    # launches total) instead of per-head per-window calls (~80 launches).
                    if promo_mask.any():
                        all_k_deq = self._dequantize_from_quant(
                            self.q_cache_k_quant, self.q_cache_k_scale,
                            self.q_cache_k_zp, self.quant_bit_width)
                        all_v_deq = self._dequantize_from_quant(
                            self.q_cache_v_quant, self.q_cache_v_scale,
                            self.q_cache_v_zp, self.quant_bit_width)
                        # One .nonzero() + .tolist() sync replaces H separate syncs
                        promo_heads, promo_qis = promo_mask.nonzero(as_tuple=True)
                        promo_wids = self.q_cache_ids[promo_heads, promo_qis].long().tolist()
                        promo_heads_list = promo_heads.tolist()
                        promo_qis_list = promo_qis.tolist()
                        for ph, pqi, pwid in zip(promo_heads_list, promo_qis_list, promo_wids):
                            promoted_q_data_k[ph].append((pwid, all_k_deq[ph, pqi]))
                            promoted_q_data_v[ph].append((pwid, all_v_deq[ph, pqi]))
                            self.q_retired_meta[(pwid, ph)] = {
                                'k_scale': self.q_cache_k_scale[ph, pqi].detach().clone(),
                                'k_zp': self.q_cache_k_zp[ph, pqi].detach().clone(),
                                'v_scale': self.q_cache_v_scale[ph, pqi].detach().clone(),
                                'v_zp': self.q_cache_v_zp[ph, pqi].detach().clone(),
                            }
                        del all_k_deq, all_v_deq

                self.window_scores.fill_(float("nan"))
                self.window_scores[:, :curr_k, 0] = final_v
                self.window_scores[:, :curr_k, 1] = final_ids
                self.window_scores[:, :curr_k, 2] = final_ids

                # ---------------------------------------------------------
                # 4. UPDATE LOCAL HISTORY
                # FIX (Bug 1+2): The old formula `seq_len - (compressed_len + omega)`
                # always evaluates to 0 because compressed_len = logical_id_map.shape[1]
                # which already includes the local zone from the previous cycle.
                # The local zone always has exactly `local_tokens_count` tokens.
                # The scoreboard already routes local-zone votes correctly via
                # logical_id_map, so we read from it directly.
                # ---------------------------------------------------------
                # FIX (Bug 2): Use ceiling division so partial-window votes are
                # preserved in local_history instead of being wiped by zero_().
                local_windows = (local_tokens_count + self.omega - 1) // self.omega

                if local_windows > 0:
                    # FIX (B6): Derive local_id_start independently of has_challenger.
                    # When has_challenger is False (very short prompts / early cycles),
                    # votes for local-zone windows were previously discarded entirely.
                    # Use the token counter to compute the correct start ID unconditionally.
                    if has_challenger:
                        local_id_start = last_id_val + 1
                    else:
                        local_id_start = max(0, (self.num_of_tokens_without_eviction - self.sink_tokens - local_tokens_count) // self.omega)
                    ids = torch.arange(local_id_start, local_id_start + local_windows, device=device, dtype=torch.long)

                    valid = (ids >= 0) & (ids < self.local_history.shape[1])
                    if valid.any():
                        ids_valid = ids[valid]
                        # Read votes directly from scoreboard — local zone votes are
                        # already correctly routed here via scatter_add_ + logical_id_map
                        self.local_history[:, ids_valid] += scoreboard[:, ids_valid]

                # If r_ratio is 100, skip physical eviction
                if self.total_cache_ratio == 100:
                    self.running_attention_votes[:, :seq_len].zero_()  # zero only the live slice
                    self.tokens_since_last_review = 0
                    return past_key_values
                
                # OPT-P2: Precompute block-boundary data ONCE for both q-cache rebuild
                # and physical eviction. Avoids recomputing and enables vectorized window lookup.
                _pre_compressed_len = self.logical_id_map.shape[1]
                _pre_num_old_blocks = max(0, (_pre_compressed_len - self.sink_tokens - local_tokens_count) // self.omega)
                if _pre_num_old_blocks > 0:
                    _pre_block_starts = self.sink_tokens + torch.arange(
                        _pre_num_old_blocks, device=device, dtype=torch.long
                    ) * self.omega
                    _pre_block_wids = self.logical_id_map[:, _pre_block_starts]  # [H, num_old_blocks]
                else:
                    _pre_block_starts = torch.zeros(0, device=device, dtype=torch.long)
                    _pre_block_wids = torch.zeros(self.num_heads, 0, device=device, dtype=torch.long)

                # --- Q-CACHE: Rebuild with ZERO-DEGRADATION routing ---
                if new_q_loser_ids is not None and self.q_windows_count > 0:
                    new_q_count = new_q_loser_ids.shape[1]
                    head_dim = past_key_values[0].shape[-1]
                    dtype_fp = past_key_values[0].dtype
                    
                    # OPT-6: Use precomputed value from _update_k_win_and_local_num
                    quant_bytes_len = self._quant_bytes_len
                    new_k_quant = torch.zeros(self.num_heads, new_q_count, self.omega, quant_bytes_len, device=device, dtype=torch.uint8)
                    new_v_quant = torch.zeros(self.num_heads, new_q_count, self.omega, quant_bytes_len, device=device, dtype=torch.uint8)
                    new_k_scale = torch.zeros(self.num_heads, new_q_count, 1, head_dim, device=device, dtype=dtype_fp)
                    new_k_zp = torch.zeros(self.num_heads, new_q_count, 1, head_dim, device=device, dtype=dtype_fp)
                    new_v_scale = torch.zeros(self.num_heads, new_q_count, self.omega, 1, device=device, dtype=dtype_fp)
                    new_v_zp = torch.zeros(self.num_heads, new_q_count, self.omega, 1, device=device, dtype=dtype_fp)
                    
                    # FIX B: Vectorized q-cache rebuild — eliminates O(H × q_count) Python
                    # loop and .item() CPU syncs. Three routing paths preserved:
                    # A) RETAINED: was in old q-cache, copy raw int8 directly
                    # B) FRESH from main cache: gather via block-boundary arithmetic
                    # C) ARCHIVED meta: per-element fallback (rare, typically 0-3 items)
                    
                    # --- Path A: Batch-identify retained windows ---
                    if self.q_cache_ids is not None:
                        # retained_mask: [H, new_q_count, old_q_count] — True where IDs match
                        retained_match = (new_q_loser_ids.unsqueeze(2) == self.q_cache_ids.unsqueeze(1))  # [H, new, old]
                        retained_any = retained_match.any(dim=2)  # [H, new_q_count] — is this slot retained?
                        # For retained slots, find which old slot they came from
                        retained_old_idx = retained_match.to(torch.uint8).argmax(dim=2)  # [H, new_q_count]
                        
                        # Batch copy for all retained slots using advanced indexing
                        h_idx = torch.arange(self.num_heads, device=device).unsqueeze(1).expand_as(retained_old_idx)
                        # Only copy where retained_any is True
                        if retained_any.any():
                            new_k_quant[retained_any] = self.q_cache_k_quant[h_idx[retained_any], retained_old_idx[retained_any]]
                            new_v_quant[retained_any] = self.q_cache_v_quant[h_idx[retained_any], retained_old_idx[retained_any]]
                            new_k_scale[retained_any] = self.q_cache_k_scale[h_idx[retained_any], retained_old_idx[retained_any]]
                            new_k_zp[retained_any] = self.q_cache_k_zp[h_idx[retained_any], retained_old_idx[retained_any]]
                            new_v_scale[retained_any] = self.q_cache_v_scale[h_idx[retained_any], retained_old_idx[retained_any]]
                            new_v_zp[retained_any] = self.q_cache_v_zp[h_idx[retained_any], retained_old_idx[retained_any]]
                    else:
                        retained_any = torch.zeros(self.num_heads, new_q_count, device=device, dtype=torch.bool)
                    
                    # --- Paths B+C: Handle non-retained windows ---
                    not_retained = ~retained_any  # [H, new_q_count]
                    if not_retained.any():
                        # OPT-P2: Batch-gather all non-retained windows from main cache
                        # using precomputed block-boundary data. One .tolist() sync replaces
                        # H × q_count individual .item() syncs.
                        nr_h, nr_qi = not_retained.nonzero(as_tuple=True)
                        nr_wids = new_q_loser_ids[nr_h, nr_qi].long()
                        nr_count = nr_h.shape[0]
                        
                        # Locate physical positions via block-boundary matching
                        if _pre_num_old_blocks > 0 and nr_count > 0:
                            # [nr_count, num_old_blocks] — block wids for each item's head
                            nr_block_wids = _pre_block_wids[nr_h]
                            nr_match = (nr_block_wids == nr_wids.unsqueeze(1))
                            nr_found = nr_match.any(dim=1)          # [nr_count]
                            nr_slot = nr_match.to(torch.uint8).argmax(dim=1)  # [nr_count]
                            nr_phys_start = self.sink_tokens + nr_slot * self.omega
                        else:
                            nr_found = torch.zeros(nr_count, device=device, dtype=torch.bool)
                            nr_phys_start = torch.zeros(nr_count, device=device, dtype=torch.long)
                        
                        # Batch-gather KV data for all found windows at once
                        offsets_om = torch.arange(self.omega, device=device, dtype=torch.long)
                        head_dim_val = past_key_values[0].shape[-1]
                        
                        # Separate Path C (archived meta) from Path B (fresh quantize)
                        nr_h_list = nr_h.tolist()
                        nr_qi_list = nr_qi.tolist()
                        nr_wids_list = nr_wids.tolist()
                        nr_found_list = nr_found.tolist()
                        
                        path_b_indices = []  # indices into nr_* arrays
                        path_c_indices = []
                        for idx in range(nr_count):
                            if (nr_wids_list[idx], nr_h_list[idx]) in self.q_retired_meta:
                                path_c_indices.append(idx)
                            else:
                                path_b_indices.append(idx)
                        
                        # --- Path B: Batch-gather + batch-quantize fresh windows ---
                        if path_b_indices:
                            pb_indices = torch.tensor(path_b_indices, device=device, dtype=torch.long)
                            pb_h = nr_h[pb_indices]
                            pb_qi = nr_qi[pb_indices]
                            pb_found = nr_found[pb_indices]
                            pb_phys = nr_phys_start[pb_indices]
                            pb_count = len(path_b_indices)
                            
                            # Gather omega tokens per window from past KV
                            # pb_positions: [pb_count, omega]
                            pb_positions = pb_phys.unsqueeze(1) + offsets_om.unsqueeze(0)
                            pb_positions = pb_positions.clamp(0, seq_len - 1)
                            # Advanced indexing: select per-head slices then gather
                            pb_gather_idx = pb_positions.unsqueeze(-1).expand(-1, -1, head_dim_val)
                            # [pb_count, seq, D] — select each item's head
                            pb_k_heads = past_key_values[0][0, pb_h]
                            pb_v_heads = past_key_values[1][0, pb_h]
                            # [pb_count, omega, D]
                            pb_k_data = torch.gather(pb_k_heads, 1, pb_gather_idx)
                            pb_v_data = torch.gather(pb_v_heads, 1, pb_gather_idx)
                            
                            # Zero-fill windows not found in main cache
                            if not pb_found.all():
                                missing = ~pb_found
                                pb_k_data[missing] = 0
                                pb_v_data[missing] = 0
                            
                            # Batch quantize: [pb_count, 1, omega, D]
                            pb_k_4d = pb_k_data.unsqueeze(1)
                            pb_v_4d = pb_v_data.unsqueeze(1)
                            pb_kq, pb_ks, pb_kz = self._quantize_k_per_window(pb_k_4d, self.quant_bit_width)
                            pb_vq, pb_vs, pb_vz = self._quantize_v_per_window(pb_v_4d, self.quant_bit_width)
                            # Scatter results back: pb_kq is [pb_count, 1, omega, quant_bytes_len]
                            new_k_quant[pb_h, pb_qi] = pb_kq[:, 0]
                            new_v_quant[pb_h, pb_qi] = pb_vq[:, 0]
                            new_k_scale[pb_h, pb_qi] = pb_ks[:, 0]
                            new_k_zp[pb_h, pb_qi] = pb_kz[:, 0]
                            new_v_scale[pb_h, pb_qi] = pb_vs[:, 0]
                            new_v_zp[pb_h, pb_qi] = pb_vz[:, 0]
                            del pb_k_heads, pb_v_heads, pb_k_data, pb_v_data
                        
                        # --- Path C: Archived meta (rare, typically 0-3 items) ---
                        for idx in path_c_indices:
                            h_val = nr_h_list[idx]
                            qi_val = nr_qi_list[idx]
                            wid_val = nr_wids_list[idx]
                            
                            # Gather this window's KV from main cache
                            if nr_found_list[idx]:
                                ps = int(nr_phys_start[idx].item())
                                k_fp = past_key_values[0][0, h_val, ps:ps+self.omega]
                                v_fp = past_key_values[1][0, h_val, ps:ps+self.omega]
                            else:
                                k_fp = torch.zeros(self.omega, head_dim, device=device, dtype=dtype_fp)
                                v_fp = torch.zeros(self.omega, head_dim, device=device, dtype=dtype_fp)
                            
                            meta = self.q_retired_meta[(wid_val, h_val)]
                            ks = meta['k_scale'].to(device)
                            kz = meta['k_zp'].to(device)
                            vs = meta['v_scale'].to(device)
                            vz = meta['v_zp'].to(device)
                            if self.quant_bit_width == 8:
                                k_q = torch.round((k_fp.unsqueeze(0) - kz) / ks).clamp(0, 255).to(torch.uint8)
                                # FIX (M3): Do NOT unsqueeze v_fp — vs is [omega,1], v_fp is [omega,D].
                                v_q = torch.round((v_fp - vz) / vs).clamp(0, 255).to(torch.uint8)
                                new_k_quant[h_val, qi_val] = k_q.squeeze(0)
                                new_v_quant[h_val, qi_val] = v_q
                            else:
                                k_q = torch.round((k_fp.unsqueeze(0) - kz) / ks).clamp(0, 15).to(torch.uint8)
                                # FIX (M3): Same unsqueeze fix for int4 path.
                                v_q = torch.round((v_fp - vz) / vs).clamp(0, 15).to(torch.uint8)
                                new_k_quant[h_val, qi_val] = ((k_q[..., 0::2] << 4) | k_q[..., 1::2]).squeeze(0)
                                new_v_quant[h_val, qi_val] = (v_q[..., 0::2] << 4) | v_q[..., 1::2]
                            new_k_scale[h_val, qi_val, 0] = ks.squeeze(0)
                            new_k_zp[h_val, qi_val, 0] = kz.squeeze(0)
                            # FIX (BUG-14): Use view instead of squeeze to avoid
                            # dimension collapse when omega==1
                            new_v_scale[h_val, qi_val] = vs.view(self.omega, 1)
                            new_v_zp[h_val, qi_val] = vz.view(self.omega, 1)
                    
                    self.q_cache_k_quant = new_k_quant
                    self.q_cache_v_quant = new_v_quant
                    self.q_cache_k_scale = new_k_scale
                    self.q_cache_k_zp = new_k_zp
                    self.q_cache_v_scale = new_v_scale
                    self.q_cache_v_zp = new_v_zp
                    self.q_cache_ids = new_q_loser_ids.float()
                    self.q_cache_scores = new_q_loser_scores
                elif self.q_windows_count > 0:
                    self.q_cache_k_quant = None
                    self.q_cache_v_quant = None
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
                # The value from line 387 is correct (local_num_tokens or local_num).
                # The old formula `seq_len - (compressed_len + omega)` always gave 0
                # because compressed_len includes the local zone.
                new_compressed_len = self.sink_tokens + curr_k * self.omega
                new_seq_len = new_compressed_len + local_tokens_count
                
                new_k = torch.zeros(1, self.num_heads, new_seq_len, head_dim, device=device, dtype=dtype_fp)
                new_v = torch.zeros(1, self.num_heads, new_seq_len, head_dim, device=device, dtype=dtype_fp)
                # FIX (BUG-3): Initialize to -1 so unset entries don't alias valid Window 0
                new_logical_id_map = torch.full((self.num_heads, new_seq_len), -1, device=device, dtype=torch.long)
                
                # OPT (Change 1): Replace CPU-bound _phys_first dict builder with
                # block-boundary arithmetic on GPU. Instead of iterating over every
                # token in logical_id_map (seq_len * num_heads iterations), we sample
                # only block start positions (num_old_blocks elements) to find each
                # window's logical ID, then do a tiny [H, curr_k, num_old_blocks]
                # broadcast match. Zero logic change — same windows, same positions.
                # OPT-P2: Reuse precomputed block-boundary data from q-cache rebuild.
                compressed_len = _pre_compressed_len
                num_old_blocks = _pre_num_old_blocks
                
                if num_old_blocks > 0:
                    block_wids = _pre_block_wids
                    # Match final_ids against block_wids: [H, curr_k, num_old_blocks]
                    # This tensor is tiny: num_heads * curr_k * num_old_blocks (e.g. 8*30*30 = 7200)
                    match = (block_wids.unsqueeze(1) == final_ids.unsqueeze(2))
                    found_in_main = match.any(dim=2)        # [H, curr_k] — is window in main cache?
                    slot_idx = match.to(torch.uint8).argmax(dim=2)  # [H, curr_k] — which old block?
                    first_phys = self.sink_tokens + slot_idx * self.omega  # physical start position
                else:
                    found_in_main = torch.zeros(self.num_heads, curr_k, device=device, dtype=torch.bool)
                    first_phys = torch.zeros(self.num_heads, curr_k, device=device, dtype=torch.long)

                # Pre-build promoted data lookup: (head, wid_float) → tensor
                _prom_k = {(_h, int(w)): k for _h in range(self.num_heads) for w, k in promoted_q_data_k[_h]}
                _prom_v = {(_h, int(w)): v for _h in range(self.num_heads) for w, v in promoted_q_data_v[_h]}

                # Precompute local logical IDs vector (same for all heads)
                # FIX (BUG-3): Compute unconditionally — when has_challenger is False,
                # derive the start window ID from the token counter so local zone
                # entries don't stay at -1 (or the old zero which aliased Window 0).
                _local_lids = None
                if local_tokens_count > 0:
                    _offsets = torch.arange(local_tokens_count, device=device, dtype=torch.long)
                    local_start_wid = max(0, (self.num_of_tokens_without_eviction - self.sink_tokens - local_tokens_count) // self.omega)
                    _local_lids = local_start_wid + (_offsets // self.omega)
                
                # 1. Sinks
                new_k[0, :, :self.sink_tokens] = past_key_values[0][0, :, :self.sink_tokens]
                new_v[0, :, :self.sink_tokens] = past_key_values[1][0, :, :self.sink_tokens]
                new_logical_id_map[:, :self.sink_tokens] = self.logical_id_map[:, :self.sink_tokens]
                
                # 2. Sticky Zone
                if found_in_main.any():
                    target_starts = self.sink_tokens + torch.arange(curr_k, device=device, dtype=torch.long) * self.omega
                    target_starts = target_starts.unsqueeze(0).expand(self.num_heads, -1)
                    offsets = torch.arange(self.omega, device=device, dtype=torch.long)
                    
                    phys_gather = (first_phys.unsqueeze(2) + offsets).view(self.num_heads, -1)
                    target_scatter = (target_starts.unsqueeze(2) + offsets).view(self.num_heads, -1)
                    
                    mask = found_in_main.unsqueeze(2).expand(-1, -1, self.omega).reshape(self.num_heads, -1)
                    
                    valid_phys = phys_gather[mask]
                    valid_target = target_scatter[mask]
                    
                    head_indices = torch.arange(self.num_heads, device=device).unsqueeze(1).expand(-1, curr_k * self.omega)
                    valid_heads = head_indices[mask]
                    
                    new_k[0, valid_heads, valid_target] = past_key_values[0][0, valid_heads, valid_phys]
                    new_v[0, valid_heads, valid_target] = past_key_values[1][0, valid_heads, valid_phys]
                    
                    flat_final_ids = final_ids.unsqueeze(2).expand(-1, -1, self.omega).reshape(self.num_heads, -1)
                    new_logical_id_map[valid_heads, valid_target] = flat_final_ids[mask].long()

                not_in_main_mask = ~found_in_main
                if not_in_main_mask.any():
                    heads, indices = not_in_main_mask.nonzero(as_tuple=True)
                    # OPT-2: Extract all wid_vals in one .tolist() call instead of
                    # O(count) per-iteration .item() CPU-GPU syncs.
                    all_wid_vals = final_ids[heads, indices].long().tolist()
                    heads_list = heads.tolist()
                    indices_list = indices.tolist()
                    for h_idx, i_idx, wid_val in zip(heads_list, indices_list, all_wid_vals):
                        new_pos = self.sink_tokens + i_idx * self.omega
                        
                        p_k = _prom_k.get((h_idx, wid_val))
                        p_v = _prom_v.get((h_idx, wid_val))
                        if p_k is not None:
                            new_k[0, h_idx, new_pos:new_pos+self.omega] = p_k
                            new_v[0, h_idx, new_pos:new_pos+self.omega] = p_v
                            new_logical_id_map[h_idx, new_pos:new_pos+self.omega] = wid_val
                        else:
                            span = self._find_logical_window_span(h_idx, wid_val, seq_len)
                            if span is not None:
                                old_start, old_end = span
                                new_k[0, h_idx, new_pos:new_pos+self.omega] = past_key_values[0][0, h_idx, old_start:old_end]
                                new_v[0, h_idx, new_pos:new_pos+self.omega] = past_key_values[1][0, h_idx, old_start:old_end]
                                new_logical_id_map[h_idx, new_pos:new_pos+self.omega] = wid_val
                            else:
                                print(f"WARNING [Layer {self.layer_idx}]: Physical eviction: window {wid_val} "
                                      f"not found or partial for head {h_idx}. Zero-filling slot.")
                                new_logical_id_map[h_idx, new_pos:new_pos+self.omega] = wid_val
                
                # 3. Local Zone
                if local_tokens_count > 0:
                    old_local_start = seq_len - local_tokens_count
                    new_local_start = new_compressed_len
                    # FIX (BUG-7): Clamp to prevent silent truncation if dynamic count drifts
                    actual_local = min(local_tokens_count, seq_len - old_local_start)
                    new_k[0, :, new_local_start:new_local_start+actual_local] = past_key_values[0][0, :, old_local_start:old_local_start+actual_local]
                    new_v[0, :, new_local_start:new_local_start+actual_local] = past_key_values[1][0, :, old_local_start:old_local_start+actual_local]
                    
                    if _local_lids is not None:
                        new_logical_id_map[:, new_local_start:new_local_start + actual_local] = _local_lids[:actual_local].unsqueeze(0)
                
                self.logical_id_map = new_logical_id_map
                updated_kv = (new_k, new_v)

                # Reset only the live slice — zeroing the full 131k buffer wastes 65x
                # more bandwidth than needed (4MB vs ~62KB for Qasper at omega=8).
                self.running_attention_votes[:, :seq_len].zero_()
                self.tokens_since_last_review = 0
                self.tokens_since_last_review = 0
                
                return updated_kv
            else:
                return past_key_values
            
    def get_ledger_data(self):
        """Tracking data is not available in the Fast Attention module.
        
        The fast-attention path omits per-token ledger tracking to avoid the
        O(N^2) memory overhead of the full prefill attention matrix.  Use the
        cumulative module (sticky_kv_logic_cummulative.py) for research analysis.
        """
        import warnings
        warnings.warn(
            "get_ledger_data() is not supported in the fast-attention module. "
            "Use the cumulative module for research analysis.",
            stacklevel=2,
        )
        return {}

    # REMOVED (Audit Bug 5): _update_window_scores_generation_vectorized was a dead method
    # never called by any code path. The active pipeline uses scatter_add_ via scoreboard.

    @staticmethod
    def _quantize_k_per_window(tensor, bit_width=8):
        """Quantize K cache: per-channel per-window, with RoPE-paired dimension tying."""
        t_min = tensor.amin(dim=2, keepdim=True)
        t_max = tensor.amax(dim=2, keepdim=True)
        half_d = tensor.shape[-1] // 2
        t_min_h1, t_min_h2 = t_min[..., :half_d], t_min[..., half_d:]
        t_max_h1, t_max_h2 = t_max[..., :half_d], t_max[..., half_d:]
        t_min_tied = torch.min(t_min_h1, t_min_h2)
        t_max_tied = torch.max(t_max_h1, t_max_h2)
        t_min = torch.cat([t_min_tied, t_min_tied], dim=-1)
        t_max = torch.cat([t_max_tied, t_max_tied], dim=-1)
        
        if bit_width == 4:
            scale = torch.clamp((t_max - t_min) / 15.0, min=1e-8)
            quantized = torch.round((tensor - t_min) / scale).clamp(0, 15).to(torch.uint8)
            packed = (quantized[..., 0::2] << 4) | quantized[..., 1::2]
            return packed, scale.to(tensor.dtype), t_min.to(tensor.dtype)
        else:
            scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
            quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
            return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)

    @staticmethod
    def _quantize_v_per_window(tensor, bit_width=8):
        """Quantize V cache: per-token per-window."""
        t_min = tensor.amin(dim=3, keepdim=True)
        t_max = tensor.amax(dim=3, keepdim=True)
        if bit_width == 4:
            scale = torch.clamp((t_max - t_min) / 15.0, min=1e-8)
            quantized = torch.round((tensor - t_min) / scale).clamp(0, 15).to(torch.uint8)
            packed = (quantized[..., 0::2] << 4) | quantized[..., 1::2]
            return packed, scale.to(tensor.dtype), t_min.to(tensor.dtype)
        else:
            scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
            quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
            return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)

    @staticmethod
    def _dequantize_from_quant(quant_tensor, scale, zero_point, bit_width=8):
        """Dequantize int8 or packed int4 tensor back to fp16."""
        if bit_width == 4:
            q_even = (quant_tensor >> 4) & 0x0F
            q_odd = quant_tensor & 0x0F
            unpacked = torch.stack((q_even, q_odd), dim=-1)
            unpacked = unpacked.view(*quant_tensor.shape[:-1], -1)
            return unpacked.to(scale.dtype) * scale + zero_point
        else:
            return quant_tensor.to(scale.dtype) * scale + zero_point

    def _evict_from_window_scores(self):
        valid_mask = ~torch.isnan(self.window_scores[:, :, 1])
        scores = torch.where(
            valid_mask,
            self.window_scores[:, :, 0],
            torch.tensor(float("-inf"), device=self.window_scores.device),
        )
        ids, orig_ids = self.window_scores[:, :, 1], self.window_scores[:, :, 2]
        
        curr_k = min(self.k_windows, int(valid_mask.sum(dim=1).min().item()))
        
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
        
        # 1. Take out sink tokens and assign array
        sinks = self.sink_indices.unsqueeze(0).expand(self.num_heads, -1)
        
        # [Steps 2 & 3 Removed: Redundant math deleted to enforce boundary synchronization]
        
        # FIX (C3): Only include genuinely valid (non-NaN) window score entries.
        # Previously, invalid NaN slots were zeroed to token index 0 via multiply-by-0,
        # inserting a phantom token-0 into the survivor set when sink_tokens == 0.
        raw_w = self.window_scores[:, :self.k_windows, 1]
        valid_w_mask = ~torch.isnan(raw_w)
        valid_k = int(valid_w_mask.all(dim=0).sum().item())
        if valid_k > 0:
            sticky_w = self.window_scores[:, :valid_k, 1].long()
            all_window_tokens = self.window_to_token_map[sticky_w]
            window_tokens = all_window_tokens.view(self.num_heads, -1)
        else:
            window_tokens = torch.zeros(self.num_heads, 0, device=device, dtype=torch.long)
        
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
        
        # FIX (Issue 2): Remove safe_len min-truncation and diff deduplication.
        # Since sinks, window_tokens, and local_zone are mutually exclusive logically,
        # there are no duplicates. Sorting provides the exact dense timeline.
        sorted_all, _ = torch.sort(all_indices_clamped, dim=1)   # [H, N]
        final_indices = sorted_all
        
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
        # Budget includes max_tokens to ensure the cache can accommodate the full
        # sequence (prompt + generation) without mid-generation reallocation.
        total_token_budget = (new_tokens + max_tokens) * self.total_cache_ratio // 100

        # --- SEQUENTIAL CARVING ALLOCATOR ---
        # Priority 1: Sinks (always kept)
        remaining = max(0, total_token_budget - self.sink_tokens)

        # Priority 2: Local zone
        if self.use_fixed_local_tokens:
            target_local_tokens = self.local_num_tokens
        else:
            target_local_tokens = (total_token_budget * self.local_cache_ratio) // 100
        self.local_num = min(target_local_tokens, remaining)
        remaining = max(0, remaining - self.local_num)

        # Priority 3+4: Split remaining between BF16 sticky and INT4 q-cache.
        # q_ratio% of remaining is the q-cache MEMORY budget (BF16-equivalent slots).
        # Compression expands those slots into more stored INT4 tokens.
        # (100-q_ratio)% goes to BF16 full-precision sticky windows.
        #
        # Example: 68 remaining, q_ratio=70, INT4 (4x compression):
        #   bf16_target    = 68 * 30% = 20 BF16 tokens
        #   q_mem_target   = 68 * 70% = 47 BF16-equivalent slots
        #   q_int4_target  = 47 * 4   = 188 INT4 tokens can be stored
        #   k_windows      = ceil(20 / 8) = 3 windows  (absorbs the 4-token remainder)
        #   q_windows      = ceil(188 / 8) = 24 windows (absorbs the 4-token remainder)
        #   No recycling — ceiling division absorbs partial windows in-place.

        bf16_bytes = 2 * self.head_dim
        quant_bytes = self.head_dim if self.quant_bit_width == 8 else (self.head_dim / 2.0)
        compression_ratio = bf16_bytes / quant_bytes

        bf16_target   = (remaining * (100 - self.q_ratio)) // 100
        q_mem_target  = remaining - bf16_target              # complement avoids double rounding
        q_int4_target = int(q_mem_target * compression_ratio)

        # Round UP to absorb the partial window rather than recycling
        # remainder tokens to local. At most omega-1 extra tokens per zone.
        # ceiling division: -(-x // y)
        self.k_windows       = -(-bf16_target   // self.omega)
        self.q_windows_count = -(-q_int4_target // self.omega)

        # Track BF16-equivalent memory actually consumed by q-cache
        q_mem_used   = int((self.q_windows_count * self.omega) / compression_ratio)
        self.q_num   = q_mem_used

        if self.k_windows == 0:
            print(f"WARNING [Layer {self.layer_idx}]: k_windows=0 — insufficient budget for sticky windows "
                  f"(budget={total_token_budget}, local={self.local_num}, sink={self.sink_tokens}, "
                  f"q_windows={self.q_windows_count}). Eviction is effectively disabled.")
        # OPT-6: Precompute quant byte width so q-cache rebuild blocks avoid repeated conditionals
        self._quant_bytes_len = self.head_dim if self.quant_bit_width == 8 else (self.head_dim // 2)

    def _clean_scores(self):
        # Hard resets for cross-document isolation
        self.gen_step = self.num_of_tokens_without_eviction = 0
        self.k_windows = 3  # Reset to constructor default
        self.local_num = 0  # FIX (M2): Reset stale local_num from previous document
        self.tokens_since_last_review = 0
        if hasattr(self, "running_attention_votes"):
            self.running_attention_votes.zero_()
        self.window_scores.fill_(float("nan"))
        self.global_token_counter.zero_()
        self.local_history.zero_()
        self._prefill_done = False
        self.logical_id_map = None
        self._dynamic_local_count = 0
        self.prompt_boundary = [-1 for _ in range(self.num_heads)]
        self.prefill_attention_matrix = None  # Reset (always None in fast-attention; kept for interface parity)
        # Reset q-cache state
        self.q_cache_k_quant = None
        self.q_cache_v_quant = None
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
        # Reset precomputed quant byte width
        self._quant_bytes_len = self.head_dim if self.quant_bit_width == 8 else (self.head_dim // 2)
