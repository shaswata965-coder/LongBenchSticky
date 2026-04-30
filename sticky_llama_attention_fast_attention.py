import torch
from torch import nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from flash_attn import flash_attn_func
from sticky_kv_logic_fast_attention import (
    repeat_kv,
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    STICKYKVCache_LayerWise
)

class Llama3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=None, base=None, device=None, scaling_factor=None, low_freq_factor=None, high_freq_factor=None, original_max_position_embeddings=None):
        super().__init__()
        import sticky_config
        max_position_embeddings = max_position_embeddings if max_position_embeddings is not None else sticky_config.MAX_POSITION_EMBEDDINGS
        base = base if base is not None else sticky_config.ROPE_THETA
        scaling_factor = scaling_factor if scaling_factor is not None else sticky_config.ROPE_SCALING_FACTOR
        low_freq_factor = low_freq_factor if low_freq_factor is not None else sticky_config.ROPE_LOW_FREQ_FACTOR
        high_freq_factor = high_freq_factor if high_freq_factor is not None else sticky_config.ROPE_HIGH_FREQ_FACTOR
        
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        if original_max_position_embeddings is None:
            original_max_position_embeddings = sticky_config.ORIGINAL_MAX_POSITION_EMBEDDINGS
        self.original_max_position_embeddings = original_max_position_embeddings
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        inv_freq = self.inv_freq
        
        # Calculate wavelengths
        wavelen = 2 * math.pi / inv_freq
        
        # Create smooth factor
        smooth = (self.original_max_position_embeddings / wavelen - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth = torch.clamp(smooth, 0, 1)

        # Apply scaling
        scaled_inv_freq = inv_freq / self.scaling_factor
        
        # Correct logic from Llama 3 reference:
        # new_freqs = (1 - smooth) * scaled_inv_freq + smooth * inv_freq
        new_inv_freq = (1 - smooth) * scaled_inv_freq + smooth * inv_freq
        
        freqs = torch.outer(t, new_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )

class STICKYLlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size ({self.hidden_size}) not divisible by num_heads ({self.num_heads})")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self._init_rope()
        
        self.kv_cache = STICKYKVCache_LayerWise(
            p_ratio=config.p_ratio,
            r_ratio=config.r_ratio,
            start_idx=config.start_idx,
            num_heads=config.num_key_value_heads,
            layer_idx=layer_idx,
            config=config
        )

    def _init_rope(self):
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        
        # Check for Llama 3 specific config
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is not None and isinstance(rope_scaling, dict):
             rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
             if rope_type == "llama3":
                 # print(f"DEBUG: Initializing Custom Llama 3 RoPE for layer {self.layer_idx}", flush=True)
                 dim = self.head_dim
                 max_pos = self.max_position_embeddings
                 base = self.rope_theta
                 
                 import sticky_config
                 factor = rope_scaling.get("factor", sticky_config.ROPE_SCALING_FACTOR)
                 low_freq = rope_scaling.get("low_freq_factor", sticky_config.ROPE_LOW_FREQ_FACTOR)
                 high_freq = rope_scaling.get("high_freq_factor", sticky_config.ROPE_HIGH_FREQ_FACTOR)
                 import sticky_config
                 orig_max_pos = rope_scaling.get("original_max_position_embeddings", sticky_config.ORIGINAL_MAX_POSITION_EMBEDDINGS)
                 
                 self.rotary_emb = Llama3RotaryEmbedding(
                     dim=dim,
                     max_position_embeddings=max_pos,
                     base=base,
                     scaling_factor=factor,
                     low_freq_factor=low_freq,
                     high_freq_factor=high_freq,
                     original_max_position_embeddings=orig_max_pos
                 )
                 return

        # Fallback to standard initialization
        try:
            self.rotary_emb = LlamaRotaryEmbedding(self.config)
        except (TypeError, AttributeError):
            dim = self.head_dim
            max_pos = getattr(self.config, "max_position_embeddings", 2048)
            base = getattr(self.config, "rope_theta", 10000.0)
            
            try:
                self.rotary_emb = LlamaRotaryEmbedding(dim, max_pos, base=base)
            except Exception:
                self.rotary_emb = LlamaRotaryEmbedding(dim, max_position_embeddings=max_pos)
                self.rotary_emb.base = base

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # 1. Update position_ids for generation (Correct RoPE indexing)
        if past_key_value is not None:
            # Overwrite the position_ids provided by the LlamaModel.forward because it
            # calculated them using the length of the currently evicted KV cache
            past_len = self.kv_cache.global_token_counter.item()
            position_ids = torch.arange(
                past_len, past_len + q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        elif position_ids is None:
            position_ids = torch.arange(0, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        # 2. Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. Causal Masking (Only materialized for decoding to avoid O(N^2) OOM spike in long prefill)
        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        attention_mask = None
        if q_len == 1:
            attention_mask = _make_causal_mask(bsz, q_len, past_len, query_states.dtype, query_states.device)

        # 4. Rotary Positional Embeddings
        kv_seq_len = key_states.shape[-2] + past_len
        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, position_ids.max().item() + 1))
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        # 5. KV Cache Concatenation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if q_len > 1:
            # --- EXPLICIT FLASH ATTENTION V2.0 FOR PREFILL (PROMPT) ---
            
            # --- 1. Generation Output via flash_attn_func (Strict FA2 backend) ---
            # Reformating tensors to (batch_size, seq_len, num_heads, head_dim)
            q_fa = query_states.transpose(1, 2)
            k_fa = key_states.transpose(1, 2)
            v_fa = value_states.transpose(1, 2)
            
            # Using original un-repeated k_fa and v_fa takes advantage of FA2's native GQA support
            attn_output = flash_attn_func(
                q_fa, 
                k_fa, 
                v_fa, 
                dropout_p=0.0, 
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True
            )
            # Reformat back to (batch_size, num_heads, seq_len, head_dim) for downstream compatibility
            attn_output = attn_output.transpose(1, 2)
            
            attn_weights_return = None

            # --- 2. Chunked Tracking Scores for Cache Eviction ---
            # FIX A+E+F: Use un-repeated key_states with grouped matmul to avoid
            # materializing key_states_rep (~134MB/layer). Reuse attn_chunk variable
            # to prevent simultaneous attn_chunk + attn_probs allocation.
            chunk_size = 512
            kv_seq_len_full = key_states.shape[-2]
            
            # Accumulated scores format: [bsz, num_key_value_heads, kv_seq_len]
            accumulated_scores = torch.zeros(
                (bsz, self.num_key_value_heads, kv_seq_len_full), 
                device=query_states.device, 
                dtype=torch.float32
            )

            # Pre-generate key indices for dynamic mask chunking
            k_indices = torch.arange(kv_seq_len_full, device=query_states.device).unsqueeze(0)  # [1, kv_seq_len]

            # Pre-transpose keys once: [bsz, kv_heads, head_dim, kv_seq_len]
            key_states_t = key_states.transpose(-2, -1)

            for i in range(0, q_len, chunk_size):
                chunk_len = min(chunk_size, q_len - i)
                q_chunk = query_states[:, :, i:i+chunk_len, :]
                
                # Dynamically construct the causal mask block for this specific chunk (O(N) memory rather than O(N^2))
                q_indices = torch.arange(i, i + chunk_len, device=query_states.device).unsqueeze(1)  # [chunk_len, 1]
                # mask_chunk is causal: valid where key_idx <= query_idx + past_len
                mask_chunk = torch.full(
                    (chunk_len, kv_seq_len_full), torch.finfo(query_states.dtype).min,
                    device=query_states.device, dtype=query_states.dtype
                )
                mask_chunk.masked_fill_(k_indices <= (q_indices + past_len), 0.0)
                
                # Expand mask to match attention chunk [1, 1, chunk_len, kv_seq_len]
                mask_chunk = mask_chunk.unsqueeze(0).unsqueeze(0)
                
                # FIX A+E+F: Grouped matmul with un-repeated keys.
                # Reshape queries [bsz, num_heads, chunk, D] → [bsz, kv_heads, groups, chunk, D]
                # Matmul broadcasts key_states_t over the group dim (no copy).
                # Softmax is per-row so group-then-softmax == softmax-then-group.
                if self.num_heads != self.num_key_value_heads:
                    q_grouped = q_chunk.view(
                        bsz, self.num_key_value_heads, self.num_key_value_groups,
                        chunk_len, self.head_dim
                    )
                    # [bsz, kv_heads, groups, chunk, D] x [bsz, kv_heads, 1, D, kv_len]
                    # → [bsz, kv_heads, groups, chunk, kv_len]
                    attn_chunk = torch.matmul(
                        q_grouped, key_states_t.unsqueeze(2)
                    ) / math.sqrt(self.head_dim)
                    attn_chunk = attn_chunk + mask_chunk
                    # Softmax per-head per-query position (dim=-1 = kv_seq_len)
                    attn_chunk = torch.softmax(attn_chunk.to(torch.float32), dim=-1).to(query_states.dtype)
                    # Average across query head groups → [bsz, kv_heads, chunk, kv_len]
                    scores_for_cache_chunk = attn_chunk.mean(dim=2)
                else:
                    attn_chunk = torch.matmul(
                        q_chunk, key_states_t
                    ) / math.sqrt(self.head_dim)
                    attn_chunk = attn_chunk + mask_chunk
                    attn_chunk = torch.softmax(attn_chunk.to(torch.float32), dim=-1).to(query_states.dtype)
                    scores_for_cache_chunk = attn_chunk
                
                # We need the sum over the q_len (which is dim=2 for scores_for_cache_chunk)
                accumulated_scores += scores_for_cache_chunk.sum(dim=2).to(torch.float32)
                del attn_chunk, scores_for_cache_chunk

            del key_states_t
            # Re-expand accumulation back to [bsz, heads, 1, kv_seq_len] so logic doesn't crash on dim count
            # since historically it expects a 4D tensor where dim=2 is q_len. Here q_len acts as "1" accumulated metric.
            accumulated_scores = accumulated_scores.unsqueeze(2)
            
            # Remove full_attn_scores matrix logging to prevent OOM
            past_key_value = self.kv_cache(past_key_value, accumulated_scores.detach(), q_len=q_len)
            
        else:
            # --- STANDARD ATTENTION FOR GENERATION (DECODING) ---
            # 6. Multi-Head / Grouped-Query Attention (only needed for generation)
            if self.num_heads != self.num_key_value_heads:
                # Reshape queries [bsz, num_heads, q_len=1, D] -> [bsz, kv_heads, groups, 1, D]
                q_grouped = query_states.reshape(
                    bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim
                )
                # [bsz, kv_heads, groups, 1, D] x [bsz, kv_heads, 1, D, kv_len]
                main_logits = torch.matmul(q_grouped, key_states.transpose(2, 3).unsqueeze(2)) / math.sqrt(self.head_dim)
                # Reshape back to mimic original behavior: [bsz, num_heads, q_len, kv_len]
                main_logits = main_logits.reshape(bsz, self.num_heads, q_len, -1)
            else:
                main_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            main_logits = main_logits + attention_mask

            q_scores_for_cache = None
            if q_len == 1 and hasattr(self.kv_cache, 'q_cache_k_quant') and self.kv_cache.q_cache_k_quant is not None:
                q_k = self.kv_cache._dequantize_from_quant(
                    self.kv_cache.q_cache_k_quant,
                    self.kv_cache.q_cache_k_scale,
                    self.kv_cache.q_cache_k_zp,
                    self.kv_cache.quant_bit_width,
                )
                q_v = self.kv_cache._dequantize_from_quant(
                    self.kv_cache.q_cache_v_quant,
                    self.kv_cache.q_cache_v_scale,
                    self.kv_cache.q_cache_v_zp,
                    self.kv_cache.quant_bit_width,
                )
                H, W, omega_dim, D = q_k.shape
                q_k = q_k.reshape(H, W * omega_dim, D).unsqueeze(0)  # [1, kv_heads, q_tokens, D]
                q_v = q_v.reshape(H, W * omega_dim, D).unsqueeze(0)

                if self.num_heads != self.num_key_value_heads:
                    q_grouped = query_states.reshape(
                        bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim
                    )
                    q_logits_grouped = torch.matmul(
                        q_grouped, q_k.transpose(2, 3).unsqueeze(2)
                    ) / math.sqrt(self.head_dim)
                    q_logits = q_logits_grouped.reshape(bsz, self.num_heads, q_len, -1)
                else:
                    q_logits = torch.matmul(query_states, q_k.transpose(2, 3)) / math.sqrt(self.head_dim)

                all_logits = torch.cat([main_logits, q_logits], dim=-1)
                attn_weights = torch.softmax(all_logits.to(torch.float32), dim=-1).to(query_states.dtype)

                main_len = main_logits.size(-1)
                attn_weights_main = attn_weights[..., :main_len]
                attn_weights_q = attn_weights[..., main_len:]

                if self.num_heads != self.num_key_value_heads:
                    attn_main_grouped = attn_weights_main.reshape(
                        bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, -1
                    )
                    value_main_grouped = value_states.unsqueeze(2)
                    out_main = torch.matmul(attn_main_grouped, value_main_grouped)

                    attn_q_grouped = attn_weights_q.reshape(
                        bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, -1
                    )
                    out_q = torch.matmul(attn_q_grouped, q_v.unsqueeze(2))

                    attn_output = (out_main + out_q).reshape(bsz, self.num_heads, q_len, self.head_dim)
                    scores_for_cache = attn_weights_main.reshape(
                        bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, -1
                    ).mean(dim=2)
                    q_scores_for_cache = attn_weights_q.reshape(
                        bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, -1
                    ).mean(dim=2)
                else:
                    attn_output = torch.matmul(attn_weights_main, value_states) + torch.matmul(attn_weights_q, q_v)
                    scores_for_cache = attn_weights_main
                    q_scores_for_cache = attn_weights_q
            else:
                attn_weights = torch.softmax(main_logits.to(torch.float32), dim=-1).to(query_states.dtype)
                if self.num_heads != self.num_key_value_heads:
                    attn_grouped = attn_weights.reshape(
                        bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, -1
                    )
                    attn_output = torch.matmul(attn_grouped, value_states.unsqueeze(2)).reshape(
                        bsz, self.num_heads, q_len, self.head_dim
                    )
                    scores_for_cache = attn_grouped.mean(dim=2)
                else:
                    attn_output = torch.matmul(attn_weights, value_states)
                    scores_for_cache = attn_weights

            past_key_value = self.kv_cache(
                past_key_value, scores_for_cache.detach(),
                q_attn_scores=q_scores_for_cache.detach() if q_scores_for_cache is not None else None,
                q_len=q_len)
                    
            attn_weights_return = attn_weights if output_attentions else None

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights_return, past_key_value