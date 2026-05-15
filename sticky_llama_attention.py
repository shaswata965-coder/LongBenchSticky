
import torch
from torch import nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from sticky_kv_logic_cummulative import (
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
        self._dbg_count = 0

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

        if self.layer_idx == 0 and not hasattr(self, "_dbg_count"):
            self._dbg_count = 0

        # 1. Update position_ids for generation (Correct RoPE indexing)
        if past_key_value is not None:
            # Overwrite the position_ids provided by the LlamaModel.forward because it
            # calculated them using the length of the currently evicted KV cache
            global_past_len = self.kv_cache.global_token_counter.item()
            position_ids = torch.arange(
                global_past_len, global_past_len + q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        elif position_ids is None:
            position_ids = torch.arange(0, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        # 2. Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. Causal Masking
        phys_past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        attention_mask = None
        if q_len == 1:
            attention_mask = _make_causal_mask(bsz, q_len, phys_past_len, query_states.dtype, query_states.device)

        # 4. Rotary Positional Embeddings
        kv_seq_len = key_states.shape[-2] + phys_past_len
        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, position_ids.max().item() + 1))
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        # 5. KV Cache Concatenation
        step1_phys_before = past_key_value[0].shape[-2] if past_key_value is not None else 0
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if self.layer_idx == 0 and self._dbg_count == 1:
            pos_dbg = position_ids.detach().flatten().tolist()
            global_tc = int(self.kv_cache.global_token_counter.item())
            print(
                f"[STEP1-CACHE original] cache_type={type(past_key_value).__name__ if past_key_value is not None else 'None'} "
                f"q_len={q_len} pos_ids={pos_dbg} global_tc={global_tc} "
                f"phys_before={step1_phys_before} concat_len={key_states.shape[-2]} "
                f"use_cache={use_cache}",
                flush=True,
            )

        past_key_value = (key_states, value_states) if use_cache else None

        # 6. Multi-Head / Grouped-Query Attention
        if self.num_heads != self.num_key_value_heads:
            q_grouped = query_states.reshape(
                bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim
            )
            main_logits = torch.matmul(q_grouped, key_states.transpose(2, 3).unsqueeze(2)) / math.sqrt(self.head_dim)
            main_logits = main_logits.reshape(bsz, self.num_heads, q_len, -1)
        else:
            main_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            main_logits = main_logits + attention_mask
        elif q_len > 1:
            # Apply causal mask in-place to avoid materializing O(N^2) mask tensor
            mask_val = torch.finfo(main_logits.dtype).min
            i_idx = torch.arange(q_len, device=main_logits.device).unsqueeze(1)
            j_idx = torch.arange(q_len + phys_past_len, device=main_logits.device).unsqueeze(0)
            main_logits.masked_fill_(i_idx + phys_past_len < j_idx, mask_val)

        # --- QUANT Q-CACHE: Joint softmax with dequantized quantized cache ---
        q_scores_for_cache = None
        if (q_len == 1 and hasattr(self.kv_cache, 'q_cache_k_quant') 
                and self.kv_cache.q_cache_k_quant is not None):
            # Dequantize on the fly
            q_k = self.kv_cache._dequantize_from_quant(
                self.kv_cache.q_cache_k_quant, self.kv_cache.q_cache_k_scale, self.kv_cache.q_cache_k_zp, self.kv_cache.quant_bit_width)
            q_v = self.kv_cache._dequantize_from_quant(
                self.kv_cache.q_cache_v_quant, self.kv_cache.q_cache_v_scale, self.kv_cache.q_cache_v_zp, self.kv_cache.quant_bit_width)
            
            # Flatten W*omega → total_tokens: [H, W*omega, D]
            H, W, omega, D = q_k.shape
            q_k = q_k.reshape(H, W * omega, D).unsqueeze(0)
            q_v = q_v.reshape(H, W * omega, D).unsqueeze(0)

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

            # Joint softmax over [main | q-cache]
            all_logits = torch.cat([main_logits, q_logits], dim=-1)
            attn_weights = all_logits.to(torch.float32)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(query_states.dtype)

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
            # FIX (C4): Track main-only attention weights for output_attentions.
            # The joint softmax attn_weights includes q-cache dimensions that
            # downstream consumers don't expect.
            attn_weights_for_output = attn_weights_main
        else:
            # Standard path (no q-cache or prefill)
            attn_weights = main_logits.to(torch.float32)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(query_states.dtype)

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
            attn_weights_for_output = attn_weights

        # Custom Sticky KV Cache Eviction
        # PASS FULL ATTENTION SCORES for Pre-fill Research Analysis
        past_key_value = self.kv_cache(
            past_key_value, scores_for_cache.detach(), 
            full_attn_scores=attn_weights_for_output.detach(),
            q_attn_scores=q_scores_for_cache.detach() if q_scores_for_cache is not None else None
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if self.layer_idx == 0 and hasattr(self, "_dbg_count") and self._dbg_count < 6:
            self._dbg_count += 1

        return attn_output, (attn_weights_for_output if output_attentions else None), past_key_value
