import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from sticky_kv_logic_fast_attention import (
    STICKYKVCache_LayerWise,
    _make_causal_mask,
)


def _layer_past_from_cache(past_key_value: Optional[Cache], layer_idx: int):
    if past_key_value is None or not isinstance(past_key_value, Cache):
        return None
    if len(past_key_value.key_cache) <= layer_idx:
        return None
    k = past_key_value.key_cache[layer_idx]
    v = past_key_value.value_cache[layer_idx]
    if k is None or v is None:
        return None
    return (k, v)


def _write_layer_cache(
    past_key_value: Cache,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    global_logical_len: int,
):
    if len(past_key_value.key_cache) <= layer_idx:
        past_key_value.key_cache.append(key_states)
        past_key_value.value_cache.append(value_states)
    else:
        past_key_value.key_cache[layer_idx] = key_states
        past_key_value.value_cache[layer_idx] = value_states
    if layer_idx == 0 and hasattr(past_key_value, "_seen_tokens"):
        past_key_value._seen_tokens = int(global_logical_len)


class STICKYQwen2Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) not divisible by num_heads ({self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.kv_cache = STICKYKVCache_LayerWise(
            p_ratio=config.p_ratio,
            r_ratio=config.r_ratio,
            start_idx=config.start_idx,
            num_heads=config.num_key_value_heads,
            layer_idx=layer_idx,
            config=config,
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

        bsz, q_len, _ = hidden_states.size()
        cache_obj = past_key_value if isinstance(past_key_value, Cache) else None
        past_kv = _layer_past_from_cache(cache_obj, self.layer_idx)

        # RoPE positions must reflect the *logical* sequence length, not the truncated
        # physical KV cache length (Sticky eviction shortens physical cache).
        if past_kv is not None:
            global_past_len = int(self.kv_cache.global_token_counter.item())
            position_ids = torch.arange(
                global_past_len, global_past_len + q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        elif position_ids is None:
            position_ids = torch.arange(0, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )

        phys_past_len = past_kv[0].shape[-2] if past_kv is not None else 0

        # Preserve HF padding masks when provided. (Fast flash-attn prefill still
        # cannot correctly handle padding; see bug report.)
        additive_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                key_mask = (attention_mask == 0).to(dtype=query_states.dtype)
                mask_val = torch.finfo(query_states.dtype).min
                additive_mask = (key_mask * mask_val)[:, None, None, :]
            else:
                additive_mask = attention_mask

        causal_mask = None
        if q_len == 1:
            causal_mask = _make_causal_mask(bsz, q_len, phys_past_len, query_states.dtype, query_states.device)

        kv_seq_len = key_states.shape[-2] + phys_past_len
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_kv is not None:
            key_states = torch.cat([past_kv[0], key_states], dim=2)
            value_states = torch.cat([past_kv[1], value_states], dim=2)

        qcache_available = (
            hasattr(self.kv_cache, "q_cache_k_quant") and self.kv_cache.q_cache_k_quant is not None
        )

        # Fast-path uses FlashAttention for q_len>1, but FlashAttention path cannot
        # incorporate the q-cache (evicted tokens). For chunked/speculative decoding
        # after eviction, fall back to the manual path so q-cache is attended/scored.
        if q_len > 1 and not qcache_available:
            q_fa = query_states.transpose(1, 2)
            k_fa = key_states.transpose(1, 2)
            v_fa = value_states.transpose(1, 2)
            attn_output = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True,
            )
            attn_output = attn_output.transpose(1, 2)
            attn_weights_return = None

            chunk_size = 512
            kv_seq_len_full = key_states.shape[-2]
            accumulated_scores = torch.zeros(
                (bsz, self.num_key_value_heads, kv_seq_len_full),
                device=query_states.device,
                dtype=torch.float32,
            )
            k_indices = torch.arange(kv_seq_len_full, device=query_states.device).unsqueeze(0)
            key_states_t = key_states.transpose(-2, -1)

            for i in range(0, q_len, chunk_size):
                chunk_len = min(chunk_size, q_len - i)
                q_chunk = query_states[:, :, i : i + chunk_len, :]
                q_indices = torch.arange(i, i + chunk_len, device=query_states.device).unsqueeze(1)
                mask_chunk = torch.full(
                    (chunk_len, kv_seq_len_full),
                    torch.finfo(query_states.dtype).min,
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                mask_chunk.masked_fill_(k_indices <= (q_indices + phys_past_len), 0.0)
                mask_chunk = mask_chunk.unsqueeze(0).unsqueeze(0)

                if self.num_heads != self.num_key_value_heads:
                    q_grouped = q_chunk.view(
                        bsz,
                        self.num_key_value_heads,
                        self.num_key_value_groups,
                        chunk_len,
                        self.head_dim,
                    )
                    attn_chunk = torch.matmul(q_grouped, key_states_t.unsqueeze(2)) / math.sqrt(self.head_dim)
                    attn_chunk = attn_chunk + mask_chunk
                    attn_chunk = torch.softmax(attn_chunk.to(torch.float32), dim=-1).to(query_states.dtype)
                    scores_for_cache_chunk = attn_chunk.mean(dim=2)
                else:
                    attn_chunk = torch.matmul(q_chunk, key_states_t) / math.sqrt(self.head_dim)
                    attn_chunk = attn_chunk + mask_chunk
                    attn_chunk = torch.softmax(attn_chunk.to(torch.float32), dim=-1).to(query_states.dtype)
                    scores_for_cache_chunk = attn_chunk

                accumulated_scores += scores_for_cache_chunk.sum(dim=2).to(torch.float32)
                del attn_chunk, scores_for_cache_chunk

            del key_states_t
            accumulated_scores = accumulated_scores.unsqueeze(2)

            if use_cache and cache_obj is not None:
                past_kv_tuple = self.kv_cache((key_states, value_states), accumulated_scores.detach(), q_len=q_len)
                _write_layer_cache(
                    cache_obj,
                    self.layer_idx,
                    past_kv_tuple[0],
                    past_kv_tuple[1],
                    int(self.kv_cache.global_token_counter.item()),
                )
                past_key_value_out = cache_obj
            elif use_cache:
                past_key_value_out = self.kv_cache((key_states, value_states), accumulated_scores.detach(), q_len=q_len)
            else:
                past_key_value_out = None

        else:
            if self.num_heads != self.num_key_value_heads:
                q_grouped = query_states.reshape(
                    bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim
                )
                main_logits = (
                    torch.matmul(q_grouped, key_states.transpose(2, 3).unsqueeze(2)) / math.sqrt(self.head_dim)
                )
                main_logits = main_logits.reshape(bsz, self.num_heads, q_len, -1)
            else:
                main_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if causal_mask is not None:
                main_logits = main_logits + causal_mask
            if additive_mask is not None:
                main_logits = main_logits + additive_mask
            elif q_len > 1:
                # causal mask for prefill/chunked decode
                mask_val = torch.finfo(main_logits.dtype).min
                i_idx = torch.arange(q_len, device=main_logits.device).unsqueeze(1)
                j_idx = torch.arange(q_len + phys_past_len, device=main_logits.device).unsqueeze(0)
                main_logits.masked_fill_(i_idx + phys_past_len < j_idx, mask_val)

            q_scores_for_cache = None
            if qcache_available:
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
                q_k = q_k.reshape(H, W * omega_dim, D).unsqueeze(0)
                q_v = q_v.reshape(H, W * omega_dim, D).unsqueeze(0)

                if self.num_heads != self.num_key_value_heads:
                    q_grouped = query_states.reshape(
                        bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim
                    )
                    q_logits_grouped = torch.matmul(q_grouped, q_k.transpose(2, 3).unsqueeze(2)) / math.sqrt(
                        self.head_dim
                    )
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
                attn_weights_for_output = attn_weights_main
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
                attn_weights_for_output = attn_weights

            past_kv_tuple = self.kv_cache(
                (key_states, value_states),
                scores_for_cache.detach(),
                q_attn_scores=q_scores_for_cache.detach() if q_scores_for_cache is not None else None,
                q_len=q_len,
            )

            if use_cache and cache_obj is not None:
                _write_layer_cache(
                    cache_obj,
                    self.layer_idx,
                    past_kv_tuple[0],
                    past_kv_tuple[1],
                    int(self.kv_cache.global_token_counter.item()),
                )
                past_key_value_out = cache_obj
            elif use_cache:
                past_key_value_out = past_kv_tuple
            else:
                past_key_value_out = None

            attn_weights_return = attn_weights_for_output if output_attentions else None

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights_return, past_key_value_out
