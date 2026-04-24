"""
run_open_ended_cummulative.py
------------------------------
Open-ended text generation using the Sticky KV Cache (cumulative variant).

Mirrors the LLaMA prompt config from main.py and wires
STICKYKVCache_LayerWise from sticky_kv_logic_cummulative into a
bespoke attention module, keeping the fast-attention model file untouched.

Usage (Kaggle / local GPU):
    python run_open_ended_cummulative.py
"""

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
    rotate_half,
)

# ── Sticky imports ────────────────────────────────────────────────────────────
import sticky_config
from configuration_sticky_llama import LlamaConfig
from sticky_kv_logic_cummulative import (
    STICKYKVCache_LayerWise,
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    repeat_kv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Llama 3 RoPE (copy of the one in sticky_llama_attention.py but wired
# to sticky_config so we don't need to import from the fast-attention file)
# ─────────────────────────────────────────────────────────────────────────────
class Llama3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=None,
        base=None,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        original_max_position_embeddings=None,
    ):
        super().__init__()
        max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else sticky_config.MAX_POSITION_EMBEDDINGS
        )
        base = base if base is not None else sticky_config.ROPE_THETA
        scaling_factor = (
            scaling_factor
            if scaling_factor is not None
            else sticky_config.ROPE_SCALING_FACTOR
        )
        low_freq_factor = (
            low_freq_factor
            if low_freq_factor is not None
            else sticky_config.ROPE_LOW_FREQ_FACTOR
        )
        high_freq_factor = (
            high_freq_factor
            if high_freq_factor is not None
            else sticky_config.ROPE_HIGH_FREQ_FACTOR
        )
        original_max_position_embeddings = (
            original_max_position_embeddings
            if original_max_position_embeddings is not None
            else sticky_config.ORIGINAL_MAX_POSITION_EMBEDDINGS
        )

        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor
        high_freq_wavelen = self.original_max_position_embeddings / self.high_freq_factor

        inv_freq = self.inv_freq
        wavelen = 2 * math.pi / inv_freq
        smooth = (self.original_max_position_embeddings / wavelen - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth = torch.clamp(smooth, 0, 1)

        scaled_inv_freq = inv_freq / self.scaling_factor
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


# ─────────────────────────────────────────────────────────────────────────────
# Sticky Attention (cumulative variant)
# ─────────────────────────────────────────────────────────────────────────────
class STICKYLlamaAttention_Cummulative(nn.Module):
    """
    Drop-in replacement for LlamaAttention that uses
    STICKYKVCache_LayerWise from sticky_kv_logic_cummulative.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
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
            raise ValueError(
                f"hidden_size ({self.hidden_size}) not divisible by num_heads ({self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()

        # ── Cumulative Sticky KV Cache ────────────────────────────────────
        self.kv_cache = STICKYKVCache_LayerWise(
            p_ratio=config.p_ratio,
            r_ratio=config.r_ratio,
            start_idx=config.start_idx,
            num_heads=config.num_key_value_heads,
            layer_idx=layer_idx,
            config=config,
        )

    # ------------------------------------------------------------------
    def _init_rope(self):
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
            if rope_type == "llama3":
                dim = self.head_dim
                max_pos = self.max_position_embeddings
                base = self.rope_theta
                factor = rope_scaling.get("factor", sticky_config.ROPE_SCALING_FACTOR)
                low_freq = rope_scaling.get("low_freq_factor", sticky_config.ROPE_LOW_FREQ_FACTOR)
                high_freq = rope_scaling.get("high_freq_factor", sticky_config.ROPE_HIGH_FREQ_FACTOR)
                orig_max_pos = rope_scaling.get(
                    "original_max_position_embeddings",
                    sticky_config.ORIGINAL_MAX_POSITION_EMBEDDINGS,
                )
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=dim,
                    max_position_embeddings=max_pos,
                    base=base,
                    scaling_factor=factor,
                    low_freq_factor=low_freq,
                    high_freq_factor=high_freq,
                    original_max_position_embeddings=orig_max_pos,
                )
                return

        # Fallback to standard LlamaRotaryEmbedding
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

    # ------------------------------------------------------------------
    def _clean_cache(self):
        self.kv_cache._clean_scores()

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # ── 1. Correct position IDs for evicted KV cache ─────────────────
        if past_key_value is not None:
            past_len = self.kv_cache.global_token_counter.item()
            position_ids = torch.arange(
                past_len, past_len + q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        elif position_ids is None:
            position_ids = torch.arange(
                0, q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

        # ── 2. Project Q, K, V ────────────────────────────────────────────
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # ── 3. Causal mask ────────────────────────────────────────────────
        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        attention_mask = _make_causal_mask(
            bsz, q_len, past_len, query_states.dtype, query_states.device
        )

        # ── 4. RoPE ───────────────────────────────────────────────────────
        kv_seq_len = key_states.shape[-2] + past_len
        cos, sin = self.rotary_emb(
            value_states, seq_len=max(kv_seq_len, position_ids.max().item() + 1)
        )
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        # ── 5. KV cache concatenation ─────────────────────────────────────
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # ── 6. GQA / MHA attention ────────────────────────────────────────
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states_rep.transpose(2, 3))
            / math.sqrt(self.head_dim)
        )
        attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.to(torch.float32)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(query_states.dtype)

        # Aggregate Q-heads → KV-heads for cache scoring
        if self.num_heads != self.num_key_value_heads:
            scores_for_cache = attn_weights.view(
                bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, -1
            ).mean(dim=2)
        else:
            scores_for_cache = attn_weights

        # ── 7. Cumulative Sticky KV eviction ──────────────────────────────
        past_key_value = self.kv_cache(
            past_key_value,
            scores_for_cache.detach(),
            full_attn_scores=attn_weights.detach(),
        )

        # ── 8. Output projection ──────────────────────────────────────────
        attn_output = torch.matmul(attn_weights, value_states_rep)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), past_key_value


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper (cumulative variant)
# ─────────────────────────────────────────────────────────────────────────────
class STICKYLlamaForCausalLM_Cummulative(LlamaForCausalLM):
    """
    LlamaForCausalLM with all attention layers replaced by
    STICKYLlamaAttention_Cummulative (uses sticky_kv_logic_cummulative).
    """

    def __init__(self, config, **kwargs):
        safe_config = copy.deepcopy(config)
        if hasattr(safe_config, "rope_scaling"):
            safe_config.rope_scaling = None
        super().__init__(safe_config)

        # Restore correct config on the model instance
        self.config = config

        print(
            f"[Cummulative] Replacing {len(self.model.layers)} attention layers …"
        )
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].self_attn = STICKYLlamaAttention_Cummulative(
                config, layer_idx
            )
        print("[Cummulative] All attention layers replaced.")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if past_key_values is not None:
            # Override incorrect slicing done by super() after KV eviction
            model_inputs["input_ids"] = input_ids[:, -1:]

            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    model_inputs["position_ids"] = position_ids[:, -1:]
                else:
                    true_seq_length = input_ids.shape[1]
                    model_inputs["position_ids"] = torch.tensor(
                        [[true_seq_length - 1]],
                        dtype=torch.long,
                        device=input_ids.device,
                    )
            else:
                model_inputs["position_ids"] = position_ids[:, -1:]

        return model_inputs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Model path (mirrors main.py) ──────────────────────────────────────
    model_path = sticky_config.MODEL_PATH

    # ── 2. Config (mirrors main.py exactly) ──────────────────────────────────
    config = LlamaConfig.from_pretrained(model_path)

    # Fix the "Version Gap": Llama 3.2 uses 'rope_type'; older transformers need 'type'
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        if "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]

    config.rope_theta = getattr(config, "rope_theta", sticky_config.ROPE_THETA)

    # Sticky-specific knobs (from sticky_config)
    config.r_ratio    = getattr(sticky_config, "R_RATIO", 20)
    config.start_idx  = getattr(sticky_config, "S_IDX",   0)
    config.alpha      = sticky_config.OMEGA   # observation window = omega chunk size

    # Local window: prefer fixed token count when LOCAL_NUM_TOKENS is set
    if hasattr(sticky_config, "LOCAL_NUM_TOKENS"):
        config.local_num_tokens = sticky_config.LOCAL_NUM_TOKENS
    else:
        config.local_num_tokens = 256          # same default as main.py

    print(
        f"[Config] r_ratio={config.r_ratio}  "
        f"p_ratio={config.p_ratio}  "
        f"local_num_tokens={config.local_num_tokens}  "
        f"omega={sticky_config.OMEGA}  "
        f"sink={sticky_config.SINK_TOKENS}"
    )

    # ── 3. Tokenizer (mirrors main.py) ───────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Chat template available: {tokenizer.chat_template is not None}")

    # ── 4. Load model with cumulative sticky cache ────────────────────────────
    model = STICKYLlamaForCausalLM_Cummulative.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # NOTE: attn_implementation is intentionally omitted here because our
        # custom forward() replaces the kernel entirely; flash_attention_2
        # would bypass our attention logic.
    )
    print(f"Model device : {model.device}")
    print(f"Model dtype  : {model.dtype}")
    print(f"RoPE scaling : {model.config.rope_scaling}")

    # ── 5. Prompt (same Valkyria Chronicles prompt from main.py) ─────────────
    messages = [
        {
            "role": "user",
            "content": (
                "The history of the Valkyria Chronicles series is deeply intertwined with its "
                "unique strategic RPG combat system known as BLiTZ (Battle of Live Tactical Zones). "
                "Developed by Sega, the franchise first debuted on the PlayStation 3 in 2008 and "
                "quickly garnered a passionate fanbase due to its gorgeous CANVAS graphics engine, "
                "which mimics the appearance of a watercolor painting in motion. The narrative "
                "frequently centers around the small, neutral nation of Gallia as it struggles to "
                "maintain its independence against the overwhelming military might of the East "
                "Europan Imperial Alliance."
                "In Valkyria Chronicles 3, the story shifts focus to a highly secretive penal "
                "military unit known as Squad 422, or 'The Nameless.' These individuals have been "
                "stripped of their names and identities, referred to only by numbers, and are "
                "tasked with executing highly dangerous black-ops missions that the regular Gallian "
                "army cannot legally undertake. The squad is completely expendable, composed of "
                "criminals, insubordinate soldiers, and those falsely accused of treason. Kurt "
                "Irving, an incredibly gifted military tactician falsely disgraced by a conspiracy, "
                "is assigned as their new commander. He must not only lead this ragtag group of "
                "outcasts to survive seemingly impossible suicide missions but also uncover the "
                "truth behind his own downfall to clear his name. The squad features a diverse "
                "cast of tragic characters, such as Riela Marcellis, a young woman ostracized for "
                "her mysterious resilience to death which stems from her latent Valkyria blood, "
                "and Imca, a fiercely independent Darcsen warrior driven entirely by a singular "
                "desire for revenge against the Imperial soldier who destroyed her village. "
                "Throughout their grueling campaign, The Nameless must face off against Calamity "
                "Raven, a formidable Imperial black-ops unit consisting entirely of Darcsens who "
                "are fighting for a promised autonomous homeland. As the war escalates, Kurt and "
                "his squad find themselves entangled in deeply complicated political machinations, "
                "constantly manipulated by their own corrupt Gallian commanders while desperately "
                "fighting to regain their true identities and prove their irreplaceable worth on "
                "the battlefield."
                "Please write a comprehensive, detailed 200-word continuation expanding on the "
                "following text. Do not stop early."
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\n{'='*60}\nFormatted Prompt\n{'='*60}")
    print(prompt)
    print("=" * 60)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input token count: {inputs['input_ids'].shape[1]}")

    # ── 6. Generate (mirrors main.py terminators + GENERATION_CONFIG) ─────────
    print("\nGenerating …")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    output = model.generate(
        **inputs,
        **sticky_config.GENERATION_CONFIG,   # max_new_tokens, do_sample, temperature
        repetition_penalty=1.1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # ── 7. Decode & print ─────────────────────────────────────────────────────
    clean_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{'='*60}\nGenerated Output\n{'='*60}")
    print(clean_output)
    print("=" * 60)

    # ── 8. Optional: dump ledger summary from layer 0 ────────────────────────
    try:
        layer0_attn = model.model.layers[0].self_attn
        ledger = layer0_attn.kv_cache.get_ledger_data()
        n = ledger["global_id"].shape[0]
        print(f"\n[Ledger] Layer-0 tracked tokens : {n}")
        print(f"[Ledger] Avg attention score (head-0) : "
              f"{ledger['attention_score'][:, 0].float().mean():.4f}")
    except Exception as e:
        print(f"[Ledger] Could not retrieve ledger data: {e}")


if __name__ == "__main__":
    main()
