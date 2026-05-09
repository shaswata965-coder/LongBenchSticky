"""
benchmark_throughput.py
=======================
Lightweight throughput benchmark for the Sticky KV Cache pipeline.

Loads a PG-19 sample with a SHORT 256-token prefill, then runs open-ended
generation (greedy) for max_new_tokens and reports throughput in tok/s.

This is designed for SPEED MEASUREMENT ONLY — no attention tracking, no
analysis output.  Set tracking_flag=0 before running to disable all
attention-weight materialization overhead.

Usage (from repo root):
    python Results/benchmark_throughput.py
"""

import torch
import numpy as np
import os
import sys
import gc
import time as _time
import random

# ---------------------------------------------------------------------------
# Path setup (Kaggle-safe)
# ---------------------------------------------------------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if SCRIPT_DIR != os.getcwd() else os.getcwd()
for _p in [REPO_ROOT, SCRIPT_DIR, os.path.join(REPO_ROOT, "Results"),
           os.path.join(REPO_ROOT, "Dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sticky_config as config

# ── Force tracking OFF for clean throughput measurement ──
config.tracking_flag = 0

from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig


# ===========================================================================
# Benchmark runner
# ===========================================================================

def run_benchmark(
    r_ratio: int,
    q_ratio: int = 0,
    quant_bit_width: int = 8,
    omega: int = 8,
    prefill_tokens: int = 256,
    max_new_tokens: int = 512,
    label: str = "",
):
    """Run a single throughput benchmark and return results dict."""

    # ── Seed for reproducibility ──
    seed = config.SEEDS[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Patch sticky_config for this run ──
    config.OMEGA = omega
    config.Q_RATIO = q_ratio
    config.QUANTIZATION_BIT_WIDTH = quant_bit_width
    config.GENERATION_CONFIG["max_new_tokens"] = max_new_tokens

    # ── Load model config ──
    model_config = LlamaConfig.from_pretrained(config.MODEL_PATH)

    # Rope-scaling compatibility shim
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
            model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
    model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)

    model_config.r_ratio = r_ratio
    model_config.start_idx = 0

    if hasattr(config, "P_RATIO"):
        model_config.p_ratio = config.P_RATIO
    elif hasattr(config, "LOCAL_NUM_TOKENS"):
        model_config.local_num_tokens = config.LOCAL_NUM_TOKENS

    # ── Load model ──
    model = STICKYLlamaForCausalLM.from_pretrained(
        config.MODEL_PATH,
        config=model_config,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device

    # ── Load dataset (Kaggle-compatible: same pattern as run_sticky_teacher_forcing) ──
    if config.dataset_tracker == 1:
        sys.path.insert(0, os.path.join(REPO_ROOT, "Dataset"))
        from pg19_loader import get_pg19_blocks
        raw_samples = get_pg19_blocks(
            config.MODEL_PATH,
            num_samples=1,
            min_tokens=max(prefill_tokens + 64, config.DATASET_MIN_TOKENS),
        )
    else:
        sys.path.insert(0, os.path.join(REPO_ROOT, "Dataset"))
        from wiki_text_loader import get_wikitext103_drift_blocks
        raw_samples = get_wikitext103_drift_blocks(
            config.MODEL_PATH,
            num_samples=1,
            min_tokens=max(prefill_tokens + 64, config.DATASET_MIN_TOKENS),
        )

    if not raw_samples:
        print("ERROR: No samples found.")
        return None

    text = raw_samples[0]["text"].strip()
    full_ids = tokenizer(text, add_special_tokens=False).input_ids
    prompt_ids = full_ids[:prefill_tokens]

    print(f"\n{'='*60}")
    print(f"  Benchmark: {label or f'R={r_ratio} Q={q_ratio}'}")
    print(f"  Prefill: {len(prompt_ids)} tokens | Gen: {max_new_tokens} tokens")
    print(f"  R_RATIO={r_ratio}  Q_RATIO={q_ratio}  QUANT_BW={quant_bit_width}  OMEGA={omega}")
    print(f"  tracking_flag={config.tracking_flag}")
    print(f"{'='*60}")

    # ── Reset caches ──
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "_clean_cache"):
            layer.self_attn._clean_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ── Prefill ──
    print("  Running prefill ...", flush=True)
    with torch.no_grad():
        prefill_out = model(
            input_ids=prompt_tensor,
            use_cache=True,
            output_attentions=False,
        )
    past_kv = prefill_out.past_key_values
    next_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token_id.item()]
    del prefill_out

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ── Timed generation loop ──
    print(f"  Running generation ({max_new_tokens} tokens) ...", flush=True)
    t0 = _time.perf_counter()

    for step in range(max_new_tokens - 1):  # -1 because we already have the first token
        with torch.no_grad():
            gen_out = model(
                input_ids=next_token_id,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=False,
            )
        past_kv = gen_out.past_key_values
        next_token_id = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token_id.item())

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Progress
        if (step + 1) % 50 == 0:
            elapsed = _time.perf_counter() - t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"    Step {step+1}/{max_new_tokens-1} | {rate:.1f} tok/s", flush=True)

        del gen_out

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = _time.perf_counter()
    gen_time = t1 - t0
    gen_count = len(generated_ids) - 1  # exclude first token (from prefill)
    throughput = gen_count / gen_time if gen_time > 0 else 0

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print(f"\n  ── Results ──")
    print(f"  Generated: {gen_count} tokens in {gen_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Peak VRAM: {peak_mem:.0f} MB")

    # Preview generated text
    preview = tokenizer.decode(generated_ids[:50], skip_special_tokens=True)
    print(f"  Preview: {preview[:200]}...")

    # Cleanup
    del model, past_kv, prompt_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label or f"R={r_ratio}_Q={q_ratio}",
        "r_ratio": r_ratio,
        "q_ratio": q_ratio,
        "quant_bit_width": quant_bit_width,
        "omega": omega,
        "prefill_tokens": len(prompt_ids),
        "generated_tokens": gen_count,
        "generation_time_s": gen_time,
        "throughput_tok_s": throughput,
        "peak_vram_mb": peak_mem,
    }


# ===========================================================================
# TRUE HuggingFace Baseline — unmodified LlamaForCausalLM
# ===========================================================================

def run_hf_baseline(prefill_tokens=256, max_new_tokens=512):
    """Run the UNMODIFIED HuggingFace LlamaForCausalLM with the same manual
    generation loop. This isolates sticky overhead from loop overhead."""
    from transformers import LlamaForCausalLM as HF_LlamaForCausalLM, AutoTokenizer

    seed = config.SEEDS[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"  TRUE HF BASELINE (no sticky wrapper)")
    print(f"  Prefill: {prefill_tokens} tokens | Gen: {max_new_tokens} tokens")
    print(f"{'='*60}")

    model = HF_LlamaForCausalLM.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    # Load same dataset
    if config.dataset_tracker == 1:
        sys.path.insert(0, os.path.join(REPO_ROOT, "Dataset"))
        from pg19_loader import get_pg19_blocks
        raw_samples = get_pg19_blocks(config.MODEL_PATH, num_samples=1,
                                       min_tokens=max(prefill_tokens + 64, config.DATASET_MIN_TOKENS))
    else:
        sys.path.insert(0, os.path.join(REPO_ROOT, "Dataset"))
        from wiki_text_loader import get_wikitext103_drift_blocks
        raw_samples = get_wikitext103_drift_blocks(config.MODEL_PATH, num_samples=1,
                                                    min_tokens=max(prefill_tokens + 64, config.DATASET_MIN_TOKENS))

    text = raw_samples[0]["text"].strip()
    full_ids = tokenizer(text, add_special_tokens=False).input_ids
    prompt_ids = full_ids[:prefill_tokens]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Prefill
    print("  Running prefill ...", flush=True)
    with torch.no_grad():
        prefill_out = model(input_ids=prompt_tensor, use_cache=True, output_attentions=False)
    past_kv = prefill_out.past_key_values
    next_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token_id.item()]
    del prefill_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Same manual generation loop
    print(f"  Running generation ({max_new_tokens} tokens) ...", flush=True)
    t0 = _time.perf_counter()
    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            gen_out = model(input_ids=next_token_id, past_key_values=past_kv,
                            use_cache=True, output_attentions=False)
        past_kv = gen_out.past_key_values
        next_token_id = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token_id.item())
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        if (step + 1) % 50 == 0:
            elapsed = _time.perf_counter() - t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"    Step {step+1}/{max_new_tokens-1} | {rate:.1f} tok/s", flush=True)
        del gen_out
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = _time.perf_counter()
    gen_time = t1 - t0
    gen_count = len(generated_ids) - 1
    throughput = gen_count / gen_time if gen_time > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print(f"\n  ── Results ──")
    print(f"  Generated: {gen_count} tokens in {gen_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Peak VRAM: {peak_mem:.0f} MB")

    del model, past_kv, prompt_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": "TRUE HF Baseline (no sticky)",
        "r_ratio": -1, "q_ratio": 0, "quant_bit_width": 0, "omega": 0,
        "prefill_tokens": len(prompt_ids),
        "generated_tokens": gen_count,
        "generation_time_s": gen_time,
        "throughput_tok_s": throughput,
        "peak_vram_mb": peak_mem,
    }


# ===========================================================================
# Main — sweeps OMEGA × config
# ===========================================================================

# ── Benchmark matrix ──
OMEGA_VALUES = [4, 8, 16]
R_RATIO_CFG = getattr(config, "R_RATIO", 20)
Q_RATIO_CFG = getattr(config, "Q_RATIO", 0) or 30
QUANT_BW_CFG = getattr(config, "QUANTIZATION_BIT_WIDTH", 4)

BENCHMARK_CONFIGS = [
    {"r_ratio": 100, "q_ratio": 0,          "quant_bit_width": 8,         "tag": "Vanilla"},
    {"r_ratio": R_RATIO_CFG, "q_ratio": 0,  "quant_bit_width": 8,         "tag": f"Evict R={R_RATIO_CFG}"},
    {"r_ratio": R_RATIO_CFG, "q_ratio": Q_RATIO_CFG, "quant_bit_width": QUANT_BW_CFG, "tag": f"Evict+Q R={R_RATIO_CFG} Q={Q_RATIO_CFG}"},
]


def main():
    print("=" * 70)
    print("  STICKY KV CACHE — THROUGHPUT BENCHMARK")
    print(f"  Model: {config.MODEL_PATH}")
    print(f"  tracking_flag: {config.tracking_flag}")
    print(f"  OMEGA sweep: {OMEGA_VALUES}")
    print("=" * 70)

    results = []

    # ── Step 0: TRUE HuggingFace baseline (isolates loop vs module overhead) ──
    hf_r = run_hf_baseline()
    if hf_r:
        results.append(hf_r)

    # ── Step 1: Sticky configs ──
    for omega_val in OMEGA_VALUES:
        for cfg in BENCHMARK_CONFIGS:
            label = f"{cfg['tag']} (ω={omega_val})"
            r = run_benchmark(
                r_ratio=cfg["r_ratio"],
                q_ratio=cfg["q_ratio"],
                quant_bit_width=cfg["quant_bit_width"],
                omega=omega_val,
                label=label,
            )
            if r:
                results.append(r)

    # ── Summary table ──
    if results:
        print(f"\n{'='*80}")
        print(f"  THROUGHPUT SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<40} {'ω':>4} {'Tok/s':>8} {'Time':>8} {'VRAM':>8}")
        print(f"  {'-'*40} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            print(f"  {r['label']:<40} {r['omega']:>4} {r['throughput_tok_s']:>7.1f}  "
                  f"{r['generation_time_s']:>6.1f}s  {r['peak_vram_mb']:>6.0f}M")
        print(f"{'='*80}")

        # ── Per-config comparison across OMEGAs ──
        print(f"\n  Per-Config OMEGA Comparison:")
        for cfg in BENCHMARK_CONFIGS:
            tag = cfg["tag"]
            cfg_results = [r for r in results if tag in r["label"]]
            if cfg_results:
                rates = ", ".join(f"ω={r['omega']}→{r['throughput_tok_s']:.1f}" for r in cfg_results)
                print(f"    {tag}: {rates}")


if __name__ == "__main__":
    main()
