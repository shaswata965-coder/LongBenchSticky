"""
run_vanilla_oracle_teacher_forcing.py
=====================================
Vanilla Oracle baseline with teacher-forced generation.

Loads the Sticky Llama model with R_RATIO=100 (full cache — no eviction),
uses PG-19 or WikiText based on sticky_config.dataset_tracker, splits each
sample in half:
    - First half  → prefill (prompt)
    - Second half → teacher-forced generation (capped at max_new_tokens)

At each generation step the attention row is accumulated into a per-token
cumulative score vector.  At every OMEGA boundary the cumulative vector is
sliced into omega-sized windows (excluding the first `sink_tokens` positions)
and a sorted snapshot is stored.

Output is a compressed .npz file with exactly the same schema that
`calculate_layer_information_retention.py` and `calculate_window_jaccard.py`
expect as the "vanilla" baseline, plus a `teacher_forcing: True` flag in
metadata so downstream scripts can distinguish it from open-ended runs.

Usage (from repo root):
    python Results/run_vanilla_oracle_teacher_forcing.py
"""

import torch
import numpy as np
import os
import sys
import gc
import time as _time
import random

# ---------------------------------------------------------------------------
# Path setup — make repo root importable (Kaggle-safe: no __file__)
# ---------------------------------------------------------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()  # Kaggle / Jupyter fallback
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if SCRIPT_DIR != os.getcwd() else os.getcwd()
for _p in [REPO_ROOT, SCRIPT_DIR, os.path.join(REPO_ROOT, "Results"), os.path.join(REPO_ROOT, "Dataset")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig
import sticky_config as config
from npz_io import save_results_npz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_window_scores(cumul_vector, sink_tokens, omega):
    """
    Given a 1-D cumulative attention vector for one head, compute per-window
    scores by summing each omega-sized block *after* the sink tokens.

    Returns a list of [score, window_id] pairs sorted by score descending.
    """
    non_sink = cumul_vector[sink_tokens:]
    num_windows = len(non_sink) // omega
    if num_windows == 0:
        return []
    windowed = non_sink[: num_windows * omega].reshape(num_windows, omega)
    window_sums = windowed.sum(axis=1)
    # Build [score, id] pairs and sort descending by score
    ws_list = [[float(window_sums[w]), int(w)] for w in range(num_windows)]
    ws_list.sort(key=lambda x: x[0], reverse=True)
    return ws_list


def _group_average_attention(attn_tensor, num_kv_heads, group_size):
    """
    Group-average Q-head attention weights to KV-head granularity.

    Args:
        attn_tensor: [num_q_heads, q_len, kv_len]  (already squeezed batch dim)
    Returns:
        [num_kv_heads, q_len, kv_len]
    """
    num_q_heads, q_len, kv_len = attn_tensor.shape
    return (
        attn_tensor
        .view(num_kv_heads, group_size, q_len, kv_len)
        .mean(dim=1)
    )


def _extract_cache_windows(kv_cache, h):
    """
    Extract all alive windows with their cumulative scores directly
    from the cache's internal state (window_scores, q_cache_scores,
    local_history).

    Uses the same score source as the eviction logic, avoiding the
    external cumulative vector whose prefill component is shared
    with the sticky runner (which inflates Jaccard).

    Returns [[score, wid], ...] sorted by score descending.
    """
    ws_list = []
    seen_ids = set()

    # 1. Sticky windows — from window_scores [H, max_windows, 3]
    ws = kv_cache.window_scores
    valid_mask = ~torch.isnan(ws[h, :, 1])
    valid_k = int(valid_mask.sum().item())
    if valid_k > 0:
        scores = ws[h, :valid_k, 0].cpu().tolist()
        ids = ws[h, :valid_k, 1].cpu().long().tolist()
        for score, wid in zip(scores, ids):
            ws_list.append([score, wid])
            seen_ids.add(wid)

    # 2. Q-cache windows — from q_cache_ids/scores
    if kv_cache.q_cache_ids is not None and kv_cache.q_cache_ids.shape[1] > 0:
        q_scores = kv_cache.q_cache_scores[h].cpu().tolist()
        q_ids = kv_cache.q_cache_ids[h].cpu().long().tolist()
        for score, wid in zip(q_scores, q_ids):
            if wid not in seen_ids:
                ws_list.append([score, wid])
                seen_ids.add(wid)

    # 3. Local zone windows — IDs from logical_id_map, scores from local_history
    if kv_cache.logical_id_map is not None:
        lid = kv_cache.logical_id_map[h].cpu().tolist()
        local_wids = set()
        for w in lid:
            if w >= 0 and w not in seen_ids:
                local_wids.add(w)
        for wid in local_wids:
            if wid < kv_cache.local_history.shape[1]:
                score = float(kv_cache.local_history[h, wid].item())
            else:
                score = 0.0
            ws_list.append([score, wid])
            seen_ids.add(wid)

    # Sort by score descending
    ws_list.sort(key=lambda x: x[0], reverse=True)
    return ws_list


# ===========================================================================
# Main
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # 0. Configuration
    # -----------------------------------------------------------------------
    omega = config.OMEGA
    sink_tokens = config.SINK_TOKENS
    tracked_layers = config.TRACKED_LAYERS
    max_new_tokens = config.GENERATION_CONFIG.get("max_new_tokens", 512)
    seed = config.SEEDS[0]

    # --- DIAGNOSTIC: Log active config values to detect stale overrides ---
    print(f"\n{'#'*70}")
    print(f"# [VANILLA RUNNER] ACTIVE CONFIG FROM sticky_config (id={id(config)}):")
    print(f"#   R_RATIO            = {getattr(config, 'R_RATIO', 'MISSING')}")
    print(f"#   OMEGA              = {config.OMEGA}")
    print(f"#   SINK_TOKENS        = {config.SINK_TOKENS}")
    print(f"#   Q_RATIO            = {getattr(config, 'Q_RATIO', 'MISSING')}")
    print(f"#   QUANTIZATION_BIT_WIDTH = {getattr(config, 'QUANTIZATION_BIT_WIDTH', 'MISSING')}")
    print(f"#   LOCAL_NUM_TOKENS   = {getattr(config, 'LOCAL_NUM_TOKENS', 'NOT SET')}")
    print(f"#   P_RATIO            = {getattr(config, 'P_RATIO', 'NOT SET')}")
    print(f"#   max_new_tokens     = {max_new_tokens}")
    print(f"#   Config file path   = {getattr(config, '__file__', 'UNKNOWN')}")
    print(f"{'#'*70}\n")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Output path — flat file for Kaggle compatibility
    OUTPUT_FILE = "vanilla_oracle_teacher_forcing.npz"

    if os.path.exists(OUTPUT_FILE):
        print(f"Removing existing {OUTPUT_FILE} ...")
        os.remove(OUTPUT_FILE)

    # -----------------------------------------------------------------------
    # 1. Load model with R_RATIO=100  (full cache — no eviction)
    # -----------------------------------------------------------------------
    print(f"Loading StickyLlama from {config.MODEL_PATH} with R_RATIO=100 ...")

    model_config = LlamaConfig.from_pretrained(config.MODEL_PATH)

    # Rope-scaling compatibility shim
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
            model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
    model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)

    # Override cache ratios for full-cache vanilla oracle
    model_config.r_ratio = 100       # 100 % of sequence retained
    model_config.p_ratio = 0         # No local sliding window needed
    model_config.local_num_tokens = 0
    model_config.q_ratio = 0         # No quantized q-cache for vanilla oracle
    model_config.quant_bit_width = 8 # Default (unused since q_ratio=0)
    model_config.start_idx = 0

    # Monkey-patch sticky_config so the cache constructor (which imports
    # directly from sticky_config) picks up vanilla-appropriate values.
    import sticky_config as _sc
    _saved_q_ratio = getattr(_sc, 'Q_RATIO', 0)
    _saved_local   = getattr(_sc, 'LOCAL_NUM_TOKENS', 0)
    _saved_qbw     = getattr(_sc, 'QUANTIZATION_BIT_WIDTH', 8)
    _sc.Q_RATIO = 0
    _sc.LOCAL_NUM_TOKENS = 0
    _sc.QUANTIZATION_BIT_WIDTH = 8

    model = STICKYLlamaForCausalLM.from_pretrained(
        config.MODEL_PATH,
        config=model_config,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Restore sticky_config to original values
    _sc.Q_RATIO = _saved_q_ratio
    _sc.LOCAL_NUM_TOKENS = _saved_local
    _sc.QUANTIZATION_BIT_WIDTH = _saved_qbw

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"Model loaded on {device}.")

    num_q_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    group_size = num_q_heads // num_kv_heads
    tracked_heads = list(range(num_kv_heads))

    # -----------------------------------------------------------------------
    # 2. Load dataset
    # -----------------------------------------------------------------------
    if config.dataset_tracker == 1:
        sys.path.insert(0, os.path.join(REPO_ROOT, "Dataset"))
        from pg19_loader import get_pg19_blocks
        raw_samples = get_pg19_blocks(
            config.MODEL_PATH,
            num_samples=config.NUM_SAMPLES,
            min_tokens=config.DATASET_MIN_TOKENS,
        )
    else:
        sys.path.insert(0, os.path.join(REPO_ROOT, "Dataset"))
        from wiki_text_loader import get_wikitext103_drift_blocks
        raw_samples = get_wikitext103_drift_blocks(
            config.MODEL_PATH,
            num_samples=config.NUM_SAMPLES,
            min_tokens=config.DATASET_MIN_TOKENS,
        )

    print(f"Loaded {len(raw_samples)} samples.")

    # -----------------------------------------------------------------------
    # 3. Process each sample
    # -----------------------------------------------------------------------
    results = []

    for idx, raw_sample in enumerate(raw_samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(raw_samples)}")
        print(f"{'='*60}")

        text = raw_sample["text"].strip()

        # --- 3a. Tokenize and split in half ---
        full_tokens = tokenizer(text, add_special_tokens=False).input_ids
        half = len(full_tokens) // 2
        prompt_ids = full_tokens[:half]
        teacher_ids = full_tokens[half:]

        # Cap teacher tokens at max_new_tokens
        teacher_ids = teacher_ids[:max_new_tokens]
        prefill_len = len(prompt_ids)
        num_teacher_steps = len(teacher_ids)
        max_total_tokens = prefill_len + num_teacher_steps

        print(f"  Full tokens: {len(full_tokens)}")
        print(f"  Prompt (prefill): {prefill_len}")
        print(f"  Teacher tokens:   {num_teacher_steps} (capped at {max_new_tokens})")

        # --- 3b. Reset caches ---
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "_clean_cache"):
                layer.self_attn._clean_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 3c. Prefill ---
        print(f"  Running prefill ({prefill_len} tokens) ...")
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            prefill_out = model(
                input_ids=prompt_tensor,
                use_cache=True,
                output_attentions=True,
            )
        past_kv = prefill_out.past_key_values

        # Initialise cumulative attention arrays (per layer × kv_heads × max_total_tokens)
        cumulative = {
            li: np.zeros((num_kv_heads, max_total_tokens), dtype=np.float32)
            for li in tracked_layers
        }

        # --- Extract prefill attention ---
        prefill_data = {}
        prefill_window_scores = {}

        for layer_idx in tracked_layers:
            attn = prefill_out.attentions[layer_idx]  # [1, Q, prefill_len, prefill_len]
            # Group-average to KV heads: [KV, prefill_len, prefill_len]
            attn_kv = _group_average_attention(attn[0].float(), num_kv_heads, group_size)
            # Column sum = total attention received per token: [KV, prefill_len]
            kv_importance = attn_kv.sum(dim=1).cpu().numpy()

            cumulative[layer_idx][:, :prefill_len] = kv_importance

            layer_data = {}
            layer_ws = {}
            for h in tracked_heads:
                layer_data[str(h)] = kv_importance[h].tolist()
                layer_ws[str(h)] = _compute_window_scores(
                    kv_importance[h], sink_tokens, omega
                )
            prefill_data[str(layer_idx)] = layer_data
            prefill_window_scores[str(layer_idx)] = layer_ws

        # Free prefill attention tensors
        del prefill_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 3d. Teacher-forced generation ---
        print(f"  Running teacher-forced generation ({num_teacher_steps} steps) ...")
        generation_data = []
        generation_window_scores = []

        t0 = _time.time()

        for step in range(num_teacher_steps):
            token_id = teacher_ids[step]
            token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)

            with torch.no_grad():
                gen_out = model(
                    input_ids=token_tensor,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                )
            past_kv = gen_out.past_key_values

            # Current total tokens in the cache
            total_tokens = prefill_len + step + 1

            # Accumulate attention into cumulative array AND keep fresh per-step
            fresh_attn = {}  # {layer_idx: kv_row [KV, kv_len]}
            for layer_idx in tracked_layers:
                attn = gen_out.attentions[layer_idx]  # [1, Q, 1, kv_len]
                attn_kv = _group_average_attention(
                    attn[0].float(), num_kv_heads, group_size
                )
                # attn_kv shape: [KV, 1, kv_len]
                kv_row = attn_kv[:, 0, :].cpu().numpy()  # [KV, kv_len]
                actual_kv_len = kv_row.shape[1]
                cumulative[layer_idx][:, :actual_kv_len] += kv_row
                fresh_attn[layer_idx] = kv_row

            # Snapshot at OMEGA boundaries or final step
            is_save_step = (
                ((step + 1) % omega == 0) or (step == num_teacher_steps - 1)
            )

            if is_save_step:
                step_data = {}
                step_ws = {}
                for layer_idx in tracked_layers:
                    layer_step = {}
                    layer_ws = {}
                    for h in tracked_heads:
                        # Cumulative attention snapshot for this head
                        cumul_snapshot = cumulative[layer_idx][h, :total_tokens].copy()
                        layer_step[str(h)] = cumul_snapshot

                        # Use THIS STEP's fresh attention for window ranking.
                        # This gives vanilla's top-K as what a base model attends
                        # to RIGHT NOW — independent of prefill history — so the
                        # Recall metric tests whether sticky retains the windows
                        # that matter at each generation step.
                        fresh_vec = fresh_attn[layer_idx][h, :total_tokens]
                        layer_ws[str(h)] = _compute_window_scores(
                            fresh_vec, sink_tokens, omega
                        )
                    step_data[str(layer_idx)] = layer_step
                    step_ws[str(layer_idx)] = layer_ws

                generation_data.append(step_data)
                generation_window_scores.append(step_ws)
            else:
                # Empty dict for non-snapshot steps (matches existing schema)
                generation_data.append({})
                generation_window_scores.append({})

            # Progress logging
            if (step + 1) % 25 == 0 or step == 0:
                elapsed = _time.time() - t0
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                num_ws = 0
                if is_save_step and generation_window_scores:
                    # Report window count from last snapshot
                    last_ws = generation_window_scores[-1]
                    for _l in last_ws.values():
                        for _h in _l.values():
                            num_ws = len(_h)
                            break
                        break
                print(f"    Step {step+1}/{num_teacher_steps} | "
                      f"{rate:.1f} tok/s | "
                      f"total_tokens={total_tokens} | "
                      f"{'[SNAPSHOT] windows=' + str(num_ws) if is_save_step else ''}",
                      flush=True)

            del gen_out
            # Periodic VRAM cleanup
            if (step + 1) % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed_total = _time.time() - t0
        print(f"  Generation done in {elapsed_total:.1f}s "
              f"({num_teacher_steps / elapsed_total:.1f} tok/s)")

        # --- 3e. Build result entry ---
        result_entry = {
            "metadata": {
                "sha256": raw_sample["sha256"],
                "article_index": raw_sample["article_index"],
                "token_count_input": prefill_len,
                "generated_token_count": num_teacher_steps,
                "generated_token_ids": teacher_ids,
                "truncation_char_index": half,
                "teacher_forcing": True,
            },
            "tracked_layers": tracked_layers,
            "tracked_heads": tracked_heads,
            "prefill_attention": prefill_data,
            "prefill_window_scores": prefill_window_scores,
            "generation_attention": generation_data,
            "generation_window_scores": generation_window_scores,
        }
        results.append(result_entry)

        # Cleanup
        del past_kv, cumulative
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 4. Save
    # -----------------------------------------------------------------------
    save_results_npz(results, OUTPUT_FILE)
    print(f"\nSaved vanilla oracle teacher-forcing results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
