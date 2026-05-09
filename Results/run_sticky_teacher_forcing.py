"""
run_sticky_teacher_forcing.py
=============================
Sticky KV Cache baseline with teacher-forced generation.

Mirror of run_vanilla_oracle_teacher_forcing.py but with the actual
R_RATIO / Q_RATIO / LOCAL_NUM_TOKENS from sticky_config (eviction active).

Loads the same dataset, same split, same teacher tokens.  The only
differences vs the vanilla oracle are:
    - Eviction and quantization are applied (R_RATIO from config)
    - Attention tracker snapshots are recorded automatically by the cache
    - Per-token cumulative attention is mapped from compressed physical
      positions back to global token positions via logical_id_map

Output is a compressed .npz file with the same schema as the vanilla oracle,
plus serialised attention-tracker snapshots and window-ledger lifecycle data.

Usage (from repo root):
    python Results/run_sticky_teacher_forcing.py
"""

import torch
import numpy as np
import os
import sys
import gc
import time as _time
import random

# ---------------------------------------------------------------------------
# Path setup (Kaggle-safe: no __file__)
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
    Compute per-window scores from a 1-D cumulative attention vector.
    Skips the first `sink_tokens` positions (not part of any window).
    Returns [[score, window_id], ...] sorted by score descending.
    """
    non_sink = cumul_vector[sink_tokens:]
    num_windows = len(non_sink) // omega
    if num_windows == 0:
        return []
    windowed = non_sink[: num_windows * omega].reshape(num_windows, omega)
    window_sums = windowed.sum(axis=1)
    ws_list = [[float(window_sums[w]), int(w)] for w in range(num_windows)]
    ws_list.sort(key=lambda x: x[0], reverse=True)
    return ws_list


def _group_average_attention(attn_tensor, num_kv_heads, group_size):
    """Group-average Q-head attention to KV-head granularity."""
    num_q_heads, q_len, kv_len = attn_tensor.shape
    return (
        attn_tensor
        .view(num_kv_heads, group_size, q_len, kv_len)
        .mean(dim=1)
    )


def _extract_cache_windows(kv_cache, h):
    """
    Extract all alive windows (sticky + q-cache + local) with their
    cumulative scores directly from the cache's internal state.

    This avoids the scatter-back reconstruction that inflates Jaccard
    by making sticky and vanilla scores nearly identical.

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


def _scatter_attention_to_global(
    kv_importance_np,      # [H, kv_len]  float32  — attention over physical cache
    logical_id_map_np,     # [H, compressed_len] int64 — from pre-forward snapshot
    cumulative,            # [H, max_total] float32 — running accumulator (modified in-place)
    sink_tokens,
    omega,
    new_token_global_pos,  # int — global position of the newly appended token
):
    """
    Map attention weights from compressed physical positions to global
    token positions and accumulate into `cumulative`.

    Physical layout after eviction:
        [0 .. sink_tokens)                          → sink tokens  (global = physical)
        [sink_tokens .. sink_tokens + k*omega)      → sticky zone  (window-aligned)
        [sink_tokens + k*omega .. compressed_len)   → local zone   (window-aligned)
        [compressed_len .. kv_len)                  → new tokens since last eviction

    For each physical position p in the compressed region:
        wid = logical_id_map[h, p]
        if wid == -1:  global = p  (sink)
        else:          global = sink_tokens + wid * omega + (p - sink_tokens) % omega

    Tokens appended after the last eviction (positions >= compressed_len)
    are sequential starting from new_token_global_pos - (num_appended - 1).
    """
    num_heads = kv_importance_np.shape[0]
    kv_len = kv_importance_np.shape[1]
    compressed_len = logical_id_map_np.shape[1] if logical_id_map_np is not None else 0
    max_total = cumulative.shape[1]

    for h in range(num_heads):
        # --- Compressed region (mapped by logical_id_map) ---
        usable = min(compressed_len, kv_len)
        if usable > 0 and logical_id_map_np is not None:
            wids = logical_id_map_np[h, :usable]
            phys = np.arange(usable, dtype=np.int64)
            is_sink = wids == -1

            offsets = (phys - sink_tokens) % omega
            global_pos = np.where(
                is_sink,
                phys,
                sink_tokens + wids * omega + offsets,
            )
            # Clamp to valid range
            valid = (global_pos >= 0) & (global_pos < max_total)
            np.add.at(
                cumulative[h],
                global_pos[valid],
                kv_importance_np[h, :usable][valid],
            )

        # --- Tokens appended since last eviction ---
        num_appended = kv_len - compressed_len
        if num_appended > 0:
            # The most recent token is at new_token_global_pos.
            # Earlier appended tokens are at sequential positions before it.
            for j in range(num_appended):
                gp = new_token_global_pos - (num_appended - 1 - j)
                if 0 <= gp < max_total:
                    cumulative[h, gp] += kv_importance_np[h, compressed_len + j]


def _extract_tracker_arrays(model, tracked_layers, num_kv_heads):
    """
    Extract attention-tracker snapshot and ledger data from the model
    after a full generation run.

    Returns a dict of numpy arrays keyed for NPZ storage.
    """
    arrays = {}

    for layer_idx in tracked_layers:
        kv_cache = model.model.layers[layer_idx].self_attn.kv_cache
        tracker = kv_cache.attn_tracker

        # --- Snapshots ---
        for snap_key in tracker.list_snapshots():
            snap = tracker.get_snapshot(snap_key)
            if snap is None:
                continue
            tag = f"L{layer_idx}_snap_{snap_key}"
            arrays[f"tracker_{tag}_window_ids"] = snap.window_ids.numpy()
            arrays[f"tracker_{tag}_step_scores"] = snap.step_scores.numpy()
            arrays[f"tracker_{tag}_cumulative_scores"] = snap.cumulative_scores.numpy()
            # Zones encoded as uint8 indices: 0=sticky, 1=quantized, 2=local
            zone_map = {"sticky": 0, "quantized": 1, "local": 2}
            arrays[f"tracker_{tag}_zones"] = np.array(
                [zone_map.get(z, 255) for z in snap.zones], dtype=np.uint8
            )

        # --- Ledger lifecycle data ---
        for h in range(num_kv_heads):
            entries = tracker.ledger.get_all(h)
            if not entries:
                continue
            wids = []
            cumul_scores = []
            statuses = []
            zones = []
            for wid in sorted(entries.keys()):
                e = entries[wid]
                wids.append(e.window_id)
                cumul_scores.append(e.last_cumulative_score)
                statuses.append(0 if e.status == "alive" else 1)
                zones.append({"sticky": 0, "quantized": 1, "local": 2}.get(e.current_zone, 255))
            arrays[f"tracker_L{layer_idx}_ledger_H{h}_wids"] = np.array(wids, dtype=np.int32)
            arrays[f"tracker_L{layer_idx}_ledger_H{h}_cumul"] = np.array(cumul_scores, dtype=np.float32)
            arrays[f"tracker_L{layer_idx}_ledger_H{h}_status"] = np.array(statuses, dtype=np.uint8)
            arrays[f"tracker_L{layer_idx}_ledger_H{h}_zone"] = np.array(zones, dtype=np.uint8)

    return arrays


# ===========================================================================
# Main
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # 0. Configuration  (uses actual sticky_config values — eviction ON)
    # -----------------------------------------------------------------------
    omega = config.OMEGA
    sink_tokens = config.SINK_TOKENS
    tracked_layers = config.TRACKED_LAYERS
    max_new_tokens = config.GENERATION_CONFIG.get("max_new_tokens", 512)
    seed = config.SEEDS[0]

    # --- DIAGNOSTIC: Log active config values to detect stale overrides ---
    print(f"\n{'#'*70}")
    print(f"# [STICKY RUNNER] ACTIVE CONFIG FROM sticky_config (id={id(config)}):")
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
    OUTPUT_FILE = "sticky_teacher_forcing.npz"

    if os.path.exists(OUTPUT_FILE):
        print(f"Removing existing {OUTPUT_FILE} ...")
        os.remove(OUTPUT_FILE)

    # -----------------------------------------------------------------------
    # 1. Load model with ACTUAL sticky config  (eviction active)
    # -----------------------------------------------------------------------
    print(f"Loading StickyLlama from {config.MODEL_PATH} with R_RATIO={getattr(config, 'R_RATIO', 'N/A')} ...")

    model_config = LlamaConfig.from_pretrained(config.MODEL_PATH)

    # Rope-scaling compatibility shim
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
            model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
    model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)

    # Use actual sticky config ratios (eviction active)
    model_config.r_ratio = getattr(config, "R_RATIO", 50)

    if hasattr(config, "P_RATIO"):
        model_config.p_ratio = config.P_RATIO
    elif hasattr(config, "LOCAL_NUM_TOKENS"):
        model_config.local_num_tokens = config.LOCAL_NUM_TOKENS

    model_config.start_idx = getattr(config, "S_IDX", 0)

    model = STICKYLlamaForCausalLM.from_pretrained(
        config.MODEL_PATH,
        config=model_config,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

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
    # 2. Load dataset  (identical to vanilla oracle — same samples, same split)
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
    all_tracker_arrays = {}  # Tracker data keyed by sample index

    for idx, raw_sample in enumerate(raw_samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(raw_samples)}")
        print(f"{'='*60}")

        text = raw_sample["text"].strip()

        # --- 3a. Tokenize and split in half (identical to vanilla oracle) ---
        full_tokens = tokenizer(text, add_special_tokens=False).input_ids
        half = len(full_tokens) // 2
        prompt_ids = full_tokens[:half]
        teacher_ids = full_tokens[half:]

        # Cap teacher tokens at max_new_tokens
        teacher_ids = teacher_ids[:max_new_tokens]
        prefill_len = len(prompt_ids)
        num_teacher_steps = len(teacher_ids)
        max_total_tokens = prefill_len + num_teacher_steps

        # Patch max_new_tokens to actual teacher length so the KV cache
        # computes its budget from the real sequence, not the config ceiling.
        config.GENERATION_CONFIG["max_new_tokens"] = num_teacher_steps
        effective_budget = int((prefill_len + num_teacher_steps) * getattr(config, 'R_RATIO', 20) / 100)
        print(f"  Full tokens: {len(full_tokens)}")
        print(f"  Prompt (prefill): {prefill_len}")
        print(f"  Teacher tokens:   {num_teacher_steps} (capped at {max_new_tokens})")
        print(f"  Budget: {effective_budget} tokens "
              f"(= ({prefill_len}+{num_teacher_steps}) × {getattr(config, 'R_RATIO', 20)}%)")

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

        # Cumulative attention in GLOBAL token space
        cumulative = {
            li: np.zeros((num_kv_heads, max_total_tokens), dtype=np.float32)
            for li in tracked_layers
        }

        # --- Extract prefill attention ---
        # During prefill the cache is NOT yet compressed, so physical == global.
        prefill_data = {}
        prefill_window_scores = {}

        for layer_idx in tracked_layers:
            attn = prefill_out.attentions[layer_idx]  # [1, Q, prefill_len, prefill_len]
            attn_kv = _group_average_attention(attn[0].float(), num_kv_heads, group_size)
            kv_importance = attn_kv.sum(dim=1).cpu().numpy()  # [KV, prefill_len]

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

        del prefill_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 3d. Teacher-forced generation (with eviction) ---
        print(f"  Running teacher-forced generation ({num_teacher_steps} steps) ...")
        generation_data = []
        generation_window_scores = []

        t0 = _time.time()

        for step in range(num_teacher_steps):
            token_id = teacher_ids[step]
            token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)

            # ── Capture PRE-FORWARD logical_id_map for attention mapping ──
            # The attention weights returned by the forward pass are computed
            # BEFORE the eviction call, so we need the mapping that was valid
            # at that point (i.e., the post-eviction state from the *previous* step).
            pre_lid_maps = {}
            for layer_idx in tracked_layers:
                kv_cache = model.model.layers[layer_idx].self_attn.kv_cache
                if kv_cache.logical_id_map is not None:
                    pre_lid_maps[layer_idx] = kv_cache.logical_id_map.cpu().numpy().copy()
                else:
                    pre_lid_maps[layer_idx] = None

            with torch.no_grad():
                gen_out = model(
                    input_ids=token_tensor,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                )
            past_kv = gen_out.past_key_values

            # Global position of the newly generated token
            new_token_global = prefill_len + step

            # Accumulate attention → global positions
            for layer_idx in tracked_layers:
                attn = gen_out.attentions[layer_idx]  # [1, Q, 1, kv_len]
                attn_kv = _group_average_attention(
                    attn[0].float(), num_kv_heads, group_size
                )
                kv_importance = attn_kv[:, 0, :].cpu().numpy()  # [KV, kv_len]

                _scatter_attention_to_global(
                    kv_importance,
                    pre_lid_maps[layer_idx],
                    cumulative[layer_idx],
                    sink_tokens,
                    omega,
                    new_token_global,
                )

            # Snapshot at OMEGA boundaries or final step
            total_tokens = prefill_len + step + 1
            is_save_step = (
                ((step + 1) % omega == 0) or (step == num_teacher_steps - 1)
            )

            if is_save_step:
                step_data = {}
                step_ws = {}
                for layer_idx in tracked_layers:
                    layer_step = {}
                    layer_ws = {}
                    kv_cache = model.model.layers[layer_idx].self_attn.kv_cache
                    for h in tracked_heads:
                        cumul_snapshot = cumulative[layer_idx][h, :total_tokens].copy()
                        layer_step[str(h)] = cumul_snapshot

                        # Use cache-internal cumulative scores directly.
                        # The external cumulative vector shares identical prefill
                        # attention with vanilla, inflating Jaccard by making both
                        # sides agree on rankings due to shared prefill history.
                        layer_ws[str(h)] = _extract_cache_windows(kv_cache, h)
                    step_data[str(layer_idx)] = layer_step
                    step_ws[str(layer_idx)] = layer_ws

                generation_data.append(step_data)
                generation_window_scores.append(step_ws)
            else:
                generation_data.append({})
                generation_window_scores.append({})

            # Progress
            if (step + 1) % 25 == 0 or step == 0:
                elapsed = _time.time() - t0
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                # Report cache size and alive windows on snapshot steps
                cache_len = past_kv[0][0].shape[2] if past_kv is not None else 0
                alive_info = ""
                if is_save_step and generation_window_scores:
                    last_ws = generation_window_scores[-1]
                    for _l in last_ws.values():
                        for _h in _l.values():
                            alive_info = f"[SNAPSHOT] alive_windows={len(_h)}"
                            break
                        break
                print(f"    Step {step+1}/{num_teacher_steps} | "
                      f"{rate:.1f} tok/s | "
                      f"cache_len={cache_len} | "
                      f"total_tokens={total_tokens} | "
                      f"{alive_info}",
                      flush=True)

            del gen_out
            if (step + 1) % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed_total = _time.time() - t0
        print(f"  Generation done in {elapsed_total:.1f}s "
              f"({num_teacher_steps / elapsed_total:.1f} tok/s)")

        # --- 3e. Extract attention tracker data ---
        tracker_arrays = _extract_tracker_arrays(model, tracked_layers, num_kv_heads)
        for k, v in tracker_arrays.items():
            all_tracker_arrays[f"sample_{idx}_{k}"] = v

        # --- 3f. Build result entry ---
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
        del past_kv, cumulative, pre_lid_maps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 4. Save  (standard schema + tracker arrays)
    # -----------------------------------------------------------------------
    # Save everything in one pass — avoids reading the file back just to append tracker arrays
    save_results_npz(results, OUTPUT_FILE, extra_arrays=all_tracker_arrays if all_tracker_arrays else None)
    if all_tracker_arrays:
        print(f"  (Included {len(all_tracker_arrays)} tracker arrays in the same file)")

    print(f"\nSaved sticky teacher-forcing results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
