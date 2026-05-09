import json
import os
import numpy as np
import sys
import argparse
from collections import defaultdict

from npz_io import load_results_npz
from sticky_config import OMEGA, SINK_TOKENS

try:
    from sticky_config import LOCAL_NUM_TOKENS
    use_fixed_local_tokens = True
    P_RATIO = None
except ImportError:
    from sticky_config import P_RATIO
    use_fixed_local_tokens = False
    LOCAL_NUM_TOKENS = None

try:
    from sticky_config import K_TOP
except ImportError:
    raise RuntimeError(
        "[calculate_window_recall] K_TOP is not defined in sticky_config.py.\n"
        "Add  K_TOP = <number>  to sticky_config.py before running this script.\n"
        "Examples: K_TOP = 0.5  (50% of vanilla windows)\n"
        "          K_TOP = 20   (fixed top-20)"
    )

# K_TOP can be:
#   float in (0.0, 1.0] → ratio of vanilla's total windows (e.g. 0.5 = top 50%)
#   int > 0             → fixed number of top windows
_K_TOP_IS_RATIO = isinstance(K_TOP, float) and 0.0 < K_TOP <= 1.0
_K_TOP_IS_FIXED = isinstance(K_TOP, int) and K_TOP > 0
if not (_K_TOP_IS_RATIO or _K_TOP_IS_FIXED):
    raise ValueError(
        f"[calculate_window_recall] K_TOP must be a positive int or a float in (0,1], got: {K_TOP!r}\n"
        "Fix sticky_config.py: K_TOP = 0.5  (ratio) or  K_TOP = 20  (fixed)"
    )

def _resolve_k(k_top, n_vanilla):
    """Convert K_TOP (ratio or fixed) to an absolute integer.
    
    When K_TOP is a ratio, K is computed against VANILLA's window count
    so that K stays constant across sticky configurations. This ensures
    adding q-cache windows can only improve recall (more alive IDs to
    match against, same target set).
    """
    if isinstance(k_top, float) and k_top <= 1.0:
        return max(1, int(round(k_top * n_vanilla)))
    return int(k_top)

def get_local_num(new_tokens, max_tokens=100, total_cache_ratio=20):
    total_token_budget = (new_tokens + max_tokens) * total_cache_ratio // 100
    if use_fixed_local_tokens:
        target_local_tokens = LOCAL_NUM_TOKENS
    else:
        target_local_tokens = (total_token_budget * P_RATIO) // 100
    return min(target_local_tokens, total_token_budget)

# Default paths — flat files for Kaggle compatibility
VANILLA_PATH = "vanilla_oracle_teacher_forcing.npz"
STICKY_PATH = "sticky_teacher_forcing.npz"
DETAILED_OUTPUT_PATH = "teacher_forcing_recall_results.json"

# ---------------------------------------------------------------------------
# Core metric: Recall@K
# ---------------------------------------------------------------------------

def calculate_recall(v_ws, s_ws, k_top, gen_diag=None):
    """
    Recall@K: what fraction of vanilla's top-K windows are in
    the sticky cache's alive set?

    Recall = |vanilla_top_K ∩ sticky_alive_set| / K

    K is resolved against VANILLA's window count so that K stays
    constant across sticky configurations (different Q_RATIO, OMEGA, etc.).
    This ensures adding q-cache windows can only improve recall.
      - float (0,1] → ratio of vanilla's total windows
      - int > 0     → fixed count
    Then capped at vanilla's total (can't ask for more than exist).

    Args:
        v_ws:   Vanilla window scores [[score, wid], ...]
        s_ws:   Sticky window scores [[score, wid], ...] (ALL alive windows)
        k_top:  K_TOP value (ratio or fixed)
        gen_diag: Optional diagnostic dict
    Returns:
        recall: float in [0.0, 1.0]
    """
    if len(v_ws) == 0 or len(s_ws) == 0:
        if gen_diag is not None:
            gen_diag['empty_comparisons'] += 1
            gen_diag['sticky_zero_windows'] += (1 if len(s_ws) == 0 else 0)
            gen_diag['vanilla_zero_windows'] += (1 if len(v_ws) == 0 else 0)
        return 0.0

    # Resolve K from vanilla's window count (constant across sticky configs)
    resolved_k = _resolve_k(k_top, len(v_ws))
    effective_k = min(len(v_ws), resolved_k)
    if effective_k == 0:
        return 0.0

    if gen_diag is not None:
        gen_diag['total_comparisons'] += 1
        gen_diag['sticky_window_counts'].append(len(s_ws))
        gen_diag['vanilla_window_counts'].append(len(v_ws))
        gen_diag['effective_k_values'].append(effective_k)

    # Vanilla's top-K by score
    v_ws_sorted = sorted(v_ws, key=lambda x: float(x[0]), reverse=True)
    v_top_k_ids = set(int(x[1]) for x in v_ws_sorted[:effective_k])

    # Sticky's FULL alive set (all window IDs, regardless of score)
    s_alive_ids = set(int(x[1]) for x in s_ws)

    # How many of vanilla's top-K are alive in sticky?
    hits = len(v_top_k_ids & s_alive_ids)
    recall = hits / effective_k

    return recall


def calculate_jaccard_prefill(v_ws, s_ws, k_top, debug_info=None):
    """
    Top-K Jaccard for PREFILL only (both caches are identical here).
    Kept as a sanity check — should always be ~1.0.
    """
    if len(v_ws) == 0 or len(s_ws) == 0:
        return 0.0

    resolved_k = _resolve_k(k_top, len(v_ws))
    effective_k = min(len(v_ws), len(s_ws), resolved_k)
    if effective_k == 0:
        return 0.0

    v_ws_sorted = sorted(v_ws, key=lambda x: float(x[0]), reverse=True)
    v_top_k_ids = set(int(x[1]) for x in v_ws_sorted[:effective_k])

    s_ws_sorted = sorted(s_ws, key=lambda x: float(x[0]), reverse=True)
    s_top_k_ids = set(int(x[1]) for x in s_ws_sorted[:effective_k])

    intersection = len(v_top_k_ids & s_top_k_ids)
    union = len(v_top_k_ids | s_top_k_ids)

    jaccard = intersection / union if union > 0 else 0.0

    if debug_info and jaccard < 1.0:
        L, H = debug_info
        print(f"\n[DEBUG] Jaccard Divergence at PREFILL Layer {L}, Head {H} | Jaccard = {jaccard:.4f}")
        missed = v_top_k_ids - s_top_k_ids
        print(f"  -> Vanilla top-K MISSED by sticky top-K: {sorted(list(missed))}")

    return jaccard


# ---------------------------------------------------------------------------
# Layer/head aggregation
# ---------------------------------------------------------------------------

def get_layer_head_recall(v_layer_data, s_layer_data, k, gen_diag=None):
    """Compute Recall@K for each head in a layer."""
    recalls = {}
    for h_str in v_layer_data:
        if h_str not in s_layer_data:
            continue
        v_ws = v_layer_data[h_str]
        s_ws = s_layer_data[h_str]
        recalls[int(h_str)] = calculate_recall(v_ws, s_ws, k, gen_diag=gen_diag)
    return recalls


def get_layer_head_jaccard_prefill(v_layer_data, s_layer_data, k, layer=None):
    """Compute Jaccard for each head in a layer (prefill only)."""
    jaccards = {}
    for h_str in v_layer_data:
        if h_str not in s_layer_data:
            continue
        v_ws = v_layer_data[h_str]
        s_ws = s_layer_data[h_str]
        debug_info = (layer, h_str)
        jaccards[int(h_str)] = calculate_jaccard_prefill(v_ws, s_ws, k, debug_info=debug_info)
    return jaccards


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _k_label():
    """Human-readable label for the K_TOP setting."""
    if isinstance(K_TOP, float) and K_TOP <= 1.0:
        return f"Top-{K_TOP:.0%}"
    return f"Top-{K_TOP}"

def print_summary_table(title, metric_name, layer_averages, overall_value):
    k_label = _k_label()
    print(f"\n{'=' * 60}")
    print(f"=== {title} {metric_name} ({k_label} Windows) ===")
    print(f"{'=' * 60}")
    print(f"{'Layer':<10} | {'Head':<10} | {metric_name:<15}")
    print("-" * 60)

    sorted_layers = sorted(layer_averages.keys())
    for L in sorted_layers:
        sorted_heads = sorted(layer_averages[L].keys())
        for H in sorted_heads:
            print(f"{L:<10} | {H:<10} | {layer_averages[L][H]:.4f}")

    print("-" * 60)
    print(f"{'OVERALL':<10} | {'ALL':<10} | {overall_value:.4f}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- DIAGNOSTIC: Log active config values to detect stale overrides ---
    import sticky_config as _diag_cfg
    print(f"\n{'#'*70}")
    print(f"# [RECALL] ACTIVE CONFIG FROM sticky_config (id={id(_diag_cfg)}):")
    print(f"#   OMEGA              = {OMEGA}")
    print(f"#   SINK_TOKENS        = {SINK_TOKENS}")
    print(f"#   K_TOP              = {K_TOP}")
    print(f"#   LOCAL_NUM_TOKENS   = {LOCAL_NUM_TOKENS}")
    print(f"#   P_RATIO            = {P_RATIO}")
    print(f"#   use_fixed_local    = {use_fixed_local_tokens}")
    print(f"#   R_RATIO            = {getattr(_diag_cfg, 'R_RATIO', 'MISSING')}")
    print(f"#   Q_RATIO            = {getattr(_diag_cfg, 'Q_RATIO', 'MISSING')}")
    print(f"#   Config file path   = {getattr(_diag_cfg, '__file__', 'UNKNOWN')}")
    print(f"{'#'*70}\n")

    print(f"Vanilla path: {VANILLA_PATH}")
    print(f"Sticky path:  {STICKY_PATH}")
    k_desc = f"{K_TOP:.0%} of sticky alive windows" if isinstance(K_TOP, float) and K_TOP <= 1.0 else f"fixed {K_TOP}"
    print(f"K_TOP:        {K_TOP} ({k_desc})")
    if not os.path.exists(VANILLA_PATH) or not os.path.exists(STICKY_PATH):
        print("Error: Missing teacher-forcing NPZ files. Run both baselines first:")
        print(f"  python Results/run_vanilla_oracle_teacher_forcing.py")
        print(f"  python Results/run_sticky_teacher_forcing.py")
        sys.exit(1)

    if os.path.exists(DETAILED_OUTPUT_PATH):
        print(f"Removing existing {DETAILED_OUTPUT_PATH} to prevent appending bugs...")
        os.remove(DETAILED_OUTPUT_PATH)

    # --- Version gate: crash if NPZ files are from old code ---
    EXPECTED_VERSION = b"v3_alive_filter"
    for tag, path in [("Vanilla", VANILLA_PATH), ("Sticky", STICKY_PATH)]:
        raw = np.load(path, allow_pickle=False)
        if "__version__" not in raw:
            raise RuntimeError(
                f"FATAL: {tag} NPZ ({path}) has NO version tag. "
                f"Re-run the runner with the updated code (npz_io with v3_alive_filter)."
            )
        ver = raw["__version__"].tobytes()
        if ver != EXPECTED_VERSION:
            raise RuntimeError(
                f"FATAL: {tag} NPZ ({path}) version mismatch: "
                f"got {ver!r}, expected {EXPECTED_VERSION!r}. Re-run the runner."
            )
        print(f"  {tag} NPZ version: {ver.decode()} ✓")
        raw.close()

    v_data = load_results_npz(VANILLA_PATH, skip_attention=True)
    s_data = load_results_npz(STICKY_PATH, skip_attention=True)

    prefill_jaccards = []
    prefill_lh = {}
    gen_recalls = []       # ALL steps (for JSON)
    gen_lh = {}            # ALL steps per layer/head (for JSON)
    gen_last_lh = {}       # LAST step only per layer/head/sample (for summary table)
    gen_last_recalls = []  # LAST step only (for overall summary)

    gen_diag = {
        'empty_comparisons': 0,
        'sticky_zero_windows': 0,
        'vanilla_zero_windows': 0,
        'total_comparisons': 0,
        'sticky_window_counts': [],
        'vanilla_window_counts': [],
        'effective_k_values': []
    }

    detailed_records = []

    num_samples = min(len(v_data), len(s_data))

    print(f"\nProcessing {num_samples} parallel samples...")
    for idx in range(num_samples):
        print(f"  Sample {idx+1}/{num_samples} ...", flush=True)
        v = v_data[idx]
        s = s_data[idx]

        prefill_seq_len = v["metadata"].get("token_count_input", 0)
        v_gen_steps = v.get("generation_window_scores", [])
        s_gen_steps = s.get("generation_window_scores", [])

        sample_record = {
            "sample_index": idx,
            "layers": {}
        }

        # --- Prefill Stage (Jaccard — sanity check, should be ~1.0) ---
        v_pre = v.get("prefill_window_scores", {})
        s_pre = s.get("prefill_window_scores", {})

        if len(v_pre) > 0 and len(s_pre) > 0:
            for l_str in v_pre:
                if l_str not in s_pre: continue
                layer_id = int(l_str)

                if layer_id not in prefill_lh:
                    prefill_lh[layer_id] = {}
                if str(layer_id) not in sample_record["layers"]:
                    sample_record["layers"][str(layer_id)] = {"heads": {}}

                head_jaccards = get_layer_head_jaccard_prefill(
                    v_pre[l_str], s_pre[l_str], K_TOP, layer=layer_id
                )
                for h, j_score in head_jaccards.items():
                    if h not in prefill_lh[layer_id]:
                        prefill_lh[layer_id][h] = []
                    prefill_lh[layer_id][h].append(j_score)
                    prefill_jaccards.append(j_score)

                    if str(h) not in sample_record["layers"][str(layer_id)]["heads"]:
                        sample_record["layers"][str(layer_id)]["heads"][str(h)] = {
                            "prefill_steps": [],
                            "generation_steps": []
                        }

                    sample_record["layers"][str(layer_id)]["heads"][str(h)]["prefill_steps"].append({
                        "step": 0,
                        "jaccard_similarity": j_score
                    })

        # --- Generation Stage (Recall@K) ---
        steps = min(len(v_gen_steps), len(s_gen_steps))
        # Track last-step recalls for this sample (overwritten each OMEGA step)
        sample_last_recalls = {}  # {layer_id: {head: recall}}

        for step in range(steps):
            # Only evaluate at OMEGA boundaries (when sticky updates rankings)
            if (step + 1) % OMEGA != 0:
                continue

            v_g = v_gen_steps[step]
            s_g = s_gen_steps[step]

            gen_seq_len = prefill_seq_len + step + 1

            # === DIAGNOSTIC DUMP: last step of sample 0 ===
            if idx == 0 and step == steps - 1:
                print(f"\n{'!'*70}")
                print(f"DIAGNOSTIC DUMP — Sample 0, Step {step} (seq_len={gen_seq_len})")
                print(f"{'!'*70}")
                diag_l = list(v_g.keys())[0] if v_g else None
                if diag_l and diag_l in s_g:
                    diag_h = list(v_g[diag_l].keys())[0]
                    v_ws_diag = v_g[diag_l].get(diag_h, [])
                    s_ws_diag = s_g[diag_l].get(diag_h, [])
                    v_sorted = sorted(v_ws_diag, key=lambda x: float(x[0]), reverse=True)
                    ek = min(len(v_sorted), _resolve_k(K_TOP, len(v_ws_diag)))
                    print(f"  Layer={diag_l}, Head={diag_h}")
                    print(f"  Vanilla total windows: {len(v_ws_diag)}")
                    print(f"  Sticky  total windows (alive): {len(s_ws_diag)}")
                    print(f"  Effective K: {ek}")
                    print(f"  Vanilla Top-{ek} [score, wid]:")
                    for w in v_sorted[:ek]:
                        print(f"    score={float(w[0]):.6f}  wid={int(w[1])}")

                    # Recall computation
                    v_top_ids = set(int(x[1]) for x in v_sorted[:ek])
                    s_alive_ids = set(int(x[1]) for x in s_ws_diag)
                    hits = v_top_ids & s_alive_ids
                    misses = v_top_ids - s_alive_ids
                    recall = len(hits) / ek if ek > 0 else 0.0
                    print(f"  Sticky alive window IDs ({len(s_alive_ids)}): "
                          f"{sorted(list(s_alive_ids))[:30]}{'...' if len(s_alive_ids) > 30 else ''}")
                    print(f"  Vanilla Top-{ek} HIT in alive set:    {sorted(hits)}")
                    print(f"  Vanilla Top-{ek} MISSED by alive set: {sorted(misses)}")
                    print(f"  Recall@{ek} = {len(hits)}/{ek} = {recall:.4f}")

                    if len(v_ws_diag) == len(s_ws_diag):
                        v_arr = np.array(v_ws_diag)
                        s_arr = np.array(s_ws_diag)
                        print(f"  Arrays IDENTICAL? {np.array_equal(v_arr, s_arr)}")
                    else:
                        print(f"  Arrays have DIFFERENT lengths — data is distinct")
                print(f"{'!'*70}\n")

            for l_str in v_g:
                if l_str not in s_g: continue
                layer_id = int(l_str)

                if layer_id not in gen_lh:
                    gen_lh[layer_id] = {}
                if str(layer_id) not in sample_record["layers"]:
                    sample_record["layers"][str(layer_id)] = {"heads": {}}

                head_recalls = get_layer_head_recall(
                    v_g[l_str], s_g[l_str], K_TOP, gen_diag=gen_diag
                )
                for h, r_score in head_recalls.items():
                    # Store ALL steps (for JSON)
                    if h not in gen_lh[layer_id]:
                        gen_lh[layer_id][h] = []
                    gen_lh[layer_id][h].append(r_score)
                    gen_recalls.append(r_score)

                    # Overwrite last-step tracker (final value = last OMEGA step)
                    if layer_id not in sample_last_recalls:
                        sample_last_recalls[layer_id] = {}
                    sample_last_recalls[layer_id][h] = r_score

                    if str(h) not in sample_record["layers"][str(layer_id)]["heads"]:
                        sample_record["layers"][str(layer_id)]["heads"][str(h)] = {
                            "prefill_steps": [],
                            "generation_steps": []
                        }

                    sample_record["layers"][str(layer_id)]["heads"][str(h)]["generation_steps"].append({
                        "step": step,
                        "recall_at_k": r_score
                    })

        # After all steps for this sample, collect last-step recalls
        for layer_id, heads in sample_last_recalls.items():
            if layer_id not in gen_last_lh:
                gen_last_lh[layer_id] = {}
            for h, r_score in heads.items():
                if h not in gen_last_lh[layer_id]:
                    gen_last_lh[layer_id][h] = []
                gen_last_lh[layer_id][h].append(r_score)
                gen_last_recalls.append(r_score)

        detailed_records.append(sample_record)

    # --- Per-sample ALL-STEP breakdown ---
    print(f"\n{'=' * 70}")
    print(f"ALL-STEP RECALL PER LAYER/HEAD (average across all OMEGA steps)")
    print(f"{'=' * 70}")
    for layer_id in sorted(gen_lh.keys()):
        for h in sorted(gen_lh[layer_id].keys()):
            scores = gen_lh[layer_id][h]
            print(f"  L{layer_id} H{h}: {len(scores)} steps → avg={np.mean(scores):.4f}, "
                  f"min={min(scores):.4f}, max={max(scores):.4f}")
    print(f"{'=' * 70}")

    # Save Detailed JSON (contains ALL steps)
    with open(DETAILED_OUTPUT_PATH, "w") as f:
        json.dump(detailed_records, f, indent=4)
    print(f"Saved detailed results to {DETAILED_OUTPUT_PATH}")

    # Calculate Aggregates — AVERAGE over ALL generation steps
    prefill_averages = {L: {H: np.mean(scores) for H, scores in heads.items()} for L, heads in prefill_lh.items()}
    overall_prefill = np.mean(prefill_jaccards) if prefill_jaccards else 0.0

    gen_avg_averages = {L: {H: np.mean(scores) for H, scores in heads.items()} for L, heads in gen_lh.items()}
    overall_gen_avg = np.mean(gen_recalls) if gen_recalls else 0.0

    print_summary_table("PREFILL", "Jaccard", prefill_averages, overall_prefill)

    # --- Generation Diagnostics ---
    if gen_diag['total_comparisons'] > 0 or gen_diag['empty_comparisons'] > 0:
        total = gen_diag['total_comparisons'] + gen_diag['empty_comparisons']
        print(f"\n{'=' * 80}")
        print(f"GENERATION DIAGNOSTICS (Window Pool Analysis)")
        print(f"{'=' * 80}")
        print(f"  Total comparison attempts:  {total}")
        print(f"  Valid comparisons:          {gen_diag['total_comparisons']}")
        print(f"  Empty comparisons (=0.0):   {gen_diag['empty_comparisons']} ({100*gen_diag['empty_comparisons']/max(1,total):.1f}%)")
        print(f"    -> Sticky had 0 windows:  {gen_diag['sticky_zero_windows']}")
        print(f"    -> Vanilla had 0 windows: {gen_diag['vanilla_zero_windows']}")
        if gen_diag['sticky_window_counts']:
            s_counts = gen_diag['sticky_window_counts']
            v_counts = gen_diag['vanilla_window_counts']
            k_vals = gen_diag['effective_k_values']
            print(f"  Sticky alive windows:       min={min(s_counts)}, max={max(s_counts)}, avg={np.mean(s_counts):.1f}")
            print(f"  Vanilla total windows:      min={min(v_counts)}, max={max(v_counts)}, avg={np.mean(v_counts):.1f}")
            print(f"  Effective K:                min={min(k_vals)}, max={max(k_vals)}, avg={np.mean(k_vals):.1f}")
            print(f"  Alive ratio (sticky/vanilla): {np.mean(s_counts)/max(1,np.mean(v_counts)):.2%}")
        print(f"{'=' * 80}")

    print_summary_table("GENERATION (avg all steps)", "Recall@K", gen_avg_averages, overall_gen_avg)

    # Also print last-step for reference
    gen_last_averages = {L: {H: np.mean(scores) for H, scores in heads.items()} for L, heads in gen_last_lh.items()}
    overall_gen_last = np.mean(gen_last_recalls) if gen_last_recalls else 0.0
    print_summary_table("GENERATION (last step)", "Recall@K", gen_last_averages, overall_gen_last)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Window Recall@K — Vanilla Oracle vs Sticky Teacher-Forcing"
    )
    parser.add_argument("--vanilla", type=str, default=VANILLA_PATH,
                        help="Path to vanilla oracle teacher-forcing .npz")
    parser.add_argument("--sticky", type=str, default=STICKY_PATH,
                        help="Path to sticky teacher-forcing .npz")
    parser.add_argument("--output", type=str, default=DETAILED_OUTPUT_PATH,
                        help="Path to write detailed Recall JSON")
    parser.add_argument("--k", type=float, default=K_TOP,
                        help=f"Top-K: float in (0,1] for ratio, int >1 for fixed count "
                             f"(default: {K_TOP} from sticky_config)")
    args, _ = parser.parse_known_args()

    # Override globals from CLI args
    VANILLA_PATH = args.vanilla
    STICKY_PATH = args.sticky
    DETAILED_OUTPUT_PATH = args.output
    # Interpret CLI --k: values > 1 are fixed int, values in (0,1] are ratios
    K_TOP = int(args.k) if args.k > 1.0 else args.k

    main()
