"""
calculate_topk_prefill_vs_gen.py — Prefill vs Last-Gen-Step Top-K Comparison

Takes an .npz file (from pure_vanilla or sticky_cumulative baseline runner)
and outputs a JSON reporting the cumulative top-K WINDOW indices per head per layer
at two checkpoints:
    1. End of Prefill  (step 0 cumulative)
    2. End of Generation (last step cumulative)

Also computes overlap (Jaccard) and rank correlation (Spearman) between the two
sets to quantify whether the model is biased toward prefill or generation attention.

Usage:
    python calculate_topk_prefill_vs_gen.py
"""

import json
import numpy as np
import os
import sys
from scipy.stats import spearmanr
from npz_io import load_results_npz
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import sticky_config as config
    OMEGA = getattr(config, "OMEGA", 1)
except ImportError:
    OMEGA = 1

INPUT_PATH = "sticky_baseline_results.npz"
OUTPUT_PATH = "topk_prefill_vs_gen.json"
K_TOP = 10  # Note: You can change this to 10 to match your Jaccard K if desired

def jaccard_overlap(set_a, set_b):
    """Jaccard Similarity between two index sets."""
    a, b = set(set_a), set(set_b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0

def rank_correlation(p_dict, g_dict, union_ids):
    """
    Spearman rank correlation between prefill and gen scores
    for the union of top-K window IDs.
    """
    if len(union_ids) < 3:
        return 0.0  # Spearman needs at least 3 data points

    p_vals = []
    g_vals = []
    
    for wid in sorted(list(union_ids)):
        # If a window was dropped or not in top-K pool, its active score is essentially 0.0
        p_vals.append(float(p_dict.get(wid, 0.0)))
        g_vals.append(float(g_dict.get(wid, 0.0)))

    # Handle case where all values are identical
    if np.std(p_vals) == 0 or np.std(g_vals) == 0:
        return 1.0 if np.array_equal(p_vals, g_vals) else 0.0

    corr, _ = spearmanr(p_vals, g_vals)
    return float(corr) if not np.isnan(corr) else 0.0

def extract_topk_windows(window_list, k):
    """
    Given a list of [score, window_id] pairs, return the top-K window_ids
    and an id->score dictionary for all candidate windows.
    """
    if not window_list:
        return [], {}
        
    score_dict = {int(x[1]): float(x[0]) for x in window_list}
    sorted_ws = sorted(window_list, key=lambda x: float(x[0]), reverse=True)
    
    effective_k = min(k, len(sorted_ws))
    topk_ids = [int(x[1]) for x in sorted_ws[:effective_k]]
    
    return topk_ids, score_dict


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Missing NPZ result file at {INPUT_PATH}. Please run the baseline first.")
        sys.exit(1)

    if os.path.exists(OUTPUT_PATH):
        print(f"Removing existing {OUTPUT_PATH} to prevent appending bugs...")
        os.remove(OUTPUT_PATH)

    print(f"Loading results from: {INPUT_PATH} (Skipping massive attention matrices!)")
    data = load_results_npz(INPUT_PATH, skip_attention=True)

    K = K_TOP
    print(f"Extracting Top-{K} WINDOWS at Prefill End vs Generation End...")

    all_samples = []
    aggregate = {}

    for sample_idx, sample in enumerate(data):
        meta = sample["metadata"]
        
        # Use Window Scores directly instead of raw token attention arrays!
        prefill_ws = sample.get("prefill_window_scores", {})
        gen_ws_list = sample.get("generation_window_scores", [])

        num_gen_steps = len(gen_ws_list)
        if num_gen_steps == 0:
            print(f"  Sample {sample_idx}: No generation steps, skipping.")
            continue

        last_step_ws = gen_ws_list[-1]  # Last generation step (cumulative)

        target_steps = list(range(0, num_gen_steps, OMEGA))
        if (num_gen_steps - 1) not in target_steps:
             target_steps.append(num_gen_steps - 1)

        sample_result = {
            "sample_index": sample_idx,
            "sha256": meta.get("sha256", ""),
            "prefill_len": meta.get("token_count_input", 0),
            "num_gen_steps": num_gen_steps,
            "k": K,
            "layers": {}
        }

        for layer_str in sorted(prefill_ws.keys(), key=lambda x: int(x)):
            if layer_str not in last_step_ws:
                continue

            layer_prefill = prefill_ws[layer_str]
            layer_gen = last_step_ws[layer_str]

            layer_result = {}

            for head_str in sorted(layer_prefill.keys(), key=lambda x: int(x)):
                if head_str not in layer_gen:
                    continue

                p_list = layer_prefill[head_str]
                
                # Compute Top-K WINDOWS for prefill
                prefill_topk_idx, p_dict = extract_topk_windows(p_list, K)
                
                # Build timeline across OMEGA flush points
                timeline = []
                for step in target_steps:
                    step_ws = gen_ws_list[step]
                    if layer_str in step_ws and head_str in step_ws[layer_str]:
                        g_list_step = step_ws[layer_str][head_str]
                    else:
                        g_list_step = []
                        
                    gen_topk_step, g_dict_step = extract_topk_windows(g_list_step, K)
                    j_step = jaccard_overlap(prefill_topk_idx, gen_topk_step)
                    union_step = set(prefill_topk_idx) | set(gen_topk_step)
                    sp_step = rank_correlation(p_dict, g_dict_step, union_step)
                    
                    shared_step = set(prefill_topk_idx) & set(gen_topk_step)
                    gen_only_step = set(gen_topk_step) - set(prefill_topk_idx)
                    
                    timeline.append({
                        "step": step,
                        "jaccard_overlap": round(j_step, 4),
                        "spearman_rank_corr": round(sp_step, 4),
                        "prefill_origin": len(shared_step),
                        "generation_origin": len(gen_only_step)
                    })

                # Compute final state logic for summary table
                g_list = layer_gen[head_str]
                gen_topk_idx, g_dict = extract_topk_windows(g_list, K)

                # Overlap metrics
                jaccard = jaccard_overlap(prefill_topk_idx, gen_topk_idx)

                gen_only = sorted(set(gen_topk_idx) - set(prefill_topk_idx))
                shared = sorted(set(prefill_topk_idx) & set(gen_topk_idx))

                union = set(prefill_topk_idx) | set(gen_topk_idx)
                spearman = rank_correlation(p_dict, g_dict, union)

                head_result = {
                    "timeline": timeline,
                    "prefill_topk_indices": prefill_topk_idx,
                    "gen_topk_indices": gen_topk_idx,
                    "shared_indices": shared,
                    "gen_only_indices": gen_only,
                    "jaccard_overlap": round(jaccard, 4),
                    "spearman_rank_corr": round(spearman, 4),
                    "prefill_origin": len(shared),
                    "generation_origin": len(gen_only),
                }

                layer_result[head_str] = head_result

                # Accumulate for summary
                if layer_str not in aggregate:
                    aggregate[layer_str] = {}
                if head_str not in aggregate[layer_str]:
                    aggregate[layer_str][head_str] = {
                        "jaccard": [], "spearman": [],
                        "generation_origin_count": [], "prefill_origin_count": [],
                        "total_windows": []
                    }
                agg = aggregate[layer_str][head_str]
                agg["jaccard"].append(jaccard)
                agg["spearman"].append(spearman)
                agg["generation_origin_count"].append(len(gen_only))
                agg["prefill_origin_count"].append(len(shared))
                agg["total_windows"].append(len(p_list))

            sample_result["layers"][layer_str] = layer_result

        all_samples.append(sample_result)

    # Build aggregate summary
    summary_per_head = {}
    summary_per_layer = {}

    for layer_str in sorted(aggregate.keys(), key=int):
        layer_jaccards = []
        layer_spearmans = []
        layer_generation_origin = []
        layer_prefill_origin = []
        layer_total_windows = []

        summary_per_head[layer_str] = {}
        for head_str in sorted(aggregate[layer_str].keys(), key=int):
            agg = aggregate[layer_str][head_str]
            j_mean = float(np.mean(agg["jaccard"]))
            s_mean = float(np.mean(agg["spearman"]))
            go_mean = float(np.mean(agg["generation_origin_count"]))
            po_mean = float(np.mean(agg["prefill_origin_count"]))
            tw_mean = float(np.mean(agg["total_windows"]))

            summary_per_head[layer_str][head_str] = {
                "jaccard_mean": round(j_mean, 4),
                "spearman_mean": round(s_mean, 4),
                "avg_generation_origin": round(go_mean, 2),
                "avg_prefill_origin": round(po_mean, 2),
                "avg_total_windows": round(tw_mean, 1),
                "data_points": len(agg["jaccard"])
            }

            layer_jaccards.extend(agg["jaccard"])
            layer_spearmans.extend(agg["spearman"])
            layer_generation_origin.extend(agg["generation_origin_count"])
            layer_prefill_origin.extend(agg["prefill_origin_count"])
            layer_total_windows.extend(agg["total_windows"])

        summary_per_layer[layer_str] = {
            "jaccard_mean": round(float(np.mean(layer_jaccards)), 4) if layer_jaccards else 0.0,
            "spearman_mean": round(float(np.mean(layer_spearmans)), 4) if layer_spearmans else 0.0,
            "avg_generation_origin": round(float(np.mean(layer_generation_origin)), 2) if layer_generation_origin else 0.0,
            "avg_prefill_origin": round(float(np.mean(layer_prefill_origin)), 2) if layer_prefill_origin else 0.0,
            "avg_total_windows": round(float(np.mean(layer_total_windows)), 1) if layer_total_windows else 0.0,
        }

    # Print summary table
    print("\n" + "=" * 128)
    print(f"TOP-{K} WINDOWS: PREFILL vs GENERATION — PER-LAYER SUMMARY")
    print("=" * 144)
    print(f"| {'Layer':<6} | {'Pool Size':<10} | {'Jaccard':<10} | {'Spearman':<10} | {'Prefill-Origin':<15} | {'Generation-Origin':<18} | {'Bias':<20} |")
    print("-" * 144)

    for layer_str in sorted(summary_per_layer.keys(), key=int):
        s = summary_per_layer[layer_str]
        j = s["jaccard_mean"]
        sp = s["spearman_mean"]
        pre_orig = s["avg_prefill_origin"]
        gen_orig = s["avg_generation_origin"]
        tw = s["avg_total_windows"]

        if pre_orig > gen_orig * 1.5:
            bias = "⇐ Prefill-biased"
        elif gen_orig > pre_orig * 1.5:
            bias = "⇒ Gen-biased"
        else:
            bias = "≈ Balanced"

        print(f"| {layer_str:<6} | {tw:<10.1f} | {j:<10.4f} | {sp:<10.4f} | {pre_orig:<15.1f} | {gen_orig:<18.1f} | {bias:<20} |")

    print("=" * 144)

    print(f"\n{'=' * 144}")
    print(f"TOP-{K} WINDOWS: PREFILL vs GENERATION — PER-HEAD DETAIL")
    print(f"{'=' * 144}")
    print(f"| {'Layer':<6} | {'Head':<6} | {'Pool Size':<10} | {'Jaccard':<10} | {'Spearman':<10} | {'Prefill-Origin':<15} | {'Generation-Origin':<18} |")
    print("-" * 144)

    for layer_str in sorted(summary_per_head.keys(), key=int):
        for head_str in sorted(summary_per_head[layer_str].keys(), key=int):
            h = summary_per_head[layer_str][head_str]
            print(f"| {layer_str:<6} | {head_str:<6} | {h['avg_total_windows']:<10.1f} | {h['jaccard_mean']:<10.4f} | {h['spearman_mean']:<10.4f} | {h['avg_prefill_origin']:<15.1f} | {h['avg_generation_origin']:<18.1f} |")

    print("=" * 128)

    # Write output
    output = {
        "config": {
            "k": K,
            "input_file": INPUT_PATH,
            "num_samples": len(all_samples),
        },
        "summary_per_layer": summary_per_layer,
        "summary_per_head": summary_per_head,
        "per_sample_detail": all_samples,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
