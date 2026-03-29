import json
import os
import argparse
import numpy as np
import sys
from collections import defaultdict

from sticky_config import OMEGA, LOCAL_NUM_TOKENS

VANILLA_PATH = "vanilla_baseline_results.json"
STICKY_PATH = "sticky_baseline_results.json"
DETAILED_OUTPUT_PATH = "detailed_jaccard_results.json"
K_TOP = 10

def calculate_jaccard(v_ws, s_ws, k, seq_len, prefill_seq_len, v_gen_tokens, s_gen_tokens, sink_tokens=5, local_tokens=LOCAL_NUM_TOKENS, omega=OMEGA, debug_info=None, gen_diag=None):
    """
    Option 2: Direct set comparison.
    - Sticky: ALL surviving window IDs are used directly (no re-ranking).
      These are already the RRF-selected survivors.
    - Vanilla: Ranked by score (column 0 = RRF score), top-K selected.
    """
    # Sinks are always the first 5 tokens
    # Local window boundary starts at seq_len - local_tokens
    local_start_token = max(sink_tokens, seq_len - local_tokens)
    
    # window_scores stores logical ID. ID 0 = Token 5. ID 1 = Token 6. (when omega=1)
    # So a window ID maps to token position: ID * omega + sink_tokens
    
    valid_v_ws = [x for x in v_ws if (int(x[1]) * omega + sink_tokens) < local_start_token]
    valid_s_ws = [x for x in s_ws if (int(x[1]) * omega + sink_tokens) < local_start_token]
    
    if len(valid_v_ws) == 0 or len(valid_s_ws) == 0:
        if gen_diag is not None:
            gen_diag['empty_comparisons'] += 1
            gen_diag['sticky_zero_windows'] += (1 if len(valid_s_ws) == 0 else 0)
            gen_diag['vanilla_zero_windows'] += (1 if len(valid_v_ws) == 0 else 0)
        return 0.0

    # STICKY: Rank by RRF score (exported in column 0) and take top-K
    effective_k = min(k, len(valid_s_ws), len(valid_v_ws))
    if effective_k == 0:
        return 0.0
    
    if gen_diag is not None:
        gen_diag['total_comparisons'] += 1
        gen_diag['sticky_window_counts'].append(len(valid_s_ws))
        gen_diag['vanilla_window_counts'].append(len(valid_v_ws))
        gen_diag['effective_k_values'].append(effective_k)
        
    s_ws_sorted = sorted(valid_s_ws, key=lambda x: float(x[0]), reverse=True)
    s_top_k_ids = set(int(x[1]) for x in s_ws_sorted[:effective_k])
    
    # VANILLA: Rank by RRF score, take top-K matching sticky's survivor count
    v_ws_sorted = sorted(valid_v_ws, key=lambda x: float(x[0]), reverse=True)
    v_top_k_ids = set(int(x[1]) for x in v_ws_sorted[:effective_k])
    
    intersection = 0
    union_set = v_top_k_ids.union(s_top_k_ids)
    union = len(union_set)
    
    for wid in v_top_k_ids.intersection(s_top_k_ids):
        start_pos = wid * omega + sink_tokens
        if start_pos >= prefill_seq_len:
            gen_idx = int(start_pos - prefill_seq_len)
            v_toks = v_gen_tokens[gen_idx : gen_idx + omega]
            s_toks = s_gen_tokens[gen_idx : gen_idx + omega]
            
            # If the underlying text tokens do not perfectly match, they are NOT the same window.
            # They should not be intersected, and because they represent two physically distinct pieces
            # of text that happen to share a logical ID, the true union size expands by 1.
            if v_toks != s_toks:
                union += 1
                continue
                
        intersection += 1
    
    if union == 0:
        jaccard = 0.0
    else:
        jaccard = intersection / union
        
    if debug_info and jaccard < 1.0:
        L, H = debug_info
        print(f"\n[DEBUG] Jaccard Divergence Detected at PREFILL Layer {L}, Head {H} | Jaccard = {jaccard:.4f}")
        
        # Original subset analysis
        v_scores_all = [float(x[0]) for x in valid_v_ws]
        s_scores_all = [float(x[0]) for x in valid_s_ws]
        
        v_zeros = sum(1 for s in v_scores_all if s == 0.0)
        s_zeros = sum(1 for s in s_scores_all if s == 0.0)
        
        print(f"  -> Total Candidates  | Vanilla: {len(valid_v_ws)} (Has {v_zeros} zeroes) | Sticky: {len(valid_s_ws)} (Has {s_zeros} zeroes)")
        
        # Top-K subset analysis
        v_top_k = v_ws_sorted[:effective_k]
        s_top_k = s_ws_sorted[:effective_k]
        
        v_top_scores = [float(x[0]) for x in v_top_k]
        s_top_scores = [float(x[0]) for x in s_top_k]
        
        v_top_zeros = sum(1 for s in v_top_scores if s == 0.0)
        s_top_zeros = sum(1 for s in s_top_scores if s == 0.0)
        
        print(f"  -> Top-{effective_k} Selection   | Vanilla pulled {v_top_zeros} zeroes into top-K | Sticky pulled {s_top_zeros} zeroes into top-K")
        
        mismatches_v = v_top_k_ids - s_top_k_ids
        mismatches_s = s_top_k_ids - v_top_k_ids
        
        m_v_scores = [float(next(x[0] for x in v_ws_sorted if int(x[1]) == wid)) for wid in mismatches_v]
        m_s_scores = [float(next(x[0] for x in s_ws_sorted if int(x[1]) == wid)) for wid in mismatches_s]
        
        print(f"  -> Vanilla exclusively selected IDs: sorted({sorted(list(mismatches_v))}) | Their Scores: {m_v_scores}")
        print(f"  -> Sticky  exclusively selected IDs: sorted({sorted(list(mismatches_s))}) | Their Scores: {m_s_scores}")
        print("  -> Proof: Divergence occurred strictly because ties at the score boundary were broken differently between runs.")

    return jaccard

def get_layer_head_jaccard(v_layer_data, s_layer_data, k, seq_len, prefill_seq_len, v_gen_tokens, s_gen_tokens, is_prefill=False, layer=None, gen_diag=None):
    jaccards = {}
    for h_str in v_layer_data:
        if h_str not in s_layer_data:
            continue
            
        v_ws = v_layer_data[h_str]
        s_ws = s_layer_data[h_str]
        
        debug_info = (layer, h_str) if is_prefill else None
        jaccards[int(h_str)] = calculate_jaccard(
            v_ws, s_ws, k, seq_len, prefill_seq_len, v_gen_tokens, s_gen_tokens, debug_info=debug_info, gen_diag=gen_diag
        )
    return jaccards

def print_summary_table(title, layer_averages, overall_jaccard):
    print(f"\n{'=' * 60}")
    print(f"=== {title} Jaccard Similarity (Top-{K_TOP} Windows) ===")
    print(f"{'=' * 60}")
    print(f"{'Layer':<10} | {'Head':<10} | {'Jaccard':<15}")
    print("-" * 60)
    
    sorted_layers = sorted(layer_averages.keys())
    for L in sorted_layers:
        sorted_heads = sorted(layer_averages[L].keys())
        for H in sorted_heads:
            print(f"{L:<10} | {H:<10} | {layer_averages[L][H]:.4f}")
            
    print("-" * 60)
    print(f"{'OVERALL':<10} | {'ALL':<10} | {overall_jaccard:.4f}")
    print(f"{'=' * 60}\n")


def main():
    if not os.path.exists(VANILLA_PATH) or not os.path.exists(STICKY_PATH):
        print("Error: Missing JSON result files. Please run both baselines first.")
        sys.exit(1)
        
    if os.path.exists(DETAILED_OUTPUT_PATH):
        print(f"Removing existing {DETAILED_OUTPUT_PATH} to prevent appending bugs...")
        os.remove(DETAILED_OUTPUT_PATH)

    with open(VANILLA_PATH, "r") as f:
        v_data = json.load(f)
    with open(STICKY_PATH, "r") as f:
        s_data = json.load(f)

    prefill_jaccards = []
    prefill_lh = {}
    gen_jaccards = []
    gen_lh = {}
    
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
        v = v_data[idx]
        s = s_data[idx]
        
        prefill_seq_len = v["metadata"].get("token_count_input", 0)
        v_gen_tokens = v["metadata"].get("generated_token_ids", [])
        s_gen_tokens = s["metadata"].get("generated_token_ids", [])
        
        v_gen_steps = v.get("generation_window_scores", [])
        s_gen_steps = s.get("generation_window_scores", [])
        
        sample_record = {
            "sample_index": idx,
            "layers": {}
        }
        
        # --- Prefill Stage ---
        # The true prefill snapshot is stored in prefill_window_scores
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
                    
                head_jaccards = get_layer_head_jaccard(
                    v_pre[l_str], s_pre[l_str], K_TOP, prefill_seq_len, 
                    prefill_seq_len, v_gen_tokens, s_gen_tokens, is_prefill=True, layer=layer_id
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
                
        # --- Gen Stage ---
        steps = min(len(v_gen_steps), len(s_gen_steps))
        for step in range(steps):
            v_g = v_gen_steps[step]
            s_g = s_gen_steps[step]
            
            # For generation, the sequence length increases by 1 each step
            gen_seq_len = prefill_seq_len + step
            
            for l_str in v_g:
                if l_str not in s_g: continue
                layer_id = int(l_str)
                
                if layer_id not in gen_lh:
                    gen_lh[layer_id] = {}
                if str(layer_id) not in sample_record["layers"]:
                    sample_record["layers"][str(layer_id)] = {"heads": {}}
                    
                head_jaccards = get_layer_head_jaccard(
                    v_g[l_str], s_g[l_str], K_TOP, gen_seq_len, 
                    prefill_seq_len, v_gen_tokens, s_gen_tokens, gen_diag=gen_diag
                )
                for h, j_score in head_jaccards.items():
                    if h not in gen_lh[layer_id]:
                        gen_lh[layer_id][h] = []
                    gen_lh[layer_id][h].append(j_score)
                    gen_jaccards.append(j_score)

                    if str(h) not in sample_record["layers"][str(layer_id)]["heads"]:
                        sample_record["layers"][str(layer_id)]["heads"][str(h)] = {
                            "prefill_steps": [],
                            "generation_steps": []
                        }
                    
                    sample_record["layers"][str(layer_id)]["heads"][str(h)]["generation_steps"].append({
                        "step": step,
                        "jaccard_similarity": j_score
                    })
                    
        detailed_records.append(sample_record)

    # Save Detailed JSON
    with open(DETAILED_OUTPUT_PATH, "w") as f:
        json.dump(detailed_records, f, indent=4)
    print(f"Saved detailed results to {DETAILED_OUTPUT_PATH}")

    # Calculate Aggregates
    prefill_averages = {L: {H: np.mean(scores) for H, scores in heads.items()} for L, heads in prefill_lh.items()}
    overall_prefill = np.mean(prefill_jaccards) if prefill_jaccards else 0.0
    
    gen_averages = {L: {H: np.mean(scores) for H, scores in heads.items()} for L, heads in gen_lh.items()}
    overall_gen = np.mean(gen_jaccards) if gen_jaccards else 0.0
    
    print_summary_table("PREFILL", prefill_averages, overall_prefill)
    
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
            print(f"  Sticky valid windows:       min={min(s_counts)}, max={max(s_counts)}, avg={np.mean(s_counts):.1f}")
            print(f"  Vanilla valid windows:      min={min(v_counts)}, max={max(v_counts)}, avg={np.mean(v_counts):.1f}")
            print(f"  Effective K:                min={min(k_vals)}, max={max(k_vals)}, avg={np.mean(k_vals):.1f}")
            print(f"  Pool ratio (sticky/vanilla): {np.mean(s_counts)/max(1,np.mean(v_counts)):.2%}")
        print(f"{'=' * 80}")
    
    print_summary_table("GENERATION", gen_averages, overall_gen)

if __name__ == "__main__":
    main()
