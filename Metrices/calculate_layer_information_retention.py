import json
import numpy as np
import os
import sys
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import argparse
from npz_io import load_results_npz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_attention_mass_retention(vanilla_attn, sticky_attn):
    """
    Calculates Attention Mass Retention (AMR) on all tokens.
    What percentage of the vanilla attention sum is retained by the active tokens in sticky?
    """
    v = np.array(vanilla_attn)
    s = np.array(sticky_attn)
    
    # Active tokens in sticky (where the ledger score > 0, assuming -1 or 0 for inactive)
    # Sticky code sets inactive tokens to 0 in full_row when it reconstructs the vector
    # We ensure s is numeric to prevent dictionary type errors
    active_mask = (s > 0.0)
    
    # Sum the vanilla probability mass for the tokens that sticky kept alive
    retained_mass = np.sum(v[active_mask])
    
    # Normalizing just in case vanilla doesn't exactly sum to 1.0 due to fp precision
    total_mass = np.sum(v)
    if total_mass == 0:
        return 0.0
    return retained_mass / total_mass

def calculate_cosine_similarity(vanilla_attn, sticky_attn):
    """
    Calculates Cosine Similarity between the vanilla and sticky attention vectors.
    """
    v = np.array(vanilla_attn)
    s = np.array(sticky_attn)
        
    if np.sum(v) == 0 or np.sum(s) == 0 or len(v) == 0:
        return 0.0
        
    # cosine() function from scipy returns distance, so 1 - dist = similarity
    return 1 - cosine(v, s)

def calculate_kl_divergence(vanilla_attn, sticky_attn):
    """
    Calculates KL Divergence: KL(Vanilla || Sticky) for all tokens.
    Since Sticky has zeros (evicted tokens), we add epsilon to avoid log(0) or div/0.
    Returns inverted KL, scaled between 0 and 1, where 1 is identical.
    """
    v = np.array(vanilla_attn)
    s = np.array(sticky_attn)
        
    if len(v) == 0:
        return 1.0 # Perfect similarity on an empty set
    
    # Add small epsilon to avoid division by zero and log(0)
    epsilon = 1e-10
    v_safe = v + epsilon
    s_safe = s + epsilon
    
    # Normalize so they are true probability distributions summing to 1
    v_prob = v_safe / np.sum(v_safe)
    s_prob = s_safe / np.sum(s_safe)
    
    # Scipy entropy calculates KL divergence if two arguments are passed
    kl_div = entropy(v_prob, s_prob)
    
    # Convert to a 0-1 metric where 1 is perfect retention (0 divergence)
    # y = 1 / (1 + x) transforms [0, infinity) -> (1, 0]
    return 1 / (1 + kl_div)

def calculate_missed_mass_drift(vanilla_attn, sticky_attn):
    """
    Calculates Attention Drift via Missed Mass for all tokens.
    This looks at the tokens the Sticky cache explicitly dropped/evicted
    (where Sticky attention == 0) and sums up the attention the Vanilla 
    model gave to those EXACT same tokens. 
    It measures what portion of the 'attention pie' slipped through the cracks.
    """
    v = np.array(vanilla_attn)
    s = np.array(sticky_attn)
    
    # The sticky baseline natively exports the array padded with zeros at exactly 
    # the correct globally evicted positions. However, a 1-token length mismatch 
    # can occur at the very tail end between vanilla and sticky tracking.
    # We cleanly truncate the trailing mismatch to preserve global positional alignment!
    min_len = min(len(v), len(s))
    v = v[:min_len]
    s = s[:min_len]
        
    # Identify tokens explicitly evicted by sticky (or missing entirely)
    evicted_mask = (s <= 0.0)
    
    # Sum the vanilla attention given to these evicted tokens
    missed_mass = np.sum(v[evicted_mask])
    
    total_mass = np.sum(v)
    if total_mass == 0:
        return 0.0
        
    return missed_mass / total_mass

def calculate_sparsity(sticky_attn):
    """
    Calculates sparsity: what percentage of total tokens were evicted (set to 0)?
    """
    s = np.array(sticky_attn)
        
    if len(s) == 0:
        return 0.0
        
    active_count = np.sum(s > 0)
    return 1.0 - (active_count / len(s))

def calculate_global_lir(vanilla_attn, sticky_attn):
    """
    Calculates Global LIR (Regretted Eviction Score).
    
    For every token that sticky evicted (attention = 0), this measures how much
    vanilla attention that token STILL receives. If vanilla assigns high attention
    to evicted tokens, the eviction was 'regretted' — important information was lost.
    
    Score = 1 - (vanilla_attention_to_evicted / total_vanilla_attention)
    Range: [0, 1] where 1.0 = no regretted evictions (perfect), 0.0 = everything important was evicted.
    
    Unlike per-region metrics (AMR, cosine, etc.), this operates on the FULL sequence
    without trimming sinks or local tokens, capturing the global eviction impact.
    """
    v = np.array(vanilla_attn, dtype=float)
    s = np.array(sticky_attn, dtype=float)
    
    # The sticky baseline natively exports the array padded with zeros at exactly 
    # the correct globally evicted positions. We cleanly truncate any trailing length 
    # mismatches to preserve global positional alignment!
    min_len = min(len(s), len(v))
    s = s[:min_len]
    v = v[:min_len]
    
    total_vanilla = np.sum(v)
    if total_vanilla == 0:
        return 1.0  # No attention to lose
    
    # Evicted tokens: where sticky has zero attention
    evicted_mask = (s <= 0.0)
    regretted_mass = np.sum(v[evicted_mask])
    
    return 1.0 - (regretted_mass / total_vanilla)

def main():
    parser = argparse.ArgumentParser(description="Calculate Layer-wise Information Retention (LIR)")
    parser.add_argument("--vanilla", type=str, default="pure_vanilla_baseline_results.npz")
    parser.add_argument("--sticky", type=str, default="sticky_baseline_results.npz")
    parser.add_argument("--output", type=str, default="lir_comparison.json")
    
    # In Jupyter notebooks/Kaggle, system adds arguments like '-f' which break standard parse_args()
    args, unknown = parser.parse_known_args()

    if os.path.exists(args.output):
        print(f"Removing existing {args.output} to prevent appending bugs...")
        os.remove(args.output)

    print(f"Loading Vanilla results from: {args.vanilla}")
    vanilla_data = load_results_npz(args.vanilla)

    print(f"Loading Sticky results from: {args.sticky}")
    sticky_data = load_results_npz(args.sticky)

    if len(vanilla_data) != len(sticky_data):
        print(f"Warning: Number of samples differ! Vanilla: {len(vanilla_data)}, Sticky: {len(sticky_data)}")
        # Only process up to the shortest list
        min_len = min(len(vanilla_data), len(sticky_data))
        vanilla_data = vanilla_data[:min_len]
        sticky_data = sticky_data[:min_len]

    # Metrics structured by Layer
    metrics_by_layer_gen = {}
    metrics_by_layer_prefill = {}
    
    # Per-head detail: {phase: {layer: {head: {metric: [values]}}}} 
    head_detail_prefill = {}
    head_detail_gen = {}
    
    total_samples = len(vanilla_data)
    
    print("Processing data across samples...")
    for sample_idx in range(total_samples):
        v_sample = vanilla_data[sample_idx]
        s_sample = sticky_data[sample_idx]
        
        # --- PREFILL ---
        v_pre = v_sample.get("prefill_attention", {})
        s_pre = s_sample.get("prefill_attention", {})
        
        for layer_idx_str, v_layer_data in v_pre.items():
            if layer_idx_str not in s_pre: continue
            s_layer_data = s_pre[layer_idx_str]
            
            if layer_idx_str not in metrics_by_layer_prefill:
                metrics_by_layer_prefill[layer_idx_str] = {
                    "amr": [], "cosine": [], "kl_inv": [], "sparsity": [], "missed_mass": [], "global_lir": []
                }
            if layer_idx_str not in head_detail_prefill:
                head_detail_prefill[layer_idx_str] = {}
                
            for head_idx_str, v_head_data in v_layer_data.items():
                if head_idx_str not in s_layer_data: continue
                
                s_head_data = s_layer_data[head_idx_str]
                
                # Convert the raw lists to flat float arrays
                # We use np.array(..., dtype=float).flatten() to force it, catching any objects
                try:
                    v_flat = np.array(v_head_data, dtype=float).flatten()
                    s_flat = np.array(s_head_data, dtype=float).flatten()
                except (ValueError, TypeError):
                    # If this happens, the structure is too complex (e.g. jagged lists). 
                    # We skip this head to avoid crashing the metric calculation.
                    continue
                
                # Prevent NaN
                v_flat = np.nan_to_num(v_flat)
                s_flat = np.nan_to_num(s_flat)
                
                min_len = min(len(v_flat), len(s_flat))
                
                if min_len > 0:
                    v_arr = v_flat[:min_len]
                    s_arr = s_flat[:min_len]
                    
                    amr = calculate_attention_mass_retention(v_arr, s_arr)
                    cos_sim = calculate_cosine_similarity(v_arr, s_arr)
                    kl_inv = calculate_kl_divergence(v_arr, s_arr)
                    sparsity = calculate_sparsity(s_arr)
                    missed_mass = calculate_missed_mass_drift(v_arr, s_arr)
                    glir = calculate_global_lir(v_flat, s_flat)
                    
                    metrics_by_layer_prefill[layer_idx_str]["amr"].append(amr)
                    metrics_by_layer_prefill[layer_idx_str]["cosine"].append(cos_sim)
                    metrics_by_layer_prefill[layer_idx_str]["kl_inv"].append(kl_inv)
                    metrics_by_layer_prefill[layer_idx_str]["sparsity"].append(sparsity)
                    metrics_by_layer_prefill[layer_idx_str]["missed_mass"].append(missed_mass)
                    metrics_by_layer_prefill[layer_idx_str]["global_lir"].append(glir)
                    
                    # Per-head tracking
                    if head_idx_str not in head_detail_prefill[layer_idx_str]:
                        head_detail_prefill[layer_idx_str][head_idx_str] = {
                            "amr": [], "cosine": [], "kl_inv": [], "sparsity": [], "missed_mass": [], "global_lir": []
                        }
                    hd = head_detail_prefill[layer_idx_str][head_idx_str]
                    hd["amr"].append(amr)
                    hd["cosine"].append(cos_sim)
                    hd["kl_inv"].append(kl_inv)
                    hd["sparsity"].append(sparsity)
                    hd["missed_mass"].append(missed_mass)
                    hd["global_lir"].append(glir)

        v_gen = v_sample.get("generation_attention", [])
        s_gen = s_sample.get("generation_attention", [])
        
        # Ensure we only compare matching generation steps
        num_steps = min(len(v_gen), len(s_gen))
        
        for step_idx in range(num_steps):
            v_step = v_gen[step_idx]
            s_step = s_gen[step_idx]
            
            # Iterate through layers in this step
            for layer_idx_str in v_step.keys():
                if layer_idx_str not in s_step:
                    continue
                    
                v_layer = v_step[layer_idx_str]
                s_layer = s_step[layer_idx_str]
                
                if layer_idx_str not in metrics_by_layer_gen:
                    metrics_by_layer_gen[layer_idx_str] = {
                        "amr": [], "cosine": [], "kl_inv": [], "sparsity": [], "missed_mass": [], "global_lir": []
                    }
                if layer_idx_str not in head_detail_gen:
                    head_detail_gen[layer_idx_str] = {}
                
                # Iterate through heads in this layer
                for head_idx_str in v_layer.keys():
                    if head_idx_str not in s_layer:
                        continue
                        
                    try:
                        v_head = np.array(v_layer[head_idx_str], dtype=float).flatten()
                        s_head = np.array(s_layer[head_idx_str], dtype=float).flatten()
                    except (ValueError, TypeError):
                        continue
                    
                    v_head = np.nan_to_num(v_head)
                    s_head = np.nan_to_num(s_head)
                    
                    # Ensure same length (due to generation step growing)
                    min_len = min(len(v_head), len(s_head))
                    if min_len == 0:
                        continue
                        
                    v_arr = v_head[:min_len]
                    s_arr = s_head[:min_len]
                    
                    # Calculate metrics
                    amr = calculate_attention_mass_retention(v_arr, s_arr)
                    cos_sim = calculate_cosine_similarity(v_arr, s_arr)
                    kl_inv = calculate_kl_divergence(v_arr, s_arr)
                    sparsity = calculate_sparsity(s_arr)
                    
                    # Missed Mass and Global LIR use FULL arrays, not truncated,
                    # because truncation removes the evicted tokens from view
                    missed_mass = calculate_missed_mass_drift(v_head, s_head)
                    glir = calculate_global_lir(v_head, s_head)
                    
                    if layer_idx_str not in metrics_by_layer_gen:
                        metrics_by_layer_gen[layer_idx_str] = {
                            "amr": [], "cosine": [], "kl_inv": [], "sparsity": [], "missed_mass": [], "global_lir": []
                        }
                    
                    metrics_by_layer_gen[layer_idx_str]["amr"].append(amr)
                    metrics_by_layer_gen[layer_idx_str]["cosine"].append(cos_sim)
                    metrics_by_layer_gen[layer_idx_str]["kl_inv"].append(kl_inv)
                    metrics_by_layer_gen[layer_idx_str]["sparsity"].append(sparsity)
                    metrics_by_layer_gen[layer_idx_str]["missed_mass"].append(missed_mass)
                    metrics_by_layer_gen[layer_idx_str]["global_lir"].append(glir)
                    
                    # Per-head tracking
                    if head_idx_str not in head_detail_gen[layer_idx_str]:
                        head_detail_gen[layer_idx_str][head_idx_str] = {
                            "amr": [], "cosine": [], "kl_inv": [], "sparsity": [], "missed_mass": [], "global_lir": []
                        }
                    hd = head_detail_gen[layer_idx_str][head_idx_str]
                    hd["amr"].append(amr)
                    hd["cosine"].append(cos_sim)
                    hd["kl_inv"].append(kl_inv)
                    hd["sparsity"].append(sparsity)
                    hd["missed_mass"].append(missed_mass)
                    hd["global_lir"].append(glir)


    def print_and_aggregate_metrics(metrics_dict, stage_name):
        print("\n" + "="*110)
        print(f"{stage_name.upper()} LAYER-WISE INFORMATION RETENTION (LIR) RESULTS")
        print("="*110)
        print(f"| {'Layer':<5} | {'Mass Retained':<15} | {'Missed Mass':<12} | {'Cosine Sim':<12} | {'Inv KL Div':<12} | {'Global LIR':<12} | {'Sparsity':<10} |")
        print("-" * 110)
        
        stage_results = {}
        sorted_layers = sorted(metrics_dict.keys(), key=lambda x: int(x))
        
        for layer in sorted_layers:
            data = metrics_dict[layer]
            
            avg_amr = np.mean(data["amr"]) if data["amr"] else 0.0
            avg_missed = np.mean(data["missed_mass"]) if data["missed_mass"] else 0.0
            avg_cos = np.mean(data["cosine"]) if data["cosine"] else 0.0
            avg_kl = np.mean(data["kl_inv"]) if data["kl_inv"] else 0.0
            avg_spars = np.mean(data["sparsity"]) if data["sparsity"] else 0.0
            avg_glir = np.mean(data["global_lir"]) if data.get("global_lir") else 0.0
            
            print(f"| {layer:<5} | {avg_amr:>14.2%} | {avg_missed:>11.2%} | {avg_cos:>11.4f} | {avg_kl:>11.4f} | {avg_glir:>11.4f} | {avg_spars:>9.2%} |")
            
            stage_results[layer] = {
                "attention_mass_retained_mean": float(avg_amr),
                "missed_mass_drift_mean": float(avg_missed),
                "cosine_similarity_mean": float(avg_cos),
                "inverse_kl_divergence_mean": float(avg_kl),
                "global_lir_mean": float(avg_glir),
                "cache_sparsity_mean": float(avg_spars),
                "data_points": len(data["amr"])
            }
        print("="*110)
        return stage_results

    summary_results = {}
    if metrics_by_layer_prefill:
        summary_results["prefill"] = print_and_aggregate_metrics(metrics_by_layer_prefill, "Prefill")
    if metrics_by_layer_gen:
        summary_results["generation"] = print_and_aggregate_metrics(metrics_by_layer_gen, "Generation")
        
    print("\nNote: 'Cache Sparsity' shows what % of tokens were evicted from the cache.")
    print("Note: All LIR metrics are bounded (0 to 1), where 1.0 (100%) is perfect retention.")
    
    # Write aggregate to file (existing behaviour)
    with open(args.output, 'w') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nSaved aggregate summary to {args.output}")
    
    # Build and write per-head detailed JSON
    def _aggregate_head_detail(hd_dict):
        """Convert {layer: {head: {metric: [values]}}} → {layer: {head: {metric_mean: float}}}."""
        result = {}
        for layer_str in sorted(hd_dict.keys(), key=int):
            result[layer_str] = {}
            for head_str in sorted(hd_dict[layer_str].keys(), key=int):
                h = hd_dict[layer_str][head_str]
                result[layer_str][head_str] = {
                    "amr_mean": float(np.mean(h["amr"])) if h["amr"] else 0.0,
                    "missed_mass_mean": float(np.mean(h["missed_mass"])) if h["missed_mass"] else 0.0,
                    "cosine_mean": float(np.mean(h["cosine"])) if h["cosine"] else 0.0,
                    "kl_inv_mean": float(np.mean(h["kl_inv"])) if h["kl_inv"] else 0.0,
                    "global_lir_mean": float(np.mean(h["global_lir"])) if h["global_lir"] else 0.0,
                    "sparsity_mean": float(np.mean(h["sparsity"])) if h["sparsity"] else 0.0,
                    "data_points": len(h["amr"])
                }
        return result
    
    detailed_results = {}
    if head_detail_prefill:
        detailed_results["prefill"] = _aggregate_head_detail(head_detail_prefill)
    if head_detail_gen:
        detailed_results["generation"] = _aggregate_head_detail(head_detail_gen)
    
    detailed_output = args.output.replace(".json", "_detailed.json")
    with open(detailed_output, 'w') as f:
        json.dump(detailed_results, f, indent=4)
    print(f"Saved per-head detailed summary to {detailed_output}")

if __name__ == "__main__":
    main()
