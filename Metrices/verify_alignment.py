import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results"))
from npz_io import load_results_npz

def verify_alignment(vanilla_file, sticky_file):
    print(f"Loading {vanilla_file}...")
    try:
        v_data = load_results_npz(vanilla_file)
    except FileNotFoundError:
        print(f"Error: {vanilla_file} not found. Please ensure the file is generated/downloaded.")
        return

    print(f"Loading {sticky_file}...")
    try:
        s_data = load_results_npz(sticky_file)
    except FileNotFoundError:
        print(f"Error: {sticky_file} not found. Please ensure the file is generated/downloaded.")
        return

    print(f"\nLoaded {len(v_data)} pure vanilla samples and {len(s_data)} sticky samples.")
    if len(v_data) == 0 or len(s_data) == 0:
        return

    sample = 0
    print(f"\n--- Analyzing Sample {sample} ---")
    v_s = v_data[sample]
    s_s = s_data[sample]

    print("\n[ Metadata ]")
    print(f"  Vanilla generated: {v_s.get('metadata', {}).get('generated_token_count', 'N/A')} tokens. (Input: {v_s.get('metadata', {}).get('token_count_input', 'N/A')})")
    print(f"  Sticky generated:  {s_s.get('metadata', {}).get('generated_token_count', 'N/A')} tokens. (Input: {s_s.get('metadata', {}).get('token_count_input', 'N/A')})")

    layer = str(v_s['tracked_layers'][0])
    head = str(v_s['tracked_heads'][0])

    print(f"\n[ Prefill Stage ] (Testing Layer {layer}, Head {head})")
    v_pre = v_s['prefill_attention'][layer][head]
    s_pre = s_s['prefill_attention'][layer][head]
    
    # Clean arrays
    v_pre_arr = np.array(v_pre, dtype=float).flatten()
    s_pre_arr = np.array(s_pre, dtype=float).flatten()
    
    print(f"  Vanilla shape: {v_pre_arr.shape}")
    print(f"  Sticky shape:  {s_pre_arr.shape}")
    
    if len(v_pre_arr) > 0 and len(s_pre_arr) > 0:
        print(f"  Vanilla first 5 scores: {np.round(v_pre_arr[:5], 4)}")
        print(f"  Sticky first 5 scores:  {np.round(s_pre_arr[:5], 4)}")

        v_sum = np.sum(v_pre_arr)
        s_sum = np.sum(s_pre_arr)
        print(f"  Vanilla total sum: {v_sum:.4f}")
        print(f"  Sticky total sum:  {s_sum:.4f}")
        
    print(f"\n[ Generation Stage - Step 1 ] (Testing Layer {layer}, Head {head})")
    
    if len(v_s['generation_attention']) > 0 and len(s_s['generation_attention']) > 0:
        v_gen1 = v_s['generation_attention'][0][layer][head]
        s_gen1 = s_s['generation_attention'][0][layer][head]
        
        v_gen1_arr = np.array(v_gen1, dtype=float).flatten()
        s_gen1_arr = np.array(s_gen1, dtype=float).flatten()
        
        print(f"  Vanilla Step 1 shape: {v_gen1_arr.shape}")
        print(f"  Sticky Step 1 shape:  {s_gen1_arr.shape}")
        
        if len(v_gen1_arr) > 0 and len(s_gen1_arr) > 0:
            print(f"  Vanilla Step 1 first 5 scores: {np.round(v_gen1_arr[:5], 4)}")
            print(f"  Sticky Step 1 first 5 scores:  {np.round(s_gen1_arr[:5], 4)}")
            print(f"  Vanilla Step 1 total sum: {np.sum(v_gen1_arr):.4f}")
            print(f"  Sticky Step 1 total sum:  {np.sum(s_gen1_arr):.4f}")

    print(f"\n[ Generation Stage - Step 50 / Last Step ] (Testing Layer {layer}, Head {head})")
    last_step = min(len(v_s['generation_attention']), len(s_s['generation_attention'])) - 1
    
    if last_step > 0:
        v_genL = v_s['generation_attention'][last_step][layer][head]
        s_genL = s_s['generation_attention'][last_step][layer][head]
        
        v_genL_arr = np.array(v_genL, dtype=float).flatten()
        s_genL_arr = np.array(s_genL, dtype=float).flatten()
        
        print(f"  Vanilla Step {last_step+1} shape: {v_genL_arr.shape}")
        print(f"  Sticky Step {last_step+1} shape:  {s_genL_arr.shape}")
        
        if len(v_genL_arr) > 0 and len(s_genL_arr) > 0:
            print(f"  Vanilla Step {last_step+1} total sum: {np.sum(v_genL_arr):.4f}")
            print(f"  Sticky Step {last_step+1} total sum:  {np.sum(s_genL_arr):.4f}")
            
            non_zero_v = np.sum(v_genL_arr > 0)
            non_zero_s = np.sum(s_genL_arr > 0)
            print(f"  Vanilla non-zero entries (alive): {non_zero_v} ({non_zero_v/len(v_genL_arr)*100:.1f}%)")
            print(f"  Sticky non-zero entries (alive):  {non_zero_s} ({non_zero_s/len(s_genL_arr)*100:.1f}%)")
            
            # Simple Mass Retained calc test
            active_mask = (s_genL_arr > 0.0)
            if active_mask.shape == v_genL_arr.shape:
                retained_mass = np.sum(v_genL_arr[active_mask])
                total_mass = np.sum(v_genL_arr)
                if total_mass > 0:
                    print(f"\n  [Test Metric] Raw AMR Calculation: {retained_mass/total_mass*100:.2f}% Mass Retained")

if __name__ == '__main__':
    v_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", "pure_vanilla_baseline_results.npz")
    s_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", "sticky_baseline_results.npz")
    
    # If the user downloaded them into the current directory instead:
    if not os.path.exists(v_file) and os.path.exists("pure_vanilla_baseline_results.npz"):
        v_file = "pure_vanilla_baseline_results.npz"
    if not os.path.exists(s_file) and os.path.exists("sticky_baseline_results.npz"):
        s_file = "sticky_baseline_results.npz"

    verify_alignment(v_file, s_file)
