import argparse
import json
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def process_json(filepath, output_dir):
    print(f"\nProcessing {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = data.get("per_sample_detail", [])
    if not samples:
        print("  No sample detail found in JSON. Skipping.")
        return

    k_top = data.get("config", {}).get("k", 10)

    # Gather data aggregated across samples: layer -> head -> step -> { 'prefill': [], 'gen': [] }
    agg = {}
    
    for sample in samples:
        layers = sample.get("layers", {})
        for layer_str, layer_data in layers.items():
            if layer_str not in agg:
                agg[layer_str] = {}
            for head_str, head_data in layer_data.items():
                if head_str not in agg[layer_str]:
                    agg[layer_str][head_str] = {}
                
                timeline = head_data.get("timeline", [])
                for t in timeline:
                    step = t["step"]
                    if step not in agg[layer_str][head_str]:
                        agg[layer_str][head_str][step] = {'prefill': [], 'gen': []}
                    agg[layer_str][head_str][step]['prefill'].append(t["prefill_origin"])
                    agg[layer_str][head_str][step]['gen'].append(t["generation_origin"])

    # Base name for output PNGs
    basename = os.path.splitext(os.path.basename(filepath))[0]

    # Create one figure per layer
    for layer_str in sorted(agg.keys(), key=int):
        heads = sorted(agg[layer_str].keys(), key=int)
        num_heads = len(heads)
        
        # Grid layout: max 4 columns
        cols = min(4, num_heads) if num_heads > 0 else 1
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), squeeze=False)
        fig.suptitle(f"Prefill vs Generation Bias - Layer {layer_str} ({basename})", fontsize=16, fontweight='bold')
        
        for idx, head_str in enumerate(heads):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            
            steps = sorted(agg[layer_str][head_str].keys())
            prefill_means = [np.mean(agg[layer_str][head_str][s]['prefill']) for s in steps]
            gen_means = [np.mean(agg[layer_str][head_str][s]['gen']) for s in steps]
            
            ax.plot(steps, prefill_means, label='Prefill Origin', color='#1f77b4', marker='o', markersize=4, linewidth=2)
            ax.plot(steps, gen_means, label='Generation Origin', color='#ff7f0e', marker='s', markersize=4, linewidth=2)
            
            # Optionally plot intersection/union to signify correlation but keeping it clean with just origins
            
            ax.set_title(f"Head {head_str}", fontsize=12)
            ax.set_xlabel("Generation Step", fontsize=10)
            ax.set_ylabel("Avg Top-K Count", fontsize=10)
            
            ax.set_ylim(-0.5, k_top + 0.5)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if idx == 0:
                ax.legend(loc='best')
                
        # Hide empty subplots
        for idx in range(num_heads, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r, c].set_visible(False)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        out_name = f"{basename}_L{layer_str}_bias.png"
        out_path = os.path.join(output_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved plot: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Prefill vs Generation Bias from JSON timeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .json file or directory containing .json files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the plots. Defaults to the folder of the input file.")
    args = parser.parse_args()

    # Gather JSON files
    input_files = []
    if os.path.isdir(args.input):
        input_files = glob.glob(os.path.join(args.input, "*.json"))
        if not input_files:
            print(f"Error: No .json files found in directory {args.input}")
            sys.exit(1)
        # Default output dir is the input dir if not specified
        output_dir = args.output_dir or args.input
    elif os.path.isfile(args.input):
        input_files = [args.input]
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    else:
        print(f"Error: {args.input} is strictly not a valid file or directory.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Discovered {len(input_files)} JSON file(s).")
    for filepath in input_files:
        # Avoid processing non-target jsons if directory has them
        with open(filepath, 'r') as f:
            try:
                test = json.load(f)
                if "config" not in test or "summary_per_layer" not in test:
                    print(f"  Skipping {filepath} (Not a top-K bias JSON format)")
                    continue
            except:
                continue
                
        process_json(filepath, output_dir)

if __name__ == "__main__":
    main()
