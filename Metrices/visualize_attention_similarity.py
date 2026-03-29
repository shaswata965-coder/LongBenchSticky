import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import argparse

# Define default path for the detailed metrics JSON
INPUT_PATH = "detailed_jaccard_results.json"

def load_and_prepare_data(filepath):
    """
    Loads detailed Jaccard results and flattens into a Pandas DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run the calculating script first.")
        return None
        
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Flatten the hierarchical JSON into a list of records
    records = []
    print("Processing JSON into DataFrame...")
    for sample in data:
        sample_idx = sample.get("sample_index", 0)
        layers = sample.get("layers", {})
        
        for layer_str, layer_data in layers.items():
            layer_idx = int(layer_str)
            heads = layer_data.get("heads", {})
            
            for head_str, head_data in heads.items():
                head_idx = int(head_str)
                
                # prefill steps
                for p_step in head_data.get("prefill_steps", []):
                    records.append({
                        "sample": sample_idx,
                        "layer": layer_idx,
                        "head": head_idx,
                        "phase": "prefill",
                        "step": p_step["step"],
                        "similarity": p_step["jaccard_similarity"]
                    })
                    
                # generation steps
                for g_step in head_data.get("generation_steps", []):
                    records.append({
                        "sample": sample_idx,
                        "layer": layer_idx,
                        "head": head_idx,
                        "phase": "generation",
                        "step": g_step["step"],
                        "similarity": g_step["jaccard_similarity"]
                    })
                    
    df = pd.DataFrame(records)
    print(f"DataFrame created with {len(df)} records.")
    return df

def plot_similarity_over_time(df, output_dir):
    """
    Plots line graphs of average similarity over steps for both prefill and generation.
    Ideal for seeing if similarity decays over time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    phases = ["prefill", "generation"]
    
    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            axes[i].set_title(f"No data for {phase}")
            continue
            
        # seaborn lineplot automatically calculates mean and standard deviation
        sns.lineplot(data=phase_df, x="step", y="similarity", errorbar='sd', ax=axes[i], color='dodgerblue')
        
        axes[i].set_title(f"Average Jaccard Similarity Over Time ({phase.capitalize()})", fontsize=14, weight='bold')
        axes[i].set_xlabel(f"{phase.capitalize()} Step", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Jaccard Similarity", fontsize=12)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    output_filename = os.path.join(output_dir, "similarity_over_time_both.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {output_filename}")
    
    plt.show()
    phase_df = df[df['phase'] == phase]
    if phase_df.empty:
        print(f"No data for phase: {phase}")
        return
        
    plt.figure(figsize=(10, 6))
    
    # seaborn lineplot automatically calculates mean and standard deviation (or CI)
    # We use errorbar='sd' (or 'ci' for confidence interval) to show spread
    sns.lineplot(data=phase_df, x="step", y="similarity", errorbar='sd', color='dodgerblue')
    
    plt.title(f"Average Jaccard Similarity Over Time ({phase.capitalize()})", fontsize=14, weight='bold')
    plt.xlabel("Generation Step", fontsize=12)
    plt.ylabel("Jaccard Similarity", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f"similarity_over_time_{phase}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {output_filename}")
    
    # In a Kaggle Notebook, plt.show() will render the plot inline
    plt.show()

def plot_layer_head_heatmap(df, output_dir):
    """
    Plots heatmaps of average similarity for each layer and head (Prefill vs Generation).
    Highlights structural differences (e.g., specific layers/heads diverging).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    phases = ["prefill", "generation"]

    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            axes[i].set_title(f"No Data for {phase.capitalize()}")
            continue
            
        # Calculate mean similarity per layer and per head across all samples/steps
        heatmap_data = phase_df.groupby(['layer', 'head'])['similarity'].mean().reset_index()
        
        # Pivot data into a 2D matrix for the heatmap
        pivot_df = heatmap_data.pivot(index="layer", columns="head", values="similarity")
        
        sns.heatmap(pivot_df, cmap="viridis", annot=False, vmin=0, vmax=1, 
                    cbar_kws={'label': 'Mean Jaccard' if i==1 else ''}, ax=axes[i])
        
        axes[i].set_title(f"Average Jaccard Similarity ({phase.capitalize()})", fontsize=14, weight='bold')
        axes[i].set_xlabel("Attention Head Index", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Transformer Layer Index", fontsize=12)
        else:
            axes[i].set_ylabel("") # Hide y label for second plot
            
        # Reverse the Y-axis so Layer 0 is at bottom (standard plot convention)
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_heatmap_both.png"), dpi=300, bbox_inches='tight')
    print("Saved similarity_heatmap_both.png")
    
    plt.show()

def plot_layerwise_distribution(df, output_dir):
    """
    Plots box plots showing the distribution of similarity scores across heads for each layer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    phases = ["prefill", "generation"]

    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            axes[i].set_title(f"No Data for {phase.capitalize()}")
            continue
            
        # Use seaborn's boxplot to show the distribution
        sns.boxplot(data=phase_df, x="layer", y="similarity", palette="husl", fliersize=2, ax=axes[i])
        
        axes[i].set_title(f"Layer-wise Jaccard Distribution ({phase.capitalize()})", fontsize=14, weight='bold')
        axes[i].set_xlabel("Transformer Layer Index", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Jaccard Similarity", fontsize=12)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_distribution_both.png"), dpi=300, bbox_inches='tight')
    print("Saved similarity_distribution_both.png")
    
    plt.show()

def main():
    # Set global plotting aesthetics
    sns.set_theme(style="whitegrid")
    
    parser = argparse.ArgumentParser(description="Visualize Jaccard Similarity Results")
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="Path to detailed_jaccard_results.json")
    parser.add_argument("--output_dir", type=str, default="./Jaccard", help="Directory to save the plots")
    args, unknown = parser.parse_known_args()
    
    filepath = args.input
    output_dir = args.output_dir
    
    # Kaggle Specific Logic:
    # If the file isn't in the current directory, look recursively in Kaggle's input directory
    if not os.path.exists(filepath):
        print(f"Local file {filepath} not found. Searching in Kaggle /kaggle/input/ path...")
        kaggle_paths = glob.glob("/kaggle/input/**/detailed_jaccard_results.json", recursive=True)
        if kaggle_paths:
            filepath = kaggle_paths[0]
            print(f"Found input file at: {filepath}")
        else:
            print("No data file found. Exiting.")
            return

    df = load_and_prepare_data(filepath)
    if df is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n--- Generating Line Plots (Prefill & Generation) ---")
        plot_similarity_over_time(df, output_dir)
        
        print("\n--- Generating Heatmaps (Prefill & Generation) ---")
        plot_layer_head_heatmap(df, output_dir)
        
        print("\n--- Generating Box Plots (Prefill & Generation) ---")
        plot_layerwise_distribution(df, output_dir)
        
        print(f"\nAll visualizations complete! Saved to {output_dir}")

if __name__ == "__main__":
    main()
