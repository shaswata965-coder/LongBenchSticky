import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import argparse
import shutil

# Define default path for the LIR metrics JSON
INPUT_PATH = "lir_comparison.json"

# ═══════════════════════════════════════════════════════════
# Premium Color Palette
# ═══════════════════════════════════════════════════════════
COLORS = {
    "amr":        "#4361EE",   # Vivid Blue
    "missed":     "#E63946",   # Crimson
    "cosine":     "#2EC4B6",   # Teal
    "kl_inv":     "#FF9F1C",   # Amber
    "global_lir": "#7209B7",   # Deep Violet
    "sparsity":   "#8D99AE",   # Slate Grey
    "bg_dark":    "#0F1624",   # Dashboard Dark
    "bg_card":    "#1A2332",   # Card Dark
    "text":       "#E8ECF1",   # Light text
    "grid":       "#2A3444",   # Muted grid
    "accent":     "#4CC9F0",   # Cyan accent
}

METRIC_LABELS = {
    "amr": "Attention Mass\nRetained",
    "missed": "Missed Mass\n(Drift)",
    "cosine": "Cosine\nSimilarity",
    "kl_inv": "Inverse KL\nDivergence",
    "global_lir": "Global LIR\n(Regret Score)",
    "sparsity": "Cache\nSparsity",
}

METRIC_LABELS_SHORT = {
    "amr": "Mass Retained",
    "missed": "Missed Mass",
    "cosine": "Cosine Sim",
    "kl_inv": "Inv KL Div",
    "global_lir": "Global LIR",
    "sparsity": "Sparsity",
}


def setup_dark_style():
    """Apply a sleek dark research-grade style."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg_dark"],
        "axes.facecolor": COLORS["bg_card"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    })


def load_and_prepare_data(filepath):
    """Loads LIR results and flattens into a Pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
        
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    records = []
    
    if "generation" in data or "prefill" in data:
        for phase, phase_data in data.items():
            for layer_str, metrics in phase_data.items():
                layer_idx = int(layer_str)
                records.append({
                    "layer": layer_idx,
                    "phase": phase,
                    "amr": metrics.get("attention_mass_retained_mean", 0) * 100,
                    "missed": metrics.get("missed_mass_drift_mean", 0) * 100,
                    "cosine": metrics.get("cosine_similarity_mean", 0) * 100,
                    "kl_inv": metrics.get("inverse_kl_divergence_mean", 0) * 100,
                    "global_lir": metrics.get("global_lir_mean", 0) * 100,
                    "sparsity": metrics.get("cache_sparsity_mean", 0) * 100
                })
    else:
        for layer_str, metrics in data.items():
            layer_idx = int(layer_str)
            records.append({
                "layer": layer_idx,
                "phase": "generation", 
                "amr": metrics.get("attention_mass_retained_mean", 0) * 100,
                "missed": metrics.get("missed_mass_drift_mean", 0) * 100,
                "cosine": metrics.get("cosine_similarity_mean", 0) * 100,
                "kl_inv": metrics.get("inverse_kl_divergence_mean", 0) * 100,
                "global_lir": metrics.get("global_lir_mean", 0) * 100,
                "sparsity": metrics.get("cache_sparsity_mean", 0) * 100
            })
                    
    df = pd.DataFrame(records)
    print(f"DataFrame created with {len(df)} records.")
    return df


def plot_radar_comparison(df, output_dir):
    """
    Radar/Spider Chart: Compare all metrics at a glance for each phase.
    Each layer is a separate polygon on the radar.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    metrics = ["amr", "cosine", "kl_inv", "global_lir"]
    labels = [METRIC_LABELS_SHORT[m] for m in metrics]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(8 * len(phases), 7), 
                              subplot_kw=dict(polar=True))
    if len(phases) == 1:
        axes = [axes]
    
    for ax_idx, phase in enumerate(phases):
        ax = axes[ax_idx]
        phase_df = df[df['phase'] == phase].sort_values(by="layer")
        
        ax.set_facecolor(COLORS["bg_card"])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10,
                          color=COLORS["text"])
        ax.set_ylim(0, 105)
        ax.set_rgrids([20, 40, 60, 80, 100], labels=["20", "40", "60", "80", "100"],
                      fontsize=8, color=COLORS["text"], alpha=0.5)
        ax.spines['polar'].set_color(COLORS["grid"])
        ax.grid(color=COLORS["grid"], alpha=0.3)
        
        cmap = plt.cm.cool
        layers = phase_df['layer'].values
        for i, (_, row) in enumerate(phase_df.iterrows()):
            values = [row[m] for m in metrics]
            values += values[:1]
            color = cmap(i / max(1, len(layers) - 1))
            ax.plot(angles, values, 'o-', linewidth=1.8, color=color, alpha=0.85, 
                    markersize=5, label=f"Layer {int(row['layer'])}")
            ax.fill(angles, values, color=color, alpha=0.06)
        
        ax.set_title(f"{phase.capitalize()} Phase", pad=20, fontsize=14, 
                     fontweight='bold', color=COLORS["accent"])
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8,
                 framealpha=0.3, edgecolor=COLORS["grid"])
    
    fig.suptitle("LIR Radar — Metric Overview Per Layer", fontsize=16, 
                 fontweight='bold', color=COLORS["text"], y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_radar.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def plot_gradient_heatmap(df, output_dir):
    """
    Premium gradient heatmap with custom colormap and annotations.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    metrics = ['amr', 'missed', 'cosine', 'kl_inv', 'global_lir']
    labels = [METRIC_LABELS_SHORT[m] for m in metrics]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(7 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]
    
    # Custom diverging colormap: Dark Red → Dark → Teal
    cmap = sns.diverging_palette(10, 170, s=80, l=55, n=256, as_cmap=True)
    
    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase].sort_values(by="layer")
        ax = axes[i]
        
        heatmap_data = phase_df.set_index('layer')[metrics].T
        heatmap_data.index = labels
        
        sns.heatmap(heatmap_data, cmap=cmap, annot=True, fmt=".1f", 
                    cbar_kws={'label': 'Score (%)' if i == len(phases) - 1 else '',
                              'shrink': 0.8},
                    linewidths=1.5, linecolor=COLORS["bg_dark"],
                    ax=ax, vmin=0, vmax=100,
                    annot_kws={"size": 10, "weight": "bold"})
        
        ax.set_title(f"{phase.capitalize()}", fontsize=14, fontweight='bold',
                     color=COLORS["accent"], pad=12)
        ax.set_xlabel("Transformer Layer", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(colors=COLORS["text"])
    
    fig.suptitle("LIR Intensity Heatmap", fontsize=16, fontweight='bold',
                 color=COLORS["text"], y=1.04)
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def plot_trend_with_fill(df, output_dir):
    """
    Trend lines with gradient fill underneath — shows metric evolution through depth.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    metrics = ['amr', 'cosine', 'kl_inv', 'global_lir']
    colors_list = [COLORS[m] for m in metrics]
    labels_list = [METRIC_LABELS_SHORT[m] for m in metrics]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(9 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]
    
    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase].sort_values(by="layer")
        ax = axes[i]
        layers = phase_df['layer'].values
        
        for j, metric in enumerate(metrics):
            values = phase_df[metric].values
            ax.plot(layers, values, '-o', color=colors_list[j], linewidth=2.2,
                    markersize=6, label=labels_list[j], zorder=3)
            ax.fill_between(layers, 0, values, color=colors_list[j], alpha=0.08)
        
        # Missed mass as a danger underlay
        missed_vals = phase_df['missed'].values
        ax.fill_between(layers, 0, missed_vals, color=COLORS["missed"], alpha=0.15,
                        hatch='///', label="Missed Mass (Danger Zone)")
        ax.plot(layers, missed_vals, '--', color=COLORS["missed"], linewidth=1.5,
                alpha=0.7)
        
        ax.set_title(f"{phase.capitalize()} — Metric Trend Through Depth",
                     fontsize=14, fontweight='bold', color=COLORS["accent"])
        ax.set_xlabel("Transformer Layer", fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.set_xticks(layers)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.5,
                 edgecolor=COLORS["grid"])
    
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_trends.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def plot_global_lir_gauge(df, output_dir):
    """
    Gauge-style visualization for Global LIR — the headline metric.
    Shows the overall 'health' of the eviction strategy.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4))
    if len(phases) == 1:
        axes = [axes]
    
    for i, phase in enumerate(phases):
        ax = axes[i]
        phase_df = df[df['phase'] == phase]
        avg_glir = phase_df['global_lir'].mean()
        
        # Create a horizontal progress bar
        bar_width = 0.6
        ax.barh(0, 100, height=bar_width, color=COLORS["bg_dark"], 
                edgecolor=COLORS["grid"], linewidth=1.5, zorder=1)
        
        # Color gradient based on score
        if avg_glir >= 80:
            bar_color = "#2EC4B6"  # Green-teal
        elif avg_glir >= 60:
            bar_color = "#FF9F1C"  # Amber warning
        else:
            bar_color = "#E63946"  # Red danger
        
        ax.barh(0, avg_glir, height=bar_width, color=bar_color,
                edgecolor="none", zorder=2, alpha=0.9)
        
        # Score text
        ax.text(avg_glir / 2, 0, f"{avg_glir:.1f}%", ha='center', va='center',
                fontsize=20, fontweight='bold', color='white', zorder=3)
        
        # Threshold markers
        for threshold, label in [(60, "Warning"), (80, "Good")]:
            ax.axvline(x=threshold, color=COLORS["text"], linestyle=':', alpha=0.4, zorder=1)
            ax.text(threshold, 0.45, label, ha='center', va='bottom', fontsize=7,
                    color=COLORS["text"], alpha=0.6)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.8)
        ax.set_yticks([])
        ax.set_xlabel("Global LIR Score (%)", fontsize=10)
        ax.set_title(f"{phase.capitalize()}", fontsize=13, fontweight='bold',
                     color=COLORS["accent"], pad=10)
        ax.grid(False)
    
    fig.suptitle("Global LIR — Eviction Regret Score", fontsize=15, fontweight='bold',
                 color=COLORS["text"], y=1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_global_gauge.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def plot_stacked_retention(df, output_dir):
    """
    Stacked area chart showing what happens to vanilla attention mass:
    - Retained (AMR) vs Missed (Drift) — they should roughly complement each other.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(9 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]
    
    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase].sort_values(by="layer")
        ax = axes[i]
        layers = phase_df['layer'].values
        amr = phase_df['amr'].values
        missed = phase_df['missed'].values
        
        ax.fill_between(layers, 0, amr, color=COLORS["amr"], alpha=0.7, 
                        label="Retained by Cache")
        ax.fill_between(layers, amr, amr + missed, color=COLORS["missed"], alpha=0.7,
                        label="Lost to Eviction (Drift)")
        
        # The gap between (amr + missed) and 100 is from sinks/local tokens
        remaining = 100 - (amr + missed)
        remaining = np.clip(remaining, 0, 100)
        ax.fill_between(layers, amr + missed, amr + missed + remaining, 
                        color=COLORS["sparsity"], alpha=0.3, label="Sinks + Local")
        
        ax.set_title(f"{phase.capitalize()} — Attention Mass Budget",
                     fontsize=14, fontweight='bold', color=COLORS["accent"])
        ax.set_xlabel("Transformer Layer", fontsize=11)
        ax.set_ylabel("Attention Mass (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_xticks(layers)
        ax.legend(loc='lower left', fontsize=9, framealpha=0.5)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_attention_budget.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def main():
    setup_dark_style()
    
    parser = argparse.ArgumentParser(description="Visualize LIR Results (Premium Edition)")
    parser.add_argument("--input", type=str, default=INPUT_PATH, 
                        help="Path to lir_comparison.json")
    parser.add_argument("--output_dir", type=str, default="./LIR", 
                        help="Directory to save plots")
    args, unknown = parser.parse_known_args()

    filepath = args.input
    output_dir = args.output_dir

    # Kaggle Specific Logic:
    # If the file isn't in the current directory, look recursively in Kaggle's input directory
    if not os.path.exists(filepath):
        print(f"Local file {filepath} not found. Searching in Kaggle /kaggle/input/ path...")
        kaggle_paths = glob.glob("/kaggle/input/**/lir_comparison.json", recursive=True)
        if kaggle_paths:
            filepath = kaggle_paths[0]
            print(f"Found input file at: {filepath}")
        else:
            print("No data file found. Exiting.")
            return

    df = load_and_prepare_data(filepath)
    if df is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy source JSON to LIR directory for reference
        lir_json_copy = os.path.join(output_dir, "lir_results.json")
        if os.path.abspath(filepath) != os.path.abspath(lir_json_copy):
            shutil.copy2(filepath, lir_json_copy)
            print(f"Copied source JSON to {lir_json_copy}")
        
        print("\n🎨 Generating Premium LIR Visualizations...")
        
        print("\n  [1/5] Radar Chart — Metric Overview")
        plot_radar_comparison(df, output_dir)
        
        print("  [2/5] Gradient Heatmap — Intensity Matrix")
        plot_gradient_heatmap(df, output_dir)
        
        print("  [3/5] Trend Lines — Depth Evolution")
        plot_trend_with_fill(df, output_dir)
        
        print("  [4/5] Global LIR Gauge — Eviction Health")
        plot_global_lir_gauge(df, output_dir)
        
        print("  [5/5] Stacked Area — Attention Mass Budget")
        plot_stacked_retention(df, output_dir)
        
        print(f"\n✅ All visualizations saved to: {output_dir}/")
        print(f"   └── lir_results.json (source data)")
        print(f"   └── lir_radar.png")
        print(f"   └── lir_heatmap.png")
        print(f"   └── lir_trends.png")
        print(f"   └── lir_global_gauge.png")
        print(f"   └── lir_attention_budget.png")

if __name__ == "__main__":
    main()
