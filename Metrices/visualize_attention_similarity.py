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

# Default path for the detailed metrics JSON
INPUT_PATH = "detailed_jaccard_results.json"

# ═══════════════A════════════════════════════════════════════
# Premium Color Palette (consistent with LIR visualizer)
# ═══════════════════════════════════════════════════════════
COLORS = {
    "prefill":    "#2EC4B6",   # Teal
    "generation": "#FF6B6B",   # Coral
    "highlight":  "#FFD166",   # Gold
    "accent":     "#4CC9F0",   # Cyan accent
    "diverge_lo": "#E63946",   # Crimson (bad)
    "diverge_hi": "#2EC4B6",   # Teal (good)
    "bg_dark":    "#0F1624",   # Dashboard Dark
    "bg_card":    "#1A2332",   # Card Dark
    "text":       "#E8ECF1",   # Light text
    "grid":       "#2A3444",   # Muted grid
    "subtle":     "#8D99AE",   # Slate Grey
}

NUM_WORST = 10  # How many worst-offender (layer, head) pairs to show


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


# ═══════════════════════════════════════════════════════════
# Plot 1: Similarity Over Time (Prefill & Generation)
# ═══════════════════════════════════════════════════════════
def plot_similarity_over_time(df, output_dir):
    """
    Line graphs of average similarity over steps for both prefill and generation.
    Shows overall trend with standard deviation shading.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    phases = ["prefill", "generation"]
    phase_colors = [COLORS["prefill"], COLORS["generation"]]
    
    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            axes[i].set_title(f"No data for {phase}")
            continue

        sns.lineplot(data=phase_df, x="step", y="similarity", errorbar='sd', 
                     ax=axes[i], color=phase_colors[i], linewidth=2.2)
        
        axes[i].set_title(f"Average Jaccard Similarity Over Time ({phase.capitalize()})",
                          fontsize=14, weight='bold', color=COLORS["accent"])
        axes[i].set_xlabel(f"{phase.capitalize()} Step", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Jaccard Similarity", fontsize=12)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, linestyle='--', alpha=0.4)
        
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_over_time_both.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 2: Layer × Head Heatmap (Prefill & Generation)
# ═══════════════════════════════════════════════════════════
def plot_layer_head_heatmap(df, output_dir):
    """
    Heatmaps of average similarity for each layer and head (Prefill vs Generation).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    phases = ["prefill", "generation"]
    
    cmap = sns.diverging_palette(10, 170, s=80, l=55, n=256, as_cmap=True)

    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            axes[i].set_title(f"No Data for {phase.capitalize()}")
            continue
            
        heatmap_data = phase_df.groupby(['layer', 'head'])['similarity'].mean().reset_index()
        pivot_df = heatmap_data.pivot(index="layer", columns="head", values="similarity")
        
        sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt=".3f", vmin=0, vmax=1, 
                    cbar_kws={'label': 'Mean Jaccard' if i == 1 else '', 'shrink': 0.8},
                    linewidths=1.5, linecolor=COLORS["bg_dark"],
                    annot_kws={"size": 9, "weight": "bold"},
                    ax=axes[i])
        
        axes[i].set_title(f"Mean Jaccard ({phase.capitalize()})",
                          fontsize=14, weight='bold', color=COLORS["accent"])
        axes[i].set_xlabel("Attention Head Index", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Transformer Layer Index", fontsize=12)
        else:
            axes[i].set_ylabel("")
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_heatmap_both.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 3: Layer-wise Box Plot Distribution
# ═══════════════════════════════════════════════════════════
def plot_layerwise_distribution(df, output_dir):
    """
    Box plots showing distribution of similarity scores across heads for each layer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    phases = ["prefill", "generation"]
    
    palette = sns.color_palette("husl", n_colors=df['layer'].nunique())

    for i, phase in enumerate(phases):
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            axes[i].set_title(f"No Data for {phase.capitalize()}")
            continue
            
        sns.boxplot(data=phase_df, x="layer", y="similarity", palette=palette,
                    fliersize=2, ax=axes[i])
        
        axes[i].set_title(f"Layer-wise Jaccard Distribution ({phase.capitalize()})",
                          fontsize=14, weight='bold', color=COLORS["accent"])
        axes[i].set_xlabel("Transformer Layer Index", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Jaccard Similarity", fontsize=12)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.4)
        
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_distribution_both.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 4: Per-Layer Trend Decomposition
# ═══════════════════════════════════════════════════════════
def plot_per_layer_trends(df, output_dir):
    """
    Individual line per layer showing average similarity over generation steps.
    Color-coded by layer index, with separate panels for prefill vs generation.
    Makes it immediately obvious which layers degrade fastest.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    layers_sorted = sorted(df['layer'].unique())
    
    cmap = plt.cm.cool
    n_layers = len(layers_sorted)

    fig, axes = plt.subplots(1, len(phases), figsize=(10 * len(phases), 6), sharey=True)
    if len(phases) == 1:
        axes = [axes]
    
    for ax_idx, phase in enumerate(phases):
        ax = axes[ax_idx]
        phase_df = df[df['phase'] == phase]
        
        for li, layer in enumerate(layers_sorted):
            layer_df = phase_df[phase_df['layer'] == layer]
            if layer_df.empty:
                continue
            avg_per_step = layer_df.groupby('step')['similarity'].mean()
            color = cmap(li / max(1, n_layers - 1))
            ax.plot(avg_per_step.index, avg_per_step.values, '-', color=color,
                    linewidth=1.8, alpha=0.85, label=f"Layer {layer}")
        
        ax.set_title(f"Per-Layer Similarity Trend ({phase.capitalize()})",
                     fontsize=14, fontweight='bold', color=COLORS["accent"])
        ax.set_xlabel(f"{phase.capitalize()} Step", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Jaccard Similarity", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower left', fontsize=8, framealpha=0.5, 
                  edgecolor=COLORS["grid"], ncol=2)
    
    fig.suptitle("Layer-Decomposed Similarity — Which Layers Diverge?",
                 fontsize=16, fontweight='bold', color=COLORS["text"], y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_per_layer_trends.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 5: Prefill vs Generation Divergence Heatmap (Delta)
# ═══════════════════════════════════════════════════════════
def plot_divergence_heatmap(df, output_dir):
    """
    Shows the DIFFERENCE in mean similarity (prefill - generation) per (layer, head).
    Positive = generation is worse. Red = largest drop. Blue/teal = stable.
    """
    prefill_df = df[df['phase'] == 'prefill']
    gen_df = df[df['phase'] == 'generation']
    
    if prefill_df.empty or gen_df.empty:
        print("  Skipping divergence heatmap — need both prefill and generation data.")
        return
    
    prefill_mean = prefill_df.groupby(['layer', 'head'])['similarity'].mean().reset_index()
    prefill_mean.rename(columns={'similarity': 'prefill_sim'}, inplace=True)
    
    gen_mean = gen_df.groupby(['layer', 'head'])['similarity'].mean().reset_index()
    gen_mean.rename(columns={'similarity': 'gen_sim'}, inplace=True)
    
    merged = prefill_mean.merge(gen_mean, on=['layer', 'head'], how='outer').fillna(0)
    merged['delta'] = merged['prefill_sim'] - merged['gen_sim']
    
    pivot_df = merged.pivot(index="layer", columns="head", values="delta")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Red = large positive delta (generation degraded). Blue = negative (generation improved)
    cmap = sns.diverging_palette(220, 10, s=80, l=55, n=256, as_cmap=True)
    max_delta = max(abs(pivot_df.values.min()), abs(pivot_df.values.max()), 0.01)
    
    sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt=".3f",
                vmin=-max_delta, vmax=max_delta,
                cbar_kws={'label': 'Δ Similarity (Prefill - Generation)', 'shrink': 0.8},
                linewidths=1.5, linecolor=COLORS["bg_dark"],
                annot_kws={"size": 10, "weight": "bold"},
                ax=ax)
    
    ax.set_title("Divergence Heatmap — Where Does Generation Degrade?",
                 fontsize=15, fontweight='bold', color=COLORS["accent"], pad=15)
    ax.set_xlabel("Attention Head Index", fontsize=12)
    ax.set_ylabel("Transformer Layer Index", fontsize=12)
    ax.invert_yaxis()
    
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_divergence_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 6: Worst-Offender Bar Chart
# ═══════════════════════════════════════════════════════════
def plot_worst_offenders(df, output_dir):
    """
    Ranks the bottom-N (layer, head) pairs by average generation similarity,
    flagging the most divergent attention heads.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(8 * len(phases), 7))
    if len(phases) == 1:
        axes = [axes]
    
    for ax_idx, phase in enumerate(phases):
        ax = axes[ax_idx]
        phase_df = df[df['phase'] == phase]
        
        # Average per (layer, head)
        avg_df = phase_df.groupby(['layer', 'head'])['similarity'].mean().reset_index()
        avg_df['label'] = avg_df.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)
        
        # Sort ascending (worst first) and take bottom N
        worst = avg_df.nsmallest(NUM_WORST, 'similarity')
        
        # Color by severity
        colors = []
        for sim in worst['similarity']:
            if sim < 0.5:
                colors.append("#E63946")  # Crimson — terrible
            elif sim < 0.75:
                colors.append("#FF9F1C")  # Amber — warning
            elif sim < 0.9:
                colors.append("#FFD166")  # Gold — marginal
            else:
                colors.append("#2EC4B6")  # Teal — good
        
        bars = ax.barh(worst['label'], worst['similarity'], color=colors, 
                       edgecolor=COLORS["grid"], linewidth=0.8, height=0.6)
        
        # Annotate values
        for bar, sim in zip(bars, worst['similarity']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{sim:.4f}", va='center', ha='left', fontsize=10,
                    fontweight='bold', color=COLORS["text"])
        
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Mean Jaccard Similarity", fontsize=12)
        ax.set_title(f"Bottom-{NUM_WORST} Worst Offenders ({phase.capitalize()})",
                     fontsize=14, fontweight='bold', color=COLORS["accent"])
        ax.invert_yaxis()  # Worst at top
        ax.axvline(x=0.9, color=COLORS["text"], linestyle=':', alpha=0.4, label="0.9 threshold")
        ax.axvline(x=0.75, color="#FF9F1C", linestyle=':', alpha=0.4, label="0.75 threshold")
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.5)
    
    fig.suptitle("Worst-Offending (Layer, Head) Pairs",
                 fontsize=16, fontweight='bold', color=COLORS["text"], y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_worst_offenders.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 7: Per-Head Strip Plot (Head Variance Within Layers)
# ═══════════════════════════════════════════════════════════
def plot_per_head_strip(df, output_dir):
    """
    Within each layer, show the distribution of per-step similarity per individual head
    as a strip/swarm plot, making head-level variance visible.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    layers_sorted = sorted(df['layer'].unique())
    n_layers = len(layers_sorted)
    
    # For generation, do a per-layer facet with heads on x-axis
    for phase in phases:
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            continue
        
        n_cols = min(4, n_layers)
        n_rows = int(np.ceil(n_layers / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                  sharey=True)
        axes_flat = np.array(axes).flatten() if n_layers > 1 else [axes]
        
        palette = sns.color_palette("husl", n_colors=phase_df['head'].nunique())
        
        for idx, layer in enumerate(layers_sorted):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            layer_df = phase_df[phase_df['layer'] == layer]
            
            if layer_df.empty:
                ax.set_visible(False)
                continue
            
            sns.stripplot(data=layer_df, x="head", y="similarity", 
                         palette=palette, size=3, alpha=0.5, jitter=True, ax=ax)
            
            # Overlay mean markers
            head_means = layer_df.groupby('head')['similarity'].mean()
            ax.scatter(range(len(head_means)), head_means.values, 
                      color='white', s=60, zorder=5, edgecolors=COLORS["accent"],
                      linewidths=2, marker='D', label='Mean')
            
            ax.set_title(f"Layer {layer}", fontsize=12, fontweight='bold',
                        color=COLORS["accent"])
            ax.set_xlabel("Head" if idx >= (n_rows - 1) * n_cols else "")
            ax.set_ylabel("Jaccard" if idx % n_cols == 0 else "")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_layers, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        fig.suptitle(f"Per-Head Similarity Distribution ({phase.capitalize()})",
                     fontsize=16, fontweight='bold', color=COLORS["text"], y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, f"similarity_per_head_strip_{phase}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
        print(f"  Saved {path}")
        plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 8: Summary Statistics Panel
# ═══════════════════════════════════════════════════════════
def plot_summary_panel(df, output_dir):
    """
    Text-based annotation panel showing overall stats: mean, std, min,
    worst layer, worst head for each phase.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    y_pos = 0.92
    
    ax.text(0.5, y_pos + 0.06, "JACCARD SIMILARITY — SUMMARY STATISTICS",
            ha='center', va='top', fontsize=18, fontweight='bold',
            color=COLORS["accent"], transform=ax.transAxes)
    
    for phase in phases:
        phase_df = df[df['phase'] == phase]
        
        overall_mean = phase_df['similarity'].mean()
        overall_std = phase_df['similarity'].std()
        overall_min = phase_df['similarity'].min()
        overall_max = phase_df['similarity'].max()
        
        # Per (layer, head) averages
        lh_avg = phase_df.groupby(['layer', 'head'])['similarity'].mean()
        worst_idx = lh_avg.idxmin()
        best_idx = lh_avg.idxmax()
        
        # Per-layer averages
        layer_avg = phase_df.groupby('layer')['similarity'].mean()
        worst_layer = layer_avg.idxmin()
        best_layer = layer_avg.idxmax()
        
        color = COLORS["prefill"] if phase == "prefill" else COLORS["generation"]
        
        ax.text(0.5, y_pos, f"══════ {phase.upper()} ══════",
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=color, transform=ax.transAxes)
        y_pos -= 0.08
        
        stats_text = (
            f"Overall Mean: {overall_mean:.4f}  |  Std: {overall_std:.4f}  |  "
            f"Min: {overall_min:.4f}  |  Max: {overall_max:.4f}\n"
            f"Worst Head: L{worst_idx[0]}H{worst_idx[1]} ({lh_avg[worst_idx]:.4f})  |  "
            f"Best Head: L{best_idx[0]}H{best_idx[1]} ({lh_avg[best_idx]:.4f})\n"
            f"Worst Layer: {worst_layer} (avg {layer_avg[worst_layer]:.4f})  |  "
            f"Best Layer: {best_layer} (avg {layer_avg[best_layer]:.4f})\n"
            f"Total data points: {len(phase_df):,}  |  "
            f"Layers: {sorted(phase_df['layer'].unique())}  |  "
            f"Heads: {sorted(phase_df['head'].unique())}"
        )
        
        ax.text(0.5, y_pos, stats_text,
                ha='center', va='top', fontsize=11,
                color=COLORS["text"], transform=ax.transAxes,
                family='monospace', linespacing=1.6)
        y_pos -= 0.30
    
    # Delta summary
    if "prefill" in phases and "generation" in phases:
        p_mean = df[df['phase'] == 'prefill']['similarity'].mean()
        g_mean = df[df['phase'] == 'generation']['similarity'].mean()
        delta = p_mean - g_mean
        
        delta_color = "#E63946" if delta > 0.05 else "#FF9F1C" if delta > 0.01 else "#2EC4B6"
        ax.text(0.5, y_pos, f"Δ (Prefill - Generation): {delta:+.4f}",
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=delta_color, transform=ax.transAxes)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_summary_panel.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Plot 9: Per-Layer Head Variance Bar Chart
# ═══════════════════════════════════════════════════════════
def plot_head_variance_per_layer(df, output_dir):
    """
    For each layer, calculates the variance of mean similarity across heads.
    High variance = some heads are good, some are bad = inconsistent behaviour.
    """
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(8 * len(phases), 6), sharey=True)
    if len(phases) == 1:
        axes = [axes]
    
    for ax_idx, phase in enumerate(phases):
        ax = axes[ax_idx]
        phase_df = df[df['phase'] == phase]
        
        # Mean per (layer, head), then variance across heads per layer
        lh_mean = phase_df.groupby(['layer', 'head'])['similarity'].mean().reset_index()
        layer_variance = lh_mean.groupby('layer')['similarity'].var().reset_index()
        layer_variance.columns = ['layer', 'head_variance']
        layer_variance = layer_variance.sort_values('layer')
        
        colors = []
        for v in layer_variance['head_variance']:
            if v > 0.01:
                colors.append("#E63946")
            elif v > 0.005:
                colors.append("#FF9F1C")
            elif v > 0.001:
                colors.append("#FFD166")
            else:
                colors.append("#2EC4B6")
        
        bars = ax.bar(layer_variance['layer'].astype(str), layer_variance['head_variance'],
                      color=colors, edgecolor=COLORS["grid"], linewidth=0.8)
        
        # Annotate
        for bar, v in zip(bars, layer_variance['head_variance']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f"{v:.4f}", ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=COLORS["text"])
        
        ax.set_title(f"Head Variance Per Layer ({phase.capitalize()})",
                     fontsize=14, fontweight='bold', color=COLORS["accent"])
        ax.set_xlabel("Transformer Layer", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Variance of Head Means", fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Legend for color thresholds
        patches = [
            mpatches.Patch(color="#E63946", label="High (>0.01)"),
            mpatches.Patch(color="#FF9F1C", label="Medium (>0.005)"),
            mpatches.Patch(color="#FFD166", label="Low (>0.001)"),
            mpatches.Patch(color="#2EC4B6", label="Minimal (≤0.001)"),
        ]
        ax.legend(handles=patches, fontsize=8, loc='upper right', framealpha=0.5)
    
    fig.suptitle("Inter-Head Consistency — Which Layers Have Divergent Heads?",
                 fontsize=16, fontweight='bold', color=COLORS["text"], y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_head_variance.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def main():
    setup_dark_style()
    
    parser = argparse.ArgumentParser(description="Visualize Jaccard Similarity Results (Enhanced)")
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="Path to detailed_jaccard_results.json")
    parser.add_argument("--output_dir", type=str, default="./Jaccard", help="Directory to save the plots")
    args, unknown = parser.parse_known_args()
    
    filepath = args.input
    output_dir = args.output_dir
    
    # Kaggle Specific Logic
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
        
        print("\n🎨 Generating Enhanced Jaccard Visualizations...\n")
        
        print("  [1/8] Similarity Over Time")
        plot_similarity_over_time(df, output_dir)
        
        print("  [2/8] Layer × Head Heatmap")
        plot_layer_head_heatmap(df, output_dir)
        
        print("  [3/8] Layer-wise Distribution (Box Plots)")
        plot_layerwise_distribution(df, output_dir)
        
        print("  [4/8] Per-Layer Trend Decomposition")
        plot_per_layer_trends(df, output_dir)
        
        print("  [5/8] Divergence Heatmap (Prefill − Generation)")
        plot_divergence_heatmap(df, output_dir)
        
        print("  [6/8] Worst-Offender Bar Chart")
        plot_worst_offenders(df, output_dir)
        
        print("  [7/8] Per-Head Strip Plots")
        plot_per_head_strip(df, output_dir)
        
        print("  [8/8] Summary Statistics Panel")
        plot_summary_panel(df, output_dir)
        
        print("  [BONUS] Head Variance Per Layer")
        plot_head_variance_per_layer(df, output_dir)
        
        print(f"\n✅ All Jaccard visualizations saved to: {output_dir}/")
        print(f"   └── similarity_over_time_both.png")
        print(f"   └── similarity_heatmap_both.png")
        print(f"   └── similarity_distribution_both.png")
        print(f"   └── similarity_per_layer_trends.png")
        print(f"   └── similarity_divergence_heatmap.png")
        print(f"   └── similarity_worst_offenders.png")
        print(f"   └── similarity_per_head_strip_prefill.png")
        print(f"   └── similarity_per_head_strip_generation.png")
        print(f"   └── similarity_summary_panel.png")
        print(f"   └── similarity_head_variance.png")

if __name__ == "__main__":
    main()
