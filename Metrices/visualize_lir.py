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

# Define default paths for the LIR metrics JSON
INPUT_PATH = "lir_comparison.json"
DETAILED_INPUT_PATH = "lir_comparison_detailed.json"

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

METRIC_COLORS = {
    "amr": "#4361EE",
    "missed": "#E63946",
    "cosine": "#2EC4B6",
    "kl_inv": "#FF9F1C",
    "global_lir": "#7209B7",
    "sparsity": "#8D99AE",
}

NUM_WORST = 10  # Top-N worst offenders


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


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def load_and_prepare_data(filepath):
    """Loads LIR aggregate results and flattens into a Pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
        
    print(f"Loading aggregate data from {filepath}...")
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


def load_detailed_data(filepath):
    """
    Loads per-head detailed LIR results into a DataFrame.
    Structure: {phase: {layer: {head: {metric_mean: float}}}}
    """
    if not os.path.exists(filepath):
        print(f"Per-head detailed file {filepath} not found. Skipping per-head plots.")
        return None
    
    print(f"Loading per-head detailed data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    records = []
    for phase, phase_data in data.items():
        for layer_str, layer_data in phase_data.items():
            layer_idx = int(layer_str)
            for head_str, metrics in layer_data.items():
                head_idx = int(head_str)
                records.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "phase": phase,
                    "amr": metrics.get("amr_mean", 0) * 100,
                    "missed": metrics.get("missed_mass_mean", 0) * 100,
                    "cosine": metrics.get("cosine_mean", 0) * 100,
                    "kl_inv": metrics.get("kl_inv_mean", 0) * 100,
                    "global_lir": metrics.get("global_lir_mean", 0) * 100,
                    "sparsity": metrics.get("sparsity_mean", 0) * 100,
                    "data_points": metrics.get("data_points", 0),
                })
    
    df = pd.DataFrame(records)
    print(f"Per-head DataFrame created with {len(df)} records.")
    return df


# ═══════════════════════════════════════════════════════════
# ORIGINAL PLOTS (preserved, enhanced with dark theme)
# ═══════════════════════════════════════════════════════════

def plot_radar_comparison(df, output_dir):
    """Radar/Spider Chart: Compare all metrics at a glance for each phase."""
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
    """Premium gradient heatmap with custom colormap and annotations."""
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    metrics = ['amr', 'missed', 'cosine', 'kl_inv', 'global_lir']
    labels = [METRIC_LABELS_SHORT[m] for m in metrics]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(7 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]
    
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
    """Trend lines with gradient fill showing metric evolution through depth."""
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    metrics = ['amr', 'cosine', 'kl_inv', 'global_lir']
    colors_list = [METRIC_COLORS[m] for m in metrics]
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
    """Gauge-style visualization for Global LIR — the headline metric."""
    phases = [p for p in ["prefill", "generation"] if not df[df['phase'] == p].empty]
    
    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4))
    if len(phases) == 1:
        axes = [axes]
    
    for i, phase in enumerate(phases):
        ax = axes[i]
        phase_df = df[df['phase'] == phase]
        avg_glir = phase_df['global_lir'].mean()
        
        bar_width = 0.6
        ax.barh(0, 100, height=bar_width, color=COLORS["bg_dark"], 
                edgecolor=COLORS["grid"], linewidth=1.5, zorder=1)
        
        if avg_glir >= 80:
            bar_color = "#2EC4B6"
        elif avg_glir >= 60:
            bar_color = "#FF9F1C"
        else:
            bar_color = "#E63946"
        
        ax.barh(0, avg_glir, height=bar_width, color=bar_color,
                edgecolor="none", zorder=2, alpha=0.9)
        
        ax.text(avg_glir / 2, 0, f"{avg_glir:.1f}%", ha='center', va='center',
                fontsize=20, fontweight='bold', color='white', zorder=3)
        
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
    """Stacked area chart showing attention mass budget: retained vs missed."""
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


# ═══════════════════════════════════════════════════════════
# NEW PLOTS — Per-Head Divergence (require detailed data)
# ═══════════════════════════════════════════════════════════

def plot_per_head_heatmaps(detail_df, output_dir):
    """
    Per-head heatmap: For each key metric (amr, cosine, global_lir),
    show a (layer × head) heatmap side-by-side for prefill vs generation.
    """
    focus_metrics = ["amr", "cosine", "global_lir"]
    focus_labels = ["Mass Retained (%)", "Cosine Similarity (%)", "Global LIR (%)"]
    phases = [p for p in ["prefill", "generation"] if not detail_df[detail_df['phase'] == p].empty]
    
    for metric, label in zip(focus_metrics, focus_labels):
        fig, axes = plt.subplots(1, len(phases), figsize=(8 * len(phases), 7))
        if len(phases) == 1:
            axes = [axes]
        
        cmap = sns.diverging_palette(10, 170, s=80, l=55, n=256, as_cmap=True)
        
        for i, phase in enumerate(phases):
            ax = axes[i]
            phase_df = detail_df[detail_df['phase'] == phase]
            
            if phase_df.empty:
                ax.set_title(f"No data for {phase}")
                continue
            
            pivot = phase_df.pivot(index="layer", columns="head", values=metric)
            
            sns.heatmap(pivot, cmap=cmap, annot=True, fmt=".1f",
                        vmin=0, vmax=100,
                        cbar_kws={'label': label if i == len(phases) - 1 else '', 'shrink': 0.8},
                        linewidths=1.5, linecolor=COLORS["bg_dark"],
                        annot_kws={"size": 9, "weight": "bold"},
                        ax=ax)
            
            ax.set_title(f"{phase.capitalize()} — {label}",
                         fontsize=14, fontweight='bold', color=COLORS["accent"])
            ax.set_xlabel("Attention Head", fontsize=11)
            if i == 0:
                ax.set_ylabel("Transformer Layer", fontsize=11)
            else:
                ax.set_ylabel("")
            ax.invert_yaxis()
        
        fig.suptitle(f"Per-Head {metric.upper()} — Layer × Head Breakdown",
                     fontsize=16, fontweight='bold', color=COLORS["text"], y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, f"lir_per_head_{metric}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
        print(f"  Saved {path}")
        plt.close()


def plot_delta_heatmaps(detail_df, output_dir):
    """
    Prefill vs Generation Delta heatmap: (prefill_metric - generation_metric)
    per (layer, head). Positive = generation is worse. Red = largest drop.
    """
    prefill_df = detail_df[detail_df['phase'] == 'prefill']
    gen_df = detail_df[detail_df['phase'] == 'generation']
    
    if prefill_df.empty or gen_df.empty:
        print("  Skipping delta heatmaps — need both prefill and generation data.")
        return
    
    focus_metrics = ["amr", "cosine", "global_lir", "missed"]
    focus_labels = ["Δ Mass Retained (%)", "Δ Cosine Sim (%)", "Δ Global LIR (%)", "Δ Missed Mass (%)"]
    
    n_metrics = len(focus_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 7))
    
    cmap_div = sns.diverging_palette(220, 10, s=80, l=55, n=256, as_cmap=True)
    
    for idx, (metric, label) in enumerate(zip(focus_metrics, focus_labels)):
        ax = axes[idx]
        
        p_pivot = prefill_df.pivot(index="layer", columns="head", values=metric)
        g_pivot = gen_df.pivot(index="layer", columns="head", values=metric)
        
        # Align indices
        common_layers = p_pivot.index.intersection(g_pivot.index)
        common_heads = p_pivot.columns.intersection(g_pivot.columns)
        
        delta = p_pivot.loc[common_layers, common_heads] - g_pivot.loc[common_layers, common_heads]
        
        max_abs = max(abs(delta.values.min()), abs(delta.values.max()), 0.1)
        
        sns.heatmap(delta, cmap=cmap_div, annot=True, fmt=".1f",
                    vmin=-max_abs, vmax=max_abs,
                    cbar_kws={'label': label if idx == n_metrics - 1 else '', 'shrink': 0.7},
                    linewidths=1.5, linecolor=COLORS["bg_dark"],
                    annot_kws={"size": 8, "weight": "bold"},
                    ax=ax)
        
        ax.set_title(label, fontsize=12, fontweight='bold', color=COLORS["accent"])
        ax.set_xlabel("Head", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Layer", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.invert_yaxis()
    
    fig.suptitle("Prefill − Generation Delta — Where Does Generation Degrade?",
                 fontsize=16, fontweight='bold', color=COLORS["text"], y=1.03)
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_delta_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def plot_head_variance_per_layer(detail_df, output_dir):
    """
    For each layer, shows the variance across heads' metric values.
    High variance = inconsistent head behavior within a layer.
    """
    phases = [p for p in ["prefill", "generation"] if not detail_df[detail_df['phase'] == p].empty]
    focus_metrics = ["amr", "cosine", "global_lir"]
    
    fig, axes = plt.subplots(len(focus_metrics), len(phases), 
                              figsize=(7 * len(phases), 4 * len(focus_metrics)),
                              sharey='row')
    if len(phases) == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, phase in enumerate(phases):
        phase_df = detail_df[detail_df['phase'] == phase]
        
        for row_idx, metric in enumerate(focus_metrics):
            ax = axes[row_idx, col_idx]
            
            layer_var = phase_df.groupby('layer')[metric].var().reset_index()
            layer_var.columns = ['layer', 'variance']
            layer_var = layer_var.sort_values('layer')
            
            colors = []
            for v in layer_var['variance']:
                if v > 5.0:
                    colors.append("#E63946")
                elif v > 2.0:
                    colors.append("#FF9F1C")
                elif v > 0.5:
                    colors.append("#FFD166")
                else:
                    colors.append("#2EC4B6")
            
            bars = ax.bar(layer_var['layer'].astype(str), layer_var['variance'],
                         color=colors, edgecolor=COLORS["grid"], linewidth=0.8)
            
            for bar, v in zip(bars, layer_var['variance']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{v:.2f}", ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color=COLORS["text"])
            
            ax.set_title(f"{METRIC_LABELS_SHORT[metric]} ({phase.capitalize()})",
                        fontsize=12, fontweight='bold', color=METRIC_COLORS[metric])
            ax.set_xlabel("Layer" if row_idx == len(focus_metrics) - 1 else "")
            if col_idx == 0:
                ax.set_ylabel("Variance", fontsize=10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.suptitle("Per-Layer Head Variance — Which Layers Have Inconsistent Heads?",
                 fontsize=16, fontweight='bold', color=COLORS["text"], y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_head_variance.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


def plot_per_head_trend_lines(detail_df, output_dir):
    """
    Within each phase, show per-head metric values across layers as trend lines.
    Each head is a separate line, color-coded.
    """
    phases = [p for p in ["prefill", "generation"] if not detail_df[detail_df['phase'] == p].empty]
    focus_metrics = ["global_lir", "cosine", "amr"]
    
    for metric in focus_metrics:
        fig, axes = plt.subplots(1, len(phases), figsize=(10 * len(phases), 6), sharey=True)
        if len(phases) == 1:
            axes = [axes]
        
        for ax_idx, phase in enumerate(phases):
            ax = axes[ax_idx]
            phase_df = detail_df[detail_df['phase'] == phase].sort_values(by=['layer', 'head'])
            
            heads = sorted(phase_df['head'].unique())
            cmap = plt.cm.Set2
            
            for hi, head in enumerate(heads):
                head_df = phase_df[phase_df['head'] == head].sort_values('layer')
                color = cmap(hi / max(1, len(heads) - 1))
                ax.plot(head_df['layer'], head_df[metric], '-o', color=color,
                        linewidth=2, markersize=6, alpha=0.85, label=f"Head {head}")
            
            ax.set_title(f"{METRIC_LABELS_SHORT[metric]} ({phase.capitalize()})",
                        fontsize=14, fontweight='bold', color=COLORS["accent"])
            ax.set_xlabel("Transformer Layer", fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel(f"{METRIC_LABELS_SHORT[metric]} (%)", fontsize=11)
            ax.set_ylim(-5, 105)
            ax.set_xticks(sorted(phase_df['layer'].unique()))
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='lower left', fontsize=8, framealpha=0.5, ncol=2)
        
        fig.suptitle(f"Per-Head {metric.upper()} Through Depth",
                     fontsize=16, fontweight='bold', color=COLORS["text"], y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, f"lir_per_head_trend_{metric}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
        print(f"  Saved {path}")
        plt.close()


def plot_divergence_summary(detail_df, output_dir):
    """
    Auto-identify and display the top-N most divergent (layer, head) pairs.
    Uses global_lir as the primary divergence signal.
    """
    phases = [p for p in ["prefill", "generation"] if not detail_df[detail_df['phase'] == p].empty]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    
    y_pos = 0.95
    
    ax.text(0.5, y_pos, "LIR DIVERGENCE SUMMARY — TOP-10 WORST (LAYER, HEAD) PAIRS",
            ha='center', va='top', fontsize=18, fontweight='bold',
            color=COLORS["accent"], transform=ax.transAxes)
    y_pos -= 0.06
    
    for phase in phases:
        phase_df = detail_df[detail_df['phase'] == phase]
        
        color = "#2EC4B6" if phase == "prefill" else "#FF6B6B"
        
        ax.text(0.5, y_pos, f"══════ {phase.upper()} ══════",
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=color, transform=ax.transAxes)
        y_pos -= 0.05
        
        # Overall stats
        overall_glir = phase_df['global_lir'].mean()
        overall_amr = phase_df['amr'].mean()
        overall_cos = phase_df['cosine'].mean()
        
        ax.text(0.5, y_pos,
                f"Overall — Global LIR: {overall_glir:.1f}%  |  AMR: {overall_amr:.1f}%  |  Cosine: {overall_cos:.1f}%",
                ha='center', va='top', fontsize=11, color=COLORS["text"],
                transform=ax.transAxes, family='monospace')
        y_pos -= 0.04
        
        # Worst offenders by global_lir
        worst = phase_df.nsmallest(NUM_WORST, 'global_lir')
        
        header = f"{'Rank':<5} {'L/H':<8} {'GlobLIR':>10} {'AMR':>10} {'Cosine':>10} {'Missed':>10} {'Sparsity':>10}"
        ax.text(0.5, y_pos, header,
                ha='center', va='top', fontsize=10, color=COLORS["accent"],
                transform=ax.transAxes, family='monospace', fontweight='bold')
        y_pos -= 0.03
        
        for rank, (_, row) in enumerate(worst.iterrows(), 1):
            lh = f"L{int(row['layer'])}H{int(row['head'])}"
            line = f"{rank:<5} {lh:<8} {row['global_lir']:>9.1f}% {row['amr']:>9.1f}% {row['cosine']:>9.1f}% {row['missed']:>9.1f}% {row['sparsity']:>9.1f}%"
            
            # Color by severity
            if row['global_lir'] < 60:
                line_color = "#E63946"
            elif row['global_lir'] < 80:
                line_color = "#FF9F1C"
            else:
                line_color = COLORS["text"]
            
            ax.text(0.5, y_pos, line,
                    ha='center', va='top', fontsize=10, color=line_color,
                    transform=ax.transAxes, family='monospace')
            y_pos -= 0.025
        
        y_pos -= 0.03
    
    # Delta if both phases present
    if "prefill" in phases and "generation" in phases:
        p_glir = detail_df[detail_df['phase'] == 'prefill']['global_lir'].mean()
        g_glir = detail_df[detail_df['phase'] == 'generation']['global_lir'].mean()
        delta = p_glir - g_glir
        delta_color = "#E63946" if delta > 5 else "#FF9F1C" if delta > 1 else "#2EC4B6"
        ax.text(0.5, y_pos,
                f"Δ Global LIR (Prefill − Generation): {delta:+.1f}%",
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=delta_color, transform=ax.transAxes)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "lir_divergence_summary.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    setup_dark_style()
    
    parser = argparse.ArgumentParser(description="Visualize LIR Results (Enhanced Edition)")
    parser.add_argument("--input", type=str, default=INPUT_PATH, 
                        help="Path to lir_comparison.json")
    parser.add_argument("--detailed_input", type=str, default=None,
                        help="Path to lir_comparison_detailed.json (auto-detected if not set)")
    parser.add_argument("--output_dir", type=str, default="./LIR", 
                        help="Directory to save plots")
    args, unknown = parser.parse_known_args()

    filepath = args.input
    output_dir = args.output_dir

    # Kaggle fallback
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
    if df is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy source JSON for reference
    lir_json_copy = os.path.join(output_dir, "lir_results.json")
    if os.path.abspath(filepath) != os.path.abspath(lir_json_copy):
        shutil.copy2(filepath, lir_json_copy)
        print(f"Copied source JSON to {lir_json_copy}")
    
    print("\n🎨 Generating LIR Visualizations (Enhanced Edition)...\n")
    
    # ── Original Aggregate Plots ──
    print("  [1/5] Radar Chart — Metric Overview")
    plot_radar_comparison(df, output_dir)
    
    print("  [2/5] Gradient Heatmap — Intensity Matrix")
    plot_gradient_heatmap(df, output_dir)
    
    print("  [3/5] Trend Lines — Depth Evolution")
    plot_trend_with_fill(df, output_dir)
    
    print("  [4/5] Global LIR Gauge — Eviction Health")
    plot_global_lir_gauge(df, output_dir)
    
    print("  [5/5] Stacked Area — Attention Mass Budget")
    plot_stacked_retention(df, output_dir)
    
    # ── New Per-Head Plots (require detailed JSON) ──
    detailed_path = args.detailed_input
    if detailed_path is None:
        # Auto-detect from aggregate path
        detailed_path = filepath.replace(".json", "_detailed.json")
    
    # Kaggle fallback for detailed
    if not os.path.exists(detailed_path):
        kaggle_detail = glob.glob("/kaggle/input/**/lir_comparison_detailed.json", recursive=True)
        if kaggle_detail:
            detailed_path = kaggle_detail[0]
    
    detail_df = load_detailed_data(detailed_path)
    
    if detail_df is not None and not detail_df.empty:
        print("\n  ── Per-Head Divergence Analysis ──\n")
        
        print("  [6] Per-Head Heatmaps (AMR, Cosine, Global LIR)")
        plot_per_head_heatmaps(detail_df, output_dir)
        
        print("  [7] Prefill vs Generation Delta Heatmaps")
        plot_delta_heatmaps(detail_df, output_dir)
        
        print("  [8] Head Variance Per Layer")
        plot_head_variance_per_layer(detail_df, output_dir)
        
        print("  [9] Per-Head Trend Lines")
        plot_per_head_trend_lines(detail_df, output_dir)
        
        print("  [10] Divergence Summary Panel")
        plot_divergence_summary(detail_df, output_dir)
    else:
        print("\n  ⚠ Per-head detailed data not found. Run calculate_layer_information_retention.py")
        print("    to generate lir_comparison_detailed.json, then re-run this script.")
    
    print(f"\n✅ All LIR visualizations saved to: {output_dir}/")
    print(f"   └── lir_results.json (source data)")
    print(f"   └── lir_radar.png")
    print(f"   └── lir_heatmap.png")
    print(f"   └── lir_trends.png")
    print(f"   └── lir_global_gauge.png")
    print(f"   └── lir_attention_budget.png")
    if detail_df is not None and not detail_df.empty:
        print(f"   └── lir_per_head_amr.png")
        print(f"   └── lir_per_head_cosine.png")
        print(f"   └── lir_per_head_global_lir.png")
        print(f"   └── lir_delta_heatmap.png")
        print(f"   └── lir_head_variance.png")
        print(f"   └── lir_per_head_trend_global_lir.png")
        print(f"   └── lir_per_head_trend_cosine.png")
        print(f"   └── lir_per_head_trend_amr.png")
        print(f"   └── lir_divergence_summary.png")

if __name__ == "__main__":
    main()
