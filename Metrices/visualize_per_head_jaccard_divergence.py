"""
Per-Head Jaccard Similarity Divergence Line Graphs
===================================================
For each tracked layer, produces a subplot with one line per KV head
showing Jaccard similarity at every generation step.

Reads: detailed_jaccard_results.json  (produced by calculate_window_jaccard.py)
Saves: per_head_jaccard_divergence.png  — main faceted figure
       per_head_jaccard_individual/      — one standalone PNG per layer
"""

import json
import os
import argparse
import shutil
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ═══════════════════════════════════════════════════════════
# Premium Color Palette (consistent with existing visualizers)
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

# Curated per-head palette — 8 distinct, vibrant, dark-mode-friendly colors
HEAD_PALETTE = [
    "#F94144",  # Red
    "#F3722C",  # Orange
    "#F9C74F",  # Yellow-Gold
    "#90BE6D",  # Green
    "#43AA8B",  # Teal-Green
    "#4CC9F0",  # Cyan
    "#577590",  # Slate-Blue
    "#9B5DE5",  # Purple
]

INPUT_PATH = "detailed_jaccard_results.json"


def setup_dark_style():
    """Apply a sleek dark research-grade style matching the existing visualizers."""
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
    Loads detailed Jaccard results JSON and flattens into a DataFrame.
    Returns a DataFrame with columns: [sample, layer, head, phase, step, similarity]
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run calculate_window_jaccard.py first.")
        return None

    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)

    records = []
    for sample in data:
        sample_idx = sample.get("sample_index", 0)
        layers = sample.get("layers", {})

        for layer_str, layer_data in layers.items():
            layer_idx = int(layer_str)
            heads = layer_data.get("heads", {})

            for head_str, head_data in heads.items():
                head_idx = int(head_str)

                for p_step in head_data.get("prefill_steps", []):
                    records.append({
                        "sample": sample_idx,
                        "layer": layer_idx,
                        "head": head_idx,
                        "phase": "prefill",
                        "step": p_step["step"],
                        "similarity": p_step["jaccard_similarity"],
                    })

                for g_step in head_data.get("generation_steps", []):
                    records.append({
                        "sample": sample_idx,
                        "layer": layer_idx,
                        "head": head_idx,
                        "phase": "generation",
                        "step": g_step["step"],
                        "similarity": g_step["jaccard_similarity"],
                    })

    df = pd.DataFrame(records)
    print(f"DataFrame created with {len(df):,} records "
          f"({df['layer'].nunique()} layers, {df['head'].nunique()} heads, "
          f"{df['sample'].nunique()} samples).")
    return df


# ═══════════════════════════════════════════════════════════
# MAIN FIGURE: Faceted per-layer, per-head line graph
# ═══════════════════════════════════════════════════════════
def plot_per_head_divergence_faceted(df, output_dir):
    """
    Creates a single large figure with one subplot per tracked layer.
    Each subplot has 8 lines (one per KV head) plus a thick dashed mean line.
    X-axis = generation step, Y-axis = Jaccard similarity.
    Uses sample-averaged values at each step.
    """
    gen_df = df[df["phase"] == "generation"]
    if gen_df.empty:
        print("  No generation data found. Skipping faceted plot.")
        return

    layers_sorted = sorted(gen_df["layer"].unique())
    n_layers = len(layers_sorted)
    heads_sorted = sorted(gen_df["head"].unique())
    n_heads = len(heads_sorted)

    # Layout: 2 columns, ceil(n_layers/2) rows
    n_cols = 2
    n_rows = int(np.ceil(n_layers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(9 * n_cols, 5 * n_rows),
                             sharex=False, sharey=True)
    axes_flat = np.array(axes).flatten() if n_layers > 1 else [axes]

    for idx, layer in enumerate(layers_sorted):
        ax = axes_flat[idx]
        layer_df = gen_df[gen_df["layer"] == layer]

        # Per-head lines (sample-averaged at each step)
        all_steps = sorted(layer_df["step"].unique())
        mean_across_heads = np.zeros(len(all_steps))

        for h_idx, head in enumerate(heads_sorted):
            head_df = layer_df[layer_df["head"] == head]
            avg_per_step = head_df.groupby("step")["similarity"].mean()

            color = HEAD_PALETTE[h_idx % len(HEAD_PALETTE)]
            ax.plot(avg_per_step.index, avg_per_step.values, '-',
                    color=color, linewidth=1.6, alpha=0.85,
                    label=f"Head {head}")


            # Accumulate for mean line
            for si, s in enumerate(all_steps):
                if s in avg_per_step.index:
                    mean_across_heads[si] += avg_per_step[s]

        # Mean across heads
        mean_across_heads /= max(1, n_heads)
        ax.plot(all_steps, mean_across_heads, '--',
                color="white", linewidth=2.4, alpha=0.9,
                label="Mean (all heads)",
                path_effects=[pe.Stroke(linewidth=4, foreground=COLORS["bg_dark"]),
                              pe.Normal()])

        # Threshold line at 0.9
        ax.axhline(y=0.9, color=COLORS["highlight"], linestyle=':', alpha=0.35,
                   linewidth=1.0)
        ax.axhline(y=1.0, color=COLORS["subtle"], linestyle='-', alpha=0.15,
                   linewidth=0.6)

        # Per-layer annotation: min Jaccard
        min_sim = layer_df.groupby("step")["similarity"].mean().min()
        ax.annotate(f"min={min_sim:.3f}",
                    xy=(0.98, 0.04), xycoords='axes fraction',
                    fontsize=9, color=COLORS["diverge_lo"] if min_sim < 0.9 else COLORS["diverge_hi"],
                    fontweight='bold', ha='right', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=COLORS["bg_dark"],
                              edgecolor=COLORS["grid"], alpha=0.9))

        ax.set_title(f"Layer {layer}", fontsize=13, fontweight='bold',
                     color=COLORS["accent"], pad=8)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel("Generation Step", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Jaccard Similarity", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Tick formatting
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    # Hide empty subplots if n_layers is odd
    for idx in range(n_layers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend at bottom
    head_handles = [Line2D([0], [0], color=HEAD_PALETTE[i], linewidth=2.0,
                           label=f"Head {heads_sorted[i]}")
                    for i in range(n_heads)]
    head_handles.append(Line2D([0], [0], color="white", linewidth=2.4,
                               linestyle='--', label="Mean (all heads)"))
    head_handles.append(Line2D([0], [0], color=COLORS["highlight"],
                               linewidth=1.0, linestyle=':', alpha=0.6,
                               label="0.9 threshold"))

    fig.legend(handles=head_handles, loc='lower center',
               ncol=min(n_heads + 2, 5), fontsize=10,
               framealpha=0.6, edgecolor=COLORS["grid"],
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-Head Jaccard Similarity Divergence Across Generation Steps",
                 fontsize=18, fontweight='bold', color=COLORS["text"],
                 y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = os.path.join(output_dir, "per_head_jaccard_divergence.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# INDIVIDUAL LAYER FIGURES: One standalone PNG per layer
# ═══════════════════════════════════════════════════════════
def plot_per_head_divergence_individual(df, output_dir):
    """
    Saves one polished PNG per layer, each containing per-head Jaccard lines.
    Useful for zoomed-in analysis of individual layers.
    """
    gen_df = df[df["phase"] == "generation"]
    if gen_df.empty:
        print("  No generation data found. Skipping individual layer plots.")
        return

    subdir = os.path.join(output_dir, "per_head_jaccard_individual")
    os.makedirs(subdir, exist_ok=True)

    layers_sorted = sorted(gen_df["layer"].unique())
    heads_sorted = sorted(gen_df["head"].unique())
    n_heads = len(heads_sorted)
    n_samples = gen_df["sample"].nunique()

    for layer in layers_sorted:
        layer_df = gen_df[gen_df["layer"] == layer]
        all_steps = sorted(layer_df["step"].unique())

        fig, ax = plt.subplots(figsize=(14, 6))

        mean_across_heads = np.zeros(len(all_steps))

        for h_idx, head in enumerate(heads_sorted):
            head_df = layer_df[layer_df["head"] == head]
            avg_per_step = head_df.groupby("step")["similarity"].mean()
            std_per_step = head_df.groupby("step")["similarity"].std().fillna(0)

            color = HEAD_PALETTE[h_idx % len(HEAD_PALETTE)]

            # Main line
            ax.plot(avg_per_step.index, avg_per_step.values, '-',
                    color=color, linewidth=2.0, alpha=0.9,
                    label=f"Head {head}")


            # Mark first divergence (first step where Jaccard < 1.0)
            below_1 = avg_per_step[avg_per_step < 0.999]
            if not below_1.empty:
                first_step = below_1.index[0]
                first_val = below_1.iloc[0]
                ax.scatter([first_step], [first_val], s=60, color=color,
                           zorder=5, edgecolors="white", linewidths=1.2)

            # Accumulate for mean
            for si, s in enumerate(all_steps):
                if s in avg_per_step.index:
                    mean_across_heads[si] += avg_per_step[s]

        # Mean line
        mean_across_heads /= max(1, n_heads)
        ax.plot(all_steps, mean_across_heads, '--',
                color="white", linewidth=2.8, alpha=0.9,
                label="Mean (all heads)",
                path_effects=[pe.Stroke(linewidth=5, foreground=COLORS["bg_dark"]),
                              pe.Normal()])

        # Threshold markers
        ax.axhline(y=0.9, color=COLORS["highlight"], linestyle=':', alpha=0.4,
                   linewidth=1.0, label="0.9 threshold")
        ax.axhline(y=1.0, color=COLORS["subtle"], linestyle='-', alpha=0.15,
                   linewidth=0.6)

        # Stats annotation box
        overall_mean = layer_df["similarity"].mean()
        overall_min = layer_df.groupby("step")["similarity"].mean().min()
        overall_std = layer_df["similarity"].std()

        stats_text = (f"μ = {overall_mean:.4f}\n"
                      f"σ = {overall_std:.4f}\n"
                      f"min = {overall_min:.4f}")
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                fontweight='bold',
                color=COLORS["text"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["bg_dark"],
                          edgecolor=COLORS["accent"], alpha=0.9, linewidth=1.5))

        ax.set_title(f"Layer {layer} — Per-Head Jaccard Similarity Over Generation",
                     fontsize=15, fontweight='bold', color=COLORS["accent"], pad=12)
        ax.set_xlabel("Generation Step", fontsize=12)
        ax.set_ylabel("Jaccard Similarity", fontsize=12)
        ax.set_ylim(-0.03, 1.06)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        ax.legend(loc='lower left', fontsize=9, framealpha=0.6,
                  edgecolor=COLORS["grid"], ncol=3)

        plt.tight_layout()
        path = os.path.join(subdir, f"jaccard_layer_{layer}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
        plt.close()

    print(f"  Saved {len(layers_sorted)} individual layer plots to {subdir}/")


# ═══════════════════════════════════════════════════════════
# OVERLAY FIGURE: All layers on one plot (head-averaged)
# ═══════════════════════════════════════════════════════════
def plot_layer_overlay(df, output_dir):
    """
    All layers overlaid on a single plot, each layer showing its head-averaged
    Jaccard trajectory. Lets you compare cross-layer behaviour at a glance.
    """
    gen_df = df[df["phase"] == "generation"]
    if gen_df.empty:
        print("  No generation data found. Skipping overlay plot.")
        return

    layers_sorted = sorted(gen_df["layer"].unique())
    n_layers = len(layers_sorted)

    # Use a separate gradient palette for layer distinction
    layer_cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(14, 6))

    for li, layer in enumerate(layers_sorted):
        layer_df = gen_df[gen_df["layer"] == layer]
        avg_per_step = layer_df.groupby("step")["similarity"].mean()
        std_per_step = layer_df.groupby("step")["similarity"].std().fillna(0)

        color = layer_cmap(li / max(1, n_layers - 1))

        ax.plot(avg_per_step.index, avg_per_step.values, '-',
                color=color, linewidth=2.2, alpha=0.9,
                label=f"Layer {layer}")

        # (no fill_between — clean lines only)

    ax.axhline(y=0.9, color=COLORS["highlight"], linestyle=':', alpha=0.4,
               linewidth=1.0, label="0.9 threshold")

    ax.set_title("Head-Averaged Jaccard Similarity — All Layers Overlaid",
                 fontsize=16, fontweight='bold', color=COLORS["accent"], pad=14)
    ax.set_xlabel("Generation Step", fontsize=12)
    ax.set_ylabel("Jaccard Similarity (head-averaged)", fontsize=12)
    ax.set_ylim(-0.03, 1.06)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(loc='lower left', fontsize=10, framealpha=0.6,
              edgecolor=COLORS["grid"], ncol=2)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_head_jaccard_layer_overlay.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# FULL GRID: Every layer × every head — clean lines, no shadows
# ═══════════════════════════════════════════════════════════
def plot_full_layer_head_grid(df, output_dir):
    """
    Faceted grid: one subplot per layer, each showing one clean line per head.
    No shadows, no fill_between — pure lines showing per-head behaviour.
    Also produces an overlay of ALL layer×head combos on a single plot.
    """
    gen_df = df[df["phase"] == "generation"]
    if gen_df.empty:
        print("  No generation data found. Skipping full layer×head grid.")
        return

    layers_sorted = sorted(gen_df["layer"].unique())
    heads_sorted = sorted(gen_df["head"].unique())
    n_layers = len(layers_sorted)
    n_heads = len(heads_sorted)

    # --- PART A: One subplot per layer, each head a clean line ---
    n_cols = 2
    n_rows = int(np.ceil(n_layers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(9 * n_cols, 4.5 * n_rows),
                             sharex=False, sharey=True)
    axes_flat = np.array(axes).flatten() if n_layers > 1 else [axes]

    for idx, layer in enumerate(layers_sorted):
        ax = axes_flat[idx]
        layer_df = gen_df[gen_df["layer"] == layer]

        for h_idx, head in enumerate(heads_sorted):
            head_df = layer_df[layer_df["head"] == head]
            avg_per_step = head_df.groupby("step")["similarity"].mean()

            color = HEAD_PALETTE[h_idx % len(HEAD_PALETTE)]
            ax.plot(avg_per_step.index, avg_per_step.values, '-',
                    color=color, linewidth=1.5, alpha=0.9,
                    label=f"Head {head}")

        ax.axhline(y=0.9, color=COLORS["highlight"], linestyle=':', alpha=0.35,
                   linewidth=1.0)

        ax.set_title(f"Layer {layer}", fontsize=13, fontweight='bold',
                     color=COLORS["accent"], pad=8)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel("Generation Step", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Jaccard Similarity", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    for idx in range(n_layers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    head_handles = [Line2D([0], [0], color=HEAD_PALETTE[i % len(HEAD_PALETTE)],
                           linewidth=2.0, label=f"Head {heads_sorted[i]}")
                    for i in range(n_heads)]
    head_handles.append(Line2D([0], [0], color=COLORS["highlight"],
                               linewidth=1.0, linestyle=':', alpha=0.6,
                               label="0.9 threshold"))
    fig.legend(handles=head_handles, loc='lower center',
               ncol=min(n_heads + 1, 6), fontsize=10,
               framealpha=0.6, edgecolor=COLORS["grid"],
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-Head Jaccard Similarity — All Layers (Clean Lines)",
                 fontsize=18, fontweight='bold', color=COLORS["text"],
                 y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = os.path.join(output_dir, "per_head_jaccard_clean_grid.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()

    # --- PART B: All layer×head combos on a single overlay ---
    fig, ax = plt.subplots(figsize=(16, 7))

    # Use a 2D palette: layer sets hue via colormap, head shifts linestyle
    layer_cmap = plt.cm.turbo

    for li, layer in enumerate(layers_sorted):
        layer_df = gen_df[gen_df["layer"] == layer]
        base_color = layer_cmap(li / max(1, n_layers - 1))

        for h_idx, head in enumerate(heads_sorted):
            head_df = layer_df[layer_df["head"] == head]
            avg_per_step = head_df.groupby("step")["similarity"].mean()

            # Vary alpha slightly by head for visual separation
            alpha = 0.5 + 0.5 * (h_idx / max(1, n_heads - 1))
            lw = 1.0 + 0.3 * (h_idx / max(1, n_heads - 1))
            linestyle = ['-', '--', '-.', ':'][h_idx % 4]

            ax.plot(avg_per_step.index, avg_per_step.values,
                    linestyle=linestyle, color=base_color,
                    linewidth=lw, alpha=alpha,
                    label=f"L{layer} H{head}")

    ax.axhline(y=0.9, color=COLORS["highlight"], linestyle=':', alpha=0.4,
               linewidth=1.0)

    ax.set_title("All Layers × All Heads — Jaccard Similarity (Clean Lines)",
                 fontsize=16, fontweight='bold', color=COLORS["accent"], pad=14)
    ax.set_xlabel("Generation Step", fontsize=12)
    ax.set_ylabel("Jaccard Similarity", fontsize=12)
    ax.set_ylim(-0.03, 1.06)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    # Compact legend: layers by color, heads by linestyle
    layer_handles = [Line2D([0], [0], color=layer_cmap(i / max(1, n_layers - 1)),
                            linewidth=2.5, label=f"Layer {layers_sorted[i]}")
                     for i in range(n_layers)]
    head_style_handles = [Line2D([0], [0], color=COLORS["text"],
                                  linestyle=['-', '--', '-.', ':'][i % 4],
                                  linewidth=1.5,
                                  label=f"Head {heads_sorted[i]}")
                          for i in range(n_heads)]

    all_handles = layer_handles + [Line2D([], [], color='none')] + head_style_handles
    ax.legend(handles=all_handles, loc='lower left', fontsize=8,
              framealpha=0.7, edgecolor=COLORS["grid"],
              ncol=max(3, n_layers // 4 + 1))

    plt.tight_layout()
    path = os.path.join(output_dir, "all_layer_head_overlay_clean.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# FIRST-DIVERGENCE HEATMAP: When does each (layer, head) first drop below 1.0?
# ═══════════════════════════════════════════════════════════
def plot_first_divergence_heatmap(df, output_dir):
    """
    Heatmap where each cell is the generation step at which a (layer, head) pair
    first drops below Jaccard=1.0. Cells that never diverge show '—'.
    """
    import seaborn as sns

    gen_df = df[df["phase"] == "generation"]
    if gen_df.empty:
        print("  No generation data found. Skipping first-divergence heatmap.")
        return

    layers_sorted = sorted(gen_df["layer"].unique())
    heads_sorted = sorted(gen_df["head"].unique())

    diverge_matrix = np.full((len(layers_sorted), len(heads_sorted)), np.nan)
    annot_matrix = np.empty((len(layers_sorted), len(heads_sorted)), dtype=object)

    for li, layer in enumerate(layers_sorted):
        for hi, head in enumerate(heads_sorted):
            subset = gen_df[(gen_df["layer"] == layer) & (gen_df["head"] == head)]
            avg = subset.groupby("step")["similarity"].mean().sort_index()
            below = avg[avg < 0.999]
            if not below.empty:
                first_step = below.index[0]
                diverge_matrix[li, hi] = first_step
                annot_matrix[li, hi] = str(int(first_step))
            else:
                diverge_matrix[li, hi] = np.nan
                annot_matrix[li, hi] = "—"

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use a sequential palette (lower step = diverges earlier = worse)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    # Mask NaNs for proper coloring
    masked = np.ma.masked_invalid(diverge_matrix)
    max_step = np.nanmax(diverge_matrix) if not np.all(np.isnan(diverge_matrix)) else 1

    im = ax.imshow(masked, cmap=cmap, aspect='auto',
                   vmin=0, vmax=max_step if max_step > 0 else 1)

    # Annotate cells
    for li in range(len(layers_sorted)):
        for hi in range(len(heads_sorted)):
            text_color = "white" if (not np.isnan(diverge_matrix[li, hi])
                                     and diverge_matrix[li, hi] > max_step * 0.5) else COLORS["text"]
            if annot_matrix[li, hi] == "—":
                text_color = COLORS["diverge_hi"]
            ax.text(hi, li, annot_matrix[li, hi],
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color=text_color)

    ax.set_xticks(range(len(heads_sorted)))
    ax.set_xticklabels([f"Head {h}" for h in heads_sorted], fontsize=10)
    ax.set_yticks(range(len(layers_sorted)))
    ax.set_yticklabels([f"Layer {l}" for l in layers_sorted], fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("First Divergence Step", fontsize=11, color=COLORS["text"])
    cbar.ax.yaxis.set_tick_params(color=COLORS["text"])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS["text"])

    ax.set_title("First Divergence Step — When Does Each (Layer, Head) Drop Below 1.0?",
                 fontsize=14, fontweight='bold', color=COLORS["accent"], pad=14)
    ax.set_xlabel("KV Head Index", fontsize=12)
    ax.set_ylabel("Transformer Layer", fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_head_first_divergence_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=COLORS["bg_dark"])
    print(f"  Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
def main():
    setup_dark_style()

    parser = argparse.ArgumentParser(
        description="Per-Head Jaccard Similarity Divergence Visualizer")
    parser.add_argument("--input", type=str, default=INPUT_PATH,
                        help="Path to detailed_jaccard_results.json")
    parser.add_argument("--output_dir", type=str, default="./Cummulative_Jaccard",
                        help="Directory to save the plots")
    args, _ = parser.parse_known_args()

    filepath = args.input
    output_dir = args.output_dir

    # Kaggle fallback
    if not os.path.exists(filepath):
        print(f"Local file {filepath} not found. Searching Kaggle input paths...")
        kaggle_paths = glob.glob("/kaggle/input/**/detailed_jaccard_results.json",
                                 recursive=True)
        if kaggle_paths:
            filepath = kaggle_paths[0]
            print(f"Found input file at: {filepath}")
        else:
            print("No data file found. Exiting.")
            return

    df = load_and_prepare_data(filepath)
    if df is None:
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing output folder: {output_dir}")
    os.makedirs(output_dir)

    print("\n🎨 Generating Per-Head Jaccard Divergence Visualizations...\n")

    print("  [1/5] Faceted per-layer, per-head divergence lines")
    plot_per_head_divergence_faceted(df, output_dir)

    print("  [2/5] Individual layer PNGs (one per tracked layer)")
    plot_per_head_divergence_individual(df, output_dir)

    print("  [3/5] All-layers overlay (head-averaged, clean lines)")
    plot_layer_overlay(df, output_dir)

    print("  [4/5] Full layer×head grid + overlay (clean lines)")
    plot_full_layer_head_grid(df, output_dir)

    print("  [5/5] First-divergence step heatmap")
    plot_first_divergence_heatmap(df, output_dir)

    print(f"\n✅ All per-head divergence plots saved to: {output_dir}/")
    print(f"   └── per_head_jaccard_divergence.png         (faceted, no shadows)")
    print(f"   └── per_head_jaccard_layer_overlay.png      (overlay, clean lines)")
    print(f"   └── per_head_jaccard_clean_grid.png         (per-head per-layer grid)")
    print(f"   └── all_layer_head_overlay_clean.png        (every L×H on one plot)")
    print(f"   └── per_head_first_divergence_heatmap.png   (heatmap)")
    print(f"   └── per_head_jaccard_individual/             (one per layer)")


if __name__ == "__main__":
    main()
