"""
Script 8: Comprehensive Visualization of All Experiment Results
================================================================
Generates 7 publication-ready charts from BirdNET experiment results:

  1. Accuracy comparison bar chart (all experiments)
  2. Data scaling curve (samples vs accuracy)
  3. Confidence distribution violin plots
  4. Per-species accuracy heatmap (species x experiment)
  5. Confusion matrix heatmap (best model — Exp2)
  6. Exp1 vs Exp2 per-species delta chart
  7. Confidence calibration box plots

Ground truth is derived from the folder structure:
  results/<experiment>/<species_folder>/<file>.BirdNET.results.csv
  → true species = <species_folder>

Usage:
    python scripts/08_visualize_all.py
    python scripts/08_visualize_all.py --output_dir results/figures
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import RESULTS_DIR, SPECIES_NAMES, SPECIES_COMMON_NAMES

# ============================================================
# Constants
# ============================================================
EXPERIMENTS = {
    "Exp1_NoNoise":    "Exp1 (No Noise)",
    "Exp2_WithNoise":  "Exp2 (With Noise)",
    "Exp3_FewShot_10": "FS-10",
    "Exp3_FewShot_25": "FS-25",
    "Exp3_FewShot_50": "FS-50",
}

# Display order for charts (full models first, then few-shot ascending)
DISPLAY_ORDER = [
    "Exp1_NoNoise", "Exp2_WithNoise",
    "Exp3_FewShot_10", "Exp3_FewShot_25", "Exp3_FewShot_50",
]

# Color palette
COLORS = {
    "Exp1_NoNoise":    "#3498db",
    "Exp2_WithNoise":  "#e74c3c",
    "Exp3_FewShot_10": "#f39c12",
    "Exp3_FewShot_25": "#9b59b6",
    "Exp3_FewShot_50": "#2ecc71",
}

MYSTERY_FOLDER = "Mystery mystery"

# ============================================================
# Data Loading
# ============================================================

def load_experiment(exp_dir: Path) -> pd.DataFrame:
    """
    Load all BirdNET result CSVs for one experiment.
    Adds 'true_species' column from the parent folder name.
    Skips Mystery folder and non-species files.
    """
    rows = []
    for species_dir in sorted(exp_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        if species_dir.name == MYSTERY_FOLDER:
            continue
        if species_dir.name not in SPECIES_NAMES:
            continue

        true_species = species_dir.name
        for csv_file in species_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df["true_species"] = true_species
                df["source_file"] = csv_file.stem
                rows.append(df)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    # Standardize column names
    col_map = {}
    for col in combined.columns:
        lower = col.lower().strip()
        if "scientific" in lower:
            col_map[col] = "predicted_species"
        elif "confidence" in lower:
            col_map[col] = "confidence"
    combined.rename(columns=col_map, inplace=True)

    return combined


def load_all_experiments(results_dir: Path) -> dict:
    """Load all experiments into a dict of DataFrames."""
    data = {}
    for exp_key in DISPLAY_ORDER:
        exp_path = results_dir / exp_key
        if exp_path.is_dir():
            print(f"  Loading {exp_key}...")
            df = load_experiment(exp_path)
            if not df.empty:
                data[exp_key] = df
                print(f"    {len(df)} detections, {df['true_species'].nunique()} species")
    return data


# ============================================================
# Metrics Computation
# ============================================================

def compute_per_species_accuracy(df: pd.DataFrame) -> dict:
    """Compute accuracy per species. Returns {species: (correct, total, accuracy)}."""
    results = {}
    for species in SPECIES_NAMES:
        mask = df["true_species"] == species
        subset = df[mask]
        if len(subset) == 0:
            results[species] = (0, 0, 0.0)
            continue
        correct = (subset["predicted_species"] == species).sum()
        total = len(subset)
        results[species] = (int(correct), int(total), correct / total * 100)
    return results


def compute_overall_accuracy(df: pd.DataFrame) -> float:
    """Compute overall accuracy across all species detections."""
    species_mask = df["true_species"].isin(SPECIES_NAMES)
    subset = df[species_mask]
    if len(subset) == 0:
        return 0.0
    correct = (subset["predicted_species"] == subset["true_species"]).sum()
    return correct / len(subset) * 100


def build_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build confusion matrix (true x predicted) for known species."""
    species = SPECIES_NAMES
    matrix = pd.DataFrame(0, index=species, columns=species)

    species_set = set(species)
    for _, row in df.iterrows():
        true = row["true_species"]
        pred = row["predicted_species"]
        if true in species_set and pred in species_set:
            matrix.loc[true, pred] += 1

    return matrix


# ============================================================
# Chart 1: Accuracy Comparison Bar Chart
# ============================================================

def chart1_accuracy_comparison(exp_data: dict, output_dir: Path):
    """Bar chart comparing overall accuracy across all experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    accuracies = []
    colors = []

    for exp_key in DISPLAY_ORDER:
        if exp_key not in exp_data:
            continue
        acc = compute_overall_accuracy(exp_data[exp_key])
        names.append(EXPERIMENTS[exp_key])
        accuracies.append(acc)
        colors.append(COLORS[exp_key])

    bars = ax.bar(names, accuracies, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy Across Experiments", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, label="Random baseline (50%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)

    path = output_dir / "chart1_accuracy_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Chart 2: Data Scaling Curve
# ============================================================

def chart2_scaling_curve(exp_data: dict, output_dir: Path):
    """Line plot: training samples vs accuracy."""
    # Approximate samples per species for each experiment
    sample_points = {
        "Exp3_FewShot_10": 10,
        "Exp3_FewShot_25": 25,
        "Exp3_FewShot_50": 50,
        "Exp2_WithNoise":  240,  # ~7207 / 30 species
    }

    x_vals = []
    y_vals = []
    labels = []

    for exp_key in ["Exp3_FewShot_10", "Exp3_FewShot_25", "Exp3_FewShot_50", "Exp2_WithNoise"]:
        if exp_key not in exp_data:
            continue
        acc = compute_overall_accuracy(exp_data[exp_key])
        x_vals.append(sample_points[exp_key])
        y_vals.append(acc)
        labels.append(EXPERIMENTS[exp_key])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(x_vals, y_vals, "o-", color="#e74c3c", linewidth=2.5, markersize=10, zorder=5)

    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(f"{label}\n{y:.1f}%", (x, y),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Training Samples per Species", fontsize=12)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax.set_title("Data Scaling Curve: How Much Training Data Do You Need?", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xscale("log")
    ax.set_xticks(x_vals)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.fill_between([0, 50], 0, 100, alpha=0.05, color="red", label="Insufficient data zone")
    ax.fill_between([50, 300], 0, 100, alpha=0.05, color="green", label="Usable range")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "chart2_scaling_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Chart 3: Confidence Distribution Violin Plots
# ============================================================

def chart3_confidence_violins(exp_data: dict, output_dir: Path):
    """Overlaid histograms of confidence distributions across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))

    bins = np.linspace(0, 1, 51)

    for exp_key in DISPLAY_ORDER:
        if exp_key not in exp_data:
            continue
        conf = exp_data[exp_key]["confidence"].dropna().values
        if len(conf) == 0:
            continue
        label = f"{EXPERIMENTS[exp_key]} (med={np.median(conf):.2f}, n={len(conf)})"
        ax.hist(conf, bins=bins, alpha=0.4, color=COLORS[exp_key],
                edgecolor=COLORS[exp_key], linewidth=0.8, label=label, density=True)

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Confidence Score Distributions", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend(loc="upper center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "chart3_confidence_distributions.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Chart 4: Per-Species Accuracy Heatmap
# ============================================================

def chart4_species_accuracy_heatmap(exp_data: dict, output_dir: Path):
    """Heatmap: species (rows) x experiment (cols), color = accuracy %."""
    # Build accuracy matrix
    exp_keys = [k for k in DISPLAY_ORDER if k in exp_data]
    species_list = SPECIES_NAMES

    matrix = np.full((len(species_list), len(exp_keys)), np.nan)

    for col_idx, exp_key in enumerate(exp_keys):
        per_sp = compute_per_species_accuracy(exp_data[exp_key])
        for row_idx, sp in enumerate(species_list):
            _, total, acc = per_sp[sp]
            if total > 0:
                matrix[row_idx, col_idx] = acc

    # Sort species by Exp2 accuracy (or Exp1 if Exp2 missing)
    sort_col = exp_keys.index("Exp2_WithNoise") if "Exp2_WithNoise" in exp_keys else 0
    sort_vals = matrix[:, sort_col].copy()
    sort_vals[np.isnan(sort_vals)] = -1
    sort_order = np.argsort(sort_vals)[::-1]
    matrix = matrix[sort_order]
    sorted_species = [species_list[i] for i in sort_order]

    # Common names for y-axis
    y_labels = [SPECIES_COMMON_NAMES.get(sp, sp) for sp in sorted_species]
    x_labels = [EXPERIMENTS[k] for k in exp_keys]

    fig, ax = plt.subplots(figsize=(10, 12))

    cmap = LinearSegmentedColormap.from_list("accuracy",
                                              ["#d32f2f", "#ff9800", "#fdd835", "#8bc34a", "#2e7d32"])

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                text = "-"
                color = "gray"
            else:
                text = f"{val:.0f}"
                color = "white" if val < 40 or val > 80 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Per-Species Accuracy Across Experiments (%)", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Accuracy (%)")
    cbar.ax.tick_params(labelsize=9)

    path = output_dir / "chart4_species_accuracy_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Chart 5: Confusion Matrix (Exp2)
# ============================================================

def chart5_confusion_matrix(exp_data: dict, output_dir: Path):
    """30x30 confusion matrix heatmap for the best model (Exp2)."""
    target_key = "Exp2_WithNoise"
    if target_key not in exp_data:
        print("  [SKIP] Chart 5: Exp2_WithNoise not found")
        return

    cm = build_confusion_matrix(exp_data[target_key])

    # Normalize rows to percentages
    row_sums = cm.sum(axis=1)
    cm_norm = cm.div(row_sums, axis=0) * 100
    cm_norm = cm_norm.fillna(0)

    # Sort by diagonal (accuracy) descending
    diag = np.diag(cm_norm.values)
    sort_order = np.argsort(diag)[::-1]
    sorted_species = [cm_norm.index[i] for i in sort_order]
    cm_norm = cm_norm.loc[sorted_species, sorted_species]
    cm_raw = cm.loc[sorted_species, sorted_species]

    # Common names
    common_labels = [SPECIES_COMMON_NAMES.get(sp, sp)[:25] for sp in sorted_species]

    fig, ax = plt.subplots(figsize=(16, 14))

    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm.values, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    # Annotate cells with significant values
    for i in range(len(sorted_species)):
        for j in range(len(sorted_species)):
            val = cm_norm.values[i, j]
            raw = cm_raw.values[i, j]
            if val >= 3:  # Only show >= 3%
                color = "white" if val > 60 else "black"
                fontsize = 6 if i != j else 7
                fontweight = "bold" if i == j else "normal"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=fontsize, fontweight=fontweight, color=color)

    ax.set_xticks(range(len(common_labels)))
    ax.set_xticklabels(common_labels, rotation=90, ha="center", fontsize=7)
    ax.set_yticks(range(len(common_labels)))
    ax.set_yticklabels(common_labels, fontsize=7)
    ax.set_xlabel("Predicted Species", fontsize=11)
    ax.set_ylabel("True Species", fontsize=11)
    ax.set_title("Confusion Matrix — Exp2 (With Noise) — Row-Normalized %",
                 fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, label="Row %")
    cbar.ax.tick_params(labelsize=9)

    path = output_dir / "chart5_confusion_matrix_exp2.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Chart 6: Exp1 vs Exp2 Per-Species Delta
# ============================================================

def chart6_exp1_vs_exp2_delta(exp_data: dict, output_dir: Path):
    """Horizontal bar chart showing per-species accuracy change from Exp1 to Exp2."""
    if "Exp1_NoNoise" not in exp_data or "Exp2_WithNoise" not in exp_data:
        print("  [SKIP] Chart 6: Need both Exp1 and Exp2")
        return

    acc1 = compute_per_species_accuracy(exp_data["Exp1_NoNoise"])
    acc2 = compute_per_species_accuracy(exp_data["Exp2_WithNoise"])

    species_deltas = []
    for sp in SPECIES_NAMES:
        _, t1, a1 = acc1[sp]
        _, t2, a2 = acc2[sp]
        if t1 > 0 and t2 > 0:
            species_deltas.append((sp, a2 - a1, a1, a2))

    # Sort by delta
    species_deltas.sort(key=lambda x: x[1])

    species_names = [SPECIES_COMMON_NAMES.get(s[0], s[0]) for s in species_deltas]
    deltas = [s[1] for s in species_deltas]
    colors = ["#2e7d32" if d >= 0 else "#c62828" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 10))

    bars = ax.barh(range(len(species_names)), deltas, color=colors, edgecolor="white", height=0.7)

    for idx, (bar, delta) in enumerate(zip(bars, deltas)):
        x_pos = bar.get_width() + (0.3 if delta >= 0 else -0.3)
        ha = "left" if delta >= 0 else "right"
        ax.text(x_pos, idx, f"{delta:+.1f}pp", va="center", ha=ha, fontsize=8)

    ax.set_yticks(range(len(species_names)))
    ax.set_yticklabels(species_names, fontsize=8)
    ax.set_xlabel("Accuracy Change (percentage points)", fontsize=11)
    ax.set_title("Per-Species Accuracy: Exp2 (Noise) vs Exp1 (No Noise)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    # Summary annotation
    improved = sum(1 for d in deltas if d > 0)
    declined = sum(1 for d in deltas if d < 0)
    unchanged = sum(1 for d in deltas if d == 0)
    ax.text(0.98, 0.02,
            f"Improved: {improved} | Declined: {declined} | Unchanged: {unchanged}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    path = output_dir / "chart6_exp1_vs_exp2_delta.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Chart 7: Confidence Calibration Box Plots
# ============================================================

def chart7_confidence_boxplots(exp_data: dict, output_dir: Path):
    """Side-by-side box plots of confidence per experiment."""
    fig, ax = plt.subplots(figsize=(11, 6))

    plot_data = []
    plot_names = []
    plot_colors = []

    for exp_key in DISPLAY_ORDER:
        if exp_key not in exp_data:
            continue
        conf = exp_data[exp_key]["confidence"].dropna().values
        if len(conf) > 0:
            plot_data.append(conf)
            plot_names.append(EXPERIMENTS[exp_key])
            plot_colors.append(COLORS[exp_key])

    bp = ax.boxplot(plot_data, patch_artist=True, notch=True,
                    widths=0.5, showfliers=False,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    means = [np.mean(d) for d in plot_data]
    ax.scatter(range(1, len(means) + 1), means, marker="D", color="black",
               s=40, zorder=5, label="Mean")

    # Stats text below each box
    for idx, data in enumerate(plot_data):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        ax.text(idx + 1, -0.08,
                f"Q1={q1:.2f}\nQ3={q3:.2f}",
                ha="center", fontsize=7, color="gray")

    ax.set_xticklabels(plot_names, fontsize=10)
    ax.set_ylabel("Confidence Score", fontsize=12)
    ax.set_title("Confidence Calibration Across Experiments", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="0.5 threshold")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    path = output_dir / "chart7_confidence_boxplots.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate all experiment visualizations.")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR / "figures"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE VISUALIZATION")
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print("=" * 60)

    # Load all experiments
    print("\nLoading experiment data...")
    exp_data = load_all_experiments(results_dir)

    if not exp_data:
        print("[ERROR] No experiment data found!")
        return

    print(f"\nLoaded {len(exp_data)} experiments. Generating charts...\n")

    # Generate all 7 charts
    print("--- Chart 1: Accuracy Comparison ---")
    chart1_accuracy_comparison(exp_data, output_dir)

    print("--- Chart 2: Data Scaling Curve ---")
    chart2_scaling_curve(exp_data, output_dir)

    print("--- Chart 3: Confidence Violins ---")
    chart3_confidence_violins(exp_data, output_dir)

    print("--- Chart 4: Species Accuracy Heatmap ---")
    chart4_species_accuracy_heatmap(exp_data, output_dir)

    print("--- Chart 5: Confusion Matrix (Exp2) ---")
    chart5_confusion_matrix(exp_data, output_dir)

    print("--- Chart 6: Exp1 vs Exp2 Delta ---")
    chart6_exp1_vs_exp2_delta(exp_data, output_dir)

    print("--- Chart 7: Confidence Box Plots ---")
    chart7_confidence_boxplots(exp_data, output_dir)

    print(f"\n{'=' * 60}")
    print(f"ALL DONE — {7} charts saved to {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
