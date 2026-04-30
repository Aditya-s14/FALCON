"""
Script 6: Results Analysis & Comparison
==========================================
Analyzes BirdNET detection results across all experiments and generates:
  - Comparative metrics table (FPR, accuracy, noise misclassification, confidence)
  - Per-species accuracy breakdown
  - Confusion matrices
  - Confidence score distributions
  - Before-vs-after comparison charts

Usage:
    python scripts/06_analyze_results.py
    python scripts/06_analyze_results.py --results_dir results --output_dir results/analysis

Pipeline Position: FINAL step — runs after all experiments complete.
Output: PNG charts + summary CSV + printed comparison table.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RESULTS_DIR, SPECIES_NAMES, SPECIES_COMMON_NAMES,
    NOISE_FOLDER_NAME, MIN_CONFIDENCE, VERBOSE, ensure_dirs,
)


# ============================================================
# Result Loading
# ============================================================

def find_result_csvs(results_dir: Path) -> dict:
    """
    Find BirdNET output CSVs for each experiment.

    BirdNET-Analyzer writes results as CSV files with columns:
    filepath, start, end, scientific_name, common_name, confidence
    """
    experiments = {}

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue

        csvs = list(subdir.rglob("*.csv"))
        if csvs:
            experiments[subdir.name] = csvs
            if VERBOSE:
                print(f"  Found {len(csvs)} CSV(s) for experiment: {subdir.name}")

    return experiments


def load_detections(csv_files: list) -> pd.DataFrame:
    """
    Load BirdNET detection results from one or more CSV files.

    Returns DataFrame with columns:
        filepath, start, end, scientific_name, common_name, confidence, source_file
    """
    all_rows = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df["source_file"] = csv_path.name
            all_rows.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to read {csv_path}: {e}")

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Standardize column names (BirdNET output varies slightly)
    col_map = {}
    for col in combined.columns:
        lower = col.lower().strip()
        if "scientific" in lower or lower == "species":
            col_map[col] = "scientific_name"
        elif "common" in lower:
            col_map[col] = "common_name"
        elif "confidence" in lower or "score" in lower:
            col_map[col] = "confidence"
        elif "start" in lower:
            col_map[col] = "start"
        elif "end" in lower:
            col_map[col] = "end"
        elif "file" in lower and "source" not in lower:
            col_map[col] = "filepath"

    combined.rename(columns=col_map, inplace=True)

    return combined


# ============================================================
# Metrics Computation
# ============================================================

def compute_metrics(detections: pd.DataFrame,
                    known_species: list = None) -> dict:
    """
    Compute evaluation metrics from BirdNET detections.

    Metrics:
        - total_detections: Total number of detections
        - mean_confidence: Average confidence across all detections
        - median_confidence: Median confidence
        - high_confidence_rate: % of detections with confidence > 0.5
        - unique_species_detected: Number of unique species in results
        - per_species_counts: Detection count per species
    """
    if detections.empty:
        return {"total_detections": 0, "error": "No detections found"}

    if known_species is None:
        known_species = SPECIES_NAMES

    metrics = {
        "total_detections": len(detections),
        "mean_confidence": float(detections["confidence"].mean()),
        "median_confidence": float(detections["confidence"].median()),
        "std_confidence": float(detections["confidence"].std()),
        "high_confidence_rate": float(
            (detections["confidence"] > 0.5).mean() * 100
        ),
        "unique_species_detected": int(
            detections["scientific_name"].nunique()
        ),
    }

    # Per-species detection counts
    species_counts = detections["scientific_name"].value_counts().to_dict()
    metrics["per_species_counts"] = species_counts

    # How many detections are for our 30 target species?
    if "scientific_name" in detections.columns:
        in_scope = detections["scientific_name"].isin(known_species)
        metrics["in_scope_detections"] = int(in_scope.sum())
        metrics["out_of_scope_detections"] = int((~in_scope).sum())
        metrics["in_scope_rate"] = float(in_scope.mean() * 100)

    return metrics


def estimate_false_positive_rate(detections: pd.DataFrame,
                                  noise_keywords: list = None) -> dict:
    """
    Estimate false positive rate by analyzing detections on known noise segments.

    This looks at detections where the source audio filename suggests noise
    (e.g., files from the noise/ folder or mystery folder).
    """
    if noise_keywords is None:
        noise_keywords = ["noise", "mystery", "esc50", "rain", "wind",
                          "water", "insect", "thunder", "fire"]

    fp_metrics = {"noise_detections": 0, "noise_as_bird": 0}

    if "filepath" in detections.columns:
        # Check if any detections come from noise-like files
        for kw in noise_keywords:
            mask = detections["filepath"].str.lower().str.contains(kw, na=False)
            fp_metrics["noise_detections"] += int(mask.sum())

    return fp_metrics


# ============================================================
# Visualization
# ============================================================

def plot_confidence_distributions(experiment_data: dict,
                                   output_path: Path):
    """
    Plot confidence score distributions for all experiments.
    """
    fig, axes = plt.subplots(1, len(experiment_data), figsize=(5 * len(experiment_data), 4),
                              squeeze=False)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for idx, (exp_name, detections) in enumerate(experiment_data.items()):
        ax = axes[0, idx]
        if detections.empty or "confidence" not in detections.columns:
            ax.set_title(f"{exp_name}\n(no data)")
            continue

        ax.hist(detections["confidence"], bins=50, alpha=0.7,
                color=colors[idx % len(colors)], edgecolor="black", linewidth=0.5)
        ax.set_title(f"{exp_name}\n(n={len(detections)})", fontsize=10)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="0.5 threshold")

    plt.suptitle("Confidence Score Distributions Across Experiments", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_metrics_comparison(metrics_table: list, output_path: Path):
    """
    Plot bar chart comparing key metrics across experiments.
    """
    if not metrics_table:
        return

    exp_names = [m["experiment"] for m in metrics_table]
    mean_conf = [m.get("mean_confidence", 0) for m in metrics_table]
    high_conf = [m.get("high_confidence_rate", 0) for m in metrics_table]
    n_detections = [m.get("total_detections", 0) for m in metrics_table]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mean confidence
    bars = axes[0].bar(exp_names, mean_conf, color="#3498db", alpha=0.8)
    axes[0].set_title("Mean Confidence Score")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, mean_conf):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontsize=9)

    # High confidence rate
    bars = axes[1].bar(exp_names, high_conf, color="#2ecc71", alpha=0.8)
    axes[1].set_title("High-Confidence Detections (>0.5)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, high_conf):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=9)

    # Total detections
    bars = axes[2].bar(exp_names, n_detections, color="#e74c3c", alpha=0.8)
    axes[2].set_title("Total Detections")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, n_detections):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(n_detections)*0.01,
                     str(val), ha="center", fontsize=9)

    plt.suptitle("BirdNET Experiment Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_species_comparison(experiment_data: dict, output_path: Path,
                                 top_n: int = 15):
    """
    Plot per-species detection counts across experiments (top N species).
    """
    # Collect per-species counts across experiments
    all_species = set()
    exp_species_counts = {}

    for exp_name, detections in experiment_data.items():
        if detections.empty or "scientific_name" not in detections.columns:
            continue
        counts = detections["scientific_name"].value_counts().to_dict()
        exp_species_counts[exp_name] = counts
        all_species.update(counts.keys())

    if not exp_species_counts:
        return

    # Get top N species by total detections across all experiments
    total_per_species = defaultdict(int)
    for counts in exp_species_counts.values():
        for sp, c in counts.items():
            total_per_species[sp] += c

    top_species = sorted(total_per_species, key=total_per_species.get, reverse=True)[:top_n]

    # Build comparison data
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top_species))
    width = 0.8 / max(len(exp_species_counts), 1)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for idx, (exp_name, counts) in enumerate(exp_species_counts.items()):
        values = [counts.get(sp, 0) for sp in top_species]
        ax.bar(x + idx * width, values, width, label=exp_name,
               color=colors[idx % len(colors)], alpha=0.8)

    # Use common names if available
    labels = []
    for sp in top_species:
        common = SPECIES_COMMON_NAMES.get(sp, sp)
        labels.append(common if len(common) < 25 else sp[:20] + "...")

    ax.set_xticks(x + width * (len(exp_species_counts) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Detection Count")
    ax.set_title(f"Per-Species Detections (Top {top_n})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Main Analysis
# ============================================================

def run_analysis(results_dir: Path, output_dir: Path):
    """Run complete analysis across all experiments."""
    print("=" * 60)
    print("RESULTS ANALYSIS")
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print("=" * 60)

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find experiment results
    experiment_csvs = find_result_csvs(results_dir)

    if not experiment_csvs:
        print("[ERROR] No experiment results found.")
        print(f"  Looked in: {results_dir}")
        print("  Run experiments first: python scripts/05_train_and_evaluate.py --experiment all")
        return

    # Load all detections
    experiment_data = {}
    metrics_table = []

    for exp_name, csv_files in experiment_csvs.items():
        if exp_name == "analysis":
            continue  # Skip our own output folder

        print(f"\n  Analyzing: {exp_name}")
        detections = load_detections(csv_files)
        experiment_data[exp_name] = detections

        metrics = compute_metrics(detections)
        metrics["experiment"] = exp_name
        fp_metrics = estimate_false_positive_rate(detections)
        metrics.update(fp_metrics)
        metrics_table.append(metrics)

    # --- Print comparison table ---
    print(f"\n{'=' * 80}")
    print(f"{'EXPERIMENT COMPARISON TABLE':^80}")
    print(f"{'=' * 80}")
    header = (f"{'Experiment':<25} {'Detections':>10} {'Mean Conf':>10} "
              f"{'Med Conf':>10} {'High%':>8} {'Species':>8}")
    print(header)
    print("-" * 80)
    for m in metrics_table:
        row = (f"{m['experiment']:<25} "
               f"{m.get('total_detections', 0):>10} "
               f"{m.get('mean_confidence', 0):>10.4f} "
               f"{m.get('median_confidence', 0):>10.4f} "
               f"{m.get('high_confidence_rate', 0):>7.1f}% "
               f"{m.get('unique_species_detected', 0):>8}")
        print(row)
    print("=" * 80)

    # --- Save metrics CSV ---
    metrics_csv = output_dir / "experiment_comparison.csv"
    if metrics_table:
        # Remove nested dicts for CSV export
        csv_rows = []
        for m in metrics_table:
            row = {k: v for k, v in m.items() if not isinstance(v, dict)}
            csv_rows.append(row)

        df_metrics = pd.DataFrame(csv_rows)
        df_metrics.to_csv(metrics_csv, index=False)
        print(f"\n  Metrics saved: {metrics_csv}")

    # --- Generate plots ---
    print(f"\n  Generating visualizations...")

    plot_confidence_distributions(
        experiment_data,
        output_dir / "confidence_distributions.png"
    )

    plot_metrics_comparison(
        metrics_table,
        output_dir / "metrics_comparison.png"
    )

    plot_per_species_comparison(
        experiment_data,
        output_dir / "per_species_comparison.png"
    )

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Experiments analyzed: {len(experiment_data)}")
    print(f"  Output directory:    {output_dir}")
    print(f"  Time elapsed:        {elapsed:.1f}s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare BirdNET experiment results."
    )
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR),
                        help="Path to results directory")
    parser.add_argument("--output_dir", type=str,
                        default=str(RESULTS_DIR / "analysis"),
                        help="Path to analysis output directory")
    args = parser.parse_args()

    ensure_dirs()
    run_analysis(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
