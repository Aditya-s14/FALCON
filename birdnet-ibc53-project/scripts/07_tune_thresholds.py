"""
Script 7: Threshold Tuning Utility
=====================================
Interactive helper for tuning noise detection thresholds on IBC53 data.

This script:
  1. Randomly samples N segments from your segmented audio
  2. Extracts features (RMS, spectral flatness, ZCR) for each
  3. Plots feature distributions to help you pick thresholds
  4. Tests different threshold combinations and shows classification stats
  5. Exports a tuning report CSV

Tuning these thresholds is part of the RESEARCH CONTRIBUTION for Paper 1.

Usage:
    python scripts/07_tune_thresholds.py
    python scripts/07_tune_thresholds.py --n_samples 200 --input_dir data/segments
"""

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    SEGMENTS_DIR, RESULTS_DIR, SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD, NOISE_FLATNESS_THRESHOLD,
    NOISE_ZCR_THRESHOLD, ensure_dirs,
)
from scripts.s02_classify_segments_lib import extract_features, classify_segment


def collect_samples(input_dir: Path, n_samples: int = 200,
                     seed: int = 42) -> list:
    """
    Randomly sample audio segments from the segmented directory.

    Returns list of (filepath, species_folder) tuples.
    """
    random.seed(seed)
    all_files = []

    for species_dir in sorted(input_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        for f in species_dir.glob("*.wav"):
            all_files.append((f, species_dir.name))

    if len(all_files) <= n_samples:
        sample = all_files
    else:
        sample = random.sample(all_files, n_samples)

    print(f"  Sampled {len(sample)} segments from {len(list(input_dir.iterdir()))} folders")
    return sample


def extract_all_features(samples: list) -> list:
    """Extract features for all sampled segments."""
    results = []
    for filepath, species_folder in samples:
        features = extract_features(filepath)
        if features:
            features["filepath"] = str(filepath)
            features["species_folder"] = species_folder
            features["filename"] = filepath.name
            results.append(features)
    print(f"  Extracted features for {len(results)}/{len(samples)} segments")
    return results


def plot_feature_distributions(features_list: list, output_dir: Path):
    """
    Plot histograms and scatter plots of the three features.
    These plots help you visually identify natural threshold boundaries.
    """
    rms_vals = [f["rms"] for f in features_list]
    flat_vals = [f["spectral_flatness"] for f in features_list]
    zcr_vals = [f["zcr"] for f in features_list]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Histograms
    axes[0, 0].hist(rms_vals, bins=50, color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0, 0].axvline(x=SILENCE_RMS_THRESHOLD, color="red", linestyle="--", label=f"Threshold={SILENCE_RMS_THRESHOLD}")
    axes[0, 0].set_title("RMS Energy Distribution")
    axes[0, 0].set_xlabel("RMS")
    axes[0, 0].legend()

    axes[0, 1].hist(flat_vals, bins=50, color="#2ecc71", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0, 1].axvline(x=NOISE_FLATNESS_THRESHOLD, color="red", linestyle="--", label=f"Threshold={NOISE_FLATNESS_THRESHOLD}")
    axes[0, 1].set_title("Spectral Flatness Distribution")
    axes[0, 1].set_xlabel("Spectral Flatness")
    axes[0, 1].legend()

    axes[0, 2].hist(zcr_vals, bins=50, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0, 2].axvline(x=NOISE_ZCR_THRESHOLD, color="red", linestyle="--", label=f"Threshold={NOISE_ZCR_THRESHOLD}")
    axes[0, 2].set_title("Zero Crossing Rate Distribution")
    axes[0, 2].set_xlabel("ZCR")
    axes[0, 2].legend()

    # Row 2: Scatter plots (feature pairs)
    axes[1, 0].scatter(rms_vals, flat_vals, alpha=0.4, s=10, c="#3498db")
    axes[1, 0].axhline(y=NOISE_FLATNESS_THRESHOLD, color="red", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(x=SILENCE_RMS_THRESHOLD, color="orange", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("RMS Energy")
    axes[1, 0].set_ylabel("Spectral Flatness")
    axes[1, 0].set_title("RMS vs Flatness")

    axes[1, 1].scatter(rms_vals, zcr_vals, alpha=0.4, s=10, c="#2ecc71")
    axes[1, 1].axhline(y=NOISE_ZCR_THRESHOLD, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(x=SILENCE_RMS_THRESHOLD, color="orange", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("RMS Energy")
    axes[1, 1].set_ylabel("ZCR")
    axes[1, 1].set_title("RMS vs ZCR")

    axes[1, 2].scatter(flat_vals, zcr_vals, alpha=0.4, s=10, c="#e74c3c")
    axes[1, 2].axhline(y=NOISE_ZCR_THRESHOLD, color="red", linestyle="--", alpha=0.5)
    axes[1, 2].axvline(x=NOISE_FLATNESS_THRESHOLD, color="orange", linestyle="--", alpha=0.5)
    axes[1, 2].set_xlabel("Spectral Flatness")
    axes[1, 2].set_ylabel("ZCR")
    axes[1, 2].set_title("Flatness vs ZCR")

    plt.suptitle("Feature Distributions for Threshold Tuning\n"
                 "(Red lines = current thresholds)", fontsize=13)
    plt.tight_layout()

    output_path = output_dir / "threshold_tuning_features.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature plots saved: {output_path}")


def grid_search_thresholds(features_list: list, output_dir: Path):
    """
    Test a grid of threshold combinations and report classification
    distributions for each. Helps identify the best balance.
    """
    silence_rms_range = [0.005, 0.01, 0.015, 0.02, 0.03]
    flatness_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    zcr_range = [0.05, 0.08, 0.1, 0.12, 0.15]

    results = []
    total = len(silence_rms_range) * len(flatness_range) * len(zcr_range)
    print(f"\n  Testing {total} threshold combinations...")

    for s_rms in silence_rms_range:
        for n_flat in flatness_range:
            for n_zcr in zcr_range:
                counts = {"bird": 0, "noise": 0, "silence": 0}
                for feat in features_list:
                    label = classify_segment(feat, s_rms, n_flat, n_zcr)
                    counts[label] += 1

                total_segs = sum(counts.values())
                results.append({
                    "silence_rms": s_rms,
                    "noise_flatness": n_flat,
                    "noise_zcr": n_zcr,
                    "bird": counts["bird"],
                    "noise": counts["noise"],
                    "silence": counts["silence"],
                    "bird_pct": f"{100*counts['bird']/total_segs:.1f}",
                    "noise_pct": f"{100*counts['noise']/total_segs:.1f}",
                    "silence_pct": f"{100*counts['silence']/total_segs:.1f}",
                })

    # Save grid search results
    import csv as csv_mod
    output_csv = output_dir / "threshold_grid_search.csv"
    fieldnames = ["silence_rms", "noise_flatness", "noise_zcr",
                  "bird", "noise", "silence", "bird_pct", "noise_pct", "silence_pct"]

    with open(output_csv, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Grid search results saved: {output_csv}")

    # Print top 10 most balanced results (noise between 5-20%)
    balanced = [r for r in results if 5.0 <= float(r["noise_pct"]) <= 20.0]
    balanced.sort(key=lambda r: abs(float(r["noise_pct"]) - 12.0))

    print(f"\n  TOP 10 BALANCED THRESHOLD COMBINATIONS:")
    print(f"  {'RMS':>6} {'Flat':>6} {'ZCR':>6} | {'Bird%':>6} {'Noise%':>7} {'Silence%':>9}")
    print(f"  {'-'*50}")
    for r in balanced[:10]:
        print(f"  {r['silence_rms']:>6.3f} {r['noise_flatness']:>6.2f} {r['noise_zcr']:>6.3f} | "
              f"{r['bird_pct']:>6}% {r['noise_pct']:>6}% {r['silence_pct']:>8}%")


def main():
    parser = argparse.ArgumentParser(
        description="Threshold tuning utility for noise detection."
    )
    parser.add_argument("--input_dir", type=str, default=str(SEGMENTS_DIR),
                        help="Path to segmented audio directory")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of random segments to sample")
    parser.add_argument("--output_dir", type=str,
                        default=str(RESULTS_DIR / "tuning"),
                        help="Output directory for tuning results")
    args = parser.parse_args()

    ensure_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("THRESHOLD TUNING UTILITY")
    print("=" * 60)

    # Step 1: Sample segments
    samples = collect_samples(Path(args.input_dir), args.n_samples)

    # Step 2: Extract features
    features_list = extract_all_features(samples)

    if not features_list:
        print("[ERROR] No features extracted. Check your segments directory.")
        return

    # Step 3: Plot feature distributions
    plot_feature_distributions(features_list, output_dir)

    # Step 4: Grid search thresholds
    grid_search_thresholds(features_list, output_dir)

    print(f"\n  NEXT STEP: Listen to ~50 segments and compare with classifications.")
    print(f"  Then update thresholds in configs/config.py")


if __name__ == "__main__":
    main()
