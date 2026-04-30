"""
Script 4: Build Final Dataset
================================
Assembles the BirdNET-compatible dataset from:
  1. Bird segments (classified by 02_classify_segments.py)
  2. ESC-50 noise files (extracted by 03_extract_esc50_noise.py)
  3. Pipeline-extracted noise from species folders
  4. Noise from Mystery mystery folder

Creates TWO dataset variants:
  - processed/          → 30 species + noise folder (for Exp 2)
  - processed_no_noise/ → 30 species only (for Exp 1)

Also creates few-shot subsets for Experiment 3.

Usage:
    python scripts/04_build_dataset.py
    python scripts/04_build_dataset.py --classifications_csv results/segment_classifications.csv

Pipeline Position: FOURTH step — runs after 02 and 03.
Output: BirdNET-format folder structure ready for training.
"""

import argparse
import csv
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    SEGMENTS_DIR, PROCESSED_DIR, PROCESSED_NO_NOISE_DIR,
    RESULTS_DIR, DATA_DIR, SPECIES_NAMES, NOISE_FOLDER_NAME,
    MYSTERY_FOLDER_NAME, VERBOSE, ensure_dirs,
)


def load_classifications(csv_path: Path) -> list:
    """Load segment classifications from CSV produced by script 02."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} segment classifications from {csv_path.name}")
    return rows


def build_dataset(classifications_csv: Path,
                  segments_dir: Path,
                  output_with_noise: Path,
                  output_no_noise: Path,
                  esc50_noise_dir: Path = None) -> dict:
    """
    Build BirdNET-compatible dataset from classified segments.

    Dataset structure (output_with_noise):
        processed/
            Pellorneum ruficeps/
                seg_0001.wav
                ...
            ... (28 more species)
            noise/
                esc50_rain_001.wav
                mystery_seg_0012.wav
                species_noise_0045.wav

    Dataset structure (output_no_noise):
        processed_no_noise/
            Pellorneum ruficeps/
                seg_0001.wav
            ... (same species, no noise folder)
    """
    print("=" * 60)
    print("STAGE 4: Build BirdNET-Compatible Dataset")
    print(f"Classifications: {classifications_csv}")
    print(f"With noise:      {output_with_noise}")
    print(f"Without noise:   {output_no_noise}")
    print("=" * 60)

    start_time = time.time()

    # Load classifications
    rows = load_classifications(classifications_csv)

    # Separate by species folder and classification
    species_bird_segments = defaultdict(list)   # species -> [filepaths]
    noise_segments = []                          # all noise from pipeline
    silence_segments = []                        # discarded
    mystery_bird_segments = []                   # mystery birds (not used)
    mystery_noise_segments = []                  # mystery noise

    for row in rows:
        folder = row["species_folder"]
        filepath = Path(row["filepath"])
        label = row["classification"]

        if folder == MYSTERY_FOLDER_NAME:
            if label == "noise":
                mystery_noise_segments.append(filepath)
            elif label == "bird":
                mystery_bird_segments.append(filepath)
            # silence is discarded
        elif folder in SPECIES_NAMES:
            if label == "bird":
                species_bird_segments[folder].append(filepath)
            elif label == "noise":
                noise_segments.append(filepath)
            # silence is discarded
        else:
            # Unknown folder — check if it's a species variant name
            if label == "bird":
                species_bird_segments[folder].append(filepath)
            elif label == "noise":
                noise_segments.append(filepath)

    # --- Build WITH-NOISE dataset ---
    print(f"\n  Building WITH-NOISE dataset...")
    if output_with_noise.exists():
        shutil.rmtree(output_with_noise)

    stats_with_noise = {"species_segments": 0, "noise_segments": 0, "species_count": 0}

    # Copy bird segments
    for species_name in SPECIES_NAMES:
        src_files = species_bird_segments.get(species_name, [])
        if not src_files:
            # Fallback: check for the folder directly in segments_dir
            seg_folder = segments_dir / species_name
            if seg_folder.is_dir():
                src_files = sorted(seg_folder.glob("*.wav"))

        if src_files:
            dest_dir = output_with_noise / species_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in src_files:
                if f.is_file():
                    shutil.copy2(str(f), str(dest_dir / f.name))
                    stats_with_noise["species_segments"] += 1
            stats_with_noise["species_count"] += 1

            if VERBOSE:
                print(f"    {species_name:40s} | {len(src_files):4d} bird segments")

    # Build noise folder (3 sources combined)
    noise_dir = output_with_noise / NOISE_FOLDER_NAME
    noise_dir.mkdir(parents=True, exist_ok=True)

    # Source 1: Pipeline-extracted noise from species folders
    for f in noise_segments:
        if f.is_file():
            dest_name = f"species_noise_{f.name}"
            shutil.copy2(str(f), str(noise_dir / dest_name))
            stats_with_noise["noise_segments"] += 1

    # Source 2: Mystery mystery noise
    for f in mystery_noise_segments:
        if f.is_file():
            dest_name = f"mystery_{f.name}"
            shutil.copy2(str(f), str(noise_dir / dest_name))
            stats_with_noise["noise_segments"] += 1

    # Source 3: ESC-50 noise (already extracted by script 03)
    if esc50_noise_dir is None:
        esc50_noise_dir = output_with_noise / NOISE_FOLDER_NAME
        # Check if ESC-50 files were previously extracted to a staging dir
        staging_noise = PROCESSED_DIR / NOISE_FOLDER_NAME
        if staging_noise.is_dir() and staging_noise != noise_dir:
            esc50_noise_dir = staging_noise

    # Look for ESC-50 files in the staging noise directory
    esc50_staging = DATA_DIR / "processed" / NOISE_FOLDER_NAME
    if esc50_staging.is_dir() and esc50_staging != noise_dir:
        for f in esc50_staging.glob("esc50_*.wav"):
            shutil.copy2(str(f), str(noise_dir / f.name))
            stats_with_noise["noise_segments"] += 1

    print(f"\n    Noise folder: {stats_with_noise['noise_segments']} total segments")
    print(f"      - Pipeline noise from species: {len(noise_segments)}")
    print(f"      - Mystery mystery noise:       {len(mystery_noise_segments)}")
    print(f"      - Mystery mystery birds (set aside): {len(mystery_bird_segments)}")

    # --- Build NO-NOISE dataset ---
    print(f"\n  Building NO-NOISE dataset...")
    if output_no_noise.exists():
        shutil.rmtree(output_no_noise)

    stats_no_noise = {"species_segments": 0, "species_count": 0}

    for species_name in SPECIES_NAMES:
        src_dir = output_with_noise / species_name
        if src_dir.is_dir():
            dest_dir = output_no_noise / species_name
            shutil.copytree(str(src_dir), str(dest_dir))
            n_files = len(list(dest_dir.glob("*.wav")))
            stats_no_noise["species_segments"] += n_files
            stats_no_noise["species_count"] += 1

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"DATASET BUILD COMPLETE")
    print(f"  WITH-NOISE dataset:")
    print(f"    Species:   {stats_with_noise['species_count']}")
    print(f"    Bird segs: {stats_with_noise['species_segments']}")
    print(f"    Noise segs:{stats_with_noise['noise_segments']}")
    print(f"    Total:     {stats_with_noise['species_segments'] + stats_with_noise['noise_segments']}")
    print(f"    Path:      {output_with_noise}")
    print(f"  NO-NOISE dataset:")
    print(f"    Species:   {stats_no_noise['species_count']}")
    print(f"    Bird segs: {stats_no_noise['species_segments']}")
    print(f"    Path:      {output_no_noise}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return {
        "with_noise": stats_with_noise,
        "no_noise": stats_no_noise,
    }


def build_fewshot_subsets(source_dir: Path, output_base: Path,
                          sample_sizes: list = None,
                          seed: int = 42) -> dict:
    """
    Create few-shot subsets for Experiment 3 (data size sensitivity).

    For each sample_size, randomly sample that many segments per species
    from the full dataset, plus include all noise files.

    Args:
        source_dir: Path to the full WITH-NOISE dataset.
        output_base: Base directory for few-shot datasets.
        sample_sizes: List of per-species sample counts (e.g., [10, 25, 50]).
        seed: Random seed for reproducibility.
    """
    if sample_sizes is None:
        sample_sizes = [10, 25, 50]

    print(f"\n  Building few-shot subsets: {sample_sizes}")
    random.seed(seed)
    stats = {}

    for n_samples in sample_sizes:
        subset_dir = output_base / f"fewshot_{n_samples}"
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
        subset_dir.mkdir(parents=True, exist_ok=True)

        total_segments = 0

        for species_dir in sorted(source_dir.iterdir()):
            if not species_dir.is_dir():
                continue

            dest_dir = subset_dir / species_dir.name

            if species_dir.name == NOISE_FOLDER_NAME:
                # Copy ALL noise files (not subsampled)
                shutil.copytree(str(species_dir), str(dest_dir))
                total_segments += len(list(dest_dir.glob("*.wav")))
            else:
                # Subsample bird species
                all_files = sorted(species_dir.glob("*.wav"))
                selected = random.sample(all_files, min(n_samples, len(all_files)))

                dest_dir.mkdir(parents=True, exist_ok=True)
                for f in selected:
                    shutil.copy2(str(f), str(dest_dir / f.name))
                total_segments += len(selected)

        stats[n_samples] = total_segments
        print(f"    fewshot_{n_samples}: {total_segments} total segments -> {subset_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build BirdNET-compatible datasets from classified segments."
    )
    parser.add_argument("--classifications_csv", type=str,
                        default=str(RESULTS_DIR / "segment_classifications.csv"),
                        help="Path to segment classifications CSV")
    parser.add_argument("--segments_dir", type=str, default=str(SEGMENTS_DIR),
                        help="Path to segmented audio directory")
    parser.add_argument("--output_with_noise", type=str, default=str(PROCESSED_DIR),
                        help="Path for dataset with noise class")
    parser.add_argument("--output_no_noise", type=str, default=str(PROCESSED_NO_NOISE_DIR),
                        help="Path for dataset without noise class")
    parser.add_argument("--build_fewshot", action="store_true",
                        help="Also build few-shot subsets for Experiment 3")
    args = parser.parse_args()

    ensure_dirs()

    build_dataset(
        classifications_csv=Path(args.classifications_csv),
        segments_dir=Path(args.segments_dir),
        output_with_noise=Path(args.output_with_noise),
        output_no_noise=Path(args.output_no_noise),
    )

    if args.build_fewshot:
        build_fewshot_subsets(
            source_dir=Path(args.output_with_noise),
            output_base=DATA_DIR / "fewshot_subsets",
        )


if __name__ == "__main__":
    main()
