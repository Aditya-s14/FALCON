"""
Script 3: ESC-50 Noise Extraction
====================================
Extracts relevant environmental noise classes from the ESC-50 dataset
and copies them to the processed noise folder.

ESC-50 provides pre-labeled environmental sounds at 44.1kHz / 5-second clips.
BirdNET auto-resamples to 48kHz and crops to 3 seconds internally during training.

Usage:
    python scripts/03_extract_esc50_noise.py
    python scripts/03_extract_esc50_noise.py --esc50_dir data/esc50 --output_dir data/processed/noise

Pipeline Position: THIRD step — can run in parallel with 02_classify_segments.py
Output: Noise WAV files copied to the processed noise folder.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    ESC50_DIR, ESC50_AUDIO_DIR, ESC50_META_CSV,
    PROCESSED_DIR, ESC50_NOISE_CATEGORIES, NOISE_FOLDER_NAME,
    VERBOSE, ensure_dirs,
)


def extract_esc50_noise(esc50_dir: Path, output_dir: Path,
                         categories: list = None) -> dict:
    """
    Extract environmental noise files from ESC-50 dataset.

    Args:
        esc50_dir: Root ESC-50 directory (contains audio/ and meta/).
        output_dir: Output directory for noise files.
        categories: List of ESC-50 category names to extract.

    Returns:
        dict of stats per category.
    """
    if categories is None:
        categories = ESC50_NOISE_CATEGORIES

    audio_dir = esc50_dir / "audio"
    meta_csv = esc50_dir / "meta" / "esc50.csv"

    print("=" * 60)
    print("ESC-50 Noise Extraction")
    print(f"ESC-50 path:  {esc50_dir}")
    print(f"Output path:  {output_dir}")
    print(f"Categories:   {categories}")
    print("=" * 60)

    # Validate paths
    if not audio_dir.is_dir():
        print(f"[ERROR] ESC-50 audio directory not found: {audio_dir}")
        print("  Run: git clone https://github.com/karolpiczak/ESC-50.git data/esc50")
        return {}

    if not meta_csv.is_file():
        print(f"[ERROR] ESC-50 metadata CSV not found: {meta_csv}")
        return {}

    # Read metadata
    df = pd.read_csv(str(meta_csv))
    print(f"  Total ESC-50 recordings: {len(df)}")
    print(f"  Available categories: {sorted(df['category'].unique())}")

    # Filter for noise categories
    noise_df = df[df["category"].isin(categories)]
    print(f"  Matched noise recordings: {len(noise_df)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    stats = {}
    total_copied = 0
    total_missing = 0

    for category in categories:
        cat_df = noise_df[noise_df["category"] == category]
        copied = 0
        missing = 0

        for _, row in cat_df.iterrows():
            src = audio_dir / row["filename"]
            # Prefix with esc50_ and category for traceability
            dst_name = f"esc50_{category}_{row['filename']}"
            dst = output_dir / dst_name

            if src.is_file():
                shutil.copy2(str(src), str(dst))
                copied += 1
            else:
                missing += 1
                if VERBOSE:
                    print(f"  [WARN] File not found: {src}")

        stats[category] = {"copied": copied, "missing": missing}
        total_copied += copied
        total_missing += missing

        if VERBOSE:
            print(f"  {category:20s} | copied={copied:3d}, missing={missing}")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"ESC-50 EXTRACTION COMPLETE")
    print(f"  Files copied:   {total_copied}")
    print(f"  Files missing:  {total_missing}")
    print(f"  Output:         {output_dir}")
    print(f"  Time elapsed:   {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract environmental noise categories from ESC-50 dataset."
    )
    parser.add_argument("--esc50_dir", type=str, default=str(ESC50_DIR),
                        help="Path to ESC-50 dataset root directory")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROCESSED_DIR / NOISE_FOLDER_NAME),
                        help="Path to output noise folder")
    args = parser.parse_args()

    ensure_dirs()
    extract_esc50_noise(
        esc50_dir=Path(args.esc50_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
