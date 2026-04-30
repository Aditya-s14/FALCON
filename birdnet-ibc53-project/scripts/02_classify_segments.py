"""
Script 2: Energy-Based Noise Detection (Segment Classifier)
=============================================================
Classifies each 3-second audio segment as 'bird', 'noise', or 'silence'
using three signal processing features:
  - RMS Energy (volume)
  - Spectral Flatness (tonal vs noise-like)
  - Zero Crossing Rate (signal structure)

This is the CORE of the CST contribution (Paper 1).

Usage:
    python scripts/02_classify_segments.py
    python scripts/02_classify_segments.py --input_dir data/segments --output_csv results/segment_classifications.csv

Pipeline Position: SECOND step — runs after 01_segment_audio.py
Output: CSV with per-segment features and classifications.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    SEGMENTS_DIR, RESULTS_DIR, SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD, NOISE_FLATNESS_THRESHOLD,
    NOISE_ZCR_THRESHOLD, VERBOSE, ensure_dirs,
)


def extract_features(audio_path: Path, sr: int = SAMPLE_RATE) -> Optional[dict]:
    """
    Extract signal processing features from a single audio segment.

    Args:
        audio_path: Path to the 3-second WAV file.
        sr: Sample rate to load at.

    Returns:
        dict with keys: rms, spectral_flatness, zcr, duration_s
        or None if loading fails.
    """
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load {audio_path.name}: {e}")
        return None

    # Feature 1: RMS Energy — overall volume
    rms = float(np.sqrt(np.mean(y ** 2)))

    # Feature 2: Spectral Flatness — tonal (near 0) vs noise-like (near 1)
    flatness_frames = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = float(np.mean(flatness_frames))

    # Feature 3: Zero Crossing Rate — signal structure
    zcr_frames = librosa.feature.zero_crossing_rate(y)
    zcr = float(np.mean(zcr_frames))

    return {
        "rms": rms,
        "spectral_flatness": spectral_flatness,
        "zcr": zcr,
        "duration_s": len(y) / sr,
    }


def classify_segment(features: dict,
                     silence_rms: float = SILENCE_RMS_THRESHOLD,
                     noise_flatness: float = NOISE_FLATNESS_THRESHOLD,
                     noise_zcr: float = NOISE_ZCR_THRESHOLD) -> str:
    """
    Classify a segment as 'bird', 'noise', or 'silence' based on features.

    Decision Logic:
        1. If RMS energy < silence_rms → 'silence'
        2. If spectral_flatness > noise_flatness AND zcr > noise_zcr → 'noise'
        3. Otherwise → 'bird'

    Args:
        features: dict from extract_features().
        silence_rms: RMS threshold below which = silence.
        noise_flatness: Spectral flatness threshold above which = noise-like.
        noise_zcr: ZCR threshold above which (combined with flatness) = noise.

    Returns:
        Classification label: 'bird', 'noise', or 'silence'.

    WARNING: The default thresholds are STARTING POINTS. You MUST tune
    them by listening to ~50-100 segments. This tuning is part of the
    research contribution.
    """
    if features["rms"] < silence_rms:
        return "silence"
    elif features["spectral_flatness"] > noise_flatness and features["zcr"] > noise_zcr:
        return "noise"
    else:
        return "bird"


def classify_all_segments(input_dir: Path, output_csv: Path,
                          silence_rms: float = SILENCE_RMS_THRESHOLD,
                          noise_flatness: float = NOISE_FLATNESS_THRESHOLD,
                          noise_zcr: float = NOISE_ZCR_THRESHOLD) -> dict:
    """
    Classify all segmented audio files and write results to CSV.

    Args:
        input_dir: Root directory containing species subfolders with segments.
        output_csv: Path to the output CSV file.

    Returns:
        dict of summary statistics.
    """
    print("=" * 60)
    print("STAGE 2: Energy-Based Noise Detection")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_csv}")
    print(f"Thresholds: silence_rms={silence_rms}, "
          f"noise_flatness={noise_flatness}, noise_zcr={noise_zcr}")
    print("=" * 60)

    start_time = time.time()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Collect all species folders
    species_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    counts = {"bird": 0, "noise": 0, "silence": 0, "errors": 0}
    rows = []

    for species_dir in species_dirs:
        species_name = species_dir.name
        audio_files = sorted(
            [f for f in species_dir.iterdir()
             if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")]
        )

        species_counts = {"bird": 0, "noise": 0, "silence": 0}

        for audio_file in audio_files:
            features = extract_features(audio_file)
            if features is None:
                counts["errors"] += 1
                continue

            label = classify_segment(features, silence_rms, noise_flatness, noise_zcr)
            species_counts[label] += 1
            counts[label] += 1

            rows.append({
                "species_folder": species_name,
                "filename": audio_file.name,
                "filepath": str(audio_file),
                "rms": f"{features['rms']:.6f}",
                "spectral_flatness": f"{features['spectral_flatness']:.6f}",
                "zcr": f"{features['zcr']:.6f}",
                "duration_s": f"{features['duration_s']:.2f}",
                "classification": label,
            })

        if VERBOSE:
            total = sum(species_counts.values())
            print(f"  {species_name:40s} | "
                  f"bird={species_counts['bird']:4d}  "
                  f"noise={species_counts['noise']:3d}  "
                  f"silence={species_counts['silence']:3d}  "
                  f"(total={total})")

    # Write CSV
    fieldnames = ["species_folder", "filename", "filepath",
                  "rms", "spectral_flatness", "zcr",
                  "duration_s", "classification"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start_time
    total_segments = counts["bird"] + counts["noise"] + counts["silence"]

    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"  Total segments:  {total_segments}")
    print(f"  Bird:            {counts['bird']:5d} ({100*counts['bird']/max(total_segments,1):.1f}%)")
    print(f"  Noise:           {counts['noise']:5d} ({100*counts['noise']/max(total_segments,1):.1f}%)")
    print(f"  Silence:         {counts['silence']:5d} ({100*counts['silence']/max(total_segments,1):.1f}%)")
    print(f"  Errors:          {counts['errors']}")
    print(f"  Results saved:   {output_csv}")
    print(f"  Time elapsed:    {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Classify audio segments as bird/noise/silence using signal features."
    )
    parser.add_argument("--input_dir", type=str, default=str(SEGMENTS_DIR),
                        help="Path to segmented audio directory")
    parser.add_argument("--output_csv", type=str,
                        default=str(RESULTS_DIR / "segment_classifications.csv"),
                        help="Path to output CSV file")
    parser.add_argument("--silence_rms", type=float, default=SILENCE_RMS_THRESHOLD,
                        help=f"RMS threshold for silence (default: {SILENCE_RMS_THRESHOLD})")
    parser.add_argument("--noise_flatness", type=float, default=NOISE_FLATNESS_THRESHOLD,
                        help=f"Spectral flatness threshold for noise (default: {NOISE_FLATNESS_THRESHOLD})")
    parser.add_argument("--noise_zcr", type=float, default=NOISE_ZCR_THRESHOLD,
                        help=f"ZCR threshold for noise (default: {NOISE_ZCR_THRESHOLD})")
    args = parser.parse_args()

    ensure_dirs()
    classify_all_segments(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
        silence_rms=args.silence_rms,
        noise_flatness=args.noise_flatness,
        noise_zcr=args.noise_zcr,
    )


if __name__ == "__main__":
    main()
