"""
Script 1: Audio Segmentation
==============================
Takes raw IBC53 audio recordings (10-30s each) and splits them into
3-second non-overlapping chunks matching BirdNET's internal window size.

Usage:
    python scripts/01_segment_audio.py
    python scripts/01_segment_audio.py --input_dir data/ibc53 --output_dir data/segments

Pipeline Position: FIRST step in the CST pipeline.
Output: Segmented WAV files organized by species folder.
"""

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    IBC53_RAW_DIR, SEGMENTS_DIR, SAMPLE_RATE,
    SEGMENT_LENGTH, SEGMENT_SAMPLES, MIN_SEGMENT_RATIO,
    SPECIES_NAMES, MYSTERY_FOLDER_NAME, VERBOSE, ensure_dirs,
)


def segment_single_file(input_path: Path, output_dir: Path,
                         sr: int = SAMPLE_RATE,
                         segment_length: float = SEGMENT_LENGTH) -> int:
    """
    Segment a single audio file into fixed-length chunks.

    Args:
        input_path: Path to the input WAV file.
        output_dir: Directory to write segmented WAV files.
        sr: Target sample rate (default: 48kHz for BirdNET).
        segment_length: Length of each segment in seconds (default: 3.0).

    Returns:
        Number of segments created.
    """
    try:
        y, _ = librosa.load(str(input_path), sr=sr, mono=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load {input_path.name}: {e}")
        return 0

    segment_samples = int(segment_length * sr)
    min_samples = int(segment_samples * MIN_SEGMENT_RATIO)
    segments_created = 0

    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]

        # Discard segments shorter than 50% of target length
        if len(segment) < min_samples:
            continue

        # Pad short segments (>50% but <100%) with zeros
        if len(segment) < segment_samples:
            segment = librosa.util.fix_length(segment, size=segment_samples)

        # Write segment
        filename = f"{input_path.stem}_seg{segments_created:04d}.wav"
        out_path = output_dir / filename
        sf.write(str(out_path), segment, sr)
        segments_created += 1

    return segments_created


def segment_species_folder(species_dir: Path, output_base: Path) -> dict:
    """
    Segment all audio files in a single species folder.

    Returns:
        dict with stats: {files_processed, segments_created, errors}
    """
    species_name = species_dir.name
    output_dir = output_base / species_name
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        [f for f in species_dir.iterdir()
         if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")]
    )

    stats = {"files_processed": 0, "segments_created": 0, "errors": 0}

    for audio_file in audio_files:
        n_segments = segment_single_file(audio_file, output_dir)
        if n_segments > 0:
            stats["files_processed"] += 1
            stats["segments_created"] += n_segments
        else:
            stats["errors"] += 1

    return stats


def run_segmentation(input_dir: Path, output_dir: Path,
                     include_mystery: bool = True):
    """
    Run segmentation on all selected species + optionally the Mystery folder.

    Args:
        input_dir: Root IBC53 directory containing species folders.
        output_dir: Root output directory for segmented audio.
        include_mystery: Whether to also segment the Mystery mystery folder.
    """
    print("=" * 60)
    print("STAGE 1: Audio Segmentation")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Segment length: {SEGMENT_LENGTH}s @ {SAMPLE_RATE}Hz")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    total_stats = {"files_processed": 0, "segments_created": 0, "errors": 0}

    # --- Process 30 selected species ---
    folders_found = 0
    for species_name in SPECIES_NAMES:
        species_dir = input_dir / species_name
        if not species_dir.is_dir():
            # Try case-insensitive match
            matches = [d for d in input_dir.iterdir()
                       if d.is_dir() and d.name.lower() == species_name.lower()]
            if matches:
                species_dir = matches[0]
            else:
                print(f"  [WARN] Species folder not found: {species_name}")
                continue

        folders_found += 1
        stats = segment_species_folder(species_dir, output_dir)
        total_stats["files_processed"] += stats["files_processed"]
        total_stats["segments_created"] += stats["segments_created"]
        total_stats["errors"] += stats["errors"]

        if VERBOSE:
            print(f"  [{folders_found:2d}/30] {species_name:40s} "
                  f"| {stats['files_processed']:3d} files -> "
                  f"{stats['segments_created']:4d} segments")

    # --- Process Mystery mystery folder ---
    if include_mystery:
        mystery_dir = input_dir / MYSTERY_FOLDER_NAME
        if mystery_dir.is_dir():
            print(f"\n  Processing Mystery mystery folder...")
            stats = segment_species_folder(mystery_dir, output_dir)
            total_stats["files_processed"] += stats["files_processed"]
            total_stats["segments_created"] += stats["segments_created"]
            total_stats["errors"] += stats["errors"]
            print(f"  Mystery mystery: {stats['files_processed']} files -> "
                  f"{stats['segments_created']} segments")
        else:
            print(f"  [WARN] Mystery mystery folder not found at {mystery_dir}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SEGMENTATION COMPLETE")
    print(f"  Species folders found: {folders_found}/30")
    print(f"  Files processed:       {total_stats['files_processed']}")
    print(f"  Segments created:      {total_stats['segments_created']}")
    print(f"  Errors:                {total_stats['errors']}")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Segment IBC53 audio into 3-second chunks for BirdNET."
    )
    parser.add_argument("--input_dir", type=str, default=str(IBC53_RAW_DIR),
                        help="Path to raw IBC53 dataset directory")
    parser.add_argument("--output_dir", type=str, default=str(SEGMENTS_DIR),
                        help="Path to output segmented audio directory")
    parser.add_argument("--no_mystery", action="store_true",
                        help="Skip the Mystery mystery folder")
    args = parser.parse_args()

    ensure_dirs()
    run_segmentation(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        include_mystery=not args.no_mystery,
    )


if __name__ == "__main__":
    main()
