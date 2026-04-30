"""
Master Pipeline Runner
========================
Runs the complete BirdNET + IBC53 pipeline end-to-end.

Usage:
    python run_pipeline.py                    # Run everything
    python run_pipeline.py --stage 1          # Run only Stage 1 (segmentation)
    python run_pipeline.py --stage 1-4        # Run Stages 1 through 4
    python run_pipeline.py --stage 5 --experiment exp2  # Run only Exp 2

Stages:
    1. Audio Segmentation       (01_segment_audio.py)
    2. Noise Detection          (02_classify_segments.py)
    3. ESC-50 Extraction        (03_extract_esc50_noise.py)
    4. Dataset Building         (04_build_dataset.py)
    5. Training & Evaluation    (05_train_and_evaluate.py)
    6. Results Analysis         (06_analyze_results.py)
    7. Threshold Tuning         (07_tune_thresholds.py)  [optional, run separately]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_stage(script_name: str, extra_args: list = None, description: str = ""):
    """Run a pipeline script."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'#' * 70}")
    print(f"# STAGE: {description}")
    print(f"# Script: {script_name}")
    print(f"{'#' * 70}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Stage failed: {description}")
        print(f"  Script: {script_name}")
        print(f"  Return code: {result.returncode}")
        return False
    return True


def parse_stage_range(stage_str: str) -> list:
    """Parse stage argument like '1', '1-4', '2-6'."""
    if "-" in stage_str:
        start, end = stage_str.split("-")
        return list(range(int(start), int(end) + 1))
    else:
        return [int(stage_str)]


def main():
    parser = argparse.ArgumentParser(
        description="Run the BirdNET + IBC53 pipeline (all stages or specific ones)."
    )
    parser.add_argument("--stage", type=str, default="1-6",
                        help="Stage(s) to run: '1', '3', '1-4', '1-6' (default: 1-6)")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["baseline", "exp1", "exp2", "exp3", "all"],
                        help="Which experiment to run in Stage 5 (default: all)")
    parser.add_argument("--autotune", action="store_true",
                        help="Use keras-tuner in Stage 5")
    parser.add_argument("--build_fewshot", action="store_true",
                        help="Build few-shot subsets in Stage 4")
    args = parser.parse_args()

    stages = parse_stage_range(args.stage)

    print("=" * 70)
    print("BirdNET + IBC53: NOISE-AWARE BIRD AUDIO CLASSIFICATION PIPELINE")
    print(f"Running stages: {stages}")
    print("=" * 70)

    start_time = time.time()
    results = {}

    if 1 in stages:
        results[1] = run_stage(
            "01_segment_audio.py",
            description="Audio Segmentation (3-second chunks)"
        )

    if 2 in stages:
        results[2] = run_stage(
            "02_classify_segments.py",
            description="Energy-Based Noise Detection"
        )

    if 3 in stages:
        results[3] = run_stage(
            "03_extract_esc50_noise.py",
            description="ESC-50 Noise Extraction"
        )

    if 4 in stages:
        extra = []
        if args.build_fewshot:
            extra.append("--build_fewshot")
        results[4] = run_stage(
            "04_build_dataset.py",
            extra_args=extra if extra else None,
            description="Build BirdNET-Compatible Dataset"
        )

    if 5 in stages:
        extra = ["--experiment", args.experiment]
        if args.autotune:
            extra.append("--autotune")
        results[5] = run_stage(
            "05_train_and_evaluate.py",
            extra_args=extra,
            description="BirdNET Training & Evaluation"
        )

    if 6 in stages:
        results[6] = run_stage(
            "06_analyze_results.py",
            description="Results Analysis & Comparison"
        )

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE")
    print(f"  Stages run: {list(results.keys())}")
    print(f"  Results:    {results}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
