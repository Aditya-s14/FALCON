"""
Script 5: BirdNET Training & Evaluation
==========================================
Runs all experiments:
  - Baseline: Pre-trained BirdNET on IBC53 (no training)
  - Exp 1: Fine-tune WITHOUT noise class
  - Exp 2: Fine-tune WITH noise class (KEY experiment)
  - Exp 3: Few-shot data size sensitivity (10/25/50 samples)

Each experiment:
  1. Trains a custom BirdNET classifier (except baseline)
  2. Evaluates on the IBC53 test set
  3. Saves detection results as CSV

Usage:
    python scripts/05_train_and_evaluate.py --experiment baseline
    python scripts/05_train_and_evaluate.py --experiment exp1
    python scripts/05_train_and_evaluate.py --experiment exp2
    python scripts/05_train_and_evaluate.py --experiment exp3
    python scripts/05_train_and_evaluate.py --experiment all

Pipeline Position: FIFTH step — runs after 04_build_dataset.py
Requires: birdnet-analyzer[train] installed
Output: Trained .tflite classifiers + detection CSV results
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    IBC53_RAW_DIR, PROCESSED_DIR, PROCESSED_NO_NOISE_DIR,
    DATA_DIR, MODELS_DIR, RESULTS_DIR, MIN_CONFIDENCE,
    VERBOSE, ensure_dirs,
)


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and print output."""
    print(f"\n  [{description}]")
    print(f"  Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # BirdNET prints progress to stderr
            print(result.stderr)

        if result.returncode != 0:
            print(f"  [ERROR] Command failed with return code {result.returncode}")
            return False
        return True

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Command timed out after 1 hour")
        return False
    except FileNotFoundError:
        print(f"  [ERROR] Command not found. Is birdnet-analyzer installed?")
        print(f"  Run: pip install birdnet-analyzer[train]")
        return False


def run_baseline(test_dir: Path, results_dir: Path) -> bool:
    """
    Baseline experiment: Run pre-trained BirdNET on IBC53 with no fine-tuning.
    """
    print("=" * 60)
    print("EXPERIMENT: Baseline (Pre-trained BirdNET)")
    print("=" * 60)

    output_dir = results_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "birdnet_analyzer.analyze",
        str(test_dir),
        "--output", str(output_dir),
        "--min_conf", str(MIN_CONFIDENCE),
        "--rtype", "csv",
    ]

    return run_command(cmd, "Baseline evaluation")


def run_train_and_evaluate(train_dir: Path, test_dir: Path,
                            model_dir: Path, results_dir: Path,
                            classifier_name: str,
                            experiment_name: str,
                            autotune: bool = False) -> bool:
    """
    Train a custom BirdNET classifier and evaluate it.

    Args:
        train_dir: Directory with BirdNET-format training data.
        test_dir: Directory with test audio for evaluation.
        model_dir: Output directory for trained model.
        results_dir: Output directory for evaluation results.
        classifier_name: Name for the .tflite classifier file.
        experiment_name: Display name for this experiment.
        autotune: Whether to use keras-tuner for hyperparameter tuning.
    """
    print("=" * 60)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"  Training data: {train_dir}")
    print(f"  Classifier:    {classifier_name}")
    print("=" * 60)

    model_dir.mkdir(parents=True, exist_ok=True)
    exp_results_dir = results_dir / classifier_name
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Train ---
    train_cmd = [
        sys.executable, "-m", "birdnet_analyzer.train",
        str(train_dir),
        "--output", str(model_dir),
    ]
    if autotune:
        train_cmd.append("--autotune")

    success = run_command(train_cmd, f"Training {classifier_name}")
    if not success:
        print(f"  [ERROR] Training failed for {experiment_name}")
        return False

    # Verify classifier was created
    # BirdNET saves as CustomClassifier.tflite by default
    classifier_path = model_dir / f"{classifier_name}.tflite"
    if not classifier_path.is_file():
        # Check alternative names
        alt_paths = list(model_dir.rglob("*.tflite"))
        if alt_paths:
            classifier_path = alt_paths[0]
            print(f"  Classifier found at: {classifier_path}")
        else:
            print(f"  [ERROR] No .tflite classifier found in {model_dir}")
            print(f"  Contents of {model_dir}: {list(model_dir.iterdir())}")
            return False

    # --- Step 2: Evaluate ---
    eval_cmd = [
        sys.executable, "-m", "birdnet_analyzer.analyze",
        str(test_dir),
        "--output", str(exp_results_dir),
        "-c", str(classifier_path),
        "--min_conf", str(MIN_CONFIDENCE),
        "--rtype", "csv",
    ]

    success = run_command(eval_cmd, f"Evaluating {classifier_name}")
    if not success:
        print(f"  [WARN] Evaluation failed for {experiment_name}")

    return True


def run_experiment(experiment: str, test_dir: Path = None,
                   autotune: bool = False):
    """
    Run one or all experiments.

    Args:
        experiment: One of 'baseline', 'exp1', 'exp2', 'exp3', 'all'.
        test_dir: Path to test audio (defaults to IBC53_RAW_DIR).
        autotune: Whether to use hyperparameter tuning.
    """
    if test_dir is None:
        test_dir = IBC53_RAW_DIR

    start_time = time.time()
    results = {}

    if experiment in ("baseline", "all"):
        results["baseline"] = run_baseline(test_dir, RESULTS_DIR)

    if experiment in ("exp1", "all"):
        results["exp1"] = run_train_and_evaluate(
            train_dir=PROCESSED_NO_NOISE_DIR,
            test_dir=test_dir,
            model_dir=MODELS_DIR / "exp1_no_noise",
            results_dir=RESULTS_DIR,
            classifier_name="Exp1_NoNoise",
            experiment_name="Exp 1: Fine-tune WITHOUT Noise Class",
            autotune=autotune,
        )

    if experiment in ("exp2", "all"):
        results["exp2"] = run_train_and_evaluate(
            train_dir=PROCESSED_DIR,
            test_dir=test_dir,
            model_dir=MODELS_DIR / "exp2_with_noise",
            results_dir=RESULTS_DIR,
            classifier_name="Exp2_WithNoise",
            experiment_name="Exp 2: Fine-tune WITH Noise Class (KEY)",
            autotune=autotune,
        )

    if experiment in ("exp3", "all"):
        fewshot_base = DATA_DIR / "fewshot_subsets"
        for n_samples in [10, 25, 50]:
            fewshot_dir = fewshot_base / f"fewshot_{n_samples}"
            if fewshot_dir.is_dir():
                results[f"exp3_{n_samples}"] = run_train_and_evaluate(
                    train_dir=fewshot_dir,
                    test_dir=test_dir,
                    model_dir=MODELS_DIR / f"exp3_fewshot_{n_samples}",
                    results_dir=RESULTS_DIR,
                    classifier_name=f"Exp3_FewShot_{n_samples}",
                    experiment_name=f"Exp 3: Few-Shot ({n_samples} samples/species)",
                    autotune=autotune,
                )
            else:
                print(f"  [WARN] Few-shot subset not found: {fewshot_dir}")
                print(f"  Run: python scripts/04_build_dataset.py --build_fewshot")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"  Results: {results}")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Models directory:  {MODELS_DIR}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train BirdNET custom classifiers and evaluate on IBC53."
    )
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["baseline", "exp1", "exp2", "exp3", "all"],
                        help="Which experiment to run (default: all)")
    parser.add_argument("--test_dir", type=str, default=str(IBC53_RAW_DIR),
                        help="Path to test audio directory")
    parser.add_argument("--autotune", action="store_true",
                        help="Use keras-tuner for hyperparameter optimization")
    args = parser.parse_args()

    ensure_dirs()
    run_experiment(
        experiment=args.experiment,
        test_dir=Path(args.test_dir),
        autotune=args.autotune,
    )


if __name__ == "__main__":
    main()
