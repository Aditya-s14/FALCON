import os
import pandas as pd
import glob
from collections import defaultdict

RESULTS_DIR = "/Users/adityas/Desktop/FALCON_DL/FALCON_DL/birdnet-ibc53-project/results"
EXPERIMENTS = ["baseline", "Exp1_NoNoise", "Exp2_WithNoise", "Exp3_FewShot_10", "Exp3_FewShot_25", "Exp3_FewShot_50"]

def analyze_experiment(exp_name):
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    species_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]

    species_results = {}
    all_correct_conf = []
    all_incorrect_conf = []
    total_detections = 0

    for species in sorted(species_dirs):
        sp_dir = os.path.join(exp_dir, species)
        csv_files = glob.glob(os.path.join(sp_dir, "*.BirdNET.results.csv"))

        files_correct = 0
        files_total = 0

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except:
                continue
            if df.empty:
                continue

            total_detections += len(df)

            # Confidence tracking
            correct_mask = df["Scientific name"] == species
            correct_confs = df.loc[correct_mask, "Confidence"].tolist()
            incorrect_confs = df.loc[~correct_mask, "Confidence"].tolist()
            all_correct_conf.extend(correct_confs)
            all_incorrect_conf.extend(incorrect_confs)

            # Top-1 per segment: keep highest confidence per segment
            df_top1 = df.loc[df.groupby(["Start (s)", "End (s)"])["Confidence"].idxmax()]

            # File-level: species with most top-1 detections
            species_counts = df_top1["Scientific name"].value_counts()
            predicted_species = species_counts.index[0]

            files_total += 1
            if predicted_species == species:
                files_correct += 1

        if files_total > 0:
            acc = files_correct / files_total
            species_results[species] = (files_correct, files_total, acc)
        else:
            species_results[species] = (0, 0, 0.0)

    # Overall
    total_correct = sum(v[0] for v in species_results.values())
    total_files = sum(v[1] for v in species_results.values())
    overall_acc = total_correct / total_files if total_files > 0 else 0.0

    mean_correct_conf = sum(all_correct_conf) / len(all_correct_conf) if all_correct_conf else 0.0
    mean_incorrect_conf = sum(all_incorrect_conf) / len(all_incorrect_conf) if all_incorrect_conf else 0.0

    species_above_50 = [s for s, v in species_results.items() if v[2] > 0.5]

    # Best/worst
    sorted_by_acc = sorted(species_results.items(), key=lambda x: x[1][2], reverse=True)
    best_3 = sorted_by_acc[:3]
    worst_3 = sorted_by_acc[-3:]

    return {
        "overall_acc": overall_acc,
        "total_correct": total_correct,
        "total_files": total_files,
        "total_detections": total_detections,
        "mean_correct_conf": mean_correct_conf,
        "mean_incorrect_conf": mean_incorrect_conf,
        "num_species_above_50": len(species_above_50),
        "total_species": len(species_results),
        "best_3": best_3,
        "worst_3": worst_3,
        "species_results": species_results,
    }

# Run all experiments
print("=" * 120)
print(f"{'Experiment':<22} {'Overall Acc':>11} {'Correct/Total':>14} {'Detections':>11} {'Conf(correct)':>14} {'Conf(incorrect)':>16} {'Sp>50%':>7}")
print("=" * 120)

all_results = {}
for exp in EXPERIMENTS:
    r = analyze_experiment(exp)
    all_results[exp] = r
    print(f"{exp:<22} {r['overall_acc']:>10.1%} {r['total_correct']:>6}/{r['total_files']:<6} {r['total_detections']:>11,} {r['mean_correct_conf']:>14.4f} {r['mean_incorrect_conf']:>16.4f} {r['num_species_above_50']:>4}/{r['total_species']}")

print("=" * 120)

# Per-species detail
print("\n\nPER-SPECIES ACCURACY (all experiments side by side):")
print("-" * 140)
header = f"{'Species':<35}"
for exp in EXPERIMENTS:
    short = exp.replace("Exp3_FewShot_", "FS").replace("Exp1_NoNoise", "Exp1").replace("Exp2_WithNoise", "Exp2")
    header += f" {short:>12}"
print(header)
print("-" * 140)

all_species = sorted(all_results["baseline"]["species_results"].keys())
for sp in all_species:
    row = f"{sp:<35}"
    for exp in EXPERIMENTS:
        sr = all_results[exp]["species_results"].get(sp)
        if sr and sr[1] > 0:
            row += f" {sr[2]:>11.0%}"
        else:
            row += f" {'N/A':>11}"
    print(row)

# Best and worst per experiment
print("\n\nBEST & WORST SPECIES PER EXPERIMENT:")
print("-" * 100)
for exp in EXPERIMENTS:
    r = all_results[exp]
    print(f"\n{exp}:")
    print(f"  Best:  ", ", ".join(f"{s} ({v[2]:.0%})" for s, v in r["best_3"]))
    print(f"  Worst: ", ", ".join(f"{s} ({v[2]:.0%})" for s, v in r["worst_3"]))
