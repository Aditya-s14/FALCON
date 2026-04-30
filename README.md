# FALCON

**FALCON** is a research-grade pipeline for **noise-aware fine-tuning and evaluation of BirdNET** on **Indian bird-call audio** (IBC53 subset). The goal is to improve robustness when recordings contain environmental noise, silence, and ambiguous segments—common failure modes for off-the-shelf bioacoustic models in regional field conditions.

This repository contains:

- **`birdnet-ibc53-project/`** — executable pipeline (segmentation → noise handling → dataset build → BirdNET train/eval → analysis).
- **Documentation and figures** — experiment write-up (`documentation.md`), IEEE draft (`paper_ieee.tex`), methodology helper (`generate_methodology.py`), and result charts under `figures/`.

---

## Table of contents

1. [Problem and approach](#problem-and-approach)
2. [High-level results](#high-level-results)
3. [Repository layout](#repository-layout)
4. [Tech stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Quick start](#quick-start)
7. [Data setup](#data-setup)
8. [Pipeline stages](#pipeline-stages)
9. [Experiments](#experiments)
10. [Configuration](#configuration)
11. [Optional scripts](#optional-scripts)
12. [What is not in Git](#what-is-not-in-git)
13. [Troubleshooting](#troubleshooting)
14. [License and citation](#license-and-citation)

---

## Problem and approach

**Problem:** BirdNET and similar models are strong globally but can **over-predict species** on noisy or non-bird segments when every window is forced into a species label. That inflates false positives and lowers confidence on challenging regional data.

**Approach:**

1. **Segment** long recordings into BirdNET-compatible windows (3 s @ 48 kHz mono).
2. **Label segments heuristically** as bird-like vs noise-like vs silence using **RMS**, **spectral flatness**, and **zero-crossing rate** (tunable thresholds).
3. **Augment training** with **ESC-50** environmental noise categories aligned to the project config.
4. **Build BirdNET-format datasets** — with and without a dedicated **`noise`** training class (BirdNET uses this as a non-event / suppression signal during training).
5. **Fine-tune** with `birdnet-analyzer[train]` and **evaluate** with BirdNET’s analysis tooling; compare metrics, confusion structure, and confidence calibration.

Detailed per-experiment tables and discussion live in [`documentation.md`](documentation.md).

---

## High-level results

Summary values are documented in [`documentation.md`](documentation.md) (IBC53 test protocol described there).

| Comparison | Exp 1 (no noise class) | Exp 2 (with noise class) |
|------------|------------------------|---------------------------|
| **Overall accuracy** | 74.56% | **75.61%** (+1.05 pp) |
| **Median confidence** | 0.4348 | **0.7436** (+71.0%) |
| **Total detections** | 14,489 | 13,979 (**−510**, fewer spurious outputs) |

**Few-shot (Exp 3)** overall accuracy on the same test setup: **10.81%** (10 samples/class), **35.09%** (25), **60.86%** (50)—see `documentation.md` for the full scaling table and per-species breakdown.

---

## Repository layout

```
<repository-root>/
├── README.md                 ← You are here
├── documentation.md          ← Experiment metrics, tables, narrative
├── paper_ieee.tex            ← IEEE-style draft
├── generate_methodology.py   ← Doc generation helper (python-docx)
├── figures/                  ← Exported plots for papers/slides
└── birdnet-ibc53-project/    ← Main Python project
    ├── configs/
    │   └── config.py         ← Paths, species list, audio + threshold constants
    ├── scripts/
    │   ├── 01_segment_audio.py
    │   ├── 02_classify_segments.py
    │   ├── 03_extract_esc50_noise.py
    │   ├── 04_build_dataset.py
    │   ├── 05_train_and_evaluate.py
    │   ├── 06_analyze_results.py
    │   ├── 07_tune_thresholds.py
    │   ├── 08_visualize_all.py
    │   └── 09_live_recognition.py
    ├── run_pipeline.py       ← Orchestrates stages 1–6 (see below)
    ├── requirements.txt
    ├── pyproject.toml
    ├── tests/
    └── README.md             ← Short in-project readme (duplicates quick start)
```

---

## Tech stack

| Area | Technologies |
|------|----------------|
| Language | **Python 3.11** (see `requirements.txt`; avoid unsupported minor versions called out there) |
| Model / training | **BirdNET** via **`birdnet-analyzer[train]==2.4.0`** (TensorFlow-based training; `.tflite` export for inference) |
| Audio | **Librosa**, **SoundFile** |
| Data | **NumPy**, **Pandas** |
| Plots | **Matplotlib** |
| Datasets | **Kaggle CLI** (IBC53 download), **ESC-50** (Git clone) |
| Optional | **Keras Tuner** (hyperparameter search flag in training stage) |
| Live demo | **`09_live_recognition.py`** uses **TensorFlow** + **SoundDevice** (install separately if you use this script) |

---

## Prerequisites

- **Python 3.11** (recommended; match `birdnet-ibc53-project/requirements.txt`).
- **Git** (to clone ESC-50).
- **Kaggle account + API credentials** if you download IBC53 via the Kaggle CLI (`~/.kaggle/kaggle.json`).
- **Disk space** — raw audio, segments, and training artifacts are large; plan tens of GB for a full reproduction.

---

## Quick start

All commands below assume you work inside **`birdnet-ibc53-project/`**.

```bash
cd birdnet-ibc53-project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# After data are in place (see next section):
python run_pipeline.py                     # default: stages 1–6
python run_pipeline.py --stage 1-4         # preprocessing + dataset build only
python run_pipeline.py --stage 5-6         # train/eval + analysis
python run_pipeline.py --stage 5 --experiment exp2
python run_pipeline.py --stage 4 --build_fewshot
```

---

## Data setup

Paths are centralized in [`birdnet-ibc53-project/configs/config.py`](birdnet-ibc53-project/configs/config.py). By default the pipeline expects:

| Asset | Typical location | How to obtain |
|--------|------------------|---------------|
| **IBC53** (Indian bird calls) | `birdnet-ibc53-project/data/ibc53/` | e.g. `kaggle datasets download -d arghyasahoo/ibc53-indian-bird-call-dataset` then unzip into `data/ibc53/` |
| **ESC-50** | `birdnet-ibc53-project/data/esc50/` | `git clone https://github.com/karolpiczak/ESC-50.git data/esc50` |

Verify folder names match what `config.py` expects (scientific species names for IBC53, ESC-50 `audio/` + `meta/esc50.csv`).

---

## Pipeline stages

The orchestrator [`run_pipeline.py`](birdnet-ibc53-project/run_pipeline.py) runs **stages 1–6** by default. Stage **7** is a standalone utility (threshold tuning), not invoked by `run_pipeline.py`.

| Stage | Script | Purpose |
|------:|--------|---------|
| **1** | `01_segment_audio.py` | Resample/segment audio into **3 s** windows at **48 kHz** mono |
| **2** | `02_classify_segments.py` | **Bird / noise / silence** heuristics (RMS, spectral flatness, ZCR) |
| **3** | `03_extract_esc50_noise.py` | Pull selected **ESC-50** noise categories into the training corpus |
| **4** | `04_build_dataset.py` | Assemble **BirdNET-style** folder layouts (`processed`, `processed_no_noise`, optional few-shot subsets) |
| **5** | `05_train_and_evaluate.py` | **Fine-tune** BirdNET and run **evaluation** (`baseline`, `exp1`, `exp2`, `exp3`, or `all`) |
| **6** | `06_analyze_results.py` | Aggregate BirdNET outputs, compare runs, produce analysis plots |
| **7** | `07_tune_thresholds.py` | **Optional** — empirical threshold tuning for stage 2 features |
| **8** | `08_visualize_all.py` | **Optional** — full figure suite (run after results exist) |
| **9** | `09_live_recognition.py` | **Optional** — **live microphone** inference with a `.tflite` model |

CLI flags on `run_pipeline.py`:

- `--stage` — e.g. `3`, `1-4`, `5-6` (default `1-6`)
- `--experiment` — `baseline` \| `exp1` \| `exp2` \| `exp3` \| `all` (stage 5)
- `--autotune` — enable Keras Tuner path in training (if configured in stage 5)
- `--build_fewshot` — pass through to dataset builder in stage 4

---

## Experiments

| ID | Idea | Training data (high level) |
|----|------|----------------------------|
| **Baseline** | Off-the-shelf BirdNET | No project fine-tuning |
| **Exp 1** | Domain adaptation **without** explicit noise class | ~30 species, `processed_no_noise` |
| **Exp 2** | **Noise-aware** fine-tuning | ~30 species + **`noise`** folder (ESC-50 + pipeline-derived noise) |
| **Exp 3** | **Few-shot** sensitivity | 10 / 25 / 50 samples per species (+ noise), under `data/fewshot_subsets/` |

**Important nuance (Exp 2):** BirdNET’s `noise` directory acts as a **non-event / suppression** signal during training; the deployed label set for species predictions remains the fine-tuned species set—see `documentation.md` for the exact interpretation and metrics.

---

## Configuration

Edit **[`birdnet-ibc53-project/configs/config.py`](birdnet-ibc53-project/configs/config.py)** for:

- Directory roots (`DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`, …)
- Audio parameters (`SAMPLE_RATE`, `SEGMENT_LENGTH`, …)
- Noise gate starting thresholds (`SILENCE_RMS_THRESHOLD`, `NOISE_FLATNESS_THRESHOLD`, `NOISE_ZCR_THRESHOLD`)
- Species selection (`SELECTED_SPECIES`) and ESC-50 category list

Thresholds are **starting points**; tuning is part of the methodology (`07_tune_thresholds.py`).

---

## Optional scripts

### Full visualization pass

After you have results under `results/`, you can regenerate the full chart suite:

```bash
cd birdnet-ibc53-project
source venv/bin/activate
python scripts/08_visualize_all.py
```

### Live microphone inference

Requires a trained **`.tflite`** model path and extra dependencies (**TensorFlow**, **SoundDevice**) if not already satisfied by your environment:

```bash
cd birdnet-ibc53-project
source venv/bin/activate
python scripts/09_live_recognition.py   # inspect script header for CLI args
```

---

## What is not in Git

To keep the repository cloneable and within typical Git hosting limits, **`.gitignore` excludes large generated trees**, including:

- `birdnet-ibc53-project/data/`
- `birdnet-ibc53-project/models/`
- `birdnet-ibc53-project/results/`
- `birdnet-ibc53-project/logs/`
- local virtual environments (`venv/`, etc.)

You are expected to **reproduce** those directories by running the pipeline after downloading IBC53 and ESC-50.

---

## Troubleshooting

| Symptom | Likely cause | What to try |
|--------|--------------|-------------|
| Kaggle download fails | Missing API key | Create `~/.kaggle/kaggle.json` with valid credentials |
| BirdNET train fails | Wrong Python / CUDA / TF stack | Use the Python version pinned in `requirements.txt`; check BirdNET Analyzer docs for your OS |
| Stage 2 labels look wrong | Thresholds not tuned for your mic/environment | Run `07_tune_thresholds.py` and update constants in `config.py` |
| Live script import errors | Optional deps not installed | `pip install sounddevice tensorflow` (versions compatible with your OS) |
| Git push rejected for huge artifacts | Accidentally staged `data/` or `venv/` | Keep artifacts ignored; use Git LFS only if you intentionally version large binaries |

---

## License and citation

- **ESC-50** and **BirdNET** / **birdnet-analyzer** are third-party projects—follow their respective licenses when redistributing data or model weights.
- If you use this repository in academic work, cite **BirdNET**, the **IBC53** dataset source you obtained from Kaggle, and **ESC-50** appropriately, and reference your copy of `paper_ieee.tex` / `documentation.md` as supplementary material.

---

## Related links

- **BirdNET Analyzer** (training and analysis tooling used in stage 5): see the upstream BirdNET / birdnet-analyzer documentation for CLI details.
- **ESC-50**: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
- **Remote repository**: [https://github.com/Aditya-s14/FALCON](https://github.com/Aditya-s14/FALCON)
