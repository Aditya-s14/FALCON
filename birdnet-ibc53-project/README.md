# BirdNET + IBC53: Noise-Aware Bird Audio Classification Pipeline

A dual-domain academic project that evaluates and improves BirdNET's reliability on Indian bird recordings through dataset-level engineering.

## Project Structure

```
birdnet-ibc53-project/
├── configs/
│   └── config.py              # Central configuration (paths, thresholds, species)
├── scripts/
│   ├── 01_segment_audio.py    # Stage 1: Split audio into 3s chunks
│   ├── 02_classify_segments.py # Stage 2: Classify segments (bird/noise/silence)
│   ├── 03_extract_esc50_noise.py # Stage 3: Extract ESC-50 noise files
│   ├── 04_build_dataset.py    # Stage 4: Assemble BirdNET-format dataset
│   ├── 05_train_and_evaluate.py # Stage 5: Train & evaluate all experiments
│   ├── 06_analyze_results.py  # Stage 6: Compare results & generate plots
│   └── 07_tune_thresholds.py  # Utility: Threshold tuning helper
├── run_pipeline.py            # Master runner (all stages)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

```bash
# 1. Create environment (Python 3.11 required)
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data
kaggle datasets download -d arghyasahoo/ibc53-indian-bird-call-dataset
# Unzip to data/ibc53/
git clone https://github.com/karolpiczak/ESC-50.git data/esc50

# 4. Run full pipeline
python run_pipeline.py

# Or run individual stages:
python run_pipeline.py --stage 1-4     # CST pipeline only (Paper 1)
python run_pipeline.py --stage 5-6     # Training + analysis (Paper 2)
python run_pipeline.py --stage 5 --experiment exp2  # Key experiment only
```

## Pipeline Stages

| Stage | Script | Description | Paper |
|-------|--------|-------------|-------|
| 1 | `01_segment_audio.py` | Segment raw audio → 3s chunks @ 48kHz | CST |
| 2 | `02_classify_segments.py` | Classify segments via RMS/flatness/ZCR | CST |
| 3 | `03_extract_esc50_noise.py` | Extract ESC-50 noise categories | CST |
| 4 | `04_build_dataset.py` | Build BirdNET-format dataset (with/without noise) | CST |
| 5 | `05_train_and_evaluate.py` | Train classifiers + evaluate experiments | DS/DL |
| 6 | `06_analyze_results.py` | Compare metrics, generate visualizations | DS/DL |
| 7 | `07_tune_thresholds.py` | Threshold tuning utility (run separately) | CST |

## Experiments

| Experiment | Training Data | Tests |
|------------|--------------|-------|
| Baseline | None (pre-trained BirdNET) | Generalization to Indian species |
| Exp 1 | 30 species, NO noise | Domain adaptation without noise handling |
| Exp 2 | 30 species + noise class | **KEY**: Impact of noise-aware training |
| Exp 3 | 10/25/50 samples + noise | Data efficiency / few-shot capability |

## Configuration

All paths, thresholds, and species lists are in `configs/config.py`. Key settings to tune:

- `SILENCE_RMS_THRESHOLD` — RMS below which = silence (default: 0.01)
- `NOISE_FLATNESS_THRESHOLD` — Spectral flatness above which = noise (default: 0.5)
- `NOISE_ZCR_THRESHOLD` — ZCR above which (with flatness) = noise (default: 0.1)

Use `07_tune_thresholds.py` to empirically tune these on your data.

## Dataset

- **IBC53**: 30 selected species (809 files, ~4854 segments)
- **ESC-50**: 7 noise categories (~280 files)
- **Mystery mystery**: 443 unclassified files (noise source + pipeline demo)
- **Total training classes**: 31 (30 species + 1 noise)
