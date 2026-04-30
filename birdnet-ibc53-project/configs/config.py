"""
BirdNET + IBC53 Project Configuration
======================================
Central configuration for all pipeline scripts.
Modify paths, thresholds, and species lists here.

Project: Noise-Aware Bird Audio Classification Pipeline
Version: 2.0 | March 2026
"""

import os
from pathlib import Path

# ============================================================
# PROJECT ROOT (auto-detected from this file's location)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# DATA PATHS
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
IBC53_RAW_DIR = DATA_DIR / "ibc53"                      # Raw IBC53 dataset (53 species + Mystery)
ESC50_DIR = DATA_DIR / "esc50"                           # ESC-50 dataset clone
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_META_CSV = ESC50_DIR / "meta" / "esc50.csv"

# Pipeline output directories
PROCESSED_DIR = DATA_DIR / "processed"                   # Full noise-aware dataset (30 species + noise)
PROCESSED_NO_NOISE_DIR = DATA_DIR / "processed_no_noise" # 30 species only (Experiment 1)
SEGMENTS_DIR = DATA_DIR / "segments"                     # Intermediate segmented audio
MYSTERY_DIR = DATA_DIR / "mystery_processed"             # Mystery mystery folder processing output

# ============================================================
# MODEL & RESULTS PATHS
# ============================================================
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# AUDIO PARAMETERS
# ============================================================
SAMPLE_RATE = 48000          # BirdNET's expected sample rate (Hz)
SEGMENT_LENGTH = 3.0         # BirdNET's internal window size (seconds)
SEGMENT_SAMPLES = int(SEGMENT_LENGTH * SAMPLE_RATE)  # 144000 samples
MIN_SEGMENT_RATIO = 0.5      # Keep segments with >= 50% of target length
AUDIO_FORMAT = "WAV"
CHANNELS = 1                 # Mono

# ============================================================
# NOISE DETECTION THRESHOLDS (STARTING POINTS — TUNE THESE!)
# ============================================================
# These are initial values. You MUST tune them empirically by
# listening to ~50-100 IBC53 segments. Tuning is part of the
# research contribution.

SILENCE_RMS_THRESHOLD = 0.01       # Below this RMS = silence
NOISE_FLATNESS_THRESHOLD = 0.5     # Above this spectral flatness = noise-like
NOISE_ZCR_THRESHOLD = 0.1          # Above this ZCR (combined with flatness) = noise

# ============================================================
# ESC-50 NOISE CATEGORIES
# ============================================================
ESC50_NOISE_CATEGORIES = [
    "rain",
    "wind",
    "water_drops",
    "insects",
    "thunderstorm",
    "crackling_fire",
    "pouring_water",
]

# ============================================================
# SELECTED SPECIES (30 species with >= 10 audio files)
# ============================================================
# Format: (scientific_name, common_name, expected_files)
# The folder names in IBC53 use scientific names.

SELECTED_SPECIES = [
    # Tier 1 — Strong Data (15+ files) — 25 species
    ("Pellorneum ruficeps",       "Puff-throated Babbler",           102),
    ("Cuculus micropterus",       "Indian Cuckoo",                    66),
    ("Sphenocichla humei",        "Blackish-breasted Babbler",        43),
    ("Glaucidium cuculoides",     "Asian Barred Owlet",               41),
    ("Pomatorhinus ruficollis",   "Streak-breasted Scimitar Babbler", 41),
    ("Loriculus vernalis",        "Vernal Hanging Parrot",            34),
    ("Arborophila torqueola",     "Hill Partridge",                   33),
    ("Cyornis poliogenys",        "Pale-chinned Flycatcher",          29),
    ("Pnoepyga pusilla",          "Pygmy Cupwing",                    29),
    ("Cyornis unicolor",          "Pale Blue Flycatcher",             28),
    ("Psittacula eupatria",       "Alexandrine Parakeet",             28),
    ("Alcippe cinerea",           "Yellow-throated Fulvetta",         24),
    ("Macronus gularis",          "Pin-striped Tit-Babbler",          24),
    ("Liocichla phoenicea",       "Crimson-faced Liocichla",          23),
    ("Todiramphus chloris",       "Collared Kingfisher",              23),
    ("Psilopogon lineatus",       "Lineated Barbet",                  22),
    ("Chloropsis",                "Leafbird sp.",                     21),
    ("Dicrurus andamanensis",     "Andaman Drongo",                   20),
    ("Rimator malacoptilus",      "Long-billed Wren-Babbler",         20),
    ("Motacilla citreola",        "Citrine Wagtail",                  19),
    ("Sturnia malabarica",        "Chestnut-tailed Starling",         18),
    ("Acridotheres fuscus",       "Jungle Myna",                      17),
    ("Centropus andamanensis",    "Brown Coucal",                     16),
    ("Chelidorhynx hypoxanthus",  "Yellow-bellied Fantail",           16),
    ("Stachyridopsis ambigua",    "Buff-chested Babbler",             15),
    # Tier 2 — Acceptable Data (10-14 files) — 5 species
    ("Chloropsis jerdoni",        "Jerdon's Leafbird",                13),
    ("Arachnothera magna",        "Streaked Spiderhunter",            11),
    ("Argya longirostris",        "Slender-billed Babbler",           11),
    ("Phylloscopus inornatus",    "Yellow-browed Warbler",            11),
    ("Polyplectron bicalcaratum", "Grey Peacock-Pheasant",            11),
]

# Quick lookup sets
SPECIES_NAMES = [s[0] for s in SELECTED_SPECIES]
SPECIES_COMMON_NAMES = {s[0]: s[1] for s in SELECTED_SPECIES}

# ============================================================
# MYSTERY MYSTERY FOLDER
# ============================================================
MYSTERY_FOLDER_NAME = "Mystery mystery"
MYSTERY_EXPECTED_FILES = 443

# ============================================================
# BIRDNET NON-EVENT CLASS NAMES (any of these work)
# ============================================================
NOISE_FOLDER_NAME = "noise"  # BirdNET recognizes: noise, other, background, silence

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
EXPERIMENTS = {
    "baseline": {
        "description": "Pre-trained BirdNET, no fine-tuning",
        "training_data": None,
        "classifier": None,
    },
    "exp1_no_noise": {
        "description": "Fine-tuned on 30 species WITHOUT noise class",
        "training_data": "processed_no_noise",
        "classifier": "Exp1_NoNoise",
    },
    "exp2_with_noise": {
        "description": "Fine-tuned on 30 species WITH noise class (KEY experiment)",
        "training_data": "processed",
        "classifier": "Exp2_WithNoise",
    },
    "exp3_fewshot": {
        "description": "Data size sensitivity: 10/25/50 samples per species + noise",
        "classifier_prefix": "Exp3_FewShot",
        "sample_sizes": [10, 25, 50],
    },
}

# Minimum confidence for BirdNET analysis
MIN_CONFIDENCE = 0.1

# ============================================================
# LOGGING
# ============================================================
LOG_DIR = PROJECT_ROOT / "logs"
VERBOSE = True


def ensure_dirs():
    """Create all required project directories."""
    dirs = [
        DATA_DIR, IBC53_RAW_DIR, ESC50_DIR,
        PROCESSED_DIR, PROCESSED_NO_NOISE_DIR,
        SEGMENTS_DIR, MYSTERY_DIR,
        MODELS_DIR, RESULTS_DIR, LOG_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration for verification."""
    print("=" * 60)
    print("BirdNET + IBC53 Project Configuration")
    print("=" * 60)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"IBC53 data:       {IBC53_RAW_DIR}")
    print(f"ESC-50 data:      {ESC50_DIR}")
    print(f"Processed output: {PROCESSED_DIR}")
    print(f"Models output:    {MODELS_DIR}")
    print(f"Results output:   {RESULTS_DIR}")
    print(f"Sample rate:      {SAMPLE_RATE} Hz")
    print(f"Segment length:   {SEGMENT_LENGTH} s")
    print(f"Selected species: {len(SELECTED_SPECIES)}")
    print(f"Noise categories: {len(ESC50_NOISE_CATEGORIES)}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
