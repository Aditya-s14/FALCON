# FALCON

**F**ine-tuned **A**coustic **L**earning for **C**lassification **O**f **N**ature (Birds)

A project for bird species identification using acoustic analysis and deep learning, focused on the Sundarbans/West Bengal region of India.

---

## Project Overview

FALCON aims to build a fine-tuned bird sound classification model using the BirdNET framework. The project collects location-specific bird audio recordings from the Xeno-Canto database and prepares them for training a custom classifier optimized for birds found in the Sundarbans delta region.

---

## Repository Structure

```
FALCON/
├── README.md                    # This file
├── collect_bird_audio.py        # Script to download bird audio from Xeno-Canto
├── BirdNET-Analyzer/            # BirdNET library for bird sound analysis
└── output/                      # Downloaded and organized audio data
    ├── xeno_canto/              # Raw audio downloads
    └── organized_audio/         # Audio files organized by species
```

---

## File & Folder Descriptions

### `collect_bird_audio.py`

Python script for collecting bird audio recordings from the [Xeno-Canto](https://xeno-canto.org/) database.

**Features:**
- Queries Xeno-Canto API v3 for recordings from a specific geographic region
- Downloads audio files with parallel processing for faster downloads
- Automatically organizes files into species-specific folders
- Saves metadata for all downloaded recordings

**Configuration:**
- **Target Region:** India (West Bengal / Sundarbans area)
- **Bounding Box:** `21.5504, 88.2518, 22.2017, 89.0905`
- **Max Recordings:** 100 (configurable)

**Usage:**
```bash
# Set your Xeno-Canto API key (get one free at https://xeno-canto.org/account)
export XENO_CANTO_API_KEY='your-api-key-here'

# Run the script
python collect_bird_audio.py
```

---

### `BirdNET-Analyzer/`

The BirdNET library - a deep learning solution for avian diversity monitoring developed by the Cornell Lab of Ornithology and Chemnitz University of Technology.

**Key Components:**

| Directory | Description |
|-----------|-------------|
| `birdnet_analyzer/` | Core Python package |
| `birdnet_analyzer/analyze/` | Audio analysis modules |
| `birdnet_analyzer/train/` | Model training utilities |
| `birdnet_analyzer/embeddings/` | Audio embedding extraction |
| `birdnet_analyzer/species/` | Species list management |
| `birdnet_analyzer/segments/` | Audio segmentation tools |
| `birdnet_analyzer/gui/` | Graphical user interface |
| `birdnet_analyzer/labels/` | Species labels in 30+ languages |
| `birdnet_analyzer/example/` | Example audio files |

**Capabilities:**
- Identifies 6,500+ bird species worldwide
- Processes audio files in various formats (WAV, MP3, FLAC)
- Supports batch processing of large audio datasets
- Includes tools for training custom classifiers
- Provides both CLI and GUI interfaces

**Documentation:** [BirdNET-Analyzer Docs](https://birdnet-team.github.io/BirdNET-Analyzer/)

**Citation:**
```bibtex
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

---

### `output/`

Contains all downloaded and processed audio data.

#### `output/xeno_canto/`

Raw audio files downloaded from Xeno-Canto.

| File | Description |
|------|-------------|
| `XC*.mp3` | Audio recordings (named by Xeno-Canto ID) |
| `metadata.json` | Complete metadata for all downloaded recordings |

**Metadata includes:** Species info, recording location, date, recordist, quality rating, and more.

#### `output/organized_audio/`

Audio files organized into species-specific folders for training.

**Folder naming convention:** `Genus species_Common Name`

**Current Species (30 total):**

| Species | Common Name | Recordings |
|---------|-------------|------------|
| *Anas poecilorhyncha* | Indian Spot-billed Duck | 15 |
| *Dendrocygna javanica* | Lesser Whistling Duck | 10 |
| *Anser indicus* | Bar-headed Goose | 10 |
| *Prinia lepida* | Delicate Prinia | 8 |
| *Anser anser* | Greylag Goose | 7 |
| *Tadorna ferruginea* | Ruddy Shelduck | 7 |
| *Arborophila torqueola* | Hill Partridge | 6 |
| *Canis aureus* | Golden Jackal | 3 |
| *Funambulus pennantii* | Northern Palm Squirrel | 3 |
| *Nettapus coromandelianus* | Cotton Pygmy Goose | 3 |
| *Spatula querquedula* | Garganey | 3 |
| ... | ... | ... |

> Note: The dataset includes some non-bird species (mammals, amphibians) that were recorded in the region.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aditya-s14/FALCON.git
   cd FALCON
   ```

2. **Install BirdNET-Analyzer:**
   ```bash
   pip install birdnet-analyzer
   # Or install from local directory:
   pip install -e BirdNET-Analyzer/
   ```

3. **Set up Xeno-Canto API key (for downloading more audio):**
   ```bash
   export XENO_CANTO_API_KEY='your-api-key'
   ```

### Quick Start

**Analyze an audio file:**
```bash
python -m birdnet_analyzer.analyze --i path/to/audio.mp3
```

**Download more bird audio:**
```bash
python collect_bird_audio.py
```

---

## Project Workflow

1. **Data Collection** (`collect_bird_audio.py`)
   - Query Xeno-Canto for regional bird recordings
   - Download and organize audio by species

2. **Data Preparation**
   - Review and clean audio samples
   - Balance dataset across species

3. **Model Training** (using BirdNET-Analyzer)
   - Fine-tune BirdNET on regional species
   - Validate model performance

4. **Deployment**
   - Deploy model for real-time bird identification

---

## Team Members

*Add team member names and roles here*

---

## License

- **FALCON Project:** MIT License
- **BirdNET-Analyzer:** 
  - Source Code: MIT License
  - Models: CC BY-NC-SA 4.0 (Non-commercial use)

---

## Acknowledgments

- [BirdNET](https://birdnet.cornell.edu/) - Cornell Lab of Ornithology
- [Xeno-Canto](https://xeno-canto.org/) - Bird sound database
- [K. Lisa Yang Center for Conservation Bioacoustics](https://www.birds.cornell.edu/ccb/)
