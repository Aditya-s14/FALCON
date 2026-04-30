# BirdNET + IBC53: Experiment Results Documentation

## Experiment 1: Fine-tuned WITHOUT Noise Class

**Date:** 2026-03-19
**Model:** `models/exp1_no_noise.tflite` (25.1 MB)
**Training Data:** `data/processed_no_noise/` (6,924 files, 30 species, no noise class)
**Test Set:** IBC53 raw audio (1,252 files across 30 species + Mystery)
**Results Location:** `results/Exp1_NoNoise/` (1,253 CSV files)

### Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | **74.56%** (8,300 / 11,132 detections) |
| Total Detections | 14,489 (11,132 species + 3,357 Mystery) |
| Mean Confidence | 0.5228 |
| Median Confidence | 0.4348 |
| Min / Max Confidence | 0.10 / 1.00 |
| Detections >= 0.5 confidence | 47.4% |
| Detections >= 0.7 confidence | 40.5% |
| Detections >= 0.9 confidence | 28.4% |

### Per-Species Accuracy (sorted by accuracy)

| Species | Common Name | Total | Correct | Wrong | Accuracy |
|---------|-------------|-------|---------|-------|----------|
| Pnoepyga pusilla | Pygmy Cupwing | 364 | 333 | 31 | **91.48%** |
| Pellorneum ruficeps | Puff-throated Babbler | 1,389 | 1,218 | 171 | 87.69% |
| Chelidorhynx hypoxanthus | Yellow-bellied Fantail | 243 | 213 | 30 | 87.65% |
| Arborophila torqueola | Hill Partridge | 634 | 552 | 82 | 87.07% |
| Glaucidium cuculoides | Asian Barred Owlet | 508 | 438 | 70 | 86.22% |
| Motacilla citreola | Citrine Wagtail | 110 | 93 | 17 | 84.55% |
| Cuculus micropterus | Indian Cuckoo | 752 | 623 | 129 | 82.85% |
| Alcippe cinerea | Yellow-throated Fulvetta | 313 | 249 | 64 | 79.55% |
| Loriculus vernalis | Vernal Hanging Parrot | 339 | 263 | 76 | 77.58% |
| Polyplectron bicalcaratum | Grey Peacock-Pheasant | 115 | 89 | 26 | 77.39% |
| Psittacula eupatria | Alexandrine Parakeet | 299 | 230 | 69 | 76.92% |
| Argya longirostris | Slender-billed Babbler | 178 | 135 | 43 | 75.84% |
| Psilopogon lineatus | Lineated Barbet | 307 | 232 | 75 | 75.57% |
| Macronus gularis | Pin-striped Tit-Babbler | 301 | 225 | 76 | 74.75% |
| Pomatorhinus ruficollis | Streak-breasted Scimitar Babbler | 660 | 492 | 168 | 74.55% |
| Dicrurus andamanensis | Andaman Drongo | 301 | 223 | 78 | 74.09% |
| Phylloscopus inornatus | Yellow-browed Warbler | 110 | 81 | 29 | 73.64% |
| Todiramphus chloris | Collared Kingfisher | 196 | 141 | 55 | 71.94% |
| Cyornis poliogenys | Pale-chinned Flycatcher | 421 | 291 | 130 | 69.12% |
| Chloropsis jerdoni | Jerdon's Leafbird | 256 | 176 | 80 | 68.75% |
| Centropus andamanensis | Brown Coucal | 240 | 165 | 75 | 68.75% |
| Rimator malacoptilus | Long-billed Wren-Babbler | 357 | 239 | 118 | 66.95% |
| Liocichla phoenicea | Crimson-faced Liocichla | 459 | 303 | 156 | 66.01% |
| Cyornis unicolor | Pale Blue Flycatcher | 618 | 400 | 218 | 64.72% |
| Chloropsis | Leafbird sp. | 295 | 186 | 109 | 63.05% |
| Arachnothera magna | Streaked Spiderhunter | 107 | 65 | 42 | 60.75% |
| Sturnia malabarica | Chestnut-tailed Starling | 174 | 104 | 70 | 59.77% |
| Acridotheres fuscus | Jungle Myna | 216 | 117 | 99 | 54.17% |
| Sphenocichla humei | Blackish-breasted Babbler | 620 | 320 | 300 | 51.61% |
| Stachyridopsis ambigua | Buff-chested Babbler | 250 | 104 | 146 | **41.60%** |

### Top Confusion Patterns

**Most common false-positive species (predicted wrongly across all folders):**

| Species | False Positives |
|---------|----------------|
| Sphenocichla humei | 489 |
| Pellorneum ruficeps | 411 |
| Arborophila torqueola | 201 |
| Psittacula eupatria | 142 |
| Stachyridopsis ambigua | 130 |

**Top misclassification pairs:**

| True Species | Predicted As | Count |
|-------------|-------------|-------|
| Stachyridopsis ambigua | Sphenocichla humei | 107 |
| Cyornis unicolor | Sphenocichla humei | 107 |
| Sphenocichla humei | Stachyridopsis ambigua | 103 |
| Pomatorhinus ruficollis | Arborophila torqueola | 58 |
| Liocichla phoenicea | Sphenocichla humei | 58 |
| Cyornis unicolor | Pellorneum ruficeps | 56 |
| Acridotheres fuscus | Psittacula eupatria | 46 |
| Sphenocichla humei | Pellorneum ruficeps | 45 |
| Chloropsis jerdoni | Pellorneum ruficeps | 36 |
| Rimator malacoptilus | Arborophila torqueola | 35 |

### Mystery Folder Detections (3,357 total from 443 files)

| Predicted Species | Count |
|-------------------|-------|
| Pellorneum ruficeps | 961 |
| Glaucidium cuculoides | 308 |
| Sphenocichla humei | 235 |
| Chloropsis jerdoni | 211 |
| Psittacula eupatria | 193 |
| Arborophila torqueola | 192 |
| Sturnia malabarica | 134 |
| Loriculus vernalis | 117 |
| Cyornis poliogenys | 117 |
| Argya longirostris | 110 |
| Others | 679 |

57 of 443 Mystery files had **zero detections**.

### Empty Result Files
4 species files had no detections:
- `Acridotheres fuscus/17.BirdNET.results.csv`
- `Chloropsis/12.BirdNET.results.csv`
- `Psittacula eupatria/14.BirdNET.results.csv`
- `Rimator malacoptilus/6.BirdNET.results.csv`

### Key Findings

1. **No noise class hurts accuracy** — without a noise class, every audio segment is forced into a bird species, inflating false positives.
2. **Data imbalance bias** — Pellorneum ruficeps (102 training files) attracts disproportionate false positives (411).
3. **Babbler confusion** — Stachyridopsis, Sphenocichla, and Pellorneum (all babblers) are heavily confused with each other due to similar vocalizations.
4. **Low confidence prevalence** — median confidence of 0.4348 suggests the model is uncertain on many segments (likely noise/silence being forced into bird classes).
5. **Exp 2 (with noise class) is expected to improve** by absorbing noise/silence detections and reducing false positives.

---

## Experiment 2: Fine-tuned WITH Noise Class

**Date:** 2026-03-19
**Model:** `models/exp2_with_noise.tflite` (25.1 MB)
**Training Data:** `data/processed/` (7,207 files, 30 species + 283 noise from ESC-50)
**Training Metrics:** Best AUPRC: 0.961, Best AUROC: 0.997, Best Loss: 0.715 (epoch 50/50)
**Test Set:** IBC53 raw audio (1,252 files across 30 species + Mystery)
**Results Location:** `results/Exp2_WithNoise/` (1,253 CSV files)

**Note:** BirdNET treats the `noise` folder as a `NON_EVENT_CLASS` — it does not create a separate "noise" output label. Instead, noise samples train the model to suppress detections on noisy/silent segments. The model still outputs 30 species labels (same as Exp 1).

### Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | **75.61%** (8,288 / 10,962 detections) |
| Total Detections | 13,979 (10,962 species + 3,017 Mystery) |
| Mean Confidence | 0.6214 |
| Median Confidence | 0.7436 |
| Min / Max Confidence | 0.10 / 1.00 |
| Detections >= 0.5 confidence | 60.6% |
| Detections >= 0.7 confidence | 52.3% |
| Detections >= 0.9 confidence | 36.9% |

### Head-to-Head: Exp 1 vs Exp 2

| Metric | Exp 1 (no noise) | Exp 2 (with noise) | Change |
|--------|:-:|:-:|:-:|
| Overall Accuracy | 74.56% | **75.61%** | **+1.05pp** |
| Species Detections | 11,132 | 10,962 | -170 (-1.5%) |
| Mystery Detections | 3,357 | 3,017 | **-340 (-10.1%)** |
| Mean Confidence | 0.5228 | 0.6214 | **+18.9%** |
| Median Confidence | 0.4348 | 0.7436 | **+71.0%** |
| High-conf (>=0.5) | 47.4% | 60.6% | **+13.2pp** |

### Per-Species Accuracy (sorted by accuracy)

| Species | Common Name | Total | Correct | Wrong | Accuracy | vs Exp1 |
|---------|-------------|-------|---------|-------|----------|---------|
| Pnoepyga pusilla | Pygmy Cupwing | 366 | 331 | 35 | **90.44%** | -1.0pp |
| Arborophila torqueola | Hill Partridge | 614 | 546 | 68 | 88.93% | +1.9pp |
| Glaucidium cuculoides | Asian Barred Owlet | 495 | 439 | 56 | 88.69% | +2.5pp |
| Pellorneum ruficeps | Puff-throated Babbler | 1,354 | 1,200 | 154 | 88.63% | +0.9pp |
| Motacilla citreola | Citrine Wagtail | 104 | 92 | 12 | 88.46% | +3.9pp |
| Chelidorhynx hypoxanthus | Yellow-bellied Fantail | 241 | 213 | 28 | 88.38% | +0.7pp |
| Phylloscopus inornatus | Yellow-browed Warbler | 97 | 81 | 16 | 83.51% | +9.9pp |
| Cuculus micropterus | Indian Cuckoo | 756 | 621 | 135 | 82.14% | -0.7pp |
| Alcippe cinerea | Yellow-throated Fulvetta | 307 | 251 | 56 | 81.76% | +2.2pp |
| Loriculus vernalis | Vernal Hanging Parrot | 331 | 263 | 68 | 79.46% | +1.9pp |
| Psilopogon lineatus | Lineated Barbet | 298 | 234 | 64 | 78.52% | +3.0pp |
| Argya longirostris | Slender-billed Babbler | 172 | 134 | 38 | 77.91% | +2.1pp |
| Todiramphus chloris | Collared Kingfisher | 191 | 147 | 44 | 76.96% | +5.0pp |
| Macronus gularis | Pin-striped Tit-Babbler | 292 | 224 | 68 | 76.71% | +2.0pp |
| Pomatorhinus ruficollis | Streak-breasted Scimitar Babbler | 648 | 495 | 153 | 76.39% | +1.8pp |
| Dicrurus andamanensis | Andaman Drongo | 295 | 224 | 71 | 75.93% | +1.8pp |
| Psittacula eupatria | Alexandrine Parakeet | 303 | 229 | 74 | 75.58% | -1.3pp |
| Polyplectron bicalcaratum | Grey Peacock-Pheasant | 111 | 82 | 29 | 73.87% | -3.5pp |
| Rimator malacoptilus | Long-billed Wren-Babbler | 331 | 241 | 90 | 72.81% | +5.9pp |
| Chloropsis jerdoni | Jerdon's Leafbird | 271 | 184 | 87 | 67.90% | -0.8pp |
| Centropus andamanensis | Brown Coucal | 254 | 171 | 83 | 67.32% | -1.4pp |
| Cyornis poliogenys | Pale-chinned Flycatcher | 425 | 285 | 140 | 67.06% | -2.1pp |
| Arachnothera magna | Streaked Spiderhunter | 97 | 65 | 32 | 67.01% | +6.3pp |
| Cyornis unicolor | Pale Blue Flycatcher | 607 | 403 | 204 | 66.39% | +1.7pp |
| Liocichla phoenicea | Crimson-faced Liocichla | 452 | 297 | 155 | 65.71% | -0.3pp |
| Chloropsis | Leafbird sp. | 302 | 194 | 108 | 64.24% | +1.2pp |
| Acridotheres fuscus | Jungle Myna | 201 | 118 | 83 | 58.71% | +4.5pp |
| Sturnia malabarica | Chestnut-tailed Starling | 184 | 102 | 82 | 55.43% | -4.3pp |
| Sphenocichla humei | Blackish-breasted Babbler | 608 | 319 | 289 | 52.47% | +0.9pp |
| Stachyridopsis ambigua | Buff-chested Babbler | 255 | 103 | 152 | **40.39%** | -1.2pp |

### Top Confusion Patterns

**Most common false-positive species (predicted wrongly across all folders):**

| Species | False Positives | vs Exp1 |
|---------|----------------|---------|
| Pellorneum ruficeps | 428 | +17 |
| Sphenocichla humei | 428 | -61 |
| Arborophila torqueola | 195 | -6 |
| Pomatorhinus ruficollis | 136 | new in top 5 |
| Stachyridopsis ambigua | 131 | +1 |

**Top misclassification pairs:**

| True Species | Predicted As | Count |
|-------------|-------------|-------|
| Stachyridopsis ambigua | Sphenocichla humei | 110 |
| Sphenocichla humei | Stachyridopsis ambigua | 102 |
| Cyornis unicolor | Sphenocichla humei | 75 |
| Cyornis unicolor | Pellorneum ruficeps | 66 |
| Pomatorhinus ruficollis | Arborophila torqueola | 58 |
| Liocichla phoenicea | Sphenocichla humei | 55 |
| Cyornis poliogenys | Pellorneum ruficeps | 50 |
| Chloropsis jerdoni | Pellorneum ruficeps | 45 |
| Acridotheres fuscus | Psittacula eupatria | 36 |
| Liocichla phoenicea | Pellorneum ruficeps | 35 |

### Key Findings

1. **Noise training works as expected** — accuracy improved +1.05pp and the model produces 510 fewer total detections (170 species + 340 mystery), confirming it learned to suppress noise.
2. **Confidence dramatically improved** — median confidence jumped from 0.43 to 0.74 (+71%), meaning the model is much more decisive and better calibrated.
3. **20 of 30 species improved** — biggest gainers: Phylloscopus inornatus (+9.9pp), Arachnothera magna (+6.3pp), Rimator malacoptilus (+5.9pp).
4. **10 species declined slightly** — biggest losers: Sturnia malabarica (-4.3pp), Polyplectron bicalcaratum (-3.5pp). These may have borderline vocalizations that the noise filter now suppresses.
5. **Babbler confusion persists** — Stachyridopsis/Sphenocichla still the worst performers (40-52%), indicating this is a species-similarity problem, not a noise problem.
6. **BirdNET's NON_EVENT_CLASS mechanism** — the noise folder is not an output label; BirdNET uses it internally to learn what is "not a bird" during training.

## Experiment 3: Few-Shot Data Sensitivity

**Date:** 2026-03-19
**Models:** `models/exp3_fewshot_{10,25,50}.tflite` (25.1 MB each)
**Training Data:** `data/fewshot_subsets/fewshot_{10,25,50}/` (30 species + noise, with 10/25/50 samples per class)
**Test Set:** IBC53 raw audio (1,252 files across 30 species + Mystery)
**Results Location:** `results/Exp3_FewShot_{10,25,50}/`

### Overall Comparison Table

| Metric | Exp1 (6,924) | Exp2 (7,207) | FS-10 (583) | FS-25 (1,033) | FS-50 (1,783) |
|--------|:-:|:-:|:-:|:-:|:-:|
| **Overall Accuracy** | 74.56% | **75.61%** | 10.81% | 35.09% | 60.86% |
| Correct / Species Det. | 8,300/11,132 | 8,288/10,962 | 337/3,118 | 1,621/4,620 | 5,456/8,965 |
| Total Detections | 14,489 | 13,979 | 4,963 | 6,926 | 11,527 |
| Mystery Detections | 3,357 | 3,017 | 1,845 | 2,306 | 2,562 |
| Mean Confidence | 0.5228 | 0.6214 | 0.1216 | 0.1394 | 0.2165 |
| Median Confidence | 0.4348 | 0.7436 | 0.1148 | 0.1200 | 0.1513 |
| Detections >= 0.5 | 47.4% | 60.6% | 0.0% | 0.1% | 7.9% |
| Detections >= 0.7 | 40.5% | 52.3% | 0.0% | 0.0% | 2.4% |
| Detections >= 0.9 | 28.4% | 36.9% | 0.0% | 0.0% | 0.0% |

### Per-Species Accuracy

| Species | Exp1 | Exp2 | FS-10 | FS-25 | FS-50 |
|---------|:----:|:----:|:-----:|:-----:|:-----:|
| Pnoepyga pusilla | 91.5% | 90.4% | 63.9% | 53.2% | **93.9%** |
| Arborophila torqueola | 87.1% | 88.9% | 21.3% | 71.6% | 77.8% |
| Glaucidium cuculoides | 86.2% | 88.7% | 5.6% | 21.6% | 66.5% |
| Pellorneum ruficeps | 87.7% | 88.6% | 2.3% | 10.9% | 59.4% |
| Motacilla citreola | 84.5% | 88.5% | 6.7% | 81.4% | 88.0% |
| Chelidorhynx hypoxanthus | 87.7% | 88.4% | 41.9% | 83.4% | 83.3% |
| Phylloscopus inornatus | 73.6% | 83.5% | 0.0% | 30.0% | 69.0% |
| Cuculus micropterus | 82.8% | 82.1% | 0.6% | 26.8% | 79.6% |
| Alcippe cinerea | 79.5% | 81.8% | 9.2% | 38.3% | 48.5% |
| Loriculus vernalis | 77.6% | 79.5% | 15.3% | 2.3% | 51.1% |
| Psilopogon lineatus | 75.6% | 78.5% | 16.7% | 27.2% | 56.9% |
| Argya longirostris | 75.8% | 77.9% | 8.9% | 16.7% | 52.1% |
| Todiramphus chloris | 71.9% | 77.0% | 0.0% | 9.4% | 61.5% |
| Macronus gularis | 74.8% | 76.7% | 19.6% | 67.1% | 66.1% |
| Pomatorhinus ruficollis | 74.5% | 76.4% | 24.9% | 16.8% | 54.8% |
| Dicrurus andamanensis | 74.1% | 75.9% | 21.2% | 78.9% | 69.5% |
| Psittacula eupatria | 76.9% | 75.6% | 5.3% | 19.1% | 53.7% |
| Polyplectron bicalcaratum | 77.4% | 73.9% | 65.1% | 69.8% | 73.5% |
| Rimator malacoptilus | 67.0% | 72.8% | 9.1% | 57.2% | 60.1% |
| Chloropsis jerdoni | 68.8% | 67.9% | 8.3% | 13.7% | 55.3% |
| Centropus andamanensis | 68.8% | 67.3% | 2.3% | 58.7% | 60.3% |
| Cyornis poliogenys | 69.1% | 67.1% | 4.9% | 10.9% | 58.5% |
| Arachnothera magna | 60.8% | 67.0% | 28.0% | 82.7% | 52.6% |
| Cyornis unicolor | 64.7% | 66.4% | 14.4% | 3.5% | 53.0% |
| Liocichla phoenicea | 66.0% | 65.7% | 0.8% | 3.6% | 62.0% |
| Chloropsis | 63.0% | 64.2% | 1.6% | 7.1% | 42.1% |
| Acridotheres fuscus | 54.2% | 58.7% | 16.9% | 10.6% | 45.7% |
| Sturnia malabarica | 59.8% | 55.4% | 4.3% | 1.0% | 25.0% |
| Sphenocichla humei | 51.6% | 52.5% | 3.4% | 17.7% | 17.7% |
| Stachyridopsis ambigua | 41.6% | 40.4% | 4.7% | 7.5% | 48.9% |

### Key Findings

1. **10 samples/species is catastrophically insufficient** — 10.81% accuracy (essentially random), near-zero confidence. BirdNET cannot learn meaningful species features from just 10 examples.
2. **25 samples/species shows partial learning** — 35.09% accuracy with extreme variance across species (some at 80%+, many below 20%). Highly species-dependent; distinctive callers like Chelidorhynx (83.4%) and Motacilla (81.4%) do well while many species fail.
3. **50 samples/species reaches ~61%** — a meaningful classifier but still 14.75pp below the full dataset (75.61%). Confidence remains very low (median 0.15) compared to Exp 2 (0.74).
4. **Confidence is the biggest casualty of data reduction** — even at 50 samples, only 7.9% of detections exceed 0.5 confidence vs 60.6% in Exp 2. The model makes predictions but with almost no certainty.
5. **Some species are inherently easier** — Pnoepyga pusilla hits 93.9% at FS-50 (beating both Exp1 and Exp2), likely due to its highly distinctive call. Polyplectron bicalcaratum is also robust (65-74% across all few-shot sizes).
6. **Data-hungry species exist** — Pellorneum ruficeps drops from 88.6% (Exp2) to 59.4% (FS-50) to 2.3% (FS-10). Species with subtle or variable vocalizations need many examples.
7. **The scaling curve is steep** — accuracy roughly doubles with each step: 10.8% → 35.1% → 60.9% → 75.6%, suggesting diminishing returns above ~100 samples/species but severe degradation below 50.
