"""
Standalone feature extraction and classification functions.
Importable by multiple scripts without circular dependencies.
"""

import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    SAMPLE_RATE, SILENCE_RMS_THRESHOLD,
    NOISE_FLATNESS_THRESHOLD, NOISE_ZCR_THRESHOLD,
)


def extract_features(audio_path, sr: int = SAMPLE_RATE) -> Optional[dict]:
    """
    Extract signal processing features from a single audio segment.
    """
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as e:
        return None

    rms = float(np.sqrt(np.mean(y ** 2)))
    flatness_frames = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = float(np.mean(flatness_frames))
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
    Classify a segment as 'bird', 'noise', or 'silence'.
    """
    if features["rms"] < silence_rms:
        return "silence"
    elif features["spectral_flatness"] > noise_flatness and features["zcr"] > noise_zcr:
        return "noise"
    else:
        return "bird"
