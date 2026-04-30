"""
Importable library version of classify_segments functions.
Used by 07_tune_thresholds.py to avoid circular imports.
"""

# Re-export from the main script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import _classify_lib as _lib

extract_features = _lib.extract_features
classify_segment = _lib.classify_segment
