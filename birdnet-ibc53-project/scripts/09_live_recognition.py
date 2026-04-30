"""
Script 9: Live Bird Sound Recognition
=======================================
Captures audio from your microphone in real-time and identifies bird species
using the fine-tuned BirdNET model.

How it works:
  1. Continuously records audio from your microphone at 48kHz mono
  2. Buffers into 3-second segments (BirdNET's window)
  3. Runs each segment through the .tflite model
  4. Displays top predictions with confidence scores

Usage:
    python scripts/09_live_recognition.py
    python scripts/09_live_recognition.py --model models/exp2_with_noise.tflite
    python scripts/09_live_recognition.py --threshold 0.3 --top_k 3
    python scripts/09_live_recognition.py --list-devices

Controls:
    Press Ctrl+C to stop.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import SPECIES_COMMON_NAMES, MODELS_DIR


# ============================================================
# Constants
# ============================================================
SAMPLE_RATE = 48000
SEGMENT_DURATION = 3.0  # seconds
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # 144000


# ============================================================
# Model
# ============================================================

class BirdClassifier:
    """Wraps a fine-tuned BirdNET .tflite model for inference."""

    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load labels (same name as model, with _Labels.txt)
        label_path = Path(model_path).with_name(
            Path(model_path).stem + "_Labels.txt"
        )
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        self.labels = label_path.read_text().strip().splitlines()
        self.common_names = {
            sp: SPECIES_COMMON_NAMES.get(sp, sp) for sp in self.labels
        }

    def predict(self, audio: np.ndarray) -> list:
        """
        Run inference on a 3-second audio chunk.

        Args:
            audio: float32 array of shape (144000,) — 3s at 48kHz mono

        Returns:
            List of (scientific_name, common_name, confidence) sorted by confidence desc
        """
        # Ensure correct shape and type
        if len(audio) < SEGMENT_SAMPLES:
            audio = np.pad(audio, (0, SEGMENT_SAMPLES - len(audio)))
        elif len(audio) > SEGMENT_SAMPLES:
            audio = audio[:SEGMENT_SAMPLES]

        audio = audio.astype(np.float32).reshape(1, SEGMENT_SAMPLES)

        # Run model
        self.interpreter.resize_tensor_input(
            self.input_details[0]["index"], audio.shape
        )
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]["index"], audio)
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # Build results
        results = []
        for idx, score in enumerate(scores):
            sp = self.labels[idx]
            results.append((sp, self.common_names[sp], float(score)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results


# ============================================================
# Display
# ============================================================

def clear_line():
    """Move cursor up and clear line."""
    sys.stdout.write("\033[F\033[K")


def format_bar(confidence: float, width: int = 20) -> str:
    """Create a visual confidence bar."""
    filled = int(confidence * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def display_predictions(results: list, threshold: float, top_k: int,
                        segment_num: int, display_lines: int):
    """Display top predictions in a clean format."""
    # Clear previous output
    for _ in range(display_lines):
        clear_line()

    filtered = [(sp, common, conf) for sp, common, conf in results if conf >= threshold]
    top = filtered[:top_k]

    print(f"--- Segment #{segment_num} " + "-" * 40)

    if not top:
        print(f"  (no detections above {threshold:.0%} confidence)")
        return 2

    lines = 1
    for rank, (sp, common, conf) in enumerate(top, 1):
        bar = format_bar(conf)
        print(f"  {rank}. {common:<35s} {bar} {conf:.1%}")
        lines += 1

    return lines


# ============================================================
# Main Loop
# ============================================================

def run_live(model_path: str, threshold: float, top_k: int,
             device: int = None, overlap: float = 0.0):
    """Run live bird sound recognition."""

    print("=" * 55)
    print("  LIVE BIRD SOUND RECOGNITION")
    print("=" * 55)
    print(f"  Model:      {Path(model_path).name}")
    print(f"  Threshold:  {threshold:.0%}")
    print(f"  Top-K:      {top_k}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Segment:    {SEGMENT_DURATION}s ({SEGMENT_SAMPLES} samples)")

    # List audio devices
    if device is not None:
        print(f"  Device:     #{device}")
    else:
        default = sd.query_devices(kind="input")
        print(f"  Device:     {default['name']} (default)")

    print("=" * 55)
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    classifier = BirdClassifier(model_path)
    print(f"OK ({len(classifier.labels)} species)")
    print()
    print("Listening... (press Ctrl+C to stop)")
    print()

    # Audio buffer
    buffer = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    buffer_pos = 0
    segment_num = 0
    display_lines = 0

    # Calculate step size (how many new samples per segment)
    overlap_samples = int(overlap * SAMPLE_RATE)
    step_samples = SEGMENT_SAMPLES - overlap_samples

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        nonlocal buffer, buffer_pos

        if status:
            pass  # Ignore overflow warnings in live mode

        # Take mono channel
        mono = indata[:, 0].astype(np.float32)

        # Fill buffer
        remaining = SEGMENT_SAMPLES - buffer_pos
        if len(mono) <= remaining:
            buffer[buffer_pos:buffer_pos + len(mono)] = mono
            buffer_pos += len(mono)
        else:
            buffer[buffer_pos:] = mono[:remaining]
            buffer_pos = SEGMENT_SAMPLES  # Mark as full

    # Print initial placeholder lines
    print(f"--- Waiting for first {SEGMENT_DURATION}s of audio... ---")
    print("  (play a bird sound near the microphone)")
    display_lines = 2

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", device=device,
                            callback=audio_callback,
                            blocksize=int(SAMPLE_RATE * 0.1)):  # 100ms blocks
            while True:
                # Wait until buffer is full
                if buffer_pos >= SEGMENT_SAMPLES:
                    segment_num += 1
                    audio_chunk = buffer.copy()

                    # Shift buffer for overlap
                    if overlap_samples > 0:
                        buffer[:overlap_samples] = buffer[step_samples:]
                        buffer_pos = overlap_samples
                    else:
                        buffer_pos = 0

                    # Run inference
                    results = classifier.predict(audio_chunk)

                    # Display
                    display_lines = display_predictions(
                        results, threshold, top_k, segment_num, display_lines
                    )
                else:
                    time.sleep(0.05)  # Small sleep to avoid busy-waiting

    except KeyboardInterrupt:
        print(f"\n\nStopped after {segment_num} segments.")


def list_devices():
    """List available audio input devices."""
    print("\nAvailable audio input devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " (DEFAULT)" if idx == sd.default.device[0] else ""
            print(f"  [{idx}] {dev['name']}{default}")
            print(f"       Channels: {dev['max_input_channels']}, "
                  f"Sample Rate: {dev['default_samplerate']:.0f} Hz")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Live bird sound recognition using fine-tuned BirdNET."
    )
    parser.add_argument(
        "--model", type=str,
        default=str(MODELS_DIR / "exp2_with_noise.tflite"),
        help="Path to .tflite model (default: exp2_with_noise)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Minimum confidence to display (default: 0.2)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to show (default: 5)"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (use --list-devices to see options)"
    )
    parser.add_argument(
        "--overlap", type=float, default=1.5,
        help="Overlap between segments in seconds (default: 1.5)"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices and exit"
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        return

    run_live(
        model_path=args.model,
        threshold=args.threshold,
        top_k=args.top_k,
        device=args.device,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
