"""
Script 01: Transcribe Audio with WhisperX
==========================================
Purpose: Transcribe an audio file and extract word-level timestamps
         using WhisperX. Works for any audio file.

Usage:
    python scripts/transcribe.py --audio <path_to_audio>

Example:
    python scripts/transcribe.py --audio data/raw/brenebrown_5min.mp3

Input:  any .mp3 audio file
Output: data/processed/<audio_name>_transcript.json
"""

import whisperx
import json
import argparse
from pathlib import Path


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with word-level timestamps using WhisperX"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="Whisper model size: tiny, base, small, medium (default: base)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save transcript (default: data/processed)"
    )
    return parser.parse_args()


# Main
def main():
    args = parse_args()

    # Build paths from arguments
    audio_path = Path(args.audio)
    output_path = Path(args.output_dir) / f"{audio_path.stem}_transcript.json"

    # Checks
    assert audio_path.exists(), f"Audio file not found: {audio_path}"
    print(f"Audio file:  {audio_path}")
    print(f"Output file: {output_path}")

    # Step 1: Load audio
    print("\nStep 1: Loading audio...")
    audio = whisperx.load_audio(str(audio_path))
    print(f"Duration: {len(audio)/16000:.1f} seconds")

    # Step 2: Transcribe
    print(f"\nStep 2: Transcribing with Whisper ({args.model} model)...")
    model = whisperx.load_model(args.model, device="cpu", compute_type="int8")
    result = model.transcribe(audio, batch_size=4)
    print(f"Segments found: {len(result['segments'])}")

    # Step 3: Align for word-level timestamps 
    print("\nStep 3: Aligning for word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code="en", device="cpu"
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device="cpu"
    )
    print(f"Words aligned: {len(result['word_segments'])}")

    # Step 4: Save
    print("\nStep 4: Saving transcript...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_path}")

    # Step 5: Sanity check
    print("\nSanity check — first 10 words:")
    print(f"{'Word':<20} {'Start':>8} {'End':>8} {'Duration':>10}")
    print("-" * 50)
    for w in result["word_segments"][:10]:
        word  = w.get("word",  "")
        start = w.get("start", 0)
        end   = w.get("end",   0)
        print(f"{word:<20} {start:>8.3f} {end:>8.3f} {end-start:>10.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()