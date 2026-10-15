#!/usr/bin/env python3
"""Transcribe an audio file to text with a Hugging Face Whisper model.

Usage:
    python transcribe.py path/to/audio.mp3
    python transcribe.py meeting.wav --model openai/whisper-small --language en -o meeting.txt
    python transcribe.py interview.m4a --timestamps
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL = "openai/whisper-small"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert speech in an audio file to text using a Whisper model from Hugging Face."
    )
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, m4a, flac, ogg, ...)")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-l", "--language", default=None,
        help="Spoken language, e.g. 'en', 'hindi', 'fr'. Omit to auto-detect.",
    )
    parser.add_argument(
        "-t", "--task", choices=["transcribe", "translate"], default="transcribe",
        help="'transcribe' keeps the original language; 'translate' outputs English.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write the text to this file instead of only printing it.",
    )
    parser.add_argument(
        "--timestamps", action="store_true",
        help="Prefix each segment with [start -> end] timestamps.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Hardware to run on (default: auto-detect).",
    )
    return parser.parse_args()


def pick_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_time(seconds):
    if seconds is None:
        return "--:--:--"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def convert_m4a_to_mp3(m4a_path, out_dir):
    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg is required to convert m4a files. Install it with 'brew install ffmpeg'.")

    mp3_path = Path(out_dir) / (Path(m4a_path).stem + ".mp3")
    print(f"Converting {m4a_path} to mp3...", file=sys.stderr)
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(m4a_path), "-vn", "-q:a", "2", str(mp3_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Error: could not convert {m4a_path} to mp3:\n{result.stderr.strip()}")
    return str(mp3_path)


def transcribe(args, audio_path):
    import torch
    from transformers import pipeline
    from transformers.utils import logging

    logging.set_verbosity_error()
    device = pick_device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {args.model} on {device}...", file=sys.stderr)
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        dtype=dtype,
        device=device,
    )

    generate_kwargs = {"task": args.task}
    if args.language:
        generate_kwargs["language"] = args.language

    print(f"Transcribing {audio_path}...", file=sys.stderr)
    result = asr(
        audio_path,
        # Timestamps are always requested: Whisper needs them to transcribe audio longer than 30 seconds.
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )

    if not args.timestamps:
        return result["text"].strip()

    lines = []
    for chunk in result["chunks"]:
        start, end = chunk["timestamp"]
        lines.append(f"[{format_time(start)} -> {format_time(end)}] {chunk['text'].strip()}")
    return "\n".join(lines)


def main():
    args = parse_args()

    if not Path(args.audio).is_file():
        sys.exit(f"Error: audio file not found: {args.audio}")

    # m4a files are converted to a temporary mp3 first; the temp file is deleted afterwards.
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = args.audio
        if Path(args.audio).suffix.lower() == ".m4a":
            audio_path = convert_m4a_to_mp3(args.audio, tmp_dir)
        text = transcribe(args, audio_path)
    print(text)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Saved transcript to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
