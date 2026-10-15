# voice-to-text

Turn speech in an audio file into text, using a [Whisper](https://huggingface.co/openai/whisper-small) model from Hugging Face. Everything runs locally on your machine.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) (needed to read audio files)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

## Setup

```bash
git clone https://github.com/techbyavanti/voice-to-text.git
cd voice-to-text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the model (about 1 GB for `whisper-small`), so it takes a while. Later runs reuse the downloaded copy.

## Usage

```bash
python transcribe.py path/to/audio.mp3
```

The transcript is printed to the terminal. Supported formats include mp3, wav, m4a, flac and ogg. m4a files are converted to mp3 automatically.

### Examples

Save the transcript to a file:

```bash
python transcribe.py meeting.wav -o meeting.txt
```

Add timestamps to each segment:

```bash
python transcribe.py interview.m4a --timestamps
```

```
[00:00:00 -> 00:00:04] Thanks for joining today.
[00:00:04 -> 00:00:09] Let's start with a quick introduction.
```

Set the spoken language instead of auto-detecting it:

```bash
python transcribe.py talk.mp3 --language hindi
```

Translate speech in any language into English text:

```bash
python transcribe.py talk.mp3 --task translate
```

Use a different model (larger is more accurate but slower):

```bash
python transcribe.py audio.mp3 --model openai/whisper-large-v3
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `audio` | Path to the audio file | required |
| `-m`, `--model` | Hugging Face model id | `openai/whisper-small` |
| `-l`, `--language` | Spoken language, e.g. `en`, `hindi`, `fr` | auto-detect |
| `-t`, `--task` | `transcribe` (keep original language) or `translate` (output English) | `transcribe` |
| `-o`, `--output` | Also write the transcript to this file | print only |
| `--timestamps` | Prefix each segment with `[start -> end]` | off |
| `--device` | `auto`, `cpu`, `cuda` or `mps` (Apple Silicon) | `auto` |

## Choosing a model

| Model | Speed | Accuracy |
| --- | --- | --- |
| `openai/whisper-tiny` | fastest | lowest |
| `openai/whisper-base` | fast | fair |
| `openai/whisper-small` | moderate | good (default) |
| `openai/whisper-medium` | slow | better |
| `openai/whisper-large-v3` | slowest | best |

For English-only audio, the `.en` variants (for example `openai/whisper-small.en`) are a little more accurate.
