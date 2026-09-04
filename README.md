# Video Profanity Censor

A command-line application that detects and censors profanity in video files. It transcribes the audio track with [Whisper](https://github.com/openai/whisper) (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)), matches spoken words against a profanity list, and produces a censored copy of the video where each profane word is either muted or replaced with a beep tone. A text report of every detection is generated alongside the output.

## Features

- Word-level detection using Whisper transcription with precise timestamps
- Two censoring modes: **mute** (silence) or **tone** (beep)
- Optional subtitle pre-filtering (SRT/ASS/SSA) to skip transcription when subtitles show no profanity
- Custom profanity lists with morphological matching (stemming), so variants of a word are caught
- Automatic Whisper model-size selection based on available system RAM
- A detailed detection report for every run

## Requirements

- **Python 3.11 or newer**
- **FFmpeg** installed and available on your `PATH` (used for probing, audio extraction, and output assembly)

### Installing FFmpeg

- **Windows:** `winget install Gyan.FFmpeg` (or `choco install ffmpeg`), then restart your terminal
- **macOS:** `brew install ffmpeg`
- **Linux (Debian/Ubuntu):** `sudo apt install ffmpeg`

Verify it is available:

```bash
ffmpeg -version
```

If this command fails, the app will report that FFmpeg is not installed or not found in `PATH`.

## First-Time Setup

These steps get the app installed from a fresh clone. Run them from the project root (the `clearplay` folder that contains `pyproject.toml`).

### 1. Get the code

```bash
git clone <your-repo-url>
cd clearplay
```

### 2. Create and activate a virtual environment

A virtual environment keeps the app's dependencies isolated from your system Python.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the application

Install it as an editable package. This also registers the `video-profanity-censor` command:

```bash
pip install -e .
```

To include the development tools (pytest, hypothesis, coverage), install the `dev` extra instead:

```bash
pip install -e ".[dev]"
```

### 4. Verify the install

```bash
video-profanity-censor --help
```

You should see the usage help. You can also run it as a module: `python -m video_profanity_censor --help`.

> **Note on the first run:** The first time you process a video, faster-whisper downloads the selected Whisper model (medium or large, depending on your RAM). This is a one-time download per model size and may take a few minutes. Later runs reuse the cached model.

## Usage

The only required argument is the path to a video file. By default the app mutes profanity and writes a new file next to the input.

### Basic example

```bash
video-profanity-censor movie.mp4
```

This produces:

- `movie_censored.mp4` — the censored video (only created if profanity is found)
- `movie_censored_report.txt` — a report of every detected instance

### Common examples

Replace profanity with a beep tone and choose the output name:

```bash
video-profanity-censor movie.mp4 --output clean.mp4 --mode tone
```

Use an external subtitle file to speed things up (transcription is skipped when subtitles contain no profanity):

```bash
video-profanity-censor movie.mkv --subtitle-path movie.srt
```

Use your own profanity list and force a specific model size:

```bash
video-profanity-censor movie.mp4 --profanity-list my_words.txt --model-size large
```

### Supported input formats

`.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv` (maximum 50 GB, must contain at least one audio track).

### Command-line options

| Option | Description | Default |
| --- | --- | --- |
| `input` | Path to the input video file (required) | — |
| `--output`, `-o` | Path for the censored output video | `<input>_censored.<ext>` |
| `--mode` | Censoring mode: `mute` (silence) or `tone` (beep) | `mute` |
| `--audio-track` | Index of the audio track to process | `0` |
| `--profanity-list` | Path to a custom profanity list (one word per line) | bundled default list |
| `--report-path` | Path for the detection report | `<output>_report.txt` |
| `--subtitle-path` | External subtitle file (SRT/ASS/SSA) for pre-filtering | none |
| `--disable-subtitle-prefilter` | Skip subtitle scanning and transcribe the full audio | off |
| `--model-size` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` | auto (by RAM) |

### Custom profanity lists

A profanity list is a plain-text file with one word per line. Blank lines and lines starting with `#` are ignored:

```text
# my custom list
word1
word2
```

Matching uses stemming, so common morphological variants of a listed word are also detected.

### Model size selection

When `--model-size` is not given, the app picks a Whisper model based on available system RAM:

- **16 GB or more** → `large` (best accuracy)
- **Less than 16 GB** → `medium` (good balance of speed and accuracy)

Transcription runs on the CPU. Smaller models are faster but less accurate; larger models are more accurate but slower and use more memory.

## How it works

The pipeline runs these stages in order:

1. **Validate** the input file (format, size, container, audio track)
2. **Extract** the selected audio track
3. **Select** the Whisper model size from available RAM
4. **Scan subtitles** (if enabled) to find candidate profanity regions
5. **Transcribe** the audio to word-level timestamps
6. **Detect** profanity by matching transcribed words against the list
7. **Censor** the audio (mute or tone) at each detected timestamp
8. **Assemble** the censored audio back into a copy of the video
9. **Generate** the detection report

If no profanity is found, no output video is created — only the report is written.

## Development

Install the dev dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
