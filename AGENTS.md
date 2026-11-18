# video-to-doc (vtd) Agent Instructions

## Project Overview
`vtd` turns videos, audio files, and URLs into structured Markdown documents using deterministic processing with optional AI synthesis.

## Installation
```bash
./install.sh
```
- Installs dependencies via Homebrew (macOS) or apt-get (Linux)
- Creates Python venv and installs package in editable mode with all extras
- Downloads default Whisper model (~1.5GB) to `~/.cache/vtd/models` (or to `VTD_MODEL_DIR` when set)

## Core Commands
- **Deterministic run**: `vtd video.mp4` 
- **AI synthesis**: `vtd video.mp4 --with-ai`
- **PDF copies of the documents**: add `--pdf` (needs pandoc and xelatex)
- **Verified bibliography**: add `--references` to `--with-ai` (footnotes and a bibliography; works are looked up in Open Library and Crossref, in `src/vtd/references.py`)
- **Who speaks when**: add `--speakers` (needs the `speakers` extra: sherpa-onnx and numpy)
- **With specific provider**: `vtd video.mp4 --with-ai --provider openai`

## Testing
To run tests:
```bash
source .venv/bin/activate
pip install -e ".[dev]"  # Install test dependencies
pytest tests/            # Run all tests
pytest tests/test_cli.py -v  # Run specific test file
```

## Key Features
- **Stages**: extract-audio, extract-frames, transcribe, identify-speakers (only with `--speakers`), ocr-speakers, detect-sections, render-starter, synthesize
- **Content types**: generic (default), meeting, lecture, podcast
- **Providers**: anthropic (default), openai, ollama (local model, no key)
- **Languages**: multilingual by default; specify with `--language` flag

## Workspace Layout
Per input file, vtd creates a `.vtd/` directory:
```
<basename>.vtd/
├── manifest.json
├── audio.wav
├── frames/
├── transcript.txt
├── speakers.json
├── sections.json
├── notes.starter.md  # deterministic output
├── notes.md          # AI-augmented (with --with-ai)
└── ai-cache/         # cached LLM responses
```

## Configuration
Optional `~/.config/vtd/config.toml` with defaults section and ai section.
Whisper models are kept in `~/.cache/vtd/models`; change it with `--model-dir` or `model_dir` under `[defaults]` (e.g. an external disk).

## Development Notes
- Run with `source .venv/bin/activate` to use the installed package
- Tests require `pip install -e ".[dev]"`
- Entry point: `src/vtd/cli.py`
- All AI providers in `src/vtd/ai/`