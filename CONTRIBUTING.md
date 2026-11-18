# Contributing to vtd

Thanks for taking the time. Bug reports, fixes and small improvements are all welcome.

## Reporting a problem

Open an issue with:

- the command you ran and what you expected,
- what happened instead (the last lines of output; `whisper.log` in the workspace when transcription failed),
- your OS, Python version and `vtd --version`.

Please do not attach private recordings, transcripts or API keys. A short synthetic clip that shows the problem is ideal.

## Setting up

You need Python 3.11+, and `ffmpeg` and `tesseract` if you want the integration tests to run rather than be skipped.

```bash
git clone https://github.com/Mariotutal/video-to-doc.git
cd video-to-doc
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Then, before every pull request:

```bash
pytest
ruff check .
```

The tests never download a Whisper model and never call an AI provider: both are faked. Please keep it that way, so the suite stays fast and free to run.

## How the code is organised

- `src/vtd/cli.py` parses the command line; `config.py` merges flags, environment and `~/.config/vtd/config.toml` into one `VTDConfig`.
- `src/vtd/pipeline.py` decides which stages run. The stages are modules in `src/vtd/stages/`, each with a function taking `(workspace, config, manifest)` that reads and writes files in the workspace: `run` in most, `run_starter` in `render.py`. The AI stage is `synthesize` in `src/vtd/ai_synth.py`. The `STAGES` table in `pipeline.py` lists the non-AI ones in order.
- `manifest.json` in the workspace records which stages finished and with which options. That record is what lets `--resume` skip work that is still valid, so a new option that changes a stage's output must be added to `_stage_options` in `pipeline.py`.
- `src/vtd/ai/base.py` holds every prompt, the response cache and the answer parsing. A vendor module only says how to attach a frame and how to send one request.
- `src/vtd/vtt.py` reads and writes the transcript as timed cues, and `read_json` in `utils.py` reads the workspace's JSON files; stages go through them instead of parsing on their own.
- Documents are Jinja2 templates in `src/vtd/templates/` (the four AI ones share `_parts.md.j2`); their fixed text lives in `src/vtd/stages/labels.py`, one table per output language.

## What a pull request should include

- A test for the behaviour you change or add. Tests are named after the behaviour they check, in plain words.
- An update to `README.md` when a flag, a config key or user-visible behaviour changes, and a line in `CHANGELOG.md`.
- Error messages a user can act on: say what is wrong and what to do about it, and raise `VTDError` so it is shown without a traceback.
- No secrets, recordings or workspaces (`*.vtd/` is ignored by git).

Keep pull requests focused: one change per pull request is easier to review than several bundled together.

## Adding things

- **An output language:** add a table to `LABELS` in `stages/labels.py` and the language to `LANGUAGE_NAMES` in `config.py`.
- **A content type:** add it to `CONTENT_TYPES` in `config.py`, write `templates/full_<type>.md.j2`, and add its entries to `SOURCE_NOUNS` and `FOLLOWUP_INSTRUCTIONS` in `ai/base.py`.
- **An AI provider:** subclass `ChatProvider` in `ai/`, implement `_frame_block` and `_request`, and register it in `PROVIDERS` in `config.py` and in `_build_provider` in `ai_synth.py`.

## License

By contributing you agree that your contribution is released under the [MIT license](LICENSE).
