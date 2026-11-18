# Changelog

Notable changes to vtd. Versions follow [semantic versioning](https://semver.org/).

## 0.1.0

First public release.

- Without AI, on your machine: audio and frame extraction with `ffmpeg`, transcription with `whisper.cpp`, on-screen speaker labels with `tesseract`, sections cut where the speaker pauses, and a starter Markdown document.
- Inputs: video files, audio files and URLs (through `yt-dlp`). `--teams-captions` reads silent Microsoft Teams recordings from their visible live captions.
- AI notes (`--with-ai`) with Anthropic, OpenAI, or a model on your machine through Ollama; four content types (generic, meeting, lecture, podcast); English or Spanish.
- Quotations in the notes are checked against the transcript; follow-ups are read from each section's transcript; long sections are split and short ones joined; lines Whisper repeats are removed.
- `--speakers`: tell who speaks when, locally, with sherpa-onnx.
- `--references`: cite the books and papers the recording refers to, only when Open Library or Crossref has them.
- `--pdf`: the documents as PDF, with an index, the cited frames and footnotes.
- `--resume` redoes only the stages whose options changed; `--only` runs chosen stages; model answers are cached in the workspace.
- Configuration file at `~/.config/vtd/config.toml`; `--model-dir` keeps the Whisper models in another folder.
