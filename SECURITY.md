# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting instead: on the repository page, open the **Security** tab and choose **Report a vulnerability**.

Include what you found, how to reproduce it and which version or commit you used. You should get a first answer within a week.

## What to keep in mind when using vtd

- **API keys.** vtd reads a key from `--api-key`, an environment variable or `~/.config/vtd/config.toml`. It never writes the key to the workspace, the manifest or the logs. Prefer the environment variable; a key passed with `--api-key` can end up in your shell history.
- **What leaves your machine.** Nothing, unless you ask for it: `--with-ai` sends transcript excerpts and sampled frames to Anthropic or OpenAI (with `--provider ollama` they go to the Ollama server, which is the one on your machine unless `$OLLAMA_HOST` points elsewhere), `--references` sends the title and the first author's surname of each work the model proposes, which can be names and titles said in the recording but not the transcript itself, as searches to `openlibrary.org` and `api.crossref.org`, and a URL input is downloaded with `yt-dlp`. Models are downloaded on first use: Whisper models from Hugging Face, and with `--speakers` two speaker models from the sherpa-onnx releases on GitHub (the voice model is checked against a known SHA-256; the other comes as an archive that is unpacked without following links).
- **Workspaces contain your content.** A `*.vtd/` folder holds the transcript, frames and notes of the recording. Treat it like the recording itself.
- **Untrusted media.** Files are decoded by `ffmpeg`, `whisper.cpp` and `tesseract`. Keep those tools up to date if you process files from sources you do not trust.
