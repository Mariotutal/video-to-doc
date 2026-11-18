"""`vtd` command-line entry point."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import __version__
from .config import CONTENT_TYPES, LANGUAGE_NAMES, PROVIDERS, resolve_config
from .pipeline import AI_STAGE_NAME, STAGE_NAMES, check_before_download, run_pipeline
from .utils import VTDError, emit

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _parse_stage_list(spec: str | None) -> list[str] | None:
    """The stages named with --only; None when the option was not given."""
    if spec is None:
        return None
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGE_NAMES]
    if unknown or not requested:
        problem = f"Unknown stage(s): {', '.join(unknown)}." if unknown else "--only names no stage."
        raise VTDError(f"{problem} Valid stages: {', '.join(STAGE_NAMES)}")
    return requested


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vtd",
        description="Turn a video or audio file (or URL) into a structured Markdown document.",
    )
    p.add_argument(
        "source",
        type=str,
        help="Path to the input file (.mp4/.mp3/.wav/.m4a/etc.) or a URL (yt-dlp will fetch).",
    )
    p.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output workspace dir, used exactly as given. Default: <source-name>.vtd next to "
             "the input (in the current directory for URLs).",
    )

    # Transcription knobs
    p.add_argument(
        "--whisper-model",
        default=None,
        help="Whisper model variant. Default: medium (multilingual). Examples: small, medium.en, large-v3-turbo.",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Folder where Whisper models are kept and downloaded to, e.g. on an external "
             "disk. Default: ~/.cache/vtd/models.",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Spoken language code (e.g. 'en', 'es', 'fr') or 'auto' (default). Forwarded to whisper-cli.",
    )

    # Frame / video knobs
    p.add_argument(
        "--frame-rate",
        type=int,
        default=None,
        help="One frame every N seconds (default 60; 5 with --teams-captions).",
    )
    p.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Read speaker-name labels and slide text on frames with OCR (on by default; "
             "--no-ocr skips it, --ocr overrides `ocr_enabled = false` in the config file).",
    )
    p.add_argument(
        "--min-frame-words",
        type=int,
        default=None,
        help="With --with-ai: a frame with fewer readable words than this is taken for a face "
             "or an empty screen and left out of the notes (default 8; 0 keeps every frame, "
             "for recordings of diagrams or photos).",
    )
    p.add_argument(
        "--teams-captions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Transcribe visible Microsoft Teams live captions instead of audio "
             "(--no-teams-captions overrides the config file).",
    )

    # Section detection knobs
    p.add_argument(
        "--silence-threshold-db",
        type=int,
        default=None,
        help="Decibel threshold below which audio counts as silence (default -30).",
    )
    p.add_argument(
        "--silence-min-s",
        type=float,
        default=None,
        help="Minimum silence duration to register (default 2.0 s).",
    )
    p.add_argument(
        "--section-gap-s",
        type=float,
        default=None,
        help="Silence longer than this is treated as a section boundary (default 8.0 s).",
    )
    p.add_argument(
        "--min-section-s",
        type=float,
        default=None,
        help="A section shorter than this many seconds is joined to the next one, so a few "
             "words between two pauses do not become a section (default 120; 0 keeps them).",
    )
    p.add_argument(
        "--max-section-chars",
        type=int,
        default=None,
        help="A section with more transcript than this many characters is split at a pause, "
             "so that the AI stage reads all of it (default 6000; 0 never splits, otherwise at "
             "least 500).",
    )

    # Content / output
    p.add_argument(
        "--content-type",
        choices=CONTENT_TYPES,
        default=None,
        help="Document template style. Default: generic.",
    )
    p.add_argument(
        "--output-language",
        choices=tuple(LANGUAGE_NAMES),
        default=None,
        help="Language for the AI-authored document. Default: en.",
    )

    # Pipeline control
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages already completed with the same options; redo the rest. "
             "Without it, every stage is re-run.",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated stage list (naming `synthesize` enables AI). Re-running a "
             f"stage marks the stages built on it as not done. Valid: {', '.join(STAGE_NAMES)}",
    )

    p.add_argument(
        "--speakers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tell who speaks when: each change of voice in the transcript is marked "
             "Speaker 1, Speaker 2, … and the AI notes use those marks. Runs locally; needs "
             "the sherpa-onnx package and downloads two models (47 MB) on first use.",
    )
    p.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="With --speakers: how many people speak, when you know it (more reliable "
             "than letting vtd work it out).",
    )
    p.add_argument(
        "--speaker-threshold",
        type=float,
        default=None,
        help="With --speakers and no --num-speakers: how different two voices must be to "
             "count as two people (default 0.7; lower finds more speakers).",
    )
    p.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also write each document as a PDF next to it, with an index and the cited "
             "frames shown. Needs pandoc and xelatex.",
    )

    # AI
    p.add_argument(
        "--with-ai",
        action="store_true",
        help="Enable the AI synthesis stage. Requires an API key, except with --provider ollama.",
    )
    p.add_argument(
        "--provider",
        choices=tuple(PROVIDERS),
        default=None,
        help="AI provider. Default: anthropic. `ollama` uses a model on this machine: no key, "
             "nothing leaves it.",
    )
    p.add_argument(
        "--ai-model",
        default=None,
        help="Model id passed to the provider. Default depends on provider.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Override provider API key. Otherwise reads from env or config. Not used by ollama.",
    )

    p.add_argument(
        "--references",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="With --with-ai: cite the books and papers the recording refers to, as footnotes "
             "and in a bibliography with further reading. Only works found in Crossref or "
             "Open Library are cited; the authors and titles searched are sent to them.",
    )

    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra diagnostic information.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"vtd {__version__}",
    )
    return p


def _resolve_source(source: str, verbose: bool) -> Path:
    """Return a local Path to the source media. If a URL, download via yt-dlp."""
    if URL_RE.match(source):
        from .urls import download_url
        return download_url(source, verbose=verbose)
    p = Path(source)
    if not p.exists():
        raise VTDError(f"Source not found: {source}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        stages = _parse_stage_list(args.only)  # a typo here must fail before any download
        cfg = resolve_config(args)
        if stages and AI_STAGE_NAME in stages:
            cfg.ai.enabled = True  # asking for the stage by name is asking for AI
        if stages and "identify-speakers" in stages:
            cfg.speakers = True
        if args.references and not cfg.ai.enabled:
            raise VTDError("--references works on the AI notes: add --with-ai.")
        if URL_RE.match(args.source) and not args.resume:
            check_before_download(cfg, stages)
        source_path = _resolve_source(args.source, verbose=args.verbose)
    except VTDError as e:
        emit(str(e), kind="error")
        return 1

    if args.out:
        workspace = args.out
    elif URL_RE.match(args.source):
        # Downloads live in a hidden cache; keep the workspace where the user is.
        workspace = Path.cwd() / (source_path.name + cfg.workspace_suffix)
    else:
        workspace = source_path.with_name(source_path.name + cfg.workspace_suffix)

    try:
        run_pipeline(
            source=source_path,
            workspace=workspace,
            config=cfg,
            resume=args.resume,
            only_stages=stages,
            verbose=args.verbose,
        )
    except VTDError as e:
        emit(str(e), kind="error")
        return 1
    except KeyboardInterrupt:
        emit("Interrupted.", kind="warn")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
