"""Render the deterministic 'starter' Markdown document from the workspace artifacts."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import VTDError, read_json
from ..vtt import parse_vtt
from .extract_frames import list_frames
from .identify_speakers import cue_speakers, labelled_text, speaker_names, speaker_turns
from .labels import LABELS


def hms(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    total = int(round(seconds))
    return str(timedelta(seconds=total))


TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def _load_template(name: str) -> str:
    path = TEMPLATES / name
    if not path.exists():
        raise VTDError(f"Template {name} not found at {path}; the vtd install is incomplete.")
    return path.read_text()


def built_with(config: VTDConfig, manifest: Manifest) -> dict:
    """The options the existing transcript and frames were actually made with.

    Documents describe the workspace, not the flags of the run that renders them.
    """
    transcript = manifest.stage_options.get("transcribe", {})
    frames = manifest.stage_options.get("extract-frames", {})
    return {
        "teams_captions": transcript.get("teams_captions", config.teams_captions),
        "whisper_model": transcript.get("whisper_model", config.whisper_model),
        "frame_rate_seconds": frames.get("frame_rate_seconds", config.frame_rate_seconds),
    }


def render_template(name: str, *, config: VTDConfig, manifest: Manifest, **context) -> str:
    """Render a bundled template with the shared filters, labels and workspace facts."""
    _load_template(name)  # a clear error if the install lost its templates
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                      autoescape=False, keep_trailing_newline=True)
    env.filters["hms"] = hms
    # The file name alone: a full path in a heading is noise, and a private detail.
    env.filters["basename"] = lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
    labels = LABELS[config.output_language]
    built = built_with(config, manifest)
    return env.get_template(name).render(
        manifest=manifest,
        config=config,
        built=built,
        t=labels,
        transcript_source=(
            labels["teams_captions"] if built["teams_captions"]
            else f"Whisper `{built['whisper_model']}`"
        ),
        **context,
    )


def _transcript(workspace: Path, config: VTDConfig) -> str:
    """The transcript, with each change of speaker marked when speakers were identified."""
    turns = speaker_turns(workspace)
    if turns:
        cues = parse_vtt(workspace / "transcript.vtt")
        return labelled_text(cues, cue_speakers(cues, turns),
                             LABELS[config.output_language]["speaker"], speaker_names(workspace))
    path = workspace / "transcript.txt"
    return path.read_text() if path.exists() else ""


def run_starter(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    frames = list_frames(workspace / "frames")
    rendered = render_template(
        "starter.md.j2",
        config=config,
        manifest=manifest,
        transcript=_transcript(workspace, config),
        speakers=read_json(workspace / "speakers.json", []),
        sections=read_json(workspace / "sections.json", []),
        frames=[str(f.relative_to(workspace)) for f in frames],
        workspace_name=workspace.name,
        voices=len({turn["speaker"] for turn in speaker_turns(workspace)}),
    )
    (workspace / config.starter_doc_name).write_text(rendered)
