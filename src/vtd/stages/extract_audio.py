"""Extract 16 kHz mono PCM WAV from the source video."""
from __future__ import annotations

from pathlib import Path

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import require_binary
from ..utils import run as sh_run


def run(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    require_binary("ffmpeg", "ffmpeg")
    out = workspace / "audio.wav"
    source = Path(manifest.source_video)
    # Write to a temp name so an interrupted run never leaves a truncated audio.wav.
    partial = workspace / "audio.partial.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-loglevel", "error",
        str(partial),
    ]
    sh_run(cmd)
    partial.replace(out)
