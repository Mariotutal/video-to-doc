"""Sample frames from the video with near-duplicate suppression."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import read_json, require_binary
from ..utils import run as sh_run

TIMESTAMPS_NAME = "timestamps.json"
# A frame is also taken where the picture changes this much (a new slide, a screen
# share starting), but not more often than every SCENE_GAP_S seconds.
SCENE_CHANGE = 0.3
SCENE_GAP_S = 10
_FRAME_NUMBER = re.compile(r"frame_(\d+)\.jpg$")
_SHOWINFO_PTS = re.compile(r"Parsed_showinfo.*?pts_time:\s*(-?[\d.]+)")


def list_frames(frames_dir: Path) -> list[Path]:
    """Sampled frames in capture order (numeric, so frame_10000 follows frame_9999)."""
    numbered = [(int(m.group(1)), p) for p in frames_dir.glob("frame_*.jpg")
                if (m := _FRAME_NUMBER.search(p.name))]
    return [p for _, p in sorted(numbered)]


def _parse_showinfo_times(ffmpeg_log: str) -> list[float]:
    """Capture time of each frame ffmpeg wrote, in output order."""
    return [max(0.0, float(t)) for t in _SHOWINFO_PTS.findall(ffmpeg_log)]


def frame_times(workspace: Path, rate: int) -> list[tuple[Path, float]]:
    """Return (frame, capture time in seconds) for every sampled frame.

    Near-duplicate frames are dropped during extraction, so a frame's number says
    nothing about when it was captured; the real times are in timestamps.json.
    Workspaces without that file fall back to (number - 1) * rate.
    """
    frames_dir = workspace / "frames"
    recorded = read_json(frames_dir / TIMESTAMPS_NAME, {})
    timed: list[tuple[Path, float]] = []
    for frame in list_frames(frames_dir):
        if frame.name in recorded:
            timed.append((frame, float(recorded[frame.name])))
        else:
            number = int(_FRAME_NUMBER.search(frame.name).group(1))
            timed.append((frame, float((number - 1) * rate)))
    return timed


def run(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    require_binary("ffmpeg", "ffmpeg")
    frames_dir = workspace / "frames"
    rate = max(1, int(config.frame_rate_seconds))
    source = Path(manifest.source_video)
    width = 1920 if config.teams_captions else 1280
    duration = manifest.duration_seconds or 0.0

    limit: list[str] = []
    if 0 < duration < rate:
        # Shorter than one sampling interval: `fps` would emit nothing and ffmpeg
        # would fail, so take the opening frame instead.
        vf = f"showinfo,scale={width}:-1"
        limit = ["-frames:v", "1"]
    elif config.teams_captions:
        # Keep every sample: live-caption timestamps rely on the frame index.
        vf = f"fps=1/{rate},showinfo,scale={width}:-1"
    else:
        # One frame every `rate` seconds, plus one where the picture changes, so a slide
        # shown between two ticks is not missed. Near-identical frames are then dropped.
        gap = min(SCENE_GAP_S, rate)
        pick = (
            f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,{rate})"
            f"+gt(scene\\,{SCENE_CHANGE})*gte(t-prev_selected_t\\,{gap})"
        )
        vf = (f"select='{pick}',mpdecimate=hi=64*200:lo=64*100:frac=0.33,"
              f"showinfo,scale={width}:-1")

    # Extract into a temp dir so an interrupted run never leaves a partial frame set.
    partial = workspace / "frames.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-i", str(source),
        "-vf", vf,
        *limit,
        "-q:v", "3",
        "-fps_mode", "vfr",
        "-loglevel", "info",  # showinfo reports each kept frame's timestamp here
        str(partial / "frame_%04d.jpg"),
    ]
    try:
        proc = sh_run(cmd, capture=True)
        times = _parse_showinfo_times(proc.stderr)
        frames = list_frames(partial)
        # showinfo can log a frame or two more than ffmpeg writes when output is capped.
        if len(times) >= len(frames):
            (partial / TIMESTAMPS_NAME).write_text(json.dumps(
                {f.name: t for f, t in zip(frames, times, strict=False)}, indent=2
            ))
        shutil.rmtree(frames_dir, ignore_errors=True)
        partial.replace(frames_dir)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
