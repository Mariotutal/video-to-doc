"""OCR the bottom-left corner of each frame to extract participant name labels.

Most video-conferencing tools (Meet, Zoom, Teams) overlay the speaker's name in
the bottom-left of each tile. Cropping that region keeps OCR fast and accurate.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pytesseract
from PIL import Image

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import emit, read_json
from .extract_frames import frame_times


def _crop_label(image_path: Path):
    img = Image.open(image_path)
    w, h = img.size
    # Bottom-left corner: 35% wide × 10% tall, with a small inset.
    left = int(w * 0.01)
    right = int(w * 0.35)
    top = int(h * 0.88)
    bottom = int(h * 0.99)
    return img.crop((left, top, right, bottom))


FRAME_TEXT_NAME = "frame_text.json"
_WORD = re.compile(r"[^\W\d_]{3,}")


def _words_shown(image_path: Path) -> int:
    """How many words can be read in a frame: many for a slide, almost none for a face."""
    with Image.open(image_path) as img:
        img.thumbnail((1280, 1280))  # enough for slide text, and much faster
        return len(_WORD.findall(pytesseract.image_to_string(img)))


def frame_text(workspace: Path) -> dict[str, int]:
    """Words read in each frame, by file name; empty when frames were not read."""
    return read_json(workspace / FRAME_TEXT_NAME, {})


def _clean(raw: str) -> str:
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    # Allow letters, spaces, hyphens, apostrophes; reject obvious junk.
    line = re.sub(r"[^A-Za-z\-' ]+", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) < 3 or len(line) > 40:
        return ""
    if line.lower() in {"sharing", "muted", "you", "host"}:
        return ""
    return line


def run(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    out = workspace / "speakers.json"
    if config.teams_captions:
        return  # the Teams caption transcriber writes speakers.json itself

    rate = max(1, int(config.frame_rate_seconds))
    frames = frame_times(workspace, rate)
    if not frames:
        out.write_text("[]")
        return

    if shutil.which("tesseract") is None:
        emit("tesseract not found on PATH; skipping speaker OCR (--no-ocr silences this).",
             kind="warn")
        out.write_text("[]")
        return

    name_to_seconds: dict[str, list[float]] = defaultdict(list)
    name_to_count: Counter[str] = Counter()
    words: dict[str, int] = {}
    unreadable = 0

    for frame_path, sec in frames:
        try:
            crop = _crop_label(frame_path)
            text = pytesseract.image_to_string(crop, config="--psm 7")
            words[frame_path.name] = _words_shown(frame_path)
        except (OSError, pytesseract.TesseractError):
            unreadable += 1
            continue
        name = _clean(text)
        if not name:
            continue
        name_to_seconds[name].append(sec)
        name_to_count[name] += 1

    if unreadable:
        emit(f"ocr-speakers: OCR failed on {unreadable} of {len(frames)} frames.", kind="warn")

    speakers = []
    for name, count in name_to_count.most_common():
        if count < 2:
            continue  # likely OCR noise — require 2+ confirmations
        seconds = sorted(name_to_seconds[name])
        speakers.append({
            "name": name,
            "frame_count": count,
            "first_seen_s": seconds[0],
            "last_seen_s": seconds[-1],
        })

    out.write_text(json.dumps(speakers, indent=2))
    (workspace / FRAME_TEXT_NAME).write_text(json.dumps(words, indent=2))
