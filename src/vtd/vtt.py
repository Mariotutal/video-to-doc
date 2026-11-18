"""The transcript as cues: reading and writing transcript.vtt, and picking cues from it.

A cue is {start_s, end_s, text}. Every stage that reads or writes the transcript goes
through here, so the file has one format and one way of telling speech from silence.
"""
from __future__ import annotations

import re
from pathlib import Path

# What whisper.cpp writes for a stretch with nothing to transcribe: [BLANK_AUDIO].
BLANK = "[BLANK"

_CUE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*\n(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)


def stamp(seconds: float) -> str:
    """Seconds as the HH:MM:SS.mmm of a VTT file."""
    whole, ms = divmod(round(max(0.0, seconds) * 1000), 1000)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{ms:03d}"


def _seconds(hours: str, minutes: str, seconds: str, ms: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000.0


def parse_vtt(vtt_path: Path) -> list[dict]:
    """Return [{start_s, end_s, text}] from a transcript.vtt file; [] when there is none."""
    if not vtt_path.exists():
        return []
    cues: list[dict] = []
    for match in _CUE.finditer(vtt_path.read_text()):
        *times, line = match.groups()
        cues.append({
            "start_s": _seconds(*times[:4]),
            "end_s": _seconds(*times[4:]),
            "text": line.strip(),
        })
    return cues


def write_vtt(vtt_path: Path, cues: list[dict]) -> None:
    """Write cues as a VTT file that `parse_vtt` reads back."""
    vtt_path.write_text("WEBVTT\n\n" + "".join(
        f"{stamp(c['start_s'])} --> {stamp(c['end_s'])}\n{c['text']}\n\n" for c in cues))


def is_spoken(text: str) -> bool:
    """Whether a cue's text is something said, not an empty line or a silence mark."""
    return bool(text.strip()) and BLANK not in text


def cue_text(cues: list[dict]) -> str:
    """The spoken text of some cues, one line per cue, as it is given to the AI stage."""
    return "\n".join(c["text"].strip() for c in cues if is_spoken(c["text"]))


def cues_in_section(cues: list[dict], start: float, end: float, *, is_last: bool) -> list[dict]:
    """Cues whose midpoint falls in [start, end), so each cue lands in exactly one section.

    The last section also takes cues that run past the probed duration.
    """
    picked = []
    for c in cues:
        mid = (c["start_s"] + c["end_s"]) / 2
        if mid >= start and (is_last or mid < end):
            picked.append(c)
    return picked
