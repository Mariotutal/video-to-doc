"""Heuristic section detection from audio silence gaps and VTT cues."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import run as sh_run
from ..vtt import cue_text, cues_in_section, is_spoken, parse_vtt

MIN_SECTION_S = 5.0
# This long with nothing in the transcript is a break, whatever the room sounded like.
BREAK_S = 120.0


def _detect_silences(audio: Path, *, threshold_db: int, min_s: float) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) of silence intervals."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", str(audio),
        "-af", f"silencedetect=n={threshold_db}dB:d={min_s}",
        "-f", "null",
        "-",
    ]
    return _parse_silences(sh_run(cmd, capture=True).stderr)


def _parse_silences(text: str) -> list[tuple[float, float]]:
    """Pair each silence_start with the silence_end that follows it.

    ffmpeg can report a slightly negative start for leading silence, and a silence
    still open at end-of-file has no end; neither may shift the other pairs.
    """
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for m in re.finditer(r"silence_(start|end): (-?[\d.]+)", text):
        value = max(0.0, float(m.group(2)))
        if m.group(1) == "start":
            start = value
        elif start is not None:
            silences.append((start, value))
            start = None
    return silences


def _mid(cue: dict) -> float:
    return (cue["start_s"] + cue["end_s"]) / 2


def _split_points(cues: list[dict], max_chars: int) -> list[float]:
    """Times at which to cut one section's cues so that no part's text exceeds `max_chars`.

    Someone who talks for an hour without a long pause would otherwise make one section
    too long to be summarised whole. Each cut falls between two cues, at the longest
    pause found in the last stretch of text that still fits. `cues` must be ordered by
    midpoint, which is how cues are assigned to sections.
    """
    sizes = [len(cue_text([c])) + 1 for c in cues]  # +1: the line break joining the cues
    total = sum(sizes) - 1
    if total <= max_chars:
        return []
    # Parts of about the same size, so the last one is not a leftover of a few lines.
    even = -(-total // -(-total // max_chars))
    max_chars = min(max_chars, even + max(200, even // 10))
    points: list[float] = []
    first = 0
    while sum(sizes[first:]) - 1 > max_chars:
        # `last` is the final cue of this part: as many cues as fit, but at least one.
        used, last = sizes[first], first
        while last + 1 < len(cues) and used + sizes[last + 1] - 1 <= max_chars:
            last += 1
            used += sizes[last]
        if last + 1 >= len(cues):
            break
        # Among the endings that leave the part at least 60% full, take the longest pause.
        candidates, filled = [], 0
        for i in range(first, last + 1):
            filled += sizes[i]
            if filled >= 0.6 * max_chars or i == last:
                candidates.append((cues[i + 1]["start_s"] - cues[i]["end_s"], i))
        _pause, end = max(candidates)
        before, after = _mid(cues[end]), _mid(cues[end + 1])
        cut = (cues[end]["end_s"] + cues[end + 1]["start_s"]) / 2
        if not before < cut <= after:  # overlapping cues: stay between their midpoints
            cut = (before + after) / 2
        points.append(cut)
        first = end + 1
    return points


def _merge_short_sections(boundaries: list[float], min_s: float) -> list[float]:
    """Drop boundaries that would leave a section shorter than `min_s`.

    A short section joins the one after it; a short last one joins the one before.
    Pauses mark where someone stopped talking, not where the subject changed, so two
    pauses close together would otherwise make a section of a sentence or two.
    """
    if min_s <= 0 or len(boundaries) < 3:
        return boundaries
    kept = [boundaries[0]]
    for boundary in boundaries[1:-1]:
        if boundary - kept[-1] >= min_s:
            kept.append(boundary)
    if len(kept) > 1 and boundaries[-1] - kept[-1] < min_s:
        kept.pop()
    return [*kept, boundaries[-1]]


def _split_long_sections(boundaries: list[float], cues: list[dict], max_chars: int,
                         min_s: float = MIN_SECTION_S) -> list[float]:
    """Add boundaries inside every section whose transcript is longer than `max_chars`."""
    if max_chars <= 0:
        return boundaries
    result = [boundaries[0]]
    last = len(boundaries) - 2
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        inside = cues_in_section(cues, start, end, is_last=(i == last))
        # Overlapping cues (two people talking) are not in midpoint order as listed.
        spoken = sorted((c for c in inside if is_spoken(c["text"])), key=_mid)
        cuts = _split_points(spoken, max_chars)
        # A last part much shorter than the others (a closing sentence, an announcement)
        # stays with the one before it, even if that makes it a little long.
        tail = max(MIN_SECTION_S, min(min_s, (end - start) / (len(cuts) + 1) / 2))
        for cut in cuts:
            if cut - result[-1] >= MIN_SECTION_S and end - cut >= tail:
                result.append(cut)
        result.append(end)
    return result


def _breaks(cues: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of every stretch of `BREAK_S` or more between two spoken cues."""
    spoken = sorted((c for c in cues if is_spoken(c["text"])), key=lambda c: c["start_s"])
    breaks = []
    said_until = spoken[0]["end_s"] if spoken else 0.0
    for cue in spoken[1:]:
        if cue["start_s"] - said_until >= BREAK_S:
            breaks.append((said_until, cue["start_s"]))
        said_until = max(said_until, cue["end_s"])
    return breaks


def _with_breaks(boundaries: list[float], breaks: list[tuple[float, float]]) -> list[float]:
    """The boundaries with every break as a section of its own.

    A boundary that fell inside a break is dropped; one a few seconds from its edge
    gives way to the edge, so no sliver is left beside the break.
    """
    if not breaks:
        return boundaries
    first, last = boundaries[0], boundaries[-1]
    edges = {edge for pair in breaks for edge in pair
             if edge - first >= MIN_SECTION_S and last - edge >= MIN_SECTION_S}
    clear = [b for b in boundaries[1:-1]
             if not any(start - MIN_SECTION_S < b < end + MIN_SECTION_S for start, end in breaks)]
    result = [first]
    for boundary in sorted({*clear, *edges}):
        # Two breaks with a word between them: the word joins the second one's section.
        if boundary - result[-1] >= MIN_SECTION_S:
            result.append(boundary)
    return [*result, last]


def run(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    out = workspace / "sections.json"
    audio = workspace / "audio.wav"
    vtt = workspace / "transcript.vtt"
    duration = manifest.duration_seconds or 0.0

    cues = parse_vtt(vtt)
    if config.teams_captions or not audio.exists():
        # Teams recordings are often silent: a stretch with no captions is the pause.
        silences = [(a["end_s"], b["start_s"]) for a, b in zip(cues, cues[1:], strict=False)]
    else:
        silences = _detect_silences(audio, threshold_db=config.silence_threshold_db,
                                    min_s=config.silence_min_s)

    # Find boundaries: silence gaps > section_gap_s
    section_gap_s = float(config.section_gap_s)
    boundaries: list[float] = [0.0]
    for s_start, s_end in silences:
        if (s_end - s_start) >= section_gap_s:
            boundaries.append(s_end)
    breaks = _breaks(cues)
    if not boundaries or boundaries[-1] < duration:
        boundaries.append(duration)
    # A closing remark shorter than a section would otherwise be dropped with its cues.
    if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < MIN_SECTION_S:
        del boundaries[-2]

    boundaries = _merge_short_sections(boundaries, float(config.min_section_s))
    boundaries = _split_long_sections(boundaries, cues, config.max_section_chars,
                                      max(MIN_SECTION_S, float(config.min_section_s)))
    boundaries = _with_breaks(boundaries, breaks)

    sections: list[dict] = []
    last = len(boundaries) - 2
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < MIN_SECTION_S:
            continue  # skip slivers
        section_cues = cues_in_section(cues, start, end, is_last=(i == last))
        is_silent = not any(is_spoken(c["text"]) for c in section_cues)
        sections.append({
            "index": i,
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "duration_s": round(end - start, 2),
            "is_silent": is_silent,
            "cue_count": len(section_cues),
            "title": None,
        })

    out.write_text(json.dumps(sections, indent=2))
