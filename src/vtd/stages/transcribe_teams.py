"""Transcribe Microsoft Teams live captions visible in sampled video frames."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pytesseract
from PIL import Image, ImageStat

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import VTDError, emit, require_binary
from ..vtt import write_vtt
from .extract_frames import list_frames

_SPEAKER_RE = re.compile(
    r"^([A-Z][A-Za-z'’-]{1,30}),\s*"
    r"([A-Z][A-Za-z'’-]{1,30})(?:\s+[a-z|])?$"
)
_PLAIN_SPEAKER_RE = re.compile(
    r"^[A-Z][A-Za-z'’-]+(?:\s+(?:[A-Z]|[A-Z][A-Za-z'’-]+)){1,3}$"
)


def _speaker_name(text: str) -> str | None:
    cleaned = re.sub(r"^[^A-Za-z]+", "", text.strip())
    match = _SPEAKER_RE.fullmatch(cleaned)
    if match:
        return f"{match.group(1)}, {match.group(2)}"
    return cleaned if _PLAIN_SPEAKER_RE.fullmatch(cleaned) else None


def _parse_entries(lines: list[tuple[str, int, int, int]]) -> list[tuple[str, str]]:
    """Turn positioned OCR lines into (speaker, caption) entries."""
    entries: list[tuple[str, str]] = []
    speaker: str | None = None
    speaker_left = last_top = last_height = 0
    text: list[str] = []

    def flush() -> None:
        if speaker and text:
            caption = re.sub(r"\s+", " ", " ".join(text)).strip()
            if len(caption) >= 2:
                entries.append((speaker, caption))

    for raw, left, top, height in sorted(lines, key=lambda line: (line[2], line[1])):
        found_speaker = _speaker_name(raw)
        if found_speaker:
            flush()
            speaker = found_speaker
            speaker_left = left
            last_top = top
            last_height = height
            text = []
            continue
        if not speaker:
            continue
        if abs(left - speaker_left) > 140 or top - last_top > max(50, last_height * 3):
            continue
        raw = re.sub(r"^\s*[|l]\s+", "I ", raw)
        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", raw).strip()
        if left < speaker_left - 10 and " " in cleaned:
            first, rest = cleaned.split(" ", 1)
            if first in {"|", "l", "I"}:
                cleaned = f"I {rest}"
            elif len(first) <= 3:
                cleaned = rest
        cleaned = re.sub(r"(?<!\S)\|(?!\S)", "I", cleaned).replace("Imean ", "I mean ")
        # Strip observed Teams avatar/icon OCR; add a layout profile
        # if another conferencing client needs different cleanup.
        words = cleaned.split()
        while words:
            token = re.sub(r"[^A-Za-z0-9]", "", words[0])
            if token.lower() in {"bs", "ee", "fe"} or token.isdigit() or (
                len(token) == 1 and token != "I" and words[0] != "A"
            ):
                words.pop(0)
                continue
            break
        cleaned = " ".join(words)
        if cleaned:
            text.append(cleaned)
            last_top = top
            last_height = height

    flush()
    return entries


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _merge_records(
    records: list[dict],
    entries: list[tuple[str, str]],
    second: int,
    interval: int,
) -> list[dict]:
    """Deduplicate the repeated and gradually expanding Teams caption window."""
    records = [dict(record) for record in records]
    for speaker, caption in entries:
        speaker_key = _normal(speaker)
        caption_key = _normal(caption)
        if not caption_key:
            continue
        merged = False
        for record in reversed(records[-20:]):
            if _normal(record["speaker"]) != speaker_key:
                continue
            old_key = _normal(record["text"])
            similar = SequenceMatcher(None, old_key, caption_key).ratio() >= 0.86
            if old_key in caption_key or caption_key in old_key or similar:
                if len(caption_key) > len(old_key):
                    record["text"] = caption
                record["end_s"] = second + interval
                merged = True
                break
        if not merged:
            records.append({
                "speaker": speaker,
                "text": caption,
                "start_s": second,
                "end_s": second + interval,
            })
    return records


def _canonicalize_speakers(records: list[dict]) -> list[dict]:
    """Fold rare OCR variants into speaker names observed repeatedly."""
    counts = Counter(_normal(record["speaker"]) for record in records)
    canonical = {key for key, count in counts.items() if count >= 2}
    displays = {
        key: Counter(
            record["speaker"] for record in records if _normal(record["speaker"]) == key
        ).most_common(1)[0][0]
        for key in canonical
    }
    cleaned: list[dict] = []
    for record in records:
        record = dict(record)
        key = _normal(record["speaker"])
        if key in canonical:
            record["speaker"] = displays[key]
        elif canonical:
            match = max(canonical, key=lambda candidate: SequenceMatcher(None, key, candidate).ratio())
            if SequenceMatcher(None, key, match).ratio() >= 0.65:
                record["speaker"] = displays[match]
        cleaned.append(record)
    return cleaned


def _ocr_lines(frame_path: Path) -> list[tuple[str, int, int, int]]:

    with Image.open(frame_path) as image:
        width, height = image.size
        # Teams live captions occupy the lower portion of the meeting window.
        crop = image.crop((0, int(height * 0.68), int(width * 0.72), height))
        gray = crop.convert("L")
        data = pytesseract.image_to_data(
            crop,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][index]) >= 30:
            grouped[(data["block_num"][index], data["par_num"][index], data["line_num"][index])].append(index)

    lines: list[tuple[str, int, int, int]] = []
    for indexes in grouped.values():
        left = min(data["left"][i] for i in indexes)
        top = min(data["top"][i] for i in indexes)
        right = max(data["left"][i] + data["width"][i] for i in indexes)
        bottom = max(data["top"][i] + data["height"][i] for i in indexes)
        line_height = bottom - top
        pad = max(4, line_height // 2)
        background = gray.crop((
            max(0, left - pad),
            max(0, top - pad),
            min(gray.width, right + pad),
            min(gray.height, bottom + pad),
        ))
        if ImageStat.Stat(background).median[0] > 110:
            continue
        text = " ".join(data["text"][i].strip() for i in indexes)
        lines.append((text, left, top, line_height))
    return lines


def run(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    require_binary("tesseract", "tesseract")
    transcript = workspace / "transcript.txt"
    vtt = workspace / "transcript.vtt"
    frames = list_frames(workspace / "frames")
    if not frames:
        raise VTDError("No sampled frames found — run extract-frames first.")

    interval = max(1, int(config.frame_rate_seconds))
    records: list[dict] = []
    for index, frame in enumerate(frames):
        records = _merge_records(records, _parse_entries(_ocr_lines(frame)), index * interval, interval)
        if (index + 1) % 100 == 0:
            emit(f"teams-captions: OCR {index + 1}/{len(frames)} frames")

    records = _canonicalize_speakers(records)
    if not records:
        raise VTDError(
            "No Teams live captions were detected. Ensure captions are visible "
            "in the lower part of the recording."
        )

    transcript.write_text("\n\n".join(
        f"{record['speaker']}: {record['text']}" for record in records
    ) + "\n")
    write_vtt(vtt, [{"start_s": r["start_s"], "end_s": r["end_s"],
                     "text": f"{r['speaker']}: {r['text']}"} for r in records])

    # speakers.json has the shape the OCR stage gives it: frame_count is frames, so a
    # caption seen from one frame to a later one counts every frame it stayed up.
    frames_shown: dict[str, int] = defaultdict(int)
    seconds: dict[str, list[int]] = defaultdict(list)
    for record in records:
        frames_shown[record["speaker"]] += (record["end_s"] - record["start_s"]) // interval
        seconds[record["speaker"]].extend([record["start_s"], record["end_s"]])
    speakers = [{
        "name": name,
        "frame_count": count,
        "first_seen_s": min(seconds[name]),
        "last_seen_s": max(seconds[name]),
    } for name, count in sorted(frames_shown.items(), key=lambda item: -item[1])]

    (workspace / "speakers.json").write_text(json.dumps(speakers, indent=2))
