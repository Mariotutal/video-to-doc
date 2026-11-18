"""Transcribe audio.wav using whisper.cpp. Auto-downloads the model on first use."""
from __future__ import annotations

import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import VTDError, emit, require_binary
from ..vtt import parse_vtt, write_vtt

# Whisper can fall into a loop over silence or music and repeat one line many times.
# From this many identical lines in a row, only the first is kept.
REPEAT_RUN = 3
# A loop is Whisper going round a few phrases over a stretch with no speech (a break,
# music): at least this many lines that only say again what was just said.
LOOP_RUN = 30
LOOP_MEMORY = 4  # lines before a stretch that its echoes may be repeating
LOOP_LINES = 5  # a loop goes round a few phrases; people who repeat themselves use more
LOOP_OTHER = 0.1  # share of lines said once that a loop may hold, against its echoes
LOOP_SLACK = 5  # and how many it may hold before that share is looked at
RAW_SUFFIX = ".raw"  # transcript.raw.vtt / .txt: what Whisper wrote, when lines were removed

HUGGINGFACE_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


MIN_MODEL_BYTES = 50 * 1024 * 1024  # anything smaller is not a whole model


def _model_path(config: VTDConfig) -> Path:
    return config.model_dir / f"ggml-{config.whisper_model}.bin"


def _has_model(target: Path) -> bool:
    try:
        return target.stat().st_size > MIN_MODEL_BYTES
    except OSError:
        return False


def prepare_model_dir(config: VTDConfig) -> Path:
    """Path of the model file, making sure it is there or its folder can take a download.

    The folder may be on an external disk, so say plainly when it cannot be reached.
    """
    target = _model_path(config)
    if _has_model(target):
        return target
    folder = config.model_dir
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if not os.access(folder, os.W_OK | os.X_OK):
            raise PermissionError(f"{folder} is not writable")
    except OSError as e:
        raise VTDError(
            f"Whisper model `{config.whisper_model}` is not in {folder} and that folder "
            f"cannot be used for a download ({e}). If it is on an external disk, check the "
            "disk is connected, or pick another folder with --model-dir."
        ) from e
    return target


def _download_model(config: VTDConfig) -> Path:
    target = prepare_model_dir(config)
    if _has_model(target):
        return target
    model_name = config.whisper_model
    url = f"{HUGGINGFACE_BASE}/ggml-{model_name}.bin"
    emit(f"Downloading whisper model `{model_name}` from {url} → {target}")

    tmp = target.with_suffix(".bin.partial")
    last_mb_reported = 0
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            read = 0
            chunk = 1024 * 1024  # 1 MB
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                read += len(buf)
                mb = read // (50 * 1024 * 1024)
                if mb > last_mb_reported:
                    last_mb_reported = mb
                    pct = f"{(100 * read / total):.0f}%" if total else "?"
                    emit(f"  …{read // (1024*1024)} MB ({pct})")
        tmp.rename(target)
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise VTDError(f"Failed to download whisper model: {e}") from e

    if not _has_model(target):
        target.unlink(missing_ok=True)
        raise VTDError(f"Downloaded model is too small; URL may be wrong: {url}")
    emit(f"Model ready: {target}", kind="done")
    return target


def _same_line(text: str) -> str:
    # A line of signs only (♪, ...) is compared as it is written.
    return re.sub(r"[\W_]+", " ", text).strip().casefold() or text.strip()


def drop_loops(cues: list[dict]) -> list[dict]:
    """Cues without the stretches where a few phrases go round and round.

    A line is an echo when the same line was said shortly before. A stretch runs while
    no two new lines follow each other and no more than `LOOP_LINES` different lines
    are echoed. It is a loop when it holds `LOOP_RUN` echoes or more and, once they
    begin, hardly anything else: people who answer one another, count or use a filler
    word say new things in between. Of a loop, the echoes are dropped, and so are the
    lines that first appear after it has begun and are then repeated; lines said only
    once are kept.
    """
    lines = [_same_line(c["text"]) for c in cues]
    kept: list[dict] = []
    i = 0
    while i < len(cues):
        heard = {line for line in lines[max(0, i - LOOP_MEMORY):i] if line}
        echoes: list[int] = []
        looped: set[str] = set()
        new_in_a_row = 0
        others = 0  # lines that are not echoes, since the first echo
        j = i
        while j < len(cues):
            if lines[j] and lines[j] in heard:
                if lines[j] not in looped and len(looped) == LOOP_LINES:
                    break  # too many different lines for a loop: the stretch ends here
                looped.add(lines[j])
                echoes.append(j)
                new_in_a_row = 0
            else:
                new_in_a_row += 1
                others += bool(echoes)
                # Too much else is being said for this to be a loop: stop reading.
                if new_in_a_row == 2 or others > LOOP_SLACK + len(echoes) * LOOP_OTHER:
                    break
                heard.add(lines[j])
            j += 1
        # A line said twice in passing is not what the loop goes round.
        times = Counter(lines[k] for k in echoes)
        echoes = [k for k in echoes if times[lines[k]] >= REPEAT_RUN]
        if len(echoes) >= LOOP_RUN:
            looped = {lines[k] for k in echoes}
            j = echoes[-1] + 1  # what follows the last echo is not part of the loop
            begun = echoes[0]
            inside = set(echoes) | {k for k in range(begun, j) if lines[k] in looped}
            said_once = j - begun - len(inside)
            if said_once <= len(echoes) * LOOP_OTHER:
                kept += [cues[k] for k in range(i, j) if k not in inside]
                i = j
                continue
        kept.append(cues[i])
        i += 1
    return kept


def drop_repeats(cues: list[dict]) -> list[dict]:
    """Cues without the echoes: loops are removed, and of a run of identical lines
    only the first is kept."""
    cues = drop_loops(cues)
    kept: list[dict] = []
    i = 0
    while i < len(cues):
        j = i + 1
        line = _same_line(cues[i]["text"])
        while j < len(cues) and line and _same_line(cues[j]["text"]) == line:
            j += 1
        kept += cues[i:j] if j - i < REPEAT_RUN else cues[i:i + 1]
        i = j
    return kept


def clean_transcript(workspace: Path) -> int:
    """Remove repeated lines from transcript.vtt and .txt; returns how many were removed.

    What Whisper wrote is kept beside them as transcript.raw.vtt and .txt.
    """
    vtt, txt = workspace / "transcript.vtt", workspace / "transcript.txt"
    cues = parse_vtt(vtt)
    kept = drop_repeats(cues)
    removed = len(cues) - len(kept)
    if not removed:
        return 0
    for path in (vtt, txt):
        if path.exists():
            path.replace(path.with_name(f"{path.stem}{RAW_SUFFIX}{path.suffix}"))
    write_vtt(vtt, kept)
    txt.write_text("".join(f"{c['text']}\n" for c in kept))
    return removed


def run(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    if config.teams_captions:
        from . import transcribe_teams

        transcribe_teams.run(workspace, config, manifest)
        return

    require_binary("whisper-cli", "whisper-cpp")
    audio = workspace / "audio.wav"
    if not audio.exists():
        raise VTDError("audio.wav not found — extract-audio stage must run first.")

    out_base = workspace / "transcript"  # whisper-cli adds .txt and .vtt
    model_path = _download_model(config)

    log_path = workspace / "whisper.log"
    cmd = [
        "whisper-cli",
        "-m", str(model_path),
        "-f", str(audio),
        "-otxt",
        "-ovtt",
        "-of", str(out_base),
        "-t", "8",
    ]
    # Language: 'auto' lets whisper detect; otherwise pass through as -l <code>.
    lang = (config.language or "auto").lower()
    if lang and lang != "auto":
        cmd += ["-l", lang]
    else:
        cmd += ["-l", "auto"]

    with log_path.open("w") as log_f:
        import subprocess
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        proc.wait()
        if proc.returncode != 0:
            raise VTDError(
                f"whisper-cli exited with code {proc.returncode}. See {log_path}."
            )

    removed = clean_transcript(workspace)
    if removed:
        emit(f"transcribe: removed {removed} lines that only repeated the one before them "
             f"(Whisper's own output is kept as transcript{RAW_SUFFIX}.vtt)", kind="warn")

    # Record auto-detected language if whisper printed it.
    try:
        log = log_path.read_text(errors="ignore")
        import re as _re
        m = _re.search(r"auto-detected language:\s*([a-z\-]+)", log, _re.IGNORECASE)
        if m:
            manifest.detected_language = m.group(1).lower()
    except OSError:
        pass
