"""Common utilities: subprocess wrapper, ffprobe, logging helpers."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


class VTDError(RuntimeError):
    """User-facing error. CLI prints message and exits non-zero."""


def require_binary(name: str, package: str | None = None) -> str:
    """Locate a binary on PATH; raise VTDError with install hint if missing."""
    found = shutil.which(name)
    if found:
        return found
    hint = (
        f" (install `{package}`, e.g. with Homebrew or your system's package manager)"
        if package else ""
    )
    raise VTDError(f"Required binary `{name}` not found on PATH{hint}.")


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Wrapper around subprocess.run. Failures surface as VTDError, not tracebacks."""
    try:
        if capture:
            # ffmpeg echoes source metadata, which is not always valid UTF-8.
            return subprocess.run(
                cmd, check=check, capture_output=True, encoding="utf-8", errors="replace"
            )
        return subprocess.run(cmd, check=check)
    except FileNotFoundError as e:
        raise VTDError(f"Required binary `{cmd[0]}` not found on PATH.") from e
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip()[-400:]
        suffix = f": {detail}" if detail else "."
        raise VTDError(f"`{cmd[0]}` failed with exit code {e.returncode}{suffix}") from e


def ffprobe_info(media: Path) -> dict:
    """Return {duration, width, height, has_video, has_audio}."""
    require_binary("ffprobe", "ffmpeg")
    cp = run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(media),
        ],
        capture=True,
    )
    data = json.loads(cp.stdout)
    duration = float(data.get("format", {}).get("duration", 0.0))
    width = height = None
    has_video = False
    has_audio = False
    for s in data.get("streams", []):
        kind = s.get("codec_type")
        if kind == "video":
            has_video = True
            if width is None:
                width = int(s.get("width", 0)) or None
                height = int(s.get("height", 0)) or None
        elif kind == "audio":
            has_audio = True
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "has_video": has_video,
        "has_audio": has_audio,
    }


def emit(msg: str, *, kind: str = "info") -> None:
    """Single user-facing print path. `kind` is one of info/warn/error/done."""
    prefix = {
        "info": "•",
        "warn": "!",
        "error": "✗",
        "done": "✓",
    }.get(kind, "•")
    stream = sys.stderr if kind in ("warn", "error") else sys.stdout
    print(f"{prefix} {msg}", file=stream, flush=True)


def read_json(path: Path, default=None, *, required: str = ""):
    """What a workspace JSON file holds, or `default` when the file is not there.

    A file that is there but cannot be read as JSON is a damaged workspace: with
    `required` (the stage that writes the file) that is an error saying how to redo it;
    otherwise it is reported and `default` is used.
    """
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        if required:
            raise VTDError(f"{path.name} not found — run the {required} stage first.") from None
        return default
    except (OSError, ValueError) as e:
        if required:
            raise VTDError(
                f"{path.name} cannot be read ({e}). Redo it with `--only {required}`.") from e
        if path not in _UNREADABLE:  # said once: some files are read for every section
            _UNREADABLE.add(path)
            emit(f"{path.name} cannot be read ({e}); going on without it.", kind="warn")
        return default


_UNREADABLE: set[Path] = set()


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def with_fence_state(lines: list[str]):
    """Yield (line, inside) for Markdown lines, `inside` being True within a fenced block.

    The fence lines themselves count as inside. A block closes only on a fence of the
    same character, at least as long as the one that opened it, with nothing after it.
    """
    opened: tuple[str, int] | None = None
    for line in lines:
        fence = _FENCE.match(line)
        if opened is None:
            if fence and not (fence.group(1)[0] == "`" and "`" in fence.group(2)):
                opened = (fence.group(1)[0], len(fence.group(1)))
                yield line, True
            else:
                yield line, False
            continue
        yield line, True
        if (fence and fence.group(1)[0] == opened[0] and len(fence.group(1)) >= opened[1]
                and not fence.group(2).strip()):
            opened = None
