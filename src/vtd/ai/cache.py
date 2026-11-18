"""On-disk response cache so re-runs with --resume don't re-spend tokens."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(*parts: Any) -> str:
    """Stable hash of arbitrary JSON-serializable inputs."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        elif isinstance(p, Path):
            h.update(str(p).encode())
            try:
                h.update(p.stat().st_size.to_bytes(8, "big"))
            except OSError:
                pass
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()


def cache_dir(workspace: Path) -> Path:
    d = workspace / "ai-cache"
    d.mkdir(exist_ok=True)
    return d


def load(workspace: Path, key: str) -> str | None:
    p = cache_dir(workspace) / f"{key}.json"
    if not p.exists():
        return None
    try:
        response = json.loads(p.read_text())["response"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return response if isinstance(response, str) else None


def store(workspace: Path, key: str, prompt_summary: str, response: Any) -> None:
    p = cache_dir(workspace) / f"{key}.json"
    p.write_text(json.dumps({
        "prompt_summary": prompt_summary,
        "response": response,
    }, indent=2))
