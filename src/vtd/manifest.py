"""Workspace manifest — tracks completed pipeline stages for resumability."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .utils import emit

SCHEMA_VERSION = 1


@dataclass
class Manifest:
    schema_version: int = SCHEMA_VERSION
    source_video: str = ""                  # historical name; may be audio
    source_sha256_prefix: str = ""          # not full hash; just for change detection
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    has_video: bool = True
    has_audio: bool = True
    detected_language: str | None = None    # filled by transcribe stage when known
    stages_completed: list[str] = field(default_factory=list)
    # The options each completed stage was run with, so --resume can tell
    # whether its result still matches what is being asked for.
    stage_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_used: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    updated_utc: str = ""

    def mark_stage(self, stage: str, options: dict[str, Any] | None = None) -> None:
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)
        if options is not None:
            self.stage_options[stage] = options
        self.updated_utc = datetime.now(UTC).isoformat(timespec="seconds")

    def has_stage(self, stage: str) -> bool:
        return stage in self.stages_completed

    def reset_stage(self, stage: str) -> None:
        if stage in self.stages_completed:
            self.stages_completed.remove(stage)
        self.stage_options.pop(stage, None)
        self.updated_utc = datetime.now(UTC).isoformat(timespec="seconds")


def manifest_path(workspace: Path) -> Path:
    return workspace / "manifest.json"


def load(workspace: Path) -> Manifest:
    p = manifest_path(workspace)
    if not p.exists():
        return Manifest(created_utc=datetime.now(UTC).isoformat(timespec="seconds"))
    try:
        raw = json.loads(p.read_text())
        if not isinstance(raw, dict):
            raise ValueError("manifest is not a JSON object")
    except ValueError:
        # Manifest corrupt; start fresh but keep a backup
        backup = p.with_suffix(".json.bak")
        if not backup.exists():
            backup.write_text(p.read_text())
        return Manifest(created_utc=datetime.now(UTC).isoformat(timespec="seconds"))
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        emit(f"{p} was written by another vtd version; fields this one does not know are ignored.",
             kind="warn")
    # Unknown fields (from another vtd version) are dropped rather than treated as corruption.
    known = {f.name for f in fields(Manifest)}
    return Manifest(**{k: v for k, v in raw.items() if k in known})


def save(workspace: Path, manifest: Manifest) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    target = manifest_path(workspace)
    partial = target.with_suffix(".json.tmp")
    partial.write_text(json.dumps(asdict(manifest), indent=2))
    os.replace(partial, target)  # atomic: an interrupted save never leaves half a manifest


def video_hash_prefix(video: Path) -> str:
    """Hash only the first and last 1 MB plus the size — fast and stable enough."""
    h = hashlib.sha256()
    size = video.stat().st_size
    h.update(str(size).encode())
    with video.open("rb") as f:
        h.update(f.read(1024 * 1024))
        if size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, 2)
            h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]
