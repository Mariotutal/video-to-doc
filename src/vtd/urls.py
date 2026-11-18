"""URL handling — download remote videos to a local cache via yt-dlp."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from .utils import VTDError, emit

CACHE_ROOT = Path.home() / ".cache" / "vtd" / "downloads"
# Anything ffmpeg can read; yt-dlp keeps the native extension for single-file downloads.
MEDIA_SUFFIXES = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts",
    ".m4a", ".mp3", ".wav", ".opus", ".ogg", ".aac", ".flac",
}
# yt-dlp names un-merged single streams like `Title.f137.mp4`.
_STREAM_PART = re.compile(r"\.f\d+\.\w+$")


def _media_files(directory: Path) -> list[Path]:
    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES and not _STREAM_PART.search(p.name)
    ]


def download_url(url: str, *, verbose: bool = False) -> Path:
    """Download `url` with yt-dlp into the cache; return the resulting media file path.

    If the URL has already been downloaded into the cache, the existing file is reused.
    """
    if shutil.which("yt-dlp") is None:
        raise VTDError(
            "yt-dlp not found on PATH. Install it (`brew install yt-dlp` or "
            "`pip install yt-dlp`) to use URL inputs."
        )

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    target_dir = CACHE_ROOT / url_hash
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = _media_files(target_dir)
    if existing:
        if verbose:
            emit(f"URL cache hit: {existing[0]}")
        return existing[0]

    # Download into a scratch dir; only a finished file is moved into the cache.
    partial_dir = target_dir / "partial"
    shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir()
    emit(f"Downloading {url} via yt-dlp → {target_dir}")
    cmd = [
        "yt-dlp",
        "-o", str(partial_dir / "%(title)s.%(ext)s"),
        "--no-playlist",
        "--restrict-filenames",
        # Prefer a single file; pick a reasonable format.
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        url,
    ]
    if not verbose:
        cmd += ["--quiet", "--no-warnings", "--progress"]
    try:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise VTDError(f"yt-dlp exited with code {proc.returncode} for {url}")
        downloaded = sorted(_media_files(partial_dir), key=lambda p: p.stat().st_size, reverse=True)
        if not downloaded:
            raise VTDError(f"yt-dlp finished but no media file was produced for {url}")
        media = target_dir / downloaded[0].name
        downloaded[0].replace(media)
    finally:
        shutil.rmtree(partial_dir, ignore_errors=True)
    emit(f"Downloaded: {media}", kind="done")
    return media
