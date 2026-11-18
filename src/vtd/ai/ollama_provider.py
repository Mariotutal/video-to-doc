"""Ollama provider — a model running on this machine, through Ollama's local HTTP API.

Nothing leaves the machine and no API key or extra package is needed.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import VTDConfig
from ..utils import VTDError, emit
from .base import ChatProvider

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_PORT = 11434
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
# Ollama's default context is too small for a section's transcript plus its frames,
# and input that does not fit is dropped without an error. So ask for enough.
NUM_CTX = 16384
# A model that cannot stop thinking spends output tokens on it before the answer.
THINKING_ROOM = 2048
# A local model can take minutes per answer, more so the first time it is loaded.
REQUEST_TIMEOUT_S = 1800
_TOLD_BLIND: set[str] = set()


def ollama_host() -> str:
    """Where the Ollama server is: $OLLAMA_HOST (as Ollama itself reads it) or localhost."""
    value = (os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).strip().rstrip("/")
    url = urllib.parse.urlsplit(value if "://" in value else f"http://{value}")
    # 0.0.0.0 is what people export so that `ollama serve` listens everywhere.
    name = "127.0.0.1" if url.hostname == "0.0.0.0" else url.hostname or "localhost"
    name = f"[{name}]" if ":" in name else name
    # Like Ollama's own client: no port means 11434, except https which keeps 443.
    port = url.port or (None if url.scheme == "https" else DEFAULT_PORT)
    return f"{url.scheme}://{name}" + (f":{port}" if port else "")


class OllamaProvider(ChatProvider):
    name = "ollama"
    vendor = "Ollama"

    def __init__(self, config: VTDConfig, workspace: Path):
        self._host = ollama_host()
        # A server on this machine is reached directly: a configured HTTP proxy would
        # either fail to find it or be handed the transcript.
        local = urllib.parse.urlsplit(self._host).hostname in _LOOPBACK
        self._open = (
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open
            if local else urllib.request.urlopen
        )
        self._configure(config, workspace)
        capabilities = self._capabilities()
        self._thinks = "thinking" in capabilities
        self._sees = "vision" in capabilities
        if not self._sees and self._model not in _TOLD_BLIND:
            _TOLD_BLIND.add(self._model)  # said once, however often the provider is built
            emit(f"Model `{self._model}` cannot read images; frames are not sent to it.",
                 kind="warn")

    def _call_ollama(self, path: str, payload: dict | None = None, timeout: float = 30) -> dict:
        request = urllib.request.Request(
            f"{self._host}{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._open(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("error") or e.reason
            except ValueError:
                detail = e.reason
            raise RuntimeError(f"{detail} (HTTP {e.code})") from e

    def _capabilities(self) -> list[str]:
        """What the model can do; also the check that Ollama runs and has the model."""
        try:
            installed = [m["name"] for m in self._call_ollama("/api/tags").get("models", [])]
        except (OSError, ValueError, RuntimeError) as e:
            raise VTDError(
                f"Cannot reach Ollama at {self._host} ({e}). Start it with `ollama serve`, "
                "or set $OLLAMA_HOST if it runs somewhere else."
            ) from e
        try:
            # Ollama decides whether it knows the name (it accepts several spellings of one).
            return self._call_ollama("/api/show", {"model": self._model}).get("capabilities") or []
        except (OSError, ValueError, RuntimeError) as e:
            have = ", ".join(sorted(installed)) or "none"
            raise VTDError(
                f"Ollama cannot use the model `{self._model}` ({e}). Get it with "
                f"`ollama pull {self._model}`, or pick one with --ai-model. Installed: {have}."
            ) from e

    def _frame_block(self, jpeg_b64: str) -> dict:
        return {"type": "image", "data": jpeg_b64}

    def _request(self, system: str, content: list[dict], max_tokens: int) -> tuple[str, bool]:
        user: dict = {
            "role": "user",
            "content": "\n\n".join(b["text"] for b in content if b["type"] == "text"),
        }
        images = [b["data"] for b in content if b["type"] == "image"]
        if images and self._sees:
            user["images"] = images
        payload: dict = {
            "model": self._model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, user],
            "options": {
                "num_ctx": NUM_CTX,
                "num_predict": max_tokens + (THINKING_ROOM if self._thinks else 0),
            },
        }
        if self._thinks:
            payload["think"] = False  # summaries do not need it, and it is much slower
        answer = self._call_ollama("/api/chat", payload, timeout=REQUEST_TIMEOUT_S)
        text = ((answer.get("message") or {}).get("content") or "").strip()
        return text, answer.get("done_reason") == "length"
