"""Anthropic provider — uses the official `anthropic` SDK."""
from __future__ import annotations

from pathlib import Path

from ..config import VTDConfig
from ..utils import VTDError
from .base import ChatProvider


class AnthropicProvider(ChatProvider):
    name = "anthropic"
    vendor = "Anthropic"

    def __init__(self, config: VTDConfig, workspace: Path):
        try:
            import anthropic
        except ImportError as e:
            raise VTDError(
                "anthropic package not installed. Run `pip install -e .[anthropic]` "
                "from the project root, or `pip install anthropic`."
            ) from e
        if not config.ai.api_key:
            raise VTDError("ANTHROPIC_API_KEY not set.")
        self._client = anthropic.Anthropic(api_key=config.ai.api_key)
        self._configure(config, workspace)

    def _frame_block(self, jpeg_b64: str) -> dict:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64},
        }

    def _request(self, system: str, content: list[dict], max_tokens: int) -> tuple[str, bool]:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        # `msg.content` is a list of TextBlock-likes.
        text = "".join(getattr(b, "text", "") for b in msg.content).strip()
        return text, getattr(msg, "stop_reason", None) == "max_tokens"
