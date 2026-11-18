"""OpenAI provider — uses the official `openai` SDK."""
from __future__ import annotations

from pathlib import Path

from ..config import VTDConfig
from ..utils import VTDError
from .base import ChatProvider

# Reasoning models spend output tokens on thinking before the answer, and those count
# against the same limit: a 40-token title would come back empty without room for it.
REASONING_ROOM = 2048


class OpenAIProvider(ChatProvider):
    name = "openai"
    vendor = "OpenAI"

    def __init__(self, config: VTDConfig, workspace: Path):
        try:
            import openai
        except ImportError as e:
            raise VTDError(
                "openai package not installed. Run `pip install -e .[openai]` "
                "from the project root, or `pip install openai`."
            ) from e
        if not config.ai.api_key:
            raise VTDError("OPENAI_API_KEY not set.")
        self._client = openai.OpenAI(api_key=config.ai.api_key)
        self._configure(config, workspace)

    def _frame_block(self, jpeg_b64: str) -> dict:
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}

    def _request(self, system: str, content: list[dict], max_tokens: int) -> tuple[str, bool]:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens + REASONING_ROOM,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        return text, getattr(choice, "finish_reason", None) == "length"
