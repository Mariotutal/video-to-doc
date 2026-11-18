"""Stand-ins shared by the tests: an AI provider whose client is a function."""
from types import SimpleNamespace

from vtd.ai.anthropic_provider import AnthropicProvider
from vtd.config import AIConfig, VTDConfig


def fake_anthropic(workspace, create, *, language="en", content_type="generic", model="test"):
    """An Anthropic provider set up as vtd does, whose requests go to `create(**kwargs)`."""
    provider = AnthropicProvider.__new__(AnthropicProvider)  # no SDK, no key
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider._configure(
        VTDConfig(output_language=language, content_type=content_type, ai=AIConfig(model=model)),
        workspace,
    )
    return provider


def message(text, stop_reason="end_turn"):
    """An answer as the Anthropic SDK returns it."""
    return SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason=stop_reason)


def replying(replies, record=lambda kwargs: kwargs["messages"][0]["content"][-1]["text"]):
    """(create, sent): `create` answers with `replies` in turn and notes what it was sent."""
    sent = []

    def create(**kwargs):
        sent.append(record(kwargs))
        return message(replies.pop(0))

    return create, sent
