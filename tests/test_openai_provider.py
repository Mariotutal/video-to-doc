from types import SimpleNamespace

import pytest

from vtd.ai import cache, openai_provider
from vtd.ai.openai_provider import OpenAIProvider
from vtd.utils import VTDError


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Generated text")
                )
            ]
        )


class FakeRateLimitError(Exception):
    pass


class RateLimitedCompletions:
    def create(self, **kwargs):
        raise FakeRateLimitError(
            "Rate limit reached. Please try again in 9h22m27.84s."
        )


def test_openai_provider_uses_max_completion_tokens(tmp_path):
    completions = FakeCompletions()
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    provider._model = "gpt-5.4-mini"
    provider._workspace = tmp_path
    provider._output_language = "Spanish"

    text = provider._call(
        instruction="Write a title.",
        transcript_chunk="A short transcript.",
        max_tokens=40,
        label="title",
    )

    assert text == "Generated text"
    # Room is added for a reasoning model's thinking, which counts against the limit.
    assert completions.kwargs["max_completion_tokens"] == 40 + openai_provider.REASONING_ROOM
    assert "max_tokens" not in completions.kwargs
    user_message = completions.kwargs["messages"][1]["content"][-1]["text"]
    assert "Write the final answer in Spanish." in user_message


def test_openai_rate_limit_raises_user_facing_error(tmp_path):
    FakeRateLimitError.__name__ = "RateLimitError"
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=RateLimitedCompletions())
    )
    provider._model = "gpt-5.4-mini"
    provider._workspace = tmp_path
    provider._output_language = "Spanish"

    with pytest.raises(VTDError) as exc:
        provider._call(
            instruction="Build a glossary.",
            transcript_chunk="A short transcript.",
            max_tokens=40,
            label="glossary",
        )

    assert "OpenAI rate limit reached while generating glossary" in str(exc.value)
    assert "--only synthesize" in str(exc.value)


def _provider(tmp_path, create):
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider._model = "gpt-5.4-mini"
    provider._workspace = tmp_path
    provider._output_language = "English"
    return provider


def _reply(content, finish_reason):
    return SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    ])


@pytest.mark.parametrize("content,finish_reason", [(None, "length"), ("Cut off mid", "length")])
def test_empty_or_truncated_responses_are_not_cached(tmp_path, content, finish_reason):
    replies = [_reply(content, finish_reason), _reply("Full answer", "stop")]
    provider = _provider(tmp_path, lambda **kwargs: replies.pop(0))
    call = dict(instruction="Write.", transcript_chunk="Text.", label="body")

    assert provider._call(**call) == (content or "")
    assert not list(cache.cache_dir(tmp_path).iterdir())
    assert provider._call(**call) == "Full answer"
    assert len(list(cache.cache_dir(tmp_path).iterdir())) == 1


def test_other_api_errors_are_user_facing(tmp_path):
    def create(**kwargs):
        raise ConnectionError("network down")

    provider = _provider(tmp_path, create)
    with pytest.raises(VTDError, match="OpenAI request failed while generating body"):
        provider._call(instruction="Write.", transcript_chunk="Text.", label="body")
