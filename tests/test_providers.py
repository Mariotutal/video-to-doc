from types import SimpleNamespace

import pytest

from vtd import urls
from vtd.ai import cache
from vtd.ai.base import ProviderLimitError
from vtd.utils import VTDError

from .fakes import fake_anthropic, message


def _anthropic(tmp_path, create):
    return fake_anthropic(tmp_path, create, model="claude-test")


_message = message


CALL = dict(instruction="Write.", transcript_chunk="Text.", label="body")


def test_anthropic_truncated_response_is_not_cached(tmp_path):
    replies = [_message("Cut off mid", "max_tokens"), _message("Full answer", "end_turn")]
    provider = _anthropic(tmp_path, lambda **kwargs: replies.pop(0))

    assert provider._call(**CALL) == "Cut off mid"
    assert not list(cache.cache_dir(tmp_path).iterdir())
    assert provider._call(**CALL) == "Full answer"
    assert len(list(cache.cache_dir(tmp_path).iterdir())) == 1


def test_empty_answer_is_cached_only_where_empty_is_valid(tmp_path):
    provider = _anthropic(tmp_path, lambda **kwargs: _message("", "end_turn"))

    assert provider._call(**CALL) == ""
    assert not list(cache.cache_dir(tmp_path).iterdir())
    assert provider.extract_followups("Nothing to do.") == []
    assert len(list(cache.cache_dir(tmp_path).iterdir())) == 1


def test_anthropic_errors_are_user_facing_and_limits_are_distinguished(tmp_path):
    class RateLimitError(Exception):
        pass

    def fail_with(error):
        def create(**kwargs):
            raise error
        return create

    with pytest.raises(VTDError, match="Anthropic request failed while generating body") as exc:
        _anthropic(tmp_path, fail_with(ConnectionError("network down")))._call(**CALL)
    assert not isinstance(exc.value, ProviderLimitError)

    with pytest.raises(ProviderLimitError, match="Anthropic rate limit reached"):
        _anthropic(tmp_path, fail_with(RateLimitError("slow down")))._call(**CALL)


def _fake_ytdlp(monkeypatch, tmp_path, files, returncode=0):
    """Stub yt-dlp: drop `files` into the directory given by the -o template."""
    monkeypatch.setattr(urls, "CACHE_ROOT", tmp_path / "downloads")
    monkeypatch.setattr(urls.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def fake_run(cmd):
        out_dir = urls.Path(cmd[cmd.index("-o") + 1]).parent
        for name, size in files.items():
            (out_dir / name).write_bytes(b"x" * size)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(urls.subprocess, "run", fake_run)


def test_download_keeps_only_the_finished_file(tmp_path, monkeypatch):
    _fake_ytdlp(monkeypatch, tmp_path, {"Talk.f137.mp4": 50, "Talk.mov": 80, "Talk.mov.part": 10})

    media = urls.download_url("https://example.com/v")

    assert media.name == "Talk.mov"
    assert [p.name for p in media.parent.iterdir()] == ["Talk.mov"]
    assert urls.download_url("https://example.com/v") == media  # cache hit


def test_failed_download_leaves_nothing_that_looks_like_a_cache_hit(tmp_path, monkeypatch):
    _fake_ytdlp(monkeypatch, tmp_path, {"Talk.f137.mp4": 50}, returncode=1)

    with pytest.raises(VTDError, match="yt-dlp exited with code 1"):
        urls.download_url("https://example.com/v")
    assert list((tmp_path / "downloads").rglob("*.mp4")) == []


def _recording_provider(tmp_path, replies):
    """Anthropic provider whose client records every request it is sent."""
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return replies.pop(0) if replies else _message("answer", "end_turn")

    return _anthropic(tmp_path, create), sent


def test_prompts_follow_the_content_type(tmp_path):
    provider, sent = _recording_provider(tmp_path, [])
    provider._content_type = "lecture"

    provider._call(**CALL)
    provider.extract_followups("summaries")

    assert "excerpt of a lecture transcript" in sent[0]["system"]
    followup_prompt = sent[1]["messages"][0]["content"][-1]["text"]
    assert "the lecturer left for the audience" in followup_prompt
    assert "SECTION SUMMARIES:" in followup_prompt and "TRANSCRIPT EXCERPT" not in followup_prompt


def test_changing_the_system_prompt_does_not_reuse_a_cached_answer(tmp_path):
    provider, sent = _recording_provider(tmp_path, [])

    provider._call(**CALL)
    provider._call(**CALL)
    assert len(sent) == 1  # second call came from the cache

    provider._content_type = "podcast"
    provider._call(**CALL)
    assert len(sent) == 2


def test_over_long_input_is_cut_with_a_warning(tmp_path, capsys):
    from vtd.ai.base import BODY_CHARS, SectionInput

    provider, sent = _recording_provider(tmp_path, [])
    section = SectionInput(index=0, title=None, start_s=0, end_s=10,
                           transcript_chunk="x" * (BODY_CHARS + 500), frame_paths=[])

    provider.write_section_body(section)

    assert f"input cut from {BODY_CHARS + 500} to {BODY_CHARS} characters" in capsys.readouterr().err
    assert "x" * BODY_CHARS + "\n```" in sent[0]["messages"][0]["content"][-1]["text"]


def test_incomplete_answers_are_tracked_once_and_never_cached(tmp_path):
    provider, sent = _recording_provider(tmp_path, [
        _message("Cut off mid", "max_tokens"),                 # body: truncated
        _message("No follow-ups were identified.", "end_turn"),  # follow-ups: prose,
        _message("There are none, sorry.", "end_turn"),          # ... asked again,
        _message("Nothing to list.", "end_turn"),                # ... and again
        _message("- Send the notes.", "end_turn"),             # follow-ups, a later run
        _message("", "max_tokens"),                            # title: has a fallback
    ])

    provider._call(**CALL)
    assert provider.extract_followups("summaries") == []
    assert provider.degraded == ["body", "follow-ups"]
    assert len(sent) == 4  # one for the body (a cut-off answer is not asked for again)

    # The prose answers were not cached, so asking again reaches the model and can succeed.
    assert provider.extract_followups("summaries") == ["Send the notes."]
    assert len(sent) == 5

    from vtd.ai.base import SectionInput
    title = provider.name_section(SectionInput(0, None, 0, 10, "words", []))
    assert title == "Section 1" and provider.degraded == ["body", "follow-ups"]


def test_none_is_a_complete_answer_for_lists(tmp_path):
    provider, sent = _recording_provider(tmp_path, [
        _message("NONE", "end_turn"), _message("None.", "end_turn"),
    ])

    assert provider.extract_followups("summaries") == []
    assert provider.build_glossary("summaries") == []
    assert provider.degraded == []
    assert provider.extract_followups("summaries") == [] and len(sent) == 2  # cached


def test_whole_document_calls_are_told_they_read_summaries(tmp_path):
    provider, sent = _recording_provider(tmp_path, [])
    provider._content_type = "podcast"

    provider.write_themes(["One"], "summaries")
    provider._call(**CALL)

    assert "summaries of the sections of a podcast episode" in sent[0]["system"]
    assert "transcript" not in sent[0]["system"] and "frames" not in sent[0]["system"]
    assert "excerpt of a podcast episode transcript" in sent[1]["system"]


def test_cached_answer_that_is_not_text_is_ignored(tmp_path):
    cache.store(tmp_path, "abc", "summary", {"not": "text"})
    assert cache.load(tmp_path, "abc") is None
