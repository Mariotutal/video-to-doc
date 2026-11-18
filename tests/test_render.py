import json
import re

import pytest

from vtd import ai_synth
from vtd import manifest as mf
from vtd.ai.base import ProviderLimitError
from vtd.config import CONTENT_TYPES, LANGUAGE_NAMES, VTDConfig
from vtd.stages import render
from vtd.stages.labels import LABELS


class FakeProvider:
    name = "fake"

    def __init__(self, degraded=(), limited=False):
        self.degraded = list(degraded)
        self._limited = limited

    def name_section(self, section):
        return "TITLE-FROM-MODEL"

    def write_section_body(self, section):
        return "BODY-FROM-MODEL"

    def write_themes(self, sections_titles, summaries):
        return "THEMES-FROM-MODEL"

    def extract_followups(self, summaries):
        return ["FOLLOWUP-FROM-MODEL"]

    def build_glossary(self, summaries):
        if self._limited:
            raise ProviderLimitError("rate limit")
        return []


def _workspace(tmp_path, *, speakers=True, frames=True):
    (tmp_path / "transcript.txt").write_text("hello world")
    (tmp_path / "transcript.vtt").write_text("00:00:00.000 --> 00:00:10.000\nhello world\n")
    (tmp_path / "sections.json").write_text(json.dumps([
        {"index": 0, "start_s": 0, "end_s": 10, "duration_s": 10,
         "is_silent": False, "cue_count": 1, "title": None},
        {"index": 1, "start_s": 10, "end_s": 20, "duration_s": 10,
         "is_silent": True, "cue_count": 0, "title": None},
        {"index": 2, "start_s": 20, "end_s": 30, "duration_s": 10,  # not silent, but no cues
         "is_silent": False, "cue_count": 0, "title": None},
    ]))
    if speakers:
        (tmp_path / "speakers.json").write_text(json.dumps([
            {"name": "Ana Lopez", "frame_count": 3, "first_seen_s": 0, "last_seen_s": 9},
        ]))
    if frames:
        (tmp_path / "frames").mkdir()
        (tmp_path / "frames" / "frame_0001.jpg").write_bytes(b"x")
    return tmp_path


def _synthesize(tmp_path, monkeypatch, provider=None, **config):
    monkeypatch.setattr(ai_synth, "_build_provider", lambda c, w: provider or FakeProvider())
    manifest = mf.Manifest(source_video="talk.mp4", duration_seconds=10, width=640, height=360)
    incomplete = ai_synth.synthesize(tmp_path, VTDConfig(**config), manifest)
    return (tmp_path / "notes.md").read_text(), incomplete


# Words that only appear in the English labels; a Spanish document must contain none of them.
ENGLISH_ONLY = re.compile(
    r"\b(Source|Duration|Glossary|Sections|Timeline|Chapters|Term|Definition|None detected|"
    r"Field|Value|Language|Synthesized|themes|follow-ups|Frames|References|Workspace|"
    r"Silence|Untitled|empty|segment)\b"
)


@pytest.mark.parametrize("content_type", CONTENT_TYPES)
@pytest.mark.parametrize("language", LANGUAGE_NAMES)
@pytest.mark.parametrize("populated", [True, False])
def test_every_template_renders_in_every_language(tmp_path, monkeypatch, content_type,
                                                  language, populated):
    _workspace(tmp_path, speakers=populated, frames=populated)

    notes, incomplete = _synthesize(tmp_path, monkeypatch, content_type=content_type,
                                    output_language=language)

    assert incomplete == []
    for expected in ("TITLE-FROM-MODEL", "BODY-FROM-MODEL", "THEMES-FROM-MODEL",
                     "FOLLOWUP-FROM-MODEL", LABELS[language]["no_glossary"], "talk.mp4",
                     LABELS[language]["silence"], LABELS[language]["untitled"]):
        assert expected in notes
    assert "{{" not in notes and "{%" not in notes
    if language == "es":
        assert not ENGLISH_ONLY.search(notes), ENGLISH_ONLY.search(notes).group(0)


def test_languages_define_the_same_labels():
    assert {lang: sorted(table) for lang, table in LABELS.items()} == {
        lang: sorted(LABELS["en"]) for lang in LABELS
    }
    assert set(LABELS) == set(LANGUAGE_NAMES)


def test_synthesis_reports_what_is_missing(tmp_path, monkeypatch):
    _workspace(tmp_path)
    provider = FakeProvider(degraded=["body §1"], limited=True)

    notes, incomplete = _synthesize(tmp_path, monkeypatch, provider)

    assert incomplete == ["body §1", "glossary"]
    assert LABELS["en"]["no_glossary"] in notes  # the document is still written


def test_themes_placeholder_is_in_the_output_language(tmp_path, monkeypatch):
    class NoThemes(FakeProvider):
        def write_themes(self, sections_titles, summaries):
            raise ProviderLimitError("rate limit")

    _workspace(tmp_path)

    notes, incomplete = _synthesize(tmp_path, monkeypatch, NoThemes(), output_language="es")

    assert incomplete == ["themes"]
    assert LABELS["es"]["themes_unavailable"] in notes


def test_documents_describe_what_the_workspace_was_built_with(tmp_path, monkeypatch):
    _workspace(tmp_path, speakers=False, frames=False)
    manifest = mf.Manifest(source_video="talk.mp4", duration_seconds=10, stage_options={
        "transcribe": {"teams_captions": False, "whisper_model": "small", "language": "es"},
        "extract-frames": {"frame_rate_seconds": 10, "teams_captions": False},
    })
    # Rendered by a run that passes other flags: the header must not repeat them.
    other_run = VTDConfig(whisper_model="large-v3", frame_rate_seconds=5, teams_captions=True)

    render.run_starter(tmp_path, other_run, manifest)

    starter = (tmp_path / "notes.starter.md").read_text()
    assert "Whisper `small`" in starter and "one frame every 10 s" in starter
    assert "Teams" not in starter


def test_teams_transcript_is_named_as_the_source(tmp_path, monkeypatch):
    _workspace(tmp_path)
    monkeypatch.setattr(ai_synth, "_build_provider", lambda c, w: FakeProvider())
    manifest = mf.Manifest(source_video="call.mp4", duration_seconds=10, stage_options={
        "transcribe": {"teams_captions": True, "frame_rate_seconds": 5},
    })

    ai_synth.synthesize(tmp_path, VTDConfig(content_type="meeting", output_language="es"),
                        manifest)

    notes = (tmp_path / "notes.md").read_text()
    assert LABELS["es"]["teams_captions"] in notes and "Whisper" not in notes
