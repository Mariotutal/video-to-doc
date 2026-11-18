"""Checks made before work starts, one way of reading workspace files, documents that agree."""
import json
import re
import sys

import pytest

from vtd import ai_synth, pipeline, vtt
from vtd import manifest as mf
from vtd.ai import ollama_provider
from vtd.config import CONTENT_TYPES, PROVIDERS, AIConfig, VTDConfig
from vtd.stages import identify_speakers as ids
from vtd.stages import render, transcribe_teams
from vtd.stages.labels import LABELS
from vtd.utils import VTDError, read_json

from .fakes import fake_anthropic, replying

# ---- nothing runs when the AI notes could not be written at the end -------------------


def _ai(provider, key=None):
    return VTDConfig(ai=AIConfig(enabled=True, provider=provider, api_key=key,
                                 model=PROVIDERS[provider]["default_model"]))


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_a_missing_key_stops_the_run_before_any_stage(tmp_path, monkeypatch, provider):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio")
    monkeypatch.setattr(pipeline, "ffprobe_info", lambda s: {
        "duration": 60.0, "width": None, "height": None, "has_video": False, "has_audio": True})
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    for name in pipeline.STAGES:
        monkeypatch.setitem(pipeline.STAGES, name, lambda *a: pytest.fail("a stage ran"))
    with pytest.raises(VTDError, match="no API key found"):
        pipeline.run_pipeline(source=source, workspace=tmp_path / "ws", config=_ai(provider),
                              resume=False, only_stages=None, verbose=False)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_a_provider_package_that_is_not_installed_is_named(tmp_path, monkeypatch, provider):
    monkeypatch.setitem(sys.modules, provider, None)  # importing it now fails
    with pytest.raises(VTDError, match=rf"{provider} package not installed"):
        pipeline._preflight_ai(_ai(provider, key="k"), tmp_path)


def test_ollama_not_running_is_reported_before_any_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens there
    with pytest.raises(VTDError, match="Cannot reach Ollama"):
        pipeline._preflight_ai(_ai("ollama"), tmp_path)


def test_an_unknown_provider_is_an_error_not_a_default(tmp_path):
    config = _ai("anthropic", key="k")
    config.ai.provider = "nope"
    with pytest.raises(VTDError, match="Unknown AI provider"):
        ai_synth._build_provider(config, tmp_path)


def test_the_note_about_a_model_that_cannot_see_is_given_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ollama_provider.OllamaProvider, "_capabilities", lambda self: [])
    monkeypatch.setattr(ollama_provider, "_TOLD_BLIND", set())
    for _ in range(2):  # once to check before the run, once to write the notes
        ollama_provider.OllamaProvider(_ai("ollama"), tmp_path)
    assert capsys.readouterr().err.count("cannot read images") == 1


def _stubbed_run(tmp_path, monkeypatch, first_language):
    """A finished workspace, then the order of what a second run does."""
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio")
    calls = []
    monkeypatch.setattr(pipeline, "ffprobe_info", lambda s: {
        "duration": 60.0, "width": None, "height": None, "has_video": False, "has_audio": True})
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    monkeypatch.setattr(pipeline, "_preflight_ai", lambda config, ws: calls.append("check ai"))
    monkeypatch.setattr(ai_synth, "synthesize", lambda ws, config, m: calls.append("notes") or [])
    for name in pipeline.STAGES:
        monkeypatch.setitem(pipeline.STAGES, name, lambda ws, c, m, name=name: calls.append(name))

    def run(language):
        calls.clear()
        config = _ai("anthropic", key="k")
        config.language = language
        pipeline.run_pipeline(source=source, workspace=tmp_path / "ws", config=config,
                              resume=True, only_stages=None, verbose=False)
        return list(calls)

    run(first_language)
    return run


def test_the_provider_is_checked_before_a_resume_redoes_what_the_notes_are_built_on(
        tmp_path, monkeypatch):
    run = _stubbed_run(tmp_path, monkeypatch, "es")
    assert run("es") == []  # nothing to redo: nothing to check
    again = run("en")  # the transcript is redone, so the notes will be too
    assert again[0] == "check ai" and again[1] == "transcribe" and again[-1] == "notes"


def test_a_url_run_that_leaves_the_ai_stage_out_needs_no_key(monkeypatch):
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    pipeline.check_before_download(_ai("anthropic"), ["transcribe"])
    with pytest.raises(VTDError, match="no API key found"):
        pipeline.check_before_download(_ai("anthropic"), ["transcribe", "synthesize"])
    with pytest.raises(VTDError, match="no API key found"):
        pipeline.check_before_download(_ai("anthropic"), None)


# ---- --resume ------------------------------------------------------------------------

def test_the_starter_document_is_redone_when_the_output_language_changes():
    manifest = mf.Manifest()
    manifest.mark_stage("render-starter", pipeline._stage_options("render-starter", VTDConfig()))
    assert pipeline._is_current("render-starter", VTDConfig(), manifest)
    assert not pipeline._is_current("render-starter", VTDConfig(output_language="es"), manifest)


# ---- workspace files -----------------------------------------------------------------

def test_a_workspace_file_that_is_missing_or_damaged(tmp_path, capsys):
    assert read_json(tmp_path / "absent.json", []) == []
    assert capsys.readouterr().err == ""  # a file that was never written is not a problem
    (tmp_path / "sections.json").write_text("{not json")
    for _ in range(3):  # read for every section of a document: reported once
        assert read_json(tmp_path / "sections.json", []) == []
    assert capsys.readouterr().err.count("sections.json cannot be read") == 1
    with pytest.raises(VTDError, match="--only detect-sections"):
        read_json(tmp_path / "sections.json", required="detect-sections")
    with pytest.raises(VTDError, match="run the detect-sections stage first"):
        read_json(tmp_path / "absent.json", required="detect-sections")


def test_damaged_sections_stop_the_ai_stage_with_a_message(tmp_path):
    (tmp_path / "transcript.txt").write_text("hola")
    (tmp_path / "sections.json").write_text("[{")
    with pytest.raises(VTDError, match="sections.json cannot be read"):
        ai_synth.synthesize(tmp_path, VTDConfig(), mf.Manifest())


def test_names_edited_into_something_else_are_ignored_with_a_note(tmp_path, capsys):
    (tmp_path / ids.NAMES_NAME).write_text('["Ana", "Luis"]')
    assert ids.speaker_names(tmp_path) == {}
    assert "not a table of speaker numbers" in capsys.readouterr().err


def test_a_transcript_written_is_read_back_the_same(tmp_path):
    cues = [{"start_s": 0.0, "end_s": 2.5, "text": "Hola."},
            {"start_s": 3725.25, "end_s": 3730.0, "text": "Ana: seguimos"}]
    vtt.write_vtt(tmp_path / "t.vtt", cues)
    assert vtt.parse_vtt(tmp_path / "t.vtt") == cues
    assert vtt.stamp(3725.25) == "01:02:05.250"
    assert not vtt.is_spoken(" ") and not vtt.is_spoken("[BLANK_AUDIO]") and vtt.is_spoken("sí")


def test_teams_speakers_are_counted_in_frames_like_the_label_says(tmp_path, monkeypatch):
    (tmp_path / "frames").mkdir()
    frames = [tmp_path / "frames" / f"frame_{i:04d}.jpg" for i in range(1, 5)]
    for frame in frames:
        frame.write_bytes(b"x")
    monkeypatch.setattr(transcribe_teams, "require_binary", lambda *a: "tesseract")
    monkeypatch.setattr(transcribe_teams, "list_frames", lambda folder: frames)
    seen = {1: [("Lopez, Ana", "Buenos días a todos")], 2: [("Lopez, Ana", "Buenos días a todos")],
            3: [("Lopez, Ana", "Buenos días a todos")], 4: [("Ruiz, Leo", "Gracias")]}
    monkeypatch.setattr(transcribe_teams, "_ocr_lines", lambda frame: int(frame.stem[-1]))
    monkeypatch.setattr(transcribe_teams, "_parse_entries", lambda number: seen[number])
    transcribe_teams.run(tmp_path, VTDConfig(teams_captions=True, frame_rate_seconds=5),
                         mf.Manifest())
    speakers = {s["name"]: s["frame_count"] for s in read_json(tmp_path / "speakers.json")}
    assert speakers == {"Lopez, Ana": 3, "Ruiz, Leo": 1}  # one caption, shown in three frames


# ---- documents -----------------------------------------------------------------------

def _templates():
    return {path.name: path.read_text() for path in render.TEMPLATES.glob("*.j2")}


def test_every_language_has_the_same_labels_and_all_are_used():
    keys = {language: set(table) for language, table in LABELS.items()}
    assert keys["en"] == keys["es"]
    used = set(re.findall(r"\bt\.(\w+)", "".join(_templates().values())))
    code_only = {"silence", "speaker", "no_audio", "untitled", "empty_section",
                 "themes_unavailable", "teams_captions"}
    assert used - keys["en"] == set()             # nothing a template asks for is missing
    assert keys["en"] - used - code_only == set()  # nothing left behind by an old template


def test_the_four_ai_documents_share_their_common_parts():
    templates = _templates()
    for kind in CONTENT_TYPES:
        text = templates[f"full_{kind}.md.j2"]
        for part in ("source_table()", "outline(", "glossary_table()", "timeline()", "bullets("):
            assert f"parts.{part}" in text, (kind, part)
        assert "t.section_timeline" in text  # one name for the list of sections


def _starter(tmp_path, language, turns=None):
    (tmp_path / "transcript.txt").write_text("hola a todos")
    (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:10.000\nhola a todos\n\n")
    (tmp_path / "sections.json").write_text(json.dumps([{
        "index": 0, "start_s": 0, "end_s": 10, "duration_s": 10, "is_silent": False,
        "cue_count": 1, "title": None}]))
    if turns:
        (tmp_path / ids.TURNS_NAME).write_text(json.dumps(turns))
    config = VTDConfig(output_language=language)
    render.run_starter(tmp_path, config, mf.Manifest(source_video="/v/clase.mp4"))
    return (tmp_path / config.starter_doc_name).read_text()


def test_the_starter_document_is_written_in_the_output_language(tmp_path):
    text = _starter(tmp_path, "es")
    assert text.startswith(f"# {tmp_path.name} — notas iniciales")
    for heading in ("## Fuente", "## Línea de tiempo de secciones", "## Transcripción"):
        assert heading in text
    assert not re.search(r"Source|Duration|Section timeline|content \||unknown", text)
    assert "no atribuye las intervenciones" in text


def test_the_starter_document_does_not_call_a_marked_transcript_unattributed(tmp_path):
    turns = [{"start_s": 0, "end_s": 5, "speaker": 1}, {"start_s": 5, "end_s": 10, "speaker": 2}]
    text = _starter(tmp_path, "en", turns)
    assert "marks who speaks, told apart by voice (speakers found: 2)" in text
    assert "unattributed" not in text and "Speaker 1: hola a todos" in text


def test_notes_list_the_speakers_told_apart_by_voice(tmp_path):
    (tmp_path / ids.TURNS_NAME).write_text(json.dumps([
        {"start_s": 0, "end_s": 60, "speaker": 1}, {"start_s": 60, "end_s": 75, "speaker": 2},
        {"start_s": 80, "end_s": 140, "speaker": 1}]))
    assert ai_synth._voices(tmp_path, {2: "Elena"}, "Hablante") == [
        {"name": "Hablante 1", "spoke_s": 120, "first_s": 0},
        {"name": "Elena", "spoke_s": 15, "first_s": 60}]


@pytest.mark.parametrize("language, people", [
    ("en", ["- **Speaker 1** — spoke for 0:01:00 in all, first at 0:00:00.",
            "- **Elena** — spoke for 0:00:15 in all, first at 0:01:00.",
            "- **Ana Lopez** — name shown on screen from 0:00:00, in 3 sampled frame(s)."]),
    ("es", ["- **Hablante 1** — habló 0:01:00 en total, por primera vez en 0:00:00.",
            "- **Elena** — habló 0:00:15 en total, por primera vez en 0:01:00.",
            "- **Ana Lopez** — nombre visto en pantalla desde 0:00:00, en 3 fotograma(s) "
            "muestreado(s)."]),
])
def test_meeting_notes_list_voices_and_the_names_read_on_screen(
        tmp_path, monkeypatch, language, people):
    (tmp_path / "transcript.txt").write_text("hola a todos")
    (tmp_path / "transcript.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:01:00.000\nhola a todos\n\n"
        "00:01:00.000 --> 00:01:15.000\nbuenas tardes\n\n")
    (tmp_path / "sections.json").write_text(json.dumps([{
        "index": 0, "start_s": 0, "end_s": 75, "duration_s": 75, "is_silent": False,
        "cue_count": 2, "title": None}]))
    (tmp_path / "speakers.json").write_text(json.dumps([
        {"name": "Ana Lopez", "frame_count": 3, "first_seen_s": 0, "last_seen_s": 9}]))
    (tmp_path / ids.TURNS_NAME).write_text(json.dumps([
        {"start_s": 0, "end_s": 60, "speaker": 1}, {"start_s": 60, "end_s": 75, "speaker": 2}]))
    (tmp_path / ids.NAMES_NAME).write_text('{"1": "", "2": "Elena"}')

    class Provider:
        name, degraded = "fake", []
        name_section = lambda self, section: "Título"  # noqa: E731
        write_section_body = lambda self, section: "Contenido."  # noqa: E731
        write_themes = lambda self, titles, digest: "### Tema"  # noqa: E731
        extract_followups = lambda self, digest: []  # noqa: E731
        build_glossary = lambda self, digest: []  # noqa: E731

    monkeypatch.setattr(ai_synth, "_build_provider", lambda config, workspace: Provider())
    config = VTDConfig(content_type="meeting", output_language=language)
    ai_synth.synthesize(tmp_path, config, mf.Manifest(source_video="reunion.mp4"))
    notes = (tmp_path / "notes.md").read_text()
    assert notes.split("## ")[2].splitlines()[2:5] == people


# ---- prompts -------------------------------------------------------------------------

def test_an_empty_answer_to_the_merge_keeps_every_follow_up(tmp_path):
    create, sent = replying(["", " ", ""])
    provider = fake_anthropic(tmp_path, create)
    assert provider.merge_followups(["read X", "do Y"]) == ["read X", "do Y"]
    assert len(sent) == 3 and provider.degraded == ["follow-ups"]


def test_merging_follow_ups_into_none_is_an_answer_not_a_failure(tmp_path):
    create, sent = replying(["NONE"])
    provider = fake_anthropic(tmp_path, create)
    assert provider.merge_followups(["Take a break", "Chat about the weather"]) == []
    assert len(sent) == 1 and provider.degraded == []


@pytest.mark.parametrize("kind, asked_for", [
    ("generic", "themes that recur"), ("meeting", "themes that recur"),
    ("lecture", "key concepts"), ("podcast", "notable themes")])
def test_the_closing_section_is_asked_for_as_its_heading_names_it(tmp_path, kind, asked_for):
    create, sent = replying(["### One\n\nText."])
    fake_anthropic(tmp_path, create, content_type=kind).write_themes(["A", "B"], "summaries")
    assert asked_for in sent[0] and "no heading above them" in sent[0]
    assert "Cross-cutting themes" not in sent[0]  # naming the section invites a second heading
