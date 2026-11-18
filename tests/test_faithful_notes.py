"""Notes that stay with what was said: quotes, attribution, length, tasks, outline."""
import json
from types import SimpleNamespace

import pytest

from vtd import ai_synth, export_pdf
from vtd.ai import cache
from vtd.ai.base import SectionInput, _length_hint, unquote, unsupported_quotes
from vtd.config import VTDConfig
from vtd.manifest import Manifest
from vtd.stages import detect_sections

from .fakes import fake_anthropic, replying

EXCERPT = ("Yo no recomiendo el uso de la ruta de montaña intermedia.\n"
           "Asumir la simetría también es un poco de tener fe.")


# ---- quotations ---------------------------------------------------------------

def test_a_quotation_is_checked_against_what_was_said():
    said = 'Dijo "yo no recomiendo el uso de la ruta de montaña intermedia" claramente.'
    invented = 'Dijo "la ruta intermedia puede utilizarse por conveniencia" claramente.'
    assert unsupported_quotes(said, EXCERPT) == []
    assert unsupported_quotes(invented, EXCERPT) == [
        '"la ruta intermedia puede utilizarse por conveniencia"']


def test_case_accents_punctuation_and_line_breaks_do_not_matter():
    text = "«YO NO RECOMIENDO EL USO DE LA RUTA... intermedia. Asumir la simetría»"
    assert unsupported_quotes(text, EXCERPT) == []


def test_each_stretch_around_an_ellipsis_must_have_been_said():
    text = '"yo no recomiendo el uso [...] es un poco de tener esperanza"'
    assert unsupported_quotes(text, EXCERPT) == [text]


def test_short_quoted_terms_are_not_quotations():
    assert unsupported_quotes('el llamado "efecto dominó" y la "ruta corta"', EXCERPT) == []


def test_a_quotation_that_was_not_said_loses_its_marks_not_its_words():
    text = 'Advierte que *"nunca se debe medir así"* y que "asumir la simetría también es un poco de tener fe".'
    assert unquote(text, unsupported_quotes(text, EXCERPT)) == (
        'Advierte que nunca se debe medir así y que "asumir la simetría también es un poco de tener fe".')


def _provider(tmp_path, replies):
    create, sent = replying(replies)
    return fake_anthropic(tmp_path, create, language="es", content_type="lecture"), sent


def _section(text=EXCERPT):
    return SectionInput(index=0, title="T", start_s=0, end_s=60, transcript_chunk=text, frame_paths=[])


def test_a_summary_with_a_made_up_quotation_is_asked_for_again(tmp_path):
    good = 'El docente dice "yo no recomiendo el uso de la ruta de montaña intermedia".'
    provider, sent = _provider(tmp_path, ['Dice "la ruta intermedia siempre es buena idea".', good])

    assert provider.write_section_body(_section()) == good
    assert "not in the excerpt word for word" in sent[1]
    assert '"la ruta intermedia siempre es buena idea"' in sent[1]
    assert provider.degraded == []


def test_a_quotation_still_made_up_after_the_retries_is_unquoted_and_kept(tmp_path):
    bad = 'Dice "la ruta intermedia siempre es buena idea".'
    provider, sent = _provider(tmp_path, [bad] * 2)

    body = provider.write_section_body(_section())

    assert body == "Dice la ruta intermedia siempre es buena idea."
    assert len(sent) == 2 and provider.degraded == []  # asked again once, then corrected here
    assert provider.write_section_body(_section()) == body and len(sent) == 2  # cached repaired


def test_a_cut_off_answer_is_never_corrected_and_kept_as_good(tmp_path):
    replies = [("Dice \"la ruta intermedia siempre es buena idea\".", "end_turn"),
               ("Dice \"la ruta intermedia siempre es buena idea\" y se cort", "max_tokens")]
    provider, _sent = _provider(tmp_path, [])
    provider._client.messages.create = lambda **kwargs: SimpleNamespace(
        content=[SimpleNamespace(text=replies[0][0])], stop_reason=replies.pop(0)[1])

    assert provider.write_section_body(_section()).endswith("se cort")
    assert provider.degraded == ["body §1"]
    assert not cache.cache_dir(tmp_path).exists() or not list(cache.cache_dir(tmp_path).iterdir())


def test_single_words_around_ellipses_do_not_make_a_quotation_true():
    source = "Hoy vamos a ver el modelo de tres capas. La capa de datos es la más importante de todas."
    assert unsupported_quotes('"hoy ... de ... la ... todas"', source) == ['"hoy ... de ... la ... todas"']
    assert unsupported_quotes('"la capa de datos ... vamos a ver"', source)  # out of order
    assert unsupported_quotes('"vamos a ver ... la capa de datos"', source) == []


def test_inch_marks_and_code_are_not_quotations():
    text = 'una pantalla de 15" y otra de 27" y usa `print("hola mundo desde python")` aquí'
    assert unsupported_quotes(text, EXCERPT) == []
    assert unquote(text, ['" y otra de 27"']) == text


def test_unquoting_keeps_bold_and_italics_balanced():
    text = '**Resumen "esto no se dijo nunca aquí"** y *"tampoco esto otro se dijo"* fin'
    assert unquote(text, unsupported_quotes(text, EXCERPT)) == (
        "**Resumen esto no se dijo nunca aquí** y tampoco esto otro se dijo fin")


# ---- prompts ------------------------------------------------------------------

def test_the_summary_is_asked_in_proportion_to_the_excerpt():
    assert "one or two sentences" in _length_hint(300)
    assert _length_hint(3000).startswith("Aim for 4–8 sentences")
    assert _length_hint(6000).startswith("Aim for 9–14 sentences")
    assert _length_hint(30000).startswith("Aim for 10–14 sentences")  # never an inverted range


def test_the_prompt_asks_for_recommendations_and_forbids_guessing_who_spoke(tmp_path):
    provider, sent = _provider(tmp_path, ["Resumen breve."])
    provider.write_section_body(_section("hola " * 40))
    prompt = sent[0]
    assert "what is recommended and what is advised against" in prompt
    assert "Name a person only for words the excerpt itself shows that person saying" in prompt
    assert "one or two sentences" in prompt


# ---- tasks read from each section ---------------------------------------------

def test_follow_ups_are_read_from_each_section_and_merged(tmp_path):
    provider, sent = _provider(tmp_path, [
        "- Leer el paper de García y López para la próxima semana.",
        "NONE",
        "- Leer el paper de García y López.\n- Traer un ejemplo de escala.",
        "- Leer el paper de García y López para la próxima semana.\n- Traer un ejemplo de escala.",
    ])
    noted = []
    for text in ("lean el paper", "nada aquí", "lean y traigan"):
        noted += provider.section_followups(_section(text))

    assert len(noted) == 3
    assert "reading or open question the lecturer left" in sent[0]
    assert "TRANSCRIPT EXCERPT" in sent[0]
    merged = provider.merge_followups(noted)
    assert merged == ["Leer el paper de García y López para la próxima semana.",
                      "Traer un ejemplo de escala."]
    assert "ITEMS NOTED SECTION BY SECTION" in sent[3] and "do not add any" in sent[3]


def test_a_section_whose_tasks_cannot_be_read_is_a_gap(tmp_path):
    provider, sent = _provider(tmp_path, ["No hay nada que listar aquí."] * 3)
    assert provider.section_followups(_section()) == []
    assert provider.degraded == ["follow-ups §1"] and len(sent) == 3


def test_a_merge_that_fails_keeps_every_item(tmp_path):
    provider, _sent = _provider(tmp_path, ["prosa"] * 3)
    assert provider.merge_followups(["uno", "dos"]) == ["uno", "dos"]


def test_meetings_are_still_asked_for_the_owner_of_each_item(tmp_path):
    provider, sent = _provider(tmp_path, ["NONE", "NONE"])
    provider._content_type = "meeting"
    provider.section_followups(_section())
    assert "with the owner if mentioned" in sent[0]
    provider._content_type = "lecture"
    provider.section_followups(_section("otra cosa"))
    assert "owner" not in sent[1] and "not what happened during the session" in sent[1]


def test_one_or_no_follow_up_needs_no_merge(tmp_path):
    provider, sent = _provider(tmp_path, [])
    assert provider.merge_followups([]) == [] and provider.merge_followups(["uno"]) == ["uno"]
    assert sent == []


class _Provider:
    name = "fake"
    degraded: list = []

    def name_section(self, section):
        return f"Título {section.index + 1}"

    def write_section_body(self, section):
        return "Contenido."

    def section_followups(self, section):
        return ["Leer el capítulo 2."] if section.index == 0 else []

    def merge_followups(self, items):
        return [f"final: {item}" for item in items]

    def extract_followups(self, digest):
        pytest.fail("the tasks must come from the sections, not from the summaries")

    def write_themes(self, titles, digest):
        assert "No se transcribió" not in digest and "Silencio" not in digest
        return "### Tema"

    def build_glossary(self, digest):
        return []


def _notes(tmp_path, monkeypatch):
    (tmp_path / "transcript.txt").write_text("x")
    (tmp_path / "transcript.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:50.000\nprimera parte\n\n"
        "00:20:00.000 --> 00:20:50.000\nsegunda parte\n\n")
    (tmp_path / "sections.json").write_text(json.dumps([
        {"index": 0, "start_s": 0, "end_s": 600, "duration_s": 600, "is_silent": False,
         "cue_count": 1, "title": None},
        {"index": 1, "start_s": 600, "end_s": 1200, "duration_s": 600, "is_silent": True,
         "cue_count": 0, "title": None},
        {"index": 2, "start_s": 1200, "end_s": 1800, "duration_s": 600, "is_silent": False,
         "cue_count": 1, "title": None}]))
    monkeypatch.setattr(ai_synth, "_build_provider", lambda config, workspace: _Provider())
    config = VTDConfig(content_type="lecture", output_language="es")
    manifest = Manifest(source_video="/Users/alguien/Videos/Clase 2.mp4", duration_seconds=1800)
    ai_synth.synthesize(tmp_path, config, manifest)
    return (tmp_path / "notes.md").read_text()


def test_tasks_in_the_notes_come_from_the_sections(tmp_path, monkeypatch):
    assert "- final: Leer el capítulo 2." in _notes(tmp_path, monkeypatch)


def test_a_break_is_not_a_numbered_section_but_stays_in_the_timeline(tmp_path, monkeypatch):
    lines = _notes(tmp_path, monkeypatch).split("\n")
    assert "### 1. Título 1  ·  [0:00:00 – 0:10:00]" in lines
    assert "### 2. Título 2  ·  [0:20:00 – 0:30:00]" in lines   # numbered without the break
    assert not any(line.startswith("### ") and "Silencio" in line for line in lines)
    assert "| — | 0:10:00 | 0:20:00 | 0:10:00 | Silencio |" in lines
    assert "| 2 | 0:20:00 | 0:30:00 | 0:10:00 | Título 2 |" in lines


def test_the_heading_shows_the_file_name_not_the_full_path(tmp_path, monkeypatch):
    notes = _notes(tmp_path, monkeypatch)
    assert notes.startswith("# Clase 2.mp4 — ")
    assert export_pdf._title(notes.split("\n")[0][2:], "/Users/alguien/Videos/Clase 2.mp4")[0] == "Clase 2"
    assert export_pdf._title("Clase 2.mp4.vtd — starter notes", "/Users/alguien/Videos/Clase 2.mp4") == (
        "Clase 2", "Starter notes")


# ---- sections -----------------------------------------------------------------

def _cues(count, seconds=10.0):
    return [{"start_s": i * seconds, "end_s": (i + 1) * seconds, "text": "x" * 99}
            for i in range(count)]


def test_long_sections_are_split_into_even_parts():
    cues = _cues(70)  # 6,999 characters: two parts, not one full and a leftover of ten lines
    cuts = detect_sections._split_points(cues, max_chars=6000)
    assert len(cuts) == 1 and 300 <= cuts[0] <= 400


def test_a_closing_remark_after_the_last_cut_stays_with_the_part_before():
    # 59 full lines, then a short goodbye that would otherwise be a section of 20 seconds.
    cues = _cues(118, seconds=5.0) + [
        {"start_s": 590.0 + i * 2, "end_s": 592.0 + i * 2, "text": "y " * 150} for i in range(10)]
    boundaries = detect_sections._split_long_sections([0.0, 610.0], cues, 6000, min_s=120)
    assert all(b - a >= 120 for a, b in zip(boundaries, boundaries[1:], strict=False))
