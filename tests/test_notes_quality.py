"""Cleaner transcripts, sections by subject, frames worth showing, answers checked."""
import json
import shutil
import subprocess
from argparse import Namespace

import pytest

from vtd import ai_synth
from vtd.ai import cache
from vtd.ai.base import _chapters, _is_title, written_in
from vtd.config import VTDConfig, resolve_config
from vtd.manifest import Manifest
from vtd.stages import detect_sections, extract_frames, transcribe
from vtd.stages.detect_sections import _merge_short_sections
from vtd.utils import VTDError

from .fakes import fake_anthropic, replying

# ---- 1. Lines Whisper repeats ------------------------------------------------

def _cue(i, text):
    return {"start_s": float(i), "end_s": float(i + 1), "text": text}


def test_a_line_repeated_over_and_over_is_kept_once():
    cues = [_cue(0, "Hola."), *(_cue(i, "Y lo dejo anotado en la pizarra.") for i in range(1, 9)),
            _cue(9, "Ahí estamos."), _cue(10, "y lo dejo anotado en la pizarra")]
    kept = transcribe.drop_repeats(cues)
    assert [c["text"] for c in kept] == [
        "Hola.", "Y lo dejo anotado en la pizarra.", "Ahí estamos.", "y lo dejo anotado en la pizarra"]
    assert kept[1]["start_s"] == 1.0  # the first of the run


def test_something_said_twice_is_left_alone():
    cues = [_cue(0, "Gracias."), _cue(1, "Gracias."), _cue(2, "Sigamos.")]
    assert transcribe.drop_repeats(cues) == cues


def test_a_cleaned_transcript_keeps_what_whisper_wrote_beside_it(tmp_path):
    lines = ["uno", "eco", "eco", "eco", "eco", "dos"]
    (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n" + "".join(
        f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\n{line}\n\n" for i, line in enumerate(lines)))
    (tmp_path / "transcript.txt").write_text("\n".join(lines) + "\n")

    assert transcribe.clean_transcript(tmp_path) == 3

    assert (tmp_path / "transcript.txt").read_text() == "uno\neco\ndos\n"
    assert [c["text"] for c in detect_sections.parse_vtt(tmp_path / "transcript.vtt")] == [
        "uno", "eco", "dos"]
    assert (tmp_path / "transcript.raw.txt").read_text().count("eco") == 4
    assert len(detect_sections.parse_vtt(tmp_path / "transcript.raw.vtt")) == 6


def test_a_transcript_without_repeats_is_not_rewritten(tmp_path):
    (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nuno\n\n")
    assert transcribe.clean_transcript(tmp_path) == 0
    assert not (tmp_path / "transcript.raw.vtt").exists()


# ---- 2. Sections by subject --------------------------------------------------

def test_a_short_section_joins_the_next_and_a_short_last_one_the_previous():
    assert _merge_short_sections([0, 300, 320, 350, 700, 730], 120) == [0, 300, 730]
    assert _merge_short_sections([0, 50, 400, 800], 120) == [0, 400, 800]
    assert _merge_short_sections([0, 50, 90], 120) == [0, 90]     # all short: one section
    assert _merge_short_sections([0, 50, 400], 0) == [0, 50, 400]  # switched off


def test_min_section_s_is_an_option_and_part_of_the_stage_record():
    from vtd import pipeline
    assert resolve_config(Namespace()).min_section_s == 120
    assert resolve_config(Namespace(min_section_s=0)).min_section_s == 0
    with pytest.raises(VTDError, match="min_section_s"):
        resolve_config(Namespace(min_section_s=-5))
    assert pipeline._stage_options("detect-sections", VTDConfig())["min_section_s"] == 120


@pytest.mark.parametrize("answer, expected", [
    ("1-3 :: Introducción al curso\n4-4 :: Ejercicio\n5-6 :: Cierre",
     [(1, 3, "Introducción al curso"), (4, 4, "Ejercicio"), (5, 6, "Cierre")]),
    ("- 1–2 :: **Inicio**\n- 3 :: Medio\n- 4-6 :: Fin", [(1, 2, "Inicio"), (3, 3, "Medio"), (4, 6, "Fin")]),
    ("`1-3 :: Intro`\n`4-6 :: **Fin**`", [(1, 3, "Intro"), (4, 6, "Fin")]),
    ("1-6 :: Todo", []),                 # one chapter says nothing
    ("1-3 :: A\n5-6 :: B", []),          # section 4 is missing
    ("1-4 :: A\n4-6 :: B", []),          # section 4 twice
    ("1-3 :: A\n4-5 :: B", []),          # stops before the end
    ("Here are the chapters.", []),
])
def test_chapters_are_accepted_only_when_they_cover_every_section_in_order(answer, expected):
    assert _chapters(answer, 6) == expected


class _Provider:
    name = "fake"
    degraded: list = []

    def __init__(self, chapters=None):
        self._chapters = chapters

    def name_section(self, section):
        return f"Título {section.index + 1}"

    def write_section_body(self, section):
        return "## Resumen\n\nContenido."

    def write_themes(self, titles, digest):
        return "### Tema"

    def extract_followups(self, digest):
        return []

    def build_glossary(self, digest):
        return []

    def group_sections(self, titles, digest):
        assert len(titles) == 6
        return self._chapters


def _notes(tmp_path, monkeypatch, provider, sections=6):
    (tmp_path / "transcript.txt").write_text("x")
    (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n" + "".join(
        f"00:{i:02d}:00.000 --> 00:{i:02d}:50.000\nlínea {i}\n\n" for i in range(sections)))
    (tmp_path / "sections.json").write_text(json.dumps([
        {"index": i, "start_s": i * 60, "end_s": (i + 1) * 60, "duration_s": 60,
         "is_silent": False, "cue_count": 1, "title": None} for i in range(sections)]))
    monkeypatch.setattr(ai_synth, "_build_provider", lambda config, workspace: provider)
    config = VTDConfig(content_type="lecture")
    ai_synth.synthesize(tmp_path, config, Manifest(source_video="/v/clase.mp4", duration_seconds=360))
    return (tmp_path / "notes.md").read_text()


def test_sections_are_listed_under_the_chapters_the_model_proposes(tmp_path, monkeypatch):
    notes = _notes(tmp_path, monkeypatch, _Provider([(1, 2, "Inicio"), (3, 6, "Desarrollo")]))
    lines = notes.split("\n")
    assert "### Inicio  ·  [0:00:00 – 0:02:00]" in lines
    assert "### Desarrollo  ·  [0:02:00 – 0:06:00]" in lines
    assert "#### 3. Título 3  ·  [0:02:00 – 0:03:00]" in lines
    assert lines.index("### Inicio  ·  [0:00:00 – 0:02:00]") < lines.index(
        "#### 2. Título 2  ·  [0:01:00 – 0:02:00]") < lines.index("### Desarrollo  ·  [0:02:00 – 0:06:00]")
    assert "##### Resumen" in lines and "## Resumen" not in lines  # below the section heading


def test_without_chapters_the_sections_are_listed_as_before(tmp_path, monkeypatch):
    notes = _notes(tmp_path, monkeypatch, _Provider([]))
    assert "### 3. Título 3  ·  [0:02:00 – 0:03:00]" in notes.split("\n")
    assert "#### Resumen" in notes.split("\n")


def test_few_sections_are_not_grouped(tmp_path, monkeypatch):
    provider = _Provider()
    provider.group_sections = lambda titles, digest: pytest.fail("not worth asking")
    notes = _notes(tmp_path, monkeypatch, provider, sections=3)
    assert "### 2. Título 2" in notes


# ---- 3. Frames worth showing -------------------------------------------------

def _frames(tmp_path, times):
    (tmp_path / "frames").mkdir()
    for i in range(len(times)):
        (tmp_path / "frames" / f"frame_{i + 1:04d}.jpg").write_bytes(b"jpeg")
    (tmp_path / "frames" / "timestamps.json").write_text(json.dumps(
        {f"frame_{i + 1:04d}.jpg": t for i, t in enumerate(times)}))


def test_frames_with_nothing_to_read_are_not_shown(tmp_path):
    _frames(tmp_path, [0, 60, 120, 180])
    (tmp_path / "frame_text.json").write_text(json.dumps({
        "frame_0001.jpg": 3,    # a face with a name tag
        "frame_0002.jpg": 40,   # a slide
        "frame_0003.jpg": 0,
        "frame_0004.jpg": 25}))
    picked = ai_synth._frames_for_section(tmp_path, VTDConfig(), 0, 200)
    assert [f.name for f in picked] == ["frame_0002.jpg", "frame_0004.jpg"]


def test_every_frame_can_be_kept_for_recordings_of_diagrams_or_photos(tmp_path):
    _frames(tmp_path, [0, 60])
    (tmp_path / "frame_text.json").write_text(json.dumps({"frame_0001.jpg": 1, "frame_0002.jpg": 0}))
    assert ai_synth._frames_for_section(tmp_path, VTDConfig(), 0, 200) == []
    assert len(ai_synth._frames_for_section(tmp_path, VTDConfig(min_frame_words=0), 0, 200)) == 2
    assert resolve_config(Namespace(min_frame_words=0)).min_frame_words == 0


def test_what_was_read_in_the_frames_goes_when_the_frames_are_replaced():
    from vtd import pipeline
    assert "frame_text.json" in pipeline._stage_outputs("extract-frames", VTDConfig())
    assert "frame_text.json" in pipeline._stage_outputs("ocr-speakers", VTDConfig())
    # Whisper's raw transcript must not survive beside a Teams transcript either.
    assert "transcript.raw.vtt" in pipeline._stage_outputs(
        "transcribe", VTDConfig(teams_captions=True))


def test_frames_are_all_candidates_when_they_were_never_read(tmp_path):
    _frames(tmp_path, [0, 60])  # no frame_text.json: --no-ocr, or an older workspace
    assert len(ai_synth._frames_for_section(tmp_path, VTDConfig(), 0, 200)) == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_slide_shown_between_two_ticks_is_sampled(tmp_path):
    # 40 s of one picture, a different one from 15 s to 25 s; sampled every 30 s.
    source = tmp_path / "v.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=navy:s=320x240:d=15",
        "-f", "lavfi", "-i", "mandelbrot=s=320x240:rate=25",
        "-f", "lavfi", "-i", "color=c=navy:s=320x240:d=15",
        "-filter_complex", "[1:v]trim=duration=10,setpts=PTS-STARTPTS,format=yuv420p[m];"
                           "[0:v]format=yuv420p[a];[2:v]format=yuv420p[c];[a][m][c]concat=n=3[v]",
        "-map", "[v]", "-r", "25", str(source)], check=True)
    manifest = Manifest(source_video=str(source), duration_seconds=40.0)
    extract_frames.run(tmp_path, VTDConfig(frame_rate_seconds=30), manifest)
    times = [t for _f, t in extract_frames.frame_times(tmp_path, 30)]
    assert any(15 <= t < 25 for t in times), times   # the slide in the middle
    assert times[0] < 1


# ---- 4. Answers checked and asked for again ----------------------------------

SPANISH = ("La clase presenta los instrumentos de medición y explica que la validez es la "
           "capacidad de un cuestionario para medir lo que dice medir, mientras que la "
           "confiabilidad es la consistencia de las respuestas en el tiempo.")
ENGLISH = ("The lecture introduces measurement instruments and explains that validity is the "
           "ability of a questionnaire to measure what it claims to measure, while reliability "
           "is the consistency of the answers over time.")


def test_the_language_of_an_answer_is_recognised():
    assert written_in(SPANISH, "es") and not written_in(SPANISH, "en")
    assert written_in(ENGLISH, "en") and not written_in(ENGLISH, "es")
    assert written_in(ENGLISH + ' The lecturer said: "que mida lo que dice medir".', "en")
    assert written_in("Too short to judge.", "es")
    assert written_in(ENGLISH, "fr")  # a language vtd has no word list for is not judged


def test_code_tables_lists_and_quotes_do_not_decide_the_language():
    code = SPANISH + "\n\n```python\nfor x in items:\n    if x is not None or x in seen:\n        pass\n```"
    table = SPANISH + "\n\n| Concept | Description | Example of the use |\n|---|---|---|\n| the | of | and to in |"
    books = SPANISH + "\n\n" + "\n".join(f"- The Art of Thinking and the Way to {i}" for i in range(8))
    quotes = ENGLISH + ' She said "la validez es que mida lo que dice medir y no otra cosa" and '\
        '"la confiabilidad es la consistencia de las respuestas en el tiempo para el que mide".'
    assert all(written_in(text, "es") for text in (code, table, books))
    assert written_in(quotes, "en")


def test_a_title_is_one_short_line():
    assert _is_title("Validez y confiabilidad")
    assert not _is_title("") and not _is_title("Primera línea\nsegunda")
    assert not _is_title(" ".join(["palabra"] * 20))


def _provider(tmp_path, replies, language="es"):
    create, sent = replying(replies)
    return fake_anthropic(tmp_path, create, language=language), sent


CALL = dict(instruction="Resume.", transcript_chunk="Texto.", label="body §1")


def test_an_answer_in_the_wrong_language_is_asked_for_again(tmp_path, capsys):
    provider, sent = _provider(tmp_path, [ENGLISH, SPANISH])

    assert provider._call(**CALL) == SPANISH

    assert len(sent) == 2
    assert "could not be used: it was not written in Spanish." in sent[1]
    assert sent[1].startswith(sent[0])  # the same request, with the reason added
    assert provider.degraded == []
    assert "not written in Spanish; asking again" in capsys.readouterr().err


def test_the_good_answer_is_the_one_kept_for_later_runs(tmp_path):
    provider, sent = _provider(tmp_path, [ENGLISH, SPANISH])
    provider._call(**CALL)
    assert provider._call(**CALL) == SPANISH and len(sent) == 2  # from the cache


def test_an_answer_still_wrong_after_the_retries_is_used_but_flagged(tmp_path):
    provider, sent = _provider(tmp_path, [ENGLISH] * 3)

    assert provider._call(**CALL) == ENGLISH

    assert len(sent) == 3
    assert provider.degraded == ["body §1"]
    assert not list(cache.cache_dir(tmp_path).iterdir())  # so a later run asks again


def test_a_list_in_the_wrong_format_is_asked_for_again(tmp_path):
    provider, sent = _provider(tmp_path, ["No hay pendientes.", "- Leer el capítulo 2."])
    assert provider.extract_followups("resúmenes") == ["Leer el capítulo 2."]
    assert len(sent) == 2 and provider.degraded == []
