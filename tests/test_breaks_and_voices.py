"""Breaks Whisper fills with a loop, a voice set apart in pieces, a name only mentioned."""
import json
import time

import pytest

from vtd import ai_synth, vtt
from vtd.config import VTDConfig
from vtd.manifest import Manifest
from vtd.stages import detect_sections, transcribe
from vtd.stages import identify_speakers as ids


def _cue(start, text, length=1.0):
    return {"start_s": float(start), "end_s": float(start) + length, "text": text}


# ---- a loop of a few phrases ---------------------------------------------------

def _loop(start, count):
    """Two phrases going round unevenly, with a third joining in now and then."""
    cues = []
    for i in range(count):
        text = "la casa" if i % 17 == 16 else ("que la casa", "se puede ver")[(i // 3) % 2]
        cues.append(_cue(start + i, text))
    return cues


def test_phrases_going_round_during_a_break_are_removed():
    cues = [_cue(0, "démonos 10 minutos"), _cue(1, "como les decía"), *_loop(2, 200),
            _cue(300, "sigamos"), _cue(301, "ya estamos de vuelta")]
    kept = [c["text"] for c in transcribe.drop_repeats(cues)]
    assert kept == ["démonos 10 minutos", "como les decía", "que la casa",
                    "sigamos", "ya estamos de vuelta"]


def test_what_is_said_only_once_inside_a_loop_is_kept():
    cues = [_cue(0, "bien"), *_loop(1, 30), _cue(31, "un momento"), *_loop(32, 30),
            _cue(70, "seguimos"), _cue(71, "con la clase")]
    kept = [c["text"] for c in transcribe.drop_repeats(cues)]
    assert "un momento" in kept and kept[-2:] == ["seguimos", "con la clase"]
    assert kept.count("que la casa") == 1 and "se puede ver" not in kept[2:]


def test_ordinary_speech_with_repeated_words_is_not_a_loop():
    words = ["entonces", "de paso", "de", "a la casa", "entonces", "el camino", "de paso",
             "sí", "sí", "claro", "entonces", "de", "la escala", "sí", "de paso", "bueno"]
    cues = [_cue(i, text) for i, text in enumerate(words * 4)]
    assert transcribe.drop_repeats(cues) == cues


def test_a_question_and_answer_said_a_few_times_is_not_a_loop():
    cues = [_cue(i, text) for i, text in enumerate(["¿sí?", "no", "¿sí?", "no", "¿sí?", "no",
                                                    "¿sí?", "no", "bueno", "sigamos"])]
    assert transcribe.drop_repeats(cues) == cues


@pytest.mark.parametrize("said", [
    ["¿Quién está a favor?"] + ["sí", "no"] * 12 + ["bien"],            # a show of hands
    [x for i in range(30) for x in ("vale", f"punto {i}")],             # a filler word
    ["ok", "bien"] * 12 + ["fin"],
    [x for i in range(40) for x in (f"línea {i}",) * 2],                # every line twice
])
def test_people_repeating_themselves_are_not_a_loop(said):
    cues = [_cue(i, text) for i, text in enumerate(said)]
    assert transcribe.drop_repeats(cues) == cues


def test_a_loop_followed_by_speech_that_echoes_itself_ends_where_the_loop_ends():
    after = [x for i in range(20) for x in (f"idea {i}", f"idea {i}")]
    cues = [_cue(i, text) for i, text in enumerate(
        ["bien", *(["que la casa", "se puede ver"] * 40), *after])]
    kept = [c["text"] for c in transcribe.drop_repeats(cues)]
    assert kept == ["bien", "que la casa", "se puede ver", *after]


@pytest.mark.parametrize("said", [
    lambda i: f"línea {i // 2}",                                         # every line twice
    lambda i: "Mm-hmm." if i % 2 else f"This is sentence {i} of the talk.",
    lambda i: ("Yes.", "No.", f"Reason {i}.")[i % 3],
])
def test_cleaning_a_transcript_that_echoes_itself_does_not_grow_with_its_square(said):
    def seconds(count):
        cues = [_cue(i, said(i)) for i in range(count)]
        runs = []
        for _ in range(3):
            started = time.perf_counter()
            assert len(transcribe.drop_repeats(cues)) == count
            runs.append(time.perf_counter() - started)
        return min(runs)

    # Four times the cues: about four times the work. Rereading the rest of the transcript
    # from every cue would make it sixteen, whatever the speed of the machine.
    assert seconds(20000) < 9 * max(seconds(5000), 0.001)


def test_a_loop_of_signs_is_removed_like_one_of_words():
    kept = transcribe.drop_repeats([_cue(i, "♪") for i in range(60)])
    assert [c["text"] for c in kept] == ["♪"]
    cues = [_cue(0, "Hello everyone."), *(_cue(1 + i, ("Thank you.", "♪")[i % 2]) for i in range(80)),
            _cue(90, "Back now."), _cue(91, "Let us go on.")]
    assert [c["text"] for c in transcribe.drop_repeats(cues)] == [
        "Hello everyone.", "Thank you.", "♪", "Back now.", "Let us go on."]


# ---- a break is a section of its own -------------------------------------------

def _sections(tmp_path, monkeypatch, cues, duration, silences=(), **options):
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n" + "".join(
        f"{vtt.stamp(c['start_s'])} --> {vtt.stamp(c['end_s'])}\n{c['text']}\n\n"
        for c in cues))
    monkeypatch.setattr(detect_sections, "_detect_silences", lambda *a, **k: list(silences))
    detect_sections.run(tmp_path, VTDConfig(**options), Manifest(duration_seconds=duration))
    return json.loads((tmp_path / "sections.json").read_text())


def test_a_long_stretch_with_no_transcript_is_a_silent_section(tmp_path, monkeypatch):
    talk = [_cue(i * 10, "hablamos del primer tema de hoy", 9) for i in range(60)]
    after = [_cue(1200 + i * 10, "seguimos con el segundo tema", 9) for i in range(60)]
    sections = _sections(tmp_path, monkeypatch, talk + after, 1800.0)
    silent = [s for s in sections if s["is_silent"]]
    assert [(s["start_s"], s["end_s"]) for s in silent] == [(599.0, 1200.0)]
    assert sum(s["cue_count"] for s in sections) == 120  # nothing lost around the break
    assert all(s["duration_s"] >= detect_sections.MIN_SECTION_S for s in sections)


def test_a_boundary_inside_or_beside_a_break_gives_way_to_it():
    assert detect_sections._with_breaks([0.0, 598.0, 900.0, 1201.0, 1800.0], [(599.0, 1200.0)]) \
        == [0.0, 599.0, 1200.0, 1800.0]


def test_two_breaks_with_a_word_between_them_are_two_silent_sections():
    cues = [_cue(0, "a", 99), _cue(300, "ya vuelvo", 3), _cue(500, "b", 100), _cue(600, "c", 400)]
    breaks = detect_sections._breaks(cues)
    assert breaks == [(99.0, 300.0), (303.0, 500.0)]
    assert detect_sections._with_breaks([0.0, 1000.0], breaks) == [0.0, 99.0, 300.0, 500.0, 1000.0]


def test_a_pause_shorter_than_a_break_changes_nothing():
    cues = [_cue(0, "uno", 9), _cue(100, "dos", 9)]
    assert detect_sections._breaks(cues) == []
    assert detect_sections._with_breaks([0.0, 50.0, 110.0], []) == [0.0, 50.0, 110.0]


def test_a_recording_that_starts_or_ends_with_a_break_keeps_its_ends():
    assert detect_sections._with_breaks([0.0, 400.0], [(2.0, 300.0), (399.0, 400.0)]) \
        == [0.0, 300.0, 400.0]


# ---- a voice set apart in pieces -----------------------------------------------

def _lecture_with_pieces():
    """One long voice (0), cut by short pieces the grouping set apart (7), and a guest (3)."""
    raw, at = [], 0.0
    for _ in range(40):
        raw.append((at, at + 60, 0))
        at += 60
        raw.append((at, at + 1.5, 7))  # a minute in all, never more than 1.5 s at a time
        at += 1.5
    raw.append((at + 30, at + 31.5, 7))  # a piece on its own, far from anyone
    raw.append((at + 60, at + 90, 3))
    return raw


def test_pieces_of_the_main_voice_go_back_to_it():
    turns = ids.tidy(_lecture_with_pieces())
    assert {t["speaker"] for t in turns} == {1, 2}
    heard = sum(t["end_s"] - t["start_s"] for t in turns if t["speaker"] == 1)
    assert heard == 40 * 60 + 39 * 1.5  # the last two pieces are not between its turns
    assert len(turns) == 40 + 39 + 1


def test_a_brief_voice_between_two_people_stays_unnamed():
    raw = [(0, 40, 6), (40, 41, 9), (41, 80, 2), (80, 120, 6)]
    assert [t["speaker"] for t in ids.tidy(raw)] == [1, 2, 1]


def test_someone_who_only_gives_short_answers_in_a_meeting_is_still_a_speaker():
    raw, at = [], 0.0
    for _ in range(10):
        raw += [(at, at + 40, 0), (at + 40.5, at + 43.5, 1)]
        at += 44
    raw.append((at, at + 40, 0))
    turns = ids.tidy(raw)
    assert [t["speaker"] for t in turns] == [1, 2] * 10 + [1]


@pytest.mark.parametrize("turns_of_main, brief_turns, brief_s", [
    (36, 6, 5.0),    # a moderator's questions in a webinar
    (15, 15, 4.0),   # a student's answers in a lecture
])
def test_brief_questions_or_answers_beside_a_long_talk_are_still_a_speaker(
        turns_of_main, brief_turns, brief_s):
    raw, at = [], 0.0
    for i in range(turns_of_main):
        raw.append((at, at + 100, 0))
        at += 100
        if i % (turns_of_main // brief_turns) == 0:
            raw.append((at, at + brief_s, 1))
            at += brief_s
    assert {t["speaker"] for t in ids.tidy(raw)} == {1, 2}


def test_someone_who_speaks_once_for_a_while_is_kept():
    raw = [(0, 3000, 0), (3000, 3024, 5), (3024, 3600, 0)]
    assert [t["speaker"] for t in ids.tidy(raw)] == [1, 2, 1]


def test_with_a_given_number_of_speakers_no_voice_is_merged_away():
    turns = ids.tidy(_lecture_with_pieces(), 0.0)
    assert {t["speaker"] for t in turns} == {1, 2, 3}


# ---- a name only mentioned -----------------------------------------------------

def test_a_name_the_speaker_only_mentions_is_not_theirs():
    assert ai_synth._names_someone_else(
        "Margaret Holloway", "fue al inicio", "hay una autora que se llama Margaret Holloway")
    assert ai_synth._names_someone_else("Nina Roberts", "", "es mamá de Nina Roberts, corredora")
    assert ai_synth._names_someone_else("Ana", "hasta mañana a todos", "le pregunté a Ana ayer")


def test_a_name_the_speaker_gives_as_their_own_or_is_called_by_is_kept():
    assert not ai_synth._names_someone_else("Pablo Andrés", "Bienvenidos.", "Hola, soy Pablo.")
    assert not ai_synth._names_someone_else("Elena", "Elena, por favor.", "Gracias, buenas noches.")
    assert not ai_synth._names_someone_else("Ana Pérez", "", "Buenas, mi nombre es Ana Pérez.")
    assert not ai_synth._names_someone_else("Ana", "", "Hi, this is Ana from support.")
    assert not ai_synth._names_someone_else("Lucía Ramírez", "", "Yo creo que sí lo vimos.")
    assert not ai_synth._names_someone_else("Ana", "", "Hola, Ana aquí. Gracias.")
    assert not ai_synth._names_someone_else("Ana", "", "Me presento, Ana.")
    assert not ai_synth._names_someone_else("Ana", "", "Hola, aquí Ana")
    assert not ai_synth._names_someone_else("Ana", "", "Hola a todos, mi nombre completo es Ana")
    assert not ai_synth._names_someone_else("Ana", "Ana, ¿nos cuentas? Adelante. Sí.", "Claro.")


def test_names_only_mentioned_are_left_out_of_the_names_file(tmp_path):
    cues = [_cue(0, "Bienvenidos a la clase.", 9), _cue(10, "Elena, por favor.", 9),
            _cue(20, "Gracias, soy maestra.", 9),
            _cue(30, "Hay una autora que se llama Margaret Holloway.", 9)]
    turns = [{"start_s": 0, "end_s": 19, "speaker": 1}, {"start_s": 20, "end_s": 29, "speaker": 2},
             {"start_s": 30, "end_s": 39, "speaker": 3}]
    (tmp_path / ids.TURNS_NAME).write_text(json.dumps(turns))

    class Provider:
        def name_speakers(self, first_words, word):
            return {2: "Elena", 3: "Margaret Holloway"}

    _speakers, names = ai_synth._speakers(tmp_path, Provider(), cues, "Hablante")
    assert names == {2: "Elena"}
    assert json.loads((tmp_path / ids.NAMES_NAME).read_text()) == {"1": "", "2": "Elena", "3": ""}
