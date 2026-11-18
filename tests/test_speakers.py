"""--speakers: who speaks when, and how that reaches the documents."""
import hashlib
import io
import json
import tarfile
from argparse import Namespace

import pytest

from vtd import ai_synth, pipeline
from vtd import manifest as mf
from vtd.ai.base import SectionInput, _speaker_names
from vtd.config import VTDConfig, resolve_config
from vtd.stages import identify_speakers as ids
from vtd.stages import render
from vtd.utils import VTDError

from .fakes import fake_anthropic, replying


def _cue(start, end, text):
    return {"start_s": float(start), "end_s": float(end), "text": text}


CUES = [_cue(0, 10, "Bienvenidos."), _cue(10, 20, "Pablo Andrés."), _cue(20, 30, "Hola, soy Pablo."),
        _cue(30, 31, "Gracias."), _cue(31, 40, "Elena."), _cue(40, 50, "Soy maestra.")]
TURNS = [{"start_s": 0, "end_s": 19, "speaker": 1}, {"start_s": 20, "end_s": 30, "speaker": 2},
         {"start_s": 31, "end_s": 39, "speaker": 1}, {"start_s": 40, "end_s": 50, "speaker": 3}]


# ---- turns --------------------------------------------------------------------

def test_speakers_are_numbered_as_they_first_speak_and_brief_voices_dropped():
    raw = [(0, 40, 6), (40, 41, 9), (41, 80, 2), (80, 120, 6), (120, 121, 9)]
    assert ids.tidy(raw) == [
        {"start_s": 0, "end_s": 40, "speaker": 1},
        {"start_s": 41, "end_s": 80, "speaker": 2},
        {"start_s": 80, "end_s": 120, "speaker": 1},
    ]


def test_a_short_recording_keeps_the_person_who_spoke_for_ten_seconds():
    raw = [(0, 10, 0), (10, 20, 1), (20, 31, 0)]
    assert {t["speaker"] for t in ids.tidy(raw)} == {1, 2}


def test_with_a_given_number_of_speakers_every_voice_found_is_kept(tmp_path, monkeypatch):
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(ids, "require", lambda: None)
    monkeypatch.setattr(ids, "models", lambda config: (tmp_path / "s", tmp_path / "e"))
    monkeypatch.setattr(ids, "_diarize",
                        lambda audio, seg, emb, config: [(0, 3000, 1), (3000, 3005, 2)])
    ids.run(tmp_path, VTDConfig(speakers=True, num_speakers=2), mf.Manifest())
    assert {t["speaker"] for t in ids.speaker_turns(tmp_path)} == {1, 2}
    ids.run(tmp_path, VTDConfig(speakers=True), mf.Manifest())
    assert {t["speaker"] for t in ids.speaker_turns(tmp_path)} == {1}  # 5 s of 50 min: dropped


def test_each_line_gets_the_speaker_who_overlaps_it_longest():
    assert ids.cue_speakers(CUES, TURNS) == [1, 1, 2, None, 1, 3]
    split = [_cue(15, 26, "a caballo entre dos")]  # 4 s of speaker 1, 6 s of speaker 2
    assert ids.cue_speakers(split, TURNS) == [2]
    assert ids.cue_speakers(CUES, []) == [None] * 6


def test_the_transcript_marks_every_change_of_speaker():
    text = ids.labelled_text(CUES, ids.cue_speakers(CUES, TURNS), "Hablante", {3: "Elena"})
    assert text.split("\n") == [
        "Hablante 1: Bienvenidos.", "Pablo Andrés.", "Hablante 2: Hola, soy Pablo.",
        "Hablante ?: Gracias.", "Hablante 1: Elena.", "Elena: Soy maestra."]


def test_names_file_is_read_tolerantly(tmp_path):
    assert ids.speaker_names(tmp_path) == {}
    (tmp_path / ids.NAMES_NAME).write_text(json.dumps({"1": "Ricardo", "2": "", "3": " Elena "}))
    assert ids.speaker_names(tmp_path) == {1: "Ricardo", 3: "Elena"}
    (tmp_path / ids.NAMES_NAME).write_text("not json")
    assert ids.speaker_names(tmp_path) == {}


# ---- the stage ----------------------------------------------------------------

def test_the_stage_writes_the_turns_it_finds(tmp_path, monkeypatch, capsys):
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(ids, "require", lambda: None)
    monkeypatch.setattr(ids, "models", lambda config: (tmp_path / "s.onnx", tmp_path / "e.onnx"))
    monkeypatch.setattr(ids, "_diarize",
                        lambda audio, seg, emb, config: [(0, 40, 6), (40, 41, 9), (41, 80, 2)])
    ids.run(tmp_path, VTDConfig(speakers=True), mf.Manifest())
    assert [t["speaker"] for t in ids.speaker_turns(tmp_path)] == [1, 2]
    assert "2 speakers (1 voices heard too briefly" in capsys.readouterr().out


def test_missing_packages_are_named_before_any_stage_runs(monkeypatch):
    def missing():
        raise VTDError("--speakers needs the sherpa-onnx and numpy packages.")
    monkeypatch.setattr(ids, "require", missing)
    monkeypatch.setattr(pipeline, "require_binary", lambda name, pkg=None: None)
    with pytest.raises(VTDError, match="sherpa-onnx"):
        pipeline._preflight(["extract-audio", "identify-speakers"], VTDConfig(speakers=True))
    pipeline._preflight(["extract-audio"], VTDConfig())  # not asked for: not checked


def test_speakers_are_identified_only_when_asked_and_never_in_teams_mode(capsys):
    manifest = mf.Manifest()
    assert "identify-speakers" not in pipeline._stages_to_run(None, VTDConfig(), manifest)
    stages = pipeline._stages_to_run(None, VTDConfig(speakers=True), manifest)
    assert stages.index("extract-audio") < stages.index("identify-speakers") < stages.index(
        "render-starter")
    teams = VTDConfig(speakers=True, teams_captions=True)
    assert "identify-speakers" not in pipeline._stages_to_run(None, teams, manifest)
    assert "Teams captions already name the speakers" in capsys.readouterr().out


def test_documents_are_rebuilt_when_the_speakers_change():
    manifest = mf.Manifest()
    assert pipeline._dependents("identify-speakers", VTDConfig(), manifest) == [
        "render-starter", "synthesize"]
    assert pipeline._stage_options("identify-speakers", VTDConfig(num_speakers=5)) == {
        "num_speakers": 5, "speaker_threshold": 0.7}


def test_options_from_the_command_line_and_the_config_file(monkeypatch):
    from vtd import config as config_module
    from vtd.cli import build_parser
    parse = build_parser().parse_args
    assert resolve_config(Namespace()).speakers is False
    cfg = resolve_config(parse(["v.mp4", "--speakers", "--num-speakers", "6"]))
    assert cfg.speakers is True and cfg.num_speakers == 6
    monkeypatch.setattr(config_module, "_load_file_config",
                        lambda: {"defaults": {"speakers": True, "speaker_threshold": 0.9}})
    cfg = resolve_config(parse(["v.mp4", "--no-speakers"]))
    assert cfg.speakers is False and cfg.speaker_threshold == 0.9
    with pytest.raises(VTDError, match="speaker_threshold"):
        resolve_config(Namespace(speaker_threshold=0))


# ---- model download -----------------------------------------------------------

def _archive():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:bz2") as tar:
        info = tarfile.TarInfo(ids.SEGMENTATION_MODEL)
        info.size = 4
        tar.addfile(info, io.BytesIO(b"onnx"))
    return data.getvalue()


def test_models_are_downloaded_once_into_the_model_folder(tmp_path, monkeypatch):
    voice = b"voice model"
    served = {ids.EMBEDDING_URL: voice, ids.SEGMENTATION_URL: _archive()}
    asked = []
    monkeypatch.setattr(ids.urllib.request, "urlopen",
                        lambda url: asked.append(url) or io.BytesIO(served[url]))
    monkeypatch.setattr(ids, "EMBEDDING_SHA256", hashlib.sha256(voice).hexdigest())
    config = VTDConfig(model_dir=tmp_path / "models")

    segmentation, embedding = ids.models(config)

    assert segmentation.read_bytes() == b"onnx" and embedding.read_bytes() == voice
    assert segmentation.parent.parent == tmp_path / "models" / "speakers"
    assert not list((tmp_path / "models" / "speakers").glob("*.tar.bz2"))
    ids.models(config)
    assert len(asked) == 2  # nothing is fetched again


def test_a_download_that_is_not_the_expected_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ids.urllib.request, "urlopen", lambda url: io.BytesIO(b"something else"))
    with pytest.raises(VTDError, match="not the expected file"):
        ids.models(VTDConfig(model_dir=tmp_path))
    assert not (tmp_path / "speakers" / ids.EMBEDDING_MODEL).exists()


def test_a_cut_off_archive_leaves_no_half_model_behind(tmp_path, monkeypatch):
    voice = b"voice model"
    cut = _archive()[:-40]
    served = {ids.EMBEDDING_URL: voice, ids.SEGMENTATION_URL: cut}
    monkeypatch.setattr(ids.urllib.request, "urlopen", lambda url: io.BytesIO(served[url]))
    monkeypatch.setattr(ids, "EMBEDDING_SHA256", hashlib.sha256(voice).hexdigest())
    with pytest.raises(VTDError, match="Could not unpack"):
        ids.models(VTDConfig(model_dir=tmp_path))
    folder = tmp_path / "speakers"
    assert not (folder / ids.SEGMENTATION_MODEL).exists()
    assert sorted(p.name for p in folder.iterdir()) == [ids.EMBEDDING_MODEL]


def test_a_connection_cut_mid_download_is_reported_and_cleaned_up(tmp_path, monkeypatch):
    import http.client

    class Dropped(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(b"half")

    monkeypatch.setattr(ids.urllib.request, "urlopen", lambda url: Dropped())
    with pytest.raises(VTDError, match="Failed to download"):
        ids.models(VTDConfig(model_dir=tmp_path))
    assert list((tmp_path / "speakers").iterdir()) == []


def test_an_unreachable_model_folder_is_explained(tmp_path):
    (tmp_path / "not-a-disk").write_text("")
    with pytest.raises(VTDError, match="external disk"):
        ids.models(VTDConfig(model_dir=tmp_path / "not-a-disk" / "models"))


# ---- the AI stage -------------------------------------------------------------

def _provider(tmp_path, replies):
    create, sent = replying(replies, lambda kwargs: {
        "system": kwargs["system"], "text": kwargs["messages"][0]["content"][-1]["text"]})
    return fake_anthropic(tmp_path, create, language="es", content_type="lecture"), sent


def test_with_marked_speakers_the_model_is_told_to_follow_the_marks(tmp_path):
    provider, sent = _provider(tmp_path, ["Resumen.", "Título corto"])
    section = SectionInput(index=0, title=None, start_s=0, end_s=50, frame_paths=[],
                           transcript_chunk="Hablante 1: hola\nElena: soy maestra",
                           speaker_word="Hablante", plain_chunk="hola\nsoy maestra")
    provider.write_section_body(section)
    provider.name_section(section)

    assert "credit words only to the speaker marked on them" in sent[0]["system"]
    assert "a mark such as 'Hablante 3'" in sent[0]["text"]
    assert "Never put a name on a numbered speaker yourself" in sent[0]["text"]
    assert "Hablante 1: hola" in sent[0]["text"]
    assert "Hablante 1:" not in sent[1]["text"]  # the title is asked from the plain text


def test_without_speakers_the_model_is_told_not_to_guess(tmp_path):
    provider, sent = _provider(tmp_path, ["Resumen."])
    provider.write_section_body(SectionInput(index=0, title="T", start_s=0, end_s=5,
                                             transcript_chunk="hola", frame_paths=[]))
    assert "The transcript does not say who is speaking" in sent[0]["system"]
    assert "do not carry the first name over" in sent[0]["text"]


@pytest.mark.parametrize("answer, expected", [
    ("Hablante 1 :: Ricardo Soto\nHablante 2 :: UNKNOWN\nHablante 3 :: Elena", {1: "Ricardo Soto", 3: "Elena"}),
    ("- **Hablante 2** :: `Pablo Andrés`", {2: "Pablo Andrés"}),
    ("Hablante 1 :: Ana\nHablante 2 :: Ana", {}),            # the same name twice proves a guess
    ("Hablante 1 :: una persona que no dice su nombre en ningún momento", {}),
    ("No hay nombres.", {}),
    ("Hablante 4 :: Desconocido\nHablante 5 :: N/A\nHablante 6 :: UNKNOWN (not stated)\nHablante 7 :: ¿?", {}),
    ("Hablante 1 :: Ana\nHablante 2 :: ana.", {}),            # the same name, written differently
    ("Hablante 5 :: Pablo Andrés (docente)", {}),               # a name with a guess attached
])
def test_speaker_names_are_read_strictly(answer, expected):
    assert _speaker_names(answer) == expected


def test_a_name_the_model_puts_on_a_numbered_speaker_is_removed():
    from vtd.ai.base import _unnamed
    text = ("Ana Pérez (Hablante 3), una profesora de otra ciudad, y Hablante 4 (Rosa de los "
            "Santos) hablan; Hablante 5 (un médico) escucha.")
    assert _unnamed("Hablante", text) == (
        "Hablante 3, una profesora de otra ciudad, y Hablante 4 hablan; Hablante 5 (un médico) escucha.")
    assert _unnamed("", text) == text


def test_the_marks_do_not_take_room_from_the_excerpt(tmp_path):
    from vtd.ai.base import BODY_CHARS, _room
    plain = "\n".join(["palabras " * 4] * 190)                     # about 7,000 characters
    marked = "\n".join(f"Hablante {i % 3 + 1}: {'palabras ' * 4}" for i in range(190))
    assert len(marked) > BODY_CHARS > len(plain)
    section = SectionInput(index=0, title="T", start_s=0, end_s=9, frame_paths=[],
                           transcript_chunk=marked, speaker_word="Hablante", plain_chunk=plain)
    assert _room(section) >= len(marked)
    provider, sent = _provider(tmp_path, ["Resumen."])
    provider.write_section_body(section)
    assert marked in sent[0]["text"]  # nothing was cut


def test_follow_ups_are_asked_knowing_the_speakers_are_marked(tmp_path):
    provider, sent = _provider(tmp_path, ["NONE"])
    provider.section_followups(SectionInput(
        index=0, title="T", start_s=0, end_s=9, frame_paths=[],
        transcript_chunk="Hablante 2: envío el archivo", speaker_word="Hablante",
        plain_chunk="envío el archivo"))
    assert "credit words only to the speaker marked on them" in sent[0]["system"]


def test_first_words_show_the_line_before_each_speaker():
    sample = ai_synth._first_words(CUES, ids.cue_speakers(CUES, TURNS), "Hablante")
    assert "BEFORE Hablante 2: Pablo Andrés.\nHablante 2:\nHola, soy Pablo." in sample
    assert "BEFORE Hablante 3: Elena.\nHablante 3:\nSoy maestra." in sample
    assert "Gracias." not in sample  # nobody identified said it


class _Provider:
    name = "fake"
    degraded: list = []

    def __init__(self):
        self.excerpts, self.asked_names = [], 0

    def name_speakers(self, first_words, word):
        self.asked_names += 1
        return {2: "Pablo Andrés"}

    def name_section(self, section):
        return "Título"

    def write_section_body(self, section):
        self.excerpts.append(section.transcript_chunk)
        assert section.speaker_word == "Hablante" and "Hablante" not in section.plain_chunk
        return "Contenido."

    def write_themes(self, titles, digest):
        return "### Tema"

    def extract_followups(self, digest):
        return []

    def build_glossary(self, digest):
        return []


def _workspace(tmp_path):
    (tmp_path / "transcript.txt").write_text("x")
    (tmp_path / "transcript.vtt").write_text("WEBVTT\n\n" + "".join(
        f"00:00:{int(c['start_s']):02d}.000 --> 00:00:{int(c['end_s']):02d}.000\n{c['text']}\n\n"
        for c in CUES))
    (tmp_path / "sections.json").write_text(json.dumps([
        {"index": 0, "start_s": 0, "end_s": 50, "duration_s": 50, "is_silent": False,
         "cue_count": 6, "title": None}]))
    (tmp_path / ids.TURNS_NAME).write_text(json.dumps(TURNS))


def test_the_notes_are_written_from_a_transcript_that_says_who_speaks(tmp_path, monkeypatch):
    _workspace(tmp_path)
    provider = _Provider()
    monkeypatch.setattr(ai_synth, "_build_provider", lambda config, workspace: provider)
    config = VTDConfig(content_type="lecture", output_language="es")
    ai_synth.synthesize(tmp_path, config, mf.Manifest(source_video="/v/c.mp4", duration_seconds=50))

    assert provider.excerpts[0].split("\n")[:3] == [
        "Hablante 1: Bienvenidos.", "Pablo Andrés.", "Pablo Andrés: Hola, soy Pablo."]
    assert json.loads((tmp_path / ids.NAMES_NAME).read_text()) == {"1": "", "2": "Pablo Andrés", "3": ""}


def test_names_that_could_not_be_asked_are_asked_again_next_time(tmp_path, monkeypatch):
    _workspace(tmp_path)
    provider = _Provider()
    provider.name_speakers = lambda first_words, word: None  # no usable answer
    monkeypatch.setattr(ai_synth, "_build_provider", lambda config, workspace: provider)
    config = VTDConfig(content_type="lecture", output_language="es")
    ai_synth.synthesize(tmp_path, config, mf.Manifest(source_video="/v/c.mp4", duration_seconds=50))
    assert not (tmp_path / ids.NAMES_NAME).exists()
    assert provider.excerpts[0].startswith("Hablante 1: Bienvenidos.")


def test_names_corrected_by_hand_are_kept_and_not_asked_for_again(tmp_path, monkeypatch):
    _workspace(tmp_path)
    (tmp_path / ids.NAMES_NAME).write_text(json.dumps({"1": "Ricardo", "2": "", "3": "Elena"}))
    provider = _Provider()
    monkeypatch.setattr(ai_synth, "_build_provider", lambda config, workspace: provider)
    config = VTDConfig(content_type="lecture", output_language="es")
    ai_synth.synthesize(tmp_path, config, mf.Manifest(source_video="/v/c.mp4", duration_seconds=50))

    assert provider.asked_names == 0
    assert provider.excerpts[0].startswith("Ricardo: Bienvenidos.")
    assert "Elena: Soy maestra." in provider.excerpts[0]


def test_the_starter_document_marks_the_speakers_too(tmp_path):
    _workspace(tmp_path)
    assert render._transcript(tmp_path, VTDConfig(output_language="es")).startswith(
        "Hablante 1: Bienvenidos.")
    (tmp_path / ids.TURNS_NAME).unlink()
    assert render._transcript(tmp_path, VTDConfig()) == "x"
