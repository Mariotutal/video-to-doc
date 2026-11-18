import json

from vtd.ai.base import ProviderLimitError
from vtd.ai_synth import _summary_digest, synthesize
from vtd.config import VTDConfig
from vtd.manifest import Manifest
from vtd.utils import VTDError


def test_summary_digest_shortens_every_section_instead_of_dropping_the_last():
    sections = [
        {"index": 0, "title": "Inicio", "body": "a" * 50},
        {"index": 1, "title": "Decision", "body": "b" * 50},
    ]

    digest = _summary_digest(sections, limit=80)

    assert "Section 1: Inicio" in digest
    assert "Section 2: Decision\nbbbb" in digest
    assert len(digest) <= 80


class GlossaryLimitedProvider:
    name = "fake"

    def name_section(self, section):
        return "Resumen"

    def write_section_body(self, section):
        return "Contenido de la seccion."

    def write_themes(self, sections_titles, transcript):
        return "### Tema\n\nDetalle."

    def extract_followups(self, transcript):
        return ["Revisar una decision pendiente."]

    def build_glossary(self, transcript):
        raise ProviderLimitError("OpenAI rate limit reached while generating glossary.")


def test_synthesize_writes_document_when_glossary_is_rate_limited(tmp_path, monkeypatch):
    workspace = tmp_path
    (workspace / "transcript.txt").write_text("Hola mundo.")
    (workspace / "transcript.vtt").write_text(
        "00:00:00.000 --> 00:00:10.000\nHola mundo.\n"
    )
    (workspace / "sections.json").write_text(json.dumps([
        {
            "index": 0,
            "start_s": 0,
            "end_s": 10,
            "duration_s": 10,
            "is_silent": False,
            "cue_count": 1,
            "title": None,
        }
    ]))
    (workspace / "speakers.json").write_text("[]")

    monkeypatch.setattr(
        "vtd.ai_synth._build_provider",
        lambda config, workspace: GlossaryLimitedProvider(),
    )
    cfg = VTDConfig(output_language="es")
    manifest = Manifest(
        source_video="video.mp4",
        duration_seconds=10,
        has_video=True,
        width=100,
        height=100,
    )

    synthesize(workspace, cfg, manifest)

    notes = (workspace / "notes.md").read_text()
    assert "# video.mp4 — notas" in notes
    assert "## Glosario" in notes
    assert "No se generaron entradas de glosario" in notes


def test_synthesize_fails_when_a_tail_call_breaks_for_another_reason(tmp_path, monkeypatch):
    class BrokenProvider(GlossaryLimitedProvider):
        def build_glossary(self, transcript):
            raise VTDError("OpenAI request failed while generating glossary: network down")

    (tmp_path / "transcript.txt").write_text("Hola mundo.")
    (tmp_path / "transcript.vtt").write_text("00:00:00.000 --> 00:00:10.000\nHola mundo.\n")
    (tmp_path / "sections.json").write_text(json.dumps([{
        "index": 0, "start_s": 0, "end_s": 10, "duration_s": 10,
        "is_silent": False, "cue_count": 1, "title": None,
    }]))
    monkeypatch.setattr("vtd.ai_synth._build_provider", lambda config, workspace: BrokenProvider())

    try:
        synthesize(tmp_path, VTDConfig(), Manifest(source_video="video.mp4", duration_seconds=10))
    except VTDError as e:
        assert "network down" in str(e)
    else:
        raise AssertionError("a non-limit provider failure must not be swallowed")
    assert not (tmp_path / "notes.md").exists()
