"""--pdf: the documents as PDF, through pandoc."""
import shutil
from argparse import Namespace

import pytest

from vtd import ai_synth, export_pdf, pipeline
from vtd import manifest as mf
from vtd.config import VTDConfig, resolve_config
from vtd.export_pdf import pdf_markdown
from vtd.utils import VTDError

NOTES = """# /videos/Clase 1.mp4 — notas de la clase

| Campo | Valor |
|---|---|
| Fuente | `/videos/Clase 1.mp4` |

### 1. Inicio

Texto con $5 y un \\comando.

_Diapositivas / elementos visuales referenciados:_ `frames/frame_0001.jpg`, `frames/frame_0002.jpg`
"""


def _workspace(tmp_path, *, notes=True):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "frame_0001.jpg").write_bytes(b"jpeg")
    (tmp_path / "notes.starter.md").write_text("# Clase 1.mp4.vtd — starter notes\n\nhola\n")
    manifest = mf.Manifest(source_video="/videos/Clase 1.mp4")
    manifest.mark_stage("render-starter")
    if notes:
        (tmp_path / "notes.md").write_text(NOTES)
        manifest.mark_stage("synthesize")
    return manifest


def test_heading_becomes_the_title_and_the_source_path_its_file_name(tmp_path):
    title, subtitle, body = pdf_markdown(NOTES, tmp_path, "/videos/Clase 1.mp4")
    assert (title, subtitle) == ("Clase 1", "Notas de la clase")
    assert "# /videos" not in body
    assert "| Fuente | `Clase 1.mp4` |" in body


def test_starter_title_drops_the_workspace_suffix(tmp_path):
    title, subtitle, _body = pdf_markdown("# Clase 1.mp4.vtd — starter notes\n", tmp_path)
    assert (title, subtitle) == ("Clase 1", "Starter notes")


def test_cited_frames_are_shown_as_images_when_the_file_exists(tmp_path):
    _workspace(tmp_path)
    _title, _subtitle, body = pdf_markdown(NOTES, tmp_path)
    assert "![](frames/frame_0001.jpg){ width=85% }" in body
    assert "frame_0002" not in body          # cited, but not on disk
    assert "elementos visuales" not in body  # the line that listed the file names


def test_the_transcript_block_becomes_text_that_wraps_with_nothing_read_as_markup(tmp_path):
    starter = "## Transcript\n\n```\nCuesta $5, a_b y *esto* [no] es #markup\n\nsegunda\n```\n"
    _t, _s, body = pdf_markdown(starter, tmp_path, transcript=True)
    assert "```" not in body
    assert "| Cuesta \\$5\\, a\\_b y \\*esto\\* \\[no\\] es \\#markup" in body
    assert "| segunda" in body


def test_a_fence_spoken_of_inside_the_transcript_does_not_end_it(tmp_path):
    starter = "```\nprimera\n```\nsigue siendo transcripción\n```\n"
    _t, _s, body = pdf_markdown(starter, tmp_path, transcript=True)
    assert "| sigue siendo transcripción" in body and "| \\`\\`\\`" in body


def test_quotes_in_the_transcript_are_left_for_proper_typesetting(tmp_path):
    _t, _s, body = pdf_markdown('```\ndijo "hola" y \'adiós\'\n```\n', tmp_path, transcript=True)
    assert '| dijo "hola" y \'adiós\'' in body


def test_plain_code_blocks_in_ai_notes_stay_code(tmp_path):
    notes = "```\n+--+\n|  |\n+--+\n```\n"
    assert pdf_markdown(notes, tmp_path)[2] == notes


def test_images_other_than_the_cited_frames_are_not_let_through(tmp_path):
    _workspace(tmp_path)
    (tmp_path / "secret.jpg").write_bytes(b"jpeg")
    notes = ("![x](/etc/hosts) y ![y](http://127.0.0.1/x.png)\n\n"
             "_Frames:_ `frames/../secret.jpg`, `frames/frame_0001.jpg`\n")
    _t, _s, body = pdf_markdown(notes, tmp_path)
    assert "!\\[x](/etc/hosts)" in body and "!\\[y](http" in body  # shown as text
    assert body.count("![](") == 1 and "![](frames/frame_0001.jpg)" in body


@pytest.mark.parametrize("heading, source, expected", [
    ("/v/Part 1 — intro.mp4 — notes", "/v/Part 1 — intro.mp4", ("Part 1 — intro", "Notes")),
    ("Part 1 — intro.mp4.vtd — starter notes", "", ("Part 1 — intro", "Starter notes")),
    ("just a name", "", ("just a name", "")),
])
def test_title_survives_a_dash_in_the_file_name(heading, source, expected):
    assert export_pdf._title(heading, source) == expected


def test_code_in_a_named_language_and_an_unclosed_block_are_left_alone(tmp_path):
    _t, _s, body = pdf_markdown("```python\nprint(1)\n```\n\n```\nnever closed\n", tmp_path)
    assert "```python\nprint(1)\n```" in body
    assert "```\nnever closed" in body


def test_missing_tools_are_named_with_what_to_install(monkeypatch):
    monkeypatch.setattr(export_pdf.shutil, "which", lambda name: None)
    with pytest.raises(VTDError, match="needs pandoc.*brew install pandoc"):
        export_pdf.pdf_tools()
    monkeypatch.setattr(export_pdf.shutil, "which",
                        lambda name: "/bin/pandoc" if name == "pandoc" else None)
    with pytest.raises(VTDError, match="xelatex"):
        export_pdf.pdf_tools()


def test_missing_pdf_tools_stop_the_run_before_any_stage(monkeypatch):
    monkeypatch.setattr(export_pdf.shutil, "which", lambda name: None)
    with pytest.raises(VTDError, match="--pdf needs pandoc"):
        pipeline._preflight([], VTDConfig(pdf=True))
    pipeline._preflight([], VTDConfig())  # not asked for: not checked


@pytest.fixture
def fake_pandoc(monkeypatch):
    """Record pandoc runs and write the file pandoc would; `fail` makes runs fail."""
    monkeypatch.setattr(export_pdf, "pdf_tools", lambda: ("pandoc", "xelatex"))
    runs = []

    def run(command, body, workspace):
        runs.append({"command": command, "body": body, "workspace": workspace})
        if run.fail(command):
            return False, "! LaTeX Error: something"
        target = command[command.index("-o") + 1]
        assert target.startswith("/")  # pandoc runs in the workspace: no relative paths
        open(target, "wb").write(b"%PDF-fake")
        return True, run.messages
    run.fail = lambda command: False
    run.messages = ""
    monkeypatch.setattr(export_pdf, "_run_pandoc", run)
    return runs, run


def test_both_documents_are_exported_next_to_their_markdown(tmp_path, fake_pandoc, monkeypatch):
    runs, _run = fake_pandoc
    manifest = _workspace(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    relative = tmp_path.relative_to(tmp_path.parent)  # as when vtd is run next to the video

    written = export_pdf.export_pdfs(relative, VTDConfig(output_language="es"), manifest)

    assert [p.name for p in written] == ["notes.starter.pdf", "notes.pdf"]
    assert (tmp_path / "notes.pdf").read_bytes() == b"%PDF-fake"
    assert not list(tmp_path.glob("*.partial.pdf"))
    assert runs[1]["workspace"] == tmp_path.resolve()
    assert "lang=es" in runs[1]["command"] and "title=Clase 1" in runs[1]["command"]


def test_notes_are_not_exported_when_the_ai_stage_did_not_complete(tmp_path, fake_pandoc):
    manifest = _workspace(tmp_path)
    manifest.reset_stage("synthesize")  # a stale notes.md is still on disk
    written = export_pdf.export_pdfs(tmp_path, VTDConfig(), manifest)
    assert [p.name for p in written] == ["notes.starter.pdf"]


def test_a_latex_without_the_language_package_still_gets_a_pdf(tmp_path, fake_pandoc):
    runs, run = fake_pandoc
    run.fail = lambda command: "lang=es" in command
    manifest = _workspace(tmp_path, notes=False)
    export_pdf.export_pdfs(tmp_path, VTDConfig(output_language="es"), manifest)
    assert len(runs) == 2 and "lang=es" not in runs[1]["command"]
    assert (tmp_path / "notes.starter.pdf").exists()


def test_a_failed_conversion_is_reported_and_leaves_no_half_written_pdf(tmp_path, fake_pandoc):
    _runs, run = fake_pandoc
    run.fail = lambda command: True
    manifest = _workspace(tmp_path, notes=False)
    (tmp_path / "notes.starter.pdf").write_bytes(b"%PDF-earlier")
    with pytest.raises(VTDError, match="notes.starter.pdf: pandoc failed.*LaTeX Error"):
        export_pdf.export_pdfs(tmp_path, VTDConfig(), manifest)
    assert (tmp_path / "notes.starter.pdf").read_bytes() == b"%PDF-earlier"
    assert not list(tmp_path.glob("*.partial.pdf"))


def test_characters_the_font_cannot_show_are_reported(tmp_path, fake_pandoc, capsys):
    _runs, run = fake_pandoc
    run.messages = ("[WARNING] Missing character: There is no П in font x\n"
                    "[WARNING] Missing character: There is no 你 in font x")
    export_pdf.export_pdfs(tmp_path, VTDConfig(), _workspace(tmp_path, notes=False))
    assert "2 kinds of character it could not show" in capsys.readouterr().err


def test_pdf_can_be_asked_for_on_the_command_line_or_in_the_config_file(monkeypatch):
    from vtd import config as config_module
    from vtd.cli import build_parser
    parse = build_parser().parse_args
    assert resolve_config(Namespace()).pdf is False
    assert resolve_config(parse(["v.mp4", "--pdf"])).pdf is True
    monkeypatch.setattr(config_module, "_load_file_config", lambda: {"defaults": {"pdf": True}})
    assert resolve_config(Namespace()).pdf is True
    assert resolve_config(parse(["v.mp4", "--no-pdf"])).pdf is False  # for one run
    monkeypatch.setattr(config_module, "_load_file_config", lambda: {"defaults": {"pdf": "no"}})
    with pytest.raises(VTDError, match="pdf"):
        resolve_config(parse(["v.mp4", "--pdf"]))  # the file is checked even when overridden


def test_headings_a_model_writes_in_a_section_stay_inside_the_section():
    body = "## Resumen\ntexto\n# Otro\n#### ya dentro\n```\n# un comentario\n```\n#sin espacio"
    assert ai_synth._fit_headings(body) == (
        "#### Resumen\ntexto\n#### Otro\n#### ya dentro\n```\n# un comentario\n```\n#sin espacio")


def test_underlined_titles_in_a_section_are_kept_inside_it_too():
    body = "Título\n======\ntexto\n\nOtro\n---\n\n| a | b |\n|---|---|\n\n- punto\n---\n"
    assert ai_synth._fit_headings(body) == (
        "#### Título\ntexto\n\n#### Otro\n\n| a | b |\n|---|---|\n\n- punto\n---\n")


def test_code_inside_a_longer_or_different_fence_is_not_touched():
    body = "~~~\n```\n# comentario\n```\n# otro\n~~~\n# fuera"
    assert ai_synth._fit_headings(body) == "~~~\n```\n# comentario\n```\n# otro\n~~~\n#### fuera"


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="needs pandoc")
def test_latex_written_into_a_document_is_shown_as_text_not_run(tmp_path):
    import subprocess
    document = ("`\\input{/etc/hosts}`{=latex}\n\n```{=latex}\n\\input{/etc/hosts}\n```\n\n"
                "\\input{/etc/hosts}\n\n<b>x</b> $1+1$\n")
    _t, _s, body = pdf_markdown(document, tmp_path)
    latex = subprocess.run(["pandoc", "-f", export_pdf._INPUT_FORMAT, "-t", "latex"],
                           input=body, capture_output=True, text=True, check=True).stdout
    # Every \input must come out escaped or inside verbatim text, never as a command.
    outside_verbatim = "\n".join(
        line for line in latex.split("\n") if "\\input" in line and "\\verb" not in line
        and "textbackslash" not in line and not line.startswith("\\input{/etc"))
    assert "\\textbackslash input" in latex.replace("{}", "")
    assert outside_verbatim == ""
    assert "\\(" not in latex and "<b>" not in latex.replace("\\textless b\\textgreater", "")


@pytest.mark.skipif(shutil.which("pandoc") is None or shutil.which("xelatex") is None,
                    reason="needs pandoc and xelatex")
def test_a_real_pdf_is_made_with_pandoc(tmp_path):
    manifest = _workspace(tmp_path)
    (tmp_path / "frames" / "frame_0001.jpg").unlink()  # not a real image
    written = export_pdf.export_pdfs(tmp_path, VTDConfig(output_language="es"), manifest)
    assert [p.name for p in written] == ["notes.starter.pdf", "notes.pdf"]
    assert all(p.read_bytes().startswith(b"%PDF") for p in written)


def test_the_index_reaches_the_sections_when_they_sit_under_chapters():
    assert export_pdf._toc_depth("### Capítulo\n\n#### 1. Sección\n\n##### detalle\n") == 4
    assert export_pdf._toc_depth("### 1. Sección\n\n#### detalle del modelo\n") == 3
