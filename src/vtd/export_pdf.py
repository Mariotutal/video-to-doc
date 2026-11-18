"""Turn the Markdown documents into PDFs with pandoc (`--pdf`)."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import manifest as mf
from .config import VTDConfig
from .utils import VTDError, emit, with_fence_state

# LaTeX engines pandoc is asked to use, in order of preference. Only xelatex has been
# tried; others that read UTF-8 (lualatex, tectonic) may work if added here.
ENGINES = ("xelatex",)
# What pandoc should not read into text written by a transcriber or a model: a `$`
# is money, a backslash is a backslash, a `---` line is not a metadata block, and
# nothing in it is LaTeX or HTML to be passed through (which could read local files).
_INPUT_FORMAT = (
    "markdown-tex_math_dollars-tex_math_single_backslash-yaml_metadata_block"
    "-raw_tex-raw_attribute-raw_html-latex_macros"
)
_HEADER = "\\usepackage{float}\n\\floatplacement{figure}{H}\n"  # keep each frame where it is cited
_FRAME_LINE = re.compile(r"^_[^_`]+:_\s+((?:`frames/[^`]+`(?:,\s*)?)+)\s*$")
_FRAME = re.compile(r"`(frames/[^`]+)`")
# The start of an image. `![^1]` is a footnote after a "!", unless an address follows.
_IMAGE = re.compile(r"!\[(?!\^\d+\](?!\())")
# Every ASCII punctuation mark can be backslash-escaped. Quotes are left alone: they
# are not markup, and unescaped they are set as proper opening and closing quotes.
_PUNCTUATION = re.compile(r"([!#-&(-/:-@\[-`{-~])")


def _as_text(block: list[str]) -> list[str]:
    """A plain fenced block (the transcript) as ordinary lines that keep their breaks.

    Left as code it is set verbatim, and a spoken line longer than the page is cut off.
    """
    escaped = (_PUNCTUATION.sub(r"\\\1", line) for line in block)
    return ["", *(f"| {line}" if line.strip() else "|" for line in escaped), ""]


def pdf_tools() -> tuple[str, str]:
    """Paths of pandoc and a LaTeX engine, or an error saying what to install."""
    pandoc = shutil.which("pandoc")
    engine = next((path for path in map(shutil.which, ENGINES) if path), None)
    if pandoc and engine:
        return pandoc, engine
    missing = "pandoc" if not pandoc else f"the LaTeX engine {' or '.join(ENGINES)}"
    raise VTDError(
        f"--pdf needs {missing}, which was not found on PATH. On macOS: `brew install pandoc` "
        "and `brew install --cask mactex-no-gui`; on Debian/Ubuntu: `sudo apt-get install "
        "pandoc texlive-xetex`."
    )


def _title(heading: str, source: str) -> tuple[str, str]:
    """(title, subtitle) from a document's first heading, `<what> — <kind of document>`."""
    names = [n for n in (source, Path(source).name) if n]
    known = next((n for n in names if heading == n or heading.startswith(f"{n} — ")), None)
    if known:  # the name itself may contain a dash
        name, kind = known, heading[len(known):].removeprefix(" — ")
    elif " — " in heading:
        name, _dash, kind = heading.rpartition(" — ")
    else:
        name, kind = heading, ""
    name = Path(name.strip()).name.removesuffix(".vtd")
    kind = kind.strip()
    return Path(name).stem or name, kind[:1].upper() + kind[1:]


def pdf_markdown(markdown: str, workspace: Path, source: str = "", *,
                 transcript: bool = False) -> tuple[str, str, str]:
    """(title, subtitle, body) of a vtd document, prepared for the page.

    The heading becomes the title, the lines that cite frames show the frames
    themselves, and the full path of the source, too wide for a page, is its file name.
    With `transcript` (the starter document), the plain fenced block holding the
    transcript becomes ordinary text. No other image is let through: the text comes
    from a recording and a model, and an image could point at any file or address.
    """
    lines = markdown.split("\n")
    title, subtitle = "", ""
    if lines and lines[0].startswith("# "):
        title, subtitle = _title(lines.pop(0)[2:], source)
    plain = [i for i, line in enumerate(lines) if line.strip() == "```"]
    # From the first plain fence to the last, so a ``` spoken of inside does not end it.
    first, last = (plain[0], plain[-1]) if transcript and len(plain) >= 2 else (-1, -1)
    frames_dir = (workspace / "frames").resolve()
    out: list[str] = []
    for i, (line, fenced) in enumerate(with_fence_state(lines)):
        if first <= i <= last:
            if i == last:
                out += _as_text(lines[first + 1:last])
            continue
        cited = None if fenced else _FRAME_LINE.match(line)
        if cited:
            for frame in _FRAME.findall(cited.group(1)):
                path = (workspace / frame).resolve()
                if path.is_file() and path.is_relative_to(frames_dir):
                    out += [f"![]({frame}){{ width=85% }}", ""]
        else:
            out.append(line if fenced else _IMAGE.sub(r"!\\[", line))
    body = "\n".join(out)
    if source:
        body = body.replace(f"`{source}`", f"`{Path(source).name}`")
    return title, subtitle, body


_NUMBERED_SECTION = re.compile(r"^#### \d+\. ", re.MULTILINE)


def _toc_depth(body: str) -> int:
    """How deep the index goes: down to the sections, which sit one level lower
    when the notes group them into chapters."""
    return 4 if _NUMBERED_SECTION.search(body) else 3


def _run_pandoc(command: list[str], body: str, workspace: Path) -> tuple[bool, str]:
    """Run pandoc in the workspace (frames are cited relative to it): (succeeded, messages)."""
    done = subprocess.run(command, input=body, cwd=workspace, capture_output=True,
                          encoding="utf-8", errors="replace")
    return done.returncode == 0, (done.stderr or "").strip()


def export_pdf(document: Path, config: VTDConfig, manifest: mf.Manifest) -> Path:
    """Write `<document>.pdf` next to a Markdown document and return its path."""
    pandoc, engine = pdf_tools()
    document = document.resolve()  # pandoc runs inside the workspace, so no relative paths
    workspace = document.parent
    title, subtitle, body = pdf_markdown(
        document.read_text(), workspace, manifest.source_video,
        transcript=document.name == config.starter_doc_name,
    )
    target = document.with_suffix(".pdf")
    partial = target.with_name(target.name + ".partial.pdf")
    with tempfile.TemporaryDirectory() as tmp:
        header = Path(tmp) / "header.tex"
        header.write_text(_HEADER)
        command = [
            pandoc, "-f", _INPUT_FORMAT, "-o", str(partial), f"--pdf-engine={engine}",
            "--toc", f"--toc-depth={_toc_depth(body)}", "-H", str(header),
            "-M", f"title={title}", "-M", f"subtitle={subtitle}",
            "-V", "geometry:margin=2.2cm", "-V", "fontsize=11pt",
            "-V", "colorlinks=true", "-V", "toccolor=black",
            "-V", "block-headings=true",  # level-4 headings on their own line, not run into the text
        ]
        # The language sets hyphenation and the name of the index, but needs that
        # language's LaTeX package; a small LaTeX install may not have it.
        ok, messages = _run_pandoc([*command, "-M", f"lang={config.output_language}"],
                                   body, workspace)
        if not ok:
            ok, messages = _run_pandoc(command, body, workspace)
    if not ok:
        partial.unlink(missing_ok=True)
        detail = messages[-600:] or "no error message"
        raise VTDError(f"Could not make {target.name}: pandoc failed: {detail}")
    partial.replace(target)
    missing = messages.count("Missing character")
    if missing:
        emit(f"pdf: {target.name} has {missing} kinds of character it could not show (the PDF "
             "font covers Latin scripts only; other scripts and emoji come out blank).",
             kind="warn")
    return target


def export_pdfs(workspace: Path, config: VTDConfig, manifest: mf.Manifest) -> list[Path]:
    """PDFs of the documents this workspace has: the starter one and, if made, the AI notes."""
    documents = []
    if manifest.has_stage("render-starter"):
        documents.append(workspace / config.starter_doc_name)
    if manifest.has_stage("synthesize"):
        documents.append(workspace / config.output_doc_name)
    documents = [d for d in documents if d.is_file()]
    if not documents:
        emit("--pdf: there is no document to convert yet.", kind="warn")
    written = []
    for document in documents:
        emit(f"pdf: converting {document.name}…")
        written.append(export_pdf(document, config, manifest))
        emit(f"pdf: {written[-1]}", kind="done")
    return written
