"""AI synthesis: weave transcript + speakers + sections + frames into notes.md."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .ai.base import LLMProvider, ProviderLimitError, SectionInput
from .config import VTDConfig
from .manifest import Manifest
from .references import bibliography
from .stages.extract_frames import frame_times
from .stages.identify_speakers import (
    NAMES_NAME,
    cue_speakers,
    labelled_text,
    speaker_names,
    speaker_turns,
)
from .stages.labels import LABELS
from .stages.ocr_speakers import frame_text
from .stages.render import hms, render_template
from .utils import VTDError, emit, read_json, with_fence_state
from .vtt import cue_text, cues_in_section, parse_vtt


def _transcript_for_section(cues: list[dict], start_s: float, end_s: float,
                            *, is_last: bool = False) -> str:
    """Stitch the VTT cues belonging to [start_s, end_s) into one chunk."""
    return cue_text(cues_in_section(cues, start_s, end_s, is_last=is_last))


# Fewest sections for which chapters are asked for.
MIN_SECTIONS_FOR_CHAPTERS = 6


def _frames_for_section(workspace: Path, config: VTDConfig, start_s: float, end_s: float,
                        limit: int = 3) -> list[Path]:
    rate = max(1, int(config.frame_rate_seconds))
    picked = [f for f, sec in frame_times(workspace, rate) if start_s <= sec <= end_s]
    words = frame_text(workspace)
    least = config.min_frame_words
    if words and least > 0:
        # A frame with next to nothing to read is a face or an empty screen, not a slide.
        picked = [f for f in picked if words.get(f.name, least) >= least]
    if len(picked) <= limit:
        return picked
    # Evenly spread across the section, always keeping the first and last frame.
    last = len(picked) - 1
    return [picked[round(i * last / (limit - 1))] for i in range(limit)]


# Longest digest of section summaries, in characters, sent for themes, follow-ups and glossary.
DIGEST_CHARS = 24000
_DIGEST_SEPARATOR = "\n\n---\n\n"


def _fair_share(lengths: list[int], budget: int) -> int:
    """Longest length every item may keep so that the total fits in `budget`.

    Short items keep everything, and what they do not use goes to the longer ones.
    """
    remaining, left = budget, len(lengths)
    for length in sorted(lengths):
        if length * left > remaining:
            return max(0, remaining // left)
        remaining -= length
        left -= 1
    return max(lengths, default=0)


def _number(section: dict) -> int:
    """The number a section is shown with (its position among the ones with speech)."""
    return section.get("number") or section.get("index", 0) + 1


def _summary_digest(sections: list[dict], limit: int = DIGEST_CHARS) -> str:
    """Compact synthesis context for whole-document follow-up calls.

    When the summaries are too long together, each is shortened to a fair share, so the
    themes, follow-ups and glossary are drawn from the whole recording and not only
    from its first sections.
    """
    heads = [
        f"Section {_number(s)}: {s.get('title') or 'Section ' + str(_number(s))}\n"
        for s in sections
    ]
    bodies = [s.get("body") or "" for s in sections]

    def join(texts: list[str]) -> str:
        return _DIGEST_SEPARATOR.join(h + t for h, t in zip(heads, texts, strict=True))

    digest = join(bodies)
    if len(digest) <= limit:
        return digest
    fixed = sum(len(h) for h in heads) + len(_DIGEST_SEPARATOR) * (len(sections) - 1)
    share = _fair_share([len(b) for b in bodies], limit - fixed)
    shortened = join([b if len(b) <= share else b[:max(0, share - 1)].rstrip() + "…"
                      for b in bodies])
    if len(shortened) > limit:  # so many sections that their headings alone do not fit
        emit(f"  · section summaries cut from {len(digest)} to {limit} characters for the "
             "themes, follow-ups and glossary", kind="warn")
        return shortened[:limit]
    emit(f"  · section summaries shortened from {len(digest)} to {len(shortened)} characters "
         f"(up to {share} per section) for the themes, follow-ups and glossary", kind="warn")
    return shortened


# Headings above `level`, for the two levels a section body can sit under.
_HEADING_ABOVE = {level: re.compile(rf"^ {{0,3}}#{{1,{level - 1}}}(\s+\S)") for level in (4, 5)}
_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)\s*$")


def _fit_headings(body: str, level: int = 4) -> str:
    """Push headings a model put in a section body below the section's own heading.

    A body is placed under a level-3 heading; a `##` inside it, or a title underlined
    with `===` or `---`, would otherwise end the section in the outline of the document.
    """
    out: list[str] = []
    for line, fenced in with_fence_state(body.split("\n")):
        previous = out[-1] if out else ""
        if fenced:
            out.append(line)
        elif (_UNDERLINE.match(line) and previous.strip()
                and not previous.lstrip().startswith(("#", "|", "-", "*", "+", ">", "`", "~"))):
            out[-1] = f"{'#' * level} {previous.strip()}"  # an underlined title
        else:
            out.append(_HEADING_ABOVE[level].sub(rf"{'#' * level}\1", line))
    return "\n".join(out)


def _chapters(provider: LLMProvider, spoken: list[dict], digest: str) -> list[dict]:
    """The sections that have speech, grouped by subject, for the outline of the document.

    Each chapter is {title, start_s, end_s, sections}. When the provider cannot say
    where the subject changes, there is one untitled chapter holding every section.
    Also sets each section's heading level and fits its body under it.
    """
    grouped: list[tuple[int, int, str]] = []
    group = getattr(provider, "group_sections", None)
    if group is not None and len(spoken) >= MIN_SECTIONS_FOR_CHAPTERS:
        emit("  chapters…")
        try:
            grouped = group([r["title"] for r in spoken], digest)
        except ProviderLimitError as e:
            emit(str(e), kind="warn")
    level = 4 if grouped else 3
    for record in spoken:
        record["heading"] = "#" * level
        record["body"] = _fit_headings(record["body"], level + 1)
    if not grouped:
        grouped = [(1, len(spoken), "")]
    return [
        {"title": title, "sections": spoken[first - 1:last],
         "start_s": spoken[first - 1]["start_s"] if spoken else 0.0,
         "end_s": spoken[last - 1]["end_s"] if spoken else 0.0}
        for first, last, title in grouped
    ]


def _followups(provider: LLMProvider, noted: list[str] | None, digest: str) -> list[str]:
    """The follow-ups of the whole document.

    `noted` are the ones read from each section's transcript, when the provider can do
    that; a summary of a section often leaves out a reading or a task said in passing.
    """
    merge = getattr(provider, "merge_followups", None)
    if noted is None or merge is None:
        return provider.extract_followups(digest)
    return merge(noted)


# Longest sample of first words, in characters, sent to ask for the speakers' names.
NAMING_CHARS = 12000


CALLED_LINES = 3  # lines before a speaker's first words in which they may be called on


def _samples(cues: list[dict], speakers: list[int | None],
             per_speaker: int = 500) -> dict[int, tuple[str, str, str]]:
    """For each speaker: the line said just before they first spoke, their first words,
    and the few lines before those, in which someone may have called on them."""
    before: dict[int, str] = {}
    called: dict[int, str] = {}
    words: dict[int, list[str]] = {}
    used: dict[int, int] = {}
    for position, (cue, speaker) in enumerate(zip(cues, speakers, strict=True)):
        text = cue["text"].strip()
        if speaker is None or not text or used.get(speaker, 0) >= per_speaker:
            continue
        if speaker not in before:
            before[speaker] = cues[position - 1]["text"].strip() if position else ""
            called[speaker] = " ".join(
                c["text"].strip() for c in cues[max(0, position - CALLED_LINES):position])
            words[speaker] = []
        words[speaker].append(text)
        used[speaker] = used.get(speaker, 0) + len(text)
    return {speaker: (before[speaker], "\n".join(words[speaker]), called[speaker])
            for speaker in before}


def _first_words(cues: list[dict], speakers: list[int | None], word: str,
                 per_speaker: int = 500) -> str:
    """For each speaker, the line said just before they first spoke and their first words."""
    return "\n\n".join(
        f"BEFORE {word} {speaker}: {before}\n{word} {speaker}:\n{words}"
        for speaker, (before, words, _called) in _samples(cues, speakers, per_speaker).items()
    )[:NAMING_CHARS]


# How people say who they are: just before their name, or just after it.
_INTRODUCES = re.compile(
    r"\b(?:soy|me llamo|me presento|(?:mi )?nombre (?:completo )?es|(?:les |le )?(?:habla|saluda)|aquí|"
    r"i am|i'm|my name is|my name's|this is|it's|here is)\W+(?:\w+\W+){0,3}$", re.IGNORECASE)
_INTRODUCED = re.compile(
    r"^(?:\W*\w+){0,3}\W+(?:aquí|al habla|por aquí|here|speaking)\b", re.IGNORECASE)


def _names_someone_else(name: str, called: str, words: str) -> bool:
    """Whether a speaker only mentions `name`: they say it without saying it is theirs,
    and nobody called them by it in the lines before they spoke.

    Strict on purpose: a speaker left as a number can be named by hand, while a wrong
    name is repeated all through the notes.
    """
    first = re.compile(rf"\b{re.escape(name.split()[0])}\b", re.IGNORECASE)
    if first.search(called):
        return False
    said = [m.start() for m in first.finditer(words)]
    return bool(said) and not any(
        _INTRODUCES.search(words[:at]) or _INTRODUCED.match(words[at:]) for at in said)


def _speakers(workspace: Path, provider: LLMProvider, cues: list[dict],
              word: str) -> tuple[dict[int, int | None], dict[int, str]]:
    """({id of a cue: its speaker}, {speaker: name}) when speakers were identified.

    Names come from speaker_names.json, which is written here the first time (from what
    the speakers say of themselves) and can then be corrected by hand.
    """
    turns = speaker_turns(workspace)
    if not turns:
        return {}, {}
    speakers = cue_speakers(cues, turns)
    names_file = workspace / NAMES_NAME
    if not names_file.exists():
        ask = getattr(provider, "name_speakers", None)
        found: dict[int, str] | None = None
        if ask is not None:
            emit("  speaker names…")
            try:
                found = ask(_first_words(cues, speakers, word), word)
            except ProviderLimitError as e:
                emit(str(e), kind="warn")
        if found:
            samples = _samples(cues, speakers)
            found = {n: name for n, name in found.items()
                     if n not in samples
                     or not _names_someone_else(name, samples[n][2], samples[n][1])}
        if found is not None:  # an answer that failed is not kept: the next run asks again
            numbers = sorted({s for s in speakers if s is not None})
            names_file.write_text(json.dumps(
                {str(n): found.get(n, "") for n in numbers}, indent=1, ensure_ascii=False))
    return {id(c): s for c, s in zip(cues, speakers, strict=True)}, speaker_names(workspace)


def _voices(workspace: Path, names: dict[int, str], word: str) -> list[dict]:
    """The speakers told apart by voice, as {name, spoke_s, first_s}, in order of appearance."""
    voices: dict[int, dict] = {}
    for turn in speaker_turns(workspace):
        voice = voices.setdefault(turn["speaker"], {
            "name": names.get(turn["speaker"]) or f"{word} {turn['speaker']}",
            "spoke_s": 0.0, "first_s": turn["start_s"]})
        voice["spoke_s"] += turn["end_s"] - turn["start_s"]
    return list(voices.values())


def _build_provider(config: VTDConfig, workspace: Path) -> LLMProvider:
    p = (config.ai.provider or "anthropic").lower()
    if p == "anthropic":
        from .ai.anthropic_provider import AnthropicProvider
        return AnthropicProvider(config, workspace)
    if p == "openai":
        from .ai.openai_provider import OpenAIProvider
        return OpenAIProvider(config, workspace)
    if p == "ollama":
        from .ai.ollama_provider import OllamaProvider
        return OllamaProvider(config, workspace)
    raise VTDError(f"Unknown AI provider: {config.ai.provider}")


def synthesize(workspace: Path, config: VTDConfig, manifest: Manifest) -> list[str]:
    """Write notes.md. Returns the parts that are missing or incomplete (empty when whole)."""
    transcript_path = workspace / "transcript.txt"
    if not transcript_path.exists():
        raise VTDError("transcript.txt not found — run the transcribe stage first.")

    sections = read_json(workspace / "sections.json", required="detect-sections")
    speakers = read_json(workspace / "speakers.json", [])
    cues = parse_vtt(workspace / "transcript.vtt")

    provider = _build_provider(config, workspace)
    labels = LABELS[config.output_language]

    speaker_of, names = _speakers(workspace, provider, cues, labels["speaker"])

    section_records: list[dict] = []
    titles: list[str] = []
    # Follow-ups read from each section's transcript, when the provider can do that.
    noted: list[str] | None = [] if hasattr(provider, "section_followups") else None
    referred: list[tuple[dict, str, list]] = []  # with --references: (record, transcript, works)
    shown = 0  # sections with speech so far

    for position, s in enumerate(sections):
        if s.get("is_silent"):
            section_records.append({
                **s,
                "title": labels["silence"],
                "body": f"_{labels['no_audio']}_",
                "frames": [],
                "number": None,  # not part of the outline; listed in the timeline
            })
            titles.append(labels["silence"])
            continue

        in_section = cues_in_section(
            cues, s["start_s"], s["end_s"], is_last=(position == len(sections) - 1))
        chunk = cue_text(in_section)
        if not chunk:
            section_records.append({
                **s,
                "title": s.get("title") or labels["untitled"],
                "body": f"_{labels['empty_section']}_",
                "frames": [],
                "number": None,
            })
            titles.append(section_records[-1]["title"])
            continue

        frames = _frames_for_section(workspace, config, s["start_s"], s["end_s"])
        shown += 1
        si = SectionInput(
            index=shown - 1,  # numbered as in the document: sections without speech are not
            title=s.get("title"),
            start_s=s["start_s"],
            end_s=s["end_s"],
            transcript_chunk=chunk if not speaker_of else labelled_text(
                in_section, [speaker_of[id(c)] for c in in_section], labels["speaker"], names),
            frame_paths=frames,
            speaker_word=labels["speaker"] if speaker_of else "",
            plain_chunk=chunk if speaker_of else "",
        )
        emit(f"  section {shown} ({hms(s['start_s'])}–{hms(s['end_s'])}): synthesizing…")
        title = provider.name_section(si)
        body = provider.write_section_body(si)
        if noted is not None:
            noted += provider.section_followups(si)
        section_records.append({
            **s,
            "title": title,
            "body": body,
            "frames": [str(f.relative_to(workspace)) for f in frames],
        })
        if config.references:
            referred.append((section_records[-1], chunk, provider.section_references(si)))
        titles.append(title)

    spoken = [r for r in section_records if "number" not in r]
    for number, record in enumerate(spoken, start=1):
        record["number"] = number
    titles = [r["title"] for r in spoken]
    synthesis_context = _summary_digest(spoken)
    limited: list[str] = []  # parts replaced by a placeholder because of a provider limit

    chapters = _chapters(provider, spoken, synthesis_context)

    emit("  cross-cutting themes…")
    try:
        themes_md = provider.write_themes(titles, synthesis_context)
    except ProviderLimitError as e:
        emit(str(e), kind="warn")
        limited.append("themes")
        themes_md = f"_{labels['themes_unavailable']}_"

    emit("  follow-ups…")
    try:
        followups = _followups(provider, noted, synthesis_context)
    except ProviderLimitError as e:
        emit(str(e), kind="warn")
        limited.append("follow-ups")
        followups = []

    emit("  glossary…")
    try:
        glossary = [
            {"term": g.term, "definition": g.definition}
            for g in provider.build_glossary(synthesis_context)
        ]
    except ProviderLimitError as e:
        emit(str(e), kind="warn")
        limited.append("glossary")
        glossary = []

    works = None
    if config.references:
        emit("  references…")
        try:
            suggested = provider.further_reading(synthesis_context)
        except ProviderLimitError as e:
            emit(str(e), kind="warn")
            limited.append("further reading")
            suggested = []
        works, looked_up = bibliography(workspace, referred, suggested)
        if not looked_up:
            limited.append("references")

    rendered = render_template(
        f"full_{config.content_type}.md.j2",
        config=config,
        manifest=manifest,
        sections=section_records,
        chapters=chapters,
        speakers=speakers,
        voices=_voices(workspace, names, labels["speaker"]),
        followups=followups,
        glossary=glossary,
        works=works,
        themes_md=themes_md,
        provider_name=provider.name,
    )
    (workspace / config.output_doc_name).write_text(rendered)
    return [*getattr(provider, "degraded", []), *limited]
