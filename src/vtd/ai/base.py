"""Abstract LLM provider interface.

Every provider is responsible for translating these high-level methods into the
shape its SDK / endpoint expects. No other code in vtd talks to a provider SDK.
"""
from __future__ import annotations

import base64
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import LANGUAGE_NAMES, VTDConfig
from ..utils import VTDError, emit
from . import cache

# {source} is the kind of recording, so a lecture is not summarised as a meeting.
SYSTEM_PROMPT = (
    "You are a meticulous technical writer. Given an excerpt of a {source} "
    "transcript and a small number of representative video frames, write the "
    "Markdown asked for. Be precise, prefer concrete details over generalities, "
    "and cite transcript content as quotes when it strengthens the writing, using "
    "quotation marks only for words that are in the transcript. {attribution} "
    "Use existing domain terminology faithfully. Never invent facts not in the "
    "transcript or visible in the frames. Output Markdown only — no preamble, "
    "no closing remarks."
)
_NOBODY_MARKED = (
    "The transcript does not say who is speaking: do not credit words to a person unless "
    "the transcript itself makes plain that this person said them."
)
_SPEAKERS_MARKED = (
    "Each change of speaker in the transcript starts with who is speaking: credit words "
    "only to the speaker marked on them."
)
# The whole-document calls work from the section summaries, not from the transcript.
SUMMARY_SYSTEM_PROMPT = (
    "You are a meticulous technical writer. Given summaries of the sections of a "
    "{source}, write the Markdown asked for. Be precise, prefer concrete details "
    "over generalities, and use existing domain terminology faithfully. Never "
    "invent facts not in the summaries. Output Markdown only — no preamble, "
    "no closing remarks."
)
SOURCE_NOUNS = {
    "generic": "recording",
    "meeting": "meeting",
    "lecture": "lecture",
    "podcast": "podcast episode",
}
EXCERPT = "TRANSCRIPT EXCERPT"    # material label of the per-section calls
SUMMARIES = "SECTION SUMMARIES"   # material label of the whole-document calls
NONE = "NONE"  # what the model is asked to reply when a list has no entries
_ONE_PER_LINE = (
    " Output ONE PER LINE, prefixed with `- `, each a single short sentence. "
    f"If there are none, reply with exactly {NONE}."
)
# What "follow-ups" means depends on the content type; the templates head the list accordingly.
FOLLOWUP_KINDS = {
    "generic": (
        "open follow-up, action item, deferred decision, or 'we'll come back to that' "
        "moment"
    ),
    "lecture": (
        "exercise, assignment, recommended reading or open question the lecturer left for "
        "the audience"
    ),
    "podcast": (
        "book, article, person, product or other resource mentioned as worth following up"
    ),
}
FOLLOWUP_KINDS["meeting"] = FOLLOWUP_KINDS["generic"]
_FOLLOWUP_EXTRA = {"generic": ", with the owner if mentioned", "meeting": ", with the owner if mentioned"}
FOLLOWUP_INSTRUCTIONS = {
    kind: f"List every {what} in these section summaries{_FOLLOWUP_EXTRA.get(kind, '')}."
    for kind, what in FOLLOWUP_KINDS.items()
}
# The closing section has a different heading per content type; the prompt asks for
# what that heading promises.
THEME_KINDS = {
    "generic": "themes that recur across several sections",
    "meeting": "themes that recur across several sections",
    "lecture": "key concepts that run through several sections",
    "podcast": "notable themes that come up in several sections",
}
ITEMS = "ITEMS NOTED SECTION BY SECTION"  # material label of the follow-up merge

# Longest input, in characters, sent for a section title and a section body.
TITLE_CHARS = 6000
BODY_CHARS = 8000

_RERUN_HINT = "Cached earlier outputs were preserved; rerun with `--only synthesize`."


class ProviderLimitError(VTDError):
    """The provider refused the request because of a rate or usage limit."""


def provider_error(vendor: str, label: str, error: Exception) -> VTDError:
    """Translate an SDK exception into the user-facing error for a failed request."""
    if error.__class__.__name__ == "RateLimitError":
        wait = re.search(r"try again in ([^.]+(?:\.[0-9]+)?s)", str(error), re.IGNORECASE)
        wait_hint = f" Try again in {wait.group(1)}." if wait else ""
        return ProviderLimitError(
            f"{vendor} rate limit reached while generating {label}.{wait_hint} {_RERUN_HINT}"
        )
    return VTDError(f"{vendor} request failed while generating {label}: {error} {_RERUN_HINT}")


@dataclass
class SectionInput:
    index: int
    title: str | None
    start_s: float
    end_s: float
    transcript_chunk: str
    frame_paths: list[Path]
    # When speakers were identified: `transcript_chunk` marks each change of speaker,
    # `speaker_word` is the word used in the marks ("Speaker", "Hablante") and
    # `plain_chunk` is the same text without the marks.
    speaker_word: str = ""
    plain_chunk: str = ""


@dataclass
class GlossaryTerm:
    term: str
    definition: str


@dataclass(frozen=True)
class Work:
    """A published work as the model names it: checked in a catalog before it is cited."""
    author: str
    title: str
    year: str = ""


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def name_section(self, section: SectionInput) -> str:
        """Return a short title (≤ 7 words) for the section."""

    @abstractmethod
    def write_section_body(self, section: SectionInput) -> str:
        """Return Markdown prose for the section body."""

    @abstractmethod
    def extract_followups(self, summaries: str) -> list[str]:
        """Return a list of open follow-ups / action items, each one short sentence."""

    @abstractmethod
    def build_glossary(self, summaries: str) -> list[GlossaryTerm]:
        """Return a glossary of domain terms appearing in the section summaries."""

    @abstractmethod
    def write_themes(self, sections_titles: list[str], summaries: str) -> str:
        """Return Markdown prose for cross-cutting themes across the document."""


# How many times an answer that cannot be used is asked for again.
RETRIES = 2
# A language is judged from its commonest words; shorter answers are not judged.
_COMMON_WORDS = {
    "en": frozenset(
        "the of and to in is that it for as with was on are this be by an at from or which "
        "but not have has their they these its were been also can will about what when "
        "there a i you we he she if so then more do how why".split()),
    "es": frozenset(
        "el la los las de que y en un una es se por con para del al como su más lo no pero "
        "sus le ya o este esta son también fue ha porque muy sin sobre entre cuando "
        "hay".split()),
}
_MIN_WORDS_TO_JUDGE = 25
# Not the model's own prose: code, tables, list items (titles, names), quotes and links.
_NOT_PROSE = re.compile(
    r"```.*?```|~~~.*?~~~|`[^`\n]*`|^\s*(?:\||[-*+]\s|\d+[.)]\s).*$"
    r"|\"[^\"\n]*\"|“[^”\n]*”|«[^»\n]*»|https?://\S+",
    re.DOTALL | re.MULTILINE,
)


def written_in(text: str, language: str) -> bool:
    """Whether `text` reads as written in `language` (a code vtd can write in).

    False only when another of those languages clearly dominates what the model wrote
    itself: quotes, code, tables and list items are often in the language of the
    recording or in English, and are not counted.
    """
    prose = _NOT_PROSE.sub(" ", text)
    words = re.findall(r"[^\W\d_]+", prose.casefold())
    if language not in _COMMON_WORDS or len(words) < _MIN_WORDS_TO_JUDGE:
        return True
    hits = {code: sum(w in common for w in words) for code, common in _COMMON_WORDS.items()}
    return max(hits.values()) <= 2 * hits[language]


def _chapters(answer: str, count: int) -> list[tuple[int, int, str]]:
    """(first, last, title) of each chapter, if they cover sections 1..count in order."""
    chapters: list[tuple[int, int, str]] = []
    for line in answer.splitlines():
        m = re.match(r"^[\W_]*(\d+)\s*(?:[-–—]\s*(\d+))?\s*::\s*(.+?)\s*$", line)
        if m:
            first = int(m.group(1))
            chapters.append((first, int(m.group(2) or first), m.group(3).strip(" *_#`")))
    expected = 1
    for first, last, title in chapters:
        if first != expected or last < first or not title:
            return []
        expected = last + 1
    # One chapter holding everything says nothing about where the subject changes.
    return chapters if expected == count + 1 and len(chapters) > 1 else []


def _plain(text: str) -> str:
    """Text reduced to its words: no case, accents or punctuation, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(
        c for c in decomposed if not unicodedata.combining(c))))


# A quotation in straight, curly or angle marks. An opening straight mark does not follow
# a letter or a digit (15" is inches), and emphasis counts only when it wraps the whole.
_MARKS = r'(?<![\w"])"([^"\n]+)"|“([^”\n]+)”|«([^»\n]+)»'
_QUOTE = re.compile(rf"(\*{{1,2}})(?:{_MARKS})\1|{_MARKS}")
_CODE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)
_ELLIPSIS = re.compile(r"\[?(?:\.{3}|…)\]?")
MIN_QUOTE_WORDS = 4  # shorter quoted spans are terms or titles, not quotations


def _quotations(text: str) -> list[tuple[str, str]]:
    """(as written with its marks, the words inside) of each quotation outside code."""
    masked = _CODE.sub(lambda m: " " * len(m.group(0)), text)
    found = []
    for match in _QUOTE.finditer(masked):
        inner = next(g for g in match.groups()[1:] if g is not None)
        found.append((text[match.start():match.end()], inner))
    return found


def unsupported_quotes(text: str, source: str) -> list[str]:
    """Quotations in `text` whose words are not in `source`, as written (with marks).

    A quotation may skip words with an ellipsis; its stretches must then be in the
    source one after the other, and a stretch of a single word proves nothing.
    """
    spoken = f" {_plain(source)} "
    missing = []
    for written, quote in _quotations(text):
        parts = [p for p in (_plain(part) for part in _ELLIPSIS.split(quote)) if p]
        if sum(len(p.split()) for p in parts) < MIN_QUOTE_WORDS:
            continue
        position = 0
        for part in parts:
            position = spoken.find(f" {part} ", position) if len(part.split()) > 1 else -1
            if position < 0:
                missing.append(written)
                break
            position += len(part)
    return missing


def unquote(text: str, quotes: list[str]) -> str:
    """`text` with these quotations left as plain words, no longer presented as verbatim."""
    for written, inner in _quotations(text):
        if written in quotes:
            text = text.replace(written, inner, 1)
    return text


def _length_hint(chars: int) -> str:
    """How long a summary of an excerpt of this many characters should be."""
    if chars < 500:
        return "Write one or two sentences: the excerpt is short, and padding it is wrong."
    low = min(10, max(2, round(chars / 700)))
    high = min(14, max(low + 2, round(chars / 400)))
    return f"Aim for {low}–{high} sentences, in proportion to the excerpt; never pad."


def _who_speaks(word: str) -> str:
    """The rule of the section prompt on crediting what is said to someone."""
    if not word:
        return (  # unchanged for runs without --speakers, so their cached answers stay valid
            "Name a person only for words the excerpt itself "
            "shows that person saying (for instance when they introduce themselves); "
            "otherwise refer to them neutrally, as the speaker or a participant, in the "
            "language of your answer. A name shown on screen says who is on camera, not who "
            "is speaking, and several people may speak in one excerpt: when one person "
            "finishes and another begins, do not carry the first name over."
        )
    return (
        "Each change of speaker in the excerpt starts with who is speaking: a name, or a "
        f"mark such as '{word} 3' when the name is not known ('{word} ?' is a voice that "
        "could not be told apart). Credit every statement to the speaker marked on it, "
        "written exactly as marked. Never put a name on a numbered speaker yourself, even "
        "when a name is said nearby. A name shown on screen says who is on camera, not who "
        "is speaking."
    )


def _speaker_names(answer: str) -> dict[int, str]:
    """{speaker number: name} from `<word> N :: NAME` lines; unknown and repeated names out."""
    names: dict[int, str] = {}
    for line in answer.splitlines():
        m = re.match(r"^[\W_]*[^\d:]*?(\d+)[\s*_`]*::\s*(.+?)\s*$", line)
        if not m:
            continue
        name = m.group(2).strip(" *_`\"'.")
        plain = _plain(name)
        if (plain and len(name.split()) <= 5 and not plain.startswith("unknown")
                and plain not in _NOT_A_NAME and "(" not in name):
            names[int(m.group(1))] = name
    said = [_plain(n) for n in names.values()]
    return {number: name for number, name in names.items() if said.count(_plain(name)) == 1}


_NOT_A_NAME = {"desconocido", "desconocida", "n a", "na", "none", "ninguno", "ninguna",
               "no se sabe", "sin nombre", "anonimo", "anonima"}
_PERSON = r"[A-ZÁÉÍÓÚÑ][\wáéíóúñü]+(?:\s+(?:de|del|la|los|las|[A-ZÁÉÍÓÚÑ][\wáéíóúñü]+))*"


def _unnamed(word: str, text: str) -> str:
    """`text` without names the model attached to numbered speakers on its own.

    'Ana Pérez (Speaker 3)' and 'Speaker 3 (Ana Pérez)' become 'Speaker 3': a speaker
    is only named through speaker_names.json, never by the model in passing.
    """
    if not word:
        return text
    mark = rf"{re.escape(word)} \d+"
    text = re.sub(rf"{_PERSON}\s*\(({mark})\)", r"\1", text)
    return re.sub(rf"({mark})\s*\({_PERSON}\)", r"\1", text)


def _room(section: SectionInput) -> int:
    """Longest excerpt sent for a section: the speaker marks do not count against it."""
    marks = len(section.transcript_chunk) - len(section.plain_chunk) if section.plain_chunk else 0
    return BODY_CHARS + max(0, marks)


def _is_title(answer: str) -> bool:
    title = answer.strip()
    return 0 < len(title.split()) <= 14 and "\n" not in title


def _is_none(answer: str) -> bool:
    return answer.strip().strip("`.").upper() in ("", NONE)


def _bullets(answer: str) -> list[str]:
    items = [line.strip().lstrip("- ").strip()
             for line in answer.splitlines() if line.strip().startswith("-")]
    return [i for i in items if i]


def _terms(answer: str) -> list[GlossaryTerm]:
    terms: list[GlossaryTerm] = []
    for line in answer.splitlines():
        term, sep, definition = line.partition("::")
        term, definition = term.strip(" -*").strip(), definition.strip()
        if sep and term and definition:
            terms.append(GlossaryTerm(term=term, definition=definition))
    return terms


_NOT_AN_AUTHOR = {"author", "autor", "unknown", "desconocido", "anonymous", "anonimo", "varios"}
_NOT_A_TITLE = {"title", "titulo"}
_FIELD_NAME = re.compile(r"^(author|autor|title|titulo|título|year|año)\s*:\s*", re.IGNORECASE)
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# A date that is not a year of four digits: "c. 375 BC", "116", "4th century BC", "n.d.", "s.f."
_DATE_WORD = re.compile(
    r"\d+(st|nd|rd|th)?|c|ca|circa|bc|bce|ad|ce|ac|dc|a|d|century|siglo|n|s|f|nd|sf|unknown|"
    r"desconocido")
_YEAR = re.compile(r"\b\d{4}\b")
_YEAR_AFTER_TITLE = re.compile(r"\s*[(\[,]\s*(\d{4})\s*[)\]]?\s*$")


def _fields(line: str) -> dict[str, str]:
    """{author, title, year} of one `AUTHOR :: TITLE :: YEAR` line; {} if it is not one.

    Fields are taken by their labels when the model writes them ("Title: … :: Author:
    …"), otherwise in order. A title may hold `::` itself; the year is the first field
    after it that is a date, and what follows the year (a publisher) is left out.
    """
    parts = [part.strip(" -*_`\"“”«»") for part in _LIST_MARK.sub("", line).split("::")]
    if len(parts) < 2:
        return {}
    labels = [(_FIELD_NAME.match(part) or [""])[0].strip(" :").casefold() for part in parts]
    parts = [_FIELD_NAME.sub("", part) for part in parts]
    named = {"author": ("author", "autor"), "title": ("title", "titulo", "título")}
    if all(any(label in names for label in labels) for names in named.values()):
        which = {key: next(i for i, label in enumerate(labels) if label in names)
                 for key, names in named.items()}
        years = [part for part, label in zip(parts, labels, strict=True) if label in ("year", "año")]
        return {"author": parts[which["author"]], "title": parts[which["title"]],
                "year": " ".join(years)}

    def date(part: str) -> bool:
        return bool(_YEAR.search(part)) or all(map(_DATE_WORD.fullmatch, _plain(part).split()))

    if len(parts) > 2 and _YEAR.fullmatch(parts[1]) and not date(parts[2]):
        parts[1:3] = parts[2], parts[1]  # the year given before the title
    end = next((i for i in range(2, len(parts)) if date(parts[i])), len(parts))
    title, year = ": ".join(parts[1:end]), parts[end] if end < len(parts) else ""
    inside = _YEAR_AFTER_TITLE.search(title)
    if inside and not _YEAR.search(year) and _plain(title[:inside.start()]):
        title, year = title[:inside.start()], inside.group(1)  # "Thinking, Fast and Slow (2011)"
    return {"author": parts[0], "title": title, "year": year}


def _listed(answer: str) -> bool:
    """Whether the answer is a list of works in the format asked for, whatever becomes
    of its lines; the format given back unfilled is not."""
    found = [_fields(line) for line in answer.splitlines()]
    return any(_plain(f["author"]) and _plain(f["title"])
               and _plain(f["author"]) not in ("author", "autor")
               for f in found if f)


def _works(answer: str) -> list[Work]:
    """The works in `- AUTHOR :: TITLE :: YEAR` lines.

    Left out: a line without an author or a title, the header of a table, and a
    title in brackets, which is the model describing a work it cannot name. Of
    several years the earliest is the first publication.
    """
    works: list[Work] = []
    for found in map(_fields, answer.splitlines()):
        author, title = found.get("author", ""), found.get("title", "")
        if (_plain(author) and _plain(title) and _plain(author) not in _NOT_AN_AUTHOR
                and not title.startswith("[") and _plain(title) not in _NOT_A_TITLE):
            works.append(Work(author, title, min(_YEAR.findall(found["year"]), default="")))
    return works


_DENIAL = {"no", "not", "none", "ninguna", "ninguno", "ningun", "nada", "nothing", "sin",
           "without", "cannot", "n"}  # "n" of "N/A"


def _no_works(answer: str) -> bool:
    """Whether the model says there is no work to list, in whatever words: NONE,
    '- NONE', 'Ninguna.', 'The excerpt does not refer to any published work.'

    An empty answer says nothing. An answer that goes on to give works is not a no:
    one with `::`, with a list, or with a year in it. A refusal is one: asked again,
    the model would refuse again.
    """
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines or "::" in answer or _YEAR.search(answer):
        return False
    if any(_LIST_MARK.match(line) for line in lines) and not (
            len(lines) == 1 and _plain(lines[0]) in _DENIAL):
        return False
    return bool(_DENIAL & set(_plain(lines[0]).split()))


_WORK_FORMAT = (
    " Output ONE PER LINE as `- AUTHOR :: TITLE :: YEAR`: the first author as `Surname, "
    "Given name`, spelled correctly; the exact title in the language it was published in "
    "(not translated into the language of your answer); and the year of first publication, "
    "or `?` when you are not sure of it. Every work listed is checked against library "
    "catalogs, so give the title exactly as it was published, never a description of the "
    "work in its place. Tests, institutions and theories are not works. "
    f"If there are none, reply with exactly {NONE}."
)


class ChatProvider(LLMProvider):
    """Prompts, caching and response parsing shared by the hosted chat vendors.

    A vendor subclass only says how to attach a frame and how to send one request.
    """

    vendor = "abstract"  # display name used in error messages
    _content_type = "generic"
    _language_code = ""

    def _configure(self, config: VTDConfig, workspace: Path) -> None:
        self._model = config.ai.model
        self._workspace = workspace
        self._output_language = LANGUAGE_NAMES.get(config.output_language, config.output_language)
        self._language_code = config.output_language
        self._content_type = config.content_type

    def _system_prompt(self, material: str, speakers_marked: bool = False) -> str:
        template = SYSTEM_PROMPT if material == EXCERPT else SUMMARY_SYSTEM_PROMPT
        return template.format(
            source=SOURCE_NOUNS[self._content_type],
            attribution=_SPEAKERS_MARKED if speakers_marked else _NOBODY_MARKED,
        )

    @property
    def degraded(self) -> list[str]:
        """Labels of answers that came back cut off, empty or unusable in this run."""
        return self.__dict__.setdefault("_degraded", [])

    @abstractmethod
    def _frame_block(self, jpeg_b64: str) -> dict:
        """Content block carrying one base64-encoded JPEG frame."""

    @abstractmethod
    def _request(self, system: str, content: list[dict], max_tokens: int) -> tuple[str, bool]:
        """Send one user message; return (text, whether it hit the token limit)."""

    def _frame_blocks(self, paths: list[Path]) -> list[dict]:
        blocks: list[dict] = []
        for p in paths:
            try:
                data = p.read_bytes()
            except OSError:
                continue
            blocks.append(self._frame_block(base64.standard_b64encode(data).decode("ascii")))
        return blocks

    def _clip(self, text: str, limit: int, label: str) -> str:
        """Cut an over-long input, saying so: the answer will not cover what was dropped."""
        if len(text) <= limit:
            return text
        emit(f"  · {label}: input cut from {len(text)} to {limit} characters", kind="warn")
        return text[:limit]

    def _call(self, *, instruction: str, transcript_chunk: str, frames: list[Path] | None = None,
              max_tokens: int = 1500, label: str, material: str = EXCERPT,
              usable: Callable[[str], bool] = bool, optional: bool = False,
              check: Callable[[str], str] | None = None,
              repair: Callable[[str], str] | None = None,
              speakers_marked: bool = False, any_language: bool = False) -> str:
        """Answer one prompt, reusing the on-disk cache when the same request was seen.

        `usable` says whether an answer can be used. One that cannot (empty, or in the
        wrong format) or that was cut off is not cached, so a later run asks again; it
        is also recorded as degraded unless the caller has a fallback (`optional`).
        `check` names a further problem with an answer ("" when there is none). If that
        is the only problem left after the retries, `repair` fixes the answer by code,
        and the repaired answer is kept like a good one. `any_language` is for an
        answer that is not prose of the model's own (a list of titles as published).
        """
        frames = frames or []
        system = self._system_prompt(material, speakers_marked)
        user_text = (
            f"Write the final answer in {self._output_language}.\n\n"
            f"{instruction}\n\n{material}:\n```\n{transcript_chunk}\n```\n\n"
            # Said again after the material: smaller models otherwise answer in its language
            # or in the language of these instructions.
            f"Write the whole answer in {self._output_language}."
        )
        # The key covers everything that is sent, so no change to a prompt reuses an old answer.
        key = cache.cache_key(
            self.name,
            self._model,
            system,
            user_text,
            [(p.name, p.stat().st_size if p.exists() else 0) for p in frames],
        )
        cached = cache.load(self._workspace, key)
        if cached is not None:
            return cached

        text, problem, repairable = "", "", False
        for attempt in range(RETRIES + 1):
            asked = user_text if not attempt else (
                f"{user_text}\n\nYour previous answer could not be used: {problem} "
                "Answer again, following the instructions exactly."
            )
            content = self._frame_blocks(frames)
            content.append({"type": "text", "text": asked})
            try:
                text, truncated = self._request(system, content, max_tokens)
            except Exception as e:
                raise provider_error(self.vendor, label, e) from e
            repairable = False
            if truncated:
                emit(f"  · {label}: response hit the token limit; using it but not caching",
                     kind="warn")
                break  # asking again would stop at the same limit
            if not usable(text):
                problem = "it was empty or not in the requested format."
            elif not any_language and not written_in(text, self._language_code):
                problem = f"it was not written in {self._output_language}."
            elif check is not None and (found := check(text)):
                problem, repairable = found, repair is not None
                if repairable and attempt:
                    break  # asked once already: correct it here instead of asking a third time
            else:
                cache.store(self._workspace, key, f"{label}: {instruction[:120]}…", text)
                emit(f"  · {label}: {len(text)} chars cached at {key[:12]}…")
                return text
            again = "; asking again" if attempt < RETRIES else "; not cached"
            emit(f"  · {label}: {problem.splitlines()[0].rstrip('.:')[:200]}{again}", kind="warn")
        if repairable:
            emit(f"  · {label}: still so after asking again; corrected here", kind="warn")
            text = repair(text)
            cache.store(self._workspace, key, f"{label}: {instruction[:120]}…", text)
            return text
        if not optional and label not in self.degraded:
            self.degraded.append(label)
        return text

    def name_section(self, section: SectionInput) -> str:
        if section.title:
            return section.title
        label = f"title §{section.index + 1}"
        instruction = (
            "Give this transcript excerpt a short, specific section title — "
            "five to seven words, sentence case, no trailing punctuation. "
            "Reply with ONLY the title text."
        )
        title = self._call(
            instruction=instruction,
            transcript_chunk=self._clip(
                section.plain_chunk or section.transcript_chunk, TITLE_CHARS, label),
            max_tokens=40,
            label=label,
            usable=_is_title,
            optional=True,  # "Section N" stands in for a missing title
        ).strip().strip('"').strip("'")
        return title or f"Section {section.index + 1}"

    def write_section_body(self, section: SectionInput) -> str:
        label = f"body §{section.index + 1}"
        excerpt = section.transcript_chunk
        instruction = (
            "Write a Markdown summary of this section that is faithful to the excerpt. "
            "Include: (a) the main topic; (b) the key points in your own words, with the "
            "examples the speaker works through and the conclusions and recommendations the "
            "speaker states, keeping their direction (what is recommended and what is "
            "advised against); (c) at most two short direct quotes, in quotation marks, "
            "made only of words that appear in the excerpt exactly as quoted; (d) what a "
            "frame adds when it shows content such as slide text or a figure: describe it, "
            "do not quote slide text as if it had been spoken, and do not comment on the "
            f"setting or on how people look. {_who_speaks(section.speaker_word)} Do not guess "
            "anyone's gender. Do not add evaluation, praise or background that is not in "
            "the excerpt. "
            f"{_length_hint(min(len(section.plain_chunk or excerpt), BODY_CHARS))} "
            "Write flowing paragraphs: do not label these parts, and do not add headings."
        )

        def quotes_not_said(answer: str) -> str:
            missing = unsupported_quotes(answer, excerpt)
            return "" if not missing else (
                "these quotations are not in the excerpt word for word: "
                + "; ".join(missing[:4])
                + ". Quote the excerpt exactly, or say it in your own words without "
                "quotation marks."
            )

        return _unnamed(section.speaker_word, self._call(
            instruction=instruction,
            transcript_chunk=self._clip(excerpt, _room(section), label),
            frames=section.frame_paths,
            max_tokens=2000,
            label=label,
            check=quotes_not_said,
            # Still not verbatim after the retries: keep the words, drop the claim.
            repair=lambda answer: unquote(answer, unsupported_quotes(answer, excerpt)),
            speakers_marked=bool(section.speaker_word),
        ))

    def name_speakers(self, first_words: str, word: str) -> dict[int, str] | None:
        """Names of the numbered speakers, as far as the recording itself gives them.

        None when the model gave no usable answer, so that it is asked again another time.
        """
        instruction = (
            f"Below, for each speaker mark ('{word} 1', '{word} 2', …), is the line said "
            "just before that speaker first spoke (BEFORE) and that speaker's first words. "
            "Give a speaker's name only when the speaker says their own name, or is called "
            "by that name in the BEFORE line and then answers. Output one line per speaker "
            f"as `{word} N :: NAME`, and `{word} N :: UNKNOWN` when the name is not clear. "
            "Never guess, and never give two speakers the same name."
        )
        text = self._call(
            instruction=instruction,
            transcript_chunk=first_words,
            max_tokens=600,
            label="speaker names",
            usable=lambda answer: "::" in answer,
            optional=True,  # without names the speakers keep their numbers
            speakers_marked=True,
        )
        return _speaker_names(text) if "::" in text else None

    def section_followups(self, section: SectionInput) -> list[str]:
        """Follow-ups said in one section, read from its transcript and not from a summary."""
        label = f"follow-ups §{section.index + 1}"
        instruction = (
            f"List every {FOLLOWUP_KINDS[self._content_type]} in this transcript excerpt"
            f"{_FOLLOWUP_EXTRA.get(self._content_type, '')}, with the titles, authors or "
            "dates as they are said. Only things to do, read or settle after this point: "
            "not what happened during the session (a break, a discussion that was held), "
            "not general advice, not a summary of what was said. A remark or a piece of "
            "advice is not a task: include only what the audience is explicitly asked to "
            "do or read, or a question explicitly left open. Write each as the thing to "
            "do. When in doubt, leave it out." + _ONE_PER_LINE
        )
        text = self._call(
            instruction=instruction,
            transcript_chunk=self._clip(section.transcript_chunk, _room(section), label),
            max_tokens=400,
            label=label,
            usable=lambda answer: _is_none(answer) or bool(_bullets(answer)),
            speakers_marked=bool(section.speaker_word),
        )  # not optional: a section whose list fails is a gap, asked for again on --resume
        return _bullets(text)

    def section_references(self, section: SectionInput) -> list[Work]:
        """Published works one section refers to, read from its transcript."""
        label = f"references §{section.index + 1}"
        instruction = (
            "List the published works (books, scientific papers, reports) that this "
            "transcript excerpt refers to: those named by their title, and those a speaker "
            "plainly means when naming their author (for instance a researcher and the "
            "theory or study they are known for). A transcript misspells names and gives "
            "titles loosely or translated: use your own knowledge of the literature to "
            "give the author and title as published. Leave out a person named without an "
            "identifiable work, and never add a work the excerpt does not refer to."
            + _WORK_FORMAT
        )
        text = self._call(
            instruction=instruction,
            transcript_chunk=self._clip(section.transcript_chunk, _room(section), label),
            max_tokens=500,
            label=label,
            usable=lambda answer: _no_works(answer) or _listed(answer),
            speakers_marked=bool(section.speaker_word),
            any_language=True,
            optional=True,  # the notes are whole without it; asked again on the next run
        )
        return _works(text)

    def further_reading(self, summaries: str) -> list[Work]:
        """Works on the subjects covered that a reader could go on to: not from the recording."""
        instruction = (
            "Suggest up to 8 published works (books or scientific papers) for a reader who "
            "wants to study the subjects of these section summaries further: foundational "
            "or widely cited works on those subjects. This is the one task in which you "
            "draw on your own knowledge of the literature and not on the summaries. If the "
            "subjects are not ones with a literature of their own (a status meeting, "
            f"practical arrangements), reply with exactly {NONE}." + _WORK_FORMAT
        )
        text = self._call(
            instruction=instruction,
            transcript_chunk=summaries,
            max_tokens=700,
            label="further reading",
            material=SUMMARIES,
            usable=lambda answer: _no_works(answer) or _listed(answer),
            any_language=True,
            optional=True,
        )
        return _works(text)

    def merge_followups(self, items: list[str]) -> list[str]:
        """The follow-ups noted section by section as one list without repeats."""
        if len(items) < 2:
            return items
        instruction = (
            "Below are items noted section by section, one per line. Merge the ones that "
            "say the same thing into one, keep every distinct one in its original order, "
            "and do not add any. Drop an item only if it is not something to do, read or "
            "settle afterwards." + _ONE_PER_LINE
        )
        text = self._call(
            instruction=instruction,
            transcript_chunk="\n".join(f"- {item}" for item in items),
            max_tokens=max(900, 60 * len(items)),
            label="follow-ups",
            material=ITEMS,
            # Here an empty answer is a failure, not "none": it would drop every item.
            usable=lambda answer: bool(_bullets(answer)) or (
                bool(answer.strip()) and _is_none(answer)),
        )
        # A merge that failed or was cut off must not lose items: keep them unmerged.
        # NONE is an answer too: none of the items was something to do afterwards.
        return items if "follow-ups" in self.degraded else _bullets(text)

    # The calls below work on a digest of the section summaries already written.

    def group_sections(self, titles: list[str], summaries: str) -> list[tuple[int, int, str]]:
        """Chapters as (first section, last section, title), numbered from 1; [] if none.

        Sections are cut where the speaker pauses or where the text gets long, not
        where the subject changes; this asks where it does.
        """
        count = len(titles)
        instruction = (
            f"Below are the {count} sections of this {SOURCE_NOUNS[self._content_type]}, in "
            "order, with their summaries. Group consecutive sections that deal with the same "
            "subject into chapters: between 2 and 8 chapters, each one or more whole "
            "sections. Output one chapter per line as `FIRST-LAST :: CHAPTER TITLE`, where "
            f"FIRST and LAST are section numbers. Every section from 1 to {count} must be in "
            "exactly one chapter, in order. Chapter titles: three to eight words, sentence "
            "case.\n\nSECTION TITLES:\n"
            + "\n".join(f"{n}. {t}" for n, t in enumerate(titles, start=1))
        )
        text = self._call(
            instruction=instruction,
            transcript_chunk=summaries,
            max_tokens=600,
            label="chapters",
            material=SUMMARIES,
            usable=lambda answer: bool(_chapters(answer, count)),
            optional=True,  # without chapters the sections are listed as they are
        )
        return _chapters(text, count)

    def extract_followups(self, summaries: str) -> list[str]:
        text = self._call(
            instruction=FOLLOWUP_INSTRUCTIONS[self._content_type] + _ONE_PER_LINE,
            transcript_chunk=summaries,
            max_tokens=900,
            label="follow-ups",
            material=SUMMARIES,
            usable=lambda answer: _is_none(answer) or bool(_bullets(answer)),
        )
        return _bullets(text)

    def build_glossary(self, summaries: str) -> list[GlossaryTerm]:
        instruction = (
            "Extract a glossary of domain-specific terms used in these section summaries. "
            "Output as `TERM :: DEFINITION` on each line. Definitions should be "
            "one short sentence, faithful to how the term is used and adding nothing the "
            "summaries do not say. Prefer the terms the speaker defines or insists on; "
            "include only terms that would be unfamiliar to a generalist reader. Spell "
            "terms correctly, with their accents. "
            f"Max 25 terms. If there are none, reply with exactly {NONE}."
        )
        text = self._call(
            instruction=instruction,
            transcript_chunk=summaries,
            max_tokens=1200,
            label="glossary",
            material=SUMMARIES,
            usable=lambda answer: _is_none(answer) or bool(_terms(answer)),
        )
        return _terms(text)

    def write_themes(self, sections_titles: list[str], summaries: str) -> str:
        instruction = (
            "Looking at the section summaries and the list of section titles below, "
            f"identify 3 to 5 {THEME_KINDS[self._content_type]}. Write each in Markdown "
            "under a level-3 (`###`) heading followed by a short paragraph. Output only "
            "those: no heading above them, no introduction and no closing remark.\n\n"
            "SECTION TITLES:\n" + "\n".join(f"- {t}" for t in sections_titles)
        )
        return self._call(
            instruction=instruction,
            transcript_chunk=summaries,
            max_tokens=1200,
            label="themes",
            material=SUMMARIES,
        )
