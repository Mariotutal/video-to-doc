"""`--references`: the works a recording refers to, looked up in public catalogs.

The model proposes works (an author and a title); none is printed on its word. Each
is searched in Open Library and Crossref, and only a record with that title, whose
first author is the one proposed, is cited, as the catalog has it. A work is listed
as referred to in the recording only when its author or its title is in the
transcript; a real work the model brought in by itself is further reading.
"""
from __future__ import annotations

import http.client
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import NamedTuple

from . import __version__
from .ai import cache
from .ai.base import _CODE, Work, _plain
from .utils import emit, with_fence_state

TIMEOUT_S = 15
PAUSE_S = 1.0  # between two requests to one catalog: Crossref allows anonymous callers one a second
ROWS = 5  # records read per search: the right one is not always the first
MAX_AUTHORS = 3
_SPELLING_RATIO = 0.8   # two spellings of one name: "Vygotsky" and "Vygotskiĭ"
_HEARD_RATIO = 0.75     # a name as Whisper wrote it: "Vigotski"
_MIN_SPELLED = 6        # shorter words are the same only when equal: "Chang" is not "Zhang"
_TITLE_ALONE_WORDS = 4  # a shorter title is also what its subject is called: "Deep Learning"
_TAG = re.compile(r"<[^>]+>")  # Crossref titles carry <i>, <sub>…
_MARKUP = re.compile(r"([\\`*_\[\]<>])")
_SUBTITLE = re.compile(r":|;|\s[-–—]\s|\s\(")
_PEOPLE = re.compile(r"\s*(?:;|&|\band\b|\by\b|\bet al\b\.?)\s*", re.IGNORECASE)
_PARTICLES = {"de", "del", "la", "las", "los", "van", "von", "der", "den", "da", "di", "du", "le"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}
# Words of the names of bodies, which have no surname: the whole name is the author.
_BODIES = set(
    "organization organisation organizacion university universidad association asociacion "
    "society sociedad institute instituto council consejo committee comite group grupo "
    "ministry ministerio department departamento foundation fundacion bank banco commission "
    "comision agency agencia academy academia college center centre centro press national "
    "nacional international internacional".split())
_ARTICLES = {"the", "a", "an", "el", "la", "los", "las", "un", "una"}
_JOINING = {"and", "y"}  # written "&" as often, which is no word
# Two spellings of one word: British and American, a letter doubled or not.
_SPELLINGS = [(re.compile(r"(.)\1+"), r"\1"), (re.compile("our"), "or"),
              (re.compile("is(?=[aei])"), "iz"), (re.compile("ys(?=e)"), "yz")]
# What a catalog can answer with that is not an answer: a dropped connection, a page
# that is not JSON, JSON of another shape.
_NO_ANSWER = (OSError, http.client.HTTPException, ValueError, LookupError, TypeError,
              AttributeError)


@dataclass
class Reference:
    """A work as a catalog records it."""
    authors: list[str]  # as printed
    first: str          # the first author as the catalog names them, "Family, Given" if it says
    year: str
    title: str
    source: str  # journal, book it is part of, or publisher; "" when the catalog has none
    url: str


_asked_at: dict[str, float] = {}  # catalog host → when it was last asked


def _get_json(url: str) -> dict:
    host = urllib.parse.urlsplit(url).netloc
    time.sleep(max(0.0, _asked_at.get(host, 0.0) + PAUSE_S - time.monotonic()))
    request = urllib.request.Request(url, headers={"User-Agent": f"vtd/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.load(response)
    finally:
        _asked_at[host] = time.monotonic()


def _text(value) -> str:
    """A catalog field as one line of plain text (some come as lists, some with tags)."""
    if isinstance(value, list):
        value = value[0] if value else ""
    return " ".join(_TAG.sub("", str(value or "")).split())


def _from_crossref(item: dict) -> Reference:
    people = item.get("author") or []
    named = [_text(f"{a.get('given', '')} {a.get('family') or a.get('name', '')}") for a in people]
    lead = people[0] if people else {}
    dated = (item.get("issued") or {}).get("date-parts") or [[]]
    return Reference(
        authors=named,
        first=(f"{_text(lead['family'])}, {_text(lead.get('given'))}" if lead.get("family")
               else _text(lead.get("name"))),
        year=str((dated[0] or [""])[0] or ""),
        title=_text(item.get("title")),
        source=_text(item.get("container-title")) or _text(item.get("publisher")),
        # Old DOIs hold < and >, which would end the link they are written in.
        url="https://doi.org/" + urllib.parse.quote(item["DOI"], safe="/:;()._-"),
    )


def _from_open_library(doc: dict) -> Reference:
    authors = [_text(name) for name in doc.get("author_name") or []]
    return Reference(
        authors=authors,
        first=authors[0] if authors else "",
        year=str(doc.get("first_publish_year") or ""),
        title=_text(doc.get("title")),
        source="",  # its publishers are those of every edition, in no order
        url=f"https://openlibrary.org{doc['key']}",
    )


def _people(names: str) -> list[str]:
    """The people in "A & B", "A and B", "A; B", "A et al."."""
    return [person for person in _PEOPLE.split(names) if person.strip()]


def _searched_author(work: Work) -> str:
    """The surname of the first person proposed: a catalog asked for a given name the
    model got wrong, or for two authors run together, finds nothing."""
    people = _people(work.author)
    return (_name(people[0]).sure or [work.author])[0] if people else work.author


def _crossref(work: Work) -> list[dict]:
    query = urllib.parse.urlencode({
        "query.bibliographic": work.title, "query.author": _searched_author(work), "rows": ROWS,
        "select": "DOI,title,author,issued,container-title,publisher",
    })
    return _get_json(f"https://api.crossref.org/works?{query}")["message"]["items"]


def _open_library(work: Work) -> list[dict]:
    query = urllib.parse.urlencode({
        "title": work.title, "author": _searched_author(work), "limit": ROWS,
        "fields": "key,title,author_name,first_publish_year",
    })
    return _get_json(f"https://openlibrary.org/search.json?{query}")["docs"]


# name → (search, how one of its records is read). Open Library first: it dates a
# book by its first edition, Crossref by the edition it indexed.
CATALOGS = {
    "Open Library": (_open_library, _from_open_library),
    "Crossref": (_crossref, _from_crossref),
}


def _like(a: str, b: str, ratio: float = _SPELLING_RATIO) -> bool:
    """Whether two names are one name spelled two ways, not two names that look alike
    ("Hernández" and "Fernández", "Roberts" and "Robertson")."""
    return (a[:1] == b[:1] and abs(len(a) - len(b)) <= 1 and min(len(a), len(b)) >= _MIN_SPELLED
            and a.isalpha() and b.isalpha() and SequenceMatcher(None, a, b).ratio() >= ratio)


def _slip(a: str, b: str) -> bool:
    """Whether two words of a title are one word spelled two ways: British or American
    ("Colour", "Civilisation"), a letter doubled or not ("Programing"), two letters
    swapped, a vowel more ("Judgement"). A letter changed or put in front is another
    word ("Microeconomics" and "Macroeconomics", "Evolution" and "Revolution")."""
    if a == b:
        return True
    if not (a.isalpha() and b.isalpha()):
        return False

    def spelled(word: str) -> str:
        for pattern, plain in _SPELLINGS:
            word = pattern.sub(plain, word)
        return word

    if min(len(a), len(b)) < _MIN_SPELLED - 1:  # "Color" has five letters; "I" is not "II"
        return False
    if spelled(a) == spelled(b):
        return True
    if min(len(a), len(b)) < _MIN_SPELLED:
        return False
    short, long = sorted((a, b), key=len)
    if len(long) == len(short) + 1:  # not at the start: "Atheism" is not "Theism"
        return any(long[i] in "aeiou" and long[:i] + long[i + 1:] == short
                   for i in range(1, len(long)))
    differ = [i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y]
    return (len(a) == len(b) and len(differ) == 2 and differ[1] == differ[0] + 1
            and a[differ[0]] == b[differ[1]] and a[differ[1]] == b[differ[0]])


def _title_words(title: str) -> list[str]:
    """The words by which titles are compared: no leading article, no "and"."""
    words = [w for w in _plain(title).split() if w not in _JOINING]
    return words[1:] if len(words) > 1 and words[0] in _ARTICLES else words


def _main_title(title: str) -> str:
    """A title without its subtitle."""
    return _SUBTITLE.split(title, maxsplit=1)[0]


def _same_title(wanted: str, found: str) -> bool:
    """Whether two titles name one work: word for word, allowing two spellings of a
    word, and allowing one of them to leave out the subtitle the other carries.

    A word that differs is another work ("Volume 1" and "Volume 3", "Western" and
    "Eastern"), and a title is not the beginning of a longer one ("Mind", "Mind in
    Society") unless what follows is set apart as a subtitle.
    """
    def same(a: list[str], b: list[str]) -> bool:
        return bool(a) and len(a) == len(b) and all(map(_slip, a, b))

    a, b = _title_words(wanted), _title_words(found)
    return (same(a, b) or same(_title_words(_main_title(wanted)), b)
            or same(a, _title_words(_main_title(found))))


class _Name(NamedTuple):
    """What one person's name says about their surname, without case or accents."""
    sure: list[str]   # the surname, likeliest form first; the whole name where it may be one
    maybe: list[str]  # the word before the last: a first surname or a middle name
    initial: str      # of the given name, "" when none is given


def _name(name: str) -> _Name:
    """ "Kuhn, Thomas" and "Thomas S. Kuhn" are surely kuhn. In a longer name the word
    before the last may be a first surname ("Gabriel García Márquez") or a middle name
    ("John Stuart Mill"); in a name of two words the first may be the family name
    ("Liu Cixin"). A body's surname is its whole name ("World Health Organization").
    """
    family, comma, given = name.partition(",")
    whole = _plain(name)
    if comma and _plain(family):
        words = _plain(family).split()
        singles = [w for w in reversed(words) if w not in _PARTICLES]
        return _Name(list(dict.fromkeys([" ".join(words), *singles])), [], _plain(given)[:1])
    said_words = [w for w in whole.split() if w not in _SUFFIXES]
    words = [w for w in said_words if len(w) > 1]  # initials say nothing
    if not words or _BODIES & set(words):
        return _Name([whole] if whole else [], [], "")
    before = words[-2:-1]
    maybe = [" ".join(words[-2:]), *(w for w in before if w not in _PARTICLES)] if before else []
    return _Name(list(dict.fromkeys([words[-1], whole])), maybe,
                 said_words[0][0] if len(said_words) > 1 else "")


def _same_author(wanted: str, first: str) -> int:
    """How surely one of the people in `wanted` is the first author a record names:
    2 by surname, 1 by a surname spelled nearly the same, 0 not at all.

    The first author only: the record of a review lists the reviewer first, then the
    authors of the book reviewed. A word that may be a middle name counts unless the
    given names begin differently ("Larry David" is not "Henry David Thoreau").
    """
    theirs, likeness = _name(first), 0
    for person in _people(wanted):
        mine = _name(person)
        crossed = set(mine.sure) & set(theirs.maybe) or set(mine.maybe) & set(theirs.sure)
        if (set(mine.sure) & set(theirs.sure)
                or sorted(_plain(person).split()) == sorted(_plain(first).split())
                or (crossed and (not mine.initial or not theirs.initial
                                 or mine.initial == theirs.initial))):
            return 2
        if any(_like(a, b) for a in mine.sure for b in theirs.sure):
            likeness = 1
    return likeness


def _records(work: Work, down: set[str]) -> list[Reference]:
    """The records of `work` in every catalog that answers, in the order found.

    A catalog that does not answer twice running is added to `down` and not asked again
    in this run; a record that cannot be read is passed over.
    """
    matching = []
    for name, (search, read) in CATALOGS.items():
        if name in down:
            continue
        try:
            try:
                found = list(search(work))
            except _NO_ANSWER:  # a connection dropped once is not a catalog that is down
                found = list(search(work))
        except _NO_ANSWER as e:
            emit(f"  · references: {name} did not answer ({e}); not asked again in this run",
                 kind="warn")
            down.add(name)
            continue
        for item in found:
            try:
                record = read(item)
            except _NO_ANSWER:
                continue
            if _same_title(work.title, record.title) and _same_author(work.author, record.first):
                matching.append(record)
    return matching


def find(work: Work, workspace: Path, down: set[str]) -> Reference | None:
    """The catalog record of `work`, or None when no catalog has it.

    Catalogs hold several editions and reprints of a work, and namesakes of its
    author: a record under the very surname is preferred to one spelled nearly the
    same, then one with the very title to one that adds or lacks a subtitle, then the
    one nearest in year to the year the model gave. Nothing is remembered while a
    catalog is down, so a later run asks them all again.
    """
    key = cache.cache_key("reference", work.author, work.title, work.year)
    cached = cache.load(workspace, key)
    if cached == "":
        return None
    if cached is not None:
        try:
            record = Reference(**json.loads(cached))
        except (ValueError, TypeError):
            record = None  # kept in another shape: looked up again
        if record and _same_title(work.title, record.title) and _same_author(
                work.author, record.first):
            return record  # else: kept by a version that accepted more than this one does

    def distance(record: Reference) -> tuple[int, bool, int]:
        apart = (abs(int(record.year) - int(work.year))
                 if work.year and record.year.isdigit() else 10_000)
        return (-_same_author(work.author, record.first),
                _plain(record.title) != _plain(work.title), apart)

    chosen = min(_records(work, down), key=distance, default=None)
    if not down:
        cache.store(workspace, key, f"reference: {work.title}",
                    json.dumps(asdict(chosen)) if chosen else "")
    return chosen


def _escaped(text: str) -> str:
    """Catalog text as Markdown text: nothing in it is emphasis, a link or a tag."""
    return _MARKUP.sub(r"\\\1", text)


def cite(reference: Reference) -> str:
    """One Markdown line: Authors (year). *Title*. Source. <link>"""
    authors = ", ".join(reference.authors[:MAX_AUTHORS])
    if len(reference.authors) > MAX_AUTHORS:
        authors += " et al."
    lead = " ".join(part for part in (
        _escaped(authors), f"({reference.year})" if reference.year else "") if part)
    parts = [lead, f"*{_escaped(reference.title.rstrip('.'))}*", _escaped(reference.source)]
    return ". ".join(part.rstrip(".") for part in parts if part) + f". <{reference.url}>"


_WORD = re.compile(r"[^\W\d_]+")
_SENTENCE_END = ".?!¿¡:…"
_OPENING = "\"'“”‘’«»([{—–->*_ \t"  # what may stand between the end of a sentence and its first word


def _spoken_surnames(*names: str) -> list[str]:
    """The one-word surnames people would say or write for the authors in `names`."""
    return list(dict.fromkeys(
        surname for name in names for person in _people(name) for surname in _name(person).sure
        if " " not in surname and surname not in _PARTICLES))


def _named(text: str, surnames: list[str]) -> list[tuple[int, bool, int]]:
    """Where `text` names one of `surnames`: (which surname, whether the word opens a
    sentence or a line, where the word ends).

    Only a word written with a capital is a name: "said", "flores" and "walked" are
    not Said, Flores and Walker. Whisper's spelling of a name counts ("Vigotski"). A
    capital that opens a sentence proves nothing when the text also has the word in
    small letters ("Young children"), and no capital does in a text written in capitals.
    """
    words = list(_WORD.finditer(text))
    if sum(w.group(0).isupper() for w in words) > len(words) / 2:
        return []
    ordinary = {_plain(w.group(0)) for w in words if not w.group(0)[0].isupper()}
    found = []
    for word in words:
        plain = _plain(word.group(0))
        which = next((i for i, surname in enumerate(surnames)
                      if plain == surname or _like(surname, plain, _HEARD_RATIO)), None)
        if which is None or not word.group(0)[0].isupper():
            continue
        before = text[:word.start()].rstrip(_OPENING)
        opens = not before or before[-1] in _SENTENCE_END + "\n"
        if not (opens and plain in ordinary):
            found.append((which, opens, word.end()))
    return found


def said(work: Work, reference: Reference, transcript: str) -> bool:
    """Whether the transcript refers to the work: by its author's surname, or by a
    title long enough not to be merely what its subject is called.

    A surname is one the model proposed, or the family name the catalog records.
    """
    title = _plain(_main_title(work.title)).split()
    if len(title) >= _TITLE_ALONE_WORDS and f" {' '.join(title)} " in f" {_plain(transcript)} ":
        return True
    family = reference.first if "," in reference.first else ""  # a catalog that says which it is
    proposed = _spoken_surnames(work.author)
    recorded = [theirs for theirs in _spoken_surnames(reference.first)
                if family or any(_like(mine, theirs) for mine in proposed)]
    return bool(_named(transcript, list(dict.fromkeys([*proposed, *recorded]))))


# What a mark must not go into: the address of a link, a tag, a bare link, mathematics.
_NOT_TEXT = re.compile(
    r"\]\([^)\n]*\)|<[^>\n]+>|https?://\S+|\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
_RULE = r"\s{0,3}(=+|-+|([-*_]\s*){3,})\s*$"  # a rule, or the line under a heading
# Lines a footnote mark cannot go on: a heading, a table row, a rule, the definition of
# a link, an empty line.
_NO_NOTE_LINE = re.compile(rf"^\s*(#|\||$)|^{_RULE}|^\s{{0,3}}\[[^\]]+\]:")
_UNDERLINE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_MARKS_BEFORE = re.compile(r"(?:,?\[\^\d+\])*$")
_AFTER = re.compile(r"(?:\]\([^)\n]*\))?((?:,?\[\^\d+\])*)")  # the rest of a link, marks there


def mark(body: str, number: int, work: Work) -> str:
    """`body` with footnote `number` where the work comes up.

    After the author's surname (as `_named` finds it, preferably not opening a
    sentence), or else after the title; never inside code, a link's address, a
    heading, a table, a rule or mathematics, and after a link or a mark already
    there. Failing both, at the end of the last line of ordinary text.
    """
    lines = body.split("\n")
    below = [*lines[1:], ""]
    text = "\n".join(  # the body with everything a mark must not go into blanked out
        " " * len(line)
        if fenced or _NO_NOTE_LINE.match(line) or (line.strip() and _UNDERLINE.match(under))
        else line
        for (line, fenced), under in zip(with_fence_state(lines), below, strict=True))
    for pattern in (_CODE, _NOT_TEXT):
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)

    named = _named(text, _spoken_surnames(work.author))
    titled = re.search(rf"(?<!\w){re.escape(work.title)}(?!\w)", text, re.IGNORECASE)
    at = min(named)[2] if named else titled.end() if titled else None
    if at is None:
        last = max((i for i, line in enumerate(text.split("\n")) if line.strip()), default=None)
        if last is None:
            return f"{body.rstrip()}\n\n[^{number}]"
        at = len("\n".join(lines[:last + 1]).rstrip())
        at -= len(_MARKS_BEFORE.search(body[:at]).group(0))
    after = _AFTER.match(body, at)
    # Two marks side by side print as one number: "12".
    return f"{body[:after.end()]}{',' if after.group(1) else ''}[^{number}]{body[after.end():]}"


def bibliography(workspace: Path, mentioned: list[tuple[dict, str, list[Work]]],
                 suggested: list[Work]) -> tuple[dict, bool]:
    """What the notes print about references, and whether every lookup could be made.

    `mentioned` gives, for each section, its record, its transcript and the works the
    model says it refers to. A work found in a catalog and named in that transcript
    is cited, and the record's body gets its footnote mark; one found but not named
    there joins `suggested`, the works on the subject that nobody mentioned. Returns
    `{"cited": [{number, citation, sections}], "further": [citation]}`.
    """
    down: set[str] = set()
    missing: set[tuple[str, str]] = set()

    def looked_up(work: Work) -> Reference | None:
        reference = find(work, workspace, down)
        if reference is None and not down:
            missing.add((_plain(work.author), _plain(work.title)))
        return reference

    cited: dict[str, dict] = {}  # by link: a work mentioned in two sections is one entry
    further: dict[str, str] = {}
    for record, transcript, works in mentioned:
        for work in works:
            reference = looked_up(work)
            if reference is None:
                continue
            if not said(work, reference, transcript):
                further[reference.url] = cite(reference)
                continue
            entry = cited.setdefault(reference.url, {
                "number": len(cited) + 1, "citation": cite(reference), "sections": []})
            if record["number"] not in entry["sections"]:
                entry["sections"].append(record["number"])
                record["body"] = mark(record["body"], entry["number"], work)
    for work in suggested:
        reference = looked_up(work)
        if reference is not None:
            further[reference.url] = cite(reference)
    if missing:
        emit(f"  · references: {len(missing)} proposed work(s) not found in "
             f"{' or '.join(CATALOGS)}; left out", kind="info")
    listed = {"cited": list(cited.values()),
              "further": [text for url, text in further.items() if url not in cited]}
    return listed, not down
