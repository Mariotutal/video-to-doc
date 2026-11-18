"""`--references`: works proposed by the model, cited only when a catalog has them."""
import http.client
import json
import urllib.parse

import pytest

from vtd import ai_synth, cli, pipeline, references
from vtd.ai.base import Work, _listed, _no_works, _works
from vtd.config import CONTENT_TYPES, AIConfig, VTDConfig, resolve_config
from vtd.export_pdf import pdf_markdown
from vtd.manifest import Manifest
from vtd.references import Reference, bibliography, cite, find, mark, said

from .fakes import fake_anthropic, message

MIND = Work("Vygotsky, Lev", "Mind in Society", "1978")
KUHN = Work("Thomas Kuhn", "The Structure of Scientific Revolutions", "1962")


def _catalogs(monkeypatch, crossref=None, open_library=None):
    """Answer catalog searches from tables keyed by the title searched; returns the URLs asked.

    A table left out is a catalog that does not answer.
    """
    asked = []

    def get(url):
        asked.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        if "crossref" in url:
            if crossref is None:
                raise OSError("timed out")
            return {"message": {"items": crossref.get(query["query.bibliographic"][0], [])}}
        if open_library is None:
            raise OSError("timed out")
        return {"docs": open_library.get(query["title"][0], [])}

    monkeypatch.setattr(references, "_get_json", get)
    return asked


def _book(title, author, year=None, key="/works/OL1W"):
    """A record as Open Library returns it."""
    return {"key": key, "title": title, "author_name": [author], "first_publish_year": year}


def _paper(title, family, given="", year=None, doi="10.1/x", **more):
    """A record as Crossref returns it."""
    return {"DOI": doi, "title": [title], "author": [{"given": given, "family": family}],
            "issued": {"date-parts": [[year]]}, **more}


MIND_IN_OPEN_LIBRARY = {"Mind in Society": [
    _book("Mind in Society", "L. S. Vygotskiĭ", 1978, "/works/OL8298080W")]}
KUHN_IN_CROSSREF = {KUHN.title: [_paper(
    KUHN.title, "Kuhn", "Thomas S.", 1962, "10.7208/chicago/9780226458106.001.0001",
    publisher="University of Chicago Press")]}
KUHN_FOUND = Reference(
    ["Thomas S. Kuhn"], "Kuhn, Thomas S.", "1962", KUHN.title, "University of Chicago Press",
    "https://doi.org/10.7208/chicago/9780226458106.001.0001")


# ---- what the model proposes -----------------------------------------------------------

def test_works_are_read_one_per_line_and_incomplete_ones_are_left_out():
    answer = (
        "- Vygotsky, Lev :: Mind in Society :: 1978\n"
        "- **Thomas Kuhn** :: «The Structure of Scientific Revolutions» :: ?\n"
        "- Piaget :: :: 1950\n"
        "- :: A title without an author :: 2001\n"
        "- Autor :: Título :: Año\n"
        "- Desconocido :: Poema de mio Cid :: 1200\n"
        "- Wu :: [Artículo de 2017, título no especificado] :: 2017\n"
        "Some remark that is not a work\n"
    )
    assert _works(answer) == [MIND, Work("Thomas Kuhn", KUHN.title, "")]
    assert _works("NONE") == []


@pytest.mark.parametrize("line, work", [
    ("- Stroustrup :: C++ :: The Language :: 1985", Work("Stroustrup", "C++: The Language", "1985")),
    ("- Vygotsky :: Mind in Society :: 1978/1934", Work("Vygotsky", "Mind in Society", "1934")),
    ("- Kuhn :: The Essential Tension", Work("Kuhn", "The Essential Tension", "")),
    ("- Author: Kuhn :: Title: The Essential Tension :: Year: 1977",
     Work("Kuhn", "The Essential Tension", "1977")),
    ("- Orwell :: 1984 :: 1949", Work("Orwell", "1984", "1949")),
    ("1. Kuhn, Thomas :: The Essential Tension :: 1977",
     Work("Kuhn, Thomas", "The Essential Tension", "1977")),
    ("2) Plato :: Republic :: c. 375 BC", Work("Plato", "Republic", "")),
    ("- Tacitus :: Annals :: 116", Work("Tacitus", "Annals", "")),
    ("- Euclid :: Elements :: n.d.", Work("Euclid", "Elements", "")),
    ("- Kuhn :: The Road :: Since Structure", Work("Kuhn", "The Road: Since Structure", "")),
    ("- Kuhn, Thomas :: The Essential Tension :: 1977 :: University of Chicago Press",
     Work("Kuhn, Thomas", "The Essential Tension", "1977")),
    ("- Title: The Essential Tension :: Author: Kuhn, Thomas :: Year: 1977",
     Work("Kuhn, Thomas", "The Essential Tension", "1977")),
    ("- Kuhn, Thomas :: The Essential Tension (1977)",
     Work("Kuhn, Thomas", "The Essential Tension", "1977")),
    ("- Kuhn, Thomas :: 1977 :: The Essential Tension",
     Work("Kuhn, Thomas", "The Essential Tension", "1977")),
])
def test_a_title_may_hold_the_separator_and_the_first_year_is_the_one_kept(line, work):
    assert _works(line) == [work]


@pytest.mark.parametrize("answer, none", [
    ("NONE", True), ("Ninguna.", True), ("NINGUNO", True),
    ("NONE — the excerpt names no published work.", True),
    ("No se menciona ninguna obra publicada en el fragmento.", True),
    ("- NONE", True), ("**NONE**", True), ("N/A", True), ("Nothing.", True), ("Sin obras.", True),
    ("There are no published works referred to in this excerpt.", True),
    ("Ningún trabajo publicado.", True),
    ("", False),  # nothing said: asked again, not kept as "no works"
    ("- Kahneman, Thinking, Fast and Slow (2011)", False),  # a list in the wrong format
    ("- No Logo, Naomi Klein (1999)", False),
    ("No works.\n- Kuhn, Thomas — The Structure of Scientific Revolutions (1962)", False),
    ("- None :: Of the above :: 1990", False),
    ("None.\nKuhn, Thomas — The Structure of Scientific Revolutions (1962)", False),
    ("The speaker mentions Kuhn's famous book on paradigms.", False),
    ("The excerpt does not refer to any published work.", True),
    ("No books or papers are referred to in this excerpt.", True),
    ("No puedo ayudar con esa solicitud.", True),  # asked again, it would refuse again
])
def test_no_works_is_understood_in_other_words_but_not_from_silence(answer, none):
    assert _no_works(answer) is none


def _section(text="como decía Vigotski"):
    return ai_synth.SectionInput(index=0, title=None, start_s=0, end_s=60,
                                 transcript_chunk=text, frame_paths=[])


def test_the_section_prompt_asks_for_works_referred_to_and_accepts_none(tmp_path):
    sent = []

    def create(**kwargs):
        sent.append(kwargs["messages"][0]["content"][-1]["text"])
        return message("Ninguna.")

    provider = fake_anthropic(tmp_path, create, language="es", content_type="lecture")
    assert provider.section_references(_section()) == []
    assert "AUTHOR :: TITLE :: YEAR" in sent[0] and "como decía Vigotski" in sent[0]
    assert provider.degraded == [] and len(sent) == 1


def test_titles_in_another_language_than_the_notes_are_not_asked_for_again(tmp_path):
    english = "\n".join(  # no list marks: read by the language check as the model's prose
        f"Author{n}, Ann :: The history of the theory of the mind and of what it is for :: 19{n}0"
        for n in range(1, 6))
    sent = []

    def create(**kwargs):
        sent.append(1)
        return message(english)

    provider = fake_anthropic(tmp_path, create, language="es")
    assert len(provider.section_references(_section())) == 5
    assert len(sent) == 1 and provider.degraded == []


@pytest.mark.parametrize("answer", [
    "- Anónimo :: Lazarillo de Tormes :: 1554",
    "- Piaget, Jean :: [his book on child development] :: ?",
])
def test_a_list_whose_lines_are_all_left_out_is_an_answer_not_a_gap(tmp_path, answer):
    sent = []

    def create(**kwargs):
        sent.append(1)
        return message(answer)

    provider = fake_anthropic(tmp_path, create, language="es")
    assert _listed(answer) and provider.section_references(_section()) == []
    assert len(sent) == 1 and provider.degraded == []
    assert not _listed("- AUTHOR :: TITLE :: YEAR")  # the format given back unfilled


@pytest.mark.parametrize("answer", ["", "The speaker mentions Kuhn's famous book on paradigms."])
def test_an_answer_that_cannot_be_read_is_asked_for_again_each_run_but_is_not_a_gap(
        tmp_path, answer):
    sent = []

    def create(**kwargs):
        sent.append(1)
        return message(answer)

    provider = fake_anthropic(tmp_path, create, language="es")
    assert provider.section_references(_section()) == []
    assert provider.degraded == []  # the notes are whole without the references of a section
    asked = len(sent)
    assert asked > 1 and provider.section_references(_section()) == [] and len(sent) > asked


# ---- what a catalog confirms -----------------------------------------------------------

@pytest.mark.parametrize("wanted, found, same", [
    ("Mind in Society", "Mind in society", True),
    ("Thinking, Fast and Slow", "Thinking fast and slow", True),
    ("Mind in Society", "Mind in Society: The Development of Higher Psychological Processes", True),
    ("Sapiens: A Brief History of Humankind", "Sapiens", True),
    ("Sapiens", "Sapiens - de animales a dioses", True),
    ("Pedagogía del oprimido", "Pedagogia del oprimido", True),
    ("The Art of Computer Programing", "The Art of Computer Programming", True),  # a slip
    ("Judgment under Uncertainty: Heuristics and Biases", "Judgement under Uncertainty", True),
    ("Organizational Behavior", "Organizational Behaviour", True),
    ("Principles of Microeconomics", "Principles of Macroeconomics", False),
    ("Evolution", "Revolution", False),
    ("The Conscious Mind", "The Unconscious Mind", False),
    ("Symmetric Information", "Asymmetric Information", False),
    ("Theism", "Atheism", False),
    ("Reformation", "Deformation", False),
    ("The Structure of Scientific Revolutions", "Structure of Scientific Revolutions", True),
    ("Crime and Punishment", "Crime & Punishment", True),
    ("Civilization and Its Discontents", "Civilisation and Its Discontents", True),
    ("The Color Purple", "The Colour Purple", True),
    ("Frankenstein; or, The Modern Prometheus", "Frankenstein", True),
    ("Psychometric Theory (3rd edition)", "Psychometric theory", True),
    ("Mind in Society", "Mind and Society", False),
    ("Mind", "Mind in Society", False),  # not set apart as a subtitle
    ("Psychometric Theory", "[Psychometric Theory] [by", False),
    ("The Art of Computer Programming, Volume 1", "The Art of Computer Programming, Volume 3",
     False),
    ("The Feynman Lectures on Physics, Vol. I", "The Feynman Lectures on Physics, Vol. II", False),
    ("A History of Western Philosophy", "A History of Eastern Philosophy", False),
    ("Lectures: Volume 1", "Lectures: Volume 2", False),
    ("", "Mind in Society", False),
])
def test_titles_match_word_for_word_or_by_leaving_out_a_subtitle(wanted, found, same):
    assert references._same_title(wanted, found) is same


@pytest.mark.parametrize("wanted, first, same", [
    ("Lev Vygotsky", "L. S. Vygotskiĭ", True),
    ("Vygotsky, Lev", "Vygotsky, L. S.", True),
    ("Kahneman", "Daniel Kahneman", True),
    ("Kahneman and Tversky", "Tversky, Amos", True),
    ("Lord & Novick", "Lord, Frederic M.", True),
    ("Tim Wu", "Wu, T.", True),
    ("Humberto Maturana", "Humberto Maturana Romesin", True),
    ("García Márquez", "Garcia Marquez, Gabriel", True),
    ("Gabriel García Márquez", "Gabriel García", True),
    ("Simone de Beauvoir", "Beauvoir, Simone de", True),
    ("World Health Organization", "World Health Organization", True),
    ("Le Guin, Ursula K.", "Ursula K. Le Guin", True),
    ("Liu Cixin", "Cixin Liu", True),                     # the family name written first
    ("Haruki Murakami", "Murakami Haruki", True),
    ("Quoc V. Le", "Le, Quoc V.", True),
    ("Hernández", "Roberto Hernández Sampieri", True),    # the first of two surnames
    ("Hernández et al.", "Roberto Hernández Sampieri", True),
    ("Cervantes", "Miguel de Cervantes Saavedra", True),
    ("Kurt Vonnegut Jr.", "Vonnegut, Kurt", True),
    ("World Health Organization", "World Trade Organization", False),
    ("Harvard University", "Stanford University", False),
    ("Kurt Vonnegut Jr.", "Robert Downey Jr.", False),
    ("Hernández, Ana", "Fernández, Ana", False),          # alike, but two names
    ("Roberts", "Robertson, Ian", False),
    ("John Stuart Mill", "Stuart, Ian", False),           # a middle name, another person
    ("Larry David", "Henry David Thoreau", False),
    ("Daniel Kahneman", "Daniela Molas", False),          # a first name in common
    ("James MacGregor Burns", "William James", False),    # a given name that is a surname
    ("Peter Chang", "Aston Zhang", False),                # too short to be a slip of spelling
    ("Georg Rasch", "Wolins, Leroy", False),
    ("Lev Vygotsky", "", False),
])
def test_an_author_matches_by_surname_only(wanted, first, same):
    assert bool(references._same_author(wanted, first)) is same


def test_a_review_of_the_book_is_not_the_book(tmp_path, monkeypatch):
    rasch = Work("Rasch, Georg", "Probabilistic Models for Some Intelligence and Attainment Tests",
                 "1960")
    review = _paper(rasch.title + ".", "Wolins", "Leroy", 1982)
    review["author"].append({"given": "Georg", "family": "Rasch"})  # the author reviewed
    _catalogs(monkeypatch, crossref={rasch.title: [review]}, open_library={})
    assert find(rasch, tmp_path, set()) is None


def test_every_catalog_is_asked_and_the_answer_is_kept_in_the_workspace(tmp_path, monkeypatch):
    asked = _catalogs(
        monkeypatch,
        open_library={KUHN.title: [_book(KUHN.title, "Ian Hacking")]},  # a book about the book
        crossref=KUHN_IN_CROSSREF,
    )
    assert find(KUHN, tmp_path, set()) == KUHN_FOUND
    assert len(asked) == 2

    assert find(KUHN, tmp_path, set()) == KUHN_FOUND  # read back from the workspace
    assert len(asked) == 2


def test_catalogs_are_searched_by_the_surname_of_the_first_author(tmp_path, monkeypatch):
    asked = _catalogs(monkeypatch, crossref={}, open_library={})
    find(Work("Lord, F. Mosteller & Novick, Melvin R", "Statistical Theories"), tmp_path, set())
    find(Work("Jacob Cohen and Michael Swerdlik", "Psychological Testing"), tmp_path, set())
    authors = [urllib.parse.parse_qs(urllib.parse.urlsplit(url).query) for url in asked]
    assert [q.get("author") or q.get("query.author") for q in authors] == [
        ["lord"], ["lord"], ["cohen"], ["cohen"]]


def test_among_records_the_very_title_and_the_nearest_year_are_preferred(tmp_path, monkeypatch):
    def years(work):
        return find(work, tmp_path, set()).year

    _catalogs(monkeypatch, crossref=KUHN_IN_CROSSREF, open_library={KUHN.title: [
        _book(KUHN.title + ": 50th Anniversary Edition", "Thomas S. Kuhn", 1962, "/works/OL3W"),
        _book(KUHN.title, "Thomas S. Kuhn", 1996, "/works/OL2W")]})
    assert years(KUHN) == "1962"
    assert find(KUHN, tmp_path, set()).url.startswith("https://doi.org/")  # the very title
    assert years(Work("Thomas Kuhn", KUHN.title, "1995")) == "1996"
    assert years(Work("Thomas Kuhn", KUHN.title)) == "1996"  # no year to go by: the first found


def test_the_very_surname_is_preferred_to_one_spelled_nearly_the_same(tmp_path, monkeypatch):
    work = Work("Johnson, Mark", "Metaphors We Live By", "1980")
    _catalogs(monkeypatch, crossref={}, open_library={work.title: [
        _book(work.title, "Mary Johnston", 1980, "/works/OTHER"),
        _book(work.title, "Mark Johnson", 2003, "/works/RIGHT")]})
    assert find(work, tmp_path, set()).url.endswith("/works/RIGHT")


@pytest.mark.parametrize("kept", [
    json.dumps(Reference(["Mary Wells"], "Mary Wells", "2020", "Principles of Macroeconomics", "",
                         "u").__dict__),
    json.dumps({"authors": ["Mary Wells"], "title": "Principles of Microeconomics"}),  # older shape
    "not json",
])
def test_a_record_kept_by_other_rules_or_in_another_shape_is_looked_up_again(
        tmp_path, monkeypatch, kept):
    from vtd.ai import cache
    work = Work("Wells", "Principles of Microeconomics")
    cache.store(tmp_path, cache.cache_key("reference", work.author, work.title, work.year), "x", kept)
    asked = _catalogs(monkeypatch, crossref={}, open_library={})
    assert find(work, tmp_path, set()) is None and len(asked) == 2


def test_nothing_is_kept_while_a_catalog_is_down_so_the_better_record_can_come_later(
        tmp_path, monkeypatch):
    reprint = {KUHN.title: [_paper(KUHN.title, "Kuhn", "Thomas S.", 2012, "10.1/reprint")]}
    _catalogs(monkeypatch, crossref=reprint)  # Open Library does not answer
    assert find(KUHN, tmp_path, set()).year == "2012"
    _catalogs(monkeypatch, crossref=reprint, open_library={KUHN.title: [
        _book(KUHN.title, "Thomas S. Kuhn", 1962)]})
    assert find(KUHN, tmp_path, set()).year == "1962"


def test_a_work_no_catalog_has_is_not_cited_and_not_searched_twice(tmp_path, monkeypatch):
    asked = _catalogs(monkeypatch, crossref={}, open_library={})
    assert find(MIND, tmp_path, set()) is None
    assert find(MIND, tmp_path, set()) is None
    assert len(asked) == 2


def test_a_catalog_that_fails_is_asked_once_and_the_work_is_looked_up_again_later(
        tmp_path, monkeypatch, capsys):
    asked = _catalogs(monkeypatch, crossref=None, open_library={})
    down = set()
    assert find(MIND, tmp_path, down) is None
    assert find(KUHN, tmp_path, down) is None
    assert down == {"Crossref"}
    assert sum("crossref" in url for url in asked) == 2  # asked again once, then left alone
    assert "Crossref did not answer" in capsys.readouterr().err

    _catalogs(monkeypatch, crossref=KUHN_IN_CROSSREF, open_library={})  # back up
    assert find(KUHN, tmp_path, set()) == KUHN_FOUND


def test_a_connection_dropped_once_does_not_put_a_catalog_down(tmp_path, monkeypatch):
    failures = [OSError("connection reset by peer")]
    asked = _catalogs(monkeypatch, crossref=KUHN_IN_CROSSREF, open_library={})
    answering = references._get_json

    def flaky(url):
        if failures:
            raise failures.pop()
        return answering(url)

    monkeypatch.setattr(references, "_get_json", flaky)
    down = set()
    assert find(KUHN, tmp_path, down) == KUHN_FOUND
    assert down == set() and len(asked) == 2


@pytest.mark.parametrize("failure", [
    OSError("timed out"), http.client.IncompleteRead(b""), ValueError("not JSON")])
def test_a_connection_cut_or_a_page_that_is_not_json_is_a_catalog_that_does_not_answer(
        tmp_path, monkeypatch, failure):
    def get(url):
        raise failure

    monkeypatch.setattr(references, "_get_json", get)
    down = set()
    assert find(MIND, tmp_path, down) is None
    assert down == set(references.CATALOGS)


def test_an_answer_of_another_shape_is_a_catalog_that_does_not_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(references, "_get_json", lambda url: {"unexpected": True})
    down = set()
    assert find(MIND, tmp_path, down) is None
    assert down == set(references.CATALOGS)


def test_a_record_that_cannot_be_read_is_passed_over_and_the_others_are_used(
        tmp_path, monkeypatch):
    good = KUHN_IN_CROSSREF[KUHN.title][0]
    odd = [{"DOI": "10.1/a", "title": [KUHN.title], "author": None, "issued": None},
           {"title": [KUHN.title], "author": [{"family": "Kuhn"}]},  # no DOI
           {"DOI": "10.1/b", "title": [KUHN.title], "author": [{"family": "Kuhn"}],
            "issued": {"date-parts": [[]]}},
           good]
    _catalogs(monkeypatch, crossref={KUHN.title: odd}, open_library={KUHN.title: [{"title": 3}]})
    down = set()
    assert find(KUHN, tmp_path, down) == KUHN_FOUND
    assert down == set()


# ---- how it is written -----------------------------------------------------------------

def test_a_citation_is_written_from_the_record_with_nothing_in_it_read_as_markdown():
    reference = Reference(["Ann Lee", "Bo *Li*", "Cy Ray", "Di Poe"], "Lee, Ann", "2001",
                          "A [new] theory_of <mind>.", "Journal of_Things.",
                          "https://doi.org/10.1/x")
    assert cite(reference) == (
        r"Ann Lee, Bo \*Li\*, Cy Ray et al. (2001). *A \[new\] theory\_of \<mind\>*. "
        r"Journal of\_Things. <https://doi.org/10.1/x>")
    assert cite(Reference([], "", "", "Anonymous work", "", "https://x.test/1")) == (
        "*Anonymous work*. <https://x.test/1>")


def test_crossref_records_are_read_as_plain_text_with_a_link_that_stays_whole():
    item = _paper("On H<sub>2</sub>O", "Poe", "Ada",
                  doi="10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9",
                  **{"container-title": ["Water  Letters"]})
    item["author"].append({"name": "The Water Consortium"})
    found = references._from_crossref(item)
    assert found == Reference(
        ["Ada Poe", "The Water Consortium"], "Poe, Ada", "", "On H2O", "Water Letters",
        "https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9")
    assert "<" not in cite(found)[:-1].rsplit("<", 1)[1]


def _found(first):
    return Reference([first], first, "", "A title", "", "u")


@pytest.mark.parametrize("work, first, transcript, heard", [
    (MIND, "L. S. Vygotskiĭ", "como decía Vigotski sobre el aprendizaje", True),  # as Whisper wrote it
    (MIND, "L. S. Vygotskiĭ", "hablamos de la zona de desarrollo próximo", False),  # the idea only
    (MIND, "L. S. Vygotskiĭ", "y Lev levantó la mano", False),               # a given name
    (MIND, "L. S. Vygotskiĭ", "lo explica en mind in society, el libro", False),  # a short title
    (KUHN, "Thomas S. Kuhn", "en the structure of scientific revolutions se dice", True),
    (Work("Goodfellow, Ian", "Deep Learning"), "Ian Goodfellow",
     "Today we continue with deep learning.", False),                        # the subject's name
    (Work("Goodfellow, Ian", "Deep Learning"), "Ian Goodfellow",
     "as Goodfellow puts it in deep learning", True),
    (Work("Harari, Yuval", "Sapiens: A Brief History"), "Yuval Harari",
     "hoy comentamos sapiens, el libro", False),
    # A surname is a word with a capital, and not an ordinary word that opens a sentence.
    (Work("Said, Edward", "Orientalism"), "Edward Said", "As I said, colonial powers", False),
    (Work("Flores, Ana", "Un libro"), "Ana Flores", "había flores en la mesa", False),
    (Work("Walker, Alice", "A book"), "Alice Walker", "they walked faster", False),
    (Work("Young, Michael", "A book"), "Michael Young",
     "Young children learn. The young imitate.", False),
    (Work("Young, Michael", "A book"), "Michael Young", "Young wrote that in 1958.", True),
    (Work("Stevens, S. S.", "A paper"), "S. S. Stevens", "medición. Stevens en 1946 propone", True),
    (Work("Le Guin, Ursula K.", "A novel"), "Ursula K. Le Guin", "Le dije al grupo", False),
    (Work("John Stuart Mill", "On Liberty"), "John Stuart Mill", "our guest Stuart said", False),
    # The catalog's first author is another spelling of the one proposed, not another person,
    # unless the catalog says which words are the family name.
    (Work("Márquez, Gabriel", "Cien años"), "Gabriel García Márquez", "como dijo García", False),
    (Work("Miguel de Cervantes Saavedra", "Don Quijote"), "Cervantes Saavedra, Miguel de",
     "como escribió Cervantes en la novela", True),
    # Capitals say nothing in a text written in capitals, nor after a quote mark or a new line
    # when the word is also there in small letters.
    (Work("Said, Edward", "Orientalism"), "Edward Said", "AS I SAID, COLONIAL POWERS DREW IT", False),
    (Work("Young, Michael", "A book"), "Michael Young",
     'she asked: "Young people?"\nYoung or old, the young imitate.', False),
])
def test_a_work_is_referred_to_when_its_author_or_a_long_title_is_in_the_transcript(
        work, first, transcript, heard):
    assert said(work, _found(first), transcript) is heard


@pytest.mark.parametrize("body, work, marked", [
    ("Según Vygotsky, se aprende con otros.", MIND, "Según Vygotsky[^2], se aprende con otros."),
    ("Cita «Mind in Society» sin decir de quién.", MIND,
     "Cita «Mind in Society[^2]» sin decir de quién."),
    ("Se habla del aprendizaje.\n", MIND, "Se habla del aprendizaje.[^2]\n"),
    ("Levantó la mano.", MIND, "Levantó la mano.[^2]"),
    # Only the surname, and only with its capital.
    ("Thomas presented; Ana cited Kuhn.", KUHN, "Thomas presented; Ana cited Kuhn[^2]."),
    ("She said the West invented it, as Orientalism argued.", Work("Edward Said", "Orientalism"),
     "She said the West invented it, as Orientalism[^2] argued."),
    ("**Daniel:** read the book on two systems.", Work("Daniel Kahneman", "Thinking, Fast and Slow"),
     "**Daniel:** read the book on two systems.[^2]"),
    # Not where a mark would not be one.
    ("See `Knuth` shuffle and [Knuth](https://knuth.test/Knuth) here.", Work("Donald Knuth", "TAOCP"),
     "See `Knuth` shuffle and [Knuth](https://knuth.test/Knuth)[^2] here."),
    ("##### Kuhn\n\nParadigms.\n\n| Kuhn | 1962 |", KUHN, "##### Kuhn\n\nParadigms.[^2]\n\n| Kuhn | 1962 |"),
    ("Example:\n\n```python\nprint(1)  # Kuhn\n```", KUHN,
     "Example:[^2]\n\n```python\nprint(1)  # Kuhn\n```"),
    ("| a | b |", KUHN, "| a | b |\n\n[^2]"),
    ("See <https://example.com/Kuhn> and https://example.com/Kuhn for more.", KUHN,
     "See <https://example.com/Kuhn> and https://example.com/Kuhn for more.[^2]"),
    ("Intro text.\n\n---", KUHN, "Intro text.[^2]\n\n---"),
    ("Text.\n\nKuhn\n===", KUHN, "Text.[^2]\n\nKuhn\n==="),
    ("Text.\n\n[ref]: http://x.test", KUHN, "Text.[^2]\n\n[ref]: http://x.test"),
    ("- Paradigms\n    - Kuhn calls them so\n- Other", KUHN,
     "- Paradigms\n    - Kuhn[^2] calls them so\n- Other"),  # a nested item is text, not code
    ("Alpha is $\\alpha_{Kuhn}$, as defined.", KUHN, "Alpha is $\\alpha_{Kuhn}$, as defined.[^2]"),
    # After a mark already there, and on the author rather than a word that opens a sentence.
    ("Kuhn[^1] argued that paradigms shift.", KUHN, "Kuhn[^1],[^2] argued that paradigms shift."),
    ("- one\n- two[^1]", KUHN, "- one\n- two[^1],[^2]"),
    ("Young people join. The young imitate.", Work("Michael Young", "Meritocracy"),
     "Young people join. The young imitate.[^2]"),
    ("Young children imitate, said Michael Young.", Work("Michael Young", "Meritocracy"),
     "Young children imitate, said Michael Young[^2]."),
    ("Dice Vigotski que se aprende con otros.", MIND, "Dice Vigotski[^2] que se aprende con otros."),
])
def test_the_footnote_goes_after_the_surname_or_the_title_or_ends_the_text(body, work, marked):
    assert mark(body, 2, work) == marked


def test_works_are_cited_once_and_one_not_named_in_the_transcript_is_further_reading(
        tmp_path, monkeypatch, capsys):
    baker = Work("Baker, Frank", "Item Response Theory")
    _catalogs(monkeypatch, crossref=KUHN_IN_CROSSREF, open_library={
        **MIND_IN_OPEN_LIBRARY, baker.title: [_book(baker.title, "Frank B. Baker", 1992, "/works/B")]})
    first = {"number": 1, "body": "Vygotsky y Kuhn."}
    second = {"number": 2, "body": "Otra vez Vygotsky."}
    unknown = Work("Nobody", "A book that does not exist", "")

    works, looked_up = bibliography(tmp_path, [
        (first, "dice Vigotski, y Kuhn con sus paradigmas", [MIND, KUHN, unknown]),
        (second, "otra vez Vygotsky, y la teoría de respuesta al ítem", [MIND, baker, unknown]),
    ], [KUHN, unknown])

    assert looked_up
    assert [(w["number"], w["sections"]) for w in works["cited"]] == [(1, [1, 2]), (2, [1])]
    assert works["further"] == [
        "Frank B. Baker (1992). *Item Response Theory*. <https://openlibrary.org/works/B>"]
    assert first["body"] == "Vygotsky[^1] y Kuhn[^2]."
    assert second["body"] == "Otra vez Vygotsky[^1]."
    assert "1 proposed work(s) not found" in capsys.readouterr().out  # one work, proposed thrice


# ---- in the notes ----------------------------------------------------------------------

def _workspace(tmp_path):
    (tmp_path / "transcript.txt").write_text("como decía Vigotski en mind in society")
    (tmp_path / "transcript.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:01:00.000\ncomo decía Vigotski en mind in society\n\n")
    (tmp_path / "sections.json").write_text(json.dumps([{
        "index": 0, "start_s": 0, "end_s": 60, "duration_s": 60, "is_silent": False,
        "cue_count": 1, "title": "Aprender con otros"}]))
    return Manifest(source_video="clase.mp4", duration_seconds=60, has_video=False)


def _answers(prompt: str) -> str:
    if "published works (books, scientific papers, reports)" in prompt:
        return "- Vygotsky, Lev :: Mind in Society :: 1978"
    if "Suggest up to 8 published works" in prompt:
        return "- Kuhn, Thomas :: The Structure of Scientific Revolutions :: 1962"
    if "TERM :: DEFINITION" in prompt or "ONE PER LINE" in prompt:
        return "NONE"
    if "level-3" in prompt:
        return "### Tema\n\nSe aprende con otros."
    return "La docente recuerda a Vygotsky y la zona de desarrollo próximo."


def _synthesize(tmp_path, monkeypatch, language="es", content_type="lecture", **options):
    manifest = _workspace(tmp_path)
    config = VTDConfig(output_language=language, content_type=content_type,
                       ai=AIConfig(model="test"), **options)
    provider = fake_anthropic(
        tmp_path, lambda **k: message(_answers(k["messages"][0]["content"][-1]["text"])),
        language=language, content_type=content_type)
    monkeypatch.setattr(ai_synth, "_build_provider", lambda config, workspace: provider)
    incomplete = ai_synth.synthesize(tmp_path, config, manifest)
    return (tmp_path / "notes.md").read_text(), incomplete


def test_notes_carry_footnotes_and_a_bibliography_of_what_the_catalogs_have(tmp_path, monkeypatch):
    _catalogs(monkeypatch, crossref=KUHN_IN_CROSSREF, open_library=MIND_IN_OPEN_LIBRARY)

    notes, incomplete = _synthesize(tmp_path, monkeypatch, references=True)

    assert incomplete == []
    vygotsky = "L. S. Vygotskiĭ (1978). *Mind in Society*. <https://openlibrary.org/works/OL8298080W>"
    assert "recuerda a Vygotsky[^1] y la zona" in notes
    assert notes.endswith(
        "| Aprender con otros |\n\n## Bibliografía\n\n"
        "Todas las obras de esta lista se encontraron en Crossref u Open Library y se citan "
        "como figuran allí; las lecturas complementarias las propuso el modelo de IA. vtd no "
        "ha comprobado lo que dicen las obras.\n\n"
        "### Obras a las que se hace referencia en la grabación\n\n"
        f"- {vygotsky} (§1)\n\n"
        "### Lecturas complementarias (no mencionadas en la grabación)\n\n"
        "- Thomas S. Kuhn (1962). *The Structure of Scientific Revolutions*. University of "
        "Chicago Press. <https://doi.org/10.7208/chicago/9780226458106.001.0001>\n\n"
        f"[^1]: {vygotsky}\n\n"
    )


@pytest.mark.parametrize("content_type", CONTENT_TYPES)
def test_the_bibliography_is_a_section_of_its_own_in_every_kind_of_document(
        tmp_path, monkeypatch, content_type):
    _catalogs(monkeypatch, crossref=KUHN_IN_CROSSREF, open_library=MIND_IN_OPEN_LIBRARY)
    notes, _incomplete = _synthesize(tmp_path, monkeypatch, "en", content_type, references=True)
    assert "\n\n## Bibliography\n\n" in notes and "\n\n\n## Bibliography" not in notes
    assert notes.count("[^1]") == 2

    without, _incomplete = _synthesize(tmp_path, monkeypatch, "en", content_type)
    assert notes.replace("Vygotsky[^1]", "Vygotsky").startswith(without.rstrip("\n"))


def test_with_nothing_found_the_bibliography_says_so_and_the_text_has_no_marks(
        tmp_path, monkeypatch):
    _catalogs(monkeypatch, crossref={}, open_library={})
    notes, incomplete = _synthesize(tmp_path, monkeypatch, references=True)
    assert incomplete == []
    assert "[^" not in notes
    assert notes.count("_No se detectó nada._") >= 2 and "## Bibliografía" in notes


def test_catalogs_that_do_not_answer_leave_the_notes_incomplete_for_a_resume(
        tmp_path, monkeypatch):
    _catalogs(monkeypatch)
    notes, incomplete = _synthesize(tmp_path, monkeypatch, references=True)
    assert incomplete == ["references"]
    assert "## Bibliografía" in notes


def test_without_the_option_nothing_is_asked_searched_or_added(tmp_path, monkeypatch):
    monkeypatch.setattr(references, "_get_json", lambda url: pytest.fail("searched a catalog"))
    monkeypatch.setattr("vtd.ai.base.ChatProvider.section_references",
                        lambda *a: pytest.fail("asked a section for its works"))
    monkeypatch.setattr("vtd.ai.base.ChatProvider.further_reading",
                        lambda *a: pytest.fail("asked for further reading"))
    notes, incomplete = _synthesize(tmp_path, monkeypatch)
    assert incomplete == []
    assert "Bibliografía" not in notes and "[^" not in notes
    assert notes.endswith("| Aprender con otros |\n\n")


def test_footnotes_reach_the_pdf_as_they_are_and_images_still_do_not(tmp_path):
    notes = ("# clase.mp4 — notas\n\nVygotsky[^1] dice. ¡Qué idea![^2] ![x](/etc/passwd) "
             "![^ y](/etc/hosts) ![^3](/etc/hosts)\n\n[^1]: L. Vygotsky (1978). *Mind*.\n")
    _title, _subtitle, body = pdf_markdown(notes, tmp_path, "clase.mp4")
    assert ("Vygotsky[^1] dice. ¡Qué idea![^2] !\\[x](/etc/passwd) "
            "!\\[^ y](/etc/hosts) !\\[^3](/etc/hosts)") in body
    assert "[^1]: L. Vygotsky (1978). *Mind*." in body


# ---- the option ------------------------------------------------------------------------

def test_the_option_comes_from_the_flag_or_the_config_file(tmp_path, monkeypatch):
    from vtd import config as config_module

    parse = cli.build_parser().parse_args
    assert resolve_config(parse(["v.mp4"])).references is False
    assert resolve_config(parse(["v.mp4", "--with-ai", "--references"])).references is True

    file = tmp_path / "config.toml"
    file.write_text("[defaults]\nreferences = true\n")
    monkeypatch.setattr(config_module, "CONFIG_PATH", file)
    assert resolve_config(parse(["v.mp4"])).references is True
    assert resolve_config(parse(["v.mp4", "--no-references"])).references is False


def test_asking_for_references_without_the_ai_notes_is_an_error_before_any_work(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_pipeline", lambda **k: pytest.fail("the pipeline ran"))
    source = tmp_path / "v.mp4"
    source.write_bytes(b"")
    assert cli.main([str(source), "--references"]) == 1
    assert "add --with-ai" in capsys.readouterr().err


def test_asking_for_the_ai_stage_by_name_is_enough_for_references(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "run_pipeline", lambda **k: ran.append(k["config"].references))
    source = tmp_path / "v.mp4"
    source.write_bytes(b"")
    assert cli.main([str(source), "--only", "synthesize", "--references", "--api-key", "k"]) == 0
    assert ran == [True]


def test_turning_references_on_or_off_redoes_the_notes_on_resume():
    def current(recorded, **options):
        manifest = Manifest(source_video="v.mp4")
        manifest.mark_stage("synthesize", recorded)
        return pipeline._is_current("synthesize", VTDConfig(**options), manifest)

    with_them = pipeline._stage_options("synthesize", VTDConfig(references=True))
    without = pipeline._stage_options("synthesize", VTDConfig())
    before_the_option = {k: v for k, v in without.items() if k != "references"}

    assert current(with_them, references=True) and current(without)
    assert not current(with_them) and not current(without, references=True)
    # Notes made by a version without the option were made without references.
    assert current(before_the_option) and not current(before_the_option, references=True)
    assert current({})  # a stage that recorded no options at all is still trusted
