# Spec: `--references`, a verified bibliography for the AI notes

- **Date:** 2026-11-17
- **Drafted by:** Mario Vargas (with Claude)
- **Status:** Implemented

## Context

The AI notes summarise what was said, and the follow-ups list readings the audience was asked to do. When a speaker names a researcher, a book or a paper, the notes repeat the name and nothing more: the reader has to find the work. A language model asked for a bibliography on its own writes citations that look right and do not exist.

## Goal

With `--references` (or `references = true`), `notes.md` and its PDF carry:

- a footnote where a section's summary brings up a work the speakers named or plainly referred to, with the citation at the foot of the PDF page;
- a **Bibliography** section listing those works with the sections that mention them, and, apart from them, further reading on the subjects covered that nobody mentioned.

Every work printed was found in a public catalog, and is printed as the catalog records it (authors, year, title, publisher or journal, DOI or link).

## Non-goals

- Does not print a work that was not found in a catalog, not even marked as unverified.
- Does not check that a work says what the speaker said it says.
- Does not download, read or summarise the works.
- Does not offer citation styles (APA, BibTeX export…).
- Does not touch the starter document: it has no AI, and the option needs `--with-ai`.

## Constraints

- **Cost ceiling:** one more model call per section and one for the whole document. Catalog lookups are free and need no key, but Crossref allows anonymous callers one request a second, so each work proposed adds about a second.
- **Coexistence:** off by default; with it off, prompts, cache keys and documents are unchanged.
- **Privacy:** author names and titles are sent to `api.crossref.org` and `openlibrary.org`, also with `--provider ollama`. Only the title and the first author's surname of each proposed work are sent; these can be names and titles said in the recording, but the transcript itself is not sent. Stated in the README and SECURITY.md.
- **Other:** no new dependency (standard-library `urllib`). OpenAlex was left out: it now asks for an API key.

## Plan

1. `--references` / `references` in the CLI and config; recorded among the options of the `synthesize` stage. Given on the command line without the AI stage, it is an error before any work.
2. Two provider prompts: the works referred to in a section's transcript, and further reading from the section summaries. Both answer `- AUTHOR :: TITLE :: YEAR` lines, or `NONE`.
3. `references.py`: look each proposed work up in Open Library and Crossref, by title and first author's surname. A record is the work when its title is the same word for word (a spelling variant or a subtitle left out apart) and its first author has the surname proposed; a surname spelled nearly the same counts only as a second choice, and a word that may be a middle name only when the given names begin alike. Among several records, prefer the very surname, then the very title, then the year nearest the one the model gave. Format the citation from the record; cache each lookup in the workspace, except while a catalog is down. A catalog that fails twice running is not asked again in that run, and the notes are reported as incomplete so `--resume` retries.
4. A work found is cited as referred to in the recording only when that section's transcript has its author's surname written with a capital (as Whisper may have spelled it; a word that opens a sentence counts only if the transcript never has it in small letters), or its title when the title without its subtitle is four words or more. Otherwise it is further reading. The footnote mark goes after the surname, or after the title, never inside code, a link, a heading, a table or a rule; failing both, at the end of the section's last line of text. One number per work across the document.
5. A `bibliography()` part shared by the four templates, with labels in English and Spanish. Footnotes are standard Markdown footnotes, which pandoc sets at the foot of the page.
6. README, CHANGELOG, AGENTS.md, SECURITY.md.

## Verification

- **Tests:** `tests/test_references.py` — answer parsing, title and author matching (near-miss titles, volumes, reviews, Spanish double surnames, given names), the choice among editions, citation formatting and escaping, lookup caching, catalogs that are down or answer oddly, the transcript check, mark placement, `synthesize` end to end with fake catalogs in the four kinds of document, the PDF text, and the option in the CLI, the config file and `--resume`.
- **Evals:** none automated. The prompts are judged by hand on real recordings (see open questions).
- **Smoke test:** done with a stand-in model: lookups of real and invented works against the live catalogs, and notes turned into a PDF with pandoc, the footnotes at the foot of the page. Then a run on a two-hour recording in Spanish with a local model (Ollama, `qwen3.6:35b`); what it showed is under Risks.

## Risks

Seen on a real run and on re-reading the code, and what was done about each:

- **The model proposes works that do not exist** — on that run, 20 of 27 proposals from the local model (invented titles, a test taken for a book, a description in brackets) / none was printed: no catalog had them.
- **The model adds a real work the speakers did not name** — seen (a textbook on a theory the speaker only described) / it was footnoted as referred to / now it must be named in the section's transcript, or it is further reading.
- **A review of a book is taken for the book** — seen twice (the reviewed authors are on the review's record) / now the first author of the record must be the one proposed.
- **A different work with a like title** — found on re-reading the code (another volume, one word changed, a longer title beginning the same way, a junk record) / now titles must agree word for word.
- **A different author with a name in common** — seen (a first name) and found on re-reading the code (a given name that is someone's surname) / now surnames only.
- **The model knows the author but not the exact title** — seen / the work is left out. Not solved: it depends on the model.
- **The catalog's record is of a reprint** — seen on live lookups (a 1978 book dated 1980, a 2017 paper under a 2025 deposit) / right work, wrong year or publisher / the record nearest the model's year is preferred; otherwise it is printed as it is.
- **A second pass over the fixed code** found the transcript check too trusting (a surname that is also a word, a title that is also the subject's name), titles one letter apart taken as one, surnames that look alike, images let through to the PDF by the footnote exception, and "no works" answers that were asked for three times / fixed, each with a test.
- **An answer of the model that is neither a list nor a no** — seen on that run (one section, every time) / the notes were reported incomplete for ever / the references of a section are not a part the notes are incomplete without: such an answer is asked for again on the next run and nothing else.
- **Known and left as they are:** a surname spelled nearly like the one proposed is accepted when no record has the very surname (Johnson, Johnston), which transliterated names need; a summary sentence that opens with a word that is also a surname may get the mark; two marks at one place are written `[^1],[^2]`.
- **A name Whisper writes in small letters, or a book named only by a short title** — expected sometimes / a work that was referred to is listed as further reading; the safer of the two errors.
- **A real work is in neither catalog** — likely for older non-English books / left out; the run says how many proposals were not found.
- **A catalog changes its API or limits anonymous use** — possible / the run reports it, `--resume` retries, and the rest of the notes is unaffected.

## Rollback plan

Turn the option off. It is off by default and writes nothing outside `notes.md` and the workspace cache.

## Open questions

- [ ] **Do the prompts hold on meetings, seminars and webinars?** It was tried on lectures only.
- [ ] **Should works not found be listed as unverified?** Decided no for now.

## References

- Crossref REST API: <https://api.crossref.org/swagger-ui/index.html>
- Open Library search API: <https://openlibrary.org/dev/docs/api/search>
- Pandoc footnotes: <https://pandoc.org/MANUAL.html#footnotes>
