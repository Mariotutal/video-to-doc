"""Splitting sections that are too long to be summarised whole."""
import json

from vtd import ai_synth
from vtd.config import VTDConfig
from vtd.manifest import Manifest
from vtd.stages import detect_sections
from vtd.stages.detect_sections import _split_points, cue_text, cues_in_section


def _cues(count, *, seconds=10.0, pause_after=None, text="x" * 99):
    """`count` cues of 100 characters each (with the line break), back to back.

    `pause_after` maps a cue number to the silence that follows it.
    """
    cues, t = [], 0.0
    for i in range(count):
        cues.append({"start_s": t, "end_s": t + seconds, "text": text})
        t += seconds + (pause_after or {}).get(i, 0.0)
    return cues


def _parts(cues, cuts):
    bounds = [0.0, *cuts, cues[-1]["end_s"] + 1]
    return [cues_in_section(cues, a, b, is_last=False) for a, b in zip(bounds, bounds[1:], strict=False)]


def test_a_section_that_fits_is_left_whole():
    assert _split_points(_cues(10), max_chars=1000) == []


def test_a_long_section_is_cut_so_that_every_part_fits_and_no_cue_is_lost():
    cues = _cues(47)
    parts = _parts(cues, _split_points(cues, max_chars=1000))
    assert len(parts) == 5
    assert all(len(cue_text(p)) <= 1000 for p in parts)
    assert [c for p in parts for c in p] == cues


def test_the_cut_is_made_at_the_longest_pause_that_still_fits():
    cues = _cues(15, pause_after={6: 4.0, 8: 1.0})  # a clear pause after the 7th cue
    cuts = _split_points(cues, max_chars=1000)
    parts = _parts(cues, cuts)
    assert len(parts[0]) == 7  # not the 10 that would fit
    assert cues[6]["end_s"] < cuts[0] < cues[7]["start_s"]


def test_a_pause_too_early_to_fill_the_part_is_not_used():
    cues = _cues(15, pause_after={1: 30.0})  # after 200 characters: less than 60% of 1000
    # 1,499 characters make two even parts, so the first holds nine lines, not the ten
    # that would fit, and not the two before the early pause.
    assert len(_parts(cues, _split_points(cues, max_chars=1000))[0]) == 9


def test_one_cue_longer_than_the_limit_becomes_its_own_part():
    cues = _cues(3, text="y" * 1500)
    parts = _parts(cues, _split_points(cues, max_chars=1000))
    assert [len(p) for p in parts] == [1, 1, 1]


def test_overlapping_cues_are_still_cut_between_their_midpoints():
    cues = [{"start_s": 0.0, "end_s": 20.0, "text": "a" * 600},
            {"start_s": 5.0, "end_s": 30.0, "text": "b" * 600}]
    parts = _parts(cues, _split_points(cues, max_chars=1000))
    assert [len(p) for p in parts] == [1, 1]


def _detect(tmp_path, monkeypatch, cues, **options):
    lines = ["WEBVTT", ""]
    for c in cues:
        lines += [f"{_stamp(c['start_s'])} --> {_stamp(c['end_s'])}", c["text"], ""]
    (tmp_path / "transcript.vtt").write_text("\n".join(lines))
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(detect_sections, "_detect_silences", lambda audio, **_kw: [])
    manifest = Manifest(duration_seconds=cues[-1]["end_s"])
    detect_sections.run(tmp_path, VTDConfig(**options), manifest)
    return json.loads((tmp_path / "sections.json").read_text())


def _stamp(seconds):
    whole, ms = divmod(round(seconds * 1000), 1000)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{ms:03d}"


def test_a_recording_without_pauses_is_divided_into_sections_the_ai_stage_reads_whole(
        tmp_path, monkeypatch):
    cues = _cues(200)  # 20,000 characters and not one silence
    sections = _detect(tmp_path, monkeypatch, cues)

    assert len(sections) == 4
    assert sum(s["cue_count"] for s in sections) == 200
    assert sections[0]["start_s"] == 0 and sections[-1]["end_s"] == cues[-1]["end_s"]
    parsed = detect_sections.parse_vtt(tmp_path / "transcript.vtt")
    for position, s in enumerate(sections):
        chunk = ai_synth._transcript_for_section(
            parsed, s["start_s"], s["end_s"], is_last=position == len(sections) - 1)
        assert 0 < len(chunk) <= 6000


def test_splitting_can_be_switched_off(tmp_path, monkeypatch):
    assert len(_detect(tmp_path, monkeypatch, _cues(200), max_section_chars=0)) == 1


def test_digest_gives_every_section_a_share_when_it_is_too_long():
    sections = [{"index": i, "title": f"T{i}", "body": f"body{i} " + "z" * 3000} for i in range(10)]
    sections.append({"index": 10, "title": "Short", "body": "kept whole"})
    digest = ai_synth._summary_digest(sections, limit=5000)

    assert len(digest) <= 5000
    assert all(f"body{i} " in digest for i in range(10))  # the last sections are still there
    assert "kept whole" in digest


def test_digest_is_untouched_when_it_fits():
    sections = [{"index": 0, "title": "A", "body": "one"}, {"index": 1, "title": "B", "body": "two"}]
    assert ai_synth._summary_digest(sections) == "Section 1: A\none\n\n---\n\nSection 2: B\ntwo"


def test_digest_never_exceeds_its_limit_even_when_the_headings_alone_do_not_fit():
    sections = [{"index": i, "title": f"T{i}", "body": "z" * 100} for i in range(5)]
    assert len(ai_synth._summary_digest(sections, limit=10)) <= 10


def test_a_short_remainder_after_a_split_stays_with_the_previous_part(tmp_path, monkeypatch):
    cues = _cues(60)  # exactly 6,000 characters over 600 s ...
    cues.append({"start_s": 600.0, "end_s": 602.0, "text": "tail words"})  # ... and 2 s more
    sections = _detect(tmp_path, monkeypatch, cues)
    assert sum(s["cue_count"] for s in sections) == 61  # the tail is not discarded


def _random_cues(rng, *, overlapping):
    cues, t = [], 0.0
    for _ in range(rng.randint(1, 60)):
        text = "w" * rng.randint(1, 200)
        if overlapping:
            start = max(0.0, t + rng.choice([-3.0, 0.0, 0.0, 0.5, 2.0]))
            end = start + rng.choice([0.0, 1.0, 3.0, 30.0])
        else:  # speech at a believable pace: at most 20 characters a second
            start = t + rng.choice([0.0, 0.5, 2.0])
            end = start + len(text) / 20 + rng.choice([0.0, 1.0])
        cues.append({"start_s": start, "end_s": end, "text": text})
        t = max(t, start) + rng.choice([1.0, 4.0]) if overlapping else end
    return cues


def _texts(folder, sections):
    parsed = detect_sections.parse_vtt(folder / "transcript.vtt")
    return [ai_synth._transcript_for_section(parsed, s["start_s"], s["end_s"],
                                             is_last=position == len(sections) - 1)
            for position, s in enumerate(sections)]


def test_splitting_never_loses_or_repeats_a_cue_whatever_the_timing(tmp_path, monkeypatch):
    import random
    rng = random.Random(7)
    for case in range(150):
        cues = _random_cues(rng, overlapping=True)
        folder = tmp_path / str(case)
        folder.mkdir()
        whole = _detect(folder, monkeypatch, cues, max_section_chars=0)
        split = _detect(folder, monkeypatch, cues, max_section_chars=rng.choice([500, 1000, 6000]))

        assert sum(s["cue_count"] for s in split) == sum(s["cue_count"] for s in whole), case
        bounds = [s["start_s"] for s in split] + [s["end_s"] for s in split[-1:]]
        assert bounds == sorted(set(bounds)), case  # sections do not overlap


def test_split_sections_stay_within_the_limit_for_ordinary_speech(tmp_path, monkeypatch):
    import random
    rng = random.Random(11)
    for case in range(150):
        cues = _random_cues(rng, overlapping=False)
        folder = tmp_path / str(case)
        folder.mkdir()
        limit = rng.choice([500, 1000, 6000])
        split = _detect(folder, monkeypatch, cues, max_section_chars=limit)
        # Only a remainder shorter than a section (5 s, so about 100 characters at this
        # pace, plus the cue that crosses into it) may be added to a part that was full.
        assert all(len(text) <= limit + 310 for text in _texts(folder, split)), case
        assert "\n".join(_texts(folder, split)) == detect_sections.cue_text(cues) or not split, case
