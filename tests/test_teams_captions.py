from vtd.config import VTDConfig
from vtd.manifest import Manifest
from vtd.pipeline import _stages_to_run
from vtd.stages.transcribe_teams import (
    _canonicalize_speakers,
    _merge_records,
    _parse_entries,
)


def test_silent_teams_video_skips_audio_extraction():
    config = VTDConfig(teams_captions=True)
    manifest = Manifest(has_video=True, has_audio=False)

    stages = _stages_to_run(None, config, manifest)

    assert "extract-audio" not in stages
    assert "extract-frames" in stages
    assert "transcribe" in stages


def test_parse_entries_ignores_other_screen_columns():
    lines = [
        ("Reed, Tom", 100, 10, 18),
        ("Let's look at the agenda", 100, 34, 18),
        ("Shared Files", 600, 35, 18),
        ("Marsh, Ellen", 100, 70, 18),
        ("No.", 100, 94, 18),
    ]

    assert _parse_entries(lines) == [
        ("Reed, Tom", "Let's look at the agenda"),
        ("Marsh, Ellen", "No."),
    ]


def test_parse_entries_strips_teams_icon_noise():
    lines = [
        ("Blake, Nora", 100, 10, 18),
        ("fe Yeah, | need to check it", 100, 34, 18),
        ("Blake, Nora", 100, 70, 18),
        ("27 J And then it should be fine", 100, 94, 18),
    ]

    assert _parse_entries(lines) == [
        ("Blake, Nora", "Yeah, I need to check it"),
        ("Blake, Nora", "And then it should be fine"),
    ]


def test_parse_entries_accepts_plain_display_name():
    lines = [
        ("Priya A Shah", 276, 146, 11),
        ("And also the second point. What the chart", 275, 168, 18),
    ]

    assert _parse_entries(lines) == [
        ("Priya A Shah", "And also the second point. What the chart")
    ]

def test_merge_records_extends_repeated_live_caption():
    records = _merge_records([], [("Reed, Tom", "Let's look at the agenda")], 60, 5)
    records = _merge_records(
        records,
        [("Reed, Tom", "Let's look at the agenda together")],
        65,
        5,
    )

    assert records == [{
        "speaker": "Reed, Tom",
        "text": "Let's look at the agenda together",
        "start_s": 60,
        "end_s": 70,
    }]


def test_canonicalize_speakers_folds_rare_ocr_variant():
    records = [
        {"speaker": "Reed, Tom", "text": "One", "start_s": 0, "end_s": 5},
        {"speaker": "Reed, Tom", "text": "Two", "start_s": 5, "end_s": 10},
        {"speaker": "Reec, To", "text": "Three", "start_s": 10, "end_s": 15},
    ]

    assert {record["speaker"] for record in _canonicalize_speakers(records)} == {"Reed, Tom"}
