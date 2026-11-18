import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from vtd import manifest as mf
from vtd import pipeline
from vtd.ai_synth import _frames_for_section, _transcript_for_section
from vtd.config import VTDConfig
from vtd.stages import detect_sections, extract_frames, ocr_speakers
from vtd.stages.detect_sections import _parse_silences, cues_in_section
from vtd.stages.extract_frames import _parse_showinfo_times, frame_times, list_frames
from vtd.stages.render import _load_template
from vtd.utils import VTDError, run


def _fake_probe(_source):
    return {"duration": 60.0, "width": None, "height": None, "has_video": False, "has_audio": True}


def _run(source, workspace, *, resume=False, only=("detect-sections",), **cfg):
    pipeline.run_pipeline(
        source=source, workspace=workspace, config=VTDConfig(**cfg),
        resume=resume, only_stages=list(only), verbose=False,
    )


@pytest.fixture
def counting_stage(monkeypatch):
    """Replace detect-sections with a stage that records the gap it was run with."""
    calls = []

    def stage(workspace, config, manifest):
        calls.append(config.section_gap_s)
        (workspace / "sections.json").write_text(json.dumps({"gap": config.section_gap_s}))

    monkeypatch.setattr(pipeline, "ffprobe_info", _fake_probe)
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    monkeypatch.setitem(pipeline.STAGES, "detect-sections", stage)
    return calls


def test_rerun_without_resume_recomputes_with_new_options(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"

    _run(source, ws, section_gap_s=8.0)
    _run(source, ws, section_gap_s=100.0)

    assert counting_stage == [8.0, 100.0]
    assert json.loads((ws / "sections.json").read_text()) == {"gap": 100.0}


def test_resume_skips_completed_stage(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"

    _run(source, ws, section_gap_s=8.0)
    _run(source, ws, resume=True, section_gap_s=8.0)

    assert counting_stage == [8.0]


def test_resume_redoes_a_stage_whose_options_changed(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws, section_gap_s=8.0)

    _run(source, ws, resume=True, section_gap_s=100.0)

    assert counting_stage == [8.0, 100.0]
    assert mf.load(ws).stage_options["detect-sections"]["section_gap_s"] == 100.0


def test_resume_ignores_options_the_stage_does_not_use(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws)

    _run(source, ws, resume=True, whisper_model="large-v3", content_type="lecture")

    assert counting_stage == [8.0]


def test_stage_completed_before_options_were_recorded_is_trusted(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws)
    manifest = mf.load(ws)
    manifest.stage_options.clear()  # as written by an earlier vtd release
    mf.save(ws, manifest)

    _run(source, ws, resume=True, section_gap_s=100.0)

    assert counting_stage == [8.0]


def test_failed_rerun_keeps_the_options_of_the_restored_result(tmp_path, counting_stage,
                                                              monkeypatch):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws, section_gap_s=8.0)

    def failing(workspace, config, manifest):
        raise KeyboardInterrupt

    monkeypatch.setitem(pipeline.STAGES, "detect-sections", failing)
    with pytest.raises(KeyboardInterrupt):
        _run(source, ws, section_gap_s=100.0)
    monkeypatch.undo()
    monkeypatch.setattr(pipeline, "ffprobe_info", _fake_probe)
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)

    manifest = mf.load(ws)
    assert manifest.stage_options["detect-sections"]["section_gap_s"] == 8.0
    # The restored result was made with 8.0, so resuming with 100.0 must redo it.
    assert not pipeline._is_current("detect-sections", VTDConfig(section_gap_s=100.0), manifest)
    assert pipeline._is_current("detect-sections", VTDConfig(section_gap_s=8.0), manifest)


def test_model_folder_is_recorded_in_the_manifest_as_text(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    _run(source, tmp_path / "ws", model_dir=tmp_path / "models")
    assert mf.load(tmp_path / "ws").config_used["model_dir"] == str(tmp_path / "models")


def test_outputs_left_aside_by_a_killed_run_are_put_back(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    (ws / pipeline.BACKUP_DIR).mkdir(parents=True)
    (ws / pipeline.BACKUP_DIR / "transcript.txt").write_text("hours of whisper output")

    _run(source, ws)

    assert (ws / "transcript.txt").read_text() == "hours of whisper output"
    assert not (ws / pipeline.BACKUP_DIR).exists()


def test_failed_mode_switch_keeps_the_existing_transcript_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ffprobe_info", lambda _s: {
        "duration": 60.0, "width": 640, "height": 360, "has_video": True, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)

    def stage(name):
        def run(workspace, config, manifest):
            if name == "transcribe" and config.teams_captions:
                raise VTDError("No Teams live captions were detected.")
            if name == "transcribe":
                (workspace / "transcript.txt").write_text("whisper transcript")
        return run

    for name in pipeline.STAGES:
        monkeypatch.setitem(pipeline.STAGES, name, stage(name))
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    run = dict(source=source, workspace=tmp_path / "ws", only_stages=None, verbose=False)

    pipeline.run_pipeline(**run, config=VTDConfig(), resume=False)
    with pytest.raises(VTDError, match="No Teams live captions"):
        pipeline.run_pipeline(**run, config=VTDConfig(teams_captions=True, frame_rate_seconds=5),
                              resume=False)

    manifest = mf.load(tmp_path / "ws")
    assert (tmp_path / "ws" / "transcript.txt").read_text() == "whisper transcript"
    assert pipeline._is_current("transcribe", VTDConfig(), manifest)


def test_rerunning_a_stage_makes_resume_redo_what_was_built_on_it(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(pipeline, "ffprobe_info", _fake_probe)
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    for name in pipeline.STAGES:
        monkeypatch.setitem(
            pipeline.STAGES, name,
            lambda workspace, config, manifest, name=name: ran.append(name),
        )
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    full = dict(source=source, workspace=ws, config=VTDConfig(), verbose=False)

    pipeline.run_pipeline(**full, resume=False, only_stages=None)
    assert ran == ["extract-audio", "transcribe", "detect-sections", "render-starter"]

    ran.clear()
    pipeline.run_pipeline(**full, resume=False, only_stages=["transcribe"])
    assert ran == ["transcribe"]
    assert mf.load(ws).stages_completed == ["extract-audio", "transcribe"]

    ran.clear()
    pipeline.run_pipeline(**full, resume=True, only_stages=None)
    assert ran == ["detect-sections", "render-starter"]


def test_stages_do_not_trust_a_file_they_find(tmp_path, monkeypatch):
    # Called directly, without the pipeline: only the pipeline decides what is still valid.
    monkeypatch.setattr(detect_sections, "_detect_silences", lambda *a, **k: [])
    (tmp_path / "audio.wav").write_bytes(b"x")
    (tmp_path / "sections.json").write_text('"left over from an interrupted run"')

    detect_sections.run(tmp_path, VTDConfig(), mf.Manifest(duration_seconds=60.0))

    sections = json.loads((tmp_path / "sections.json").read_text())
    assert [(s["start_s"], s["end_s"]) for s in sections] == [(0.0, 60.0)]


def _stub_all_stages(monkeypatch, ran, *, has_video=True, fail=None):
    """Every stage writes its real output names and records that it ran."""
    monkeypatch.setattr(pipeline, "ffprobe_info", lambda _s: {
        "duration": 60.0, "width": 640, "height": 360, "has_video": has_video, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    # No provider package, key or server is needed: the AI stage itself is stubbed.
    monkeypatch.setattr(pipeline, "_preflight_ai", lambda config, workspace: None)

    def stage(name):
        def run(workspace, config, manifest):
            ran.append(name)
            if fail and fail(name, config):
                raise VTDError(f"{name} failed")
            for rel in pipeline._stage_outputs(name, config):
                if rel == "frames":
                    (workspace / rel).mkdir()
                else:
                    (workspace / rel).write_text(f"{name} output")
        return run

    for name in pipeline.STAGES:
        monkeypatch.setitem(pipeline.STAGES, name, stage(name))


def test_failed_teams_attempt_costs_nothing_on_a_finished_whisper_workspace(tmp_path,
                                                                           monkeypatch):
    ran = []
    _stub_all_stages(monkeypatch, ran,
                     fail=lambda name, config: name == "transcribe" and config.teams_captions)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    ws = tmp_path / "ws"
    run = dict(source=source, workspace=ws, only_stages=None, verbose=False)
    pipeline.run_pipeline(**run, config=VTDConfig(), resume=False)

    with pytest.raises(VTDError, match="transcribe failed"):
        pipeline.run_pipeline(**run, config=VTDConfig(teams_captions=True, frame_rate_seconds=5),
                              resume=False)
    assert (ws / "audio.wav").exists() and (ws / "transcript.txt").exists()

    ran.clear()
    pipeline.run_pipeline(**run, config=VTDConfig(), resume=True)
    # Only what the Teams attempt replaced (the frames) and what reads it is redone.
    assert ran == ["extract-frames", "ocr-speakers", "render-starter"]


def test_failed_whisper_attempt_costs_nothing_on_a_finished_teams_workspace(tmp_path,
                                                                           monkeypatch):
    ran = []
    _stub_all_stages(
        monkeypatch, ran,
        fail=lambda name, config: name == "transcribe" and not config.teams_captions,
    )
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    ws = tmp_path / "ws"
    run = dict(source=source, workspace=ws, only_stages=None, verbose=False)
    teams = VTDConfig(teams_captions=True, frame_rate_seconds=5)
    pipeline.run_pipeline(**run, config=teams, resume=False)
    (ws / "frames" / "frame_0001.jpg").write_bytes(b"teams frame")

    ran.clear()
    with pytest.raises(VTDError, match="transcribe failed"):
        pipeline.run_pipeline(**run, config=VTDConfig(), resume=True)

    # Whisper transcribes before sampling frames, so the Teams frames were never touched.
    assert ran == ["extract-audio", "transcribe"]
    assert (ws / "frames" / "frame_0001.jpg").read_bytes() == b"teams frame"
    manifest = mf.load(ws)
    assert pipeline._is_current("transcribe", teams, manifest)
    assert pipeline._is_current("extract-frames", teams, manifest)


def test_only_refuses_to_build_on_inputs_made_with_other_options(tmp_path, monkeypatch):
    ran = []
    _stub_all_stages(monkeypatch, ran)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    run = dict(source=source, workspace=tmp_path / "ws", verbose=False, resume=False)
    pipeline.run_pipeline(**run, config=VTDConfig(), only_stages=None)

    ran.clear()
    teams = VTDConfig(teams_captions=True, frame_rate_seconds=5)
    with pytest.raises(VTDError, match=r"transcribe would read extract-frames .*frame_rate_seconds"):
        pipeline.run_pipeline(**run, config=teams, only_stages=["transcribe"])
    assert ran == []

    pipeline.run_pipeline(**run, config=teams, only_stages=["extract-frames", "transcribe"])
    assert ran == ["extract-frames", "transcribe"]


def test_speaker_ocr_is_redone_once_tesseract_is_installed(tmp_path, monkeypatch):
    ran = []
    _stub_all_stages(monkeypatch, ran)
    installed = {"tesseract": None}
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: installed.get(name, name))
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    run = dict(source=source, workspace=tmp_path / "ws", only_stages=None, verbose=False,
               config=VTDConfig())
    pipeline.run_pipeline(**run, resume=False)

    ran.clear()
    pipeline.run_pipeline(**run, resume=True)
    assert ran == []  # still no tesseract: nothing to redo, nothing invalidated

    installed["tesseract"] = "/usr/bin/tesseract"
    pipeline.run_pipeline(**run, resume=True)
    assert ran == ["ocr-speakers", "render-starter"]


def test_option_added_after_a_stage_completed_does_not_invalidate_it(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws)
    manifest = mf.load(ws)
    del manifest.stage_options["detect-sections"]["teams_captions"]  # recorded by an older vtd
    mf.save(ws, manifest)

    _run(source, ws, resume=True)

    assert counting_stage == [8.0]


def test_stages_run_in_pipeline_order_whatever_order_only_lists_them(tmp_path, monkeypatch):
    ran = []
    _stub_all_stages(monkeypatch, ran)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")

    pipeline.run_pipeline(source=source, workspace=tmp_path / "ws", config=VTDConfig(),
                          resume=False, verbose=False,
                          only_stages=["extract-frames", "transcribe", "extract-audio"])

    assert ran == ["extract-audio", "transcribe", "extract-frames"]


def test_ai_notes_refuse_inputs_made_with_other_options(tmp_path, monkeypatch):
    from vtd import ai_synth

    ran = []
    _stub_all_stages(monkeypatch, ran)
    monkeypatch.setattr(ai_synth, "synthesize", lambda w, c, m: ran.append("synthesize") or [])
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    run = dict(source=source, workspace=tmp_path / "ws", verbose=False, resume=False)
    pipeline.run_pipeline(**run, config=VTDConfig(whisper_model="small"), only_stages=None)

    ran.clear()
    cfg = VTDConfig()  # --whisper-model small forgotten
    cfg.ai.enabled, cfg.ai.api_key = True, "sk-test"
    with pytest.raises(VTDError, match="synthesize would read transcribe"):
        pipeline.run_pipeline(**run, config=cfg, only_stages=["synthesize"])
    assert ran == []


def test_incomplete_rerun_of_finished_ai_notes_is_not_left_marked_done(tmp_path, monkeypatch,
                                                                      capsys):
    from vtd import ai_synth

    ran = []
    _stub_all_stages(monkeypatch, ran)
    answers = [[], ["glossary"]]
    monkeypatch.setattr(ai_synth, "synthesize", lambda w, c, m: answers.pop(0))
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    cfg = VTDConfig()
    cfg.ai.enabled, cfg.ai.api_key = True, "sk-test"
    run = dict(source=source, workspace=tmp_path / "ws", verbose=False, config=cfg, resume=False)
    pipeline.run_pipeline(**run, only_stages=None)
    assert mf.load(tmp_path / "ws").has_stage("synthesize")

    pipeline.run_pipeline(**run, only_stages=["synthesize"])

    assert not mf.load(tmp_path / "ws").has_stage("synthesize")
    out = capsys.readouterr()
    assert "missing or incomplete: glossary" in out.err and "Finished with gaps" in out.err
    assert out.out.count("All requested stages complete") == 1  # only the first run


def test_teams_sections_ignore_audio_left_from_a_whisper_run(tmp_path, monkeypatch):
    monkeypatch.setattr(detect_sections, "_detect_silences",
                        lambda *a, **k: pytest.fail("Teams mode must not read the audio"))
    (tmp_path / "audio.wav").write_bytes(b"silent recording")
    (tmp_path / "transcript.vtt").write_text(
        "00:00:00.000 --> 00:00:40.000\nAna: first topic\n\n"
        "00:01:30.000 --> 00:02:00.000\nLuis: second topic\n"
    )

    detect_sections.run(tmp_path, VTDConfig(teams_captions=True, min_section_s=0),
                        mf.Manifest(duration_seconds=120.0))

    assert len(json.loads((tmp_path / "sections.json").read_text())) == 2


def test_incomplete_ai_notes_are_retried_on_resume(tmp_path, monkeypatch):
    from vtd import ai_synth

    ran = []
    _stub_all_stages(monkeypatch, ran)
    answers = [["glossary"], []]
    monkeypatch.setattr(ai_synth, "synthesize",
                        lambda w, c, m: ran.append("synthesize") or answers.pop(0))
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    cfg = VTDConfig()
    cfg.ai.enabled, cfg.ai.api_key = True, "sk-test"
    run = dict(source=source, workspace=tmp_path / "ws", only_stages=None, verbose=False, config=cfg)

    pipeline.run_pipeline(**run, resume=False)
    assert not mf.load(tmp_path / "ws").has_stage("synthesize")

    ran.clear()
    pipeline.run_pipeline(**run, resume=True)
    assert ran == ["synthesize"]
    assert mf.load(tmp_path / "ws").has_stage("synthesize")


def test_teams_sections_come_from_gaps_between_captions(tmp_path):
    # No audio.wav in Teams mode: a stretch with no captions separates sections.
    (tmp_path / "transcript.vtt").write_text(
        "00:00:00.000 --> 00:00:20.000\nAna: hello\n\n"
        "00:00:20.000 --> 00:00:40.000\nAna: first topic\n\n"
        "00:01:30.000 --> 00:02:00.000\nLuis: second topic\n"
    )

    detect_sections.run(tmp_path, VTDConfig(min_section_s=0), mf.Manifest(duration_seconds=120.0))

    sections = json.loads((tmp_path / "sections.json").read_text())
    assert [(s["start_s"], s["end_s"], s["cue_count"]) for s in sections] == [
        (0.0, 90.0, 2), (90.0, 120.0, 1),
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("tesseract") is None,
                    reason="needs ffmpeg and tesseract")
def test_whole_pipeline_on_a_real_clip_with_transcription_stubbed(tmp_path, monkeypatch):
    clip = tmp_path / "talk.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "mandelbrot=r=1:s=320x240",
         "-f", "lavfi", "-i", "sine=f=440", "-t", "30", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )

    def fake_whisper(workspace, config, manifest):
        assert (workspace / "audio.wav").stat().st_size > 0
        (workspace / "transcript.txt").write_text("hello from the talk")
        (workspace / "transcript.vtt").write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:09.000\nhello from the talk\n"
        )

    monkeypatch.setitem(pipeline.STAGES, "transcribe", fake_whisper)
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    ws = tmp_path / "ws"

    pipeline.run_pipeline(source=clip, workspace=ws, config=VTDConfig(frame_rate_seconds=10),
                          resume=False, only_stages=None, verbose=False)

    manifest = mf.load(ws)
    assert manifest.stages_completed == [
        "extract-audio", "transcribe", "extract-frames", "ocr-speakers",
        "detect-sections", "render-starter",
    ]
    assert list_frames(ws / "frames")
    starter = (ws / "notes.starter.md").read_text()
    assert "hello from the talk" in starter and "one frame every 10 s" in starter
    assert json.loads((ws / "sections.json").read_text())[0]["cue_count"] == 1
    assert not (ws / pipeline.BACKUP_DIR).exists() and not (ws / "frames.partial").exists()


def test_resume_with_ocr_disabled_drops_speakers_and_redoes_the_document(tmp_path, monkeypatch):
    ran = []
    _stub_all_stages(monkeypatch, ran)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    ws = tmp_path / "ws"
    run = dict(source=source, workspace=ws, only_stages=None, verbose=False)
    pipeline.run_pipeline(**run, config=VTDConfig(), resume=False)

    ran.clear()
    pipeline.run_pipeline(**run, config=VTDConfig(ocr_enabled=False), resume=True)

    assert not (ws / "speakers.json").exists()
    assert ran == ["render-starter"]


def test_failed_stage_restores_every_output_it_set_aside(tmp_path, monkeypatch):
    ran = []
    _stub_all_stages(monkeypatch, ran)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    ws = tmp_path / "ws"
    run = dict(source=source, workspace=ws, only_stages=["transcribe"], verbose=False)
    pipeline.run_pipeline(**run, config=VTDConfig(), resume=False)
    (ws / "transcript.txt").write_text("first transcript")
    (ws / "notes.starter.md").write_text("stale starter")

    _stub_all_stages(monkeypatch, ran, fail=lambda name, config: name == "transcribe")
    with pytest.raises(VTDError):
        pipeline.run_pipeline(**run, config=VTDConfig(), resume=False)

    assert (ws / "transcript.txt").read_text() == "first transcript"
    assert (ws / "transcript.vtt").exists()
    assert not (ws / pipeline.BACKUP_DIR).exists()
    assert mf.load(ws).has_stage("transcribe")


def test_resume_redoes_ai_notes_when_their_options_change(tmp_path, monkeypatch):
    from vtd import ai_synth

    ran = []
    _stub_all_stages(monkeypatch, ran)
    monkeypatch.setattr(ai_synth, "synthesize",
                        lambda workspace, config, manifest, **k: ran.append("synthesize"))
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    run = dict(source=source, workspace=tmp_path / "ws", only_stages=None, verbose=False)

    def config(**kwargs):
        cfg = VTDConfig(**kwargs)
        cfg.ai.enabled, cfg.ai.api_key = True, "sk-test"
        return cfg

    pipeline.run_pipeline(**run, config=config(), resume=False)
    ran.clear()
    pipeline.run_pipeline(**run, config=config(), resume=True)
    assert ran == []
    pipeline.run_pipeline(**run, config=config(content_type="lecture"), resume=True)
    assert ran == ["synthesize"]


def test_each_content_type_has_a_template():
    from vtd.config import CONTENT_TYPES

    for content_type in CONTENT_TYPES:
        assert _load_template(f"full_{content_type}.md.j2")


def test_missing_tesseract_skips_speaker_ocr_with_a_warning(tmp_path, monkeypatch, capsys):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "frame_0001.jpg").write_bytes(b"x")
    monkeypatch.setattr(ocr_speakers.shutil, "which", lambda name: None)

    ocr_speakers.run(tmp_path, VTDConfig(), mf.Manifest())

    assert json.loads((tmp_path / "speakers.json").read_text()) == []
    assert "tesseract not found" in capsys.readouterr().err


def test_teams_mode_is_refused_on_an_existing_audio_only_workspace(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws)

    with pytest.raises(VTDError, match="requires a video stream"):
        _run(source, ws, teams_captions=True)
    assert (ws / "sections.json").exists()


def test_dependents_follow_what_each_stage_reads():
    whisper, teams, fresh = VTDConfig(), VTDConfig(teams_captions=True), mf.Manifest()
    assert pipeline._dependents("extract-frames", whisper, fresh) == [
        "ocr-speakers", "render-starter", "synthesize",
    ]
    assert pipeline._dependents("extract-frames", teams, fresh) == [
        "transcribe", "ocr-speakers", "detect-sections", "render-starter", "synthesize",
    ]
    assert pipeline._dependents("detect-sections", whisper, fresh) == [
        "render-starter", "synthesize",
    ]
    assert pipeline._dependents("render-starter", whisper, fresh) == []
    # A transcript already made by whisper does not depend on the frames, whatever is asked now.
    built_by_whisper = mf.Manifest(stage_options={"transcribe": {"teams_captions": False}})
    assert "transcribe" not in pipeline._dependents("extract-frames", teams, built_by_whisper)


def test_moved_source_is_followed(tmp_path, counting_stage):
    first = tmp_path / "a.wav"
    first.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(first, ws)
    moved = tmp_path / "elsewhere.wav"
    first.rename(moved)

    _run(moved, ws)

    assert mf.load(ws).source_video == str(moved.resolve())


def test_missing_tool_is_reported_before_any_stage_runs(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(pipeline, "ffprobe_info", _fake_probe)
    monkeypatch.setattr(pipeline, "require_binary", lambda name, pkg=None: (_ for _ in ()).throw(
        VTDError(f"Required binary `{name}` not found on PATH.")) if name == "whisper-cli" else name)
    for name in pipeline.STAGES:
        monkeypatch.setitem(pipeline.STAGES, name, lambda w, c, m, name=name: ran.append(name))
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")

    with pytest.raises(VTDError, match="whisper-cli"):
        pipeline.run_pipeline(source=source, workspace=tmp_path / "ws", config=VTDConfig(),
                              resume=False, only_stages=None, verbose=False)
    assert ran == []


def test_workspace_from_another_source_is_rejected(tmp_path, counting_stage):
    first = tmp_path / "a.wav"
    first.write_bytes(b"audio-a")
    other = tmp_path / "b.wav"
    other.write_bytes(b"different audio")
    ws = tmp_path / "ws"

    _run(first, ws)
    with pytest.raises(VTDError, match="different source"):
        _run(other, ws)
    assert mf.load(ws).source_video == str(first.resolve())


def test_failed_rerun_restores_previous_outputs(tmp_path, counting_stage, monkeypatch):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws, section_gap_s=8.0)

    def failing(workspace, config, manifest):
        raise VTDError("prerequisite missing")

    monkeypatch.setitem(pipeline.STAGES, "detect-sections", failing)
    with pytest.raises(VTDError, match="prerequisite missing"):
        _run(source, ws, section_gap_s=100.0)

    assert json.loads((ws / "sections.json").read_text()) == {"gap": 8.0}
    assert mf.load(ws).has_stage("detect-sections")
    assert not (ws / ".rerun-backup").exists()


def test_rerun_invalidates_ai_notes(tmp_path, counting_stage):
    source = tmp_path / "a.wav"
    source.write_bytes(b"audio-a")
    ws = tmp_path / "ws"
    _run(source, ws)
    manifest = mf.load(ws)
    manifest.mark_stage(pipeline.AI_STAGE_NAME)
    mf.save(ws, manifest)

    _run(source, ws, section_gap_s=100.0)

    assert not mf.load(ws).has_stage(pipeline.AI_STAGE_NAME)


def test_full_rerun_drops_results_of_stages_it_now_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ffprobe_info", lambda _s: {
        "duration": 60.0, "width": 640, "height": 360, "has_video": True, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    for name in pipeline.STAGES:
        monkeypatch.setitem(pipeline.STAGES, name, lambda workspace, config, manifest: None)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "speakers.json").write_text('[{"name": "Old Speaker"}]')

    pipeline.run_pipeline(source=source, workspace=ws, config=VTDConfig(ocr_enabled=False),
                          resume=False, only_stages=None, verbose=False)

    assert not (ws / "speakers.json").exists()


def test_negative_silence_start_does_not_shift_pairs():
    log = (
        "silence_start: -0.00133\n"
        "silence_end: 3.2 | silence_duration: 3.2\n"
        "silence_start: 40\n"
        "silence_end: 52.5 | silence_duration: 12.5\n"
        "silence_start: 90\n"  # still silent at end of file: no end
    )
    assert _parse_silences(log) == [(0.0, 3.2), (40.0, 52.5)]


def test_cue_straddling_a_boundary_lands_in_exactly_one_section():
    cues = [
        {"start_s": 0, "end_s": 28, "text": "one"},
        {"start_s": 28, "end_s": 58, "text": "straddle"},
        {"start_s": 58, "end_s": 95, "text": "runs past the end"},
    ]
    first = cues_in_section(cues, 0, 40.5, is_last=False)
    second = cues_in_section(cues, 40.5, 90, is_last=True)

    assert [c["text"] for c in first] == ["one"]
    assert [c["text"] for c in second] == ["straddle", "runs past the end"]
    assert _transcript_for_section(cues, 40.5, 90, is_last=True) == "straddle\nruns past the end"


def _make_frames(workspace, count):
    frames = workspace / "frames"
    frames.mkdir()
    for i in range(1, count + 1):
        (frames / f"frame_{i:04d}.jpg").write_bytes(b"x")
    return frames


def test_frame_times_use_recorded_timestamps_not_the_frame_number(tmp_path):
    frames = _make_frames(tmp_path, 3)
    (frames / "timestamps.json").write_text(json.dumps(
        {"frame_0001.jpg": 0, "frame_0002.jpg": 300, "frame_0003.jpg": 420}
    ))

    assert [sec for _, sec in frame_times(tmp_path, 60)] == [0.0, 300.0, 420.0]
    picked = _frames_for_section(tmp_path, VTDConfig(), 290, 430)
    assert [p.name for p in picked] == ["frame_0002.jpg", "frame_0003.jpg"]


def test_frame_times_fall_back_to_index_for_old_workspaces(tmp_path):
    _make_frames(tmp_path, 3)
    assert [sec for _, sec in frame_times(tmp_path, 60)] == [0.0, 60.0, 120.0]


def test_frames_are_spread_evenly_across_a_section(tmp_path):
    _make_frames(tmp_path, 5)
    picked = _frames_for_section(tmp_path, VTDConfig(), 0, 300)
    assert [p.name for p in picked] == ["frame_0001.jpg", "frame_0003.jpg", "frame_0005.jpg"]


def test_templates_ship_inside_the_package():
    package_dir = Path(pipeline.__file__).parent
    assert len(list((package_dir / "templates").glob("*.j2"))) == 6  # starter, four full ones, shared parts
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert pyproject["tool"]["setuptools"]["package-data"]["vtd"] == ["templates/*.j2"]
    assert "t.starter_notes" in _load_template("starter.md.j2")


def test_trailing_sliver_is_folded_into_the_last_section(tmp_path, monkeypatch):
    monkeypatch.setattr(detect_sections, "_detect_silences", lambda *a, **k: [(20.0, 57.0)])
    (tmp_path / "audio.wav").write_bytes(b"x")
    (tmp_path / "transcript.vtt").write_text(
        "00:00:01.000 --> 00:00:19.000\nintro\n\n00:00:57.500 --> 00:01:01.000\nthanks, bye\n"
    )

    detect_sections.run(tmp_path, VTDConfig(), mf.Manifest(duration_seconds=60.0))

    sections = json.loads((tmp_path / "sections.json").read_text())
    assert [(s["start_s"], s["end_s"], s["cue_count"]) for s in sections] == [(0.0, 60.0, 2)]


def test_showinfo_log_gives_capture_times_in_output_order():
    log = (
        "[Parsed_showinfo_2 @ 0x1] n:   0 pts:      0 pts_time:0       duration:60\n"
        "[Parsed_showinfo_2 @ 0x1] n:   1 pts:      5 pts_time:300     duration:60\n"
        "[out#0/image2 @ 0x2] video:12KiB pts_time:999 not a showinfo line\n"
    )
    assert _parse_showinfo_times(log) == [0.0, 300.0]


def test_frames_sort_numerically_past_9999(tmp_path):
    for n in (9999, 10000, 1000):
        (tmp_path / f"frame_{n:04d}.jpg").write_bytes(b"x")
    assert [p.name for p in list_frames(tmp_path)] == [
        "frame_1000.jpg", "frame_9999.jpg", "frame_10000.jpg",
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("seconds,rate,at_least", [(180, 60, 2), (4, 60, 1)])
def test_extract_frames_records_a_timestamp_per_frame(tmp_path, seconds, rate, at_least):
    clip = tmp_path / "clip with space.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "mandelbrot=r=1:s=160x120",
         "-t", str(seconds), "-metadata", "title=Reunión", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    ws = tmp_path / "ws"
    ws.mkdir()

    extract_frames.run(ws, VTDConfig(frame_rate_seconds=rate),
                       mf.Manifest(source_video=str(clip), duration_seconds=float(seconds)))

    times = json.loads((ws / "frames" / "timestamps.json").read_text())
    assert len(list_frames(ws / "frames")) == len(times) >= at_least
    assert list(times.values()) == sorted(times.values()) and times["frame_0001.jpg"] == 0.0
    assert not (ws / "frames.partial").exists()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")
def test_ocr_reads_speaker_labels_from_the_bottom_left_corner(tmp_path):
    from PIL import Image, ImageDraw

    frames = tmp_path / "frames"
    frames.mkdir()
    shown = ["Luis Ortega", "Luis Ortega", "Ana Lopez", "Ana Lopez", "Ana Lopez", "Seen Once"]
    for i, name in enumerate(shown, 1):
        image = Image.new("RGB", (1280, 720), (30 + 20 * i, 60, 90))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 640, 1280, 720), fill="white")
        draw.text((30, 655), name, fill="black", font_size=32)
        image.save(frames / f"frame_{i:04d}.jpg")
    (frames / "timestamps.json").write_text(json.dumps(
        {f"frame_{i:04d}.jpg": (i - 1) * 300 for i in range(1, len(shown) + 1)}
    ))

    ocr_speakers.run(tmp_path, VTDConfig(), mf.Manifest())

    speakers = json.loads((tmp_path / "speakers.json").read_text())
    assert [(s["name"], s["frame_count"], s["first_seen_s"], s["last_seen_s"]) for s in speakers] == [
        ("Ana Lopez", 3, 600.0, 1200.0),
        ("Luis Ortega", 2, 0.0, 300.0),
    ]  # a label seen in a single frame is treated as OCR noise


def test_failed_subprocess_is_a_user_facing_error():
    with pytest.raises(VTDError, match="exit code 3: boom"):
        run([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
            capture=True)
    with pytest.raises(VTDError, match="not found on PATH"):
        run(["vtd-no-such-binary"])


def test_local_provider_runs_without_an_api_key(tmp_path, monkeypatch):
    from vtd import ai_synth
    monkeypatch.setattr(ai_synth, "synthesize", lambda workspace, config, manifest: [])
    config = VTDConfig()
    config.ai.enabled, config.ai.provider, config.ai.model = True, "ollama", "qwen3.6:35b"
    assert pipeline._run_ai_stage(tmp_path, config, mf.Manifest(), resume=False) is True

    config.ai.provider = "openai"  # a hosted provider still needs one
    with pytest.raises(VTDError, match="OPENAI_API_KEY"):
        pipeline._run_ai_stage(tmp_path, config, mf.Manifest(), resume=False)
