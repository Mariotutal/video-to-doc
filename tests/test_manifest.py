from pathlib import Path

from vtd import manifest as mf


def test_round_trip(tmp_path: Path):
    m = mf.Manifest(source_video="/tmp/x.mp4", duration_seconds=42.0)
    mf.save(tmp_path, m)
    loaded = mf.load(tmp_path)
    assert loaded.source_video == "/tmp/x.mp4"
    assert loaded.duration_seconds == 42.0
    assert loaded.stages_completed == []


def test_mark_and_check_stage(tmp_path: Path):
    m = mf.Manifest()
    m.mark_stage("extract-audio")
    m.mark_stage("transcribe")
    m.mark_stage("transcribe")  # idempotent
    assert m.has_stage("extract-audio")
    assert m.has_stage("transcribe")
    assert m.stages_completed.count("transcribe") == 1


def test_reset_stage(tmp_path: Path):
    m = mf.Manifest()
    m.mark_stage("extract-audio")
    m.reset_stage("extract-audio")
    assert not m.has_stage("extract-audio")


def test_corrupt_manifest_recovers(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{not valid json")
    m = mf.load(tmp_path)
    # Returns a fresh manifest rather than blowing up
    assert m.schema_version == mf.SCHEMA_VERSION
    assert (tmp_path / "manifest.json.bak").exists()


def test_video_hash_prefix_changes_with_content(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x" * 2_500_000)
    b.write_bytes(b"y" * 2_500_000)
    assert mf.video_hash_prefix(a) != mf.video_hash_prefix(b)


def test_unknown_fields_do_not_reset_the_workspace(tmp_path: Path, capsys):
    import json

    m = mf.Manifest(source_video="/x/y.mp4")
    m.mark_stage("transcribe")
    mf.save(tmp_path, m)
    raw = json.loads((tmp_path / "manifest.json").read_text())
    raw.update(schema_version=mf.SCHEMA_VERSION + 1, added_by_a_newer_version=True)
    (tmp_path / "manifest.json").write_text(json.dumps(raw))

    loaded = mf.load(tmp_path)

    assert loaded.source_video == "/x/y.mp4"
    assert loaded.has_stage("transcribe")
    assert not (tmp_path / "manifest.json.bak").exists()
    assert "another vtd version" in capsys.readouterr().err


def test_non_numeric_schema_version_is_tolerated(tmp_path: Path):
    (tmp_path / "manifest.json").write_text('{"schema_version": "2", "source_video": "/x/y.mp4"}')
    assert mf.load(tmp_path).source_video == "/x/y.mp4"


def test_interrupted_save_keeps_the_previous_manifest(tmp_path: Path, monkeypatch):
    import pytest

    mf.save(tmp_path, mf.Manifest(source_video="/old.mp4"))

    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(mf.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        mf.save(tmp_path, mf.Manifest(source_video="/new.mp4"))
    monkeypatch.undo()

    assert mf.load(tmp_path).source_video == "/old.mp4"


def test_resetting_a_stage_forgets_the_options_it_ran_with(tmp_path: Path):
    m = mf.Manifest()
    m.mark_stage("transcribe", {"whisper_model": "medium"})
    m.reset_stage("transcribe")
    assert m.stage_options == {}
