"""Where the Whisper model is looked for and downloaded to."""
import pytest

from vtd import pipeline
from vtd.config import VTDConfig
from vtd.manifest import Manifest
from vtd.stages import transcribe
from vtd.utils import VTDError


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    monkeypatch.setattr(transcribe, "MIN_MODEL_BYTES", 10)


def _model(folder, name="medium"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"ggml-{name}.bin"
    path.write_bytes(b"x" * 100)
    return path


def test_model_is_looked_for_in_the_configured_folder(tmp_path):
    model = _model(tmp_path / "disk" / "models", "large-v3-turbo")
    config = VTDConfig(whisper_model="large-v3-turbo", model_dir=tmp_path / "disk" / "models")
    assert transcribe.prepare_model_dir(config) == model


def test_missing_model_folder_is_created_for_the_download(tmp_path):
    config = VTDConfig(model_dir=tmp_path / "disk" / "models")
    assert transcribe.prepare_model_dir(config) == tmp_path / "disk" / "models" / "ggml-medium.bin"
    assert (tmp_path / "disk" / "models").is_dir()


def test_unreachable_model_folder_is_explained(tmp_path):
    (tmp_path / "not-a-disk").write_text("a file where the disk should be")
    config = VTDConfig(model_dir=tmp_path / "not-a-disk" / "models")
    with pytest.raises(VTDError, match="external disk.*--model-dir"):
        transcribe.prepare_model_dir(config)


def test_read_only_model_folder_is_explained(tmp_path, monkeypatch):
    # Faked rather than chmod-ed: root can write to a read-only folder.
    monkeypatch.setattr(transcribe.os, "access", lambda path, mode: False)
    with pytest.raises(VTDError, match="not writable"):
        transcribe.prepare_model_dir(VTDConfig(model_dir=tmp_path / "models"))


def test_a_model_already_there_is_used_even_when_the_folder_is_read_only(tmp_path):
    folder = tmp_path / "models"
    model = _model(folder)
    folder.chmod(0o500)
    try:
        assert transcribe.prepare_model_dir(VTDConfig(model_dir=folder)) == model
    finally:
        folder.chmod(0o700)


def test_unreachable_model_folder_stops_the_run_before_any_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "require_binary", lambda name, pkg=None: None)
    (tmp_path / "not-a-disk").write_text("")
    config = VTDConfig(model_dir=tmp_path / "not-a-disk" / "models")
    with pytest.raises(VTDError, match="external disk"):
        pipeline._preflight(["extract-audio", "transcribe"], config)
    # Neither a run that skips transcription nor Teams-captions mode needs the model.
    pipeline._preflight(["extract-audio"], config)
    config.teams_captions = True
    pipeline._preflight(["transcribe"], config)


def test_whisper_is_given_the_model_from_the_configured_folder(tmp_path, monkeypatch):
    model = _model(tmp_path / "models", "small")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "audio.wav").write_bytes(b"RIFF")
    commands = []

    class Finished:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            commands.append(cmd)

        def wait(self):
            return 0

    monkeypatch.setattr(transcribe, "require_binary", lambda name, pkg=None: None)
    monkeypatch.setattr("subprocess.Popen", Finished)
    monkeypatch.setattr(transcribe.urllib.request, "urlopen",
                        lambda url: pytest.fail("must not download a model that is there"))

    config = VTDConfig(whisper_model="small", model_dir=tmp_path / "models")
    transcribe.run(ws, config, Manifest())

    assert commands[0][commands[0].index("-m") + 1] == str(model)
