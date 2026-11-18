import pytest

from vtd import cli, pipeline
from vtd import config as config_module
from vtd.cli import build_parser
from vtd.pipeline import STAGE_NAMES


def test_help_runs():
    parser = build_parser()
    # argparse exits on --help; capture by patching
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_required_source_positional():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_basic_parse():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--whisper-model", "small", "--frame-rate", "30"])
    assert args.source == "video.mp4"
    assert args.whisper_model == "small"
    assert args.model_dir is None
    assert parser.parse_args(["v.mp4", "--model-dir", "/Volumes/D/m"]).model_dir == "/Volumes/D/m"
    assert args.frame_rate == 30
    assert args.with_ai is False
    assert args.ocr is None  # unset: the config file may decide
    assert parser.parse_args(["v.mp4", "--no-ocr"]).ocr is False
    assert args.teams_captions is None  # unset: the config file may decide


def test_no_teams_captions_flag_overrides_the_config_file(monkeypatch):
    from vtd import config as config_module

    monkeypatch.setattr(
        config_module, "_load_file_config", lambda: {"defaults": {"teams_captions": True}}
    )
    args = build_parser().parse_args(["v.mp4", "--no-teams-captions"])
    assert config_module.resolve_config(args).teams_captions is False


def test_teams_captions_flag():
    args = build_parser().parse_args(["silent.mov", "--teams-captions"])
    assert args.teams_captions is True


def test_url_source_accepted():
    parser = build_parser()
    args = parser.parse_args(["https://youtu.be/dQw4w9WgXcQ"])
    assert args.source.startswith("https://")


def test_content_type_choices():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--content-type", "podcast"])
    assert args.content_type == "podcast"


def test_language_flag():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--language", "es"])
    assert args.language == "es"


def test_output_language_flag():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--output-language", "es"])
    assert args.output_language == "es"


def test_with_ai_flag():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--with-ai", "--api-key", "sk-abc", "--provider", "openai"])
    assert args.with_ai is True
    assert args.api_key == "sk-abc"
    assert args.provider == "openai"


def test_stage_names_known():
    expected = {
        "extract-audio", "extract-frames", "transcribe", "identify-speakers",
        "ocr-speakers", "detect-sections", "render-starter", "synthesize",
    }
    assert set(STAGE_NAMES) == expected


def _capture_pipeline(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda **kwargs: seen.update(kwargs))
    # The tools a URL run is checked for before the download are not what is under test.
    monkeypatch.setattr(cli, "check_before_download", lambda config, stages: None)
    return seen


def test_out_is_used_exactly_as_given(tmp_path, monkeypatch):
    seen = _capture_pipeline(monkeypatch)
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"x")

    assert cli.main([str(source), "--out", str(tmp_path / "results")]) == 0
    assert seen["workspace"] == tmp_path / "results"

    assert cli.main([str(source)]) == 0
    assert seen["workspace"] == tmp_path / "talk.mp4.vtd"


def test_only_synthesize_enables_ai(tmp_path, monkeypatch):
    seen = _capture_pipeline(monkeypatch)
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"x")

    assert cli.main([str(source), "--only", "synthesize"]) == 0
    assert seen["config"].ai.enabled is True


def test_url_workspace_is_created_in_the_current_directory(tmp_path, monkeypatch):
    seen = _capture_pipeline(monkeypatch)
    cached = tmp_path / "cache" / "Some_Talk.mp4"
    monkeypatch.setattr(cli, "_resolve_source", lambda source, verbose: cached)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["https://example.com/watch?v=1"]) == 0
    assert seen["workspace"] == tmp_path / "Some_Talk.mp4.vtd"


def test_config_errors_are_printed_not_raised(tmp_path, monkeypatch, capsys):
    from vtd import config as config_module

    _capture_pipeline(monkeypatch)
    monkeypatch.setattr(
        config_module, "_load_file_config", lambda: {"defaults": {"section_gap_s": "long"}}
    )
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"x")

    assert cli.main([str(source)]) == 1
    assert "section_gap_s" in capsys.readouterr().err


def test_stage_typo_fails_before_the_source_is_fetched(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve_source", lambda *a, **k: pytest.fail("must not download"))
    assert cli.main(["https://example.com/v", "--only", "transcribee"]) == 1
    assert "Unknown stage(s): transcribee" in capsys.readouterr().err


@pytest.mark.parametrize("spec", ["", ",", " , "])
def test_an_only_list_that_names_no_stage_is_refused_and_runs_nothing(monkeypatch, capsys, spec):
    # An empty shell variable must not turn into "run everything", paid AI included.
    monkeypatch.setattr(cli, "_resolve_source", lambda *a, **k: pytest.fail("must not run"))
    assert cli.main(["talk.mp4", "--only", spec]) == 1
    assert "--only names no stage" in capsys.readouterr().err


def test_a_missing_tool_is_reported_before_a_url_is_downloaded(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve_source", lambda *a, **k: pytest.fail("must not download"))
    monkeypatch.setattr(config_module, "_load_file_config", lambda: {})
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    monkeypatch.setattr("vtd.utils.shutil.which", lambda name: None)
    assert cli.main(["https://example.com/v"]) == 1
    assert "ffmpeg" in capsys.readouterr().err


def test_a_missing_api_key_is_reported_before_a_url_is_downloaded(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve_source", lambda *a, **k: pytest.fail("must not download"))
    monkeypatch.setattr(config_module, "_load_file_config", lambda: {})
    monkeypatch.setattr(pipeline, "_preflight", lambda to_run, config: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["https://example.com/v", "--with-ai"]) == 1
    assert "no API key found" in capsys.readouterr().err
