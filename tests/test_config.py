from argparse import Namespace

import pytest

from vtd import config as config_module
from vtd.config import resolve_config
from vtd.utils import VTDError


def test_defaults():
    cfg = resolve_config(Namespace())
    assert cfg.whisper_model == "medium"          # multilingual default
    assert cfg.language == "auto"
    assert cfg.frame_rate_seconds == 60
    assert cfg.content_type == "generic"
    assert cfg.output_language == "en"
    assert cfg.ocr_enabled is True
    assert cfg.silence_threshold_db == -30
    assert cfg.ai.enabled is False
    assert cfg.ai.provider == "anthropic"
    assert cfg.ai.api_key is None


def test_no_ocr_flag():
    cfg = resolve_config(Namespace(ocr=False))
    assert cfg.ocr_enabled is False


def test_ocr_flag_overrides_the_config_file(monkeypatch):
    monkeypatch.setattr(config_module, "_load_file_config",
                        lambda: {"defaults": {"ocr_enabled": False}})
    assert resolve_config(Namespace()).ocr_enabled is False
    assert resolve_config(Namespace(ocr=True)).ocr_enabled is True


def test_teams_captions_uses_five_second_frames():
    cfg = resolve_config(Namespace(teams_captions=True))
    assert cfg.teams_captions is True
    assert cfg.frame_rate_seconds == 5


def test_teams_captions_honors_explicit_frame_rate():
    cfg = resolve_config(Namespace(teams_captions=True, frame_rate=2))
    assert cfg.frame_rate_seconds == 2


def test_content_type_flag():
    cfg = resolve_config(Namespace(content_type="lecture"))
    assert cfg.content_type == "lecture"


def test_language_flag():
    cfg = resolve_config(Namespace(language="es"))
    assert cfg.language == "es"


def test_output_language_flag():
    cfg = resolve_config(Namespace(output_language="es"))
    assert cfg.output_language == "es"


def test_silence_knobs():
    cfg = resolve_config(Namespace(silence_threshold_db=-25, silence_min_s=1.0, section_gap_s=10.0))
    assert cfg.silence_threshold_db == -25
    assert cfg.silence_min_s == 1.0
    assert cfg.section_gap_s == 10.0


def test_cli_overrides(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ns = Namespace(
        whisper_model="small.en",
        frame_rate=30,
        with_ai=True,
        provider="anthropic",
        ai_model="claude-sonnet-4-6",
        api_key="sk-from-flag",
    )
    cfg = resolve_config(ns)
    assert cfg.whisper_model == "small.en"
    assert cfg.frame_rate_seconds == 30
    assert cfg.ai.enabled is True
    assert cfg.ai.model == "claude-sonnet-4-6"
    assert cfg.ai.api_key == "sk-from-flag"


def test_openai_provider_uses_openai_default_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ns = Namespace(
        with_ai=True,
        provider="openai",
        ai_model=None,
        api_key="sk-from-flag",
    )
    cfg = resolve_config(ns)
    assert cfg.ai.provider == "openai"
    assert cfg.ai.model == "gpt-5.4-mini"


def test_cli_provider_override_does_not_reuse_file_model(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_load_file_config",
        lambda: {"ai": {"provider": "anthropic", "model": "claude-opus-4-7"}},
    )
    ns = Namespace(
        with_ai=True,
        provider="openai",
        ai_model=None,
        api_key="sk-from-flag",
    )
    cfg = resolve_config(ns)
    assert cfg.ai.provider == "openai"
    assert cfg.ai.model == "gpt-5.4-mini"


def test_file_provider_model_is_honored(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_load_file_config",
        lambda: {"ai": {"provider": "openai", "model": "gpt-4o-mini"}},
    )
    ns = Namespace(with_ai=True, provider=None, ai_model=None, api_key="sk-from-flag")
    cfg = resolve_config(ns)
    assert cfg.ai.provider == "openai"
    assert cfg.ai.model == "gpt-4o-mini"


def test_env_falls_back_when_no_flag(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    ns = Namespace(with_ai=True, provider="anthropic", api_key=None)
    cfg = resolve_config(ns)
    assert cfg.ai.api_key == "sk-from-env"


def test_flag_beats_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    ns = Namespace(with_ai=True, provider="anthropic", api_key="sk-from-flag")
    cfg = resolve_config(ns)
    assert cfg.ai.api_key == "sk-from-flag"


def test_file_api_key_is_not_sent_to_another_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        config_module,
        "_load_file_config",
        lambda: {"ai": {"provider": "anthropic", "api_key": "sk-ant-file"}},
    )
    ns = Namespace(with_ai=True, provider="openai", ai_model=None, api_key=None)
    assert resolve_config(ns).ai.api_key is None

    ns = Namespace(with_ai=True, provider="anthropic", ai_model=None, api_key=None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_config(ns).ai.api_key == "sk-ant-file"


def test_same_provider_on_cli_keeps_file_model(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_load_file_config",
        lambda: {"ai": {"provider": "openai", "model": "gpt-custom"}},
    )
    ns = Namespace(with_ai=True, provider="openai", ai_model=None, api_key="sk-flag")
    assert resolve_config(ns).ai.model == "gpt-custom"


def test_explicit_zero_is_not_replaced_by_the_default(monkeypatch):
    monkeypatch.setattr(config_module, "_load_file_config", lambda: {})
    ns = Namespace(silence_threshold_db=0, section_gap_s=0.0, silence_min_s=0.0)
    cfg = resolve_config(ns)
    assert (cfg.silence_threshold_db, cfg.section_gap_s, cfg.silence_min_s) == (0, 0.0, 0.0)


@pytest.mark.parametrize("option, value", [
    ("frame_rate", 0), ("num_speakers", -1), ("min_frame_words", -1), ("min_section_s", -1.0),
    ("section_gap_s", -1.0), ("silence_min_s", -0.5), ("silence_threshold_db", 30),
])
def test_a_number_out_of_range_is_refused_not_quietly_changed(monkeypatch, option, value):
    monkeypatch.setattr(config_module, "_load_file_config", lambda: {})
    with pytest.raises(VTDError, match="Invalid value for"):
        resolve_config(Namespace(**{option: value}))


def test_broken_config_file_is_reported_not_ignored(tmp_path, monkeypatch):
    broken = tmp_path / "config.toml"
    broken.write_text("[defaults\nwhisper_model = ")
    monkeypatch.setattr(config_module, "CONFIG_PATH", broken)

    with pytest.raises(VTDError, match="config.toml"):
        resolve_config(Namespace())


@pytest.mark.parametrize("file_cfg,message", [
    ({"defaults": {"frame_rate_seconds": "fast"}}, "frame_rate_seconds"),
    ({"defaults": {"section_gap_s": "long"}}, "section_gap_s"),
    ({"defaults": {"content_type": "webinar"}}, "content_type"),
    ({"defaults": {"output_language": "fr"}}, "output_language"),
    ({"ai": {"provider": "no-such-provider"}}, "provider"),
    ({"defaults": ["not", "a", "table"]}, "defaults"),
    ({"defaults": {"frame_rate_seconds": float("inf")}}, "frame_rate_seconds"),
    ({"defaults": {"teams_captions": "false"}}, "teams_captions"),
    ({"defaults": {"ocr_enabled": "no"}}, "ocr_enabled"),
    ({"defaults": {"language": 5}}, "language"),
    ({"defaults": {"whisper_model": 5}}, "whisper_model"),
    ({"defaults": {"model_dir": 5}}, "model_dir"),
    ({"defaults": {"model_dir": "models"}}, "model_dir"),  # relative: differs per directory
    ({"defaults": {"model_dir": "~no-such-user-vtd/models"}}, "model_dir"),
    ({"ai": {"model": 5}}, "model"),
    ({"defaults": {"silence_min_s": float("nan")}}, "silence_min_s"),
    ({"defaults": {"silence_threshold_db": True}}, "silence_threshold_db"),
    ({"defaults": {"frame_rate_seconds": 2.7}}, "frame_rate_seconds"),
])
def test_bad_config_values_name_the_option(monkeypatch, file_cfg, message):
    monkeypatch.setattr(config_module, "_load_file_config", lambda: file_cfg)
    with pytest.raises(VTDError, match=message):
        resolve_config(Namespace())


def test_unknown_config_keys_are_warned_about(monkeypatch, capsys):
    monkeypatch.setattr(
        config_module, "_load_file_config",
        lambda: {"defaults": {"whisper_modle": "small"}, "ai": {"temperature": 0}},
    )
    resolve_config(Namespace())
    err = capsys.readouterr().err
    assert "defaults.whisper_modle" in err and "ai.temperature" in err


def test_teams_captions_can_be_set_in_the_config_file(monkeypatch):
    monkeypatch.setattr(config_module, "_load_file_config", lambda: {"defaults": {"teams_captions": True}})
    cfg = resolve_config(Namespace())
    assert cfg.teams_captions is True and cfg.frame_rate_seconds == 5


def test_config_file_in_another_encoding_is_reported(tmp_path, monkeypatch):
    latin = tmp_path / "config.toml"
    latin.write_bytes("# caf\xe9\n[defaults]\n".encode("latin-1"))
    monkeypatch.setattr(config_module, "CONFIG_PATH", latin)
    with pytest.raises(VTDError, match="config.toml"):
        resolve_config(Namespace())


def test_choices_from_the_file_are_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        config_module, "_load_file_config",
        lambda: {"defaults": {"content_type": "Meeting"}, "ai": {"provider": "OpenAI"}},
    )
    cfg = resolve_config(Namespace())
    assert (cfg.content_type, cfg.ai.provider) == ("meeting", "openai")


def test_model_dir_defaults_to_the_user_cache():
    assert resolve_config(Namespace()).model_dir == config_module.MODEL_CACHE


def test_model_dir_from_the_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config_module, "_load_file_config",
                        lambda: {"defaults": {"model_dir": "~/disk/models"}})
    assert resolve_config(Namespace()).model_dir == tmp_path / "disk" / "models"


def test_model_dir_flag_wins_and_is_taken_relative_to_the_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_load_file_config",
                        lambda: {"defaults": {"model_dir": "/Volumes/Disk/models"}})
    assert resolve_config(Namespace(model_dir="models")).model_dir == tmp_path / "models"


def test_ollama_provider_needs_no_key_and_has_its_own_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-ollama")
    cfg = resolve_config(Namespace(with_ai=True, provider="ollama"))
    assert cfg.ai.provider == "ollama"
    assert cfg.ai.model == "qwen3.6:35b"
    assert cfg.ai.api_key is None


def test_sections_are_split_before_the_ai_stage_would_cut_them():
    from vtd.ai.base import BODY_CHARS, TITLE_CHARS
    assert 0 < resolve_config(Namespace()).max_section_chars <= min(TITLE_CHARS, BODY_CHARS)


def test_max_section_chars_can_be_set_or_switched_off(monkeypatch):
    assert resolve_config(Namespace(max_section_chars=0)).max_section_chars == 0
    monkeypatch.setattr(config_module, "_load_file_config",
                        lambda: {"defaults": {"max_section_chars": 4000}})
    assert resolve_config(Namespace()).max_section_chars == 4000


@pytest.mark.parametrize("value", [-1, 10])
def test_max_section_chars_too_small_is_rejected(value):
    with pytest.raises(VTDError, match="max_section_chars"):
        resolve_config(Namespace(max_section_chars=value))
