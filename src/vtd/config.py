"""Config + CLI-flag resolution.

Resolution order (highest priority first):
  1. CLI flag
  2. Environment variable (API keys only)
  3. ~/.config/vtd/config.toml
  4. Built-in default
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .utils import VTDError, emit

CONFIG_PATH = Path.home() / ".config" / "vtd" / "config.toml"
MODEL_CACHE = Path.home() / ".cache" / "vtd" / "models"

# Everything vtd knows about each AI provider, in one place.
PROVIDERS = {
    "anthropic": {"default_model": "claude-opus-4-7", "env_var": "ANTHROPIC_API_KEY"},
    "openai": {"default_model": "gpt-5.4-mini", "env_var": "OPENAI_API_KEY"},
    # A model on this machine: no key (env_var None), nothing sent to a vendor.
    "ollama": {"default_model": "qwen3.6:35b", "env_var": None},
}
DEFAULT_PROVIDER = "anthropic"
CONTENT_TYPES = ("generic", "meeting", "lecture", "podcast")
# Output languages for AI-authored documents: code → name used in prompts.
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
}

DEFAULT_SPEAKER_THRESHOLD = 0.7
MIN_SECTION_CHARS = 500  # smaller sections would be a few sentences each

_DEFAULTS_KEYS = {
    "whisper_model", "model_dir", "language", "teams_captions", "frame_rate_seconds", "content_type",
    "output_language", "ocr_enabled", "silence_threshold_db", "silence_min_s", "section_gap_s",
    "max_section_chars", "min_section_s", "min_frame_words", "pdf", "references",
    "speakers", "num_speakers", "speaker_threshold",
}
_AI_KEYS = {"provider", "model", "api_key"}


@dataclass
class AIConfig:
    enabled: bool = False
    provider: str = DEFAULT_PROVIDER
    model: str = PROVIDERS[DEFAULT_PROVIDER]["default_model"]
    api_key: str | None = None  # never logged, never persisted by vtd


@dataclass
class VTDConfig:
    whisper_model: str = "medium"             # multilingual by default
    model_dir: Path = field(default_factory=lambda: MODEL_CACHE)  # where Whisper models are kept
    language: str = "auto"                    # whisper-cli -l value; 'auto' detects
    frame_rate_seconds: int = 60
    workspace_suffix: str = ".vtd"
    output_doc_name: str = "notes.md"
    starter_doc_name: str = "notes.starter.md"
    content_type: str = "generic"             # one of CONTENT_TYPES
    output_language: str = "en"               # one of LANGUAGE_NAMES
    ocr_enabled: bool = True                  # auto-disabled when no video stream
    teams_captions: bool = False              # OCR visible Teams live captions
    silence_threshold_db: int = -30           # ffmpeg silencedetect noise level
    silence_min_s: float = 2.0                # min silence duration to count
    section_gap_s: float = 8.0                # silence longer than this = section boundary
    # A section with more transcript than this is split at a pause; 0 = never split. The
    # default is the longest input the AI stage sends for a section title, so nothing is cut.
    max_section_chars: int = 6000
    min_section_s: float = 120.0              # a shorter section joins its neighbour; 0 = keep
    # Fewest readable words for a frame to be shown to the model and cited (a face with a
    # name tag has three or four); 0 = every frame, e.g. for diagrams or photos.
    min_frame_words: int = 8
    pdf: bool = False                         # also write the documents as PDF (needs pandoc)
    references: bool = False                  # cite the works referred to, checked in catalogs
    speakers: bool = False                    # tell who speaks when (needs sherpa-onnx)
    num_speakers: int = 0                     # how many people speak; 0 = work it out
    # How different two voices must be to count as two people when num_speakers is 0:
    # lower finds more speakers, higher merges similar voices.
    speaker_threshold: float = DEFAULT_SPEAKER_THRESHOLD
    ai: AIConfig = field(default_factory=AIConfig)


def _load_file_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except (OSError, ValueError) as e:  # ValueError covers bad TOML and bad encoding
        raise VTDError(f"Cannot read {CONFIG_PATH}: {e}") from e


def _table(file_cfg: dict, name: str, known: set[str]) -> dict:
    """A `[name]` table from the config file, warning about keys vtd does not use."""
    table = file_cfg.get(name) or {}
    if not isinstance(table, dict):
        raise VTDError(f"`{name}` in {CONFIG_PATH} must be a table, e.g. `[{name}]`.")
    for key in sorted(set(table) - known):
        emit(f"Ignoring unknown key `{name}.{key}` in {CONFIG_PATH}", kind="warn")
    return table


def _first(*values):
    """First value that is not None, so an explicit 0 is not mistaken for 'unset'."""
    return next((v for v in values if v is not None), None)


def _number(cast, key: str, *values):
    """`cast` the first set value, naming the option when it is not a number."""
    value = _first(*values)
    try:
        if isinstance(value, bool) or value != value:  # true/false and NaN are not numbers
            raise ValueError(value)
        number = cast(value)
        if cast is int and number != float(value):
            raise ValueError(value)  # 2.7 is not a whole number of seconds
        return number
    except (TypeError, ValueError, OverflowError) as e:
        raise VTDError(f"Invalid value for `{key}`: {value!r} (expected a number).") from e


def _at_least(key: str, value, low):
    """`value`, which must be `low` or more."""
    if value < low:
        raise VTDError(f"Invalid value for `{key}`: {value} (use {low} or more).")
    return value


def _text(key: str, *values) -> str:
    """The first set value, which must be text (a bare number in the file is a mistake)."""
    value = next(v for v in values if v)
    if not isinstance(value, str):
        raise VTDError(f"Invalid value for `{key}`: {value!r} (expected text in quotes).")
    return value


def _flag(key: str, table: dict, default: bool) -> bool:
    """A true/false key from the config file; a quoted "false" must not read as true."""
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise VTDError(f"Invalid value for `{key}`: {value!r} (expected true or false).")
    return value


def _choice(key: str, value: str, allowed) -> str:
    value = value.lower() if isinstance(value, str) else value
    if value not in allowed:
        raise VTDError(
            f"Invalid value for `{key}`: {value!r}. Choose one of: {', '.join(allowed)}."
        )
    return value


def _expand(value: str) -> Path:
    try:
        return Path(value).expanduser()
    except RuntimeError as e:  # `~someone` for a user that does not exist
        raise VTDError(f"Invalid value for `model_dir`: {value!r} (cannot expand `~`).") from e


def _model_dir(cli_value, file_value, default: Path) -> Path:
    """Folder holding the Whisper models, e.g. on an external disk."""
    if cli_value:
        # Typed in a shell, so a relative path means "relative to where I am".
        return _expand(_text("model_dir", cli_value)).absolute()
    if not file_value:
        return default
    path = _expand(_text("model_dir", file_value))
    if not path.is_absolute():
        # A relative path would point somewhere else from every directory vtd is run in.
        raise VTDError(
            f"Invalid value for `model_dir`: {file_value!r} "
            "(expected a full path, or one starting with `~`)."
        )
    return path


def resolve_config(cli_args) -> VTDConfig:
    """Merge defaults < file < env < CLI into a single VTDConfig."""
    file_cfg = _load_file_config()
    defaults = _table(file_cfg, "defaults", _DEFAULTS_KEYS)
    ai_file = _table(file_cfg, "ai", _AI_KEYS)

    cfg = VTDConfig()
    cfg.whisper_model = _text(
        "whisper_model",
        getattr(cli_args, "whisper_model", None), defaults.get("whisper_model"), cfg.whisper_model,
    )
    cfg.model_dir = _model_dir(
        getattr(cli_args, "model_dir", None), defaults.get("model_dir"), cfg.model_dir,
    )
    cfg.language = _text(
        "language", getattr(cli_args, "language", None), defaults.get("language"), cfg.language,
    )
    cfg.teams_captions = bool(_first(
        getattr(cli_args, "teams_captions", None),  # --teams-captions / --no-teams-captions
        _flag("teams_captions", defaults, False),
    ))
    cfg.frame_rate_seconds = _at_least("frame_rate_seconds", _number(
        int, "frame_rate_seconds",
        getattr(cli_args, "frame_rate", None),
        defaults.get("frame_rate_seconds"),
        5 if cfg.teams_captions else cfg.frame_rate_seconds,
    ), 1)
    cfg.content_type = _choice("content_type", (
        getattr(cli_args, "content_type", None)
        or defaults.get("content_type")
        or cfg.content_type
    ), CONTENT_TYPES)
    cfg.output_language = _choice("output_language", (
        getattr(cli_args, "output_language", None)
        or defaults.get("output_language")
        or cfg.output_language
    ), tuple(LANGUAGE_NAMES))
    cfg.ocr_enabled = bool(_first(
        getattr(cli_args, "ocr", None),  # --ocr / --no-ocr
        _flag("ocr_enabled", defaults, cfg.ocr_enabled),
    ))
    cfg.silence_threshold_db = _number(
        int, "silence_threshold_db",
        getattr(cli_args, "silence_threshold_db", None),
        defaults.get("silence_threshold_db"),
        cfg.silence_threshold_db,
    )
    if cfg.silence_threshold_db > 0:
        raise VTDError(
            f"Invalid value for `silence_threshold_db`: {cfg.silence_threshold_db} "
            "(a level in decibels below full volume: use 0 or less, e.g. -30)."
        )
    cfg.silence_min_s = _at_least("silence_min_s", _number(
        float, "silence_min_s",
        getattr(cli_args, "silence_min_s", None),
        defaults.get("silence_min_s"),
        cfg.silence_min_s,
    ), 0)
    cfg.section_gap_s = _at_least("section_gap_s", _number(
        float, "section_gap_s",
        getattr(cli_args, "section_gap_s", None),
        defaults.get("section_gap_s"),
        cfg.section_gap_s,
    ), 0)

    cfg.max_section_chars = _number(
        int, "max_section_chars",
        getattr(cli_args, "max_section_chars", None),
        defaults.get("max_section_chars"),
        cfg.max_section_chars,
    )
    if cfg.max_section_chars < 0 or 0 < cfg.max_section_chars < MIN_SECTION_CHARS:
        raise VTDError(
            f"Invalid value for `max_section_chars`: {cfg.max_section_chars} "
            f"(use 0 to never split a section, or at least {MIN_SECTION_CHARS})."
        )

    cfg.min_section_s = _at_least("min_section_s", _number(
        float, "min_section_s",
        getattr(cli_args, "min_section_s", None),
        defaults.get("min_section_s"),
        cfg.min_section_s,
    ), 0)

    cfg.min_frame_words = _at_least("min_frame_words", _number(
        int, "min_frame_words",
        getattr(cli_args, "min_frame_words", None),
        defaults.get("min_frame_words"),
        cfg.min_frame_words,
    ), 0)

    cfg.speakers = bool(_first(
        getattr(cli_args, "speakers", None), _flag("speakers", defaults, cfg.speakers)))
    cfg.num_speakers = _at_least("num_speakers", _number(
        int, "num_speakers",
        getattr(cli_args, "num_speakers", None), defaults.get("num_speakers"), cfg.num_speakers,
    ), 0)
    cfg.speaker_threshold = _number(
        float, "speaker_threshold",
        getattr(cli_args, "speaker_threshold", None),
        defaults.get("speaker_threshold"),
        cfg.speaker_threshold,
    )
    if not 0 < cfg.speaker_threshold < 2:
        raise VTDError(
            f"Invalid value for `speaker_threshold`: {cfg.speaker_threshold} "
            "(use a number between 0 and 2; lower finds more speakers)."
        )

    cfg.pdf = bool(_first(getattr(cli_args, "pdf", None), _flag("pdf", defaults, cfg.pdf)))
    cfg.references = bool(_first(
        getattr(cli_args, "references", None), _flag("references", defaults, cfg.references)))

    cfg.ai.enabled = bool(getattr(cli_args, "with_ai", False))
    cli_provider = getattr(cli_args, "provider", None)
    cli_model = getattr(cli_args, "ai_model", None)
    file_provider = ai_file.get("provider")
    file_model = ai_file.get("model")

    # The file's model and api_key belong to the file's provider (or the default one).
    file_owner = file_provider or cfg.ai.provider
    cfg.ai.provider = _choice(
        "provider", cli_provider or file_provider or cfg.ai.provider, tuple(PROVIDERS)
    )
    provider = PROVIDERS[cfg.ai.provider]
    file_applies = cfg.ai.provider == file_owner
    cfg.ai.model = _text(
        "model", cli_model, file_model if file_applies else None, provider["default_model"],
    )
    cfg.ai.api_key = (
        getattr(cli_args, "api_key", None)
        or (os.environ.get(provider["env_var"]) if provider["env_var"] else None)
        or (ai_file.get("api_key") if file_applies else None)
    )

    return cfg
