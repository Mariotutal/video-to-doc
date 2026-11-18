"""Stage orchestration. Loads/saves the manifest and runs stages in order."""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from . import export_pdf
from . import manifest as mf
from .config import PROVIDERS, VTDConfig
from .stages import (
    detect_sections,
    extract_audio,
    extract_frames,
    identify_speakers,
    ocr_speakers,
    render,
    transcribe,
)
from .utils import VTDError, emit, ffprobe_info, require_binary

# Stage name → callable
StageFn = Callable[[Path, VTDConfig, mf.Manifest], None]

STAGES: dict[str, StageFn] = {
    "extract-audio": extract_audio.run,
    "extract-frames": extract_frames.run,
    "transcribe": transcribe.run,
    "identify-speakers": identify_speakers.run,
    "ocr-speakers": ocr_speakers.run,
    "detect-sections": detect_sections.run,
    "render-starter": render.run_starter,
}

AI_STAGE_NAME = "synthesize"
BACKUP_DIR = ".rerun-backup"
STAGE_NAMES = list(STAGES.keys()) + [AI_STAGE_NAME]


def _stage_outputs(name: str, config: VTDConfig) -> list[str]:
    """Workspace artifacts a stage owns; set aside before the stage is re-run."""
    teams = config.teams_captions
    return {
        "extract-audio": ["audio.wav"],
        # frame_text.json describes these frames, so it goes when they are replaced.
        "extract-frames": ["frames"] + ([] if teams else ["frame_text.json"]),
        # In Teams mode the caption transcriber also writes speakers.json.
        "transcribe": ["transcript.txt", "transcript.vtt", "transcript.raw.txt",
                       "transcript.raw.vtt"] + (["speakers.json"] if teams else []),
        "identify-speakers": [identify_speakers.TURNS_NAME, identify_speakers.NAMES_NAME],
        "ocr-speakers": [] if teams else ["speakers.json", "frame_text.json"],
        "detect-sections": ["sections.json"],
        "render-starter": [config.starter_doc_name],
    }.get(name, [])


def _stage_options(name: str, config: VTDConfig) -> dict:
    """The options that shape a stage's result. Recorded when the stage completes."""
    teams = config.teams_captions
    if name == "extract-frames":
        return {"frame_rate_seconds": config.frame_rate_seconds, "teams_captions": teams}
    if name == "transcribe":
        if teams:
            return {"teams_captions": True, "frame_rate_seconds": config.frame_rate_seconds}
        return {"teams_captions": False, "whisper_model": config.whisper_model,
                "language": config.language}
    if name == "identify-speakers":
        return {"num_speakers": config.num_speakers,
                "speaker_threshold": config.speaker_threshold}
    if name == "ocr-speakers":
        # Without tesseract the stage finds nobody; recording that lets --resume
        # redo it once the tool is installed.
        return {"teams_captions": teams, "tesseract": shutil.which("tesseract") is not None}
    if name == "detect-sections":
        return {"teams_captions": teams, "silence_threshold_db": config.silence_threshold_db,
                "silence_min_s": config.silence_min_s, "section_gap_s": config.section_gap_s,
                "max_section_chars": config.max_section_chars,
                "min_section_s": config.min_section_s}
    if name == "render-starter":
        return {"output_language": config.output_language}  # its speaker marks are in it
    if name == AI_STAGE_NAME:
        return {"content_type": config.content_type, "output_language": config.output_language,
                "provider": config.ai.provider, "model": config.ai.model,
                "min_frame_words": config.min_frame_words, "references": config.references}
    return {}


# What an option was before vtd recorded it: a stage completed by a version that did
# not have the option was made without it.
_BEFORE_RECORDED = {AI_STAGE_NAME: {"references": False}}


def _recorded(name: str, manifest: mf.Manifest) -> dict:
    recorded = manifest.stage_options.get(name) or {}
    return {**_BEFORE_RECORDED.get(name, {}), **recorded} if recorded else recorded


def _changed(name: str, config: VTDConfig, manifest: mf.Manifest) -> str:
    """Human-readable differences between a stage's recorded options and the current ones."""
    recorded, current = _recorded(name, manifest), _stage_options(name, config)
    return ", ".join(
        f"{key} {value!r} → {current.get(key)!r}"
        for key, value in recorded.items() if current.get(key) != value
    )


def _is_current(name: str, config: VTDConfig, manifest: mf.Manifest) -> bool:
    """Whether a stage is done and its result was made with the options now requested.

    Only what was recorded is compared: a stage completed before an option (or any
    option) was being recorded is trusted for it.
    """
    if not manifest.has_stage(name):
        return False
    current = _stage_options(name, config)
    return all(current.get(key) == value for key, value in _recorded(name, manifest).items())


def _needs(name: str, config: VTDConfig, manifest: mf.Manifest) -> list[str]:
    """The stages whose outputs a stage reads."""
    derived = ["extract-frames", "transcribe", "identify-speakers", "ocr-speakers",
               "detect-sections"]
    # An existing transcript depends on what it was built from, not on the mode now requested.
    teams = manifest.stage_options.get("transcribe", {}).get(
        "teams_captions", config.teams_captions
    )
    return {
        "transcribe": ["extract-frames"] if teams else ["extract-audio"],
        "identify-speakers": ["extract-audio"],
        "ocr-speakers": ["extract-frames"],
        "detect-sections": ["extract-audio", "transcribe"],
        "render-starter": derived,
        AI_STAGE_NAME: derived,
    }.get(name, [])


def _dependents(name: str, config: VTDConfig, manifest: mf.Manifest) -> list[str]:
    """Every stage built, directly or indirectly, on `name`'s outputs."""
    found: list[str] = []
    for stage in STAGE_NAMES:  # pipeline order, so one pass reaches indirect dependents
        if any(need == name or need in found for need in _needs(stage, config, manifest)):
            found.append(stage)
    return found


def _check_inputs(name: str, config: VTDConfig, manifest: mf.Manifest, in_run: list[str]) -> None:
    """Refuse to build a stage on results made with other options than those requested.

    Inputs that are part of this run were already brought up to date, so this only
    triggers with --only lists that leave an input out.
    """
    if name == "transcribe":  # the transcript about to be made, not the one being replaced
        needs = ["extract-frames"] if config.teams_captions else ["extract-audio"]
    else:
        needs = _needs(name, config, manifest)
    for need in needs:
        if need in in_run or not manifest.has_stage(need) or _is_current(need, config, manifest):
            continue
        raise VTDError(
            f"{name} would read {need} results made with different options "
            f"({_changed(need, config, manifest)}). Add {need} to --only, pass the original "
            "values, or use --resume to redo what changed."
        )


def _recover_backup(workspace: Path) -> None:
    """Put back outputs a killed run left set aside, so they are never lost."""
    backup = workspace / BACKUP_DIR
    if backup.is_dir():
        for item in backup.iterdir():
            if not (workspace / item.name).exists():
                item.replace(workspace / item.name)
        shutil.rmtree(backup, ignore_errors=True)


def _preflight(to_run: list[str], config: VTDConfig) -> None:
    """Fail before any work starts if a tool a later stage needs is missing."""
    if config.pdf:
        export_pdf.pdf_tools()
    if {"extract-audio", "extract-frames", "detect-sections"} & set(to_run):
        require_binary("ffmpeg", "ffmpeg")
    if "transcribe" in to_run:
        if config.teams_captions:
            require_binary("tesseract", "tesseract")
        else:
            require_binary("whisper-cli", "whisper-cpp")
            transcribe.prepare_model_dir(config)  # e.g. an external disk that is not connected
    if "identify-speakers" in to_run:
        identify_speakers.require()


def require_api_key(config: VTDConfig) -> None:
    """Fail when the chosen provider needs a key and none was given."""
    env_hint = PROVIDERS[config.ai.provider]["env_var"]
    if env_hint and not config.ai.api_key:  # a local provider has no key
        raise VTDError(
            f"AI synthesis requested but no API key found. "
            f"Set ${env_hint}, pass --api-key, or set `provider = \"{config.ai.provider}\"` "
            "and `api_key` under `[ai]` in ~/.config/vtd/config.toml."
        )


def _preflight_ai(config: VTDConfig, workspace: Path) -> None:
    """Fail before any work starts if the AI notes could not be written at the end:
    no key, the provider's package not installed, Ollama not running or without the model."""
    require_api_key(config)
    from . import ai_synth
    ai_synth._build_provider(config, workspace)


def check_before_download(config: VTDConfig, stages: list[str] | None) -> None:
    """What can be checked before the source is known: the tools and the AI key.

    A first run needs every tool its stages use, so nothing is fetched to find out
    afterwards that one is missing.
    """
    to_run = list(STAGES) if stages is None else [s for s in stages if s in STAGES]
    if config.teams_captions:
        to_run = [s for s in to_run if s not in ("extract-audio", "identify-speakers")]
    elif not config.speakers:
        to_run = [s for s in to_run if s != "identify-speakers"]
    _preflight(to_run, config)
    if config.ai.enabled and (stages is None or AI_STAGE_NAME in stages):
        require_api_key(config)


def _discard_outputs(workspace: Path, name: str, config: VTDConfig, manifest: mf.Manifest) -> None:
    """Forget a stage that this run will not execute, so its old results are not reused."""
    for rel in _stage_outputs(name, config):
        target = workspace / rel
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    manifest.reset_stage(name)


def _run_fresh(workspace: Path, name: str, config: VTDConfig, manifest: mf.Manifest) -> None:
    """Run a stage from scratch so it reflects the current options and inputs.

    Previous outputs are set aside first and put back if the stage fails or is
    interrupted, so a run that cannot succeed never destroys earlier results.
    On success, every stage built on this one is marked as not done.
    """
    was_done = manifest.has_stage(name)
    old_options = manifest.stage_options.get(name)
    outputs = _stage_outputs(name, config)
    if name == "transcribe" and (old_options or {}).get("teams_captions"):
        outputs = [*outputs, "speakers.json"]  # written by the Teams transcript being replaced
    backup = workspace / BACKUP_DIR
    try:
        for rel in outputs:
            if (workspace / rel).exists():
                backup.mkdir(exist_ok=True)
                (workspace / rel).replace(backup / rel)
        dependents = _dependents(name, config, manifest)  # before this stage's record changes
        manifest.reset_stage(name)
        mf.save(workspace, manifest)
        STAGES[name](workspace, config, manifest)
        for dependent in dependents:
            manifest.reset_stage(dependent)  # it was built from the outputs just replaced
    except BaseException:
        # Put back whatever was set aside. If this is interrupted too, the rest stays
        # in the backup directory and the next run recovers it.
        for item in (list(backup.iterdir()) if backup.is_dir() else []):
            target = workspace / item.name
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            item.replace(target)
        if was_done:
            manifest.mark_stage(name, old_options)
            mf.save(workspace, manifest)
        shutil.rmtree(backup, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _config_dict_for_manifest(cfg: VTDConfig) -> dict:
    d = asdict(cfg)
    d["model_dir"] = str(cfg.model_dir)  # a Path cannot be written as JSON
    # Never persist the API key.
    if "ai" in d and isinstance(d["ai"], dict):
        d["ai"].pop("api_key", None)
    return d


def _stages_to_run(
    requested: list[str] | None,
    config: VTDConfig,
    manifest: mf.Manifest,
) -> list[str]:
    """Compute the effective stage list, accounting for input characteristics and config."""
    if requested is None:
        requested = list(STAGE_NAMES)
        if not config.ai.enabled:
            requested = [s for s in requested if s != AI_STAGE_NAME]
        elif AI_STAGE_NAME not in requested:
            requested.append(AI_STAGE_NAME)
    else:
        requested = [s for s in STAGE_NAMES if s in requested]  # pipeline order, not typed order

    if config.teams_captions and "extract-audio" in requested:
        requested.remove("extract-audio")
    if not config.teams_captions and {"transcribe", "extract-frames"} <= set(requested):
        # Whisper reads only the audio. Sampling frames after it means a failed transcription
        # (the slow, fallible step) never replaces frames a Teams transcript was read from.
        requested.remove("extract-frames")
        requested.insert(requested.index("transcribe") + 1, "extract-frames")

    # Audio-only sources skip frame and OCR stages.
    if not manifest.has_video:
        skipped_for_audio_only = ["extract-frames", "ocr-speakers"]
        before = list(requested)
        requested = [s for s in requested if s not in skipped_for_audio_only]
        for s in skipped_for_audio_only:
            if s in before:
                emit(f"{s}: skipped (no video stream)", kind="info")
    # Speakers are identified from the audio, and only when asked for.
    if "identify-speakers" in requested:
        if config.teams_captions:
            requested.remove("identify-speakers")
            if config.speakers:
                emit("identify-speakers: skipped (Teams captions already name the speakers)",
                     kind="info")
        elif not config.speakers:
            requested.remove("identify-speakers")
    # OCR disabled by config or flag.
    if not config.ocr_enabled and "ocr-speakers" in requested:
        emit("ocr-speakers: skipped (--no-ocr / ocr_enabled = false)", kind="info")
        requested = [s for s in requested if s != "ocr-speakers"]

    return requested


def run_pipeline(
    *,
    source: Path,
    workspace: Path,
    config: VTDConfig,
    resume: bool,
    only_stages: list[str] | None,
    verbose: bool,
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    _recover_backup(workspace)
    manifest = mf.load(workspace)

    if not manifest.source_video:
        info = ffprobe_info(source)
        manifest.source_video = str(source.resolve())
        manifest.source_sha256_prefix = mf.video_hash_prefix(source)
        manifest.duration_seconds = info["duration"]
        manifest.width = info["width"]
        manifest.height = info["height"]
        manifest.has_video = info["has_video"]
        manifest.has_audio = info["has_audio"]
        manifest.created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    elif (
        manifest.source_sha256_prefix
        and mf.video_hash_prefix(source) != manifest.source_sha256_prefix
    ):
        raise VTDError(
            f"Workspace {workspace} was created from a different source "
            f"({manifest.source_video}). Use another --out or delete the workspace."
        )
    else:
        manifest.source_video = str(source.resolve())  # same file, possibly moved

    if config.teams_captions and not manifest.has_video:
        raise VTDError(
            "Teams captions mode requires a video stream "
            "(pass --no-teams-captions if it is enabled in the config file)."
        )
    if not manifest.has_audio and not config.teams_captions:
        raise VTDError(f"No audio stream in {source} — vtd can't transcribe it.")
    mf.save(workspace, manifest)

    effective = _stages_to_run(only_stages, config, manifest)
    pending = [n for n in effective if not (resume and _is_current(n, config, manifest))]
    _preflight([n for n in pending if n in STAGES], config)
    # Redoing any stage resets the notes built on it, so the AI stage runs then too.
    if AI_STAGE_NAME in effective and config.ai.enabled and pending:
        _preflight_ai(config, workspace)
    emit(f"Workspace: {workspace}")
    if verbose:
        if config.teams_captions:
            media_kind = "video+Teams captions"
        else:
            media_kind = "audio+video" if manifest.has_video else "audio-only"
        emit(f"Input: {media_kind}; language={config.language}; model={config.whisper_model}")
        emit(f"Pipeline stages: {', '.join(effective)}")

    if only_stages is None:
        # A full run must not reuse results of stages it now skips (e.g. --no-ocr).
        # audio.wav is kept: it stays valid for the source, and the transcript made
        # from it would otherwise have to be redone if this run fails.
        for name in STAGES:
            if name in effective or name == "extract-audio":
                continue
            if manifest.has_stage(name):
                for dependent in _dependents(name, config, manifest):
                    manifest.reset_stage(dependent)
            _discard_outputs(workspace, name, config, manifest)
        mf.save(workspace, manifest)

    for name in effective:  # before anything runs, so a refusal costs nothing
        if not (resume and _is_current(name, config, manifest)):
            _check_inputs(name, config, manifest, effective)

    complete = True
    for name in effective:
        if name == AI_STAGE_NAME:
            complete = _run_ai_stage(workspace, config, manifest, resume=resume)
            continue
        # Decided stage by stage: an earlier stage in this run may have invalidated this one.
        if resume and _is_current(name, config, manifest):
            unchecked = name not in manifest.stage_options
            note = " (by an earlier vtd; options not checked)" if unchecked else ""
            emit(f"{name}: already done{note} — skipping", kind="info")
            continue
        if resume and manifest.has_stage(name):
            emit(f"{name}: options changed ({_changed(name, config, manifest)}) — redoing")
        emit(f"{name}: running…", kind="info")
        _run_fresh(workspace, name, config, manifest)
        manifest.mark_stage(name, _stage_options(name, config))
        manifest.config_used = _config_dict_for_manifest(config)
        mf.save(workspace, manifest)
        emit(f"{name}: done", kind="done")

    if config.pdf:
        export_pdf.export_pdfs(workspace, config, manifest)
    if complete:
        emit(f"All requested stages complete: {workspace}", kind="done")
    else:
        emit(f"Finished with gaps in {config.output_doc_name}: {workspace}", kind="warn")


def _run_ai_stage(
    workspace: Path,
    config: VTDConfig,
    manifest: mf.Manifest,
    *,
    resume: bool,
) -> bool:
    """Write the AI notes. Returns False when they were written with parts missing."""
    if not config.ai.enabled:
        return True
    require_api_key(config)
    if resume and _is_current(AI_STAGE_NAME, config, manifest):
        emit(f"{AI_STAGE_NAME}: already done — skipping", kind="info")
        return True

    from . import ai_synth
    emit(f"{AI_STAGE_NAME}: running with provider={config.ai.provider} model={config.ai.model}")
    incomplete = ai_synth.synthesize(workspace, config, manifest)
    manifest.config_used = _config_dict_for_manifest(config)
    if incomplete:
        # Not done, so --resume asks again for these parts (finished answers are cached).
        manifest.reset_stage(AI_STAGE_NAME)
        emit(
            f"{config.output_doc_name} was written, but these parts are missing or incomplete: "
            f"{', '.join(incomplete)}. Run the same command again with --resume added to "
            "retry them.",
            kind="warn",
        )
        mf.save(workspace, manifest)
        return False
    manifest.mark_stage(AI_STAGE_NAME, _stage_options(AI_STAGE_NAME, config))
    mf.save(workspace, manifest)
    emit(f"{AI_STAGE_NAME}: done", kind="done")
    return True
