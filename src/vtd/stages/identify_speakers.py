"""Tell who is speaking when (`--speakers`), with sherpa-onnx. Everything runs locally.

The result says "voice 1 spoke from here to there", not who voice 1 is: a name is
only known when someone says it, which the AI stage may work out afterwards.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import tarfile
import urllib.request
import wave
from pathlib import Path

from ..config import VTDConfig
from ..manifest import Manifest
from ..utils import VTDError, emit, read_json
from ..vtt import is_spoken

TURNS_NAME = "speaker_turns.json"
NAMES_NAME = "speaker_names.json"  # written by the AI stage; may be corrected by hand

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = (
    f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDING_URL = f"{_RELEASES}/speaker-recongition-models/nemo_en_titanet_small.onnx"  # sic
EMBEDDING_MODEL = "nemo_en_titanet_small.onnx"
EMBEDDING_SHA256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"

# A voice heard for less than this in the whole recording is an interjection ("thanks",
# "yes") that could not be matched to anyone, and is left without a speaker.
MIN_SPEAKER_S = 20.0
# A voice that never speaks this long at a stretch, and is this small beside the voice
# its turns fall inside, is pieces of that voice which the grouping set apart.
MIN_TURN_S = 8.0
PIECES_SHARE = 0.03
# Pieces are a couple of seconds long on average; a question or an answer is longer.
PIECES_MEAN_S = 3.0
# Someone who pauses this long in the middle of their own talk is still the one talking.
PIECES_GAP_S = 30.0
# A piece of a dropped voice this close to the same speaker on both sides is theirs.
JOIN_GAP_S = 2.0


def require() -> None:
    """Fail with what to install when the optional packages are missing."""
    try:
        import numpy  # noqa: F401
        import sherpa_onnx  # noqa: F401
    except ImportError as e:
        raise VTDError(
            "--speakers needs the sherpa-onnx and numpy packages. Run `pip install -e "
            ".[speakers]` from the project root, or `pip install sherpa-onnx numpy`."
        ) from e


def model_dir(config: VTDConfig) -> Path:
    return config.model_dir / "speakers"


def _fetch(url: str, target: Path) -> None:
    emit(f"Downloading speaker model {url} → {target}")
    partial = target.with_name(target.name + ".partial")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
    except (OSError, http.client.HTTPException) as e:  # a connection cut off mid-file too
        raise VTDError(f"Failed to download the speaker model {url}: {e}") from e
    finally:
        partial.unlink(missing_ok=True)  # also after Ctrl-C; a no-op once moved into place


def models(config: VTDConfig) -> tuple[Path, Path]:
    """Paths of the segmentation and voice models, downloaded on first use."""
    folder = model_dir(config)
    segmentation, embedding = folder / SEGMENTATION_MODEL, folder / EMBEDDING_MODEL
    if segmentation.is_file() and embedding.is_file():
        return segmentation, embedding
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise VTDError(
            f"The speaker models are not in {folder} and that folder cannot be used for a "
            f"download ({e}). If it is on an external disk, check the disk is connected, or "
            "pick another folder with --model-dir."
        ) from e
    if not embedding.is_file():
        _fetch(EMBEDDING_URL, embedding)
        digest = hashlib.sha256(embedding.read_bytes()).hexdigest()
        if digest != EMBEDDING_SHA256:
            embedding.unlink()
            raise VTDError(f"The downloaded speaker model is not the expected file: {EMBEDDING_URL}")
    if not segmentation.is_file():
        archive = folder / "segmentation.tar.bz2"
        _fetch(SEGMENTATION_URL, archive)
        unpacked = folder / "segmentation.unpacking"
        shutil.rmtree(unpacked, ignore_errors=True)
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(unpacked, filter="data")  # nothing outside it, no links out
            if not (unpacked / SEGMENTATION_MODEL).is_file():
                raise VTDError(f"The speaker model archive did not contain {SEGMENTATION_MODEL}.")
            # Moved into place only when whole: a cut-off archive must not leave half a model.
            shutil.rmtree(segmentation.parent, ignore_errors=True)
            (unpacked / SEGMENTATION_MODEL).parent.replace(segmentation.parent)
        except (tarfile.TarError, OSError, EOFError) as e:
            raise VTDError(f"Could not unpack the speaker model {archive.name}: {e}") from e
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(unpacked, ignore_errors=True)
    return segmentation, embedding


def _diarize(audio: Path, segmentation: Path, embedding: Path,
             config: VTDConfig) -> list[tuple[float, float, int]]:
    """(start, end, voice) of every stretch of speech, in order."""
    import numpy as np
    import sherpa_onnx

    threads = max(1, min(8, (os.cpu_count() or 2) - 1))
    settings = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=config.num_speakers if config.num_speakers > 0 else -1,
            threshold=config.speaker_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not settings.validate():
        raise VTDError("The speaker models could not be loaded; delete them to download again: "
                       f"{segmentation.parent.parent}")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(settings)
    with wave.open(str(audio), "rb") as wav:
        shape = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if shape != (diarizer.sample_rate, 1, 2):
            raise VTDError(f"{audio.name} is not 16 kHz mono 16-bit audio; rerun extract-audio.")
        pcm = wav.readframes(wav.getnframes())
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    result = diarizer.process(samples).sort_by_start_time()
    return [(float(s.start), float(s.end), int(s.speaker)) for s in result]


def tidy(raw: list[tuple[float, float, int]], min_s: float | None = None) -> list[dict]:
    """Turns as {start_s, end_s, speaker}, speakers numbered from 1 in order of appearance.

    Voices heard for less than `min_s` in total are dropped: they are short words the
    grouping could not attach to anyone, and dozens of one-line "speakers" help nobody.
    By default that is 20 seconds, or 2% of all the speech in a short recording, so a
    one-minute clip keeps the person who spoke for ten seconds. By default a voice that
    is pieces cut from a much longer one (see `_pieces`) is dropped as well, and its
    pieces go back to that voice. What a briefly heard voice said right between two
    turns of one speaker goes to that speaker; the rest stays unnamed.
    """
    heard: dict[int, float] = {}
    for start, end, voice in raw:
        heard[voice] = heard.get(voice, 0.0) + (end - start)
    ordered = sorted(raw)
    pieces: set[int] = set()
    if min_s is None:
        min_s = min(MIN_SPEAKER_S, 0.02 * sum(heard.values()))
        pieces = _pieces(ordered, heard, min_s)
    is_kept = [heard[voice] >= min_s and voice not in pieces for _start, _end, voice in ordered]
    near = _hosts(ordered, is_kept, JOIN_GAP_S)
    inside = _hosts(ordered, is_kept, PIECES_GAP_S)
    hosts = [inside[i] if voice in pieces else near[i] for i, (_s, _e, voice) in enumerate(ordered)]
    numbers: dict[int, int] = {}
    turns = []
    for position, (start, end, voice) in enumerate(ordered):
        if not is_kept[position]:
            if hosts[position] is None:
                continue
            voice = hosts[position]
        number = numbers.setdefault(voice, len(numbers) + 1)
        turns.append({"start_s": round(start, 2), "end_s": round(end, 2), "speaker": number})
    return turns


def _hosts(ordered: list[tuple[float, float, int]], is_kept: list[bool],
           gap_s: float) -> list[int | None]:
    """For each turn that is not kept, the voice it was cut from, or None.

    That is the voice kept on both sides of it, when it is one and the same and the
    talk between those two turns never stops for longer than `gap_s`: several
    pieces in a row inside one person's speech all go back to that person.
    """
    hosts: list[int | None] = [None] * len(ordered)
    position = 0
    while position < len(ordered):
        if is_kept[position]:
            position += 1
            continue
        run_end = position
        while run_end < len(ordered) and not is_kept[run_end]:
            run_end += 1
        if position > 0 and run_end < len(ordered) \
                and ordered[position - 1][2] == ordered[run_end][2]:
            said_until = ordered[position - 1][1]
            unbroken = True
            for start, end, _voice in ordered[position:run_end + 1]:
                unbroken = unbroken and start - said_until <= gap_s
                said_until = max(said_until, end)
            if unbroken:
                hosts[position:run_end] = [ordered[run_end][2]] * (run_end - position)
        position = run_end
    return hosts


def _pieces(ordered: list[tuple[float, float, int]], heard: dict[int, float],
            min_s: float) -> set[int]:
    """Voices that are pieces of another one, set apart by the grouping.

    Such a voice never speaks `MIN_TURN_S` at a stretch, most of its turns fall between
    two turns of one other voice, it is tiny beside that voice (`PIECES_SHARE`) and its
    turns last a couple of seconds on average (`PIECES_MEAN_S`). Someone who asks brief
    questions or gives short answers speaks longer each time, or more beside the others.
    """
    longest: dict[int, float] = {}
    for start, end, voice in ordered:
        longest[voice] = max(longest.get(voice, 0.0), end - start)
    short = {voice for voice in heard
             if heard[voice] >= min_s and longest[voice] < MIN_TURN_S}
    if not short:
        return set()
    is_kept = [heard[voice] >= min_s and voice not in short for _s, _e, voice in ordered]
    hosts = _hosts(ordered, is_kept, PIECES_GAP_S)
    turns: dict[int, int] = {}
    hosted: dict[int, dict[int, int]] = {}
    for (_start, _end, voice), host in zip(ordered, hosts, strict=True):
        if voice in short:
            turns[voice] = turns.get(voice, 0) + 1
            if host is not None:
                by = hosted.setdefault(voice, {})
                by[host] = by.get(host, 0) + 1
    pieces = set()
    for voice in short:
        host, times = max(hosted.get(voice, {}).items(), key=lambda kv: kv[1], default=(None, 0))
        if host is not None and 2 * times >= turns[voice] \
                and heard[voice] <= PIECES_SHARE * heard[host] \
                and heard[voice] <= PIECES_MEAN_S * turns[voice]:
            pieces.add(voice)
    return pieces


def run(workspace: Path, config: VTDConfig, manifest: Manifest) -> None:
    require()
    audio = workspace / "audio.wav"
    if not audio.exists():
        raise VTDError("audio.wav not found — extract-audio stage must run first.")
    segmentation, embedding = models(config)
    raw = _diarize(audio, segmentation, embedding, config)
    # When the number of people is given, every voice found is one of them.
    turns = tidy(raw, 0.0 if config.num_speakers > 0 else None)
    (workspace / TURNS_NAME).write_text(json.dumps(turns, indent=1))
    found = len({t["speaker"] for t in turns})
    voices = len({voice for _s, _e, voice in raw})
    emit(f"identify-speakers: {found} speakers"
         + (f" ({voices - found} voices heard too briefly to tell were left unnamed)"
            if voices > found else ""))


def speaker_turns(workspace: Path) -> list[dict]:
    """The recorded turns, or [] when speakers were not identified."""
    turns = read_json(workspace / TURNS_NAME, [])
    return turns if isinstance(turns, list) else []


def speaker_names(workspace: Path) -> dict[int, str]:
    """Names known for speaker numbers, from speaker_names.json ({"3": "Elena"})."""
    raw = read_json(workspace / NAMES_NAME, {})
    try:
        return {int(k): str(v).strip() for k, v in raw.items() if str(v).strip()}
    except (ValueError, AttributeError):  # not {"number": "name"}: a hand edit gone wrong
        emit(f"{NAMES_NAME} is not a table of speaker numbers and names; ignoring it.",
             kind="warn")
        return {}


def cue_speakers(cues: list[dict], turns: list[dict]) -> list[int | None]:
    """The speaker of each cue: the one whose turns overlap it longest, or None."""
    speakers: list[int | None] = []
    first = 0  # turns are in order, and so are cues: never look back before this one
    for cue in cues:
        while first < len(turns) and turns[first]["end_s"] <= cue["start_s"]:
            first += 1
        overlap: dict[int, float] = {}
        for turn in turns[first:]:
            if turn["start_s"] >= cue["end_s"]:
                break
            shared = min(turn["end_s"], cue["end_s"]) - max(turn["start_s"], cue["start_s"])
            if shared > 0:
                overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0.0) + shared
        speakers.append(max(overlap, key=overlap.get) if overlap else None)
    return speakers


def labelled_text(cues: list[dict], speakers: list[int | None], word: str,
                  names: dict[int, str] | None = None) -> str:
    """The cues as text in which each change of speaker starts a line with who speaks.

    A speaker is shown by name when it is known and as `<word> <number>` otherwise;
    a voice that could not be identified is `<word> ?`.
    """
    names = names or {}
    lines: list[str] = []
    current: object = object()  # no speaker yet, distinct from None
    for cue, speaker in zip(cues, speakers, strict=True):
        text = cue["text"].strip()
        if not is_spoken(text):
            continue
        if speaker != current:
            current = speaker
            who = f"{word} ?" if speaker is None else names.get(speaker) or f"{word} {speaker}"
            lines.append(f"{who}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)
