#!/usr/bin/env bash
# Bootstrap script for vtd.
#
# Verifies the system tools, creates a Python venv, installs vtd in editable
# mode with the [all] extra (AI providers, URL inputs, speaker identification), and
# downloads the default whisper model.
set -euo pipefail

cd "$(dirname "$0")"

say() { printf "• %s\n" "$*"; }
warn() { printf "! %s\n" "$*" >&2; }
fail() { printf "✗ %s\n" "$*" >&2; exit 1; }

# --- Dependency checks --------------------------------------------------------

OS=$(uname)

install_macos() {
  if ! command -v brew >/dev/null 2>&1; then
    fail "Homebrew not found. Install from https://brew.sh and re-run."
  fi
  local need=()
  for bin in ffmpeg whisper-cli tesseract yt-dlp; do
    if ! command -v "$bin" >/dev/null 2>&1; then
      case "$bin" in
        whisper-cli) need+=("whisper-cpp") ;;
        *)           need+=("$bin")        ;;
      esac
    fi
  done
  if (( ${#need[@]} > 0 )); then
    say "Installing missing Homebrew formulae: ${need[*]}"
    brew install "${need[@]}"
  else
    say "All required brew formulae present."
  fi
}

install_linux() {
  local missing=()
  for bin in ffmpeg tesseract yt-dlp; do
    if ! command -v "$bin" >/dev/null 2>&1; then
      case "$bin" in
        tesseract) missing+=("tesseract-ocr") ;;  # Debian/Ubuntu package name
        *)         missing+=("$bin")          ;;
      esac
    fi
  done
  # whisper-cpp is not in standard apt; let user know how to build/install.
  if ! command -v whisper-cli >/dev/null 2>&1; then
    warn "whisper-cli not found. Install from https://github.com/ggml-org/whisper.cpp (build and add to PATH)."
  fi
  if (( ${#missing[@]} > 0 )); then
    if command -v apt-get >/dev/null 2>&1; then
      say "Installing missing apt packages: ${missing[*]}"
      local as_root=()
      [[ $EUID -ne 0 ]] && as_root=(sudo)  # a root shell, as in a container, has no sudo
      "${as_root[@]}" apt-get update
      "${as_root[@]}" apt-get install -y "${missing[@]}"
    else
      warn "Missing tools: ${missing[*]}. Install with your distro's package manager."
    fi
  else
    say "All required tools present on PATH."
  fi
}

case "$OS" in
  Darwin) install_macos ;;
  Linux)  install_linux ;;
  *)      warn "Unrecognised OS: $OS. Install ffmpeg, whisper-cli, tesseract, yt-dlp manually." ;;
esac

# --- Python venv --------------------------------------------------------------

# Pick the first python in this list that's >=3.11; allow $PYTHON override.
pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null && { echo "$PYTHON"; return; }
    fail "\$PYTHON ($PYTHON) is not >=3.11"
  fi
  for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
        echo "$candidate"
        return
      fi
    fi
  done
  fail "No Python >=3.11 found. Install one (macOS: brew install python@3.14; Debian/Ubuntu: sudo apt-get install python3 python3-venv)."
}

PY=$(pick_python)
say "Using Python: $($PY --version) ($(command -v "$PY"))"

if [[ -d .venv ]]; then
  EXISTING_PY_VERSION=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "broken")
  if [[ "$EXISTING_PY_VERSION" == "broken" ]] || ! .venv/bin/python -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
    warn "Existing .venv uses Python $EXISTING_PY_VERSION. Re-creating with $PY."
    rm -rf .venv
  fi
fi

if [[ ! -d .venv ]]; then
  say "Creating Python venv at .venv/"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

say "Upgrading pip"
python -m pip install --upgrade pip >/dev/null

say "Installing vtd (editable) with [all] extras (anthropic, openai, yt-dlp, sherpa-onnx, numpy)"
pip install -e ".[all]"

# --- Whisper model cache ------------------------------------------------------

# Set VTD_MODEL_DIR to keep models elsewhere (and set `model_dir` in the config to match).
CACHE_DIR="${VTD_MODEL_DIR:-${HOME}/.cache/vtd/models}"
MODEL="${VTD_DEFAULT_MODEL:-medium}"
MODEL_FILE="${CACHE_DIR}/ggml-${MODEL}.bin"

mkdir -p "$CACHE_DIR"

# `wc -c` instead of `stat`: the stat size flag differs between macOS and Linux.
if [[ -f "$MODEL_FILE" && $(wc -c < "$MODEL_FILE") -gt 50000000 ]]; then
  say "Whisper model already cached: $MODEL_FILE"
else
  URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
  say "Downloading whisper model ${MODEL} from $URL"
  # To another name first: a download cut short must not pass for the model.
  curl --fail --location --progress-bar --output "${MODEL_FILE}.partial" "$URL"
  mv "${MODEL_FILE}.partial" "$MODEL_FILE"
fi

# --- Outro --------------------------------------------------------------------

cat <<EOF

✓ Install complete.

Next steps:
  source "$(pwd)/.venv/bin/activate"
  vtd path/to/video.mp4               # deterministic-only run
  vtd path/to/video.mp4 --with-ai     # AI notes; needs one of:
    export ANTHROPIC_API_KEY=...          (default provider)
    export OPENAI_API_KEY=...             with --provider openai
    a running Ollama with a model pulled  with --provider ollama (nothing leaves this machine)

To expose vtd outside the venv, symlink it:
  mkdir -p ~/.local/bin && ln -sf "$(pwd)/.venv/bin/vtd" ~/.local/bin/vtd   # ~/.local/bin must be on your PATH
EOF
