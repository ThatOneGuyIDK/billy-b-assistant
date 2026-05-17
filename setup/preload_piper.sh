#!/bin/bash
# One-time (online) Piper voice download for offline TTS. Run before air-gapping.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Tip: copy .env.example to .env and set TTS_VOICE if you want a non-default voice."
fi

set -a
[[ -f .env ]] && source .env
set +a

VOICE="${TTS_VOICE:-en_US-lessac-medium}"
MODEL_DIR="${PIPER_MODEL_DIR:-$HOME/.piper/models}"
mkdir -p "$MODEL_DIR"

# en_US-lessac-medium -> en / en_US / lessac / medium
if [[ ! "$VOICE" =~ ^([a-z]{2}_[A-Z]{2})-([^-]+)-([^-]+)$ ]]; then
  echo "Unsupported TTS_VOICE format: $VOICE"
  echo "Expected: locale-speaker-quality (e.g. en_US-lessac-medium)"
  exit 1
fi

LOCALE="${BASH_REMATCH[1]}"
SPEAKER="${BASH_REMATCH[2]}"
QUALITY="${BASH_REMATCH[3]}"
LANG="${LOCALE%%_*}"

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG}/${LOCALE}/${SPEAKER}/${QUALITY}"
ONNX="${MODEL_DIR}/${VOICE}.onnx"
JSON="${MODEL_DIR}/${VOICE}.onnx.json"

download() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url"
  else
    echo "Install curl or wget to download Piper models."
    exit 1
  fi
}

if [[ -s "$ONNX" && -s "$JSON" ]]; then
  echo "Piper voice already present: $ONNX"
  exit 0
fi

echo "Downloading Piper voice '${VOICE}' to ${MODEL_DIR} ..."
download "${BASE}/${VOICE}.onnx" "$ONNX"
download "${BASE}/${VOICE}.onnx.json" "$JSON"

if [[ ! -s "$ONNX" || ! -s "$JSON" ]]; then
  echo "Download failed or files are empty. Check TTS_VOICE and network."
  rm -f "$ONNX" "$JSON"
  exit 1
fi

echo "Piper ready:"
ls -lh "$ONNX" "$JSON"
