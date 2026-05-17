#!/bin/bash
# Download all offline AI assets (Whisper + Piper). Run once while online.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Whisper (STT) ==="
bash "$SCRIPT_DIR/preload_whisper.sh"

echo ""
echo "=== Piper (TTS) ==="
bash "$SCRIPT_DIR/preload_piper.sh"

echo ""
echo "=== Ollama (LLM) — if ollama is installed ==="
if command -v ollama >/dev/null 2>&1; then
  set -a
  [[ -f "$SCRIPT_DIR/../.env" ]] && source "$SCRIPT_DIR/../.env"
  set +a
  MODEL="${OLLAMA_MODEL:-llama3.2:latest}"
  echo "Pulling ${MODEL} ..."
  ollama pull "$MODEL"
else
  echo "Skip: ollama not in PATH (install separately)."
fi

echo ""
echo "All preload steps finished. Set HF_HUB_OFFLINE=1 in .env when going offline."
