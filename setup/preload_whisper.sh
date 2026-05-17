#!/bin/bash
# One-time (online) Whisper cache warm-up for the Pi. Run before air-gapping.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f venv/bin/activate ]]; then
  echo "Missing venv. Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate
set -a
[[ -f .env ]] && source .env
set +a

MODEL="${WHISPER_MODEL:-base}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Downloading Whisper model '${MODEL}' into ${HF_HOME} (can take several minutes)..."
python -c "
from faster_whisper import WhisperModel
import os
m = os.environ.get('WHISPER_MODEL', 'base')
root = os.environ.get('HF_HOME')
WhisperModel(m, device='cpu', compute_type='int8', download_root=root)
print('Whisper ready:', m)
"

echo "Done. Restart billy.service when ready."
