#!/bin/bash
# One-time song conversion helper.
# Converts a single source audio file into the WAV files Billy expects:
#   full.wav, vocals.wav, and drums.wav
#
# If you only have one mixed track, the script duplicates it into vocals/drums
# so Billy can still move the mouth and tail. That will not be as accurate as
# real separated stems, but it will work with the current playback code.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input-audio-file> [output-song-dir]"
  echo "Example: $0 ~/Downloads/blub_blub_jake.mp3 sounds/songs/blub_blub_jake"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="${2:-}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file not found: $INPUT_FILE"
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  BASE_NAME="$(basename "$INPUT_FILE")"
  BASE_NAME="${BASE_NAME%.*}"
  OUTPUT_DIR="sounds/songs/${BASE_NAME}"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but was not found in PATH."
  echo "Install it on the Pi, then rerun this script."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Converting '$INPUT_FILE' into '$OUTPUT_DIR'..."

# Billy expects 48 kHz, 16-bit, stereo WAV files.
ffmpeg -y -i "$INPUT_FILE" -ar 48000 -ac 2 -sample_fmt s16 "$OUTPUT_DIR/full.wav"
cp "$OUTPUT_DIR/full.wav" "$OUTPUT_DIR/vocals.wav"
cp "$OUTPUT_DIR/full.wav" "$OUTPUT_DIR/drums.wav"

if [[ ! -s "$OUTPUT_DIR/full.wav" || ! -s "$OUTPUT_DIR/vocals.wav" || ! -s "$OUTPUT_DIR/drums.wav" ]]; then
  echo "Conversion failed: one or more output files are empty."
  exit 1
fi

cat > "$OUTPUT_DIR/metadata.ini" <<EOF
[SONG]
title = $(basename "$OUTPUT_DIR")
keywords = $(basename "$OUTPUT_DIR")
bpm = 120
gain = 1.0
tail_threshold = 1500
compensate_tail = 0.0
head_moves =
half_tempo_tail_flap = false
EOF

echo "Done. Created:"
ls -lh "$OUTPUT_DIR/full.wav" "$OUTPUT_DIR/vocals.wav" "$OUTPUT_DIR/drums.wav" "$OUTPUT_DIR/metadata.ini"