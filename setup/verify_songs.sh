#!/bin/bash
# Check that every song folder under sounds/songs has real WAV stems.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SONGS_DIR="$ROOT/sounds/songs"
MIN_BYTES=32000
FAIL=0

echo "Checking songs in $SONGS_DIR ..."
echo

for song_dir in "$SONGS_DIR"/*; do
  [[ -d "$song_dir" ]] || continue
  name="$(basename "$song_dir")"
  ok=0
  for stem in full.wav vocals.wav drums.wav; do
    path="$song_dir/$stem"
    if [[ -f "$path" ]]; then
      size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")
      if [[ "$size" -ge "$MIN_BYTES" ]]; then
        ok=1
        printf "  OK  %-14s %8s KB  %s\n" "$stem" "$(( size / 1024 ))" "$name"
      else
        printf "  BAD %-14s %8s B   %s (too small)\n" "$stem" "$size" "$name"
        FAIL=1
      fi
    else
      printf "  --- %-14s missing     %s\n" "$stem" "$name"
    fi
  done
  if [[ "$ok" -eq 0 ]]; then
    echo "  >> '$name' is not playable — run: git pull"
    echo "     or copy the folder from your PC with scp"
    FAIL=1
  fi
  echo
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "Some songs are missing audio. Fix with:"
  echo "  cd ~/billy-b-assistant && git pull"
  exit 1
fi

echo "All songs have usable audio stems."
