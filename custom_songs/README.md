# Deprecated — use `sounds/songs/` instead

All Billy songs (including custom ones) now live in:

```text
sounds/songs/<song_name>/
  full.wav
  vocals.wav
  drums.wav
  metadata.ini
```

Folders here are still scanned for backward compatibility. Move any remaining songs to `sounds/songs/` on the Pi:

```bash
mv ~/billy-b-assistant/custom_songs/* ~/billy-b-assistant/sounds/songs/
```
