"""
Song Manager - Handles Billy song library under sounds/songs/
"""

import configparser
import random
import shutil
from pathlib import Path
from typing import Any, Optional

from .logger import logger

# ~0.3s of stereo 48 kHz 16-bit audio — catches empty or stub files on the Pi.
MIN_WAV_BYTES = 32_000


def _wav_stem_ok(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_WAV_BYTES
    except OSError:
        return False


def validate_song_stems(song_path: Path) -> tuple[bool, str]:
    """Return (ok, error_message). Requires at least one non-empty stem."""
    stems = {
        "full.wav": song_path / "full.wav",
        "vocals.wav": song_path / "vocals.wav",
        "drums.wav": song_path / "drums.wav",
    }
    usable = [name for name, path in stems.items() if _wav_stem_ok(path)]
    if usable:
        return True, ""
    missing = [name for name, path in stems.items() if not path.is_file()]
    empty = [
        name
        for name, path in stems.items()
        if path.is_file() and not _wav_stem_ok(path)
    ]
    if missing:
        return False, f"missing {', '.join(missing)}"
    if empty:
        return False, f"empty or too small: {', '.join(empty)} (run git pull or copy from PC)"
    return False, "no audio stems found"


class SongManager:
    """Manages songs in sounds/songs/ (legacy custom_songs/ still scanned for migration)."""

    def __init__(self):
        project_root = Path(__file__).parent.parent
        self.songs_dir = project_root / "sounds" / "songs"
        self.songs_dir.mkdir(parents=True, exist_ok=True)

        # Deprecated — scanned so existing Pi installs keep working until moved.
        self.legacy_songs_dir = project_root / "custom_songs"

    def _song_dirs_to_scan(self) -> list[tuple[Path, str]]:
        """Return (directory, source_label) pairs in search order."""
        dirs: list[tuple[Path, str]] = [(self.songs_dir, "sounds/songs")]
        if self.legacy_songs_dir.exists():
            dirs.append((self.legacy_songs_dir, "custom_songs"))
        return dirs

    def _resolve_song_path(self, song_name: str) -> Optional[Path]:
        for base, _label in self._song_dirs_to_scan():
            path = base / song_name
            if path.is_dir():
                return path
        return None

    def get_song_directory(self, song_name: str) -> Optional[Path]:
        """Absolute path to a song folder, or None if not found."""
        return self._resolve_song_path(song_name)

    def list_songs(self) -> list[dict[str, Any]]:
        """List all playable songs from sounds/songs and legacy custom_songs."""
        songs: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for base, source in self._song_dirs_to_scan():
            if not base.exists():
                continue
            for song_dir in sorted(base.iterdir()):
                if not song_dir.is_dir():
                    continue
                name = song_dir.name
                if name in seen_names:
                    continue
                metadata = self.get_song_metadata(name)
                if not metadata:
                    continue
                if not (
                    metadata.get("has_full")
                    or metadata.get("has_vocals")
                    or metadata.get("has_drums")
                ):
                    ok, reason = validate_song_stems(song_dir)
                    if not ok:
                        logger.warning(
                            f"Skipping song '{name}' in {source}: {reason}",
                            "⚠️",
                        )
                    continue
                metadata["source"] = source
                metadata["is_legacy"] = source == "custom_songs"
                songs.append(metadata)
                seen_names.add(name)

        return sorted(songs, key=lambda x: x.get("title", x["name"]).lower())

    def get_song_metadata(
        self, song_name: str, is_custom: Optional[bool] = None
    ) -> Optional[dict[str, Any]]:
        """Get metadata for a specific song (is_custom arg kept for compatibility)."""
        _ = is_custom
        song_path = self._resolve_song_path(song_name)
        if song_path is None:
            return None

        metadata_file = song_path / "metadata.ini"

        has_full = _wav_stem_ok(song_path / "full.wav")
        has_vocals = _wav_stem_ok(song_path / "vocals.wav")
        has_drums = _wav_stem_ok(song_path / "drums.wav")

        metadata: dict[str, Any] = {
            "name": song_name,
            "title": song_name.replace("_", " ").title(),
            "is_custom": song_path.parent == self.legacy_songs_dir,
            "keywords": "",
            "wake_words": "",
            "default": False,
            "bpm": 120.0,
            "gain": 1.0,
            "tail_threshold": 1500.0,
            "compensate_tail": 0.0,
            "head_moves": "",
            "half_tempo_tail_flap": False,
            "has_full": has_full,
            "has_vocals": has_vocals,
            "has_drums": has_drums,
        }

        if metadata_file.exists():
            config = configparser.ConfigParser()
            config.read(metadata_file)

            if config.has_section("SONG"):
                metadata.update(
                    {
                        "title": config.get("SONG", "title", fallback=metadata["title"]),
                        "keywords": config.get("SONG", "keywords", fallback=""),
                        "wake_words": config.get("SONG", "wake_words", fallback=""),
                        "default": config.getboolean("SONG", "default", fallback=False),
                        "bpm": config.getfloat("SONG", "bpm", fallback=120.0),
                        "gain": config.getfloat("SONG", "gain", fallback=1.0),
                        "tail_threshold": config.getfloat(
                            "SONG", "tail_threshold", fallback=1500.0
                        ),
                        "compensate_tail": config.getfloat(
                            "SONG", "compensate_tail", fallback=0.0
                        ),
                        "head_moves": config.get("SONG", "head_moves", fallback=""),
                        "half_tempo_tail_flap": config.getboolean(
                            "SONG", "half_tempo_tail_flap", fallback=False
                        ),
                    }
                )
        elif (song_path / "metadata.txt").exists():
            metadata.update(self._load_old_metadata(song_path / "metadata.txt"))

        if not metadata.get("wake_words") and metadata.get("keywords"):
            metadata["wake_words"] = metadata["keywords"]

        return metadata

    def _load_old_metadata(self, path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        with open(path) as f:
            for line in f:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                if key == "head_moves":
                    metadata[key] = value
                elif key in ("bpm", "tail_threshold", "gain", "compensate_tail"):
                    metadata[key] = float(value.strip())
                elif key == "half_tempo_tail_flap":
                    metadata[key] = value.strip().lower() == "true"
        return metadata

    def save_song_metadata(self, song_name: str, metadata: dict[str, Any]) -> bool:
        song_path = self.songs_dir / song_name
        song_path.mkdir(parents=True, exist_ok=True)

        metadata_file = song_path / "metadata.ini"
        config = configparser.ConfigParser()
        config["SONG"] = {
            "title": metadata.get("title", song_name.replace("_", " ").title()),
            "keywords": metadata.get("keywords", ""),
            "wake_words": metadata.get("wake_words", metadata.get("keywords", "")),
            "default": str(bool(metadata.get("default", False))).lower(),
            "bpm": str(metadata.get("bpm", 120.0)),
            "gain": str(metadata.get("gain", 1.0)),
            "tail_threshold": str(metadata.get("tail_threshold", 1500.0)),
            "compensate_tail": str(metadata.get("compensate_tail", 0.0)),
            "head_moves": metadata.get("head_moves", ""),
            "half_tempo_tail_flap": str(metadata.get("half_tempo_tail_flap", False)),
        }

        try:
            with open(metadata_file, "w") as f:
                config.write(f)
            logger.info(f"Saved metadata for song: {song_name}", "🎵")
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata for {song_name}: {e}")
            return False

    def create_song(self, song_name: str, metadata: dict[str, Any]) -> bool:
        song_path = self.songs_dir / song_name
        if song_path.exists():
            logger.warning(f"Song already exists: {song_name}")
            return False
        song_path.mkdir(parents=True, exist_ok=True)
        return self.save_song_metadata(song_name, metadata)

    def delete_song(self, song_name: str) -> bool:
        song_path = self._resolve_song_path(song_name)
        if song_path is None:
            logger.warning(f"Song not found: {song_name}")
            return False
        if song_path.parent == self.songs_dir and song_name == "fishsticks":
            logger.warning("Refusing to delete bundled example song: fishsticks")
            return False
        try:
            shutil.rmtree(song_path)
            logger.info(f"Deleted song: {song_name}", "🗑️")
            return True
        except Exception as e:
            logger.error(f"Failed to delete song {song_name}: {e}")
            return False

    def save_audio_file(self, song_name: str, file_type: str, file_data: bytes) -> bool:
        if file_type not in ["full", "vocals", "drums"]:
            logger.error(f"Invalid file type: {file_type}")
            return False

        song_path = self.songs_dir / song_name
        song_path.mkdir(parents=True, exist_ok=True)
        audio_file = song_path / f"{file_type}.wav"

        try:
            with open(audio_file, "wb") as f:
                f.write(file_data)
            logger.info(f"Saved {file_type}.wav for song: {song_name}", "🎵")
            return True
        except Exception as e:
            logger.error(f"Failed to save {file_type}.wav for {song_name}: {e}")
            return False

    def get_audio_file_path(self, song_name: str, file_type: str) -> Optional[Path]:
        if file_type not in ["full", "vocals", "drums"]:
            return None
        song_path = self._resolve_song_path(song_name)
        if song_path is None:
            return None
        audio_file = song_path / f"{file_type}.wav"
        return audio_file if audio_file.exists() else None

    def copy_example_to_custom(
        self, example_name: str, new_name: Optional[str] = None
    ) -> bool:
        """Copy a song folder within sounds/songs (kept for API compatibility)."""
        if new_name is None:
            new_name = example_name

        source_path = self._resolve_song_path(example_name)
        dest_path = self.songs_dir / new_name

        if source_path is None:
            logger.error(f"Song not found: {example_name}")
            return False
        if dest_path.exists():
            logger.warning(f"Song already exists: {new_name}")
            return False

        try:
            shutil.copytree(source_path, dest_path)
            logger.info(f"Copied song '{example_name}' to '{new_name}'", "📋")
            return True
        except Exception as e:
            logger.error(f"Failed to copy song: {e}")
            return False

    def get_dynamic_tool_description(self) -> str:
        songs = self.list_songs()
        if not songs:
            return "Plays a special Billy song. No songs are currently available."

        song_list = []
        for song in songs:
            title = song.get("title", song["name"])
            wake_words = song.get("wake_words") or song.get("keywords", "")
            if wake_words:
                song_list.append(
                    f"- '{song['name']}' ({title}): say any of [{wake_words}]"
                )
            else:
                song_list.append(f"- '{song['name']}' ({title})")

        description = (
            "Plays a special Billy song based on the given name. Available songs:\n"
        )
        description += "\n".join(song_list)
        description += (
            "\n\nIMPORTANT: Use the song folder name (first part in quotes) when calling "
            "this function, NOT the display title in parentheses."
        )
        return description


song_manager = SongManager()


def wake_phrases_for_song(song: dict[str, Any]) -> list[str]:
    """Spoken phrases that should trigger this song (from metadata.ini wake_words)."""
    seen: set[str] = set()
    phrases: list[str] = []
    for field in ("wake_words", "keywords"):
        raw = str(song.get(field, "") or "")
        for part in raw.split(","):
            phrase = part.strip().lower()
            if phrase and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
    return phrases


def collect_wake_phrase_index(
    songs: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """(phrase, song_folder_name) pairs, longest phrases first for greedy matching."""
    index: list[tuple[str, str]] = []
    for song in songs:
        name = str(song.get("name", "")).strip()
        if not name:
            continue
        for phrase in wake_phrases_for_song(song):
            index.append((phrase, name))
    index.sort(key=lambda item: len(item[0]), reverse=True)
    return index


def match_song_by_wake_phrases(
    text: str, songs: list[dict[str, Any]]
) -> str | None:
    """Return the song folder name if any wake phrase appears in text."""
    lowered = text.lower()
    for phrase, song_name in collect_wake_phrase_index(songs):
        if phrase in lowered:
            return song_name
    return None


def pick_random_song(songs: list[dict[str, Any]]) -> str | None:
    """Pick a random song from the library."""
    if not songs:
        return None
    chosen = random.choice(songs)
    return chosen.get("name") or chosen.get("title")


def pick_default_song(songs: list[dict[str, Any]]) -> str | None:
    """Song marked default=true in metadata.ini, else fishsticks, else first listed."""
    if not songs:
        return None
    for song in songs:
        if song.get("default"):
            return song.get("name")
    for song in songs:
        if str(song.get("name", "")).lower() == "fishsticks":
            return song.get("name")
    return songs[0].get("name")

