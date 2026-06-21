"""Fast rules-based workout intent routing.

This module is intentionally tiny and deterministic so it can run before the
chat model without adding noticeable latency. It classifies simple workout
utterances, stores memory notes, and returns a cleaned-up prompt for the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .logger import logger
from .song_manager import (
    match_song_by_wake_phrases,
    pick_default_song,
    pick_random_song,
    song_manager,
)
from .profile_manager import user_manager


_MEMORY_TRIGGERS = (
    "remember",
    "log",
    "save",
    "note",
    "record",
    "add this",
    "add that",
)
_AUTOMATION_TRIGGERS = (
    "timer",
    "rest",
    "countdown",
    "count down",
    "next set",
    "next exercise",
    "done with",
    "how long left",
    "how much time",
    "start workout",
    "finish workout",
    "set counter",
    "sets",
)
# Generic song requests — random pick from the full library
_GENERIC_SONG_TRIGGERS = (
    "play me a song",
    "play a song",
    "play song",
    "sing me a song",
    "sing a song",
)

_SONG_WORD_RE = re.compile(r"\bsongs?\b", re.IGNORECASE)
_SET_COUNTER_RE = re.compile(
    r"\b(?:"
    r"set\s+counter|"
    r"next\s+set|"
    r"count(?:ing)?\s+(?:my\s+)?sets?|"
    r"count\s+\d+\s+sets?|"
    r"ready\s+set\s+go"
    r")\b",
    re.IGNORECASE,
)
_WORKOUT_HINTS = (
    "set",
    "reps",
    "rep",
    "weight",
    "lbs",
    "lb",
    "kg",
    "dumbbell",
    "barbell",
    "bench",
    "squat",
    "deadlift",
    "curl",
    "press",
    "row",
    "pullup",
    "pushup",
    "workout",
    "exercise",
)


@dataclass
class WorkoutIntentResult:
    original_text: str
    normalized_text: str
    action: str = "chat"
    confidence: str = "low"
    memory_note: str | None = None
    song_name: str | None = None
    target_count: int | None = None
    spoken_sequence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def should_store_memory(self) -> bool:
        return self.memory_note is not None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_trigger_prefix(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(?:please\s+)?(?:" + "|".join(_MEMORY_TRIGGERS) + r")\b[,:\- ]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _normalize_words(text: str) -> str:
    """Lowercase text with punctuation turned into word breaks."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _looks_like_set_counter_request(text: str) -> bool:
    """Detect set-counting requests, including noisy Whisper transcripts."""
    normalized = _normalize_words(text)
    if not normalized:
        return False
    if _SET_COUNTER_RE.search(normalized):
        return True
    words = normalized.split()
    if "count" in words and ("set" in words or "sets" in words):
        return True
    if "ready" in words and "set" in words and "go" in words:
        return True
    return False


def _default_set_count(text: str) -> int:
    """Default rep/set count when the user did not say a number."""
    return _extract_count(text) or 10


def _looks_like_workout_text(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _WORKOUT_HINTS)


def _extract_timer_seconds(text: str) -> int | None:
    lowered = text.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:sec|secs|second|seconds)", lowered)
    if match:
        return max(1, int(float(match.group(1))))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)", lowered)
    if match:
        return max(1, int(float(match.group(1)) * 60))
    return None


def _extract_count(text: str) -> int | None:
    lowered = text.lower()
    match = re.search(r"(\d+(?:\.\d+)?)", lowered)
    if match:
        return max(1, int(float(match.group(1))))
    return None


def _build_memory_note(text: str) -> str:
    note = _strip_trigger_prefix(text)
    note = _clean_text(note)
    return note if note else _clean_text(text)


def _build_timer_sequence(count: int) -> list[str]:
    return [str(step) for step in range(count, 0, -1)]


def _build_set_sequence(count: int) -> list[str]:
    if count <= 1:
        return ["keep going bitch"]
    if count == 2:
        return ["up", "keep going bitch"]
    sequence = ["up"]
    sequence.extend(["down"] * max(0, count - 2))
    sequence.append("keep going bitch")
    return sequence


def _get_available_songs() -> list[dict[str, Any]]:
    try:
        return song_manager.list_songs()
    except Exception as e:
        logger.warning(f"Failed to load songs for intent routing: {e}", "⚠️")
        return []


def _mentions_song_word(text: str) -> bool:
    """True if the utterance mentions 'song' or 'songs' anywhere."""
    return bool(_SONG_WORD_RE.search(text))


def _pick_song_by_title(target_title: str | None = None) -> str | None:
    songs = _get_available_songs()
    if not songs:
        return None

    if target_title:
        target = target_title.lower()
        for song in songs:
            song_name = str(song.get("name", "")).lower()
            song_title = str(song.get("title", "")).lower()
            if song_name == target or song_title == target:
                return song.get("name") or song.get("title")

    return pick_default_song(songs)


def _pick_song_for_request(text: str) -> str | None:
    lowered = text.lower()
    songs = _get_available_songs()
    if not songs:
        return None

    matched = match_song_by_wake_phrases(lowered, songs)
    if matched:
        return matched

    if any(trigger in lowered for trigger in _GENERIC_SONG_TRIGGERS):
        return pick_random_song(songs)

    if _mentions_song_word(lowered):
        return pick_random_song(songs)

    return None


def _looks_like_song_request(text: str) -> bool:
    lowered = text.lower()
    if any(trigger in lowered for trigger in _GENERIC_SONG_TRIGGERS):
        return True
    if _mentions_song_word(lowered):
        return True
    if match_song_by_wake_phrases(lowered, _get_available_songs()):
        return True
    return False


def classify_workout_intent(text: str) -> WorkoutIntentResult:
    """Classify a user utterance with a fast deterministic pass."""
    original_text = _clean_text(text)
    lowered = original_text.lower()

    if not original_text:
        return WorkoutIntentResult(original_text="", normalized_text="", confidence="low")

    looks_like_memory = any(trigger in lowered for trigger in _MEMORY_TRIGGERS)
    looks_like_automation = any(trigger in lowered for trigger in _AUTOMATION_TRIGGERS)
    looks_like_workout = _looks_like_workout_text(lowered)

    if looks_like_memory and looks_like_workout:
        note = _build_memory_note(original_text)
        return WorkoutIntentResult(
            original_text=original_text,
            normalized_text=f"Workout memory log: {note}",
            action="log_memory",
            confidence="high",
            memory_note=note,
        )

    if _looks_like_song_request(lowered):
        song_name = _pick_song_for_request(lowered)
        normalized = f"Song request: {song_name or original_text}"
        return WorkoutIntentResult(
            original_text=original_text,
            normalized_text=normalized,
            action="song",
            confidence="high",
            song_name=song_name,
            metadata={"song_name": song_name, "song_request": original_text},
        )

    if _looks_like_set_counter_request(lowered):
        count = _default_set_count(lowered)
        return WorkoutIntentResult(
            original_text=original_text,
            normalized_text=f"Set counter request: {original_text}",
            action="set_counter",
            confidence="high",
            target_count=count,
            spoken_sequence=_build_set_sequence(count),
            metadata={"count": count},
        )

    if looks_like_automation:
        count = _extract_timer_seconds(lowered) or _extract_count(lowered) or 60
        metadata: dict[str, Any] = {}
        metadata["count"] = count

        if any(trigger in lowered for trigger in ("set counter", "sets", "next set")):
            normalized = f"Set counter request: {original_text}"
            return WorkoutIntentResult(
                original_text=original_text,
                normalized_text=normalized,
                action="set_counter",
                confidence="high",
                target_count=count,
                spoken_sequence=_build_set_sequence(count),
                metadata=metadata,
            )

        normalized = f"Timer request: {original_text}"
        return WorkoutIntentResult(
            original_text=original_text,
            normalized_text=normalized,
            action="timer",
            confidence="high",
            target_count=count,
            spoken_sequence=_build_timer_sequence(count),
            metadata=metadata,
        )

    if looks_like_workout:
        return WorkoutIntentResult(
            original_text=original_text,
            normalized_text=original_text,
            action="chat",
            confidence="medium",
        )

    return WorkoutIntentResult(
        original_text=original_text,
        normalized_text=original_text,
        action="chat",
        confidence="low",
    )


def route_workout_text(text: str) -> WorkoutIntentResult:
    """Classify a message and persist workout memory if needed."""
    result = classify_workout_intent(text)

    if result.should_store_memory:
        try:
            profile = user_manager.get_current_user()
            if profile is None:
                profile = user_manager.identify_user("guest", "high")
            if profile is not None:
                profile.add_memory(result.memory_note or result.normalized_text, category="workout")
                logger.info(f"Stored workout memory: {result.memory_note}", "💭")
        except Exception as e:
            logger.warning(f"Failed to store workout memory: {e}", "⚠️")

    return result
