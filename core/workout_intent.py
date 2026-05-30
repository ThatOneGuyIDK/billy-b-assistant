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
