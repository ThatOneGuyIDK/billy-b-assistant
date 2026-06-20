from core import workout_intent
from core.workout_intent import classify_workout_intent


def test_play_me_a_song_uses_featured_song(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {"name": "fishsticks", "title": "Fishsticks"},
            {"name": "other-song", "title": "Other Song"},
        ],
    )

    result = classify_workout_intent("play me a song")

    assert result.action == "song"
    assert result.song_name == "fishsticks"


def test_song_word_can_pick_a_random_song(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {"name": "fishsticks", "title": "Fishsticks"},
            {"name": "random-song", "title": "Random Song"},
        ],
    )
    monkeypatch.setattr(workout_intent.random, "choice", lambda items: items[1])

    result = classify_workout_intent("song")

    assert result.action == "song"
    assert result.song_name == "random-song"


def test_song_request_matches_keywords(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {"name": "fishsticks", "title": "Fishsticks", "keywords": "kanye"},
            {
                "name": "Blub Blub Jake",
                "title": "Blub Blub Jake",
                "keywords": "Blub Blub Jake",
            },
        ],
    )

    result = classify_workout_intent("play blub blub jake")

    assert result.action == "song"
    assert result.song_name == "Blub Blub Jake"