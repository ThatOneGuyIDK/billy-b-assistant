from core import workout_intent
from core.workout_intent import classify_workout_intent


def test_play_me_a_song_picks_from_library(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {"name": "fishsticks", "title": "Fishsticks", "default": True},
            {"name": "other-song", "title": "Other Song", "default": False},
        ],
    )
    monkeypatch.setattr(workout_intent, "pick_random_song", lambda songs: "other-song")

    result = classify_workout_intent("play me a song")

    assert result.action == "song"
    assert result.song_name == "other-song"


def test_song_word_can_pick_a_random_song(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {"name": "fishsticks", "title": "Fishsticks", "wake_words": "fishsticks"},
            {"name": "random-song", "title": "Random Song", "wake_words": "random"},
        ],
    )
    monkeypatch.setattr(workout_intent, "pick_random_song", lambda songs: "random-song")

    result = classify_workout_intent("song")

    assert result.action == "song"
    assert result.song_name == "random-song"


def test_song_request_matches_wake_words_from_metadata(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {
                "name": "fishsticks",
                "title": "Fishsticks",
                "wake_words": "play fishticks, fishticks",
            },
            {
                "name": "Blub Blub Jake",
                "title": "Blub Blub Jake",
                "wake_words": "happy birthday, blub blub jake",
            },
        ],
    )

    birthday = classify_workout_intent("happy birthday")
    fishticks = classify_workout_intent("play fishticks")

    assert birthday.action == "song"
    assert birthday.song_name == "Blub Blub Jake"
    assert fishticks.action == "song"
    assert fishticks.song_name == "fishsticks"


def test_bare_song_word_picks_from_library(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {"name": "fishsticks", "title": "Fishsticks", "default": True},
            {"name": "other-song", "title": "Other Song"},
        ],
    )
    monkeypatch.setattr(workout_intent, "pick_random_song", lambda songs: "other-song")

    result = classify_workout_intent("Billy Bass assistant, a song")

    assert result.action == "song"
    assert result.song_name == "other-song"


def test_play_song_phrase_picks_from_library(monkeypatch):
    monkeypatch.setattr(
        workout_intent.song_manager,
        "list_songs",
        lambda: [
            {"name": "fishsticks", "title": "Fishsticks", "default": True},
            {"name": "other-song", "title": "Other Song"},
        ],
    )
    monkeypatch.setattr(workout_intent, "pick_random_song", lambda songs: "other-song")

    result = classify_workout_intent("play song")

    assert result.action == "song"
    assert result.song_name == "other-song"
