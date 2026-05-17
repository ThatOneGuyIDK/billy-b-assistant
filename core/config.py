import asyncio
import os

from dotenv import load_dotenv

# Persona system removed: use single prompt file `core/mean_workout_prompt.txt`


# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

# === Load .env ===
load_dotenv(dotenv_path=ENV_PATH)


# Load single persona/prompt file for local-only operation
PROMPT_FILE = os.path.join(ROOT_DIR, "core", "mean_workout_prompt.txt")
try:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        INSTRUCTIONS = f.read().strip()
except Exception:
    INSTRUCTIONS = "You are Iron Mouth, a brutally sassy AI workout assistant focused entirely on fitness and motivation. Keep replies short, high-energy, and gym-focused."

# === Local Provider Config ===
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
TTS_VOICE = os.getenv("TTS_VOICE", "en_US-lessac-medium")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "96"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1536"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "1"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
THINKING_SOUND_DELAY_MS = int(os.getenv("THINKING_SOUND_DELAY_MS", "160"))

# === Provider Config ===
REALTIME_AI_PROVIDER = os.getenv("REALTIME_AI_PROVIDER", "local")

# === Modes ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Legacy DEBUG_MODE for backward compatibility
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
DEBUG_MODE_INCLUDE_DELTA = (
    os.getenv("DEBUG_MODE_INCLUDE_DELTA", "false").lower() == "true"
)
TEXT_ONLY_MODE = os.getenv("TEXT_ONLY_MODE", "false").lower() == "true"
RUN_MODE = os.getenv("RUN_MODE", "normal").lower()

# === Billy Hardware ===
BILLY_MODEL = os.getenv("BILLY_MODEL", "modern").strip().lower()
BILLY_PINS = os.getenv("BILLY_PINS", "new").strip().lower()

# === Audio Config ===
SPEAKER_PREFERENCE = os.getenv("SPEAKER_PREFERENCE")
MIC_PREFERENCE = os.getenv("MIC_PREFERENCE")
MIC_TIMEOUT_SECONDS = int(os.getenv("MIC_TIMEOUT_SECONDS", "5"))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.3"))  # RMS threshold for detecting speech
CHUNK_MS = int(os.getenv("CHUNK_MS", "40"))
PLAYBACK_LATENCY = float(os.getenv("PLAYBACK_LATENCY", "0.5"))  # USB audio buffer latency in seconds (0.5s recommended for Pi)
USE_APLAY = os.getenv("USE_APLAY", "false").lower()  # "true" = use aplay subprocess, "false" = use sounddevice (default)
PLAYBACK_VOLUME = float(os.getenv("PLAYBACK_VOLUME", "1.0"))
TTS_LENGTH_SCALE = float(os.getenv("TTS_LENGTH_SCALE", "1.08"))
TTS_NOISE_SCALE = float(os.getenv("TTS_NOISE_SCALE", "0.55"))
TTS_NOISE_W = float(os.getenv("TTS_NOISE_W", "0.7"))
TTS_SENTENCE_SILENCE = float(os.getenv("TTS_SENTENCE_SILENCE", "0.12"))
TTS_VOCAL_PROCESSING = os.getenv("TTS_VOCAL_PROCESSING", "true").lower() == "true"
TTS_VOCAL_DARKEN = float(os.getenv("TTS_VOCAL_DARKEN", "0.35"))
TTS_VOCAL_DRIVE = float(os.getenv("TTS_VOCAL_DRIVE", "0.18"))
TTS_VOCAL_LOWPASS_HZ = float(os.getenv("TTS_VOCAL_LOWPASS_HZ", "4200"))
TTS_VOCAL_LIMIT = float(os.getenv("TTS_VOCAL_LIMIT", "0.92"))
MOUTH_ARTICULATION = int(os.getenv("MOUTH_ARTICULATION", "5"))
# Motor lip-sync (Piper TTS is quieter than song stems — use lower thresholds)
MOUTH_FLAP_THRESHOLD = int(os.getenv("MOUTH_FLAP_THRESHOLD", "350"))
TAIL_FLAP_THRESHOLD = int(os.getenv("TAIL_FLAP_THRESHOLD", "900"))
MOTOR_SYNC_HEAD = os.getenv("MOTOR_SYNC_HEAD", "true").lower() == "true"
MOTOR_SYNC_TAIL = os.getenv("MOTOR_SYNC_TAIL", "true").lower() == "true"
TURN_EAGERNESS = os.getenv("TURN_EAGERNESS", "high").strip().lower()
if TURN_EAGERNESS not in {"low", "medium", "high"}:
    TURN_EAGERNESS = "medium"

# Server VAD parameters based on eagerness
# Lower silence_duration_ms = faster turn detection (more eager)
# Higher threshold = less sensitive to noise (more conservative)
SERVER_VAD_PARAMS = {
    "low": {
        "threshold": 0.6,  # Less sensitive to background noise
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,  # Wait longer before responding
    },
    "medium": {
        "threshold": 0.5,  # Balanced sensitivity
        "prefix_padding_ms": 300,
        "silence_duration_ms": 300,  # Standard wait time
    },
    "high": {
        "threshold": 0.5,  # Standard sensitivity
        "prefix_padding_ms": 300,
        "silence_duration_ms": 200,  # Standard wait time
    },
}

# === GPIO Config ===
BUTTON_PIN = 27 if BILLY_PINS == "legacy" else 24  # legacy=pin 13, new=pin 18
# If head and mouth wires are swapped on the driver board, set INVERT_HEAD_MOUTH=true in .env
INVERT_HEAD_MOUTH = os.getenv("INVERT_HEAD_MOUTH", "false").lower() == "true"

# === Software Config ===
FLAP_ON_BOOT = os.getenv("FLAP_ON_BOOT", "false").lower() == "true"
MOCKFISH = os.getenv("MOCKFISH", "false").lower() == "true"


def is_classic_billy():
    return os.getenv("BILLY_MODEL", "modern").strip().lower() == "classic"


try:
    MAIN_LOOP = asyncio.get_event_loop()
except RuntimeError:
    MAIN_LOOP = None
