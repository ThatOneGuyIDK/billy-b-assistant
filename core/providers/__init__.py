from ..config import OLLAMA_MODEL, TTS_VOICE, WHISPER_MODEL
from ..logger import logger
from ..realtime_ai_provider import voice_provider_registry
from .local_provider import LocalProvider


logger.verbose("Importing core.providers")
try:
    voice_provider_registry.register_provider(
        LocalProvider(ollama_model=OLLAMA_MODEL, whisper_model=WHISPER_MODEL, tts_voice=TTS_VOICE)
    )
    voice_provider_registry.set_default_provider("local")
    logger.success("Local provider registered")
except Exception as e:
    logger.warning(f"Could not register local provider: {e}")
