import os

from ..config import ENV_PATH
from ..logger import logger


logger.verbose("Importing core.providers")
# Register realtime AI providers
from ..config import OPENAI_API_KEY, OPENAI_MODEL, REALTIME_AI_PROVIDER, XAI_API_KEY
from ..realtime_ai_provider import voice_provider_registry
from .openai_provider import OpenAIProvider
from .xai_provider import XAIProvider
from .local_provider import LocalProvider


logger.verbose(f"OPENAI_API_KEY set: {bool(OPENAI_API_KEY)}")
logger.verbose(f"XAI_API_KEY set: {bool(XAI_API_KEY)}")
logger.verbose(f"REALTIME_AI_PROVIDER: {REALTIME_AI_PROVIDER}")

# Always register local provider (no API key needed)
try:
    local_provider = LocalProvider()
    voice_provider_registry.register_provider(local_provider)
    logger.success("Local provider registered (Ollama + Whisper + TTS)")
except Exception as e:
    logger.warning(f"Could not register local provider: {e}")

if OPENAI_API_KEY:
    openai_provider = OpenAIProvider(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    voice_provider_registry.register_provider(openai_provider)

if XAI_API_KEY:
    xai_provider = XAIProvider(api_key=XAI_API_KEY)
    voice_provider_registry.register_provider(xai_provider)

# Set the default provider based on configuration
if REALTIME_AI_PROVIDER:
    voice_provider_registry.set_default_provider(REALTIME_AI_PROVIDER)
elif OPENAI_API_KEY and not XAI_API_KEY:
    voice_provider_registry.set_default_provider("openai")
elif XAI_API_KEY and not OPENAI_API_KEY:
    voice_provider_registry.set_default_provider("xai")
elif OPENAI_API_KEY and XAI_API_KEY:
    # Both keys are set, default to OpenAI if no explicit provider is specified
    logger.info(
        "Both OpenAI and XAI API keys are set. Defaulting to OpenAI. Set REALTIME_AI_PROVIDER to choose a different default."
    )
    voice_provider_registry.set_default_provider("openai")
else:
    # No API keys - default to local provider
    logger.info("No API keys configured - defaulting to local provider")
    voice_provider_registry.set_default_provider("local")
