import asyncio
import os
import re
import wave
from typing import Optional

from .config import INSTRUCTIONS, TTS_VOICE, WAKEUP_LEAD_IN_MS
from .realtime_ai_provider import voice_provider_registry


WAKEUP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../sounds/wake-up/custom")
)
os.makedirs(WAKEUP_DIR, exist_ok=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_").lower()


def get_wakeup_path(phrase: str) -> str:
    return os.path.join(WAKEUP_DIR, f"{slugify(phrase)}.wav")


class WakeupClipGenerator:
    def __init__(self, *, voice: Optional[str] = None):

        # Get voice from persona if not specified
        if voice:
            self.voice = voice
        else:
            # Use the configured TTS voice so cached wakeups match the live voice.
            self.voice = TTS_VOICE

    async def generate(self, prompt: str, index: int) -> str:
        path = os.path.join(WAKEUP_DIR, f"{index}.wav")

        provider = voice_provider_registry.get_provider()

        print(f"🔊 Generating wakeup clip for: {prompt} → {index}")

        # Use single local INSTRUCTIONS as persona instructions
        persona_instructions = INSTRUCTIONS
        instructions = (
            "IMPORTANT: Always respond by speaking the exact user text out loud. Do not add, change or rephrase anything!\n\n"
            + persona_instructions
        )

        audio_bytes = await provider.generate_audio_clip(
            prompt="Repeat this literal message:" + prompt,
            voice=self.voice,
            instructions=instructions,
        )

        if WAKEUP_LEAD_IN_MS > 0:
            import numpy as np

            lead_samples = int(24000 * (WAKEUP_LEAD_IN_MS / 1000.0))
            if lead_samples > 0:
                audio_bytes = np.concatenate(
                    [
                        np.zeros(lead_samples, dtype=np.int16),
                        np.frombuffer(audio_bytes, dtype=np.int16),
                    ]
                ).astype(np.int16).tobytes()

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_bytes)

        print(f"✅ Saved wakeup clip: {path}")
        return path


def generate_wake_clip_async(prompt, index):
    async def _run():
        gen = WakeupClipGenerator()
        return await gen.generate(prompt, index)

    return asyncio.run(_run())
