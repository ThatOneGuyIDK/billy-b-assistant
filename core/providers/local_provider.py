"""
Local AI Provider - Runs completely offline without external APIs
Uses local models for speech-to-text, LLM, and text-to-speech
"""
import asyncio
import base64
import io
import json
import queue
import wave
from typing import Any, Optional

import numpy as np
import requests

from ..logger import logger
from ..realtime_ai_provider import RealtimeAIProvider


class LocalSession:
    """
    Wrapper that mimics a WebSocket interface for local AI processing
    This allows the session.py code to work without modifications
    """

    def __init__(self, provider, instructions: str, tools: list[dict]):
        self.provider = provider
        self.instructions = instructions
        self.tools = tools
        self.conversation_history = []
        self.audio_buffer = []
        self.message_queue = asyncio.Queue()
        self.closed = False

    async def send(self, message: str):
        """Receive messages from session.py"""
        try:
            payload = json.loads(message)
            await self._handle_message(payload)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")

    async def recv(self):
        """Send messages back to session.py"""
        message = await self.message_queue.get()
        return json.dumps(message)

    def __aiter__(self):
        """Allow async iteration over messages"""
        return self

    async def __anext__(self):
        """Get next message"""
        if self.closed and self.message_queue.empty():
            raise StopAsyncIteration
        message = await self.message_queue.get()
        return json.dumps(message)

    async def close(self):
        """Close the session"""
        self.closed = True

    async def wait_closed(self):
        """Wait for session to close (compatibility with websocket)"""
        # Already closed, nothing to wait for
        pass

    async def _handle_message(self, payload: dict):
        """Process incoming messages"""
        msg_type = payload.get("type")

        if msg_type == "session.update":
            # Session configuration update
            await self._send_message({"type": "session.updated"})

        elif msg_type == "input_audio_buffer.append":
            # Store incoming audio
            audio_b64 = payload.get("audio", "")
            if audio_b64:
                self.audio_buffer.append(audio_b64)

        elif msg_type == "input_audio_buffer.commit":
            # Transcribe accumulated audio
            if self.audio_buffer:
                audio_data = b"".join(
                    [base64.b64decode(chunk) for chunk in self.audio_buffer]
                )
                
                # Send commit event
                await self._send_message({"type": "input_audio_buffer.committed"})
                
                # Transcribe
                text = await self.provider._speech_to_text(audio_data)
                
                if text:
                    # Send transcription event
                    await self._send_message({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": text,
                    })
                    
                    # Store for LLM
                    self.conversation_history.append({"role": "user", "content": text})
                
                self.audio_buffer = []

        elif msg_type == "response.create":
            # Generate LLM response
            await self._generate_response()

        elif msg_type == "conversation.item.create":
            # Handle text input
            item = payload.get("item", {})
            content = item.get("content", [])
            for part in content:
                if part.get("type") == "input_text":
                    text = part.get("text", "")
                    self.conversation_history.append({"role": "user", "content": text})

    async def _generate_response(self):
        """Generate LLM response using Ollama"""
        if not self.conversation_history:
            return

        # Build messages
        messages = [{"role": "system", "content": self.instructions}]
        messages.extend(self.conversation_history)

        try:
            # Send response.created event
            await self._send_message({
                "type": "response.created",
                "response": {"id": "resp_local"}
            })

            # Call Ollama
            response = requests.post(
                f"{self.provider.ollama_host}/api/chat",
                json={
                    "model": self.provider.ollama_model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                assistant_text = result.get("message", {}).get("content", "")

                # Store in history
                self.conversation_history.append(
                    {"role": "assistant", "content": assistant_text}
                )

                # Send text delta
                await self._send_message({
                    "type": "response.text.delta",
                    "delta": assistant_text,
                })

                # Generate audio from text
                audio_bytes = await self.provider._text_to_speech(
                    assistant_text, self.provider.voice
                )

                if audio_bytes:
                    # Send audio in chunks
                    chunk_size = 4800  # 100ms at 24kHz
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        await self._send_message({
                            "type": "response.audio.delta",
                            "delta": base64.b64encode(chunk).decode("utf-8"),
                        })

                # Send response.done
                await self._send_message({
                    "type": "response.done",
                    "response": {
                        "id": "resp_local",
                        "status": "completed",
                    },
                })

            else:
                logger.error(f"Ollama error: {response.status_code}")
                await self._send_message({
                    "type": "error",
                    "error": {"message": f"Ollama error: {response.status_code}"},
                })

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            await self._send_message({
                "type": "error",
                "error": {"message": str(e)},
            })

    async def _send_message(self, message: dict):
        """Queue a message to send to session.py"""
        await self.message_queue.put(message)


class LocalProvider(RealtimeAIProvider):
    """
    Local-only AI provider using:
    - Whisper (faster-whisper) for speech-to-text
    - Ollama for LLM conversation
    - Piper TTS for text-to-speech
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:latest",
        whisper_model: str = "base",
        tts_voice: str = "en_US-lessac-medium",
    ):
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.whisper_model = whisper_model
        self.tts_voice = tts_voice
        self.voice = tts_voice

        # Lazy imports for optional dependencies
        self._whisper_model = None
        self._tts_engine = None

        logger.info(
            f"LocalProvider initialized: Ollama={ollama_host}, Model={ollama_model}",
            "🏠",
        )

    @property
    def default_voice(self) -> str:
        """Default TTS voice"""
        return "en_US-lessac-medium"

    def get_supported_voices(self) -> list[str]:
        """List of available Piper TTS voices"""
        return [
            "en_US-lessac-medium",
            "en_US-amy-medium",
            "en_US-danny-low",
            "en_GB-alan-medium",
            "en_GB-alba-medium",
        ]

    def get_provider_name(self) -> str:
        return "local"

    def get_provider_tools(self) -> list[dict]:
        """Return provider-specific tools (none for local)"""
        return []

    # ==================== Core Methods ====================

    async def generate_audio_clip(
        self,
        prompt: str,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """
        Generate audio from text using local TTS
        This is used for simple text-to-speech without LLM processing
        """
        if voice is None:
            voice = self.default_voice

        logger.info(f"Generating audio clip: {prompt[:50]}...", "🔊")

        # Use local TTS to convert text to audio
        audio_bytes = await self._text_to_speech(prompt, voice)

        if not audio_bytes:
            raise RuntimeError("No audio data generated from local TTS")

        return audio_bytes

    async def connect(
        self, instructions: str, tools: list[dict[str, Any]], **kwargs
    ) -> LocalSession:
        """
        Connect to local services (Ollama, Whisper, TTS)
        Returns a LocalSession object that mimics a WebSocket interface
        """
        logger.info("Connecting to local AI services...", "🔌")

        # Check Ollama availability
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.success("Ollama connection successful")
            else:
                logger.warning(f"Ollama responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            raise RuntimeError(f"Ollama not available at {self.ollama_host}")

        # Initialize Whisper model (lazy load)
        self._ensure_whisper_loaded()

        # Initialize TTS engine (lazy load)
        self._ensure_tts_loaded()

        # Create and return session wrapper
        session = LocalSession(self, instructions, tools)

        logger.success("Local provider connected and ready")
        return session

    async def send_message(self, ws, payload: dict[str, Any]):
        """
        Send a message to the session
        For LocalProvider, ws is actually a LocalSession instance
        """
        await ws.send(json.dumps(payload))

    # ==================== Private Helper Methods ====================

    def _ensure_whisper_loaded(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel

                self._whisper_model = WhisperModel(
                    self.whisper_model, device="cpu", compute_type="int8"
                )
                logger.success(f"Whisper model '{self.whisper_model}' loaded")
            except ImportError:
                logger.error(
                    "faster-whisper not installed. Install with: pip install faster-whisper"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                raise

    def _ensure_tts_loaded(self):
        """Lazy load TTS engine"""
        if self._tts_engine is None:
            # TODO: Initialize Piper TTS or alternative
            # For now, we'll use a placeholder
            logger.warning("TTS engine not yet implemented - using placeholder")
            self._tts_engine = "placeholder"

    async def _speech_to_text(self, audio_bytes: bytes) -> str:
        """Convert audio to text using Whisper"""
        self._ensure_whisper_loaded()

        try:
            # Convert raw PCM to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0  # Normalize to [-1, 1]

            # Transcribe with Whisper
            segments, _ = self._whisper_model.transcribe(audio_np, language="en")
            text = " ".join([segment.text for segment in segments])

            return text.strip()
        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            return ""

    async def _text_to_speech(self, text: str, voice: str) -> bytes:
        """Convert text to audio using local TTS"""
        self._ensure_tts_loaded()

        # TODO: Implement Piper TTS integration
        # For now, return silence as placeholder
        logger.warning("TTS not yet fully implemented - returning placeholder audio")

        # Generate 1 second of silence as placeholder
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.int16)

        return audio_data.tobytes()

    # ==================== WebSocket Compatibility Methods ====================

    def _get_websocket_uri(self) -> str:
        """Not used for local provider (no websocket)"""
        return ""

    def _get_headers(self) -> dict[str, str]:
        """Not used for local provider"""
        return {}
