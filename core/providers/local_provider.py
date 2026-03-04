"""
Local AI Provider - Runs completely offline without external APIs
Uses local models for speech-to-text, LLM, and text-to-speech
"""
import asyncio
import base64
import io
import json
import queue
import re
import wave
from typing import Any, Optional

import numpy as np
import requests

from ..config import DEBUG_MODE
from ..logger import logger
from ..realtime_ai_provider import RealtimeAIProvider


def _clean_tts_text(text: str) -> str:
    """Clean text for TTS - remove formatting, limit length, remove problematic punctuation"""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'__(.+?)__', r'\1', text)  # __bold__ -> bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # *italic* -> italic
    text = re.sub(r'_(.+?)_', r'\1', text)  # _italic_ -> italic
    text = re.sub(r'`(.+?)`', r'\1', text)  # `code` -> code
    
    # Remove list markers and extra formatting
    text = re.sub(r'^[\s]*[-*•]\s+', '', text, flags=re.MULTILINE)  # Remove bullet points
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)  # Remove numbered lists
    text = re.sub(r'^[\s]*#+\s+', '', text, flags=re.MULTILINE)  # Remove markdown headers
    
    # Replace multiple newlines with space
    text = re.sub(r'\n\n+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    
    # Remove all forbidden punctuation except periods
    text = re.sub(r'[,;:!?\-()[\]{}"\'\`]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Enforce concise responses while avoiding mid-sentence cutoffs
    max_words = 50
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    kept_sentences = []
    total_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if total_words + len(sentence_words) <= max_words:
            kept_sentences.append(sentence)
            total_words += len(sentence_words)
        else:
            break

    if kept_sentences:
        text = '. '.join(kept_sentences).strip()
    else:
        # Fallback: hard truncate only if first sentence is already too long
        text = ' '.join(text.split()[:max_words]).strip()

    # Ensure clean ending for TTS
    text = text.rstrip("'\"-_,;:")
    if text and not text.endswith('.'):
        text += '.'

    return text


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
        self._close_sentinel = {"type": "__session_closed__"}

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
        if message == self._close_sentinel:
            raise StopAsyncIteration
        return json.dumps(message)

    def __aiter__(self):
        """Allow async iteration over messages"""
        return self

    async def __anext__(self):
        """Get next message"""
        if self.closed and self.message_queue.empty():
            raise StopAsyncIteration
        message = await self.message_queue.get()
        if message == self._close_sentinel:
            raise StopAsyncIteration
        return json.dumps(message)

    async def close(self):
        """Close the session"""
        self.closed = True
        # Unblock any pending queue consumers (async for / recv)
        try:
            self.message_queue.put_nowait(self._close_sentinel)
        except Exception:
            pass

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
                if len(self.audio_buffer) <= 1:  # Log first chunk only
                    logger.info(f"🎙️ LocalSession: Received audio buffer, total chunks: {len(self.audio_buffer)}", "🔊")

        elif msg_type == "input_audio_buffer.commit":
            # Transcribe accumulated audio
            logger.info(f"🔧 Commit received with {len(self.audio_buffer)} audio chunks", "🔧")
            if self.audio_buffer:
                audio_data = b"".join(
                    [base64.b64decode(chunk) for chunk in self.audio_buffer]
                )
                logger.info(f"🔧 Audio data size: {len(audio_data)} bytes", "🔧")
                
                # Send commit event
                await self._send_message({"type": "input_audio_buffer.committed"})
                
                # Transcribe
                logger.info("🔧 Calling Whisper for transcription...", "🔧")
                text = await self.provider._speech_to_text(audio_data)
                logger.info(f"🔧 Whisper returned: '{text}'", "🔧")
                
                if text:
                    # Send transcription event
                    logger.info(f"📝 Sending transcription: {text}", "📝")
                    await self._send_message({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": text,
                    })
                    
                    # Store for LLM
                    self.conversation_history.append({"role": "user", "content": text})
                else:
                    logger.warning("⚠️ Whisper returned empty text", "⚠️")
                
                self.audio_buffer = []
            else:
                logger.warning("⚠️ Commit received but audio buffer is empty", "⚠️")

        elif msg_type == "response.create":
            # Generate LLM response in background so message loop can continue
            asyncio.create_task(self._generate_response())

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

        # Build messages with TTS optimization
        from ..config import TTS_OPTIMIZATION_PROMPT
        instructions_with_tts = self.instructions
        if TTS_OPTIMIZATION_PROMPT:
            instructions_with_tts += f"\n\n{TTS_OPTIMIZATION_PROMPT}"
        
        messages = [{"role": "system", "content": instructions_with_tts}]
        messages.extend(self.conversation_history)

        try:
            # Send response.created event
            await self._send_message({
                "type": "response.created",
                "response": {"id": "resp_local"}
            })

            # Call Ollama (in a worker thread so event loop can keep running thinking indicator)
            response = await asyncio.to_thread(
                requests.post,
                f"{self.provider.ollama_host}/api/chat",
                json={
                    "model": self.provider.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.3,
                    },
                },
                timeout=(5, 300),
            )

            if response.status_code == 200:
                result = response.json()
                assistant_text = (
                    result.get("message", {}).get("content")
                    or result.get("response")
                    or result.get("text")
                    or ""
                )
                assistant_text = assistant_text.strip()

                if not assistant_text:
                    logger.warning(f"Ollama returned empty text payload: {result}")
                    assistant_text = "I heard you, but I couldn't generate a response this time."
                
                # Clean text for TTS - remove formatting and enforce constraints
                assistant_text = _clean_tts_text(assistant_text)
                
                if not assistant_text:
                    assistant_text = "I heard you"

                # Store in history
                self.conversation_history.append(
                    {"role": "assistant", "content": assistant_text}
                )

                # Send text delta
                await self._send_message({
                    "type": "response.text.delta",
                    "delta": assistant_text,
                })

                # Send text done (helps clients that rely on done events)
                await self._send_message({
                    "type": "response.text.done",
                    "text": assistant_text,
                })

                # Generate audio from text
                audio_bytes = await self.provider._text_to_speech(
                    assistant_text, self.provider.voice
                )

                logger.info(f"🔊 Audio bytes returned: {len(audio_bytes) if audio_bytes else 0} bytes")

                if audio_bytes and len(audio_bytes) > 0:
                    # Send audio in chunks
                    chunk_size = 4800  # 100ms at 24kHz
                    num_chunks = 0
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        await self._send_message({
                            "type": "response.output_audio.delta",
                            "delta": base64.b64encode(chunk).decode("utf-8"),
                        })
                        num_chunks += 1
                    logger.info(f"🔊 Sent {num_chunks} audio chunks")
                else:
                    logger.warning("🔊 No audio data to send")

                # Send response.done
                await self._send_message({
                    "type": "response.done",
                    "response": {
                        "id": "resp_local",
                        "status": "completed",
                    },
                })

            else:
                error_msg = f"Ollama error: {response.status_code}"
                try:
                    error_body = response.json()
                    error_msg += f" - {error_body}"
                except:
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
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
        ollama_model: str = "llama2:latest",
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

        # For local provider, strip out function-calling instructions since Ollama can't call functions
        # Only keep persona/personality instructions
        simplified_instructions = instructions
        if "=== SPECIAL POWERS ===" in simplified_instructions:
            # Remove the TOOL_INSTRUCTIONS section, keep only persona
            simplified_instructions = simplified_instructions.split("=== SPECIAL POWERS ===")[0].strip()
        
        # Create and return session wrapper
        session = LocalSession(self, simplified_instructions, tools)

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
            try:
                from piper import PiperVoice
                import os
                
                # Local Piper model path
                model_dir = os.path.expanduser("~/.piper/models")
                configured_model_path = os.path.join(model_dir, f"{self.tts_voice}.onnx")
                fallback_voice = "en_US-lessac-medium"
                fallback_model_path = os.path.join(model_dir, f"{fallback_voice}.onnx")

                model_path = configured_model_path
                if not os.path.exists(configured_model_path):
                    if os.path.exists(fallback_model_path):
                        logger.warning(
                            f"Configured Piper voice '{self.tts_voice}' not found. "
                            f"Falling back to '{fallback_voice}'."
                        )
                        model_path = fallback_model_path
                    else:
                        raise FileNotFoundError(
                            f"Piper model not found at {configured_model_path} and fallback not found at {fallback_model_path}.\n"
                            f"Download voices from: https://huggingface.co/rhasspy/piper-voices"
                        )
                
                # Load Piper with local model file
                self._tts_engine = PiperVoice.load(model_path, use_cuda=False)
                
                logger.success(
                    f"Piper TTS loaded (offline) - model: {os.path.basename(model_path)}"
                )
            except ImportError:
                logger.error(
                    "piper-tts not installed. Install with: pip install piper-tts"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load Piper TTS: {e}")
                raise

    async def _speech_to_text(self, audio_bytes: bytes) -> str:
        """Convert audio to text using Whisper"""
        self._ensure_whisper_loaded()

        try:
            # Convert raw PCM to numpy array
            audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
            if audio_i16.size == 0:
                return ""

            # Skip near-silence/noise to avoid random hallucinated words
            peak = int(np.max(np.abs(audio_i16)))
            rms = float(np.sqrt(np.mean(np.square(audio_i16.astype(np.float32)))))
            if peak < 80 and rms < 1.0:
                logger.info(
                    f"Whisper skipped: audio too quiet (peak={peak}, rms={rms:.2f})"
                )
                return ""

            audio_np = audio_i16.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Transcribe with Whisper - improved settings for accuracy
            segments, _ = self._whisper_model.transcribe(
                audio_np,
                language="en",
                beam_size=10,  # Increased for better accuracy
                best_of=10,    # Increased for better accuracy
                temperature=0.0,  # Deterministic output
                vad_filter=True,  # Filter silence
                condition_on_previous_text=False,  # Don't hallucinate
            )
            text = " ".join([segment.text for segment in segments]).strip()

            logger.debug(f"Whisper transcribed: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            return ""

    def _text_to_speech_blocking(self, text: str, voice: str) -> bytes:
        """Blocking TTS synthesis helper (runs in a worker thread)."""
        self._ensure_tts_loaded()

        try:
            logger.debug(f"Generating TTS for: {text[:50]}...")

            # Generate audio chunks from Piper
            audio_chunks = []

            for chunk in self._tts_engine.synthesize(text):
                # chunk.audio_int16_bytes is the raw PCM audio at the model's native rate
                audio_bytes = chunk.audio_int16_bytes
                logger.debug(f"Piper chunk: {len(audio_bytes)} bytes at {chunk.sample_rate}Hz")
                audio_chunks.append(audio_bytes)

            # Concatenate all audio chunks
            if not audio_chunks:
                logger.warning("No audio generated from Piper")
                return np.zeros(24000, dtype=np.int16).tobytes()

            combined_audio = b''.join(audio_chunks)
            logger.info(f"🔊 Combined audio: {len(combined_audio)} bytes")

            # Convert to numpy array
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)

            # Piper generates 22050 Hz, resample to 24kHz
            if len(audio_array) > 0:
                from scipy.signal import resample_poly

                source_rate = 22050
                target_rate = 24000
                # Higher-quality resampling than linear interpolation (reduces crackle/robotic artifacts)
                audio_f32 = audio_array.astype(np.float32)
                resampled = resample_poly(audio_f32, target_rate, source_rate)
                audio_array = np.clip(resampled, -32768, 32767).astype(np.int16)

            logger.info(f"🔊 TTS generated {len(audio_array)} samples at 24kHz")
            return audio_array.tobytes()

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            # Return 1 second of silence as fallback
            return np.zeros(24000, dtype=np.int16).tobytes()

    async def _text_to_speech(self, text: str, voice: str) -> bytes:
        """Convert text to audio using offline Piper TTS."""
        return await asyncio.to_thread(self._text_to_speech_blocking, text, voice)

    # ==================== WebSocket Compatibility Methods ====================

    def _get_websocket_uri(self) -> str:
        """Not used for local provider (no websocket)"""
        return ""

    def _get_headers(self) -> dict[str, str]:
        """Not used for local provider"""
        return {}
