"""
Local AI Provider - Runs completely offline without external APIs
Uses local models for speech-to-text, LLM, and text-to-speech
"""
import asyncio
import base64
import json
import re
from typing import Any, Optional

import numpy as np
import requests
from scipy.signal import butter, sosfilt
import os
import time
import wave

from ..config import (
    DEBUG_MODE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    PIPER_MODEL_DIR,
    TTS_LENGTH_SCALE,
    TTS_VOCAL_DARKEN,
    TTS_VOCAL_DRIVE,
    TTS_VOCAL_LIMIT,
    TTS_VOCAL_LOWPASS_HZ,
    TTS_VOCAL_PROCESSING,
    TTS_NOISE_SCALE,
    TTS_NOISE_W,
    TTS_SENTENCE_SILENCE,
    WHISPER_INITIAL_PROMPT,
    WHISPER_BEST_OF,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
)
from ..logger import logger
from ..realtime_ai_provider import RealtimeAIProvider
from ..workout_intent import WorkoutIntentResult, route_workout_text


def _clean_tts_text(text: str) -> str:
    """Clean text for TTS - remove formatting, limit length, remove problematic punctuation"""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'__(.+?)__', r'\1', text)  # __bold__ -> bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # *italic* -> italic
    text = re.sub(r'_(.+?)_', r'\1', text)  # _italic_ -> italic
    text = re.sub(r'`(.+?)`', r'\1', text)  # `code` -> code
    text = text.replace('*', '')  # Remove any stray asterisks
    
    # Remove list markers and extra formatting
    text = re.sub(r'^[\s]*[-*•]\s+', '', text, flags=re.MULTILINE)  # Remove bullet points
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)  # Remove numbered lists
    text = re.sub(r'^[\s]*#+\s+', '', text, flags=re.MULTILINE)  # Remove markdown headers
    
    # Replace multiple newlines with space
    text = re.sub(r'\n\n+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    
    # Keep pause-friendly punctuation so Piper can sound more natural.
    # Normalize separators to commas/periods instead of removing everything.
    text = re.sub(r'[;:]+', ',', text)
    text = re.sub(r'[—–-]+', ', ', text)
    # Remove only punctuation that tends to hurt TTS prosody
    text = re.sub(r'[()\[\]{}"\'\`]', '', text)
    
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


_VOCAL_COLOR_FILTER = None


def _apply_vocal_color(audio_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Darken and thicken TTS audio to make the voice sound rougher and fuller."""
    if not TTS_VOCAL_PROCESSING or not audio_bytes:
        return audio_bytes

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return audio_bytes

    global _VOCAL_COLOR_FILTER
    if _VOCAL_COLOR_FILTER is None:
        nyquist = sample_rate / 2.0
        cutoff_hz = min(max(TTS_VOCAL_LOWPASS_HZ, 300.0), nyquist * 0.95)
        _VOCAL_COLOR_FILTER = butter(
            2,
            cutoff_hz / nyquist,
            btype="low",
            output="sos",
        )

    low_band = sosfilt(_VOCAL_COLOR_FILTER, audio)

    # Blend in the darker band, then saturate slightly for a rougher texture.
    blended = audio * (1.0 - TTS_VOCAL_DARKEN) + low_band * TTS_VOCAL_DARKEN
    driven = np.tanh(blended / 32768.0 * (1.0 + TTS_VOCAL_DRIVE * 6.0)) * 32768.0

    peak = float(np.max(np.abs(driven))) if driven.size else 0.0
    if peak > 0:
        driven *= min(1.0, (32767.0 * TTS_VOCAL_LIMIT) / peak)

    return np.clip(driven, -32768, 32767).astype(np.int16).tobytes()


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

    def _route_user_text(self, text: str) -> WorkoutIntentResult:
        """Run a fast workout intent pass before the LLM sees user text."""
        result = route_workout_text(text)
        if result.action != "chat":
            logger.info(
                f"Workout intent routed as {result.action} (confidence={result.confidence})",
                "🏋️",
            )
        return result

    async def _send_direct_response(self, text: str, voice: Optional[str] = None):
        """Send a prebuilt assistant response without calling Ollama."""
        response_text = _clean_tts_text(text)
        if not response_text:
            response_text = text

        await self._send_message({
            "type": "response.created",
            "response": {"id": "resp_local"},
        })

        await self._send_message({"type": "response.text.delta", "delta": response_text})

        audio_bytes = await self.provider.generate_audio_clip(
            response_text,
            voice=voice or self.voice,
        )

        chunk_size = 9600
        for start in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[start : start + chunk_size]
            if chunk:
                await self._send_message(
                    {
                        "type": "response.output_audio.delta",
                        "delta": base64.b64encode(chunk).decode("utf-8"),
                    }
                )

        await self._send_message({"type": "response.text.done", "text": response_text})
        await self._send_message(
            {
                "type": "response.done",
                "response": {"id": "resp_local", "status": "completed"},
            }
        )

    async def _run_workout_automation(self, result: WorkoutIntentResult):
        """Handle instant workout timer/set-counter commands locally."""
        count = max(1, int(result.target_count or 60))
        spoken_sequence = result.spoken_sequence or [str(count)]

        intro = "Set counter." if result.action == "set_counter" else "Timer."
        script = ". ".join([intro, *spoken_sequence])
        await self._send_direct_response(script)

    async def _run_song_automation(self, result: WorkoutIntentResult):
        """Handle song requests locally before the LLM turn runs."""
        song_name = result.song_name or result.metadata.get("song_name")
        if not song_name:
            logger.warning("Song request detected but no song was selected.")
            return

        from .. import audio

        logger.info(f"Playing requested song: {song_name}", "🎵")
        await audio.play_song(song_name)

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
                
                # Start thinking sound BEFORE transcription to eliminate awkward silence
                await self._send_message({"type": "response.thinking_started"})
                # Yield so session loop can process thinking_started immediately
                await asyncio.sleep(0)
                
                # Transcribe
                logger.info("🔧 Calling Whisper for transcription...", "🔧")
                text = await self.provider._speech_to_text(audio_data)
                logger.info(f"🔧 Whisper returned: '{text}'", "🔧")
                
                if text:
                    # Send transcription event
                    logger.info(f"📝 Sending transcription: {text}", "📝")
                    routed = self._route_user_text(text)
                    await self._send_message({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": text,
                    })

                    if routed.action in {"timer", "set_counter"}:
                        asyncio.create_task(self._run_workout_automation(routed))
                        self.audio_buffer = []
                        return
                    
                    # Store for LLM
                    self.conversation_history.append({"role": "user", "content": routed.normalized_text or text})
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
                    routed = self._route_user_text(text)
                    if routed.action in {"timer", "set_counter"}:
                        asyncio.create_task(self._run_workout_automation(routed))
                        return
                    if routed.action == "song":
                        asyncio.create_task(self._run_song_automation(routed))
                        return
                    self.conversation_history.append(
                        {"role": "user", "content": routed.normalized_text or text}
                    )

    async def _generate_response(self):
        """Generate LLM response using Ollama"""
        if not self.conversation_history:
            return

        # Build messages (TTS optimization prompt removed)
        messages = [{"role": "system", "content": self.instructions}]
        messages.extend(self.conversation_history)

        try:
            # Send response.created event
            await self._send_message({
                "type": "response.created",
                "response": {"id": "resp_local"}
            })

            def _pop_ready_sentence(buf: str):
                for i, ch in enumerate(buf):
                    if ch in ".!?" and (i + 1 == len(buf) or buf[i + 1].isspace()):
                        return buf[: i + 1], buf[i + 1 :].lstrip()
                return None, buf

            async def _send_tts_for_text(text_piece: str) -> int:
                cleaned = _clean_tts_text(text_piece)
                if not cleaned:
                    return 0
                audio_bytes = await self.provider._text_to_speech(cleaned, self.provider.voice)
                if not audio_bytes:
                    return 0

                chunk_size = 9600  # 200ms at 24kHz
                sent = 0
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i : i + chunk_size]
                    await self._send_message(
                        {
                            "type": "response.output_audio.delta",
                            "delta": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                    sent += 1
                return sent

            delta_queue: asyncio.Queue = asyncio.Queue()
            stream_done = object()
            stream_error: dict[str, Any] = {"message": ""}
            loop = asyncio.get_running_loop()

            def _stream_ollama():
                try:
                    with self.provider._ollama_session.post(
                        f"{self.provider.ollama_host}/api/chat",
                        json={
                            "model": self.provider.ollama_model,
                            "messages": messages,
                            "stream": True,
                            "options": {
                                "num_predict": OLLAMA_NUM_PREDICT,
                                "temperature": OLLAMA_TEMPERATURE,
                                "num_ctx": OLLAMA_NUM_CTX,
                            },
                        },
                        stream=True,
                        timeout=(5, 300),
                    ) as response:
                        if response.status_code != 200:
                            try:
                                err = response.json()
                                msg = f"Ollama error: {response.status_code} - {err}"
                            except Exception:
                                msg = f"Ollama error: {response.status_code} - {response.text}"
                            stream_error["message"] = msg
                            return

                        for line in response.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            try:
                                payload = json.loads(line)
                            except Exception:
                                continue

                            delta = (
                                payload.get("message", {}).get("content")
                                or payload.get("response")
                                or ""
                            )
                            if delta:
                                loop.call_soon_threadsafe(delta_queue.put_nowait, delta)

                            if payload.get("done"):
                                break
                except Exception as e:
                    stream_error["message"] = str(e)
                finally:
                    loop.call_soon_threadsafe(delta_queue.put_nowait, stream_done)

            producer_task = asyncio.create_task(asyncio.to_thread(_stream_ollama))

            full_text = ""
            pending_for_tts = ""
            total_audio_chunks = 0

            while True:
                item = await delta_queue.get()
                if item is stream_done:
                    break

                delta_text = str(item).replace("*", "")
                if not delta_text:
                    continue
                full_text += delta_text
                pending_for_tts += delta_text

                await self._send_message({"type": "response.text.delta", "delta": delta_text})

                # Start speaking as soon as we have a complete sentence.
                while True:
                    sentence, pending_for_tts = _pop_ready_sentence(pending_for_tts)
                    if not sentence:
                        break
                    total_audio_chunks += await _send_tts_for_text(sentence)

            await producer_task

            if stream_error["message"]:
                logger.error(stream_error["message"])
                await self._send_message(
                    {
                        "type": "error",
                        "error": {"message": stream_error["message"]},
                    }
                )
                return

            # Flush trailing partial sentence if any
            if pending_for_tts.strip():
                total_audio_chunks += await _send_tts_for_text(pending_for_tts)

            assistant_text = _clean_tts_text(full_text.strip())
            if not assistant_text:
                assistant_text = "I heard you"

            self.conversation_history.append({"role": "assistant", "content": assistant_text})

            await self._send_message({"type": "response.text.done", "text": assistant_text})
            logger.info(f"🔊 Sent {total_audio_chunks} audio chunks")

            await self._send_message(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_local",
                        "status": "completed",
                    },
                }
            )

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
        self._ollama_session = requests.Session()

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
            response = self._ollama_session.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = []
                try:
                    tags_payload = response.json()
                    if isinstance(tags_payload, dict):
                        for model_info in tags_payload.get("models", []):
                            if not isinstance(model_info, dict):
                                continue
                            model_name = model_info.get("name") or model_info.get("model")
                            if model_name:
                                available_models.append(str(model_name))
                except Exception:
                    available_models = []

                resolved_model = self._resolve_ollama_model(available_models)
                if resolved_model != self.ollama_model:
                    logger.warning(
                        f"Ollama model '{self.ollama_model}' is not available. Falling back to '{resolved_model}'."
                    )
                    self.ollama_model = resolved_model

                logger.success("Ollama connection successful")
            else:
                logger.warning(f"Ollama responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            raise RuntimeError(f"Ollama not available at {self.ollama_host}")

        # Whisper/TTS are warmed at boot via button preload; load on demand if skipped.
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

    def _cached_whisper_snapshot_path(self, download_root: str | None) -> str | None:
        """Return a local HF snapshot dir if the model is already on disk."""
        root = download_root or os.path.expanduser("~/.cache/huggingface")
        snapshots_dir = os.path.join(
            root,
            "hub",
            f"models--Systran--faster-whisper-{self.whisper_model}",
            "snapshots",
        )
        if not os.path.isdir(snapshots_dir):
            return None
        for snap_id in sorted(os.listdir(snapshots_dir)):
            path = os.path.join(snapshots_dir, snap_id)
            if not os.path.isdir(path):
                continue
            if os.path.isfile(os.path.join(path, "model.bin")) or os.path.isfile(
                os.path.join(path, "config.json")
            ):
                return path
        return None

    def _ensure_whisper_loaded(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel

                download_root = os.getenv("WHISPER_DOWNLOAD_ROOT") or os.getenv(
                    "HF_HOME"
                )
                cached_path = self._cached_whisper_snapshot_path(download_root)
                model_source = cached_path or self.whisper_model
                kwargs: dict[str, Any] = {
                    "device": "cpu",
                    "compute_type": "int8",
                }
                if download_root and not cached_path:
                    kwargs["download_root"] = download_root

                # HF_HUB_OFFLINE breaks snapshot resolution even when weights exist.
                saved_offline = os.environ.pop("HF_HUB_OFFLINE", None)
                try:
                    self._whisper_model = WhisperModel(model_source, **kwargs)
                finally:
                    if saved_offline is not None:
                        os.environ["HF_HUB_OFFLINE"] = saved_offline

                if cached_path:
                    logger.debug(f"Whisper loaded from cache path: {cached_path}")
                logger.success(f"Whisper model '{self.whisper_model}' loaded")
            except ImportError:
                logger.error(
                    "faster-whisper not installed. Install with: pip install faster-whisper"
                )
                raise
            except Exception as e:
                err = str(e)
                logger.error(f"Failed to load Whisper: {e}")
                if "HF_HUB_OFFLINE" in err or "outgoing traffic has been disabled" in err:
                    logger.error(
                        "Remove HF_HUB_OFFLINE=1 from .env — it blocks loading cached Whisper models."
                    )
                if "parse_error" in err or "json.exception" in err:
                    cache = os.getenv("HF_HOME", "~/.cache/huggingface")
                    logger.error(
                        f"Whisper cache for '{self.whisper_model}' looks corrupt "
                        f"(interrupted download). In .env set WHISPER_MODEL=base "
                        f"or remove: {cache}/hub/models--Systran--faster-whisper-{self.whisper_model}"
                    )
                raise

    def _ensure_tts_loaded(self):
        """Lazy load TTS engine"""
        if self._tts_engine is None:
            try:
                from piper import PiperVoice
                import os
                
                model_dir = PIPER_MODEL_DIR
                os.makedirs(model_dir, exist_ok=True)
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
                            f"Piper model not found at {configured_model_path} "
                            f"and fallback not found at {fallback_model_path}. "
                            f"Run once (online): ./setup/preload_piper.sh"
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

    def _resolve_ollama_model(self, available_models: list[str]) -> str:
        """Choose the best available Ollama model for this session."""
        if not available_models:
            return self.ollama_model

        available_lookup = {model.strip(): model.strip() for model in available_models if model}
        configured_model = self.ollama_model.strip()

        if configured_model in available_lookup:
            return available_lookup[configured_model]

        preferred_models = [
            "llama3.2:latest",
            "llama3.1:8b",
            "mistral:latest",
            "phi3:mini",
            "llama2:latest",
        ]
        for model_name in preferred_models:
            if model_name in available_lookup:
                return available_lookup[model_name]

        return available_models[0].strip()

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
                # Save the near-silent audio so it can be inspected offline
                try:
                    debug_dir = os.path.join("sounds", "response-history")
                    os.makedirs(debug_dir, exist_ok=True)
                    fname = os.path.join(debug_dir, f"whisper-skip-{int(time.time())}.wav")
                    with wave.open(fname, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(audio_bytes)
                    logger.info(f"Saved skipped (too-quiet) audio to {fname}")
                except Exception:
                    pass
                return ""

            audio_np = audio_i16.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            def _transcribe_blocking():
                return self._whisper_model.transcribe(
                    audio_np,
                    language="en",
                    beam_size=WHISPER_BEAM_SIZE,
                    best_of=WHISPER_BEST_OF,
                    temperature=0.0,  # Deterministic output
                    vad_filter=WHISPER_VAD_FILTER,
                    initial_prompt=WHISPER_INITIAL_PROMPT,
                    condition_on_previous_text=False,  # Don't hallucinate
                )

            # Run Whisper in a worker thread so event loop can keep processing
            # queued events (e.g., thinking_started) immediately.
            segments, _ = await asyncio.to_thread(_transcribe_blocking)
            text = " ".join([segment.text for segment in segments]).strip()

            logger.debug(f"Whisper transcribed: '{text}'")
            if not text:
                # Save the raw audio for offline inspection when transcription is empty
                try:
                    debug_dir = os.path.join("sounds", "response-history")
                    os.makedirs(debug_dir, exist_ok=True)
                    fname = os.path.join(debug_dir, f"whisper-failed-{int(time.time())}.wav")
                    with wave.open(fname, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(audio_bytes)
                    logger.info(f"Saved failed Whisper audio to {fname}")
                except Exception:
                    pass
            return text
        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            try:
                debug_dir = os.path.join("sounds", "response-history")
                os.makedirs(debug_dir, exist_ok=True)
                fname = os.path.join(debug_dir, f"whisper-exception-{int(time.time())}.wav")
                with wave.open(fname, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_bytes)
                logger.info(f"Saved exception Whisper audio to {fname}")
            except Exception:
                pass
            return ""

    def _text_to_speech_blocking(self, text: str, voice: str) -> bytes:
        """Blocking TTS synthesis helper (runs in a worker thread)."""
        self._ensure_tts_loaded()

        try:
            logger.debug(f"Generating TTS for: {text[:50]}...")

            # Generate audio sentence-by-sentence to avoid clipping at sentence boundaries
            audio_chunks = []
            sample_rate = 22050

            # Tiny preroll helps some USB audio paths avoid clipping onset
            preroll_samples = int(0.02 * sample_rate)  # 20 ms
            audio_chunks.append(np.zeros(preroll_samples, dtype=np.int16).tobytes())

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not sentences:
                sentences = [text]

            for idx, sentence in enumerate(sentences):
                sentence_chunks = []
                synth_kwargs = {
                    "length_scale": TTS_LENGTH_SCALE,
                    "noise_scale": TTS_NOISE_SCALE,
                    "noise_w": TTS_NOISE_W,
                    "sentence_silence": TTS_SENTENCE_SILENCE,
                }
                try:
                    chunk_iter = self._tts_engine.synthesize(sentence, **synth_kwargs)
                except TypeError:
                    # Older piper-tts versions may not support all kwargs
                    chunk_iter = self._tts_engine.synthesize(sentence)

                for chunk in chunk_iter:
                    audio_bytes = chunk.audio_int16_bytes
                    sample_rate = chunk.sample_rate
                    logger.debug(
                        f"Piper chunk: {len(audio_bytes)} bytes at {chunk.sample_rate}Hz"
                    )
                    sentence_chunks.append(audio_bytes)

                if sentence_chunks:
                    audio_chunks.append(b"".join(sentence_chunks))

                # Natural pause between sentences
                if idx < len(sentences) - 1:
                    pause_samples = int(0.10 * sample_rate)  # 100 ms
                    audio_chunks.append(np.zeros(pause_samples, dtype=np.int16).tobytes())

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

            audio_bytes = _apply_vocal_color(audio_array.tobytes(), sample_rate=24000)

            logger.info(f"🔊 TTS generated {len(audio_array)} samples at 24kHz")
            return audio_bytes

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
