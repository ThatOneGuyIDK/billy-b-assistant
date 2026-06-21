import asyncio
import base64
import json
import os
import socket
import time
from typing import Any

import numpy as np

from . import audio
from .base_tools import get_base_tools
from .config import (
    CHUNK_MS,
    DEBUG_MODE,
    DEBUG_MODE_INCLUDE_DELTA,
    INSTRUCTIONS,
    MIC_TIMEOUT_SECONDS,
    REALTIME_AI_PROVIDER,
    RUN_MODE,
    THINKING_SOUND_DELAY_MS,
    SERVER_VAD_PARAMS,
    SILENCE_THRESHOLD,
    TEXT_ONLY_MODE,
    TURN_EAGERNESS,
)
from .logger import logger
from .mic import MicManager
from .movements import move_tail_async, stop_all_motors
from .realtime_ai_provider import voice_provider_registry
from .workout_intent import WorkoutIntentResult, classify_workout_intent


def _get_dynamic_song_description():
    """Get dynamic song description based on available songs."""
    return ""


def get_instructions_with_user_context():
    return INSTRUCTIONS


def get_tools_for_current_mode():
    return get_base_tools()


class BillySession:
    def __init__(
        self,
        interrupt_event=None,
        *,
        conversation_provider=None,
        kickoff_text: str | None = None,
        kickoff_kind: str = "literal",  # "literal" | "prompt" | "raw"
        kickoff_to_interactive: bool = False,  # immediately open-mic after kickoff
        autofollowup: str = "auto",  # "auto" | "never" | "always"
    ):
        self.realtime_ai_provider = (
            conversation_provider
            or voice_provider_registry.get_provider(REALTIME_AI_PROVIDER)
        )
        self.ws = None
        self.ws_lock: asyncio.Lock = asyncio.Lock()
        self.loop = None
        self.audio_buffer = bytearray()
        self.committed = False
        self.first_text = True
        self.full_response_text = ""
        self.last_rms = 0.0
        self.last_activity = [time.time()]
        self.first_audio_received = False  # Track if we've heard any audio yet
        self.session_active = asyncio.Event()
        self.user_spoke_after_assistant = False
        self.allow_mic_input = True
        self.interrupt_event = interrupt_event or asyncio.Event()
        self.mic = MicManager()
        self.mic_running = False
        self.mic_timeout_task: asyncio.Task | None = None
        self.stop_thinking_sounds = False  # Flag to stop enqueueing thinking sounds
        self.thinking_sound_task: asyncio.Task | None = None

        # Track whenever a session is updated after creation, and OpenAI is ready to receive voice.
        self.session_initialized = False
        self.run_mode = RUN_MODE

        # Optional first line (e.g. remote trigger) before normal mic turns
        self.kickoff_text = (kickoff_text or "").strip() or None
        self.kickoff_kind = kickoff_kind
        self.kickoff_to_interactive = kickoff_to_interactive
        self.kickoff_first_turn_done = False

        # Follow-up
        self.autofollowup = autofollowup  # "auto" | "never" | "always"
        self.follow_up_expected = False
        self.follow_up_prompt: str | None = None

        # Tool args buffer (for streamed args)
        self._tool_args_buffer: dict[str, str] = {}

        # Turn-level flags for follow-up detection
        self._saw_transcript_delta = False
        self._turn_had_speech = False
        self._active_transcript_stream: str | None = None  # "audio" | "text"

        # Flag for handling "I am not X" scenarios
        self._waiting_for_name_after_denial = False
        self._added_done_text = False

        # Flags for logging mic state (reset for each session)
        self._mic_data_started = False
        self._logged_mic_blocked_1 = False
        self._logged_waiting_for_wakeup = False

    def _interrupted(self) -> bool:
        try:
            return self.interrupt_event.is_set()
        except Exception:
            return False

    async def _wait_for_playback_ready(self) -> None:
        """Wait for wake-up audio without blocking session interrupt."""
        if TEXT_ONLY_MODE or audio.playback_done_event.is_set():
            return
        while not audio.playback_done_event.is_set():
            if not self.session_active.is_set() or self._interrupted():
                return
            await asyncio.sleep(0.05)

    # ---- Websocket helpers ---------------------------------------------
    async def _ws_send_json(self, payload: dict[str, Any]):
        """Send a JSON payload over the session websocket with locking.

        This method is a small convenience to avoid repeating the lock and
        json.dumps boilerplate across the codebase.
        """
        async with self.ws_lock:
            if self.ws is not None:
                await self.realtime_ai_provider.send_message(self.ws, payload)

    # ---- Message type constants ----------------------------------------
    AUDIO_OUT_TYPES = {
        "response.output_audio",
        "response.output_audio.delta",
    }
    TRANSCRIPT_DELTA_TYPES = {
        "response.output_audio_transcript.delta",
        "response.audio_transcript.delta",
        "response.text.delta",
    }
    TRANSCRIPT_DONE_TYPES = {
        "response.output_audio_transcript.done",
        "response.audio_transcript.done",
        "response.text.done",
    }

    # ---- Private handlers -----------------------------------------------
    def _on_response_created(self):
        self._saw_transcript_delta = False
        self._turn_had_speech = False
        self.follow_up_expected = False
        self.follow_up_prompt = None
        self._active_transcript_stream = None
        self._added_done_text = False
        self._saw_follow_up_call = False

    def _on_input_speech_started(self):
        self.committed = False

    def _on_transcript_done(self, data: dict[str, Any]):
        transcript = data.get("transcript") or data.get("text") or ""
        if transcript and not self._saw_transcript_delta and not self._added_done_text:
            self.full_response_text += transcript
            self._added_done_text = True
        self.full_response_text += "\n\n"
        if DEBUG_MODE:
            logger.info(f"Transcript completed: {transcript!r}", "📝")

    def _on_audio_out(self, data: dict[str, Any]):
        # Stop enqueueing more thinking sounds
        self._stop_thinking_sound()
        self._turn_had_speech = True
        
        if TEXT_ONLY_MODE:
            return
        
        audio_b64 = data.get("audio") or data.get("delta")
        if audio_b64:
            audio_chunk = base64.b64decode(audio_b64)
            
            # Track audio in buffer
            self.audio_buffer.extend(audio_chunk)
            self.last_activity[0] = time.time()
            
            # On first chunk, clear thinking sounds then start response playback
            if len(self.audio_buffer) == len(audio_chunk):  # First chunk only
                logger.debug("First audio chunk - clearing thinking sounds from queue")
                cleared = 0
                while not audio.playback_queue.empty():
                    try:
                        audio.playback_queue.get_nowait()
                        audio.playback_queue.task_done()
                        cleared += 1
                    except Exception:
                        break
                logger.debug(f"Cleared {cleared} thinking sound chunks")
            
            # Always enqueue response audio
            audio.playback_queue.put(audio_chunk)

            if self.interrupt_event.is_set():
                logger.warning(
                    "Assistant turn interrupted. Stopping response playback.", "⛔"
                )
                while not audio.playback_queue.empty():
                    try:
                        audio.playback_queue.get_nowait()
                        audio.playback_queue.task_done()
                    except Exception:
                        break
                self.session_active.clear()
                self.interrupt_event.clear()

    def _on_transcript_delta(self, t: str, data: dict[str, Any]):
        # Choose a single transcript stream per turn to avoid duplicates
        if t.startswith("response.output_audio_transcript") or t.startswith(
            "response.audio_transcript"
        ):
            stream = "audio"
        else:
            stream = "text"
        if self._active_transcript_stream is None:
            self._active_transcript_stream = stream
        elif stream != self._active_transcript_stream:
            return
        self._turn_had_speech = True
        self._saw_transcript_delta = True
        self.allow_mic_input = False
        if self.first_text:
            logger.info("Billy: ", "🐟")
            self.first_text = False
            self.user_spoke_after_assistant = False
        # Don't log individual deltas - they're too verbose
        # Just print to console for real-time display
        # print(data.get("delta", ""), end="", flush=True)
        self.full_response_text += data.get("delta", "")

    def _on_tool_args_delta(self, data: dict[str, Any]):
        name = data.get("name")
        if name:
            self._tool_args_buffer.setdefault(name, "")
            self._tool_args_buffer[name] += data.get("arguments", "")

    def _parse_json_args(self, raw_args: str | None, tool_name: str) -> dict:
        """Parse JSON arguments with fallback for malformed JSON."""
        raw_args = raw_args or "{}"
        try:
            return json.loads(raw_args)
        except Exception as e:
            # Try to fix common JSON issues
            try:
                import re

                fixed_json = raw_args
                # Fix malformed JSON where colon appears inside quoted key: {"key:value"} -> {"key": "value"}
                fixed_json = re.sub(
                    r'{"([^"]*):([^"}]*)([}])', r'{"\1": \2\3', fixed_json
                )
                # Handle boolean values
                fixed_json = re.sub(r':(true|false)([},])', r': \1\2', fixed_json)
                args = json.loads(fixed_json)
                logger.info(
                    f"{tool_name}: fixed malformed JSON | original={raw_args!r} | fixed={fixed_json!r}",
                    "🔧",
                )
                return args
            except Exception as fix_e:
                logger.warning(
                    f"{tool_name}: failed to parse arguments: {e} | raw={raw_args!r} | fix also failed: {fix_e}"
                )
                return {}

    async def _handle_play_song(self, raw_args: str | None):
        args = self._parse_json_args(raw_args, "play_song")
        song_name = args.get("song")
        if song_name:
            logger.info(f"Assistant requested to play song: {song_name}", "🎵")
            await self.stop_session()
            await asyncio.sleep(1.0)
            await audio.play_song(song_name, interrupt_event=self.interrupt_event)

    async def _run_workout_automation(self, result: WorkoutIntentResult):
        """Speak a timer or set-counter script without calling the LLM."""
        count = max(1, int(result.target_count or 60))
        spoken_sequence = result.spoken_sequence or [str(count)]
        intro = "Set counter." if result.action == "set_counter" else "Timer."
        script = ". ".join([intro, *spoken_sequence])

        self._stop_thinking_sound()
        while not audio.playback_queue.empty():
            try:
                audio.playback_queue.get_nowait()
                audio.playback_queue.task_done()
            except Exception:
                break

        try:
            audio_bytes = await self.realtime_ai_provider.generate_audio_clip(script)
            if not audio_bytes:
                return
            chunk_size = 9600
            for start in range(0, len(audio_bytes), chunk_size):
                if self._interrupted():
                    audio.stop_playback()
                    return
                audio.playback_queue.put(audio_bytes[start : start + chunk_size])
            await asyncio.to_thread(audio.playback_queue.join)
        except Exception as e:
            logger.warning(f"Workout automation playback failed: {e}", "⚠️")

    async def _handle_song_intent(self, workout_intent):
        song_name = workout_intent.song_name or workout_intent.metadata.get(
            "song_name"
        )
        if not song_name:
            logger.warning("Song request detected but no playable song was selected.")
            return

        logger.info(f"Song request routed locally: {song_name}", "🎵")
        await self.stop_session()
        if self._interrupted():
            return
        await asyncio.sleep(1.0)
        if self._interrupted():
            return
        await audio.play_song(song_name, interrupt_event=self.interrupt_event)

    async def _on_tool_args_done(self, data: dict[str, Any]):
        name = data.get("name")
        raw_args = data.get("arguments")
        if not raw_args and name:
            raw_args = self._tool_args_buffer.pop(name, "{}")

        if name == "play_song":
            await self._handle_play_song(raw_args)
            return
        if name and DEBUG_MODE:
            logger.verbose(f"Ignoring unknown tool call: {name}", "🔧")

    async def _on_response_done(self, data: dict[str, Any]):
        self._stop_thinking_sound()
        error = data.get("status_details", {}).get("error")
        if error:
            error_type = error.get("type")
            error_message = error.get("message", "Unknown error")
            logger.error(f"Assistant error [{error_type}]: {error_message}")
        else:
            logger.success("Assistant response complete.", "✿")

        if not TEXT_ONLY_MODE:
            await asyncio.to_thread(audio.playback_queue.join)
            await asyncio.sleep(0.3)
            if len(self.audio_buffer) > 0:
                logger.verbose(
                    f"Saving audio buffer ({len(self.audio_buffer)} bytes)", "💾"
                )
                audio.rotate_and_save_response_audio(self.audio_buffer)
            else:
                logger.warning("Audio buffer was empty, skipping save.")
            self.audio_buffer.clear()
            audio.playback_done_event.set()
            self.last_activity[0] = time.time()
            self.allow_mic_input = True

        # Kickoff follow-up switch
        if self.kickoff_text and not self.kickoff_first_turn_done:
            if self._turn_had_speech:
                self.kickoff_first_turn_done = True
                if self.kickoff_to_interactive:
                    print("🔁 Kickoff complete — switching to interactive mode.")
                    self._start_mic()
                elif self.autofollowup == "auto":
                    asked_question = self._wants_follow_up_heuristic()
                    wants_follow_up = self.follow_up_expected or asked_question
                    if wants_follow_up:
                        print("🔁 Auto follow-up detected — opening mic.")
                        await self._start_mic_after_playback()
                        self.user_spoke_after_assistant = False
                        self.last_activity[0] = time.time()
                    else:
                        # No follow-up needed, close the session
                        print(
                            "🔁 Kickoff complete — no follow-up needed. Closing session."
                        )
                        await self.stop_session()
                        return
            else:
                if DEBUG_MODE:
                    logger.info(
                        "Kickoff turn ended with no speech (tool-only). Waiting for next turn.",
                        "ℹ️",
                    )

        if self.run_mode == "dory":
            logger.info("Dory mode active. Ending session after single response.", "🎣")
            await self.stop_session()

    async def _thinking_sound_loop(self):
        """Continuously enqueue short thinking tones until response arrives."""
        try:
            # Give fast responses a chance to start before any bubble is heard.
            await asyncio.sleep(max(0, THINKING_SOUND_DELAY_MS) / 1000.0)
            while (
                not self.stop_thinking_sounds
                and self.session_active.is_set()
                and not self.interrupt_event.is_set()
            ):
                # Keep a small buffer of thinking tones, avoid queue buildup.
                if audio.playback_queue.qsize() < 6:
                    audio.enqueue_thinking_tone(duration_ms=180, frequency_hz=700.0)
                await asyncio.sleep(0.28)
        except asyncio.CancelledError:
            pass

    def _start_thinking_sound(self):
        """Start adaptive thinking sound loop while waiting for response."""
        if TEXT_ONLY_MODE:
            return
        self.stop_thinking_sounds = False
        if self.thinking_sound_task and not self.thinking_sound_task.done():
            return
        self.thinking_sound_task = asyncio.create_task(self._thinking_sound_loop())

    def _stop_thinking_sound(self):
        """Stop enqueueing more thinking sounds (response is arriving)."""
        self.stop_thinking_sounds = True
        if self.thinking_sound_task and not self.thinking_sound_task.done():
            self.thinking_sound_task.cancel()
        self.thinking_sound_task = None

    # ---- Mic helpers -------------------------------------------------
    def _start_mic(self, *, retry=True):
        """
        Try to open the mic. If it fails (device busy/unavailable), optionally
        start a background retry loop with exponential backoff.
        """
        if self.mic_running or not self.session_active.is_set():
            return

        try:
            # Recreate the manager in case the previous stream left it in a bad state
            if self.mic is None:
                self.mic = MicManager()

            self.mic.start(self.mic_callback)
            self.mic_running = True
            if DEBUG_MODE:
                logger.info("Mic started", "🎤")
            if not self.mic_timeout_task or self.mic_timeout_task.done():
                self.mic_timeout_task = asyncio.create_task(self.mic_timeout_checker())

        except Exception as e:
            self.mic_running = False
            logger.error(f"Mic start failed: {e}")
            if retry and self.session_active.is_set():
                # Kick off a retry loop (non-blocking)
                asyncio.create_task(self._retry_mic_loop())

    def _stop_mic(self):
        if self.mic_running:
            try:
                self.mic.stop()
                # Small delay to ensure ALSA releases the device
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error while stopping mic: {e}")
            self.mic_running = False

    async def _retry_mic_loop(self):
        """
        Retry opening the mic a few times with backoff. Keeps the session alive
        while we wait for the input device to become available again.
        """
        if DEBUG_MODE:
            logger.verbose("Mic retry loop started", "🔁")

        # Small, bounded backoff: 0.5s → 1s → 2s → 2s → …
        delays = [0.5, 1.0, 2.0, 2.0, 2.0]
        for delay in delays:
            if not self.session_active.is_set():
                return

            await asyncio.sleep(delay)

            # Recreate MicManager to clear any stale PortAudio handles
            try:
                self.mic = MicManager()
            except Exception as e:
                logger.warning(f"MicManager recreate failed: {e}")

            try:
                self.mic.start(self.mic_callback)
                self.mic_running = True
                logger.info("Mic started after retry", "✅")
                if not self.mic_timeout_task or self.mic_timeout_task.done():
                    self.mic_timeout_task = asyncio.create_task(
                        self.mic_timeout_checker()
                    )
                return
            except Exception as e:
                self.mic_running = False
                logger.warning(f"Mic retry failed: {e}")

                # Try audio system reset for device unavailable errors
                if "Device unavailable" in str(e):
                    logger.info("Attempting audio system reset in retry loop...", "🔄")
                    try:
                        import subprocess

                        import sounddevice as sd

                        # Reset PortAudio system
                        sd._terminate()
                        await asyncio.sleep(0.5)
                        sd._initialize()

                        # Reset ALSA mixer
                        subprocess.run(
                            ["sudo", "alsactl", "restore"],
                            capture_output=True,
                            timeout=5,
                        )

                        # Kill any processes that might be using the audio device
                        subprocess.run(
                            ["sudo", "fuser", "-k", "/dev/snd/*"],
                            capture_output=True,
                            timeout=3,
                        )

                        await asyncio.sleep(2.0)
                        logger.info("Audio system reset completed in retry loop", "✅")
                    except Exception as reset_error:
                        logger.warning(
                            f"Audio reset failed in retry loop: {reset_error}"
                        )

        # All retries exhausted
        logger.error(
            "Mic unavailable after retries; keeping session but not listening."
        )

    # ------------------------------------------------------------------

    def _wants_follow_up_heuristic(self) -> bool:
        """
        Minimal, language-agnostic check: treat any question punctuation
        as an invitation to follow up.
        """
        txt = (self.full_response_text or "").strip()
        # Latin '?', Spanish '¿', CJK full-width '？', Arabic '؟', interrobang '‽'
        has_question = any(ch in txt for ch in ("?", "¿", "？", "؟", "‽"))
        logger.info(
            f"Heuristic check: text='{txt}' | has_question={has_question}", "🔍"
        )
        return has_question

    async def _start_mic_after_playback(
        self, delay: float = 0.6, retries: int = 3
    ) -> bool:
        """
        Open the mic a tad later (and retry) so ALSA has released devices.
        """
        for attempt in range(1, retries + 1):
            try:
                # Progressive delay: longer waits for later attempts
                if attempt > 1:
                    wait_time = delay * (attempt - 1) + 0.5
                    logger.info(
                        f"Waiting {wait_time:.1f}s before mic retry {attempt}...", "⏳"
                    )
                    await asyncio.sleep(wait_time)

                # Ensure mic is fully stopped before retry
                if self.mic_running:
                    self.mic.stop()
                    self.mic_running = False
                    await asyncio.sleep(0.2)  # Brief pause after stop

                if not self.mic_running:
                    self.mic.start(self.mic_callback)  # may raise
                    self.mic_running = True
                    if not self.mic_timeout_task or self.mic_timeout_task.done():
                        self.mic_timeout_task = asyncio.create_task(
                            self.mic_timeout_checker()
                        )
                print(f"🎙️ Mic opened (attempt {attempt}).")
                return True
            except Exception as e:
                logger.warning(f"Mic open failed (attempt {attempt}/{retries}): {e}")
                # For ALSA device unavailable errors, try to reset audio system
                if "Device unavailable" in str(e) and attempt < retries:
                    logger.info("Attempting audio system reset...", "🔄")
                    try:
                        import subprocess

                        import sounddevice as sd

                        # Reset PortAudio system
                        sd._terminate()
                        await asyncio.sleep(0.5)
                        sd._initialize()

                        # Reset ALSA mixer
                        subprocess.run(
                            ["sudo", "alsactl", "restore"],
                            capture_output=True,
                            timeout=5,
                        )

                        # Kill any processes that might be using the audio device
                        subprocess.run(
                            ["sudo", "fuser", "-k", "/dev/snd/*"],
                            capture_output=True,
                            timeout=3,
                        )

                        await asyncio.sleep(2.0)
                        logger.info("Audio system reset completed", "✅")
                    except Exception as reset_error:
                        logger.warning(f"Audio reset failed: {reset_error}")

        logger.error("Mic failed to open after retries.")
        return False

    async def start(self):
        self.loop = asyncio.get_running_loop()
        logger.info("Session starting...", "⏱️")

        # Debug VAD parameters
        vad_params = SERVER_VAD_PARAMS[TURN_EAGERNESS]
        logger.info(f"🔧 VAD Parameters (eagerness={TURN_EAGERNESS}): {vad_params}")
        logger.info(
            f"🔧 Audio Config: SILENCE_THRESHOLD={SILENCE_THRESHOLD}, MIC_TIMEOUT_SECONDS={MIC_TIMEOUT_SECONDS}"
        )

        # Clear all session state
        self.audio_buffer.clear()
        self.committed = False
        self.first_text = True
        self.full_response_text = ""
        self.last_activity[0] = time.time()
        self.session_active.set()
        self.user_spoke_after_assistant = False
        self.allow_mic_input = True

        # Ensure mic logging flag is reset (should already be False from __init__)
        self._mic_data_started = False

        # Debug: Log all mic-blocking conditions at session start
        logger.info(
            f"🔧 Mic state check: allow_mic_input={self.allow_mic_input}, "
            f"session_active={self.session_active.is_set()}, "
            f"playback_done_event={'SET' if audio.playback_done_event.is_set() else 'CLEAR (waiting for wake-up)'}, "
            f"TEXT_ONLY_MODE={TEXT_ONLY_MODE}",
            "🔧",
        )

        async with self.ws_lock:
            if self.ws is None:
                try:
                    self.ws = await self.realtime_ai_provider.connect(
                        instructions=get_instructions_with_user_context(),
                        tools=get_tools_for_current_mode(),
                        server_vad_params=SERVER_VAD_PARAMS[TURN_EAGERNESS],
                        text_only_mode=TEXT_ONLY_MODE,
                        voice="alloy",  # Default voice for local mode
                    )

                    # Optional scripted first utterance
                    if self.kickoff_text:
                        if self.kickoff_kind == "prompt":
                            kickoff_payload = self.kickoff_text
                        elif self.kickoff_kind == "literal":
                            kickoff_payload = (
                                "Say the user's message **verbatim**, word for word, with no additions or reinterpretation.\n"
                                "Maintain personality, but do NOT rephrase or expand.\n\n"
                                f"Repeat this literal message: {self.kickoff_text}"
                            )
                        else:
                            kickoff_payload = self.kickoff_text

                        await self.realtime_ai_provider.send_message(
                            self.ws,
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": kickoff_payload}
                                    ],
                                },
                            },
                        )
                        await self.realtime_ai_provider.send_message(
                            self.ws, {"type": "response.create"}
                        )

                except Exception as e:
                    err = str(e).lower()
                    if "invalid_api_key" in err or "api key" in err:
                        await self._play_error_sound("noapikey", str(e))
                    elif isinstance(e, socket.gaierror) or "network" in err or "dns" in err:
                        await self._play_error_sound(
                            "nowifi", "Network unreachable or DNS failed"
                        )
                    else:
                        await self._play_error_sound("error", str(e))
                    return

        if not TEXT_ONLY_MODE:
            audio.ensure_playback_worker_started(CHUNK_MS)

        await self.run_stream()

    def mic_callback(self, indata, *_):
        # Check 1: Mic input allowed and session active
        if not self.allow_mic_input or not self.session_active.is_set():
            if not hasattr(self, '_logged_mic_blocked_1'):
                logger.warning(
                    f"🔇 Mic blocked: allow_mic_input={self.allow_mic_input}, "
                    f"session_active={self.session_active.is_set()}",
                    "⚠️",
                )
                self._logged_mic_blocked_1 = True
            return

        # Check 2: Wait for wake-up sound to finish
        if not TEXT_ONLY_MODE and not audio.playback_done_event.is_set():
            if not hasattr(self, '_logged_waiting_for_wakeup'):
                logger.info("🔇 Mic waiting for wake-up sound to finish...", "⏳")
                self._logged_waiting_for_wakeup = True
            return

        # Log once when mic data starts being sent after wake-up sound
        if not self._mic_data_started and not TEXT_ONLY_MODE:
            logger.info("Mic data now being sent (wake-up sound finished)", "🎤")
            self._mic_data_started = True

        # Handle both mono (1D) and stereo (2D) input
        if indata.ndim == 1:
            samples = indata
        else:
            samples = indata[:, 0]
        
        rms = np.sqrt(np.mean(np.square(samples.astype(np.float32))))
        self.last_rms = rms

        if DEBUG_MODE:
            print(f"\r🎙️ Mic Volume: {rms:.1f}", end="", flush=True)

        if rms > SILENCE_THRESHOLD:
            if not self.first_audio_received:
                self.first_audio_received = True
                logger.info("Audio detected! Listening...")
            self.last_activity[0] = time.time()
            self.user_spoke_after_assistant = True

        audio.send_mic_audio(self.ws, samples, self.loop)

    async def run_stream(self):
        await self._wait_for_playback_ready()
        if not self.session_active.is_set() or self._interrupted():
            return

        logger.info(
            "Mic stream active. Say something..."
            if not self.kickoff_text
            else "Announcing kickoff...",
            "🎙️" if not self.kickoff_text else "📣",
        )

        try:
            # Start mic immediately only for non-kickoff sessions
            if not self.kickoff_text:
                self._start_mic()

            assert self.ws is not None
            async for message in self.ws:
                if not self.session_active.is_set() or self._interrupted():
                    print("🚪 Session marked as inactive, stopping stream loop.")
                    print()  # Add newline to end the mic volume display line
                    break
                data = json.loads(message)
                if DEBUG_MODE and (
                    DEBUG_MODE_INCLUDE_DELTA
                    or not (data.get("type") or "").endswith("delta")
                ):
                    logger.verbose(f"Raw message: {data}", "🔁")

                if data.get("type") in ("session.updated", "session_updated"):
                    self.session_initialized = True

                await self.handle_message(data)

        except Exception as e:
            logger.error(f"Error opening mic input: {e}")
            self.session_active.clear()

        finally:
            try:
                self._stop_thinking_sound()
                self._stop_mic()
                logger.info("Mic stream closed.", "🎙️")
            except Exception as e:
                logger.warning(f"Error while stopping mic: {e}")

            try:
                await self.post_response_handling()
            except Exception as e:
                logger.warning(f"Error in post_response_handling: {e}")

    async def handle_message(self, data):
        t = data.get("type") or ""

        if t == "response.created":
            self._on_response_created()
            return
        if t == "input_audio_buffer.speech_started":
            self._on_input_speech_started()
            return
        if t == "input_audio_buffer.speech_stopped":
            return
        if t == "conversation.item.input_audio_transcription.completed":
            # User's speech has been transcribed, now generate LLM response
            transcript = data.get("transcript", "")
            logger.info(f"📝 User said: {transcript}", "📝")

            route_action = data.get("route_action") or ""
            if route_action == "song":
                workout_intent = WorkoutIntentResult(
                    original_text=transcript,
                    normalized_text=f"Song request: {data.get('song_name') or transcript}",
                    action="song",
                    confidence="high",
                    song_name=data.get("song_name"),
                    metadata={"song_name": data.get("song_name"), "route_action": "song"},
                )
                await self._handle_song_intent(workout_intent)
                return

            workout_intent = classify_workout_intent(transcript)
            if workout_intent.action in {"timer", "set_counter"}:
                logger.info(
                    f"Workout automation handled locally ({workout_intent.action}); skipping LLM turn.",
                    "🏋️",
                )
                await self._run_workout_automation(workout_intent)
                return
            if workout_intent.action == "song":
                await self._handle_song_intent(workout_intent)
                return
            # Thinking sound already started earlier
            await self._ws_send_json({"type": "response.create"})
            return
        if t == "response.thinking_started":
            # Start thinking sound after a short delay so fast responses skip it.
            self._start_thinking_sound()
            return
        if t in self.TRANSCRIPT_DONE_TYPES:
            self._on_transcript_done(data)
            return
        if t in self.AUDIO_OUT_TYPES:
            self._on_audio_out(data)
            return
        if t == "input_audio_buffer.committed":
            self.committed = True
            return
        if t in self.TRANSCRIPT_DELTA_TYPES and "delta" in data:
            self._on_transcript_delta(t, data)
            return
        if t == "response.function_call_arguments.delta":
            self._on_tool_args_delta(data)
            return
        if t == "response.function_call_arguments.done":
            await self._on_tool_args_done(data)
            return
        if t == "response.done":
            await self._on_response_done(data)
            return
        if t == "error":
            error: dict[str, Any] = data.get("error") or {}
            code = error.get("code", "error").lower()
            message = error.get("message", "Unknown error")
            code = "noapikey" if "invalid_api_key" in code else "error"
            logger.error(f"API Error ({code}): {message}")
            await self._play_error_sound(code, message)
            return
        # else: ignore unrecognized messages silently

    async def mic_timeout_checker(self):
        logger.info("Mic timeout checker active", "🛡️")
        last_tail_move = 0

        while self.session_active.is_set() and not self._interrupted():
            if not self.mic_running:
                await asyncio.sleep(0.2)
                continue

            # Only count timeout after we've received audio
            if not self.first_audio_received:
                await asyncio.sleep(0.5)
                continue

            now = time.time()
            idle_seconds = now - max(self.last_activity[0], audio.last_played_time)
            timeout_offset = 2

            if idle_seconds - timeout_offset > 0.5:
                elapsed = idle_seconds - timeout_offset
                progress = min(elapsed / MIC_TIMEOUT_SECONDS, 1.0)
                bar_len = 20
                filled = int(bar_len * progress)
                bar = "█" * filled + "-" * (bar_len - filled)
                print(
                    f"\r👂 {MIC_TIMEOUT_SECONDS}s timeout: [{bar}] {elapsed:.1f}s "
                    f"| Mic Volume:: {self.last_rms:.4f} / Threshold: {SILENCE_THRESHOLD:.4f}",
                    end="",
                    flush=True,
                )

                # Disabled tail movement during timeout to avoid noise feedback loop
                # if now - last_tail_move > 1.0:
                #     move_tail_async(duration=0.2)
                #     last_tail_move = now

                if elapsed > MIC_TIMEOUT_SECONDS:
                    logger.info(
                        f"No mic activity for {MIC_TIMEOUT_SECONDS}s. Ending input...",
                        "⏱️",
                    )
                    # Move tail now that timeout triggered
                    move_tail_async(duration=0.2)
                    # Commit captured audio, then pause mic and wait for assistant response.
                    if not self.committed:
                        await self._ws_send_json({"type": "input_audio_buffer.commit"})
                        self.committed = True
                    self._stop_mic()
                    logger.info("Waiting for assistant response...", "⏳")
                    break

            await asyncio.sleep(0.5)

    async def post_response_handling(self):
        if self.full_response_text.strip():
            print(f"📝 Transcript completed: \"{self.full_response_text.strip()}\"")
        logger.verbose(f"Full response: {self.full_response_text.strip()}", "🧠")

        if not self.session_active.is_set() or self._interrupted():
            print()  # Add newline to end the mic volume display line
            logger.info(
                "Session inactive after timeout or interruption. Not restarting.", "🚪"
            )
            stop_all_motors()
            async with self.ws_lock:
                if self.ws:
                    await self.ws.close()
                    await self.ws.wait_closed()
                    self.ws = None
            return

        # Heuristic fallback (punctuation only)
        asked_question = self._wants_follow_up_heuristic()

        # Always log follow-up decision for debugging
        logger.info(
            f"Follow-up decision | mode={self.autofollowup}"
            f" | tool_expects={self.follow_up_expected}"
            f" | qmark={asked_question}"
            f" | had_speech={self._turn_had_speech}"
            f" | saw_follow_up_call={self._saw_follow_up_call}",
            "🧪",
        )

        if self.autofollowup == "always":
            wants_follow_up = True
        elif self.autofollowup == "never":
            wants_follow_up = False
        else:
            wants_follow_up = self.follow_up_expected or asked_question

        if not self._saw_follow_up_call and DEBUG_MODE:
            logger.verbose(
                "No follow-up hint from model this turn; using heuristic instead.",
                "🧭",
            )

        if wants_follow_up:
            logger.info("Follow-up expected. Keeping session open.", "🔁")
            await self._start_mic_after_playback()  # <-- changed
            self.user_spoke_after_assistant = False
            self.full_response_text = ""
            self.last_activity[0] = time.time()
            return

        logger.info("No follow-up. Ending session.", "🛑")
        stop_all_motors()
        async with self.ws_lock:
            if self.ws:
                await self.ws.close()
                await self.ws.wait_closed()
                self.ws = None

    async def stop_session(self):
        logger.info("Stopping session...", "🛑")
        self._stop_thinking_sound()

        self.session_active.clear()
        self._stop_mic()

        if self.mic_timeout_task and not self.mic_timeout_task.done():
            self.mic_timeout_task.cancel()
            self.mic_timeout_task = None

        async with self.ws_lock:
            if self.ws:
                try:
                    await self.ws.close()
                    # Add timeout to prevent hanging
                    try:
                        await asyncio.wait_for(self.ws.wait_closed(), timeout=2.0)
                    except asyncio.TimeoutError:
                        logger.warning("Session close timeout, forcing cleanup")
                except Exception as e:
                    logger.warning(f"Error closing provider session: {e}")
                finally:
                    self.ws = None

        if self.loop:
            current = asyncio.current_task(self.loop)
            for task in asyncio.all_tasks(self.loop):
                if task is current:
                    continue
                task.cancel()

    async def request_stop(self):
        logger.info("Stop requested via external signal.", "🛑")
        self.session_active.clear()

    async def _play_error_sound(self, code: str = "error", message: str | None = None):
        """
        Play an error sound based on the provided code.
        Example:
          - "error"     → sounds/error.wav
          - "nowifi"    → sounds/nowifi.wav
          - "noapikey"  → sounds/noapikey.wav
        """
        stop_all_motors()

        filename = f"{code}.wav"
        sound_path = os.path.join("sounds", filename)

        logger.error(f"Error ({code}): {message or 'No message'}")
        logger.info(f"Attempting to play {filename}...", "🔊")

        if os.path.exists(sound_path):
            await asyncio.to_thread(audio.enqueue_wav_to_playback, sound_path)
            await asyncio.to_thread(audio.playback_queue.join)
        else:
            logger.warning(f"{sound_path} not found, skipping audio playback.")

        await self.stop_session()
