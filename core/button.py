import asyncio
import contextlib
import threading
import time
from concurrent.futures import CancelledError

from . import audio, config
from .logger import logger
from .movements import move_head, move_tail


try:
    from gpiozero import Button

    gpiozero_available = True
except ImportError:
    gpiozero_available = False

if config.MOCKFISH or not gpiozero_available:
    # Mock button in mockfish mode or if gpiozero not available
    class MockButton:
        def __init__(self, pin, pull_up=True):
            self.pin = pin
            self.when_pressed = None
            self.is_pressed = False
            if config.MOCKFISH:
                logger.info(f"Mockfish: Button on pin {pin} mocked", "🐟")
            elif not gpiozero_available:
                logger.info(f"gpiozero not available: Button on pin {pin} mocked", "🐟")
                # Start thread to listen for Enter key using pynput
                import threading

                threading.Thread(target=self._listen, daemon=True).start()

        def _listen(self):
            # Use fallback input() method for mock button
            self._fallback_listen()

        def _fallback_listen(self):
            import sys

            if not sys.stdin.isatty():
                logger.warning(
                    "stdin is not a tty, mock button input not available", "⚠️"
                )
                return
            while True:
                try:
                    user_input = input("Press Enter to simulate button press: ")
                    if user_input == "" and self.when_pressed:
                        self.when_pressed()
                except (EOFError, KeyboardInterrupt):
                    break

        def close(self):
            pass

    Button = MockButton
from .movements import move_head
from .session import BillySession


# Button and session globals
is_active = False
session_thread = None
interrupt_event = threading.Event()
session_instance: BillySession | None = None
last_button_time = 0
button_debounce_delay = 0.5  # seconds debounce
_session_start_lock = threading.Lock()  # Lock to prevent concurrent session starts

# Setup hardware button
button = Button(config.BUTTON_PIN, pull_up=True)


def is_billy_speaking():
    """Return True if Billy is playing audio (wake-up or response)."""
    if not audio.playback_done_event.is_set():
        return True
    return bool(not audio.playback_queue.empty())


def on_button():
    global \
        is_active, \
        session_thread, \
        interrupt_event, \
        session_instance, \
        last_button_time

    now = time.time()
    if now - last_button_time < button_debounce_delay:
        return  # Ignore very quick repeat presses (debounce)
    last_button_time = now

    if is_active:
        logger.info("Button pressed during active session.", "🔁")
        interrupt_event.set()
        audio.stop_playback()

        if session_instance:
            try:
                logger.info("Stopping active session...", "🛑")
                # A concurrent.futures.CancelledError is expected here, because the last
                # thing that BillySession.stop_session does is `await asyncio.sleep`,
                # and that will raise CancelledError because it's a logical place to
                # stop.
                with contextlib.suppress(CancelledError):
                    future = asyncio.run_coroutine_threadsafe(
                        session_instance.stop_session(), session_instance.loop
                    )
                    # Add timeout to prevent hanging
                    try:
                        future.result(timeout=5.0)  # Wait up to 5 seconds
                        logger.success("Session stopped.")
                    except TimeoutError:
                        logger.warning("Session stop timeout, forcing cleanup")
                        future.cancel()
            except Exception as e:
                logger.warning(f"Error stopping session ({type(e)}): {e}")
            finally:
                # Always ensure cleanup
                session_instance = None
                # Wait for session thread to finish to ensure mic is fully closed
                if session_thread and session_thread.is_alive():
                    logger.info("Waiting for session thread to finish...", "⏳")
                    session_thread.join(timeout=2.0)
                    if session_thread.is_alive():
                        logger.warning("Session thread did not finish in time", "⚠️")
                        # If the session thread is stuck, release start lock so user can start fresh
                        with contextlib.suppress(Exception):
                            if _session_start_lock.locked():
                                _session_start_lock.release()
        is_active = False  # ✅ Ensure this is always set after stopping
        # In mock mode, one Enter should be enough to reset and start listening again.
        # On real hardware, keep existing behavior (press once to stop).
        if not config.MOCKFISH:
            return


    # Use lock to prevent concurrent session starts (but allow interruption above)
    if not _session_start_lock.acquire(blocking=False):
        logger.warning("Session start already in progress, ignoring button press", "⚠️")
        return

    try:
        # Ensure previous session thread is fully finished before starting new one
        if session_thread and session_thread.is_alive():
            logger.warning("Previous session thread still running, waiting...", "⏳")
            session_thread.join(timeout=2.0)
            if session_thread.is_alive():
                logger.error(
                    "Previous session thread did not finish, aborting new session", "❌"
                )
                _session_start_lock.release()
                return

        audio.ensure_playback_worker_started(config.CHUNK_MS)
        
        # Pre-load AI models before playing wake sound to avoid delay
        logger.info("🔧 Pre-loading AI models...", "🔧")
        try:
            from .realtime_ai_provider import voice_provider_registry
            provider = voice_provider_registry.get_provider()
            if hasattr(provider, '_ensure_whisper_loaded'):
                provider._ensure_whisper_loaded()
                logger.success("✅ Whisper model loaded", "🔧")
            if hasattr(provider, '_ensure_tts_loaded'):
                provider._ensure_tts_loaded()
                logger.success("✅ TTS model loaded", "🔧")
        except Exception as e:
            logger.warning(f"⚠️ Model pre-load failed (will retry later): {e}", "🔧")
        
        # Clear the playback done event so session waits for wake-up sound
        audio.playback_done_event.clear()
        logger.info("🔧 playback_done_event cleared (waiting for wake-up sound)", "🔧")
        logger.info("🔔 Playing wake-up sound before opening mic...", "🔔")
        threading.Thread(target=audio.play_random_wake_up_clip, daemon=True).start()
        is_active = True
        interrupt_event = threading.Event()  # Fresh event for each session
        logger.info("Button pressed. Listening...", "🎤")

        def run_session():
            global session_instance, is_active
            try:
                move_head("on")
                session_instance = BillySession(interrupt_event=interrupt_event)
                session_instance.last_activity[0] = time.time()
                asyncio.run(session_instance.start())
            except Exception as e:
                logger.error(f"Session error: {e}")
            finally:
                move_head("off")
                is_active = False
                session_instance = None  # Clear reference
                logger.info("Waiting for button press...", "🕐")
                # Release lock when session finishes
                with contextlib.suppress(Exception):
                    _session_start_lock.release()  # Lock might already be released

        session_thread = threading.Thread(target=run_session, daemon=True)
        session_thread.start()
        # Lock will be released by the session thread when it finishes
    except Exception as e:
        # If anything goes wrong, release the lock
        logger.error(f"Error starting session: {e}")
        with contextlib.suppress(Exception):
            _session_start_lock.release()


def start_loop():
    audio.detect_devices(debug=config.DEBUG_MODE)

    if config.FLAP_ON_BOOT:
        logger.info("Starting Billy startup animation", "🎭")
        move_head("on")
        time.sleep(0.5)
        move_tail(0.3)
        move_tail(0.3)
        move_head("off")
        time.sleep(0.5)
        move_tail(0.3)
        move_tail(0.3)
        logger.info("Billy startup animation complete", "✅")

    button.when_pressed = on_button
    logger.info(
        "Ready. Press button to start a voice session. Press Ctrl+C to quit.", "🎦"
    )
    logger.info("Waiting for button press...", "🕐")
    if config.MOCKFISH:
        logger.info("Mockfish mode: use Enter to simulate button press", "🐟")
        try:
            while True:
                input("Press Enter to simulate button press: ")
                logger.info("🔘 Button press detected (simulated)", "🔘")
                on_button()
        except KeyboardInterrupt:
            pass
    else:
        logger.info("Listening for real GPIO button press on pin 24...", "🔘")
        while True:
            time.sleep(0.1)
