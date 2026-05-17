import atexit
import contextlib
import random
import threading
import time
from threading import Lock, Thread

import numpy as np

from .config import BILLY_PINS, MOCKFISH, is_classic_billy
from .logger import logger


try:
    import lgpio

    lgpio_available = True
except ImportError:
    lgpio_available = False

if MOCKFISH or not lgpio_available:
    # Mock lgpio for development or when not available
    class MockLgpio:
        error = Exception

        @staticmethod
        def gpiochip_open(chip):
            return "mock_handle"

        @staticmethod
        def gpio_claim_output(h, pin):
            pass

        @staticmethod
        def gpio_write(h, pin, value):
            pass

        @staticmethod
        def tx_pwm(h, pin, freq, duty):
            pass

        @staticmethod
        def gpio_free(h, pin):
            pass

        @staticmethod
        def gpiochip_close(h):
            pass

    lgpio = MockLgpio
    if MOCKFISH:
        logger.info("Mockfish: GPIO mocked for development", "🐟")
    elif not lgpio_available:
        logger.info("lgpio not available: GPIO mocked", "🐟")


# === Configuration ===
USE_THIRD_MOTOR = is_classic_billy()
logger.info(f"Using third motor: {USE_THIRD_MOTOR} | Pin profile: {BILLY_PINS}", "⚙️")

# === GPIO Setup ===
h = lgpio.gpiochip_open(0)
FREQ = 10000  # PWM frequency
_gpio_active = True  # Flag to track if GPIO handle is still valid

# -------------------------------------------------------------------
# Pin mapping by profile
# -------------------------------------------------------------------
# We normalize to three "drive" pins (MOUTH, HEAD, TAIL) and up to three
# "mates" that must be held LOW for legacy wiring (GND_1..GND_3).
MOUTH = GND_1 = HEAD = TAIL = GND_2 = GND_3 = None

if BILLY_PINS == "legacy":
    # Original wiring (backwards compatible)
    # Controller 1: IN1=HEAD, IN2=TAIL (2-motor legacy) | IN3=MOUTH, IN4=GND_1
    MOUTH = 12
    HEAD = 13
    TAIL = 6
    GND_1 = 5
    if USE_THIRD_MOTOR:
        # Classic Billy (3 motors): dedicated tail bridge on second driver
        TAIL = 19  # second driver IN1 (PWM)
        GND_2 = 6  # head mate (keep LOW)
        GND_3 = 26  # tail mate (keep LOW)
else:
    # NEW quiet wiring (mates are tied to GND in hardware)
    HEAD = 27  # pin 13
    MOUTH = 22  # pin 15
    TAIL = 17  # pin 11

# Collect all pins we actually use
motor_pins = [p for p in (MOUTH, HEAD, TAIL, GND_1, GND_2, GND_3) if p is not None]

# Claim/initialize
for pin in motor_pins:
    try:
        lgpio.gpio_claim_output(h, pin)
        lgpio.gpio_write(h, pin, 0)
    except lgpio.error as e:
        if "GPIO busy" in str(e) or "busy" in str(e).lower():
            # Pin is already claimed (likely from a previous crashed instance)
            # Try to free it first, then claim it again
            logger.warning(
                f"GPIO pin {pin} is busy, attempting to free and reclaim...", "⚠️"
            )
            try:
                # Try to free the pin (may fail if not claimed by this handle, but worth trying)
                with contextlib.suppress(lgpio.error, Exception):
                    lgpio.gpio_free(h, pin)
                # Wait a bit for the kernel to clean up
                time.sleep(0.2)
                # Now try to claim it again
                lgpio.gpio_claim_output(h, pin)
                lgpio.gpio_write(h, pin, 0)
                logger.info(f"Successfully reclaimed GPIO pin {pin}", "✅")
            except Exception as free_error:
                logger.error(
                    f"Failed to free/reclaim GPIO pin {pin}: {free_error}", "❌"
                )
                raise
        else:
            # Some other GPIO error - re-raise it
            raise

# === State ===
_head_tail_lock = Lock()
_motor_watchdog_running = False
_last_flap = 0
_mouth_open_until = 0
_last_rms = 0
_last_tail_flap = 0.0
_last_head_pulse = 0.0
_sync_smoothed_rms = 0.0
head_out = False

# === PWM tracking (so watchdog can see PWM activity) ===
_pwm = {pin: {"duty": 0, "since": None} for pin in motor_pins}


def set_pwm(pin: int, duty: int):
    """Start/adjust PWM on pin and remember when it went active."""
    global _gpio_active
    if not _gpio_active:
        return  # GPIO handle already closed, skip
    try:
        lgpio.tx_pwm(h, pin, FREQ, int(duty))
    except (lgpio.error, Exception):
        # Handle already closed or invalid - ignore during shutdown
        _gpio_active = False
        return
    if duty > 0:
        _pwm[pin]["duty"] = int(duty)
        _pwm[pin]["since"] = (
            time.time() if _pwm[pin]["since"] is None else _pwm[pin]["since"]
        )
    else:
        _pwm[pin]["duty"] = 0
        _pwm[pin]["since"] = None


def clear_pwm(pin: int):
    """Stop PWM on pin and clear active since timestamp."""
    global _gpio_active
    if not _gpio_active:
        return  # GPIO handle already closed, skip
    try:
        lgpio.tx_pwm(h, pin, FREQ, 0)
    except (lgpio.error, Exception):
        # Handle already closed or invalid - ignore during shutdown
        _gpio_active = False
        return
    _pwm[pin]["duty"] = 0
    _pwm[pin]["since"] = None


# === Motor Helpers ===
def brake_motor(pin1, pin2=None):
    """Actively stop the channel: zero PWM and drive LOW."""
    global _gpio_active
    if not _gpio_active:
        return  # GPIO handle already closed, skip
    clear_pwm(pin1)
    if pin2 is not None:
        clear_pwm(pin2)
        try:
            lgpio.gpio_write(h, pin2, 0)
        except (lgpio.error, Exception):
            _gpio_active = False
            return
    try:
        lgpio.gpio_write(h, pin1, 0)
    except (lgpio.error, Exception):
        _gpio_active = False
        return


def run_motor_async(pwm_pin, low_pin=None, speed_percent=100, duration=0.3, brake=True):
    global _gpio_active
    if not _gpio_active:
        return  # GPIO handle already closed, skip
    if low_pin is not None:
        try:
            lgpio.gpio_write(h, low_pin, 0)
        except (lgpio.error, Exception):
            _gpio_active = False
            return
    set_pwm(pwm_pin, int(speed_percent))
    if brake:
        threading.Timer(duration, lambda: brake_motor(pwm_pin, low_pin)).start()
    else:
        # still auto-close after duration, but just clear PWM (no active brake)
        threading.Timer(duration, lambda: clear_pwm(pwm_pin)).start()


# === Movement Functions (keep signatures/behavior) ===
def move_mouth(speed_percent, duration, brake=False):
    run_motor_async(MOUTH, GND_1, speed_percent, duration, brake)


def stop_mouth():
    brake_motor(MOUTH, GND_1)


def move_head(state="on"):
    global head_out

    def _move_head_on():
        global _gpio_active
        if not _gpio_active:
            return  # GPIO handle already closed, skip
        # Ensure opposite input is LOW if sharing a bridge (2-motor cases)
        # For 3-motor "new" layout, mate is hard GND so this is a no-op.
        if TAIL is not None:
            try:
                lgpio.gpio_write(h, TAIL, 0)
            except (lgpio.error, Exception):
                _gpio_active = False
                return
        set_pwm(HEAD, -80)
        time.sleep(0.5)
        set_pwm(HEAD, -100)  # stay extended

    if state == "on":
        if not head_out:
            threading.Thread(target=_move_head_on, daemon=True).start()
            head_out = True
    else:
        # Brake both sides of shared bridge where relevant
        brake_motor(HEAD, TAIL)
        head_out = False


def move_tail(duration=0.2):
    """
    Tail drive matrix:
      - legacy + classic(3): TAIL has dedicated bridge => mate = GND_3
      - legacy + modern(2):  shared with HEAD => mate = HEAD
      - new    + classic(3): dedicated channel with mate tied to GND => mate = None
      - new    + modern(2):  shared bridge with HEAD => mate = HEAD
    """
    if BILLY_PINS == "legacy":
        if USE_THIRD_MOTOR and TAIL is not None and GND_3 is not None:
            run_motor_async(TAIL, GND_3, speed_percent=80, duration=duration)
        else:
            run_motor_async(TAIL, HEAD, speed_percent=80, duration=duration)
    else:
        if USE_THIRD_MOTOR:
            run_motor_async(TAIL, None, speed_percent=80, duration=duration)
        else:
            run_motor_async(TAIL, HEAD, speed_percent=80, duration=duration)


def move_tail_async(duration=0.3):
    threading.Thread(target=move_tail, args=(duration,), daemon=True).start()


def pulse_head(duration=0.14, speed_percent=80):
    """Short head emphasis — does not latch head_out (unlike move_head('on'))."""
    global _gpio_active

    if not _gpio_active or head_out or _head_tail_lock.locked():
        return

    mate = _mate_for(HEAD)

    def _pulse():
        global _gpio_active
        if not _gpio_active or head_out:
            return
        with _head_tail_lock:
            if head_out or not _gpio_active:
                return
            if mate is not None:
                try:
                    lgpio.gpio_write(h, mate, 0)
                except (lgpio.error, Exception):
                    _gpio_active = False
                    return
            run_motor_async(HEAD, mate, speed_percent=speed_percent, duration=duration)

    threading.Thread(target=_pulse, daemon=True).start()


def reset_motor_sync_state():
    """Clear envelope follower between playback sessions."""
    global _last_rms, _sync_smoothed_rms, _last_flap, _last_tail_flap, _last_head_pulse
    _last_rms = 0.0
    _sync_smoothed_rms = 0.0
    _last_flap = 0.0
    _last_tail_flap = 0.0
    _last_head_pulse = 0.0


def _mouth_duration_scale():
    """Higher MOUTH_ARTICULATION = snappier/shorter flaps (inverse of old behavior)."""
    try:
        from .config import MOUTH_ARTICULATION

        articulation = max(1, min(10, int(MOUTH_ARTICULATION)))
    except Exception:
        articulation = 1
    return max(0.2, 1.0 / articulation)


def _pcm_levels(audio: np.ndarray) -> tuple[float, float]:
    if audio.size == 0:
        return 0.0, 0.0
    samples = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    return rms, peak


def sync_motors_from_pcm_chunk(
    audio,
    *,
    mouth_threshold: int | None = None,
    tail_threshold: int | None = None,
    chunk_ms: int = 40,
    enable_mouth: bool = True,
    enable_tail: bool = True,
    enable_head: bool = True,
):
    """
    Drive mouth, tail, and head from the same PCM chunk as the speakers.
    Tuned for Piper TTS (quieter than song stems).
    """
    global \
        _last_flap, \
        _mouth_open_until, \
        _last_rms, \
        _sync_smoothed_rms, \
        _last_tail_flap, \
        _last_head_pulse

    if audio.size == 0:
        return

    try:
        from .config import (
            MOTOR_SYNC_HEAD,
            MOTOR_SYNC_TAIL,
            MOUTH_FLAP_THRESHOLD,
            TAIL_FLAP_THRESHOLD,
        )
    except Exception:
        MOUTH_FLAP_THRESHOLD = 350
        TAIL_FLAP_THRESHOLD = 900
        MOTOR_SYNC_HEAD = True
        MOTOR_SYNC_TAIL = True

    if mouth_threshold is None:
        mouth_threshold = MOUTH_FLAP_THRESHOLD
    if tail_threshold is None:
        tail_threshold = TAIL_FLAP_THRESHOLD

    if not MOTOR_SYNC_TAIL:
        enable_tail = False
    if not MOTOR_SYNC_HEAD:
        enable_head = False

    now = time.time()
    rms, peak = _pcm_levels(audio)

    # Envelope follower for beat/emphasis detection
    if _sync_smoothed_rms <= 0:
        _sync_smoothed_rms = rms
    else:
        _sync_smoothed_rms = 0.82 * _sync_smoothed_rms + 0.18 * rms
    _last_rms = rms

    quiet = rms < mouth_threshold * 0.45
    if quiet and now >= _mouth_open_until:
        if enable_mouth:
            stop_mouth()

    normalized = np.clip(rms / 32768.0, 0.0, 1.0)
    emphasis = peak / (rms + 1e-5)

    # --- Mouth (lip sync) ---
    if enable_mouth and rms > mouth_threshold and (now - _last_flap) >= 0.07:
        speed = int(np.clip(np.interp(normalized, [0.004, 0.12], [30, 100]), 30, 100))
        duration_ms = float(np.interp(normalized, [0.004, 0.12], [20, min(70, chunk_ms)]))
        duration = (duration_ms / 1000.0) * _mouth_duration_scale()
        duration = max(0.03, min(duration, chunk_ms / 1000.0))

        _last_flap = now
        _mouth_open_until = now + duration
        move_mouth(speed, duration, brake=False)

    # --- Tail (sass / emphasis on louder syllables) ---
    if (
        enable_tail
        and not head_out
        and rms > tail_threshold
        and rms >= _sync_smoothed_rms * 1.08
        and (now - _last_tail_flap) >= 0.28
    ):
        tail_dur = float(np.interp(normalized, [0.01, 0.18], [0.12, 0.28]))
        _last_tail_flap = now
        move_tail_async(duration=tail_dur)

    # --- Head (occasional nod on punchy consonants / peaks) ---
    if (
        enable_head
        and not head_out
        and rms > mouth_threshold * 1.4
        and emphasis > 2.2
        and (now - _last_head_pulse) >= 0.75
        and (now - _last_tail_flap) >= 0.15
    ):
        nod_dur = float(np.interp(emphasis, [2.2, 5.0], [0.10, 0.18]))
        _last_head_pulse = now
        pulse_head(duration=nod_dur, speed_percent=int(np.clip(65 + normalized * 35, 65, 95)))


def flap_from_pcm_chunk(
    audio, threshold=1500, min_flap_gap=0.15, chunk_ms=40, sample_rate=24000
):
    """Mouth-only sync (song vocals path). min_flap_gap/sample_rate kept for API compat."""
    _ = min_flap_gap, sample_rate
    sync_motors_from_pcm_chunk(
        audio,
        mouth_threshold=threshold,
        chunk_ms=chunk_ms,
        enable_mouth=True,
        enable_tail=False,
        enable_head=False,
    )


# === Interlude Behavior ===
def _interlude_routine():
    try:
        move_head("off")
        time.sleep(random.uniform(0.2, 2))
        flap_count = random.randint(1, 3)
        for _ in range(flap_count):
            move_tail()
            time.sleep(random.uniform(0.25, 0.9))
        if random.random() < 0.9:
            move_head("on")
            # Head movement during interlude (no logging needed)
            # Auto-turn off head after max 3 seconds to prevent getting stuck
            threading.Timer(5.0, lambda: move_head("off")).start()
    except Exception as e:
        print(f"⚠️ Interlude error: {e}")


def interlude():
    """Run head/tail interlude in a background thread if not already running."""
    if _head_tail_lock.locked():
        return
    Thread(target=lambda: _interlude_routine(), daemon=True).start()


# === Motor Watchdog (per-pin continuous activity) ===
WATCHDOG_TIMEOUT_SEC = 30  # max continuous ON time per pin
WATCHDOG_POLL_SEC = 1.0  # poll cadence


def _mate_for(pin: int):
    """
    Return the logical 'mate' input that should be LOW when 'pin' drives.
    This lets the watchdog brake a channel safely.
    """
    if pin == MOUTH:
        return GND_1
    if pin == HEAD:
        if BILLY_PINS == "legacy":
            # legacy modern shares bridge with tail
            return TAIL
        # new layout: 3-motor => mate hard GND (None); 2-motor => mate is TAIL
        return None if USE_THIRD_MOTOR else TAIL
    if pin == TAIL:
        if BILLY_PINS == "legacy":
            return GND_3 if USE_THIRD_MOTOR else HEAD
        return None if USE_THIRD_MOTOR else HEAD
    return None


def _stop_channel(pin: int):
    """Brake one channel safely (pin + its mate)."""
    global _gpio_active
    if not _gpio_active:
        return  # GPIO handle already closed, skip
    mate = _mate_for(pin)
    clear_pwm(pin)
    try:
        lgpio.gpio_write(h, pin, 0)
    except (lgpio.error, Exception):
        _gpio_active = False
        return
    if mate is not None:
        clear_pwm(mate)
        try:
            lgpio.gpio_write(h, mate, 0)
        except (lgpio.error, Exception):
            _gpio_active = False
            return


def _pin_is_active(pin: int) -> bool:
    """Active if line is HIGH or PWM duty > 0."""
    if not _gpio_active:
        # If GPIO is inactive, only check PWM state
        return _pwm.get(pin, {}).get("duty", 0) > 0
    try:
        if lgpio.gpio_read(h, pin) == 1:
            return True
    except (lgpio.error, Exception):
        # Handle might be closed, fall back to PWM state
        pass
    return _pwm.get(pin, {}).get("duty", 0) > 0


def stop_all_motors():
    global _gpio_active
    reset_motor_sync_state()
    logger.info("Stopping all motors", "🛑")
    if not _gpio_active:
        return  # GPIO handle already closed, skip
    for pin in motor_pins:
        clear_pwm(pin)
        try:
            lgpio.gpio_write(h, pin, 0)
        except (lgpio.error, Exception):
            # Handle already closed or invalid - ignore during shutdown
            _gpio_active = False
            return


def cleanup_gpio():
    """Close GPIO chip handle to prevent memory corruption on shutdown."""
    global _gpio_active
    try:
        _gpio_active = (
            False  # Mark GPIO as inactive before closing to prevent new operations
        )
        stop_all_motors()  # This will now safely skip if handle is invalid
        time.sleep(0.1)  # Give any pending timer threads a moment to check the flag

        # Free all GPIO pins before closing the chip handle
        for pin in motor_pins:
            with contextlib.suppress(lgpio.error, Exception):
                lgpio.gpio_free(h, pin)

        with contextlib.suppress(lgpio.error, Exception):
            lgpio.gpiochip_close(h)  # Handle might already be closed, ignore
        logger.info("GPIO cleanup complete", "✅")
    except Exception as e:
        logger.warning(f"GPIO cleanup error: {e}", "⚠️")


def is_motor_active():
    return any(_pin_is_active(pin) for pin in motor_pins)


def motor_watchdog():
    """Stop any single pin that stays active longer than WATCHDOG_TIMEOUT_SEC."""
    global _motor_watchdog_running
    _motor_watchdog_running = True

    # Track continuous-on start time per pin
    since_on = {pin: None for pin in motor_pins}

    while _motor_watchdog_running:
        now = time.time()
        for pin in motor_pins:
            active = _pin_is_active(pin)
            if active:
                if since_on[pin] is None:
                    since_on[pin] = now
                else:
                    if (now - since_on[pin]) >= WATCHDOG_TIMEOUT_SEC:
                        logger.warning(
                            f"Watchdog: pin {pin} active > {WATCHDOG_TIMEOUT_SEC}s → braking channel",
                            "⏱️",
                        )
                        _stop_channel(pin)
                        since_on[pin] = None
            else:
                since_on[pin] = None
        time.sleep(WATCHDOG_POLL_SEC)


def start_motor_watchdog():
    Thread(target=motor_watchdog, daemon=True).start()


def stop_motor_watchdog():
    global _motor_watchdog_running
    _motor_watchdog_running = False


# Ensure safe shutdown
atexit.register(stop_all_motors)
atexit.register(stop_motor_watchdog)
