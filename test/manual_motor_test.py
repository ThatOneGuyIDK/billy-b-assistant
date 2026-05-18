#!/usr/bin/env python3
"""
Safe manual motor test for Billy B-Assistant.

Run this on the Raspberry Pi where the hardware is connected. The script
prompts before pulsing each motor briefly (safe defaults). It uses the
movement helpers in `core/movements.py` so wiring logic and safety checks
are reused.

Usage (from repo root):
    venv/bin/python test/manual_motor_test.py
"""
import os
import sys
import time

# Allow `from core import ...` when run as a script from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import movements


def pause(msg: str = "Press ENTER to continue..."):
    try:
        input(msg)
    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(1)


def test_mouth(duration=0.5, speed=80):
    print(f"Pulsing MOUTH (pin={movements.MOUTH}) for {duration}s at {speed}%")
    movements.move_mouth(speed, duration, brake=True)
    time.sleep(duration + 0.2)


def test_head(duration=0.6):
    drive, idle = movements._head_extend_pins()
    print(
        f"Extending HEAD — software HEAD GPIO={movements.HEAD}, "
        f"drive GPIO={drive}, idle GPIO={idle} — for {duration}s"
    )
    from core.config import MOTOR_REVERSE_ALL, MOTOR_REVERSE_HEAD

    if MOTOR_REVERSE_HEAD or MOTOR_REVERSE_ALL:
        print("  (MOTOR_REVERSE_HEAD/ALL is on — head may drive the tail GPIO on a shared driver)")
    movements.move_head("on")
    time.sleep(duration)
    movements.move_head("off")
    time.sleep(0.2)


def test_tail(duration=0.5):
    print(f"Pulsing TAIL (GPIO={movements.TAIL}) for {duration}s")
    from core.config import BILLY_PINS, is_classic_billy

    if not is_classic_billy() and BILLY_PINS != "legacy":
        print("  (modern 2-motor: tail shares driver with head — opposite GPIO from head test)")
    movements.move_tail(duration=duration)
    time.sleep(duration + 0.2)


def main():
    print("=== Billy Motor Manual Test ===")
    print("Note: If you are running on a non-Pi or have MOCKFISH=true, motors will be mocked and won't move.")
    print(f"MOCKFISH flag: {movements.MOCKFISH}")
    from core.config import BILLY_MODEL, is_classic_billy

    print(f"Pin mapping -> MOUTH: {movements.MOUTH}, HEAD: {movements.HEAD}, TAIL: {movements.TAIL}")
    print(f"BILLY_MODEL={BILLY_MODEL} (classic=3 motors, modern=head+tail share one driver)")
    print(f"Separate tail driver: {is_classic_billy()}")
    print()

    pause("Ensure the area is clear and the fish can move freely. Press ENTER to test MOUTH...")
    test_mouth()

    pause("Check MOUTH movement. Press ENTER to test HEAD extend/retract...")
    test_head()

    pause("Check HEAD movement. Press ENTER to test TAIL...")
    test_tail()

    print("\nManual motor test complete. If a motor did not move, check wiring, .env pin/profile, and that lgpio is installed on the Pi.")


if __name__ == "__main__":
    main()
