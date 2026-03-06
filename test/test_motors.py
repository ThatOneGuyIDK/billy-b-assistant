#!/usr/bin/env python3
"""
Motor Test Script
Tests all three Billy motors individually to verify wiring.
"""
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import lgpio
except ImportError:
    print("❌ Error: lgpio library not found")
    print("Install with: sudo apt install python3-lgpio")
    sys.exit(1)

print("🐟 Billy Motor Test")
print("=" * 60)
print("This will test each motor individually.")
print("Watch each motor to verify it spins correctly.")
print()

# Pin assignments (NEW wiring)
HEAD = 22   # GPIO 22 (Pin 15)
MOUTH = 17  # GPIO 17 (Pin 11)
TAIL = 27   # GPIO 27 (Pin 13)

# Initialize GPIO
try:
    h = lgpio.gpiochip_open(0)
    print("✓ GPIO initialized\n")
except Exception as e:
    print(f"❌ Failed to initialize GPIO: {e}")
    sys.exit(1)

# Setup all motor pins
motors = [
    ("HEAD", HEAD, 15),
    ("MOUTH", MOUTH, 11),
    ("TAIL", TAIL, 13)
]

for name, gpio, pin in motors:
    try:
        lgpio.gpio_claim_output(h, gpio)
        lgpio.gpio_write(h, gpio, 0)
    except Exception as e:
        print(f"❌ Failed to setup {name} motor (GPIO {gpio}): {e}")
        lgpio.gpiochip_close(h)
        sys.exit(1)

print("✓ All motors initialized\n")

# Test each motor
def test_motor(name, gpio, pin):
    """Test a single motor"""
    print(f"Testing {name} motor (GPIO {gpio}, Physical Pin {pin})...")
    print("  → Motor should spin now...")
    
    # Turn on motor
    lgpio.gpio_write(h, gpio, 1)
    time.sleep(3)
    
    # Turn off motor
    lgpio.gpio_write(h, gpio, 0)
    print("  → Motor stopped")
    print()

try:
    print("=" * 60)
    print("Starting motor tests...")
    print("=" * 60)
    print()
    
    for name, gpio, pin in motors:
        test_motor(name, gpio, pin)
        time.sleep(1)  # Pause between tests
    
    print("=" * 60)
    print("✓ All motor tests complete!")
    print()
    print("Verification checklist:")
    print("  [ ] Head motor (Red/Black wires) spun")
    print("  [ ] Mouth motor (White/Yellow wires) spun")
    print("  [ ] Tail motor (Orange/Brown wires) spun")
    print()
    print("If any motor didn't spin or spun backwards:")
    print("  - Check wiring connections")
    print("  - Verify GPIO pins match the wiring guide")
    print("  - Check motor driver power supply")
    print("  - Swap motor wires if it spins backwards")
    
except KeyboardInterrupt:
    print("\n\n⚠ Test interrupted by user")
    
except Exception as e:
    print(f"\n❌ Error during test: {e}")
    
finally:
    # Cleanup
    print("\nCleaning up GPIO...")
    for name, gpio, pin in motors:
        try:
            lgpio.gpio_write(h, gpio, 0)
            lgpio.gpio_free(h, gpio)
        except:
            pass
    lgpio.gpiochip_close(h)
    print("✓ GPIO cleaned up")
