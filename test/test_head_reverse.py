#!/usr/bin/env python3
"""
Head Motor Reverse Test
Tests head motor with reversed polarity
"""
import time
import lgpio

HEAD = 27  # GPIO 27 (Pin 13)

print("🐟 Head Motor Reverse Test")
print("=" * 40)

try:
    # Initialize GPIO
    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(h, HEAD)
    lgpio.gpio_write(h, HEAD, 0)
    
    print("Head motor ON (reversed - 3 seconds)...")
    lgpio.gpio_write(h, HEAD, 0)  # Reverse: write 0 instead of 1
    time.sleep(3)
    
    print("Head motor OFF")
    lgpio.gpio_write(h, HEAD, 1)  # Reverse: write 1 instead of 0
    
    # Cleanup
    lgpio.gpio_free(h, HEAD)
    lgpio.gpiochip_close(h)
    print("✓ Test complete - did it spin the other direction?")
    
except Exception as e:
    print(f"❌ Error: {e}")
