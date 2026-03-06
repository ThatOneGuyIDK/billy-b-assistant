#!/usr/bin/env python3
"""
Head Motor Test - Quick test for head motor only
"""
import time
import lgpio

HEAD = 22  # GPIO 22 (Pin 15)

print("🐟 Head Motor Test")
print("=" * 40)

try:
    # Initialize GPIO
    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(h, HEAD)
    lgpio.gpio_write(h, HEAD, 0)
    
    print("Head motor ON (3 seconds)...")
    lgpio.gpio_write(h, HEAD, 1)
    time.sleep(3)
    
    print("Head motor OFF")
    lgpio.gpio_write(h, HEAD, 0)
    
    # Cleanup
    lgpio.gpio_free(h, HEAD)
    lgpio.gpiochip_close(h)
    print("✓ Test complete")
    
except Exception as e:
    print(f"❌ Error: {e}")
