#!/usr/bin/env python3
"""
Head Motor Extended Test - Move head further out with tx_pwm
"""
import time
import lgpio

HEAD = 27  # GPIO 27 (Pin 13)
FREQ = 10000  # PWM frequency (10kHz)

print("🐟 Head Motor Extended Test")
print("=" * 40)

try:
    # Initialize GPIO
    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(h, HEAD)
    lgpio.gpio_write(h, HEAD, 0)
    
    print("Head motor ON (100% power, 10 seconds)...")
    # Use tx_pwm with 100% duty (0-100 range)
    lgpio.tx_pwm(h, HEAD, FREQ, 100)
    time.sleep(10)
    
    print("Head motor OFF")
    lgpio.tx_pwm(h, HEAD, FREQ, 0)
    
    # Cleanup
    lgpio.gpio_free(h, HEAD)
    lgpio.gpiochip_close(h)
    print("✓ Test complete - head should have moved further out")
    
except Exception as e:
    print(f"❌ Error: {e}")
