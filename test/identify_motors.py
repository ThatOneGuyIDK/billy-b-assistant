#!/usr/bin/env python3
"""
Motor GPIO Identification Script
Run this to identify which GPIO pin controls which motor
"""
import time
import lgpio

print("🐟 Motor GPIO Identification")
print("=" * 60)
print("This will test each GPIO pin one at a time.")
print("Watch which motor moves and note it down.")
print("=" * 60)
print()

# Test GPIO pins
test_pins = [
    (22, "GPIO 22 (Pin 15) - Should be HEAD"),
    (17, "GPIO 17 (Pin 11) - Should be MOUTH"),
    (27, "GPIO 27 (Pin 13) - Should be TAIL"),
]

results = {}

try:
    h = lgpio.gpiochip_open(0)
    
    for gpio, description in test_pins:
        print(f"\nTesting {description}")
        print("-" * 60)
        
        # Setup pin
        lgpio.gpio_claim_output(h, gpio)
        lgpio.gpio_write(h, gpio, 0)
        
        input(f"Press ENTER to test GPIO {gpio}...")
        
        # Turn on
        print(f"  → GPIO {gpio} ON - WATCH WHICH MOTOR MOVES!")
        lgpio.gpio_write(h, gpio, 1)
        time.sleep(2)
        
        # Turn off
        lgpio.gpio_write(h, gpio, 0)
        print(f"  → GPIO {gpio} OFF")
        
        # Ask which motor moved
        print()
        print("Which motor moved?")
        print("  1 = Head (Red/Black wires)")
        print("  2 = Mouth (White/Yellow wires)")
        print("  3 = Tail (Orange/Brown wires)")
        choice = input("Enter 1, 2, or 3: ").strip()
        
        motor_names = {"1": "HEAD", "2": "MOUTH", "3": "TAIL"}
        motor_name = motor_names.get(choice, "UNKNOWN")
        results[gpio] = motor_name
        
        # Cleanup pin
        lgpio.gpio_free(h, gpio)
        print()
    
    # Show results
    print("=" * 60)
    print("WIRING IDENTIFICATION RESULTS:")
    print("=" * 60)
    for gpio, motor in results.items():
        print(f"  GPIO {gpio} → {motor} motor")
    
    print()
    print("Copy these values to your .env file if they're different from:")
    print("  HEAD = GPIO 22")
    print("  MOUTH = GPIO 17")
    print("  TAIL = GPIO 27")
    
    lgpio.gpiochip_close(h)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
