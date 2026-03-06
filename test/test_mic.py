#!/usr/bin/env python3
"""
Microphone Test Script
Tests if your microphone is working by showing real-time audio levels.
No API keys needed - pure audio hardware test.
"""
import time
import numpy as np
import sounddevice as sd
import warnings
import sys

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

print("🎤 Microphone Test")
print("=" * 60)
print("This will show real-time audio levels from your mic.")
print("Speak into your microphone - you should see the bars move.")
print("Press Ctrl+C to stop.\n")

# List available audio devices
print("Available audio devices:")
try:
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        marker = ">" if i == sd.default.device[0] else " "
        in_channels = d.get('max_input_channels', 0)
        if in_channels > 0:
            print(f"{marker} {i}: {d['name']} ({in_channels} in)")
except Exception as e:
    print(f"Error querying devices: {e}")

print()

# Audio settings - try to find supported sample rate
CHANNELS = 1
CHUNK_SIZE = 480
RATE = None

# Try common sample rates in order of preference
print("Detecting supported sample rate...")
for sr in [16000, 48000, 44100, 32000, 22050]:
    try:
        # Test if this sample rate works
        sd.check_input_configuration(samplerate=sr, channels=CHANNELS)
        RATE = sr
        print(f"✓ Using sample rate: {RATE} Hz\n")
        break
    except Exception as e:
        continue

if RATE is None:
    RATE = 16000  # Fallback
    print(f"⚠ Using fallback sample rate: {RATE} Hz (may have errors)\n")

def audio_callback(indata, frames, time_info, status):
    """Display audio level as a simple bar graph"""
    if status:
        print(f"\n⚠ Status: {status}")
    
    # Calculate volume (RMS - Root Mean Square)
    volume = np.sqrt(np.mean(indata**2))
    
    # Handle NaN or invalid values
    if np.isnan(volume) or not np.isfinite(volume):
        volume = 0
    
    # Scale to 0-50 for display
    bars = int(volume * 50)
    bars = min(bars, 50)  # Cap at 50
    bars = max(bars, 0)   # Prevent negative
    
    # Display as bar graph with threshold line
    bar_display = "█" * bars
    threshold_mark = "|" if bars > 0 else " "
    print(f"\rLevel: {bar_display:<50} {int(volume*100):>3}%", end="", flush=True)

try:
    print("Starting microphone stream...")
    print("Listening (speak now)...\n")
    
    with sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
        latency='low'
    ):
        # Keep the stream running until Ctrl+C
        while True:
            time.sleep(0.1)
            
except KeyboardInterrupt:
    print("\n\n✓ Microphone test stopped.")
    print("If you saw the bars move when you spoke, your mic is working!")
    
except Exception as e:
    print(f"\n\n❌ Error: {type(e).__name__}: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure your microphone is connected")
    print("2. Check device permissions (sudo chmod +rw /dev/snd/*)")
    print("3. Try a different USB port")
    print("4. Check if 'arecord -l' shows your device")
    sys.exit(1)
