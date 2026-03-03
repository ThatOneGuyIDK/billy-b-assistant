"""
Simple microphone test script - no API keys needed!
Tests if your mic is working by showing audio levels in real-time.
"""
import time
import numpy as np
import sounddevice as sd
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

print("🎤 Microphone Test")
print("=" * 50)
print("This will show real-time audio levels from your mic.")
print("Speak into your microphone - you should see the bars move.")
print("Press Ctrl+C to stop.\n")

# List available audio devices
print("Available audio devices:")
print(sd.query_devices())
print()

# Audio settings (matching Billy's config)
RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 480

def audio_callback(indata, frames, time_info, status):
    """Display audio level as a simple bar graph"""
    if status:
        print(f"Status: {status}")
    
    # Calculate volume (RMS)
    volume = np.sqrt(np.mean(indata**2))
    
    # Handle NaN or invalid values
    if np.isnan(volume) or not np.isfinite(volume):
        volume = 0
    
    # Scale to 0-50 for display
    bars = int(volume / 100)
    bars = min(bars, 50)  # Cap at 50
    bars = max(bars, 0)   # Prevent negative
    
    # Display as bar graph
    bar_display = "█" * bars
    print(f"\rLevel: {bar_display:<50} ({int(volume):>5})", end="", flush=True)

try:
    print("Starting microphone stream...")
    with sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        print("✅ Mic is active! Speak now...\n")
        while True:
            time.sleep(0.1)
            
except KeyboardInterrupt:
    print("\n\n✅ Test complete!")
except Exception as e:
    print(f"\n\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure your microphone is connected")
    print("2. Check device permissions")
    print("3. Try selecting a specific device index")
