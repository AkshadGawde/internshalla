"""
Test audio.py on its own: record 5 seconds and print basic stats.

Run from the project folder:  python -m scripts.test_audio
"""

import time

import numpy as np

import config
from audio import MicrophoneError, Recorder, looks_silent

SECONDS = 5

recorder = Recorder()
try:
    recorder.start()
except MicrophoneError as e:
    print("ERROR:", e)
    raise SystemExit(1)

print(f"Recording for {SECONDS} seconds... say something!")
time.sleep(SECONDS)
recorder.stop()

audio = recorder.read_all()
peak = float(np.max(np.abs(audio))) if audio.size else 0.0

print(f"Samples recorded : {audio.size} (expected about {SECONDS * config.SAMPLE_RATE})")
print(f"Duration         : {audio.size / config.SAMPLE_RATE:.2f} s")
print(f"Peak volume      : {peak:.3f} (0.0 = silent, 1.0 = maximum)")

if looks_silent(audio):
    print(
        "WARNING: the audio is completely silent. On macOS this usually means your "
        "terminal has no microphone permission:\n"
        "  System Settings > Privacy & Security > Microphone > turn on Terminal / "
        "VS Code, then restart that app."
    )
elif peak < 0.01:
    print("Note: very quiet - check that the right input device is selected.")
