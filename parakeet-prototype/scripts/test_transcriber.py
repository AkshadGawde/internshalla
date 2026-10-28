"""
Test transcriber.py on its own.

  python -m scripts.test_transcriber              # record 5 s from the mic, then transcribe
  python -m scripts.test_transcriber speech.wav   # transcribe a 16 kHz mono 16-bit WAV file instead

Tip (macOS) - make a test file without a microphone:
  say -o speech.wav --file-format=WAVE --data-format=LEI16@16000 "What is a REST API?"
"""

import sys
import time
import wave

import numpy as np

import config
import transcriber
from audio import MicrophoneError, Recorder

SECONDS = 5


def load_wav(path):
    """Read a 16-bit PCM WAV file into float32 samples between -1 and 1."""
    with wave.open(path, "rb") as f:
        if f.getframerate() != config.SAMPLE_RATE or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise SystemExit(f"{path} must be {config.SAMPLE_RATE} Hz, mono, 16-bit")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def record(seconds):
    recorder = Recorder()
    try:
        recorder.start()
    except MicrophoneError as e:
        raise SystemExit(f"ERROR: {e}")
    print(f"Recording for {seconds} seconds... ask a question!")
    time.sleep(seconds)
    recorder.stop()
    return recorder.read_all()


print("Loading Whisper model (the very first run downloads ~145 MB)...")
t0 = time.perf_counter()
transcriber.load_model()
print(f"Model loaded in {time.perf_counter() - t0:.1f} s")

audio = load_wav(sys.argv[1]) if len(sys.argv) > 1 else record(SECONDS)
duration = audio.size / config.SAMPLE_RATE

t0 = time.perf_counter()
text = transcriber.transcribe(audio)
elapsed = time.perf_counter() - t0

print(f"Text           : {text!r}")
print(f"Audio length   : {duration:.2f} s")
print(f"Transcribed in : {elapsed * 1000:.0f} ms ({elapsed / duration:.2f}x real time)")
