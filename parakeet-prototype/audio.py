"""
audio.py - records microphone audio in the background.

How it works:
  sounddevice opens an InputStream. While the stream is running, PortAudio
  (the C library underneath) calls our _callback() on ITS OWN thread every
  few milliseconds with a small block of new samples. The callback just copies
  that block into a thread-safe queue.Queue. Whoever wants the audio (the test
  script, or the transcription thread in app.py) takes blocks out of the queue.

The callback must be fast: if it does slow work (like running Whisper),
PortAudio's buffer overflows and audio gets lost. So: copy, enqueue, return.
"""

import queue
import sys

import numpy as np
import sounddevice as sd

import config


class MicrophoneError(Exception):
    """Raised when the microphone can't be opened (missing, busy, or no permission)."""


class Recorder:
    def __init__(self, device=None):
        # device=None means "the system's default input device".
        self.device = device
        # queue.Queue is thread-safe: the audio thread put()s, another thread get()s,
        # and the Queue handles the locking for us.
        self.queue = queue.Queue()
        self._stream = None
        # Loudness of the most recent block (0.0-1.0), for the UI's level meter.
        # Written by the audio thread, read by the UI thread. Replacing a float
        # attribute is a single atomic step in Python, so no lock is needed.
        self.level = 0.0

    @property
    def is_recording(self):
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        """Called by PortAudio on its own thread with each new block of audio."""
        if status:
            # e.g. "input overflow" - we were too slow and some samples were dropped
            print(f"[audio] {status}", file=sys.stderr)
        # indata has shape (frames, channels). We recorded mono, so take column 0.
        # .copy() is required: sounddevice reuses the indata buffer for the next block.
        self.queue.put(indata[:, 0].copy())
        self.level = float(np.abs(indata).max())

    def start(self):
        """Open the microphone and start filling the queue."""
        if self.is_recording:
            return
        self.clear()  # throw away any audio left over from a previous recording

        # 1) Is there an input device at all? (e.g. desktop PC with no mic plugged in)
        try:
            sd.query_devices(self.device, kind="input")
        except (ValueError, sd.PortAudioError) as e:
            raise MicrophoneError(
                "No microphone found. Plug one in or pick an input device in your "
                f"system sound settings.\n(details: {e})"
            ) from e

        # 2) Open the stream. This fails if the device is busy or access is denied.
        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype="float32",          # samples are floats between -1.0 and 1.0
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            self._stream = None
            raise MicrophoneError(
                "Could not open the microphone. It may be in use by another app, or "
                "this app may not have microphone permission.\n"
                "macOS: System Settings > Privacy & Security > Microphone > enable your "
                "terminal / VS Code.\n"
                "Windows: Settings > Privacy & security > Microphone > allow desktop apps.\n"
                f"(details: {e})"
            ) from e

    def stop(self):
        """Stop and close the microphone. Audio already in the queue stays there."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.level = 0.0

    def clear(self):
        """Empty the queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def read_all(self):
        """Take every block currently in the queue and join them into one 1-D array."""
        blocks = []
        while True:
            try:
                blocks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)


def looks_silent(audio):
    """True if the audio is pure digital silence (all zeros).

    On macOS, if the app has NO microphone permission, the OS does not raise an
    error - it quietly hands us zeros. Real rooms are never perfectly silent, so
    an exact 0.0 peak almost always means a permission problem.
    """
    return audio.size > 0 and float(np.max(np.abs(audio))) == 0.0
