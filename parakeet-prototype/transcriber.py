"""
transcriber.py - turns audio (a numpy array) into text with faster-whisper.

faster-whisper is a re-implementation of OpenAI's Whisper on top of
CTranslate2, a fast inference engine. Same model weights, roughly 4x faster
on CPU, and it supports int8 quantization.

The model takes a few seconds to load, so we load it ONCE and reuse it
(a module-level "singleton"). The first load ever also downloads the model
(~145 MB) from Hugging Face into ~/.cache/huggingface - after that it works
fully offline.
"""

import numpy as np
from faster_whisper import WhisperModel

import config

_model = None  # the loaded WhisperModel, created on first use


def load_model():
    """Load the Whisper model if it isn't loaded yet. Safe to call many times."""
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe(audio):
    """Transcribe a 1-D float32 numpy array (16 kHz mono) and return the text."""
    if audio is None or audio.size == 0:
        return ""

    model = load_model()
    segments, _info = model.transcribe(
        audio.astype(np.float32),
        language="en",
        # VAD = Voice Activity Detection. It cuts out the silent parts before
        # Whisper sees them. Without it, Whisper sometimes "hallucinates" text
        # like "Thank you." on pure silence.
        vad_filter=True,
    )
    # 'segments' is a generator: the actual decoding happens while we loop over it.
    return " ".join(segment.text.strip() for segment in segments).strip()
