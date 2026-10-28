"""
config.py - every setting for the app lives here.

The other modules import from this file, so changing a model or a
timing value means editing ONE line instead of hunting through the code.
"""

# ---------------------------------------------------------------------------
# Audio (microphone)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000   # samples per second; Whisper models are trained on 16 kHz audio
CHANNELS = 1          # mono - speech recognition doesn't need stereo
CHUNK_SECONDS = 3     # how much audio we collect before sending it to Whisper
                      # (smaller = more "live" but more words get cut in half)

# ---------------------------------------------------------------------------
# Speech-to-text (faster-whisper, runs locally)
# ---------------------------------------------------------------------------
WHISPER_MODEL = "base.en"     # English-only "base" model (~150 MB): good speed/accuracy balance on CPU
WHISPER_DEVICE = "cpu"        # faster-whisper has no Apple-GPU (Metal) support, so CPU it is
WHISPER_COMPUTE_TYPE = "int8" # 8-bit integer weights: ~4x smaller than float32 and faster on CPU

# ---------------------------------------------------------------------------
# LLM (Ollama, runs locally)
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"  # Ollama's default local server address
OLLAMA_MODEL = "llama3.2:3b"           # small enough to answer quickly on an 8 GB laptop
OLLAMA_TIMEOUT = 120                   # seconds to wait before giving up on Ollama
OLLAMA_KEEP_ALIVE = "30m"              # keep the model in RAM between questions (default is 5 min);
                                       # reloading it from disk adds several seconds to the first answer

# The "system prompt" sets the model's role and answer style before it sees the question.
SYSTEM_PROMPT = (
    "You are a helpful interview assistant. "
    "The user message is a transcript of someone speaking, captured by a microphone, "
    "so it may contain small transcription mistakes or filler words. "
    "Find the question being asked and answer it in 3 to 5 short, clear sentences, "
    "written the way a person would say it out loud. "
    "Do not use bullet points, headings, markdown or code blocks."
)
