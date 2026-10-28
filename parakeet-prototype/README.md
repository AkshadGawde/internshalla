# Parakeet Prototype: Local Interview Assistant

A small macOS desktop app inspired by Parakeet AI. It listens to your microphone, shows a live
transcript, and when you click **Suggest Answer** it streams an AI-generated answer to the
question into the window word by word.

**Everything runs locally.** Speech-to-text (faster-whisper) and the LLM (Llama 3.2 3B via Ollama)
both run on your own machine. After the one-time model downloads, no internet is needed and no audio
or text leaves the computer.

## Features

- Desktop window with **Start**, **Stop**, **Suggest Answer** and **Clear** buttons
- Near-live transcription: the transcript updates every 3 seconds while you speak
- A final, more accurate transcription pass over the whole recording when you press Stop
- Answers stream in as they are generated
- Latency cards: live-chunk and final speech-to-text time, time to first token, total answer time
- Dark UI with a status indicator, a blinking REC timer and a live mic level meter
- Copy-answer button, "Keep on top" toggle (overlay-style), and ⌘/Ctrl + Enter to suggest an answer
- Friendly errors for: Ollama not running, model not downloaded, no microphone, missing mic
  permission (macOS returns silence), and an empty transcript

## Architecture

```
 ┌────────────┐  16 kHz mono   ┌──────────────┐  every 3 s   ┌─────────────────────┐
 │ Microphone │ ─────────────▶ │  audio.py    │ ───────────▶ │  transcriber.py     │
 │            │  float32 blocks│  (Recorder,  │   numpy      │  faster-whisper     │
 └────────────┘                │  sounddevice)│   audio      │  base.en, CPU, int8 │
                               └──────────────┘              │  + VAD filter       │
                                                             └──────────┬──────────┘
                                                                        │ text
                                                                        ▼
 ┌────────────────────────┐   "Suggest Answer"   ┌───────────────────────────────────┐
 │  AI Answer box (UI)    │ ◀── text pieces ──── │  llm.py → Ollama /api/chat        │
 │  streamed word by word │   (generator)        │  llama3.2:3b, "stream": true      │
 └────────────────────────┘                      │  newline-delimited JSON           │
                                                 └───────────────────────────────────┘
```

**Threads (app.py).** Tkinter is single-threaded and not thread-safe, so slow work never runs on
the UI thread:

```
PortAudio audio thread ──▶ recorder.queue ──▶ transcription thread ─┐
                                                                    ├─▶ ui_queue ──▶ UI thread
                                              answer thread ────────┘   (root.after poll every 50 ms
                                                                         updates the widgets)
```

Background threads only put messages like `("answer_piece", "Hello")` into a `queue.Queue`. The UI
thread checks that queue every 50 ms with `root.after()` and is the only thread that touches widgets.

| File | Responsibility |
|---|---|
| `config.py` | All settings: sample rate, chunk length, model names, Ollama URL, system prompt |
| `audio.py` | `Recorder`: opens the mic with `sounddevice.InputStream`, callback pushes blocks into a thread-safe queue |
| `transcriber.py` | Loads the Whisper model once; `transcribe(numpy_audio) -> str` |
| `llm.py` | `stream_answer(question)`: a generator that yields text pieces from Ollama's streaming API |
| `app.py` | Tkinter window, threads, button states, latency display |
| `theme.py` | Colours, fonts and ttk styles (the look), kept separate from the behaviour in `app.py` |
| `scripts/` | Small scripts that test each module on its own |

## Setup (macOS, Apple Silicon)

**1. Install Ollama and download the model**
```bash
brew install ollama            # or download the app from https://ollama.com/download
brew services start ollama     # runs the server in the background (or: `ollama serve` in a terminal)
ollama pull llama3.2:3b        # ~2 GB
```

**2. Create a virtual environment and install dependencies** (Python 3.10+)
```bash
cd parakeet-prototype
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run**
```bash
python app.py
```
The first launch downloads the Whisper `base.en` model (~145 MB) into `~/.cache/huggingface`.

**4. Microphone permission (macOS).** The first time you press Start, macOS asks whether your
terminal (Terminal, iTerm or VS Code) may use the microphone. Click **Allow**. If you denied it
earlier: *System Settings → Privacy & Security → Microphone*, turn on your terminal app, then restart
it. Without permission macOS sends silence instead of an error; the app detects this and shows a
warning.

**Usage:** press **Start** → ask a question out loud → **Stop** → **Suggest Answer**.
The transcript box is editable, so you can fix a misheard word before asking.

### Testing each module on its own
```bash
python -m scripts.test_audio                    # record 5 s, print sample count + peak volume
python -m scripts.test_transcriber              # record 5 s, transcribe, print time taken
python -m scripts.test_transcriber speech.wav   # or transcribe a 16 kHz mono WAV file
python -m scripts.test_llm "What is REST API?"  # stream an answer, print time to first token + total
```
To make a test WAV without a microphone:
`say -o speech.wav --file-format=WAVE --data-format=LEI16@16000 "What is a REST API?"`

## Tested hardware and measured latency

| | |
|---|---|
| Machine | MacBook Pro 14" (Mac15,3), Apple M3: 8-core CPU (4 performance + 4 efficiency), 10-core GPU |
| RAM | 8 GB unified memory |
| OS / Python | macOS 26.5, Python 3.12.7 |
| Speech-to-text | faster-whisper 1.2.1, `base.en`, CPU, int8 |
| LLM | Ollama 0.40.11, `llama3.2:3b` (Q4), 100% on the Apple GPU via Metal |

| Measurement | Result |
|---|---|
| Whisper model load (cached / first run incl. download) | 0.8 s / 16 s |
| Transcribing one 3 s live chunk | ~1.0–1.3 s (≈ 0.3–0.4× real time) |
| Final pass over a ~6.5 s recording | ~1.3 s |
| Time to first token (model already in memory) | **0.4–0.5 s** |
| Time to first token (cold, model loading from disk) | ~11 s, so the app warms the model up at startup |
| Total answer time (3–5 sentences, ~90–110 tokens) | ~16–18 s (≈ 6–7 tokens/s) |

Note: these numbers were measured while Chrome, Arc and VS Code were open, with ~10 GB of swap in use.
On an 8 GB machine that memory pressure slows token generation a lot. With other apps closed, the
same model is expected to generate several times faster.

## Design decisions

- **faster-whisper `base.en` + int8 on CPU.** Good accuracy for English at better than real-time
  speed on a laptop CPU. `tiny.en` is faster but makes more mistakes; `small.en` is more accurate but
  ~3× slower. faster-whisper has no Metal (Apple GPU) backend, so CPU + int8 is the fastest option here.
- **VAD filter.** Removes silence before Whisper sees it. This stops Whisper from inventing text like
  "Thank you." on silent chunks, and makes silent chunks almost free to process.
- **Live chunks + final pass.** 3 s chunks give quick feedback, but a chunk boundary can cut a word
  in half (in testing, "SQL and NoSQL" became "SQL and and no-school"). After Stop, the whole
  recording is transcribed once more and replaces the preview, so the text sent to the LLM is accurate.
- **Llama 3.2 3B via Ollama.** Small enough (2 GB) to stay in memory on an 8 GB laptop and use the GPU.
  Ollama handles model download, quantization and Metal acceleration, and exposes a simple HTTP API.
- **Streaming + warm-up.** Streaming means the user starts reading after ~0.5 s instead of waiting
  ~17 s for the full answer. Loading the model at startup (`llm.warm_up()`, `keep_alive: 30m`)
  removes the ~11 s cold start from the first question.
- **Threads + queue + `root.after()`.** This keeps the window responsive and respects Tkinter's rule
  that only the UI thread touches widgets.

## Known limitations

- **3-second chunks can split words** in the live preview (fixed in the final transcript, but the
  preview can briefly show mistakes). Very long recordings make the final pass slower.
- **Microphone only, no system-audio capture.** It can't hear the other person in a Zoom or Meet
  call unless the sound comes out of the speakers into the mic.
- **CPU-only Whisper.** faster-whisper can't use the Apple GPU.
- **Manual question detection.** The user decides when to click Suggest Answer, and the whole
  transcript is sent as the question.
- **Small-model accuracy.** A 3B model can state wrong facts confidently. In testing it once expanded
  REST as "Representational State of Resource" (it is Representational State **Transfer**).
- **No conversation memory.** Each answer is independent.
- On 8 GB machines, answer speed depends heavily on how much other apps are using memory.

## Next steps

- **System audio capture:** WASAPI loopback on Windows (e.g. the `soundcard` library);
  ScreenCaptureKit on macOS 13+ (or a virtual device like BlackHole) to hear the interviewer.
- **Automatic question detection:** detect when the speaker finishes a question (VAD silence +
  a "?"/question-word heuristic or a tiny classifier) and answer automatically.
- **GPU / Metal acceleration:** `whisper.cpp` or `mlx-whisper` use the Apple GPU; CUDA for
  faster-whisper on Windows/NVIDIA.
- **Smaller / faster models:** `distil-whisper` or `tiny.en` for STT; `llama3.2:1b` or
  `qwen2.5:1.5b` for answers when speed matters more than depth.
- **Better live transcription:** overlapping chunks, or passing earlier text as Whisper's
  `initial_prompt` for context.
- **Fine-tuning / RAG:** fine-tune on interview Q&A, or feed in the user's resume and job description
  so answers are personalised.
- **UI:** an Electron/React or Qt front end, an always-on-top overlay window, keyboard shortcuts.

---

Built with AI-assisted coding (Claude Code); design decisions, testing and measurements are my own.
