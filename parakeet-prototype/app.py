"""
app.py - the desktop window. Wires microphone -> Whisper -> transcript -> Ollama -> answer.

Run it with:  python app.py

THREADING DESIGN (the most important idea in this file)
-------------------------------------------------------
Tkinter runs everything on ONE thread, the "UI thread", inside root.mainloop().
While a function on that thread is busy, the window can't redraw or react to
clicks - it freezes. Whisper (~1 s per chunk) and the LLM (several seconds per
answer) are far too slow to run there.

So the slow work runs on background threads:
  * PortAudio's audio thread -> puts raw audio blocks into recorder.queue
  * transcription thread     -> takes audio from recorder.queue, runs Whisper
  * answer thread            -> runs llm.stream_answer() and reads the pieces

But Tkinter is NOT thread-safe: only the UI thread may touch widgets. So the
background threads never touch widgets. Instead they put small messages like
("transcript_live", text) into self.ui_queue. The UI thread checks that queue every
50 ms using root.after() (a timer that runs a function later ON the UI thread)
and applies the updates. queue.Queue does all the locking for us.

    audio thread --> recorder.queue --> transcription thread --+
                                                               +--> ui_queue --> UI thread (after() poll) --> widgets
                                          answer thread -------+
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

import config
import llm
import transcriber
from audio import MicrophoneError, Recorder, looks_silent
from theme import COLORS, apply_theme, style_text_box

POLL_MS = 50  # how often the UI thread checks ui_queue for updates

MIC_SILENT_MSG = (
    "The microphone is sending pure silence. On macOS, allow microphone access: "
    "System Settings > Privacy & Security > Microphone > enable Terminal / VS Code, then restart it."
)

# Text and colour of the status "pill" in the header, for each state of the app.
MODES = {
    "loading": ("Loading models", "amber"),
    "ready": ("Ready", "green"),
    "recording": ("Recording", "red"),
    "finishing": ("Transcribing", "amber"),
    "answering": ("Answering", "blue"),
}

# The latency cards at the bottom: key in self.latency -> card title.
METRICS = {
    "stt": "Live chunk STT",
    "final_stt": "Final STT",
    "first_token": "First token",
    "total": "Total answer",
}


class App:
    def __init__(self, root):
        self.root = root
        self.recorder = Recorder()
        self.ui_queue = queue.Queue()        # messages from background threads to the UI thread
        self.stop_event = threading.Event()  # set by Stop to tell the transcription thread to finish

        # Simple state flags; _update_buttons() turns them into enabled/disabled buttons.
        self.model_ready = False  # Whisper loaded?
        self.recording = False    # True from Start until the last chunk has been transcribed
        self.stopping = False     # Stop pressed, still transcribing leftover audio
        self.answering = False    # an answer is streaming in
        self.rec_started = 0.0    # time.monotonic() when Start was pressed, for the REC timer
        self.latency = {}         # METRICS key -> milliseconds

        apply_theme(root)
        self._build_ui()
        self._update_buttons()
        self._set_status("Loading speech model... (the first run downloads it)")

        # Load models in the background so the window appears immediately.
        threading.Thread(target=self._load_models, daemon=True).start()
        self.root.after(POLL_MS, self._poll_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI layout
    def _build_ui(self):
        self.root.title("Parakeet Prototype")
        self.root.geometry("820x780")
        self.root.minsize(640, 600)

        frame = ttk.Frame(self.root, padding=(20, 16))
        frame.pack(fill="both", expand=True)

        # --- Header: title + subtitle on the left, status pill + "keep on top" on the right
        header = ttk.Frame(frame)
        header.pack(fill="x")
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="Parakeet Prototype", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            titles, style="Muted.TLabel",
            text=f"Local interview assistant  ·  {config.WHISPER_MODEL} + {config.OLLAMA_MODEL}  ·  100% offline",
        ).pack(anchor="w")

        right = ttk.Frame(header)
        right.pack(side="right", anchor="n")
        self.pill = ttk.Label(right, style="Pill.TLabel")
        self.pill.pack(anchor="e")
        self.on_top = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            right, text="Keep on top", variable=self.on_top,
            command=lambda: self.root.attributes("-topmost", self.on_top.get()),
        ).pack(anchor="e", pady=(6, 0))

        # --- Controls: buttons on the left, REC timer + mic level meter on the right
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(18, 14))
        self.start_btn = ttk.Button(controls, text="●  Start", style="Start.TButton", command=self.start)
        self.stop_btn = ttk.Button(controls, text="■  Stop", style="Stop.TButton", command=self.stop)
        self.suggest_btn = ttk.Button(controls, text="✦  Suggest Answer", style="Accent.TButton",
                                      command=self.suggest_answer)
        self.clear_btn = ttk.Button(controls, text="Clear", style="Ghost.TButton", command=self.clear)
        for btn in (self.start_btn, self.stop_btn, self.suggest_btn, self.clear_btn):
            btn.pack(side="left", padx=(0, 8))

        self.level_bar = ttk.Progressbar(controls, style="Level.Horizontal.TProgressbar",
                                         length=110, maximum=1.0)
        self.level_bar.pack(side="right", padx=(10, 0))
        self.rec_label = ttk.Label(controls, text="", style="Rec.TLabel", width=10, anchor="e")
        self.rec_label.pack(side="right")

        # --- Transcript
        section = ttk.Frame(frame)
        section.pack(fill="x")
        ttk.Label(section, text="TRANSCRIPT", style="Section.TLabel").pack(side="left")
        ttk.Label(section, text="editable  ·  ⌘/Ctrl + Enter to suggest", style="Muted.TLabel").pack(side="right")
        self.transcript_box = ScrolledText(frame, height=6, wrap="word")
        style_text_box(self.transcript_box)
        self.transcript_box.pack(fill="both", expand=True, pady=(6, 14))

        # --- AI answer (read-only, so state="disabled") + Copy button
        section = ttk.Frame(frame)
        section.pack(fill="x")
        ttk.Label(section, text="AI ANSWER", style="Section.TLabel").pack(side="left")
        self.copy_btn = ttk.Button(section, text="Copy", style="Ghost.TButton", command=self.copy_answer)
        self.copy_btn.pack(side="right")
        self.answer_box = ScrolledText(frame, height=9, wrap="word")
        style_text_box(self.answer_box)
        self.answer_box.config(state="disabled")
        self.answer_box.pack(fill="both", expand=True, pady=(6, 14))

        # --- Latency cards: one small card per metric
        cards = ttk.Frame(frame)
        cards.pack(fill="x")
        self.metric_values = {}
        for i, (key, title) in enumerate(METRICS.items()):
            card = ttk.Frame(cards, style="Card.TFrame", padding=(12, 8))
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            cards.columnconfigure(i, weight=1)  # all four cards share the width equally
            ttk.Label(card, text=title, style="MetricName.TLabel").pack(anchor="w")
            self.metric_values[key] = ttk.Label(card, text="—", style="MetricValue.TLabel")
            self.metric_values[key].pack(anchor="w")

        # --- Status / error message line
        self.status_label = ttk.Label(frame, text="", style="Muted.TLabel", wraplength=760)
        self.status_label.pack(fill="x", pady=(12, 0))

        # Keyboard shortcut: Cmd+Enter (macOS) / Ctrl+Enter (Windows) = Suggest Answer
        for key in ("<Command-Return>", "<Control-Return>"):
            self.root.bind(key, self._suggest_shortcut)

    def _update_buttons(self):
        def enable(button, on):
            button.config(state="normal" if on else "disabled")

        enable(self.start_btn, self.model_ready and not self.recording)
        enable(self.stop_btn, self.recording and not self.stopping)
        enable(self.suggest_btn, not self.answering)
        enable(self.clear_btn, not self.answering)

        # The header pill shows the current state, worked out from the same flags.
        if not self.model_ready:
            mode = "loading"
        elif self.recording:
            mode = "finishing" if self.stopping else "recording"
        elif self.answering:
            mode = "answering"
        else:
            mode = "ready"
        text, color = MODES[mode]
        self.pill.config(text=f"●  {text}", foreground=COLORS[color])

    def _set_status(self, text, error=False):
        self.status_label.config(text=text, foreground=COLORS["red"] if error else COLORS["muted"])

    def _show_latency(self):
        for key, label in self.metric_values.items():
            label.config(text=f"{self.latency[key]:,.0f} ms" if key in self.latency else "—")

    def _update_live_widgets(self):
        """Called every poll: blink the REC timer and move the mic level meter."""
        if self.recorder.is_recording:
            elapsed = time.monotonic() - self.rec_started
            minutes, seconds = divmod(int(elapsed), 60)
            dot_on = int(elapsed * 2) % 2 == 0  # toggles every half second -> blinking dot
            self.rec_label.config(
                text=f"{'●' if dot_on else ' '} REC {minutes}:{seconds:02d}", foreground=COLORS["red"]
            )
            # Speech peaks are usually 0.05-0.3, so scale up to make the meter move visibly.
            self.level_bar["value"] = min(self.recorder.level * 4, 1.0)
        else:
            self.rec_label.config(text="")
            self.level_bar["value"] = 0

    def _append_answer(self, text):
        # The answer box is read-only ("disabled"), so unlock it briefly to insert text.
        self.answer_box.config(state="normal")
        self.answer_box.insert("end", text)
        self.answer_box.see("end")  # auto-scroll to the newest text
        self.answer_box.config(state="disabled")

    # ------------------------------------------------------------- button actions (UI thread)
    def start(self):
        try:
            self.recorder.start()
        except MicrophoneError as e:
            self._set_status("Microphone error - see details.", error=True)
            messagebox.showerror("Microphone error", str(e))
            return
        # A Tk "mark" is a bookmark in the text box. It remembers where this recording's
        # text begins, so the final transcript can replace just this recording's preview.
        # "left" gravity = the mark stays put when text is inserted right at it.
        self.transcript_box.mark_set("rec_start", "end-1c")
        self.transcript_box.mark_gravity("rec_start", "left")
        self.stop_event.clear()
        self.recording, self.stopping = True, False
        self.rec_started = time.monotonic()
        threading.Thread(target=self._transcription_worker, daemon=True).start()
        self._set_status("Recording... ask your question, then press Stop.")
        self._update_buttons()

    def stop(self):
        self.recorder.stop()   # 1) close the mic: no new audio after this point
        self.stop_event.set()  # 2) tell the worker to transcribe what's left, then exit
        self.stopping = True
        self._set_status("Finishing transcription...")
        self._update_buttons()

    def suggest_answer(self):
        question = self.transcript_box.get("1.0", "end").strip()
        if not question:
            self._set_status("The transcript is empty - record (or type) a question first.", error=True)
            return
        self.answer_box.config(state="normal")
        self.answer_box.delete("1.0", "end")
        self.answer_box.config(state="disabled")
        self.latency.pop("first_token", None)
        self.latency.pop("total", None)
        self._show_latency()
        self.answering = True
        self._update_buttons()
        self._set_status("Generating answer...")
        threading.Thread(target=self._answer_worker, args=(question,), daemon=True).start()

    def _suggest_shortcut(self, _event):
        if not self.answering:
            self.suggest_answer()
        return "break"  # stop Tk from also inserting a newline into the transcript box

    def copy_answer(self):
        text = self.answer_box.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("Answer copied to clipboard.")

    def clear(self):
        self.transcript_box.delete("1.0", "end")
        self.answer_box.config(state="normal")
        self.answer_box.delete("1.0", "end")
        self.answer_box.config(state="disabled")
        self.latency.clear()
        self._show_latency()
        if not self.recording:
            self._set_status("Cleared. Press Start and ask a question.")

    def _on_close(self):
        self.recorder.stop()
        self.root.destroy()  # background threads are daemon threads, so they end with the app

    # ------------------------------------------------------------- background threads
    # These functions NEVER touch widgets. They only put messages into self.ui_queue.

    def _load_models(self):
        try:
            transcriber.load_model()
            # Warm-up: the first transcribe() call also loads the VAD model, so do it now
            # on 1 s of silence instead of making the user's first chunk slow.
            transcriber.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
        except Exception as e:
            self.ui_queue.put(("fatal", f"Could not load the Whisper model:\n{e}"))
            return
        self.ui_queue.put(("model_ready", None))
        try:
            llm.warm_up()  # load the LLM into memory now, so the first answer starts faster
        except llm.LLMError as e:
            self.ui_queue.put(("warning", str(e)))

    def _transcription_worker(self):
        """Live preview: transcribe every CHUNK_SECONDS of audio while recording.
        Final: after Stop, transcribe the whole recording again in one pass."""
        chunk_samples = config.SAMPLE_RATE * config.CHUNK_SECONDS
        blocks, buffered = [], 0
        all_blocks = []  # every block of this recording, kept for the final pass
        try:
            # Keep going until Stop was pressed AND all recorded audio has been taken out.
            while not (self.stop_event.is_set() and self.recorder.queue.empty()):
                try:
                    block = self.recorder.queue.get(timeout=0.1)  # wait up to 100 ms for audio
                except queue.Empty:
                    continue
                blocks.append(block)
                all_blocks.append(block)
                buffered += len(block)
                if buffered >= chunk_samples:
                    self._transcribe_and_send("live", np.concatenate(blocks))
                    blocks, buffered = [], 0

            # Fixed 3 s chunks can cut a word in half ("and NoSQL" -> "and" | "no-school").
            # One pass over the full recording lets Whisper hear every word whole, so the
            # final transcript (the one sent to the LLM) is more accurate than the preview.
            if all_blocks:
                self._transcribe_and_send("final", np.concatenate(all_blocks))
        except Exception as e:
            self.ui_queue.put(("error", f"Transcription failed: {e}"))
        finally:
            self.ui_queue.put(("recording_done", None))

    def _transcribe_and_send(self, kind, audio):
        """kind is "live" (a 3 s preview chunk) or "final" (the whole recording)."""
        if looks_silent(audio):
            self.ui_queue.put(("warning", MIC_SILENT_MSG))
            return
        start = time.perf_counter()
        text = transcriber.transcribe(audio)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.ui_queue.put(("transcript_" + kind, (text, elapsed_ms)))

    def _answer_worker(self, question):
        start = time.perf_counter()
        got_first = False
        try:
            for piece in llm.stream_answer(question):
                if not got_first:
                    got_first = True
                    self.ui_queue.put(("first_token", (time.perf_counter() - start) * 1000))
                self.ui_queue.put(("answer_piece", piece))
            self.ui_queue.put(("answer_done", (time.perf_counter() - start) * 1000))
        except llm.LLMError as e:
            self.ui_queue.put(("answer_error", str(e)))
        except Exception as e:
            self.ui_queue.put(("answer_error", f"Unexpected error: {e}"))

    # ------------------------------------------------------------- UI-thread message handling
    def _poll_ui_queue(self):
        """Runs on the UI thread every POLL_MS: apply every pending message, then re-schedule."""
        while True:
            try:
                kind, data = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_message(kind, data)
        self._update_live_widgets()
        self.root.after(POLL_MS, self._poll_ui_queue)

    def _handle_message(self, kind, data):
        if kind == "model_ready":
            self.model_ready = True
            self._set_status("Ready. Press Start and ask a question.")
        elif kind == "transcript_live":
            text, elapsed_ms = data
            if text:
                # Add a space between chunks, except at the very beginning.
                separator = " " if self.transcript_box.get("1.0", "end").strip() else ""
                self.transcript_box.insert("end", separator + text)
                self.transcript_box.see("end")
            self.latency["stt"] = elapsed_ms
            self._show_latency()
        elif kind == "transcript_final":
            # Replace this recording's live preview (everything after the "rec_start"
            # mark set in start()) with the more accurate full-recording transcript.
            text, elapsed_ms = data
            earlier_text = self.transcript_box.get("1.0", "rec_start").strip()
            self.transcript_box.delete("rec_start", "end")
            if text:
                self.transcript_box.insert("end", (" " if earlier_text else "") + text)
                self.transcript_box.see("end")
            self.latency["final_stt"] = elapsed_ms
            self._show_latency()
        elif kind == "recording_done":
            self.recording = self.stopping = False
            if not self.answering:
                self._set_status("Stopped. Click 'Suggest Answer' to answer the transcript.")
        elif kind == "first_token":
            self.latency["first_token"] = data
            self._show_latency()
        elif kind == "answer_piece":
            self._append_answer(data)
        elif kind == "answer_done":
            self.latency["total"] = data
            self._show_latency()
            self.answering = False
            self._set_status("Answer complete.")
        elif kind == "answer_error":
            self.answering = False
            self._set_status("Could not get an answer - see details.", error=True)
            messagebox.showerror("LLM error", data)
        elif kind == "warning":
            self._set_status(data, error=True)
        elif kind == "error":
            self._set_status(data, error=True)
        elif kind == "fatal":
            self._set_status(data, error=True)
            messagebox.showerror("Startup error", data)
        self._update_buttons()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
