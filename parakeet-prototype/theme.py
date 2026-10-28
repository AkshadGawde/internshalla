"""
theme.py - colours, fonts and widget styles (how the app LOOKS).

Like config.py does for settings, this keeps every visual choice in one place,
so app.py only deals with behaviour.

Why the "clam" theme? On macOS, Tkinter's default "aqua" theme draws native
buttons and ignores custom colours. "clam" is another theme built into Tkinter
that lets us set the colour of every widget.
"""

from tkinter import ttk

FONT = "Helvetica"  # Tk falls back to a similar font if this one is missing (e.g. Arial on Windows)

COLORS = {
    "bg": "#0f172a",      # window background (dark navy)
    "panel": "#1e293b",   # text boxes and cards
    "border": "#334155",
    "text": "#e2e8f0",
    "muted": "#94a3b8",   # secondary text
    "accent": "#6366f1",  # indigo - Suggest Answer
    "green": "#22c55e",
    "red": "#ef4444",
    "amber": "#f59e0b",
    "blue": "#38bdf8",
}


def apply_theme(root):
    """Set up all ttk styles. Widgets then pick a look with style="Name.TWidget"."""
    c = COLORS
    root.configure(bg=c["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")

    # "." is the default for every ttk widget.
    style.configure(".", background=c["bg"], foreground=c["text"], font=(FONT, 13))
    style.configure("TFrame", background=c["bg"])
    style.configure("Card.TFrame", background=c["panel"])
    style.configure("TLabel", background=c["bg"], foreground=c["text"])
    style.configure("Title.TLabel", font=(FONT, 20, "bold"))
    style.configure("Muted.TLabel", foreground=c["muted"], font=(FONT, 12))
    style.configure("Section.TLabel", foreground=c["muted"], font=(FONT, 11, "bold"))
    style.configure("Pill.TLabel", background=c["panel"], font=(FONT, 12, "bold"), padding=(12, 4))
    style.configure("Rec.TLabel", font=(FONT, 13, "bold"))
    style.configure("MetricName.TLabel", background=c["panel"], foreground=c["muted"], font=(FONT, 11))
    style.configure("MetricValue.TLabel", background=c["panel"], font=(FONT, 16, "bold"))

    # Buttons: a colour for normal, a darker one on hover ("active"), grey when disabled.
    def button(name, color, hover, text_color="white"):
        style.configure(
            name, background=color, foreground=text_color, font=(FONT, 13, "bold"),
            padding=(16, 8), borderwidth=0, relief="flat",
            bordercolor=color, lightcolor=color, darkcolor=color, focuscolor=color,
        )
        style.map(
            name,
            background=[("disabled", c["panel"]), ("active", hover)],
            foreground=[("disabled", c["border"])],
            bordercolor=[("disabled", c["panel"]), ("active", hover)],
            lightcolor=[("disabled", c["panel"]), ("active", hover)],
            darkcolor=[("disabled", c["panel"]), ("active", hover)],
        )

    button("Start.TButton", c["green"], "#16a34a")
    button("Stop.TButton", c["red"], "#dc2626")
    button("Accent.TButton", c["accent"], "#4f46e5")
    button("Ghost.TButton", c["panel"], c["border"], text_color=c["text"])

    # Mic level meter
    style.configure(
        "Level.Horizontal.TProgressbar", troughcolor=c["panel"], background=c["green"],
        bordercolor=c["panel"], lightcolor=c["green"], darkcolor=c["green"], thickness=8,
    )

    style.configure("TCheckbutton", foreground=c["muted"], font=(FONT, 12),
                    indicatorbackground=c["panel"], indicatorforeground=c["text"])
    style.map("TCheckbutton", background=[("active", c["bg"])],
              indicatorbackground=[("selected", c["accent"])])


def style_text_box(box):
    """Text boxes are classic Tk widgets (not ttk), so they are styled directly."""
    c = COLORS
    box.configure(
        bg=c["panel"], fg=c["text"], insertbackground=c["text"],  # insertbackground = cursor colour
        selectbackground=c["accent"], relief="flat", borderwidth=0,
        highlightthickness=1, highlightbackground=c["border"], highlightcolor=c["accent"],
        padx=12, pady=10, font=(FONT, 15), spacing2=3,  # spacing2 = extra space between wrapped lines
    )
