import tkinter as tk
from tkinter import scrolledtext
import threading
import sounddevice as sd
import numpy as np
import whisper
from transformers import pipeline
from datetime import datetime

# ---------------- LOAD MODELS ----------------
speech_model = whisper.load_model("base")

# Safe summarizer loading
summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    framework="pt"
)

is_recording = False
recorded_text = ""

# ---------------- RECORD FUNCTION ----------------
def start_recording():
    global is_recording, recorded_text
    is_recording = True
    recorded_text = ""

    status_label.config(text="🎙 Recording...", fg="red")

    duration = 30
    sample_rate = 16000

    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='float32')
    sd.wait()

    status_label.config(text="🧠 Converting...", fg="orange")

    result = speech_model.transcribe(np.squeeze(audio))
    recorded_text = result["text"].strip()

    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, recorded_text)

    generate_summary()

    is_recording = False


# ---------------- SUMMARY FUNCTION ----------------
def generate_summary():
    global recorded_text

    if recorded_text.strip() == "":
        return

    status_label.config(text="✍ Generating Summary...", fg="blue")

    try:
        summary = summarizer(
            recorded_text,
            max_length=120,
            min_length=40,
            do_sample=False
        )

        summary_text = summary[0]["summary_text"]

        summary_box.delete("1.0", tk.END)
        summary_box.insert(tk.END, summary_text)

        status_label.config(text="✔ Summary Ready", fg="green")

    except Exception as e:
        summary_box.delete("1.0", tk.END)
        summary_box.insert(tk.END, f"Error: {str(e)}")
        status_label.config(text="❌ Summary Failed", fg="red")


# ---------------- THREAD WRAPPER ----------------
def threaded_record():
    threading.Thread(target=start_recording, daemon=True).start()


# ---------------- UI ----------------
window = tk.Tk()
window.title("🎤 Speech to Text + Summary")
window.geometry("800x650")
window.configure(bg="#0F172A")

title = tk.Label(window,
                 text="Speech to Text Summarizer",
                 font=("Segoe UI", 20, "bold"),
                 bg="#0F172A",
                 fg="#00FFF5")
title.pack(pady=15)

record_btn = tk.Button(window,
                       text="Start Recording (20s)",
                       font=("Segoe UI", 12, "bold"),
                       bg="#00ADB5",
                       fg="white",
                       command=threaded_record)
record_btn.pack(pady=10)

status_label = tk.Label(window,
                        text="Idle",
                        font=("Segoe UI", 11),
                        bg="#0F172A",
                        fg="white")
status_label.pack(pady=5)

# Full Text Box
tk.Label(window, text="Full Text:",
         bg="#0F172A", fg="white").pack()

text_box = scrolledtext.ScrolledText(window,
                                     wrap=tk.WORD,
                                     height=10,
                                     bg="#111827",
                                     fg="white")
text_box.pack(fill="both", padx=20, pady=10)

# Summary Box
tk.Label(window, text="Summary:",
         bg="#0F172A", fg="white").pack()

summary_box = scrolledtext.ScrolledText(window,
                                        wrap=tk.WORD,
                                        height=8,
                                        bg="#1E293B",
                                        fg="#00FFAA")
summary_box.pack(fill="both", padx=20, pady=10)

window.mainloop()