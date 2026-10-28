#!/bin/python3
import sys
import time
import numpy as np
import sounddevice as sd
import whisper

# --- Config ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 0.015
SILENCE_TIMEOUT = 1.5
MAX_RECORD_SECONDS = 60
MIN_RECORD_SECONDS = 0.3

model_size = sys.argv[1] if len(sys.argv) > 1 else "turbo"

print(f"Loading Whisper model: {model_size} ...")
model = whisper.load_model(model_size, device="cuda")
print("Model loaded.\n")

print("🎙️  Whisper transcriber ready!")
print("   Speak into your microphone. Say '!quit' or press Ctrl+C to exit.\n")


def record_utterance():
    chunks = []
    silence_start = None
    speaking_started = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
    ) as stream:
        while True:
            data, overflowed = stream.read(stream.blocksize)
            audio_chunk = data[:, 0]
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

            if rms > SILENCE_THRESHOLD:
                speaking_started = True
                silence_start = None
                chunks.append(audio_chunk.copy())
            elif speaking_started:
                chunks.append(audio_chunk.copy())
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_TIMEOUT:
                    break

            total_duration = len(chunks) * CHUNK_DURATION
            if total_duration > MAX_RECORD_SECONDS:
                break

    if not chunks:
        return None

    audio = np.concatenate(chunks)
    total_sec = len(audio) / SAMPLE_RATE

    if total_sec < MIN_RECORD_SECONDS:
        return None

    return audio


while True:
    try:
        print("🎤 Listening... (speak now)")
        audio = record_utterance()

        if audio is None:
            print("   (no speech detected)\n")
            continue

        duration = len(audio) / SAMPLE_RATE
        print(f"   Captured {duration:.1f}s. Transcribing...")

        t0 = time.time()
        result = model.transcribe(
            audio,
            beam_size=5,
            language=None,
            fp16=True,
        )
        text = result["text"].strip()
        lang = result.get("language", "?")
        elapsed = time.time() - t0

        print(f"   Detected language: {lang}")
        print(f"   Transcribed in {elapsed:.2f}s")
        print(f"👉 {text}\n")

        if "!quit" in text.lower():
            print("👋 Goodbye!")
            break

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
