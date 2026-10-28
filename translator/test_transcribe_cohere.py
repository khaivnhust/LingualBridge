#!/bin/python3
import sys
import time
import numpy as np
import sounddevice as sd
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

# --- Config ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 0.015
SILENCE_TIMEOUT = 1.5
MAX_RECORD_SECONDS = 60
MIN_RECORD_SECONDS = 0.3

LANG_CODES = {
    "EN": "en", "ZH": "zh", "KR": "ko", "VN": "vi",
    "FR": "fr", "DE": "de", "IT": "it", "ES": "es",
    "PT": "pt", "EL": "el", "NL": "nl", "PL": "pl",
    "JA": "ja", "AR": "ar",
}

target = sys.argv[1] if len(sys.argv) > 1 else None
lang_code = LANG_CODES.get(target.upper(), None) if target else None

model_id = "CohereLabs/cohere-transcribe-03-2026"

print(f"Loading {model_id} ...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = CohereAsrForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
)
print("Model loaded.\n")

lang_msg = f"Language hint: {lang_code}" if lang_code else "Auto-detect language"
print(f"🎙️  Cohere Transcribe ready! ({lang_msg})")
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
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           language=lang_code)
        audio_chunk_index = inputs.pop("audio_chunk_index", None)
        inputs.pop("length", None)
        inputs = {k: v.to(model.device, dtype=model.dtype) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=256)
        text = processor.decode(
            outputs, skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=lang_code,
        )[0]
        elapsed = time.time() - t0

        print(f"   Transcribed in {elapsed:.2f}s (RTFx: {duration/elapsed:.1f}x)")
        print(f"👉 {text}\n")

        if "!quit" in text.lower():
            print("👋 Goodbye!")
            break

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
