#!/bin/python3
"""
Speech-to-Speech Translation Pipeline
STT (Qwen3-ASR) → Translation (Hy-MT2 GGUF) → TTS (Piper)

Usage:
    python3 pipeline.py                    # Live mic, translate to EN
    python3 pipeline.py --target ko        # Translate to Korean
    python3 pipeline.py --target vi        # Translate to Vietnamese
    python3 pipeline.py --file input.wav   # Translate from audio file
    python3 pipeline.py --benchmark        # Run latency benchmark on test sentences
"""

import os
import sys
import time
import wave
import json
import struct
import argparse
import subprocess
import tempfile

# Return freed CUDA blocks to the driver so the GGUF model can reuse VRAM
# after the torch-based STT model is unloaded during model swapping.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# torch 2.14 routes some ops (bmm) to Triton kernels that need Python dev
# headers to JIT-compile; use the eager CUDA kernels instead.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np

# ── Constants ───────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 0.015
SILENCE_TIMEOUT = 1.5
MAX_RECORD_SECONDS = 30
MIN_RECORD_SECONDS = 0.3

PIPER_DIR = os.path.expanduser("~/AI-Translator/piper")
PIPER_BIN = os.path.join(PIPER_DIR, "install/piper")
ESPEAK_DATA = os.path.join(PIPER_DIR, "install/espeak-ng-data")
VOICES_DIR = os.path.join(PIPER_DIR, "voices")
OUTPUT_DIR = os.path.expanduser("~/AI-Translator/translator/pipeline_outputs")

# Hy-MT2 expects full language names in its prompt ("into English"), not ISO
# codes. STT reports a full name; the pipeline indexes by code, so both
# directions are needed.
LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "vi": "Vietnamese",
}
NAME_TO_CODE = {name.lower(): code for code, name in LANG_MAP.items()}


def as_code(lang):
    """Normalise a language given as either a code ("ko") or a name ("Korean")."""
    if not lang:
        return lang
    key = lang.strip().lower()
    return NAME_TO_CODE.get(key, key)

PIPER_MODELS = {
    "en": os.path.join(VOICES_DIR, "en_US-lessac-medium.onnx"),
    "zh": os.path.join(VOICES_DIR, "zh_CN-huayan-medium.onnx"),
    "ko": os.path.join(VOICES_DIR, "ko_KR-kss-medium.onnx"),
    "vi": os.path.join(VOICES_DIR, "vi_VN-vais1000-medium.onnx"),
}

GGUF_PATH = os.path.expanduser(
    "~/AI-Translator/translator/models/Hy-MT2-7B-Q6_K.gguf"
)

TEST_SENTENCES = {
    "en": [
        "Hello, how are you doing today?",
        "The weather is beautiful this morning.",
        "I need to go to the store to buy some groceries.",
        "Can you help me with this problem?",
        "The meeting has been rescheduled to tomorrow afternoon.",
    ],
    "zh": [
        "你好，你今天过得怎么样？",
        "今天早上的天气真好。",
        "我需要去商店买一些日用品。",
        "你能帮我解决这个问题吗？",
        "会议已经改到明天下午了。",
    ],
    "ko": [
        "안녕하세요, 오늘 어떻게 지내세요?",
        "오늘 아침 날씨가 정말 좋아요.",
        "필요한 물건을 사러 가게에 가야 해요.",
        "이 문제를 해결하는 데 도움을 주시겠어요?",
        "회의가 내일 오후로 변경되었습니다.",
    ],
    "vi": [
        "Xin chào, hôm nay bạn thế nào?",
        "Thời tiết hôm nay thật đẹp.",
        "Tôi cần đi cửa hàng mua vài thứ.",
        "Bạn có thể giúp tôi giải quyết vấn đề này không?",
        "Cuộc họp đã được đổi sang chiều mai.",
    ],
}

# ── Audio Recording ─────────────────────────────────────────────────────────────

def record_from_mic():
    """Record audio from microphone with VAD."""
    import sounddevice as sd

    chunks = []
    silence_start = None
    speaking_started = False

    print("🎤 Listening... (speak now)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
    ) as stream:
        while True:
            data, _ = stream.read(stream.blocksize)
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


def load_audio_file(path):
    """Load audio from WAV file, resample to 16kHz mono float32."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
    return audio


def save_wav(audio, path):
    """Save float32 audio to WAV."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())


# ── STT: Qwen3-ASR ────────────────────────────────────────────────────────────

STT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"


class _ASRResult:
    def __init__(self, text, language):
        self.text = text
        self.language = language


class Qwen3ASRNative:
    """Qwen3-ASR via native transformers>=5 (Qwen3ASRForConditionalGeneration).

    Mirrors the qwen_asr.Qwen3ASRModel.transcribe() interface used below.
    """

    def __init__(self, model_id, dtype, device, max_new_tokens=256):
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device
        )
        self.max_new_tokens = max_new_tokens

    def transcribe(self, audio, language=None):
        import torch

        wav, sr = audio
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz audio, got {sr}"
        inputs = self.processor.apply_transcription_request(
            audio=np.asarray(wav, dtype=np.float32), language=language
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = out[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(gen, return_format="parsed")[0]
        return [_ASRResult(parsed["transcription"], parsed["language"])]


def init_stt(device="cpu"):
    """Load Qwen3-ASR. device='cuda' uses bf16 on GPU; 'cpu' uses float32."""
    import torch

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading STT model (Qwen3-ASR-1.7B) on {device}...")
    model = Qwen3ASRNative(STT_MODEL_ID, dtype=dtype, device=device)
    print(f"  STT ready ({device}).\n")
    return model


def transcribe(stt_model, audio):
    """Transcribe audio → (text, language)."""
    results = stt_model.transcribe(
        audio=(audio, SAMPLE_RATE),
        language=None,
    )
    text = results[0].text
    lang = results[0].language
    return text, lang


# ── Translation: Hy-MT2 GGUF ─────────────────────────────────────────────────

def init_translator():
    """Load Hy-MT2-7B Q6_K GGUF model."""
    from llama_cpp import Llama

    print("Loading Translation model (Hy-MT2-7B Q6_K)...")
    model = Llama(
        model_path=GGUF_PATH,
        n_ctx=2048,
        n_gpu_layers=999,
        n_batch=512,
        use_kvcache=False,
        verbose=False,
    )
    print("  Translation model ready.\n")
    return model


import re

def clean_output(text):
    end_pat = r"<\|?im[_e]+_?end>?|<turn\|>|<\|?eot_id\|?>|<eos>|\|eos\|>|<\|?startoftext\|?>|<\|?extra_\d+\|?>"
    first_end = re.search(end_pat, text)
    if first_end:
        text = text[:first_end.start()]
    text = re.sub(r"<\|?im[_e]+_?start>?", "", text)
    text = re.sub(r"<\|turn>system", "", text)
    text = re.sub(r"<\|turn>user", "", text)
    text = re.sub(r"<\|turn>model", "", text)
    text = re.sub(r"<\|?begin_of_text\|?>", "", text)
    text = re.sub(r"<\|?start_header_id\|?>", "", text)
    text = re.sub(r"<\|?end_header_id\|?>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def translate(translator, text, source_lang, target_lang):
    """Translate text from source_lang to target_lang.

    Both arguments may be a code ("ko") or a full name ("Korean").
    """
    src_code, tgt_code = as_code(source_lang), as_code(target_lang)
    if src_code == tgt_code:
        return text

    prompt = (f"Translate the following text into {LANG_MAP[tgt_code]}. "
              f"Note that you should only output the translated result "
              f"without any additional explanation:\n\n{text}")

    output = translator(
        prompt,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.0,
    )
    result = output["choices"][0]["text"].strip()
    return clean_output(result)


def translate_all(translator, text, source_lang):
    """Translate text to all languages except source."""
    src_code = as_code(source_lang)
    results = {}
    for target in ["en", "zh", "ko", "vi"]:
        if target == src_code:
            results[target] = text
            continue
        results[target] = translate(translator, text, src_code, target)
    return results


# ── TTS: Piper ────────────────────────────────────────────────────────────────

def tts_piper(text, lang, output_path):
    """Synthesize speech with Piper TTS."""
    cmd = [
        PIPER_BIN,
        "--model", PIPER_MODELS[lang],
        "--output_file", output_path,
        "--espeak_data", ESPEAK_DATA,
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, input=text, capture_output=True, text=True, timeout=30
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    TTS error: {result.stderr[:100]}")
        return None

    # Parse RTF
    rtf = None
    for line in result.stderr.split("\n"):
        if "Real-time factor:" in line:
            try:
                rtf = float(line.split("Real-time factor:")[1].split("(")[0].strip())
            except (ValueError, IndexError):
                pass

    return {"elapsed": elapsed, "rtf": rtf, "output": output_path}


# ── Pipeline ────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.stt = None
        self.translator = None

    def load_stt(self):
        self.stt = init_stt()

    def load_translator(self):
        self.translator = init_translator()

    def run(self, audio, source_lang=None):
        """
        Full pipeline: audio → text → translations → speech.
        Returns dict with all results and timings.
        """
        result = {
            "stt_text": None,
            "stt_lang": None,
            "translations": {},
            "tts_files": {},
            "timings": {},
        }

        # ── STT ──
        t0 = time.time()
        text, detected_lang = transcribe(self.stt, audio)
        t_stt = time.time() - t0
        result["stt_text"] = text
        result["stt_lang"] = detected_lang
        result["timings"]["stt"] = t_stt
        result["timings"]["stt_audio_sec"] = len(audio) / SAMPLE_RATE

        print(f"  STT ({t_stt:.2f}s): [{detected_lang}] {text}")

        src = as_code(source_lang or detected_lang)

        # ── Translation ──
        t0 = time.time()
        translations = translate_all(self.translator, text, src)
        t_mt = time.time() - t0
        result["translations"] = translations
        result["timings"]["mt"] = t_mt

        print(f"  MT ({t_mt:.2f}s):")
        for lang, tr in translations.items():
            marker = " ← source" if lang == src else ""
            print(f"    {lang.upper()}: {tr}{marker}")

        # ── TTS for each translation ──
        result["timings"]["tts"] = {}
        t0_total = time.time()
        for lang, tr in translations.items():
            if lang == src:
                continue
            out_path = os.path.join(OUTPUT_DIR, f"output_{lang}.wav")
            t0 = time.time()
            info = tts_piper(tr, lang, out_path)
            t_tts = time.time() - t0
            if info:
                result["tts_files"][lang] = info["output"]
                result["timings"]["tts"][lang] = t_tts
                print(f"  TTS {lang.upper()} ({t_tts:.2f}s, RTF={info['rtf']:.3f}x): {info['output']}")

        result["timings"]["tts_total"] = time.time() - t0_total
        result["timings"]["total"] = sum([
            result["timings"]["stt"],
            result["timings"]["mt"],
            result["timings"]["tts_total"],
        ])

        return result


def run_live(args):
    """Live mic translation."""
    import torch
    pipe = Pipeline()
    pipe.load_stt()
    if not args.no_translate:
        pipe.load_translator()

    target = args.target
    print(f"Target language: {target.upper()}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    while True:
        try:
            audio = record_from_mic()
            if audio is None:
                print("   (no speech detected)\n")
                continue

            duration = len(audio) / SAMPLE_RATE
            print(f"   Captured {duration:.1f}s")

            result = pipe.run(audio, source_lang=args.source)

            total = result["timings"]["total"]
            rtf_overall = total / duration if duration > 0 else 0
            print(f"\n  ⏱  Total: {total:.2f}s (RTF={rtf_overall:.3f}x, {1/rtf_overall:.1f}x realtime)")
            print(f"  📁 Output files in: {OUTPUT_DIR}")
            print()

            if "!quit" in (result["stt_text"] or "").lower():
                print("Goodbye!")
                break

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def run_benchmark(args):
    """Benchmark pipeline on test sentences."""
    import torch
    pipe = Pipeline()
    pipe.load_stt()
    pipe.load_translator()

    lang = args.lang or "en"
    sentences = TEST_SENTENCES.get(lang, TEST_SENTENCES["en"])

    print(f"\nBenchmarking pipeline: {lang.upper()} → all other languages")
    print(f"Sentence count: {len(sentences)}")
    print(f"{'='*60}\n")

    total_stt = 0
    total_mt = 0
    total_tts = 0
    total_wall = 0

    for i, sentence in enumerate(sentences):
        print(f"[{i+1}/{len(sentences)}] {sentence}")
        print(f"  {'─'*50}")

        # Synthesize the source sentence as input audio
        src_audio_path = os.path.join(OUTPUT_DIR, f"bench_input_{i:03d}.wav")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tts_piper(sentence, lang, src_audio_path)

        # Load synthesized audio
        audio = load_audio_file(src_audio_path)

        # Run pipeline
        t0 = time.time()
        result = pipe.run(audio, source_lang=lang)
        wall = time.time() - t0

        total_stt += result["timings"]["stt"]
        total_mt += result["timings"]["mt"]
        total_tts += result["timings"]["tts_total"]
        total_wall += wall

        print(f"  Total: {wall:.2f}s")
        print()

    n = len(sentences)
    print(f"{'='*60}")
    print(f"SUMMARY ({n} sentences)")
    print(f"  STT avg:  {total_stt/n:.2f}s")
    print(f"  MT avg:   {total_mt/n:.2f}s")
    print(f"  TTS avg:  {total_tts/n:.2f}s")
    print(f"  Wall avg: {total_wall/n:.2f}s")
    print(f"  Total:    {total_wall:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech-to-Speech Translation Pipeline")
    parser.add_argument("--target", default="en", choices=["en", "zh", "ko", "vi"],
                        help="Target language (default: en)")
    parser.add_argument("--source", default=None, choices=["en", "zh", "ko", "vi"],
                        help="Source language (auto-detect if not set)")
    parser.add_argument("--file", default=None, help="Input audio file (WAV)")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation (STT + TTS only)")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--lang", default="en", help="Language for benchmark")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args)
    else:
        run_live(args)
