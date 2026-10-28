#!/bin/python3
"""
ASR Benchmark: Compare Whisper turbo, Qwen3-ASR-1.7B, Cohere Transcribe
on FLEURS dataset (EN, ZH, KR, VN).

Usage:
    python3 benchmark_asr.py                    # all models, all languages
    python3 benchmark_asr.py --models qwen      # only Qwen
    python3 benchmark_asr.py --langs en zh       # only EN and ZH
    python3 benchmark_asr.py --num-samples 20    # 20 samples per language
"""

import argparse
import gc
import time
import numpy as np
import torch
import jiwer
from datasets import load_dataset

# --- Dataset configs ---
FLEURS_MAP = {
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
    "vi": "vi_vn",
}

# --- Model loaders ---

def load_whisper(model_size="turbo"):
    import whisper
    print(f"  Loading Whisper {model_size}...")
    model = whisper.load_model(model_size, device="cuda")
    return model

def load_qwen(model_id="Qwen/Qwen3-ASR-1.7B"):
    from qwen_asr import Qwen3ASRModel
    print(f"  Loading {model_id}...")
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    return model

def load_cohere(model_id="CohereLabs/cohere-transcribe-03-2026"):
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    print(f"  Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    return {"model": model, "processor": processor}

# --- Transcribe functions ---

def transcribe_whisper(model, audio_np, sr):
    result = model.transcribe(audio_np, beam_size=5, fp16=True, language=None)
    return result["text"].strip()

def transcribe_qwen(model, audio_np, sr):
    results = model.transcribe(audio=(audio_np, sr), language=None)
    return results[0].text

def transcribe_cohere(model_dict, audio_np, sr):
    import torch as _torch
    model = model_dict["model"]
    processor = model_dict["processor"]
    inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt", language=None)
    audio_chunk_index = inputs.pop("audio_chunk_index", None)
    inputs.pop("length", None)
    inputs = {k: v.to(model.device, dtype=model.dtype) if isinstance(v, _torch.Tensor) else v
              for k, v in inputs.items()}
    with _torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    text = processor.decode(outputs, skip_special_tokens=True,
                            audio_chunk_index=audio_chunk_index, language=None)[0]
    return text

# --- VRAM tracking ---

def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

# --- Main benchmark ---

def load_fleurs_samples(lang_config, num_samples, split="validation"):
    ds = load_dataset("google/fleurs", lang_config, split=split, streaming=True)
    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        text = sample["transcription"].strip()
        samples.append({"audio": audio, "sr": sr, "text": text})
    return samples

def run_benchmark(models_to_test, langs, num_samples):
    results = {}

    for lang in langs:
        fleurs_config = FLEURS_MAP[lang]
        print(f"\n{'='*60}")
        print(f"Language: {lang.upper()} ({fleurs_config}) — {num_samples} samples")
        print(f"{'='*60}")

        print(f"  Loading dataset...")
        samples = load_fleurs_samples(fleurs_config, num_samples)
        print(f"  Loaded {len(samples)} samples")

        for model_name in models_to_test:
            print(f"\n  --- {model_name} ---")
            reset_vram()

            if model_name == "whisper":
                model = load_whisper()
                transcribe_fn = lambda a, s: transcribe_whisper(model, a, s)
            elif model_name == "qwen":
                model = load_qwen()
                transcribe_fn = lambda a, s: transcribe_qwen(model, a, s)
            elif model_name == "cohere":
                model = load_cohere()
                transcribe_fn = lambda a, s: transcribe_cohere(model, a, s)
            else:
                print(f"  Unknown model: {model_name}, skipping")
                continue

            hyps = []
            refs = []
            total_audio_sec = 0
            t_start = time.time()

            for i, sample in enumerate(samples):
                audio = sample["audio"]
                sr = sample["sr"]
                ref = sample["text"]
                total_audio_sec += len(audio) / sr

                t0 = time.time()
                try:
                    hyp = transcribe_fn(audio, sr)
                except Exception as e:
                    print(f"    [{i+1}] ERROR: {e}")
                    hyp = ""
                elapsed = time.time() - t0
                audio_dur = len(audio) / sr
                rtf = elapsed / audio_dur if audio_dur > 0 else 0

                hyps.append(hyp)
                refs.append(ref)

                if (i + 1) % 5 == 0 or i == 0:
                    print(f"    [{i+1}/{len(samples)}] RTFx={1/rtf:.1f}x  hyp={hyp[:60]}")

            total_time = time.time() - t_start
            avg_rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0

            # Compute WER (character-level for CJK, word-level for others)
            use_cer = lang in ("zh", "ko")
            metric_fn = jiwer.cer if use_cer else jiwer.wer
            metric_name = "CER" if use_cer else "WER"

            scores = []
            for hyp, ref in zip(hyps, refs):
                try:
                    s = metric_fn(ref, hyp)
                    scores.append(s)
                except Exception:
                    pass
            avg_score = np.mean(scores) * 100 if scores else -1

            vram = get_vram_mb()

            key = f"{model_name}_{lang}"
            results[key] = {
                "model": model_name,
                "lang": lang,
                "metric": metric_name,
                "score": avg_score,
                "rtfx": 1 / avg_rtf if avg_rtf > 0 else 0,
                "vram_mb": vram,
                "num_samples": len(samples),
                "total_audio_sec": total_audio_sec,
            }

            print(f"\n    RESULT: {metric_name}={avg_score:.2f}%  RTFx={1/avg_rtf:.1f}x  VRAM={vram:.0f}MB")

            # Free GPU memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

    return results

def print_summary(results):
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Model':<20} {'Lang':<6} {'Metric':<6} {'Score%':<10} {'RTFx':<10} {'VRAM(MB)':<10}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['model']:<20} {r['lang']:<6} {r['metric']:<6} {r['score']:<10.2f} {r['rtfx']:<10.1f} {r['vram_mb']:<10.0f}")

    # Average across languages
    print(f"\n{'Model':<20} {'Avg Score%':<10} {'Avg RTFx':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*10}")

    models = set(r["model"] for r in results.values())
    for model in sorted(models):
        model_results = [r for r in results.values() if r["model"] == model]
        avg_score = np.mean([r["score"] for r in model_results])
        avg_rtfx = np.mean([r["rtfx"] for r in model_results])
        print(f"{model:<20} {avg_score:<10.2f} {avg_rtfx:<10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Benchmark")
    parser.add_argument("--models", nargs="+", default=["whisper", "qwen", "cohere"],
                        choices=["whisper", "qwen", "cohere"])
    parser.add_argument("--langs", nargs="+", default=["en", "zh", "ko", "vi"],
                        choices=list(FLEURS_MAP.keys()))
    parser.add_argument("--num-samples", type=int, default=30)
    args = parser.parse_args()

    print(f"ASR Benchmark")
    print(f"Models: {args.models}")
    print(f"Languages: {args.langs}")
    print(f"Samples per language: {args.num_samples}")

    results = run_benchmark(args.models, args.langs, args.num_samples)
    print_summary(results)
