#!/bin/python3
"""
Pipeline Test — Comprehensive latency, accuracy, memory, and RTF measurement.

Measures for each source language (EN/ZH/KO/VI) → all 3 target languages:
  - Model load time (STT, MT)
  - VRAM/RAM usage
  - STT latency, RTFx, CER/WER vs ground truth
  - MT per-target latency
  - TTS per-target latency + RTF
  - End-to-end wall time
  - Translation keyword accuracy

Usage:
    python3 pipeline_test.py                     # Run all languages
    python3 pipeline_test.py --lang en           # Single source language
    python3 pipeline_test.py --lang en,zh,ko     # Multiple
    python3 pipeline_test.py --num 3             # Limit sentences per lang
    python3 pipeline_test.py --no-accuracy       # Skip accuracy scoring (faster)

Output:
    pipeline_test_results.json  — full per-sentence report
    pipeline_test_summary.csv   — summary table
"""

import os
import re
import sys
import json
import time
import csv
import unicodedata
import argparse
import subprocess
from statistics import mean, median

# Return freed CUDA blocks to the driver so llama.cpp can reuse VRAM after
# the STT (torch) model is unloaded (default allocator retains a large pool).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

import pipeline as P

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
JSON_PATH = os.path.join(RESULTS_DIR, "pipeline_test_results.json")
CSV_PATH = os.path.join(RESULTS_DIR, "pipeline_test_summary.csv")

ALL_LANGS = ["en", "zh", "ko", "vi"]

# ── Reference translations for keyword accuracy scoring ───────────────────────
# For each source sentence, the expected meaning in every target language.
# Keywords chosen to be robust to rephrasing.
REFERENCE_KEYWORDS = {
    "en": {
        "Hello, how are you doing today?": {
            "zh": ["你好", "今天", "怎么样", "你好吗"],
            "ko": ["안녕하세요", "오늘", "지내"],
            "vi": ["chào", "hôm nay", "bạn", "khỏe", "nào"],
        },
        "The weather is beautiful this morning.": {
            "zh": ["天气", "好", "今天", "早上"],
            "ko": ["날씨", "아침", "좋아", "오늘"],
            "vi": ["thời tiết", "sáng", "đẹp", "hôm nay"],
        },
        "I need to go to the store to buy some groceries.": {
            "zh": ["商店", "买", "日用品", "去"],
            "ko": ["가게", "사", "식료품", "가야"],
            "vi": ["cửa hàng", "mua", "thực phẩm", "đi"],
        },
        "Can you help me with this problem?": {
            "zh": ["帮助", "解决", "问题"],
            "ko": ["도와", "문제", "해결"],
            "vi": ["giúp", "vấn đề", "này"],
        },
        "The meeting has been rescheduled to tomorrow afternoon.": {
            "zh": ["会议", "明天", "下午", "改"],
            "ko": ["회의", "내일", "오후", "변경"],
            "vi": ["cuộc họp", "mai", "chiều", "đổi", "dời"],
        },
    },
    "zh": {
        "你好，你今天过得怎么样？": {
            "en": ["hello", "today", "how"],
            "ko": ["안녕하세요", "오늘", "지내"],
            "vi": ["chào", "hôm nay", "nào"],
        },
        "今天早上的天气真好。": {
            "en": ["weather", "beautiful", "morning", "today"],
            "ko": ["날씨", "아침", "좋", "오늘"],
            "vi": ["thời tiết", "sáng", "đẹp", "hôm nay"],
        },
        "我需要去商店买一些日用品。": {
            "en": ["store", "buy", "groceries", "need"],
            "ko": ["가게", "사", "필요", "일용품"],
            "vi": ["cửa hàng", "mua", "cần", "hàng"],
        },
        "你能帮我解决这个问题吗？": {
            "en": ["help", "solve", "problem", "can"],
            "ko": ["도와", "문제", "해결", "수"],
            "vi": ["giúp", "vấn đề", "giải quyết", "không"],
        },
        "会议已经改到明天下午了。": {
            "en": ["meeting", "tomorrow", "afternoon", "rescheduled"],
            "ko": ["회의", "내일", "오후", "변경"],
            "vi": ["cuộc họp", "mai", "chiều", "đổi"],
        },
    },
    "ko": {
        "안녕하세요, 오늘 어떻게 지내세요?": {
            "en": ["hello", "today", "how", "doing"],
            "zh": ["你好", "今天", "怎么样"],
            "vi": ["chào", "hôm nay", "khỏe"],
        },
        "오늘 아침 날씨가 정말 좋아요.": {
            "en": ["weather", "morning", "beautiful", "today"],
            "zh": ["天气", "早上", "好", "今天"],
            "vi": ["thời tiết", "sáng", "đẹp"],
        },
        "필요한 물건을 사러 가게에 가야 해요.": {
            "en": ["store", "buy", "need", "things"],
            "zh": ["商店", "买", "东西", "去"],
            "vi": ["cửa hàng", "mua", "cần", "đi"],
        },
        "이 문제를 해결하는 데 도움을 주시겠어요?": {
            "en": ["help", "problem", "solve", "can"],
            "zh": ["帮助", "问题", "解决"],
            "vi": ["giúp", "vấn đề", "giải quyết"],
        },
        "회의가 내일 오후로 변경되었습니다.": {
            "en": ["meeting", "tomorrow", "afternoon", "changed"],
            "zh": ["会议", "明天", "下午", "改"],
            "vi": ["cuộc họp", "mai", "chiều", "đổi"],
        },
    },
    "vi": {
        "Xin chào, hôm nay bạn thế nào?": {
            "en": ["hello", "today", "how", "doing"],
            "zh": ["你好", "今天", "怎么样"],
            "ko": ["안녕하세요", "오늘", "지내"],
        },
        "Thời tiết hôm nay thật đẹp.": {
            "en": ["weather", "today", "beautiful"],
            "zh": ["天气", "今天", "好"],
            "ko": ["날씨", "오늘", "좋아"],
        },
        "Tôi cần đi cửa hàng mua vài thứ.": {
            "en": ["need", "store", "buy", "things"],
            "zh": ["商店", "买", "需要"],
            "ko": ["가게", "사", "필요"],
        },
        "Bạn có thể giúp tôi giải quyết vấn đề này không?": {
            "en": ["help", "solve", "problem", "can"],
            "zh": ["帮助", "解决", "问题"],
            "ko": ["도와", "해결", "문제"],
        },
        "Cuộc họp đã được đổi sang chiều mai.": {
            "en": ["meeting", "afternoon", "tomorrow", "changed"],
            "zh": ["会议", "下午", "明天", "改"],
            "ko": ["회의", "오후", "내일", "변경"],
        },
    },
}

LANG_HINT = {
    "en": "English", "zh": "Chinese", "ko": "Korean", "vi": "Vietnamese",
}


# ── Accuracy helpers (from base_model_accuracy_test.py) ───────────────────────

def strip_diacritics(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def cer_score(reference, hypothesis):
    """Character error rate for CJK languages."""
    import jiwer
    return jiwer.cer(reference, hypothesis)


def wer_score(reference, hypothesis):
    """Word error rate for space-separated languages."""
    import jiwer
    return jiwer.wer(reference, hypothesis)


def stt_error_rate(ref_text, hyp_text, lang):
    """Return CER or WER depending on language."""
    if lang in ("zh", "ko"):
        return cer_score(ref_text, hyp_text)
    return wer_score(ref_text, hyp_text)


def keyword_accuracy(output, keywords):
    """Fraction of reference keywords found in output (0-100)."""
    if not keywords:
        return 100.0
    out = output.lower()
    out_stripped = strip_diacritics(out)
    found = 0
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in out or kw_l in out_stripped:
            found += 1
    return found / len(keywords) * 100


# ── Memory / GPU measurement ──────────────────────────────────────────────────

def gpu_memory_mb():
    """Return dict of per-GPU used memory in MB, or None if no NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        results = {}
        for i, line in enumerate(out.stdout.strip().split("\n")):
            used, total = line.split(",")
            results[f"gpu{i}"] = {
                "used_mb": int(used.strip()),
                "total_mb": int(total.strip()),
            }
        return results
    except Exception:
        return None


def ram_usage_mb():
    """Return (used_mb, total_mb) for host RAM."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":")
                info[k] = int(v.strip().split()[0])  # kB
        total = info["MemTotal"] // 1024
        available = info["MemAvailable"] // 1024
        return total - available, total
    except Exception:
        return 0, 0


# ── Model loading / unload / footprint measurement ────────────────────────────

def init_stt(device="cpu"):
    """Load STT model. device='cuda' uses bf16 on GPU; 'cpu' uses float32."""
    return P.init_stt(device)


def unload_models():
    """Free references and release cached CUDA memory."""
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1.0)  # let the driver release memory


def gpu_used_mb():
    m = gpu_memory_mb()
    if m is None:
        return None
    return m["gpu0"]["used_mb"]


def measure_footprints():
    """
    Load each model to VRAM one at a time, measure footprint, unload, verify
    memory is released. Returns stats dict:
        stats["stt"] = {load_s, gpu_mb, ram_mb, freed_mb}
        stats["mt"]  = {load_s, gpu_mb, freed_mb}
    """
    print("=" * 60)
    print("VRAM FOOTPRINT MEASUREMENT (one model at a time)")
    print("=" * 60)

    gpu_total = gpu_memory_mb()["gpu0"]["total_mb"] if gpu_memory_mb() else None
    gpu_base = gpu_used_mb() or 0
    ram_base, ram_total = ram_usage_mb()

    stats = {
        "gpu_total_mb": gpu_total,
        "ram_total_mb": ram_total,
        "gpu_base_mb": gpu_base,
        "ram_base_mb": ram_base,
    }

    # ── 1. STT on GPU ──
    t0 = time.time()
    stt = init_stt("cuda")
    stt_load = time.time() - t0

    gpu_after = gpu_used_mb()
    ram_after, _ = ram_usage_mb()
    stt_gpu = (gpu_after - gpu_base) if gpu_after is not None else None
    stt_ram = ram_after - ram_base

    stats["stt"] = {
        "load_s": round(stt_load, 2),
        "gpu_mb": stt_gpu,
        "ram_mb": stt_ram,
    }
    print(f"  STT @VRAM: load={stt_load:.1f}s  GPU +{stt_gpu} MB  "
          f"(→ {gpu_after}/{gpu_total} MB)  RAM +{stt_ram} MB")

    # unload STT, verify freed
    del stt
    unload_models()
    gpu_freed = gpu_used_mb()
    freed = (gpu_base - gpu_freed) if (gpu_freed is not None and gpu_base) else None
    stats["stt"]["freed_mb"] = freed
    print(f"  unload → GPU used {gpu_freed} MB (freed {freed} MB)")

    # ── 2. MT (SLM) on GPU ──
    gpu_mid = gpu_used_mb()
    ram_mid, _ = ram_usage_mb()

    t0 = time.time()
    mt = P.init_translator()
    mt_load = time.time() - t0

    gpu_after = gpu_used_mb()
    mt_gpu = (gpu_after - gpu_mid) if gpu_after is not None else None
    mt_ram = (ram_used := ram_usage_mb()[0]) - ram_mid

    stats["mt"] = {
        "load_s": round(mt_load, 2),
        "gpu_mb": mt_gpu,
        "ram_mb": mt_ram,
    }
    print(f"  MT   @VRAM: load={mt_load:.1f}s  GPU +{mt_gpu} MB  "
          f"(→ {gpu_after}/{gpu_total} MB)  RAM +{mt_ram} MB")

    # unload MT, verify freed
    del mt
    unload_models()
    gpu_freed = gpu_used_mb()
    freed = (gpu_mid - gpu_freed) if (gpu_freed is not None and gpu_mid) else None
    stats["mt"]["freed_mb"] = freed
    print(f"  unload → GPU used {gpu_freed} MB (freed {freed} MB)")

    # peak concurrent VRAM if both were resident (informational)
    if stt_gpu is not None and mt_gpu is not None:
        stats["peak_both_mb"] = gpu_base + stt_gpu + mt_gpu
        fits = stats["peak_both_mb"] <= gpu_total
        stats["peak_both_fits"] = fits
        print(f"\n  If both loaded: peak ~{stats['peak_both_mb']} MB of {gpu_total} MB "
              f"→ {'FITS' if fits else 'EXCEEDS VRAM (must swap)'}")
    print()

    return stats


class ModelManager:
    """
    Manages STT / MT lifecycle. In swap mode only one model is resident on the
    GPU at a time (load → infer → unload → load other). In keep-loaded mode
    both are resident (STT CPU, MT GPU) for the whole run.
    """

    def __init__(self, swap=True):
        self.swap = swap
        self.stt = None
        self.mt = None
        self.stt_load_count = 0
        self.mt_load_count = 0

    def ensure_stt(self):
        """Return a ready STT model (loads it if needed)."""
        if self.stt is None:
            t0 = time.time()
            self.stt = init_stt("cuda" if self.swap else "cpu")
            self.stt_load_count += 1
            self.stt_load_s = time.time() - t0
        return self.stt

    def ensure_mt(self):
        """Return a ready MT model (loads it if needed)."""
        if self.mt is None:
            t0 = time.time()
            self.mt = P.init_translator()
            self.mt_load_count += 1
            self.mt_load_s = time.time() - t0
        return self.mt

    def swap_stt(self):
        """Make STT resident: unload MT if swapping, then ensure STT."""
        if self.swap:
            self.release_mt()
        return self.ensure_stt()

    def swap_mt(self):
        """Make MT resident: unload STT if swapping, then ensure MT."""
        if self.swap:
            self.release_stt()
        return self.ensure_mt()

    def release_stt(self):
        if self.stt is not None:
            del self.stt
            self.stt = None
            unload_models()

    def release_mt(self):
        if self.mt is not None:
            del self.mt
            self.mt = None
            unload_models()

    def peak_vram_during_test(self):
        """Estimate peak concurrent GPU memory during the test runs."""
        if self.swap:
            return gpu_used_mb()  # only one model resident
        return gpu_used_mb()


def run_language(mgr, lang, num_sentences, do_accuracy):
    """Run pipeline test for a single source language."""
    sentences = P.TEST_SENTENCES.get(lang, [])[:num_sentences]
    refs = REFERENCE_KEYWORDS.get(lang, {})
    targets = [l for l in ALL_LANGS if l != lang]

    results = []
    stt_times, mt_times, tts_times, wall_times = [], [], [], []
    stt_errors, kw_scores = [], []
    stt_rtfx = []
    stt_loads, mt_loads = [], []

    print("=" * 60)
    print(f"LANGUAGE: {lang.upper()} ({LANG_HINT.get(lang, lang)}) → {[t.upper() for t in targets]}")
    print(f"SENTENCES: {len(sentences)}")
    print("=" * 60)

    for i, sentence in enumerate(sentences):
        rec = {
            "sentence": sentence,
            "stt_hyp": None,
            "stt_error": None,
            "stt_rtfx": None,
            "mt": {},
            "tts": {},
            "timings": {},
            "keyword_accuracy": {},
        }

        # 1. Synthesize input audio via Piper
        src_audio_path = os.path.join(P.OUTPUT_DIR, f"test_{lang}_{i:03d}_input.wav")
        os.makedirs(P.OUTPUT_DIR, exist_ok=True)
        tts_info = P.tts_piper(sentence, lang, src_audio_path)

        # Get audio duration
        import wave
        with wave.open(src_audio_path, "rb") as wf:
            audio_dur = wf.getnframes() / wf.getframerate()
        rec["audio_sec"] = round(audio_dur, 2)

        audio = P.load_audio_file(src_audio_path)

        # 2. STT (swap in STT model, releasing MT from VRAM if swapping)
        stt = mgr.swap_stt()
        if mgr.stt_load_count and "stt_load_s" not in rec:
            rec["timings"]["stt_load"] = round(mgr.stt_load_s, 3)
            stt_loads.append(mgr.stt_load_s)
            print(f"    (STT model loaded: {mgr.stt_load_s:.1f}s)")

        t0 = time.time()
        text, detected = P.transcribe(stt, audio)
        stt_t = time.time() - t0
        rec["stt_hyp"] = text
        rec["stt_lang_detected"] = detected
        rec["timings"]["stt"] = round(stt_t, 3)
        stt_times.append(stt_t)
        rtfx = audio_dur / stt_t if stt_t > 0 else 0
        rec["stt_rtfx"] = round(rtfx, 2)
        stt_rtfx.append(rtfx)

        if do_accuracy:
            err = stt_error_rate(sentence, text, lang)
            rec["stt_error"] = round(err, 4)
            stt_errors.append(err)

        print(f"\n  [{i+1}/{len(sentences)}] {sentence}")
        print(f"    STT  ({stt_t:.2f}s, RTFx={rtfx:.1f}x): [{detected}] {text}")
        if do_accuracy:
            print(f"         error: {err*100:.1f}%")

        # 3. Translation (swap in MT model, releasing STT from VRAM) + TTS
        mt = mgr.swap_mt()
        if mgr.mt_load_count and "mt_load_s" not in rec:
            rec["timings"]["mt_load"] = round(mgr.mt_load_s, 3)
            mt_loads.append(mgr.mt_load_s)
            print(f"    (MT model loaded: {mgr.mt_load_s:.1f}s)")

        t0 = time.time()
        translations = P.translate_all(mt, text, lang)
        mt_t = time.time() - t0
        rec["timings"]["mt_total"] = round(mt_t, 3)
        rec["timings"]["mt_per_target"] = {}
        mt_times.append(mt_t)

        src = lang
        for target in targets:
            tr_text = translations[target]
            rec["mt"][target] = tr_text

            # per-target MT time estimate (sequential, share the batch — report aggregate)
            if do_accuracy:
                kw = keyword_accuracy(tr_text, refs.get(sentence, {}).get(target, []))
                rec["keyword_accuracy"][target] = round(kw, 1)
                kw_scores.append(kw)

            print(f"    MT→{target.upper()}: {tr_text}")

        # 4. TTS for each target
        t0_total = time.time()
        rec["timings"]["tts"] = {}
        for target in targets:
            out_path = os.path.join(P.OUTPUT_DIR, f"test_{lang}_{i:03d}_{target}.wav")
            t0 = time.time()
            info = P.tts_piper(translations[target], target, out_path)
            t_tts = time.time() - t0
            rec["timings"]["tts"][target] = round(t_tts, 3)
            if info and info["rtf"]:
                rec["tts"][target] = {
                    "rtf": round(info["rtf"], 4),
                    "file": out_path,
                }
                print(f"    TTS→{target.upper()} ({t_tts:.2f}s, RTF={info['rtf']:.3f}x)")
            else:
                rec["tts"][target] = {"rtf": None, "file": out_path}
                print(f"    TTS→{target.upper()} ({t_tts:.2f}s)")
        rec["timings"]["tts_total"] = round(time.time() - t0_total, 3)
        tts_times.append(rec["timings"]["tts_total"])

        rec["timings"]["total"] = round(stt_t + mt_t + rec["timings"]["tts_total"], 3)
        wall_times.append(rec["timings"]["total"])
        print(f"    ── TOTAL: {rec['timings']['total']:.2f}s")

        results.append(rec)

    return build_summary(
        lang, targets, results, stt_times, mt_times, tts_times, wall_times,
        stt_errors, kw_scores, stt_rtfx, stt_loads, mt_loads,
    )


def build_summary(lang, targets, results, stt_times, mt_times, tts_times,
                  wall_times, stt_errors, kw_scores, stt_rtfx,
                  stt_loads=None, mt_loads=None):
    """Compute the per-language summary dict from collected metrics."""
    stt_loads = stt_loads or []
    mt_loads = mt_loads or []
    # ── Per-language summary ──
    summary = {
        "lang": lang,
        "n_sentences": len(results),
        "stt_avg_s": round(mean(stt_times), 3) if stt_times else None,
        "stt_median_s": round(median(stt_times), 3) if stt_times else None,
        "stt_avg_rtfx": round(mean(stt_rtfx), 2) if stt_rtfx else None,
        "mt_avg_s": round(mean(mt_times), 3) if mt_times else None,
        "tts_avg_s": round(mean(tts_times), 3) if tts_times else None,
        "wall_avg_s": round(mean(wall_times), 3) if wall_times else None,
        "stt_avg_error": round(mean(stt_errors), 4) if stt_errors else None,
        "kw_avg": round(mean(kw_scores), 1) if kw_scores else None,
        "stt_model_loads": stt_loads,
        "mt_model_loads": mt_loads,
        "stt_loads_total_s": round(sum(stt_loads), 2) if stt_loads else 0,
        "mt_loads_total_s": round(sum(mt_loads), 2) if mt_loads else 0,
    }

    if stt_errors:
        summary["stt_avg_error_pct"] = round(summary["stt_avg_error"] * 100, 2)
    if kw_scores:
        # per-target kw averages
        by_target = {t: [] for t in targets}
        for rec in results:
            for t, v in rec.get("keyword_accuracy", {}).items():
                by_target[t].append(v)
        summary["kw_per_target"] = {
            t: round(mean(v), 1) for t, v in by_target.items() if v
        }

    print(f"\n  ── {lang.upper()} SUMMARY ──")
    for k, v in summary.items():
        if v is not None:
            print(f"    {k}: {v}")
    print()

    return {"summary": summary, "sentences": results}


# ── Subprocess stage workers (model isolation for VRAM swapping) ─────────────

def run_stage_process(stage, payload, label):
    """Run a model stage in a fresh subprocess so the OS fully reclaims VRAM
    when the stage exits (required because llama.cpp cannot be re-loaded in the
    same process after torch has used VRAM)."""
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="pipeline_stage_")
    in_path = os.path.join(tmpdir, "in.json")
    out_path = os.path.join(tmpdir, "out.json")
    with open(in_path, "w") as f:
        json.dump(payload, f)

    cmd = [sys.executable, os.path.abspath(__file__), "--stage", stage,
           "--stage-in", in_path, "--stage-out", out_path]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"stage '{stage}' failed (exit {proc.returncode}) after {elapsed:.1f}s:\n"
            f"{proc.stderr[-3000:]}"
        )
    with open(out_path, "r") as f:
        out = json.load(f)
    out["wall_s"] = round(elapsed, 2)
    print(f"  [subprocess {label}] done in {elapsed:.1f}s")
    return out


def stage_stt_main(payload):
    """--stage stt: load Qwen3-ASR on GPU, transcribe all audios, exit."""
    audios = payload["audios"]
    t0 = time.time()
    stt = init_stt("cuda")
    load_s = time.time() - t0
    items = []
    for path in audios:
        audio = P.load_audio_file(path)
        t1 = time.time()
        text, lang = P.transcribe(stt, audio)
        items.append({"stt_s": round(time.time() - t1, 3), "text": text,
                      "language": lang})
    out = {
        "load_s": round(load_s, 3),
        "gpu_peak_mb": gpu_used_mb(),
        "items": items,
    }
    return out


def stage_mt_main(payload):
    """--stage mt: load Hy-MT2 GGUF on GPU, translate all texts, exit."""
    texts = payload["texts"]
    src = payload["src"]
    targets = payload["targets"]
    t0 = time.time()
    mt = P.init_translator()
    load_s = time.time() - t0
    items = []
    for text in texts:
        t1 = time.time()
        translations = P.translate_all(mt, text, src)
        items.append({"mt_s": round(time.time() - t1, 3),
                      "translations": {t: translations[t] for t in targets}})
    out = {
        "load_s": round(load_s, 3),
        "gpu_peak_mb": gpu_used_mb(),
        "items": items,
    }
    return out


def run_language_swap(lang, num_sentences, do_accuracy):
    """Run the full pipeline for a source language using subprocess-isolated
    stages: each model is loaded to VRAM, used, then its process exits so the
    OS reclaims the memory before the next model loads."""
    sentences = P.TEST_SENTENCES.get(lang, [])[:num_sentences]
    refs = REFERENCE_KEYWORDS.get(lang, {})
    targets = [l for l in ALL_LANGS if l != lang]

    print("=" * 60)
    print(f"LANGUAGE: {lang.upper()} ({LANG_HINT.get(lang, lang)}) → {[t.upper() for t in targets]}")
    print(f"SENTENCES: {len(sentences)}  [swap strategy: subprocess-isolated models]")
    print("=" * 60)

    os.makedirs(P.OUTPUT_DIR, exist_ok=True)

    # ── 1. Synthesize all input audios (Piper, CPU, parent process) ──
    audio_paths, audio_durs = [], []
    for i, sentence in enumerate(sentences):
        path = os.path.join(P.OUTPUT_DIR, f"test_{lang}_{i:03d}_input.wav")
        P.tts_piper(sentence, lang, path)
        import wave
        with wave.open(path, "rb") as wf:
            audio_durs.append(wf.getnframes() / wf.getframerate())
        audio_paths.append(path)

    # ── 2. STT stage (subprocess: load → transcribe all → exit) ──
    stt_out = run_stage_process("stt", {"audios": audio_paths}, "STT")

    # ── 3. MT stage (subprocess: load → translate all → exit) ──
    mt_out = run_stage_process(
        "mt",
        {"texts": [i["text"] for i in stt_out["items"]], "src": lang,
         "targets": targets},
        "MT",
    )

    # ── 4. Output TTS + scoring (Piper, CPU, parent process) ──
    results = []
    stt_times, mt_times, tts_times, wall_times = [], [], [], []
    stt_errors, kw_scores, stt_rtfx = [], [], []
    stt_loads, mt_loads = [stt_out["load_s"]], [mt_out["load_s"]]

    for i, sentence in enumerate(sentences):
        s = stt_out["items"][i]
        m = mt_out["items"][i]
        audio_dur = audio_durs[i]
        rec = {
            "sentence": sentence,
            "audio_sec": round(audio_dur, 2),
            "stt_hyp": s["text"],
            "stt_lang_detected": s["language"],
            "stt_error": None,
            "stt_rtfx": round(audio_dur / s["stt_s"], 2) if s["stt_s"] else None,
            "mt": m["translations"],
            "tts": {},
            "timings": {
                "stt_load": round(stt_out["load_s"], 3),
                "stt": round(s["stt_s"], 3),
                "mt_load": round(mt_out["load_s"], 3),
                "mt_total": round(m["mt_s"], 3),
                "mt_per_target": {},
            },
            "keyword_accuracy": {},
        }
        stt_times.append(s["stt_s"])
        stt_rtfx.append(rec["stt_rtfx"])
        mt_times.append(m["mt_s"])

        if do_accuracy:
            err = stt_error_rate(sentence, s["text"], lang)
            rec["stt_error"] = round(err, 4)
            stt_errors.append(err)

        print(f"\n  [{i+1}/{len(sentences)}] {sentence}")
        print(f"    STT  ({s['stt_s']:.2f}s, RTFx={rec['stt_rtfx']:.1f}x): "
              f"[{s['language']}] {s['text']}")
        if do_accuracy:
            print(f"         error: {err*100:.1f}%")

        for target in targets:
            tr_text = m["translations"][target]
            if do_accuracy:
                kw = keyword_accuracy(tr_text, refs.get(sentence, {}).get(target, []))
                rec["keyword_accuracy"][target] = round(kw, 1)
                kw_scores.append(kw)
            print(f"    MT→{target.upper()}: {tr_text}")

        t0_total = time.time()
        rec["timings"]["tts"] = {}
        for target in targets:
            out_path = os.path.join(P.OUTPUT_DIR, f"test_{lang}_{i:03d}_{target}.wav")
            t0 = time.time()
            info = P.tts_piper(m["translations"][target], target, out_path)
            t_tts = time.time() - t0
            rec["timings"]["tts"][target] = round(t_tts, 3)
            rtf = info["rtf"] if info else None
            rec["tts"][target] = {"rtf": round(rtf, 4) if rtf else None,
                                  "file": out_path}
            if rtf:
                print(f"    TTS→{target.upper()} ({t_tts:.2f}s, RTF={rtf:.3f}x)")
            else:
                print(f"    TTS→{target.upper()} ({t_tts:.2f}s)")
        rec["timings"]["tts_total"] = round(time.time() - t0_total, 3)
        tts_times.append(rec["timings"]["tts_total"])

        rec["timings"]["total"] = round(s["stt_s"] + m["mt_s"] + rec["timings"]["tts_total"], 3)
        wall_times.append(rec["timings"]["total"])
        print(f"    ── TOTAL: {rec['timings']['total']:.2f}s (loads excluded)")

        results.append(rec)

    return build_summary(
        lang, targets, results, stt_times, mt_times, tts_times, wall_times,
        stt_errors, kw_scores, stt_rtfx, stt_loads, mt_loads,
    )


def main():
    parser = argparse.ArgumentParser(description="Pipeline latency & accuracy test")
    parser.add_argument("--lang", default="all", help="Source langs: 'all' or 'en,zh,ko,vi'")
    parser.add_argument("--num", type=int, default=None, help="Max sentences per lang")
    parser.add_argument("--no-accuracy", action="store_true", help="Skip accuracy scoring")
    parser.add_argument("--stage", default=None, choices=["stt", "mt"],
                        help="Internal: run a single model stage in a subprocess")
    parser.add_argument("--stage-in", default=None, help="Internal: stage input JSON path")
    parser.add_argument("--stage-out", default=None, help="Internal: stage output JSON path")
    parser.add_argument("--swap", action="store_true", dest="swap", default=True,
                        help="Swap STT/MT in/out of VRAM per stage (default; needs only ~max(stt,mt) VRAM)")
    parser.add_argument("--keep-loaded", action="store_true", dest="keep_loaded",
                        help="Keep both models loaded for the whole run (STT on CPU, MT on GPU)")
    args = parser.parse_args()

    # Internal subprocess mode: run one model stage and exit.
    if args.stage:
        with open(args.stage_in) as f:
            payload = json.load(f)
        if args.stage == "stt":
            out = stage_stt_main(payload)
        else:
            out = stage_mt_main(payload)
        with open(args.stage_out, "w") as f:
            json.dump(out, f)
        # llama.cpp's Vulkan backend can segfault in its destructors at
        # interpreter shutdown; results are written, so skip teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    swap = not args.keep_loaded

    if args.lang == "all":
        langs = ALL_LANGS
    else:
        langs = [l.strip() for l in args.lang.split(",") if l.strip() in ALL_LANGS]
    do_accuracy = not args.no_accuracy

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check GPU availability for translator
    gpu = gpu_memory_mb()
    if gpu is None:
        print("WARNING: No NVIDIA GPU detected. MT will fall back to CPU "
              "(this will be slow for a 7B model).")
        swap = False

    # Phase 1: measure each model's VRAM footprint independently
    load_stats = measure_footprints()
    if swap:
        load_stats["strategy"] = "swap"
    else:
        load_stats["strategy"] = "keep-loaded (STT cpu, MT gpu)"

    # Phase 2: run the latency/accuracy test
    mgr = ModelManager(swap=swap)
    all_results = {}
    overall_stt, overall_mt, overall_tts, overall_wall = [], [], [], []
    overall_error, overall_kw = [], []

    for lang in langs:
        if swap:
            res = run_language_swap(lang, args.num, do_accuracy)
        else:
            res = run_language(mgr, lang, args.num, do_accuracy)
        all_results[lang] = res
        s = res["summary"]
        if s["stt_avg_s"]:
            overall_stt.append(s["stt_avg_s"])
        if s["mt_avg_s"]:
            overall_mt.append(s["mt_avg_s"])
        if s["tts_avg_s"]:
            overall_tts.append(s["tts_avg_s"])
        if s["wall_avg_s"]:
            overall_wall.append(s["wall_avg_s"])
        if s["stt_avg_error"] is not None:
            overall_error.append(s["stt_avg_error"])
        if s["kw_avg"] is not None:
            overall_kw.append(s["kw_avg"])

    # record peak VRAM observed during test
    load_stats["test_peak_gpu_mb"] = gpu_used_mb()

    # ── Final report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "langs": langs,
            "max_sentences_per_lang": args.num,
            "accuracy": do_accuracy,
            "strategy": load_stats["strategy"],
        },
        "footprint": {k: v for k, v in load_stats.items()},
        "languages": all_results,
        "overall": {
            "stt_avg_s": round(mean(overall_stt), 3) if overall_stt else None,
            "mt_avg_s": round(mean(overall_mt), 3) if overall_mt else None,
            "tts_avg_s": round(mean(overall_tts), 3) if overall_tts else None,
            "wall_avg_s": round(mean(overall_wall), 3) if overall_wall else None,
            "stt_avg_error": round(mean(overall_error), 4) if overall_error else None,
            "kw_avg": round(mean(overall_kw), 1) if overall_kw else None,
        },
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── CSV summary table ──
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lang", "n", "STT(s)", "MT(s)", "TTS(s)", "wall(s)",
                         "STT RTFx", "STT err(%)", "kw acc(%)"])
        for lang, res in all_results.items():
            s = res["summary"]
            writer.writerow([
                lang.upper(), s["n_sentences"],
                s["stt_avg_s"], s["mt_avg_s"], s["tts_avg_s"], s["wall_avg_s"],
                s["stt_avg_rtfx"],
                f"{s['stt_avg_error_pct']:.1f}" if s.get("stt_avg_error_pct") else "",
                s["kw_avg"] if s["kw_avg"] is not None else "",
            ])

    # ── Console final summary ──
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    fp = load_stats
    print(f"  Strategy: {fp.get('strategy', '?')}")
    stt_fp = fp.get("stt", {})
    mt_fp = fp.get("mt", {})
    print(f"  STT VRAM footprint: +{stt_fp.get('gpu_mb')} MB "
          f"(load {stt_fp.get('load_s')}s, freed {stt_fp.get('freed_mb')} MB)")
    print(f"  MT  VRAM footprint: +{mt_fp.get('gpu_mb')} MB "
          f"(load {mt_fp.get('load_s')}s, freed {mt_fp.get('freed_mb')} MB)")
    if "peak_both_mb" in fp:
        print(f"  Peak if both resident: {fp['peak_both_mb']} / {fp.get('gpu_total_mb')} MB "
              f"→ {'FITS' if fp.get('peak_both_fits') else 'MUST SWAP'}")
    if "test_peak_gpu_mb" in fp:
        print(f"  Peak VRAM during test: {fp['test_peak_gpu_mb']} MB")
    print()
    header = f"  {'LANG':<6} {'n':>3} {'STT(s)':>8} {'MT(s)':>7} {'TTS(s)':>7} {'Wall(s)':>8} {'RTFx':>6} {'STT err':>9} {'kw acc':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lang, res in all_results.items():
        s = res["summary"]
        err = f"{s['stt_avg_error_pct']:.1f}%" if s.get("stt_avg_error_pct") is not None else "-"
        kw = f"{s['kw_avg']:.0f}%" if s["kw_avg"] is not None else "-"
        print(f"  {lang.upper():<6} {s['n_sentences']:>3} {s['stt_avg_s']:>8} {s['mt_avg_s']:>7} "
              f"{s['tts_avg_s']:>7} {s['wall_avg_s']:>8} {s['stt_avg_rtfx']:>6} {err:>9} {kw:>8}")
    o = report["overall"]
    print("  " + "-" * (len(header) - 2))
    print(f"  {'AVG':<6} {'':>3} {o['stt_avg_s']:>8} {o['mt_avg_s']:>7} {o['tts_avg_s']:>7} "
          f"{o['wall_avg_s']:>8}")
    if o["stt_avg_error"] is not None:
        print(f"  Overall STT error: {o['stt_avg_error']*100:.1f}%")
    if o["kw_avg"] is not None:
        print(f"  Overall translation keyword accuracy: {o['kw_avg']:.1f}%")
    print()
    print(f"  Full report: {JSON_PATH}")
    print(f"  Summary CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
