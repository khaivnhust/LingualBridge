#!/bin/python3
"""
Round-trip test with every model resident in VRAM at once.

  STT  Qwen3-ASR-1.7B (bf16, CUDA)
  MT   Hy-MT2-7B Q6_K GGUF (llama.cpp, all layers on GPU)
  TTS  Piper, all 4 voices (onnxruntime CUDA)

All models are loaded once up front and stay loaded; each sentence runs
audio → STT → MT (3 targets) → TTS (3 targets) with no model swapping.

Input audio is synthesized with the Piper binary (CPU, untimed), the same way
pipeline_test.py builds its inputs.

RTF = processing time / audio duration (lower is better, <1 = faster than real time).

Usage:
    python3 roundtrip_vram_test.py
    python3 roundtrip_vram_test.py --lang en,vi --num 3
"""

import os
import sys
import json
import time
import wave
import argparse
import threading
import subprocess
from statistics import mean

import pipeline as P  # sets torch env vars; import before torch
import pipeline_test as T
import torch

RESULTS_DIR = T.RESULTS_DIR
JSON_PATH = os.path.join(RESULTS_DIR, "roundtrip_vram_results.json")


class VramSampler(threading.Thread):
    """Poll nvidia-smi to record peak GPU memory while the test runs."""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            used = T.gpu_used_mb()
            if used is not None:
                self.peak = max(self.peak, used)
            time.sleep(self.interval)

    def stop(self):
        self._halt.set()
        self.join()


class _ShrinkingSession:
    """InferenceSession wrapper that releases the GPU arena after every run.

    Each Piper session otherwise keeps its peak arena (200-300 MB) forever;
    three of them overflow the ~1 GB left with STT+MT resident.
    """

    def __init__(self, session):
        import onnxruntime

        self._session = session
        self._run_options = onnxruntime.RunOptions()
        self._run_options.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage", "gpu:0")

    def run(self, output_names, input_feed, run_options=None):
        return self._session.run(output_names, input_feed, self._run_options)

    def __getattr__(self, name):
        return getattr(self._session, name)


class PiperGPU:
    """All Piper voices preloaded on the GPU via onnxruntime CUDA."""

    def __init__(self, langs):
        import onnxruntime
        from piper import PiperVoice

        # CUDA/cuDNN come from the nvidia-* pip wheels (installed with torch),
        # which are not on the loader path by default.
        onnxruntime.preload_dlls()

        self.voices = {}
        self.cpu_fallback = []
        for lang in langs:
            try:
                voice = PiperVoice.load(P.PIPER_MODELS[lang], use_cuda=False)
                # With STT+MT resident only ~1 GB of VRAM is left; the default
                # arena grows in powers of two and OOMs on long sentences.
                voice.session = _ShrinkingSession(onnxruntime.InferenceSession(
                    P.PIPER_MODELS[lang],
                    providers=[("CUDAExecutionProvider", {
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "arena_extend_strategy": "kSameAsRequested",
                    })],
                ))
                self.voices[lang] = voice
            except ValueError as e:
                # ko_KR-kss uses phoneme_type "pygoruut", which piper-tts can't
                # load; use the Piper binary on CPU like pipeline.py does.
                print(f"  Piper {lang}: cannot load on GPU ({e}); using CPU binary")
                self.cpu_fallback.append(lang)
        providers = {
            lang: v.session.get_providers()[0] for lang, v in self.voices.items()
        }
        print(f"  Piper providers: {providers}")
        if any(p != "CUDAExecutionProvider" for p in providers.values()):
            raise RuntimeError(f"Piper not on GPU: {providers}")

    def synthesize(self, text, lang, out_path):
        """Returns (synth_seconds, audio_seconds)."""
        t0 = time.time()
        if lang in self.cpu_fallback:
            P.tts_piper(text, lang, out_path)
        else:
            with wave.open(out_path, "wb") as wf:
                self.voices[lang].synthesize_wav(text, wf)
        elapsed = time.time() - t0
        with wave.open(out_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        return elapsed, dur


def load_all():
    """Load MT, STT, TTS and keep them all resident; record VRAM per model."""
    fp = {"gpu_total_mb": T.gpu_memory_mb()["gpu0"]["total_mb"]}
    base = T.gpu_used_mb()
    fp["gpu_base_mb"] = base

    t0 = time.time()
    mt = P.init_translator()
    after_mt = T.gpu_used_mb()
    fp["mt"] = {"load_s": round(time.time() - t0, 2), "gpu_mb": after_mt - base}

    t0 = time.time()
    stt = P.init_stt("cuda")
    after_stt = T.gpu_used_mb()
    fp["stt"] = {"load_s": round(time.time() - t0, 2), "gpu_mb": after_stt - after_mt}

    t0 = time.time()
    tts = PiperGPU(T.ALL_LANGS)
    after_tts = T.gpu_used_mb()
    fp["tts"] = {"load_s": round(time.time() - t0, 2), "gpu_mb": after_tts - after_stt}

    fp["tts"]["cpu_fallback_langs"] = tts.cpu_fallback
    fp["all_loaded_mb"] = after_tts
    print(f"  VRAM: base {base} → +MT {fp['mt']['gpu_mb']} → +STT {fp['stt']['gpu_mb']}"
          f" → +TTS {fp['tts']['gpu_mb']} = {after_tts}/{fp['gpu_total_mb']} MB\n")
    return stt, mt, tts, fp


def round_trip(stt, mt, tts, audio_path, lang, tag):
    """One full round trip; returns timings and outputs."""
    targets = [l for l in T.ALL_LANGS if l != lang]
    with wave.open(audio_path, "rb") as wf:
        in_dur = wf.getnframes() / wf.getframerate()
    audio = P.load_audio_file(audio_path)

    t_start = time.time()

    t0 = time.time()
    text, detected = P.transcribe(stt, audio)
    # Hand torch's cached activation blocks back so Piper's ORT arenas fit
    # in the ~1 GB left with every model resident (counted in STT time).
    torch.cuda.empty_cache()
    stt_s = time.time() - t0

    t0 = time.time()
    translations = P.translate_all(mt, text, lang)
    mt_s = time.time() - t0

    tts_info = {}
    t0 = time.time()
    for tgt in targets:
        out = os.path.join(P.OUTPUT_DIR, f"rt_{tag}_{tgt}.wav")
        synth_s, out_dur = tts.synthesize(translations[tgt], tgt, out)
        tts_info[tgt] = {"synth_s": round(synth_s, 3), "audio_s": round(out_dur, 2),
                         "rtf": round(synth_s / out_dur, 4) if out_dur else None}
    tts_s = time.time() - t0

    total_s = time.time() - t_start
    return {
        "audio_sec": round(in_dur, 2),
        "stt_hyp": text,
        "stt_lang_detected": detected,
        "mt": {t: translations[t] for t in targets},
        "tts": tts_info,
        "timings": {"stt": round(stt_s, 3), "mt": round(mt_s, 3),
                    "tts": round(tts_s, 3), "total": round(total_s, 3)},
        "rtf": {"stt": round(stt_s / in_dur, 4),
                "mt": round(mt_s / in_dur, 4),
                "tts": round(tts_s / sum(v["audio_s"] for v in tts_info.values()), 4),
                "e2e": round(total_s / in_dur, 4)},
    }


def main():
    ap = argparse.ArgumentParser(description="All-models-in-VRAM round-trip test")
    ap.add_argument("--lang", default="all")
    ap.add_argument("--num", type=int, default=None)
    args = ap.parse_args()
    langs = T.ALL_LANGS if args.lang == "all" else [
        l.strip() for l in args.lang.split(",") if l.strip() in T.ALL_LANGS]

    os.makedirs(P.OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Inputs: Piper binary on CPU, untimed (same as pipeline_test.py)
    inputs = {}
    for lang in langs:
        inputs[lang] = []
        for i, s in enumerate(P.TEST_SENTENCES[lang][:args.num]):
            path = os.path.join(P.OUTPUT_DIR, f"test_{lang}_{i:03d}_input.wav")
            P.tts_piper(s, lang, path)
            inputs[lang].append((s, path))

    print("=" * 60)
    print("LOADING ALL MODELS INTO VRAM")
    print("=" * 60)
    stt, mt, tts, fp = load_all()

    sampler = VramSampler()
    sampler.start()

    # Warm-up (CUDA/Vulkan kernel init), excluded from stats
    s0, p0 = inputs[langs[0]][0]
    t0 = time.time()
    round_trip(stt, mt, tts, p0, langs[0], "warmup")
    fp["warmup_s"] = round(time.time() - t0, 2)
    print(f"  warm-up round trip: {fp['warmup_s']}s (excluded)\n")

    report_langs = {}
    for lang in langs:
        refs = T.REFERENCE_KEYWORDS.get(lang, {})
        print("=" * 60)
        print(f"LANGUAGE: {lang.upper()}")
        print("=" * 60)
        recs = []
        for i, (sentence, path) in enumerate(inputs[lang]):
            r = round_trip(stt, mt, tts, path, lang, f"{lang}_{i:03d}")
            r["sentence"] = sentence
            r["stt_error"] = round(T.stt_error_rate(sentence, r["stt_hyp"], lang), 4)
            r["keyword_accuracy"] = {
                t: round(T.keyword_accuracy(tr, refs.get(sentence, {}).get(t, [])), 1)
                for t, tr in r["mt"].items()}
            recs.append(r)
            tm, rt = r["timings"], r["rtf"]
            print(f"  [{i+1}] {sentence}")
            print(f"      STT {tm['stt']:.2f}s (RTF {rt['stt']:.3f}) err {r['stt_error']*100:.1f}%: {r['stt_hyp']}")
            for t, tr in r["mt"].items():
                print(f"      →{t.upper()} kw {r['keyword_accuracy'][t]:.0f}%: {tr}")
            print(f"      MT {tm['mt']:.2f}s  TTS {tm['tts']:.2f}s (RTF {rt['tts']:.3f})"
                  f"  TOTAL {tm['total']:.2f}s  e2e RTF {rt['e2e']:.2f}")

        def avg(f):
            return round(mean(f(r) for r in recs), 4)

        summary = {
            "n": len(recs),
            "audio_avg_s": avg(lambda r: r["audio_sec"]),
            "stt_avg_s": avg(lambda r: r["timings"]["stt"]),
            "mt_avg_s": avg(lambda r: r["timings"]["mt"]),
            "tts_avg_s": avg(lambda r: r["timings"]["tts"]),
            "total_avg_s": avg(lambda r: r["timings"]["total"]),
            "stt_rtf": avg(lambda r: r["rtf"]["stt"]),
            "tts_rtf": avg(lambda r: r["rtf"]["tts"]),
            "e2e_rtf": avg(lambda r: r["rtf"]["e2e"]),
            "stt_err_pct": round(avg(lambda r: r["stt_error"]) * 100, 1),
            "kw_avg": round(mean(v for r in recs for v in r["keyword_accuracy"].values()), 1),
        }
        report_langs[lang] = {"summary": summary, "sentences": recs}
        print(f"  ── {lang.upper()}: {summary}\n")

    sampler.stop()
    fp["peak_gpu_mb"] = sampler.peak

    S = [v["summary"] for v in report_langs.values()]
    overall = {k: round(mean(s[k] for s in S), 4) for k in S[0] if k != "n"}
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "strategy": "all models resident in VRAM",
              "footprint": fp, "languages": report_langs, "overall": overall}
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("ROUND-TRIP REPORT (all models in VRAM)")
    print("=" * 60)
    print(f"  VRAM all loaded: {fp['all_loaded_mb']} MB, peak during run: "
          f"{fp['peak_gpu_mb']} / {fp['gpu_total_mb']} MB")
    hdr = f"  {'LANG':<5}{'audio':>7}{'STT':>7}{'MT':>7}{'TTS':>7}{'total':>8}{'STT RTF':>9}{'TTS RTF':>9}{'e2e RTF':>9}{'STT err':>9}{'kw':>6}"
    print(hdr)
    for lang, v in report_langs.items():
        s = v["summary"]
        print(f"  {lang.upper():<5}{s['audio_avg_s']:>7.2f}{s['stt_avg_s']:>7.2f}{s['mt_avg_s']:>7.2f}"
              f"{s['tts_avg_s']:>7.2f}{s['total_avg_s']:>8.2f}{s['stt_rtf']:>9.3f}{s['tts_rtf']:>9.3f}"
              f"{s['e2e_rtf']:>9.2f}{s['stt_err_pct']:>8.1f}%{s['kw_avg']:>5.0f}%")
    print(f"\n  Full report: {JSON_PATH}")


if __name__ == "__main__":
    main()
