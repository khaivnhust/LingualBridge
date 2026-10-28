#!/bin/python3
"""
Korean re-test on real human speech (Google FLEURS ko_kr test split) with all
models resident in VRAM (same setup as roundtrip_vram_test.py).

FLEURS sentences are n-way parallel, so the EN/ZH/VI human translations of the
same sentence ids serve as MT references.

  STT  CER vs the FLEURS normalized transcript (punctuation and spaces removed)
  MT   chrF vs FLEURS references, from (a) the STT output and (b) the
       ground-truth transcript ("oracle", untimed) to separate STT from MT errors
  RTF  processing time / input audio duration

Usage:
    python3 fleurs_ko_test.py            # 10 sentences
    python3 fleurs_ko_test.py --num 20
"""

import os
import re
import csv
import json
import time
import tarfile
import argparse
import unicodedata
from statistics import mean

import pipeline as P  # sets torch env vars; import before torch
import pipeline_test as T
import roundtrip_vram_test as R

FLEURS_REPO = "google/fleurs"
FLEURS_LANGS = {"ko": "ko_kr", "en": "en_us", "zh": "cmn_hans_cn", "vi": "vi_vn"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fleurs_ko")
JSON_PATH = os.path.join(R.RESULTS_DIR, "fleurs_ko_results.json")
MAX_AUDIO_S = 12.0


def read_tsv(lang):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FLEURS_REPO, f"data/{FLEURS_LANGS[lang]}/test.tsv",
                           repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append({"id": r[0], "file": r[1], "raw": r[2], "norm": r[3],
                         "samples": int(r[5])})
    return rows


def select_samples(num):
    """First `num` sentence ids (sorted) present in all 4 languages, one
    Korean recording each, at most MAX_AUDIO_S long."""
    ko = read_tsv("ko")
    refs = {l: {r["id"]: r["raw"] for r in read_tsv(l)} for l in ("en", "zh", "vi")}
    seen, picked = set(), []
    for r in sorted(ko, key=lambda r: (int(r["id"]), r["file"])):
        if r["id"] in seen or r["samples"] / 16000 > MAX_AUDIO_S:
            continue
        if all(r["id"] in refs[l] for l in refs):
            seen.add(r["id"])
            r["refs"] = {l: refs[l][r["id"]] for l in refs}
            picked.append(r)
        if len(picked) == num:
            break
    return picked


def extract_audio(samples):
    import io
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    os.makedirs(DATA_DIR, exist_ok=True)
    want = {s["file"] for s in samples
            if not os.path.exists(os.path.join(DATA_DIR, s["file"]))}
    if want:
        tar_path = hf_hub_download(FLEURS_REPO, "data/ko_kr/audio/test.tar.gz",
                                   repo_type="dataset")
        with tarfile.open(tar_path) as tf:
            for m in tf:
                name = os.path.basename(m.name)
                if name in want:
                    # FLEURS ships 32-bit float WAVs; the pipeline's wave
                    # module reader needs PCM.
                    audio, sr = sf.read(io.BytesIO(tf.extractfile(m).read()))
                    sf.write(os.path.join(DATA_DIR, name), audio, sr,
                             subtype="PCM_16")
    for s in samples:
        s["path"] = os.path.join(DATA_DIR, s["file"])


def norm_ko(text):
    """Lowercase, drop punctuation and whitespace (character-level CER)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text
                   if not unicodedata.category(ch).startswith(("P", "S")))
    return re.sub(r"\s+", "", text)


def chrf(hyp, ref):
    import sacrebleu
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def main():
    import jiwer

    ap = argparse.ArgumentParser(description="FLEURS Korean round-trip re-test")
    ap.add_argument("--num", type=int, default=10)
    args = ap.parse_args()

    samples = select_samples(args.num)
    extract_audio(samples)
    print(f"FLEURS ko_kr test: {len(samples)} sentences "
          f"({sum(s['samples'] for s in samples) / 16000:.1f}s audio)\n")

    print("=" * 60)
    print("LOADING ALL MODELS INTO VRAM")
    print("=" * 60)
    stt, mt, tts, fp = R.load_all()

    sampler = R.VramSampler()
    sampler.start()
    t0 = time.time()
    R.round_trip(stt, mt, tts, samples[0]["path"], "ko", "fleurs_warmup")
    fp["warmup_s"] = round(time.time() - t0, 2)
    print(f"  warm-up round trip: {fp['warmup_s']}s (excluded)\n")

    recs = []
    for i, s in enumerate(samples):
        r = R.round_trip(stt, mt, tts, s["path"], "ko", f"fleurs_ko_{i:03d}")
        r["fleurs_id"] = s["id"]
        r["reference"] = s["raw"]
        r["references"] = s["refs"]
        r["stt_cer"] = round(jiwer.cer(norm_ko(s["raw"]), norm_ko(r["stt_hyp"])), 4)
        r["chrf"] = {t: round(chrf(r["mt"][t], s["refs"][t]), 1) for t in r["mt"]}

        # Oracle MT: translate the ground-truth transcript (untimed)
        oracle = P.translate_all(mt, s["raw"], "ko")
        r["mt_oracle"] = {t: oracle[t] for t in r["mt"]}
        r["chrf_oracle"] = {t: round(chrf(oracle[t], s["refs"][t]), 1) for t in r["mt"]}
        recs.append(r)

        tm, rt = r["timings"], r["rtf"]
        print(f"  [{i+1}] REF: {s['raw']}")
        print(f"      STT {tm['stt']:.2f}s (RTF {rt['stt']:.3f}) CER {r['stt_cer']*100:.1f}%: {r['stt_hyp']}")
        for t in r["mt"]:
            print(f"      →{t.upper()} chrF {r['chrf'][t]:.0f} (oracle {r['chrf_oracle'][t]:.0f}): {r['mt'][t]}")
        print(f"      MT {tm['mt']:.2f}s  TTS {tm['tts']:.2f}s  TOTAL {tm['total']:.2f}s"
              f"  e2e RTF {rt['e2e']:.2f}")

    sampler.stop()
    fp["peak_gpu_mb"] = sampler.peak

    targets = list(recs[0]["mt"])
    summary = {
        "n": len(recs),
        "audio_avg_s": round(mean(r["audio_sec"] for r in recs), 2),
        "stt_avg_s": round(mean(r["timings"]["stt"] for r in recs), 3),
        "mt_avg_s": round(mean(r["timings"]["mt"] for r in recs), 3),
        "tts_avg_s": round(mean(r["timings"]["tts"] for r in recs), 3),
        "total_avg_s": round(mean(r["timings"]["total"] for r in recs), 3),
        "stt_rtf": round(mean(r["rtf"]["stt"] for r in recs), 4),
        "tts_rtf": round(mean(r["rtf"]["tts"] for r in recs), 4),
        "e2e_rtf": round(mean(r["rtf"]["e2e"] for r in recs), 4),
        "stt_cer_pct": round(mean(r["stt_cer"] for r in recs) * 100, 1),
        "chrf": {t: round(mean(r["chrf"][t] for r in recs), 1) for t in targets},
        "chrf_oracle": {t: round(mean(r["chrf_oracle"][t] for r in recs), 1) for t in targets},
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "dataset": "google/fleurs ko_kr test",
                   "footprint": fp, "summary": summary, "sentences": recs},
                  f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("FLEURS KOREAN REPORT (all models in VRAM)")
    print("=" * 60)
    print(f"  VRAM all loaded {fp['all_loaded_mb']} MB, peak {fp['peak_gpu_mb']}"
          f" / {fp['gpu_total_mb']} MB")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  Full report: {JSON_PATH}")


if __name__ == "__main__":
    main()
