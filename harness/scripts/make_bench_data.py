#!/usr/bin/env python3
"""Build the benchmark audio set: 16 kHz mono WAV, 20 utterances per language.

Prefers conversational speech because that is what a speech-to-speech
translation app actually receives:

  en -> VoxPopuli held-out   (parliamentary / semi-conversational)
  ko -> KsponSpeech          (spontaneous)
  zh -> Common Voice         (crowdsourced read -- no conversational zh exists)
  vi -> Common Voice         (crowdsourced read -- no conversational vi exists)

The zh/vi gap is a real limitation and is recorded in SUBSTITUTIONS.md: those
two languages have no spontaneous corpus in the project at all, which is also
open item 1 in WORKLOG.
"""
import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import wave


def read_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to_wav16k(src, dst):
    """Decode to 16 kHz mono 16-bit PCM using ffmpeg; returns duration seconds."""
    cmd = ["ffmpeg", "-nostdin", "-v", "quiet", "-y", "-i", src,
           "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", dst]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst):
        return None
    with wave.open(dst, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def pick(rows, n, seed, min_s=2.0, max_s=12.0):
    """Deterministic pick of n utterances within a sane duration band."""
    rng = random.Random(seed)
    ok = [r for r in rows if r.get("audio")
          and (r.get("duration") or 0) >= min_s
          and (r.get("duration") or 0) <= max_s]
    rng.shuffle(ok)
    return ok[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs-test", required=True)
    ap.add_argument("--cvtest", required=True)
    ap.add_argument("--kspon-manifest", required=True)
    ap.add_argument("--voxpopuli", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required to build the benchmark audio")

    adir = os.path.join(a.out, "audio")
    os.makedirs(adir, exist_ok=True)

    sources = {
        "en": (pick(read_manifest(a.voxpopuli), a.n, 1), a.voxpopuli),
        "ko": (pick(read_manifest(a.kspon_manifest), a.n, 2), a.kspon_manifest),
        "zh": (pick(read_manifest(os.path.join(a.cvtest, "zh.jsonl")), a.n, 3),
               os.path.join(a.cvtest, "zh.jsonl")),
        "vi": (pick(read_manifest(os.path.join(a.cvtest, "vi.jsonl")), a.n, 4),
               os.path.join(a.cvtest, "vi.jsonl")),
    }

    index, total, missing = [], 0, 0
    for lg, (rows, manpath) in sources.items():
        man = {r["audio"]: r for r in read_manifest(manpath)}
        for i, r in enumerate(rows):
            dst = os.path.join(adir, f"{lg}_{i:02d}.wav")
            dur = to_wav16k(r["audio"], dst)
            if dur is None:
                missing += 1
                os.path.exists(dst) and os.remove(dst)
                continue
            m = man.get(r["audio"], {})
            index.append({"id": f"{lg}_{i:02d}", "lang": lg, "wav": f"audio/{lg}_{i:02d}.wav",
                          "duration": round(dur, 3),
                          "text": m.get("text", ""),
                          "source": m.get("source", "")})
            total += dur
    with open(os.path.join(a.out, "audio_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    for lg in ("en", "zh", "ko", "vi"):
        c = sum(1 for x in index if x["lang"] == lg)
        print(f"    {lg}: {c} utterances")
    print(f"    total {len(index)} utterances, {total/60:.1f} min audio"
          + (f", {missing} failed to convert" if missing else ""))


if __name__ == "__main__":
    main()
