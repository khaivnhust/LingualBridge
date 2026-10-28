#!/usr/bin/env python3
"""Build ASR manifests from the KsponSpeech parquet mirror.

Why this exists
---------------
Every metric so far in this project is studio read speech (FLEURS, Common
Voice). KsponSpeech is spontaneous two-speaker Korean conversation recorded by
2,000 native speakers -- disfluencies, overlap, backchannel, real prosody. It
is the first genuinely out-of-distribution evaluation set here.

The mirror stores raw 16 kHz WAV bytes inside parquet `audio{bytes,path}`
structs. This flattens them to 16-bit PCM WAV and emits the same manifest
schema as the other corpora:

    {"audio", "text", "lang", "duration", "source", "id"}

Transcript quirks handled here:
  * a trailing newline on every row is stripped;
  * utterances are often annotated with a leading filler/noise token
    ("음/ ", "o/ ", "맞 assimilations"), which are stripped because they are
    disfluency markers rather than lexical content -- keeping them would make
    CER unreproducible against any other Korean corpus.

Usage:
    scripts/prepare_ksponspeech.py --split test
    scripts/prepare_ksponspeech.py --split valid --max-rows 500
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_RATE = 16000

# Leading disfluency/noise annotations that are not lexical content, e.g.
# "음/ ", "o/ ", "맞 assimilations". These bracket non-speech noise and are not
# tokens any ASR would emit; leaving them in the reference taxes every such
# utterance with CER the model can never avoid. The Hangul range must include
# precomposed syllables (U+AC00-D7A3), not just compatibility jamo, or "음/"
# slips through.
FILLER = re.compile(r"^\s*(?:(?:[a-zA-Z]{1,3}|[가-힣]{1,3})/\s*)+")


def norm_text(s):
    s = (s or "").strip()
    s = FILLER.sub("", s)
    return s.strip()


def write_clip(job):
    """(bytes, stem, dst) -> (n_samples, err)."""
    blob, stem, dst = job
    if os.path.exists(dst):
        try:
            import soundfile as sf
            return sf.info(dst).frames, None
        except Exception:                      # noqa: BLE001
            pass
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(blob), dtype="int16", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        if sr != SAMPLE_RATE:
            import librosa
            data = (librosa.resample(data.astype(np.float32), orig_sr=sr,
                                     target_sr=SAMPLE_RATE) * 32767).astype(np.int16)
        sf.write(dst, data, SAMPLE_RATE, subtype="PCM_16")
        return len(data), None
    except Exception as e:                      # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--DragonLine--ksponspeech"))
    ap.add_argument("--split", required=True, help="test | valid | train")
    ap.add_argument("--out", default="data/ksponspeech")
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--drop-empty", action="store_true", default=True)
    a = ap.parse_args()

    import pyarrow.parquet as pq

    snaps = glob.glob(os.path.join(a.cache, "snapshots", "*"))
    if not snaps:
        sys.exit(f"no snapshot under {a.cache}")
    files = sorted(glob.glob(os.path.join(snaps[0], "data", f"{a.split}-*.parquet")))
    if not files:
        sys.exit(f"no parquet for split {a.split}")

    wav_dir = os.path.join(ROOT, a.out, a.split)
    os.makedirs(wav_dir, exist_ok=True)
    manifest = os.path.join(ROOT, a.out, f"{a.split}.jsonl")

    jobs, rows = [], []
    seen = set()
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=2048,
                                     columns=["audio", "transcripts"]):
            for r in batch.to_pylist():
                au = r.get("audio") or {}
                blob = au.get("bytes")
                if not blob:
                    continue
                stem = os.path.splitext(os.path.basename(au.get("path") or ""))[0]
                if not stem or stem in seen:
                    continue
                seen.add(stem)
                txt = norm_text(r.get("transcripts"))
                if a.drop_empty and not txt:
                    continue
                jobs.append((blob, stem, os.path.join(wav_dir, f"{stem}.wav")))
                rows.append((stem, txt))
                if a.max_rows and len(jobs) >= a.max_rows:
                    break
            if a.max_rows and len(jobs) >= a.max_rows:
                break
        if a.max_rows and len(jobs) >= a.max_rows:
            break

    print(f"{a.split}: {len(jobs)} clips from {len(files)} parquet shards",
          flush=True)
    if not jobs:
        sys.exit("nothing to do")

    lengths, errs, samples = {}, 0, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(write_clip, j): i for i, j in enumerate(jobs)}
        for i, fu in enumerate(as_completed(futs), 1):
            n, err = fu.result()
            if err:
                errs += 1
                if len(samples) < 3:
                    samples.append(err)
            else:
                lengths[jobs[futs[fu]][1]] = n
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)} err={errs}", flush=True)

    import soundfile as sf
    n = tot = 0
    with open(manifest, "w", encoding="utf-8") as f:
        for stem, txt in rows:
            p = os.path.join(wav_dir, f"{stem}.wav")
            if not os.path.exists(p):
                continue
            i = sf.info(p)
            dur = round(i.frames / i.samplerate, 3)
            if dur <= 0:
                continue
            f.write(json.dumps({"audio": p, "text": txt, "lang": "ko",
                                "duration": dur, "source": "ksponspeech",
                                "id": stem}, ensure_ascii=False) + "\n")
            tot += dur
            n += 1

    print(f"wrote {n} rows ({tot/3600:.2f} h), {errs} errors -> {manifest}")
    for s in samples:
        print(f"  e.g. {s}")

    # duration distribution matters: eval on 2s clips is not the same as 20s
    if n:
        import random
        rs = random.Random(0)
        sample = [json.loads(l)["duration"] for l in open(manifest, encoding="utf-8")]
        s = sorted(rs.sample(sample, min(5000, len(sample))))
        print(f"  duration p05={s[len(s)//20]:.1f}s  median={s[len(s)//2]:.1f}s  "
              f"p95={s[int(len(s)*.95)]:.1f}s  max={s[-1]:.1f}s")


if __name__ == "__main__":
    main()