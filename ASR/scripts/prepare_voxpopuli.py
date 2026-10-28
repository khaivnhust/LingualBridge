#!/usr/bin/env python3
"""Build an English ASR manifest from VoxPopuli (unscripted, conversational).

Why this dataset specifically
-----------------------------
Round 3 evaluation showed a SIGNIFICANT English regression from the QAT run
(+2.83 WER, 95% CI [+0.70, +5.31], paired on 100 utterances) while vi/ko/zh
were flat or improved. English was the one language with no conversational
training audio -- only 2,602 studio-read FLEURS utterances plus a MUSAN-noised
copy of the same 2,602.

VoxPopuli is unscripted speech from public talks and panels: real disfluency,
real pacing, real rooms. That is the distribution the regression showed up on.

The schema field is `raw_text`, not `text` -- raw preserves original casing and
punctuation, which is what a real transcript looks like and what training should
target (our metric normalises casing at scoring time anyway).

Usage:
    scripts/prepare_voxpopuli.py --shards 3
"""
import argparse
import glob
import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_RATE = 16000
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--facebook--voxpopuli/snapshots/*"))


def convert(job):
    """(blob, dst) -> (seconds, error)."""
    blob, dst = job
    if os.path.exists(dst):
        try:
            import soundfile as sf
            return sf.info(dst).frames / sf.info(dst).samplerate, None
        except Exception:                        # noqa: BLE001
            pass
    # NB: must keep a .wav suffix. soundfile picks the output format from the
    # extension, and "<name>.wav.part" fails for every single clip with
    # "unable to get format from file extension" -- same trap as the ffmpeg
    # outfile earlier. VoxPopuli ships 16 kHz mono already, so this is a pure
    # dtype cast rather than a resample.
    tmp = dst + ".part.wav"
    try:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
        sf.write(tmp, x, SAMPLE_RATE, subtype="PCM_16")
        os.replace(tmp, dst)
        return len(x) / SAMPLE_RATE, None
    except Exception as e:                        # noqa: BLE001
        return 0.0, f"{type(e).__name__}: {e}"
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--out", default="data/voxpopuli")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--gold-only", action="store_true", default=True)
    ap.add_argument("--any-transcript", dest="gold_only", action="store_false")
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    import pyarrow.parquet as pq

    if not SNAP:
        raise SystemExit("VoxPopuli not found in the HF cache")
    snap = SNAP[0]
    shards = sorted(glob.glob(os.path.join(snap, "en", "train-*.parquet")))[:a.shards]
    if not shards:
        raise SystemExit(f"no en/train-*.parquet under {snap}")

    out_root = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    wav_dir = os.path.join(out_root, "en")
    os.makedirs(wav_dir, exist_ok=True)

    jobs, meta = [], []
    for sh in shards:
        pf = pq.ParquetFile(sh)
        for batch in pf.iter_batches(batch_size=512):
            for r in batch.to_pylist():
                txt = (r.get("raw_text") or "").strip()
                au = r.get("audio") or {}
                blob = au.get("bytes")
                if not blob or not txt:
                    continue
                if a.gold_only and not r.get("is_gold_transcript"):
                    continue
                aid = r.get("audio_id") or os.path.splitext(
                    os.path.basename(au.get("path") or "x"))[0]
                dst = os.path.join(wav_dir, f"{abs(hash(aid)) % (10**12):012d}.wav")
                jobs.append((blob, dst))
                meta.append((dst, txt, aid))
        print(f"  {os.path.basename(sh)}: cumulative {len(jobs)} candidates",
              flush=True)

    print(f"converting {len(jobs)} utterances with {a.workers} workers...",
          flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    secs, errs = {}, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        for i, j in enumerate(jobs):
            futs[ex.submit(convert, j)] = i
        for n, f in enumerate(as_completed(futs), 1):
            d, err = f.result()
            if err:
                errs += 1
            else:
                secs[meta[futs[f]][0]] = d
            if n % 2000 == 0:
                print(f"    {n}/{len(jobs)} err={errs}", flush=True)

    manifest = os.path.join(out_root, "en.jsonl")
    kept, total_h = 0, 0.0
    with open(manifest, "w", encoding="utf-8") as f:
        for dst, txt, aid in meta:
            d = secs.get(dst)
            if d is None or not (a.min_seconds <= d <= a.max_seconds):
                continue
            f.write(json.dumps({"audio": dst, "text": txt, "lang": "en",
                                "duration": round(d, 3), "id": str(aid),
                                "source": "voxpopuli"}, ensure_ascii=False) + "\n")
            kept += 1
            total_h += d

    print(f"wrote {kept} rows ({total_h/3600:.2f} h), {errs} decode errors")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()