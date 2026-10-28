#!/usr/bin/env python3
"""Convert Common Voice tars into ASR manifests with 16 kHz PCM WAVs.

Common Voice on HuggingFace ships audio as `{lang}_{split}_0.tar` containing
loose mp3s, plus TSV metadata. FLEURS is stored as 16-bit PCM WAV at 16 kHz, so
this converts to match and emits the same manifest schema:

    {"audio", "text", "lang", "duration", "source", "id", ...}

mp3 bytes are streamed straight from the tar into ffmpeg over stdin, so the
~20 GB of mp3 is never written to disk a second time.

Two traps in this dataset:

1. **The official train split is small.** On this mirror CV ko/train is 0.91 h
   against FLEURS ko/train's ~6.9 h. Most clips live in the `other` split,
   which exists precisely because those sentences are not assigned to
   train/dev/test. So the useful training data is mostly in `other`.

2. **`other` may contain sentences that also appear in dev/test.** CV splits by
   sentence, so `other` is excluded from the official splits for exactly that
   reason. Training on `other` while evaluating on CV test can therefore leak.
   `--exclude-split-sentences` drops from `other` any sentence that appears in
   dev or test. It is on by default; `--no-dedupe` turns it off.

Usage:
    scripts/prepare_commonvoice.py --langs ko,vi --splits train,dev,test,other
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_RATE = 16000


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def index_tar(path):
    """basename -> full member name, for every mp3 in the archive.

    Members are stored as `ko_train_0/common_voice_ko_123.mp3`, but the TSV
    `path` column is a bare basename, so tarfile.getmember() must be given the
    full name while matching is done on the basename.
    """
    idx = {}
    with tarfile.open(path, "r") as t:
        for m in t:
            if m.isfile() and m.name.lower().endswith(".mp3"):
                idx[os.path.basename(m.name)] = m.name
    return idx


def decode_one(job):
    """tar_path, member, raw_bytes -> wav on disk. Returns (dst, err)."""
    tar_path, member, blob, dst = job
    if os.path.exists(dst):
        return dst, None
    # NB: must keep a .wav suffix -- ffmpeg picks the output muxer from the
    # filename extension, and "....wav.tmp" makes it fail on all 519 clips.
    tmp = dst + ".part.wav"
    try:
        p1 = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", "pipe:0",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le", tmp],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        _, err = p1.communicate(input=blob, timeout=180)
        if p1.returncode != 0 or not os.path.exists(tmp):
            return dst, (err.decode()[:200] or
                         f"ffmpeg rc={p1.returncode}, no output")
        os.replace(tmp, dst)
        return dst, None
    except Exception as e:                              # noqa: BLE001
        return dst, f"{type(e).__name__}: {e}"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def split_sentences(cache, lang, splits):
    """Sentence texts used by the given splits, for leakage exclusion."""
    banned = set()
    for s in splits:
        p = os.path.join(cache, "transcript", lang, f"{s}.tsv")
        if os.path.exists(p):
            for r in read_tsv(p):
                if r.get("sentence"):
                    banned.add(r["sentence"].strip().lower())
    return banned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--fsicoli--common_voice_22_0"))
    ap.add_argument("--langs", required=True)
    ap.add_argument("--splits", default="train,dev,test,other")
    ap.add_argument("--out", default="data/commonvoice")
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-per-split", type=int, default=0)
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                    default=True)
    ap.add_argument("--eval-splits", default="dev,test",
                    help="splits whose sentences are excluded from training")
    ap.add_argument("--tars-only", action="store_true",
                    help="stop after locating tars (no conversion)")
    a = ap.parse_args()

    snap = glob.glob(os.path.join(a.cache, "snapshots", "*"))
    if not snap:
        sys.exit(f"no snapshot under {a.cache}")
    cache = snap[0]
    out_root = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    totals = {}

    for lang in [x.strip() for x in a.langs.split(",") if x.strip()]:
        banned = split_sentences(cache, lang,
                                 a.eval_splits.split(",")) if a.dedupe else set()
        if banned:
            print(f"{lang}: {len(banned)} sentences reserved by "
                  f"{a.eval_splits} (excluded from other)", flush=True)

        for split in [x.strip() for x in a.splits.split(",") if x.strip()]:
            tars = sorted(glob.glob(os.path.join(cache, "audio", lang,
                                                 split, "*.tar")))
            tsv = os.path.join(cache, "transcript", lang, f"{split}.tsv")
            if not tars or not os.path.exists(tsv):
                print(f"  {lang}/{split}: missing tar or tsv, skipped")
                continue

            owner = {}                       # basename -> (tar path, full member name)
            for tp in tars:
                for base, member in index_tar(tp).items():
                    owner[base] = (tp, member)
            print(f"  {lang}/{split}: {len(owner)} mp3 in "
                  f"{len(tars)} tar(s)", flush=True)
            if a.tars_only:
                continue

            rows = [r for r in read_tsv(tsv)
                    if r.get("sentence") and r.get("path")
                    and os.path.basename(r["path"]) in owner]
            dropped = 0
            if split not in a.eval_splits.split(",") and banned:
                before = len(rows)
                rows = [r for r in rows
                        if r["sentence"].strip().lower() not in banned]
                dropped = before - len(rows)
            if a.max_per_split:
                rows = rows[:a.max_per_split]
            if dropped:
                print(f"    dropped {dropped} rows sharing sentences with "
                      f"{a.eval_splits}", flush=True)

            wav_dir = os.path.join(out_root, lang, split)
            os.makedirs(wav_dir, exist_ok=True)
            kept, manifest_rows, errs, sample_errors = [], [], 0, []

            # One tar at a time: reading every archive's mp3 bytes up front
            # would hold the whole split (up to 20 GB for zh-CN) in RAM.
            by_tar = {}
            for r in rows:
                tp, _member = owner[os.path.basename(r["path"])]
                by_tar.setdefault(tp, []).append(r)

            for tp, trows in by_tar.items():
                jobs = []
                with tarfile.open(tp, "r") as t:
                    for r in trows:
                        base = os.path.basename(r["path"])
                        m = t.getmember(owner[base][1])
                        f = t.extractfile(m)
                        if f is None:
                            continue
                        stem = os.path.splitext(base)[0]
                        dst = os.path.join(wav_dir, f"{stem}.wav")
                        jobs.append((tp, m, f.read(), dst))
                        kept.append((dst, r, stem))

                print(f"    {os.path.basename(tp)}: {len(jobs)} clips",
                      flush=True)
                with ProcessPoolExecutor(max_workers=a.workers) as ex:
                    futs = [ex.submit(decode_one, j) for j in jobs]
                    for i, fu in enumerate(as_completed(futs), 1):
                        _, err = fu.result()
                        if err:
                            errs += 1
                            if len(sample_errors) < 3:
                                sample_errors.append(err)
                        if i % 2000 == 0:
                            print(f"      {i}/{len(futs)} err={errs}",
                                  flush=True)
                del jobs

            import soundfile as sf
            manifest = os.path.join(out_root, lang, f"{split}.jsonl")
            n = 0
            with open(manifest, "w", encoding="utf-8") as f:
                for dst, r, stem in kept:
                    if not os.path.exists(dst):
                        continue
                    info = sf.info(dst)
                    f.write(json.dumps({
                        "audio": dst, "text": r["sentence"].strip(),
                        "lang": lang,
                        "duration": round(info.frames / info.samplerate, 3),
                        "source": "commonvoice-22", "id": stem,
                        "client_id": r.get("client_id"),
                        "up_votes": r.get("up_votes"),
                        "down_votes": r.get("down_votes"),
                        "accents": r.get("accents"),
                    }, ensure_ascii=False) + "\n")
                    n += 1
            totals[f"{lang}/{split}"] = n
            print(f"    -> {n} rows, {errs} errors, {manifest}", flush=True)
            if errs:
                for e in sample_errors[:3]:
                    print(f"      e.g. {e}", flush=True)

    print("\nsummary:")
    for k, v in totals.items():
        print(f"  {k:<14} {v}")
    print(f"wav root: {out_root}")


if __name__ == "__main__":
    main()