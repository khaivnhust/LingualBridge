#!/usr/bin/env python3
"""Transcribe FLEURS through a llama.cpp server, writing one hypothesis per line.

Design notes
------------
This replaced a first version that kept every hypothesis in RAM, printed
progress to a log, and wrote metrics only at the very end. That version was
killed by a signal partway through the run and took 3.4 h of compute with it,
producing nothing -- and its per-request serial client used exactly one of the
server's four slots, so it was 4x slower than it needed to be.

Hence:
  * results stream to a JSONL and are flushed per line, so a kill costs at most
    the in-flight batch and `--resume` picks up where it stopped;
  * requests run on a thread pool sized to the server's n_slots;
  * metrics live in score_hyps.py, not here, so transcription and scoring are
    independently re-runnable and any two hypothesis files can be compared on
    the utterances they share.

Usage:
    scripts/eval_gguf_server.py --out results/q4km_hyps.jsonl --split test
"""
import argparse
import base64
import collections
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = ["en", "zh", "ko", "vi"]


def load_split(split):
    rows = []
    for lang in LANGS:
        p = os.path.join(ROOT, f"data/fleurs/{lang}_{split}.jsonl")
        with open(p, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def load_rows(args):
    """Either a FLEURS split, or arbitrary ASR manifests via --manifest.

    Conversational corpora (KsponSpeech, Common Voice) are not in data/fleurs,
    so --manifest is the only way to score them through the server.

    --limit applies in BOTH branches. It previously only applied to --manifest,
    which silently ran the full FLEURS test split when a subset was requested
    and broke CPU-vs-Vulkan paired comparisons on identical utterances.
    """
    if args.manifest:
        import glob as _glob
        paths = [args.manifest] if os.path.isfile(args.manifest) else sorted(
            _glob.glob(args.manifest))
        if not paths:
            sys.exit(f"no manifests matched {args.manifest}")
        rows = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    r.setdefault("lang", "?")
                    rows.append(r)
    else:
        rows = load_split(args.split)

    if args.only_audio_from:
        # Restrict to exactly the audio paths already present in a hypotheses
        # file. The paired scorer can only compare utterances both sides cover,
        # and when the reference was drawn with a different sampler (e.g.
        # dump_base_transcripts.py shuffles) a --limit subset can overlap it on
        # only a few hundred rows -- which silently destroys the sample size.
        want = set()
        with open(args.only_audio_from, encoding="utf-8") as f:
            for line in f:
                want.add(json.loads(line)["audio"])
        have = {r["audio"] for r in rows}
        missing = len(want - have)
        if missing:
            print(f"WARNING: {missing} paths in {args.only_audio_from} are not "
                  f"in the selected rows and will be absent", flush=True)
        rows = [r for r in rows if r["audio"] in want]

    if args.limit:
        # deterministic, stratified across whatever languages are present
        import random as _random
        by = {}
        for r in rows:
            by.setdefault(r["lang"], []).append(r)
        out = []
        for lang in sorted(by):
            pool = by[lang]
            _random.Random(42).shuffle(pool)
            out += pool[:args.limit]
        rows = out
    return rows


def parse_asr(text):
    """Split llama.cpp's `language X<asr_text>...` envelope.

    Returns (language, transcript). language is None when the model never
    opened the envelope, which is the escape case the metrics count.
    """
    m = re.match(r"^\s*language\s+(\S+?)<asr_text>(.*)$", text, re.S)
    if not m:
        return None, text.strip()
    return m.group(1), m.group(2).strip()


def transcribe(url, wav, timeout, retries):
    with open(wav, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = json.dumps({"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}],
        "temperature": 0.0, "max_tokens": 256}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json", "User-Agent": "eval"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:          # noqa: BLE001 - report, retry, keep going
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8399/v1/chat/completions")
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default=None,
                    help="JSONL or glob to evaluate instead of a FLEURS split")
    ap.add_argument("--limit", type=int, default=0, help="max per language")
    ap.add_argument("--only-audio-from", default=None,
                    help="hypotheses JSONL; evaluate exactly those audio paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = set()
    if a.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["audio"])
                except Exception:
                    pass
        print(f"resume: {len(done)} already done", flush=True)

    rows = [r for r in load_rows(a) if r["audio"] not in done]
    if not rows:
        print("nothing to do")
        return

    lock = threading.Lock()
    fh = open(out, "a", encoding="utf-8")
    state = {"n": 0, "fail": 0}
    t0 = time.time()

    def work(r):
        try:
            raw = transcribe(a.url, r["audio"], a.timeout, a.retries)
            lang, text = parse_asr(raw)
            rec = {"audio": r["audio"], "lang": r["lang"], "duration": r.get("duration", 0),
                   "hyp": text, "emitted_language": lang, "label": a.label}
            ok = True
        except Exception as e:                      # noqa: BLE001
            rec = {"audio": r["audio"], "lang": r["lang"], "duration": r.get("duration", 0),
                   "hyp": None, "emitted_language": None, "error": f"{type(e).__name__}: {e}"}
            ok = False
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            state["n"] += 1
            state["fail"] += (not ok)
            if state["n"] % 50 == 0 or state["n"] == len(rows):
                el = time.time() - t0
                rate = el / state["n"]
                print(f"  {state['n']}/{len(rows)}  {el:.0f}s  "
                      f"{rate:.2f}s/utt  eta {rate * (len(rows) - state['n']):.0f}s  "
                      f"fail={state['fail']}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, rows))
    fh.close()
    print(f"wrote {state['n']} rows ({state['fail']} failed) to {out} "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()