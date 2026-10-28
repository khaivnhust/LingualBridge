#!/usr/bin/env python3
"""Summarise eos_bench.sh output: last speech -> first MT token, and ASR accuracy.

  eos_report.py <dir with A.jsonl B.jsonl C.jsonl> <audio_index.json>

Each clip may be split by the VAD into several utterances; latency is reported
per utterance, accuracy per clip (utterance transcripts joined in order).
"""
import json
import re
import sys
import unicodedata
from collections import defaultdict

import jiwer

d, index = sys.argv[1], sys.argv[2]
refs = {r["id"]: r for r in json.load(open(index, encoding="utf-8"))}
METRIC = {"en": "wer", "vi": "wer", "zh": "cer", "ko": "cer"}


def norm(s, lang):
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "") if lang in ("zh", "ko") else s


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")


def err(pairs, lang):
    r = [norm(a, lang) for a, _ in pairs]
    h = [norm(b, lang) for _, b in pairs]
    keep = [(a, b) for a, b in zip(r, h) if a]
    r, h = [a for a, _ in keep], [b for _, b in keep]
    return 100 * (jiwer.cer(r, h) if METRIC[lang] == "cer" else jiwer.wer(r, h))


rows = {}
for cfg in "ABC":
    try:
        rows[cfg] = [json.loads(l) for l in open(f"{d}/{cfg}.jsonl", encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        pass

print(f"{'cfg':4}{'lang':5}{'n':>4} | {'last->MT1 med':>13}{'p90':>7}{'max':>7}{'>2s':>6} | "
      f"{'VAD->MT1 med':>12}{'ASR med':>8}{'MT1 med':>8} | {'err':>6}")
for cfg, rs in rows.items():
    by = defaultdict(list)
    for r in rs:
        by[r["id"][:2]].append(r)
    by["all"] = rs
    for lang in ["en", "vi", "zh", "ko", "all"]:
        g = by.get(lang, [])
        if not g:
            continue
        m1 = [r["last_to_mt1_ms"] for r in g if r["last_to_mt1_ms"] >= 0]
        d1 = [r["detect_to_mt1_ms"] for r in g if r["detect_to_mt1_ms"] >= 0]
        a = [r["asr_ms"] for r in g]
        t1 = [r["mt_first_token_ms"] for r in g if r["mt_first_token_ms"] >= 0]
        if lang != "all":
            clips = defaultdict(list)
            for r in g:
                clips[r["id"]].append(r["asr_text"])
            e = f"{err([(refs[k]['text'], ' '.join(v)) for k, v in clips.items()], lang):6.2f}"
        else:
            e = "     -"
        over = sum(x > 2000 for x in m1)
        print(f"{cfg:4}{lang:5}{len(g):4} | {q(m1,.5):13.0f}{q(m1,.9):7.0f}{max(m1):7.0f}"
              f"{over:4}/{len(m1):<3}| {q(d1,.5):12.0f}{q(a,.5):8.0f}{q(t1,.5):8.0f} | {e}")
    print()

if "C" in rows:
    c = rows["C"]
    print("C detail: segments/utt median", q([r["stitch_segments"] for r in c], .5),
          "| pause cuts", sum(r["n_pause_trig"] for r in c),
          "| tail audio med %.2fs" % q([r["tail_s"] for r in c], .5),
          "| tail decode med %.0f ms" % q([r["tail_ms"] for r in c], .5))
    worst = sorted(c, key=lambda r: -r["last_to_mt1_ms"])[:5]
    for r in worst:
        print(f"  worst {r['id']} {r['audio_s']:.1f}s last->MT1 {r['last_to_mt1_ms']} "
              f"asr_done {r['last_to_asr_done_ms']} tail {r['tail_s']:.2f}s/{r['tail_ms']}ms "
              f"segs {r['stitch_segments']} vad {r['vad_penalty_ms']:.0f}")
