#!/usr/bin/env python3
"""Score ASR hypothesis files against FLEURS references, with paired deltas.

Any two hypothesis files can be compared -- the BF16 base-emission dump
(`text` key) and a llama.cpp run (`hyp` key) are both accepted. With `--base`
the comparison is *paired*: only utterances present in both files are scored,
so the reported delta removes subset-sampling noise. That matters because the
quantization effect being measured (~0.2 CER) is smaller than the standard
error of an independent 250-utterance sample (~1.2 points).

Usage:
    scripts/score_hyps.py --hyps results/q4km_hyps.jsonl \\
        --base data/base_emissions_test.jsonl
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = ["en", "zh", "ko", "vi"]
METRIC = {"en": "wer", "vi": "wer", "zh": "cer", "ko": "cer"}
LANGNAME = {"en": "english", "zh": "chinese", "ko": "korean", "vi": "vietnamese"}


def normalize(text, lang):
    """Identical to train_asr.py::normalize."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        ch if not unicodedata.category(ch).startswith(("P", "S")) else " "
        for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "") if lang in ("zh", "ko") else text


def load_hyps(path):
    """Accepts one path or a glob, and both the `text` (HF emissions) and
    `hyp` (llama.cpp) schemas."""
    out = {}
    key_used = None
    paths = sorted(glob.glob(path)) if glob.has_magic(path) else [path]
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                hyp = r.get("hyp", r.get("text"))
                if hyp is None:
                    continue
                out[r["audio"]] = {
                    "hyp": hyp,
                    "emitted_language": r.get("emitted_language"),
                    "lang": r.get("lang"),
                }
                key_used = "hyp" if "hyp" in r else "text"
    if not out:
        sys.exit(f"no hypotheses loaded from {path!r}")
    return out, key_used


def load_refs(split="test", manifest=None):
    """Reference transcripts, keyed by audio path.

    `manifest` lets any ASR-style JSONL act as the reference set -- required for
    the conversational corpora (KsponSpeech, Common Voice), which are not in
    data/fleurs/ and so cannot be scored against the built-in loader.
    """
    refs = {}
    if manifest:
        paths = [manifest] if os.path.isfile(manifest) else sorted(
            glob.glob(manifest))
        if not paths:
            sys.exit(f"no manifests matched {manifest}")
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("audio") and r.get("text") is not None:
                        refs[r["audio"]] = (r["text"], r.get("lang", "?"))
        return refs
    for lang in LANGS:
        p = os.path.join(ROOT, f"data/fleurs/{lang}_{split}.jsonl")
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                refs[r["audio"]] = (r["text"], r["lang"])
    return refs


def score(hyps, refs, restrict=None):
    per = collections.defaultdict(
        lambda: {"ref": [], "hyp": [], "lid": 0, "escape": 0, "n": 0})
    for audio, h in hyps.items():
        if audio not in refs:
            continue
        if restrict is not None and audio not in restrict:
            continue
        ref, lang = refs[audio]
        d = per[lang]
        d["n"] += 1
        d["ref"].append(normalize(ref, lang))
        d["hyp"].append(normalize(h["hyp"], lang))
        lid = h["emitted_language"]
        d["lid"] += lid is None or str(lid).lower() == LANGNAME[lang]
        d["escape"] += lid is None
    return per


def metrics(per, label):
    import jiwer
    res = {"label": label}
    for lang in LANGS:
        d = per.get(lang)
        if not d or not d["n"]:
            continue
        m = METRIC[lang]
        fn = jiwer.cer if m == "cer" else jiwer.wer
        res[lang] = {"n": d["n"], m: round(fn(d["ref"], d["hyp"]) * 100, 2),
                     "lid_acc": round(100 * d["lid"] / d["n"], 1),
                     "escape_rate": round(100 * d["escape"] / d["n"], 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--base", default=None, help="paired baseline hyps file")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref-manifest", default=None,
                    help="JSONL (or glob) to score against instead of FLEURS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    refs = load_refs(a.split, a.ref_manifest)
    new, k1 = load_hyps(a.hyps)
    per_new = score(new, refs)
    rows = [metrics(per_new, f"new ({k1})")]

    if a.base:
        old, k2 = load_hyps(a.base)
        shared = set(new) & set(old)
        per_old = score(old, refs, restrict=shared)
        per_pair_new = score(new, refs, restrict=shared)
        rows = [metrics(per_old, f"base ({k2})"),
                metrics(per_pair_new, f"new  ({k1})")]
        print(f"paired on {len(shared)} shared utterances "
              f"(of {len(new)} new / {len(old)} base)\n")

    hdr = f"{'lang':<5}{'n':>6}{'metric':>7}"
    for r in rows:
        hdr += f"{r['label'][:12]:>16}"
    if len(rows) > 1:
        hdr += f"{'delta':>9}"
    print(hdr)
    print("-" * len(hdr))
    out = {}
    present = [l for l in LANGS
               if any(r.get(l) for r in rows)]
    # a non-FLEURS corpus may carry langs outside LANGS; surface them too.
    # refs values are (text, lang) tuples.
    for lang in sorted({v[1] for v in refs.values()}):
        if lang not in present and lang not in LANGS:
            present.append(lang)
    for lang in present:
        cells = [r.get(lang) for r in rows]
        if not any(cells):
            continue
        m = METRIC.get(lang, "cer")
        n = cells[0]["n"] if cells[0] else cells[1]["n"]
        line = f"{lang:<5}{n:>6}{m:>7}"
        for c in cells:
            line += f"{(c[m] if c else float('nan')):>16.2f}"
        if len(cells) > 1 and cells[0] and cells[1]:
            line += f"{cells[1][m] - cells[0][m]:>+9.2f}"
        print(line)
        out[lang] = {r["label"]: r.get(lang) for r in rows}

    for r in rows:
        lid = [r[l]["lid_acc"] for l in present if l in r]
        esc = [r[l]["escape_rate"] for l in present if l in r]
        if lid:
            print(f"{'LID acc':<18}{min(lid):.1f}-{max(lid):.1f}  "
                  f"(mean {sum(lid) / len(lid):.1f})   "
                  f"escape {max(esc):.1f}", end="   " if len(rows) > 1 else "\n")
    print()

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "paired": out}, f, indent=2, ensure_ascii=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()