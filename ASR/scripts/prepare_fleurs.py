#!/usr/bin/env python3
"""Download Google FLEURS for the 4 target languages and write JSONL manifests.

Output (one line per utterance, the same lang/text fields vocab_plan.py reads):
    {"audio": "/abs/path.wav", "text": "...", "lang": "ko", "duration": 9.3,
     "source": "fleurs", "id": "1888"}

    ASR/data/fleurs/<lang>/<split>/*.wav
    ASR/data/fleurs/<lang>_<split>.jsonl

FLEURS ids are sentence ids shared across languages (one sentence, several
speakers), so keep test/dev for evaluation only.

Usage:
    scripts/prepare_fleurs.py --splits dev,test
    scripts/prepare_fleurs.py --splits train --langs ko,vi
"""
import argparse, csv, io, json, os, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "fleurs")
FLEURS = {"en": "en_us", "zh": "cmn_hans_cn", "ko": "ko_kr", "vi": "vi_vn"}


def prepare(lang, split):
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    cfg = FLEURS[lang]
    tsv = hf_hub_download("google/fleurs", f"data/{cfg}/{split}.tsv", repo_type="dataset")
    rows = {}
    with open(tsv, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows[r[1]] = {"id": r[0], "text": r[2], "samples": int(r[5])}

    wav_dir = os.path.join(OUT, lang, split)
    os.makedirs(wav_dir, exist_ok=True)
    tar = hf_hub_download("google/fleurs", f"data/{cfg}/audio/{split}.tar.gz",
                          repo_type="dataset")
    n = 0
    manifest = os.path.join(OUT, f"{lang}_{split}.jsonl")
    with tarfile.open(tar) as tf, open(manifest, "w", encoding="utf-8") as out:
        for m in tf:
            name = os.path.basename(m.name)
            if name not in rows:
                continue
            path = os.path.join(wav_dir, name)
            if not os.path.exists(path):
                # FLEURS ships 32-bit float WAVs; store 16-bit PCM (half the size)
                audio, sr = sf.read(io.BytesIO(tf.extractfile(m).read()))
                sf.write(path, audio, sr, subtype="PCM_16")
            r = rows[name]
            out.write(json.dumps({
                "audio": path, "text": r["text"], "lang": lang,
                "duration": round(r["samples"] / 16000, 2),
                "source": "fleurs", "id": r["id"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"{lang} {split}: {n} utterances -> {manifest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", default="en,zh,ko,vi")
    ap.add_argument("--splits", default="train,dev,test")
    a = ap.parse_args()
    for split in a.splits.split(","):
        for lang in a.langs.split(","):
            prepare(lang.strip(), split.strip())


if __name__ == "__main__":
    main()
