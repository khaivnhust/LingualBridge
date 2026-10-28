#!/usr/bin/env python3
"""Build eval manifests from the "Round 3" OneVoice sample dataset.

Four Vietnamese-centric parallel corpora:

  Vie-Kor  180 ids  vie_voice/vie_text + kor_voice/kor_text
  Vie-Man  180 ids  vie_voice/vie_text + man_voice/man_text
  Vie-Eng  100 ids  Vie-Eng/wav (vi) + Eng-Vie/wav (en)

Audio is OneVoice-synthesised and ships at mixed sample rates (24 k and 48 k),
so everything is resampled to 16 kHz mono PCM to match every other corpus here.

Vie-Eng differs from the other two: its xlsx has only (id, tiếng việt,
tiếng anh - final) with no voice_path column, so paths are resolved by matching
the id prefix to the wav directories instead.

Each (audio, text) pair becomes its own row tagged with its TRUE language, so
the model is never asked to transcribe Korean against Vietnamese ground truth.
"""
import argparse
import json
import os

import openpyxl
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# unzipped "Round 3-Bộ dữ liệu mẫu" sample dataset (not in git); override with ROUND3_DIR
SRC = os.environ.get("ROUND3_DIR", "data/round3_raw")
SR = 16000


def to16k(src, dst):
    x, sr = sf.read(src, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    sf.write(dst, x, SR, subtype="PCM_16")
    return len(x) / SR


def sheet_rows(path):
    ws = openpyxl.load_workbook(path).worksheets[0]
    it = ws.iter_rows(values_only=True)
    hdr = [str(h).strip().lower() if h else "" for h in next(it)]
    return hdr, list(it)


def pair_dirs(xlsx_rel, audio_root):
    """Yield (lang, id, text, wav_path) for BOTH languages of a pair."""
    p = os.path.join(SRC, xlsx_rel)
    hdr, rows = sheet_rows(p)

    def find(*subs):
        for i, h in enumerate(hdr):
            if all(s in h for s in subs):
                return i
        return None

    ci = find("id")
    # Vie-Eng: no voice_path column; match by id prefix to the wav dirs.
    has_path = find("voice_path") is not None
    c_vie, c_kor = find("vie_text"), find("kor_text")
    c_man = find("man_text")
    c_en_v, c_en_e = find("tiếng việt"), find("tiếng anh")
    if c_vie is None:
        c_vie = c_en_v
    if c_en_e is None:
        c_en_e = find("tiếng anh - final")

    for r in rows:
        if not r or ci is None or not r[ci]:
            continue
        ident = str(r[ci]).strip()

        # Vietnamese
        t = r[c_vie] if c_vie is not None else None
        if t and str(t).strip():
            if has_path:
                w = os.path.join(SRC, audio_root, "Vie_voice", ident + ".wav")
            else:
                w = os.path.join(SRC, audio_root, "Vie-Eng", "wav", ident + ".wav")
            if os.path.exists(w):
                yield "vi", ident, str(t).strip(), w

        # second language
        for col, sub in ((c_kor, "Kor_voice"), (c_man, "Man_voice")):
            if col is None or not r[col] or not str(r[col]).strip():
                continue
            w = os.path.join(SRC, audio_root, sub, ident + ".wav")
            if os.path.exists(w):
                yield {"Kor_voice": "ko", "Man_voice": "zh"}[sub], ident, str(r[col]).strip(), w

        if c_en_e is not None and r[c_en_e] and str(r[c_en_e]).strip():
            w = os.path.join(SRC, audio_root, "Eng-Vie", "wav", ident + ".wav")
            if os.path.exists(w):
                yield "en", ident, str(r[c_en_e]).strip(), w


def english_from_e2v():
    """Eng-Vie/wav holds EV-* audio, described by a SEPARATE sheet.

    The V->E sheet only describes VE-* (Vietnamese) audio; the E->V sheet has
    its own EV-* ids and its own column order (Tiếng Anh first). Treating them
    as one sheet silently drops all 100 English clips.
    """
    p = os.path.join(SRC, "Vie-Eng dataset/Eng-Vie/OneVoice_E_to_V_samples_100.xlsx")
    if not os.path.exists(p):
        return
    hdr, rows = sheet_rows(p)

    def find(*subs):
        for i, h in enumerate(hdr):
            if all(s in h for s in subs):
                return i
        return None

    ci = find("id")
    c_en = find("tiếng anh")
    for r in rows:
        if not r or ci is None or not r[ci]:
            continue
        ident = str(r[ci]).strip()
        if c_en is None or not r[c_en] or not str(r[c_en]).strip():
            continue
        w = os.path.join(SRC, "Vie-Eng dataset", "Eng-Vie", "wav", ident + ".wav")
        if os.path.exists(w):
            yield "en", ident, str(r[c_en]).strip(), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="data/round3")
    a = ap.parse_args()
    out_root = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    rows = {"vi": [], "ko": [], "zh": [], "en": []}
    seen = set()

    print(f"  Eng-Vie/E_to_V sheet: {len(list(english_from_e2v()))} pairs")
    for xlsx, root in [
        ("Vie-Kor dataset/OneVoice_V_to_K_samples_180.xlsx", "Vie-Kor dataset"),
        ("Vie-Man dataset/OneVoice_V_to_M_samples_180.xlsx", "Vie-Man dataset"),
        ("Vie-Eng dataset/Vie-Eng/OneVoice_V_to_E_samples_100.xlsx", "Vie-Eng dataset"),
    ]:
        try:
            items = list(pair_dirs(xlsx, root))
        except Exception as e:                    # noqa: BLE001
            print(f"  {os.path.basename(xlsx)}: ERROR {e}")
            continue
        print(f"  {os.path.basename(xlsx)}: {len(items)} (audio,text) pairs")
        items += list(english_from_e2v())

        for lang, ident, text, wav in items:
            key = (lang, os.path.basename(wav))
            if key in seen:
                continue
            seen.add(key)
            d = os.path.join(out_root, lang, os.path.basename(root))
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, os.path.basename(wav))
            try:
                dur = to16k(wav, dst)
            except Exception as e:                # noqa: BLE001
                print(f"    skip {wav}: {e}")
                continue
            rows[lang].append({"audio": dst, "text": text, "lang": lang,
                               "duration": round(dur, 3), "id": ident,
                               "source": f"round3/{os.path.basename(root)}"})

    total = 0
    for lang in ("vi", "ko", "zh", "en"):
        R = sorted(rows[lang], key=lambda r: (r["source"], r["id"]))
        if not R:
            continue
        p = os.path.join(out_root, f"{lang}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in R:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{lang:<4} {len(R):>4} rows  {sum(r['duration'] for r in R)/3600:>6.2f} h"
              f"  mean={sum(r['duration'] for r in R)/len(R):.2f}s")
        total += len(R)
    print(f"TOTAL {total} utterances")


if __name__ == "__main__":
    main()