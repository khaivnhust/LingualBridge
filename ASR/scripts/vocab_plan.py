#!/usr/bin/env python3
"""Count token usage across the training corpus and pick a vocab keep-set.

The byte-token story, corrected
--------------------------------
An earlier version of this script claimed that dropping byte-level tokens from
the output head is what prevents foreign-language output. Measured against the
real tokenizer, that claim is wrong:

  '\u3053\u3093\u306b\u3061\u306f' -> ['\u00e3\u00a3\u0135\u00e3\u00a4\u0165\u00e3\u00a3\u00ab...']

That is ONE vocabulary entry, 12 characters wide, encoding 12 bytes of
Japanese. Its surface length is 12, not 1, so no len()==1 test can catch it.
And dropping single-char byte tokens is actively harmful, because the same
byte alphabet covers ASCII: '12345' -> ['1','2','3','4','5'], so deleting
byte tokens deletes the model's ability to write any number.

The real protection is corpus-driven and needs no byte awareness:

  keep exactly the tokens your 4-language training corpus uses.

Japanese and Russian tokens do not appear in a vi/en/zh/ko corpus, so they drop
out on their own. Verify that on your real corpus with --probe-foreign instead
of assuming it.

Two keep-sets are still produced, for the OOV trade-off:

  embd_keep : corpus merges + byte alphabet + all specials. Byte tokens stay
              so unseen input tokenizes instead of raising on OOV.
  out_keep  : corpus merges + format tokens + digits/punct + specials.
              Smaller, since input-only tolerance is dropped. Note this alone
              does NOT block foreign output; the corpus filter is what does.

Run against the real training corpus. Do not guess the threshold.
"""
import argparse, collections, json
from transformers import AutoTokenizer


def byte_alphabet():
    """The 256 chars GPT-2 byte-level BPE uses to represent bytes 0x00-0xFF."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c) for c in cs}


def is_byte_token(tok_str, alpha):
    """True only for single-char tokens drawn from the byte alphabet.

    Multi-char tokens are merges over several bytes; they are ordinary
    vocabulary, not byte fallbacks. Note that a foreign language can be spelled
    with ONE multi-char token, so this classification does not, and cannot,
    determine whether a token is safe to emit.
    """
    return len(tok_str) == 1 and tok_str in alpha


FOREIGN_PROBES = {
    "japanese": "こんにちは世界です私は学生です",
    "russian":  "Привет мир как дела сегодня",
    "arabic":   "مرحبا بالعالم كيف حالك اليوم",
    "thai":     "สวัสดีครับวันนี้อากาศเป็นอย่างไร",
    "hindi":    "नमस्ते दुनिया आप कैसे हैं",
    "greek":    "γεια σου κόσμε τι κάνεις σήμερα",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-tokens", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--drop-byte-from-output", type=int, default=1)
    ap.add_argument("--keep-digit-tokens", type=int, default=1,
                    help="digits and '.' are byte-alphabet tokens but load bearing for "
                         "numbers; excluding them would delete the ability to write 3.500.000")
    ap.add_argument("--probe-foreign", type=int, default=1,
                    help="check whether non-target scripts remain spellable from out_keep")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    n_full = len(tok)
    alpha = byte_alphabet()
    print(f"vocab_size={tok.vocab_size}  len(tok)={n_full}")

    surface = {}
    for i in range(n_full):
        s = tok.convert_ids_to_tokens(i)
        if s is not None:
            surface[i] = s

    byte_ids = {i for i, s in surface.items() if is_byte_token(s, alpha)}
    print(f"byte-alphabet tokens (single char, len==1): {len(byte_ids)}")

    used = collections.Counter()
    per_lang = collections.defaultdict(collections.Counter)
    n_lines = 0
    for path in a.corpus:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line) if path.endswith(".jsonl") else {"text": line}
            except json.JSONDecodeError:
                rec = {"text": line}
            t = rec.get(a.field, "")
            if not t:
                continue
            ids = tok(t, add_special_tokens=False).input_ids
            used.update(ids)
            n_lines += 1
            if rec.get("lang"):
                per_lang[rec["lang"]].update(ids)

    merge_keep = set(used)
    print(f"lines={n_lines}  distinct tokens used={len(merge_keep)} "
          f"({100*len(merge_keep)/n_full:.2f}%)")

    specials_all = set(range(151643, n_full))
    input_only = {151669, 151670, 151676}          # audio start/end/pad
    out_specials = specials_all - input_only

    # The ASR output format is fixed: "language <NAME><asr_text>". These tokens
    # are structural and must survive even if the corpus text lacks them.
    format_tokens = set()
    for name in ("Vietnamese", "English", "Chinese", "Korean"):
        for probe in (f"language {name}<asr_text>", f" {name}", "language"):
            format_tokens |= set(tok(probe, add_special_tokens=False).input_ids)
    format_tokens = {i for i in format_tokens if i < n_full}
    print(f"format tokens (language / names / <asr_text>): {len(format_tokens)}")

    # Digits and '.' are single-char byte-alphabet tokens, but ASR must emit
    # numbers. Keep them regardless of the corpus so the model can write them.
    digit_keep = set()
    if a.keep_digit_tokens:
        # Digits and punctuation, the uppercase letter X (it appears in "Xin
        # chao"), and the byte-encoded leading space marker G-with-dot, which
        # BPE prefixes to nearly every word and is not the ASCII space token.
        for ch in "0123456789.,%:?!;()'\"-XĠ":
            i = tok(ch, add_special_tokens=False).input_ids
            digit_keep |= {x for x in i if x < n_full}
        print(f"digit/punct/space tokens force-kept: {len(digit_keep)}")

    merge_out = merge_keep | format_tokens | digit_keep

    # Input keeps byte tokens so unseen input tokenizes; output does not need
    # input-only tolerance and is kept smaller.
    embd_keep = (merge_keep | byte_ids | specials_all) & set(range(n_full))
    out_keep = (merge_out | out_specials) & set(range(n_full))

    print(f"input  embedding rows: {len(embd_keep)} ({100*len(embd_keep)/n_full:.2f}%)")
    print(f"output head     rows: {len(out_keep)} ({100*len(out_keep)/n_full:.2f}%)")

    target = min(max(len(merge_keep), a.min_tokens), a.max_tokens)
    if target > len(merge_keep):
        print(f"budget {a.min_tokens}-{a.max_tokens}: corpus is smaller, keeping all used merges")
    else:
        print(f"budget {a.min_tokens}-{a.max_tokens}: keeping all {len(merge_keep)} used merges")

    res = {
        "n_full": n_full,
        "n_vocab_model": tok.vocab_size,
        "merges_used": len(merge_keep),
        "byte_ids": sorted(byte_ids),
        "input_only_specials": sorted(input_only),
        "specials_all": sorted(specials_all),
        "embd_keep": sorted(embd_keep),
        "out_keep": sorted(out_keep),
        "budget_min": a.min_tokens,
        "budget_max": a.max_tokens,
        "drop_byte_from_output": bool(a.drop_byte_from_output),
    }
    json.dump(res, open(a.out, "w"))
    print(f"wrote {a.out}")

    # invariants
    assert out_specials <= set(out_keep), "output specials missing from out_keep"
    assert specials_all <= set(embd_keep), "specials missing from embd_keep"
    assert merge_keep <= set(embd_keep), "corpus token missing from embd_keep"
    assert merge_out <= set(out_keep), "corpus/format token missing from out_keep"
    for name in ("Vietnamese", "English", "Chinese", "Korean"):
        ids = tok(f"language {name}<asr_text>", add_special_tokens=False).input_ids
        assert all(i in out_keep for i in ids), f"format broken for {name}: {ids}"

    if a.probe_foreign:
        print("\nforeign-language leak check (the property that actually matters):")
        leak_any = False
        for label, probe in FOREIGN_PROBES.items():
            ids = tok(probe, add_special_tokens=False).input_ids
            leaked = [i for i in ids if i in set(out_keep)]
            status = "LEAK" if leaked else "ok"
            if leaked:
                leak_any = True
            print(f"  {label:10s} {status:4s} {len(leaked)}/{len(ids)} tokens reachable"
                  + (f"  e.g. {[surface[i] for i in leaked[:6]]}" if leaked else ""))
        if leak_any:
            print("\n  WARNING: foreign text is partially spellable from out_keep.")
            print("  That means your corpus overlaps those scripts. Inspect before trimming.")
        else:
            print("  verified: none of these foreign scripts is spellable from out_keep")

    print("\nall invariants passed")

    if per_lang:
        print("\nper-language distinct tokens:")
        for l, c in sorted(per_lang.items()):
            print(f"  {l:12s} {len(c):6d}")


if __name__ == "__main__":
    main()
