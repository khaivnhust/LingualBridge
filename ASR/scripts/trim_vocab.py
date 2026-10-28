#!/usr/bin/env python3
"""Trim Qwen3-ASR's vocabulary to EN/VI/KO/ZH and write a loadable checkpoint.

Keep-set (all measured, nothing guessed):
  1. every token the tokenizer produces on the reference text: a Wikipedia
     sample per language plus the training transcripts (count >= --min-count)
  2. the single-char token of every character seen in that text, so no
     in-language character falls back to byte spelling
  3. the 256-char byte alphabet (any input still tokenizes), digits/punct,
     the "language <Name><asr_text>" format tokens, and all special tokens
  4. merge closure: every BPE ancestor of a kept token. Without it the
     trimmed tokenizer would need merges whose inputs no longer exist.

Kept merges are exactly those whose result is kept, so the trimmed tokenizer
segments in-language text identically to the original (verified below).

Unlike vocab_plan.py's separate input/output sets, this produces ONE
vocabulary: the HF model ties embed_tokens and lm_head, so both must match.
A smaller output-only head (n_vocab_out) can still be applied at GGUF export.

Output: <out>/ (model.safetensors, configs, trimmed tokenizer, processor files,
keep_ids.json mapping new id -> original id)

Usage:
    scripts/trim_vocab.py --out weights/qwen3-asr-1.7B-4lang-trim \\
        --corpus data/fleurs/*_train.jsonl
"""
import argparse, collections, glob, json, os, random, re, shutil, sys, unicodedata

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vocab_plan import byte_alphabet, FOREIGN_PROBES  # noqa: E402

LANGS = {"en": "English", "vi": "Vietnamese", "ko": "Korean", "zh": "Chinese"}
WIKI = "wikimedia/wikipedia"
WIKI_SNAPSHOT = "20231101"


def wiki_text(lang, max_chars, cache_dir):
    """First `max_chars` characters of one Wikipedia shard, as lines."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WIKI, f"{WIKI_SNAPSHOT}.{lang}/train-00000-of-"
                           f"{ {'en': '00041', 'vi': '00004', 'ko': '00003', 'zh': '00006'}[lang] }.parquet",
                           repo_type="dataset", cache_dir=cache_dir)
    lines, n = [], 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["text"], batch_size=256):
        for text in batch.column(0).to_pylist():
            for line in text.split("\n"):
                line = line.strip()
                if len(line) >= 20:
                    lines.append(line)
                    n += len(line)
            if n >= max_chars:
                return lines
    return lines


# Characters the 4 target languages (and English code-switching) can need.
# Han covers Chinese (and the Hanja that appears in Korean); kana, Cyrillic,
# Greek, Arabic, Thai, Devanagari, emoji, etc. are excluded.
ALLOWED_RANGES = [
    (0x0000, 0x024F),   # ASCII, Latin-1, Latin Extended-A/B
    (0x0300, 0x036F),   # combining diacritics (decomposed Vietnamese)
    (0x1E00, 0x1EFF),   # Latin Extended Additional (Vietnamese precomposed)
    (0x2000, 0x2BFF),   # punctuation, currency, number forms, arrows, math, misc symbols
    (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F),
    (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF),               # Hangul
    (0x3000, 0x303F),   # CJK punctuation
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),  # Han
    (0xFF00, 0xFFEF),   # fullwidth forms
]


def allowed_char(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ALLOWED_RANGES)


def _byte_decoder():
    """Inverse of GPT-2 bytes_to_unicode: token char -> byte value."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


def token_text(tok_str):
    """Decoded text of a byte-level token, or None if it is partial UTF-8."""
    try:
        return bytes(BYTE_DECODER[c] for c in tok_str).decode("utf-8")
    except (UnicodeDecodeError, KeyError):
        return None


def numeral_probe_ids(tok, n_regular):
    """Force-keep the tokens the model needs to *spell* numbers.

    Trimming on frequency alone drops spelled-out numerals ("twentieth",
    "六", "百") long before it drops digits, because they are rarer in the
    corpus than the digits that FLEURS transcripts prefer. The model then
    drifts to whichever numeral spelling survived: "20th" -> "Twentieth",
    "6比6" -> "六比六". Those are real content changes, not cosmetic ones, so
    numeral vocabulary is force-kept rather than left to the frequency cut.

    Both orthographies are kept. Deciding that only one may be emitted is a
    language-policy question the corpus cannot answer, and keeping both costs
    a few hundred tokens.
    """
    words = [
        # English cardinals, including the compounds that tokenize whole
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
        "thirty forty fifty sixty seventy eighty ninety hundred thousand million "
        "billion trillion",
        # English ordinals, which are the ones actually observed drifting
        "first second third fourth fifth sixth seventh eighth ninth tenth "
        "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
        "eighteenth nineteenth twentieth thirtieth fortieth fiftieth hundredth",
        # ordinal suffixes, the "20th"/"3rd" path
        "1st 2nd 3rd 4th 10th 11th 12th 13th 21st 22nd 23rd 100th 101st",
        # Vietnamese
        "không một hai ba bốn năm sáu bảy tám chín mười trăm nghìn triệu tỷ "
        "thứ nhất thứ hai thứ ba thứ tư thứ năm thứ sáu thứ bảy thứ tám thứ chín thứ mười",
        # Chinese numerals, simplified + traditional digits
        "零一二三四五六七八九十百千万亿萬億兩两百廿卅",
        # common numeric contexts
        "1st 2nd 3rd 2024 2025 2026 100% 1,000 3.14 10:30 -5 +5 1/2",
    ]
    ids = set()
    for enc in tok.encode_batch(words, add_special_tokens=False):
        ids.update(i for i in enc.ids if i < n_regular)
    # full-width / half-width variants: the zh diffs showed "（FAA）" where the
    # base emitted "(FAA)". Keep both so punctuation style cannot drift.
    punct = list("（）「」【】，、。：；？！％－／")
    for enc in tok.encode_batch(punct + [" " + c for c in punct],
                                add_special_tokens=False):
        ids.update(i for i in enc.ids if i < n_regular)
    return ids


def manifest_text(patterns):
    lines = []
    for pat in patterns:
        for p in glob.glob(pat if os.path.isabs(pat) else os.path.join(ROOT, pat)):
            with open(p, encoding="utf-8") as f:
                lines += [json.loads(l)["text"] for l in f if l.strip()]
    return lines


def main():
    ap = argparse.ArgumentParser(description="Trim Qwen3-ASR vocab to EN/VI/KO/ZH")
    ap.add_argument("--base", default="Qwen/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", nargs="*", default=[],
                    help="training manifests (JSONL with 'text'); globs allowed")
    ap.add_argument("--emissions", nargs="*", default=[],
                    help="manifests of BASE-MODEL greedy transcripts "
                         "(scripts/dump_base_transcripts.py). Every token they "
                         "use is force-kept, and their coverage is reported per "
                         "--min-count in the sweep. Build from TRAIN and verify "
                         "on TEST so the threshold is not fitted to the report set.")
    ap.add_argument("--wiki-chars", type=int, default=20_000_000,
                    help="characters of Wikipedia text per language")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--mode", choices=["frequency", "coverage"], default="frequency",
                    help="frequency: keep tokens seen >= --min-count times (small, "
                         "but byte-spells rare in-language orthography). "
                         "coverage: keep every in-script token (larger, but "
                         "in-language spelling is always available)")
    ap.add_argument("--verify-audio", nargs="*",
                    default=[f"data/fleurs/{l}_test.jsonl" for l in LANGS],
                    help="manifests for base-vs-trimmed transcription check. "
                         "Defaults to TEST: acceptance is measured on the split "
                         "you report numbers on, so never choose the threshold on it.")
    ap.add_argument("--verify-per-lang", type=int, default=200,
                    help="25-utterance samples miss rare-token regressions")
    ap.add_argument("--min-equivalent", type=float, default=95.0,
                    help="acceptance gate: %% of test transcripts whose CONTENT is "
                         "unchanged vs base, ignoring punctuation and LID tag")
    ap.add_argument("--sweep-only", action="store_true",
                    help="print the vocab size for several --min-count values and exit")
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    base_dir = snapshot_download(a.base)
    tok = Tokenizer.from_file(os.path.join(base_dir, "tokenizer.json"))
    tj = json.load(open(os.path.join(base_dir, "tokenizer.json"), encoding="utf-8"))
    vocab = tj["model"]["vocab"]                      # str -> id (regular tokens)
    id2tok = {i: s for s, i in vocab.items()}
    added = {t["id"]: t["content"] for t in tj["added_tokens"]}
    n_regular = len(vocab)
    print(f"base: {n_regular} regular + {len(added)} added tokens, {len(tj['model']['merges'])} merges")

    # ── 1-2. measure usage on reference text ──
    texts = {}
    for lang in LANGS:
        texts[lang] = wiki_text(lang, a.wiki_chars, None)
        print(f"wiki {lang}: {len(texts[lang])} lines, {sum(map(len, texts[lang]))/1e6:.1f}M chars")
    corpus = manifest_text(a.corpus)
    print(f"training transcripts: {len(corpus)} lines")

    counts = collections.Counter()
    chars = set()
    all_lines = [l for ls in texts.values() for l in ls] + corpus
    for i in range(0, len(all_lines), 4096):
        chunk = all_lines[i:i + 4096]
        for enc in tok.encode_batch(chunk, add_special_tokens=False):
            counts.update(enc.ids)
        for line in chunk:
            chars.update(line)
    # Wikipedia quotes foreign names and scripts; drop any token whose text has
    # a character outside the 4 target scripts. Partial-UTF-8 tokens can't be
    # classified and are dropped here; the closure re-adds those that kept
    # tokens are built from.
    in_script = lambda i: (lambda t: t is not None and all(map(allowed_char, t)))(
        token_text(id2tok[i]))
    chars = {c for c in chars if allowed_char(c)}

    # transcripts are the target distribution: keep everything they use
    transcript_ids = set()
    for enc in tok.encode_batch(corpus, add_special_tokens=False):
        transcript_ids.update(i for i in enc.ids if i < n_regular)

    # The base model's own orthography is the real target distribution. Its
    # greedy output differs from the human references (number formatting,
    # punctuation, synonyms), so without this the trim silently byte-spells
    # tokens the model actually emits.
    emissions = manifest_text(a.emissions)
    emission_ids = set()
    if emissions:
        for enc in tok.encode_batch(emissions, add_special_tokens=False):
            emission_ids.update(i for i in enc.ids if i < n_regular)
        print(f"base emissions: {len(emissions)} lines, {len(emission_ids)} distinct tokens")

    # ── coverage mode ────────────────────────────────────────────────────────
    # Frequency trimming fails here. Any token below the cut is byte-spelled,
    # and the model then mis-spells rare in-language orthography that no corpus
    # reliably contains: 드래곤->드래군, 깁슨->깁스, "gợ gĩ"->"gợ ý". Those are real
    # content errors, not cosmetic ones. Coverage mode instead keeps EVERY
    # token whose text lies in the target scripts and drops only out-of-script
    # ones, so in-language spelling is always available and the trim becomes a
    # scarcity trim rather than a coverage trim.
    coverage_ids = set()
    if a.mode == "coverage":
        for i in range(n_regular):
            if in_script(i):
                coverage_ids.add(i)
        out_script = n_regular - len(coverage_ids)
        print(f"coverage mode: keeping all {len(coverage_ids)} in-script tokens, "
              f"dropping {out_script} out-of-script ({100 * out_script / n_regular:.1f}%)")

    def used_at(min_count, with_emissions=True):
        if a.mode == "coverage":
            s = set(coverage_ids)
        else:
            s = {i for i, c in counts.items()
                 if c >= min_count and i < n_regular and in_script(i)}
        s |= transcript_ids
        return (s | emission_ids) if with_emissions else s

    used = used_at(a.min_count)
    if a.mode == "coverage":
        print(f"tokens used (coverage): {len(used)}  "
              f"in-script chars: {len(chars)}")
    else:
        print(f"tokens used (count>={a.min_count}, in-script): {len(used)}  "
              f"in-script chars: {len(chars)}")

    char_tokens = set()
    for enc in tok.encode_batch(sorted(chars), add_special_tokens=False):
        char_tokens.update(i for i in enc.ids if i < n_regular)

    # ── 3. structural tokens ──
    alpha = byte_alphabet()
    byte_ids = {vocab[c] for c in alpha if c in vocab}
    forced = set()
    probes = [f"language {n}<asr_text>" for n in LANGS.values()] + \
             [f" {n}" for n in LANGS.values()] + list("0123456789") + \
             [" " + c for c in "0123456789"] + list(".,%:?!;()'\"-")
    for enc in tok.encode_batch(probes, add_special_tokens=False):
        forced.update(i for i in enc.ids if i < n_regular)
    num_ids = numeral_probe_ids(tok, n_regular)
    if a.mode == "coverage":
        # Numerals are already in-script, so the frequency cut does not reach
        # them and probing would only bias the model toward spelled-out forms
        # (it caused 1963 -> 一九六三年 in the min-count run). Digits are also
        # the overwhelmingly more common ASR output, so keep only those.
        forced |= {i for enc in tok.encode_batch(list("0123456789") +
                                                 [" " + c for c in "0123456789"],
                                                 add_special_tokens=False)
                   for i in enc.ids if i < n_regular}
        print(f"structural forced: {len(forced)} "
              f"({len(probes)} base probes + digits only; numeral probes "
              f"skipped in coverage mode)")
    else:
        forced |= num_ids
        print(f"structural forced: {len(forced)} "
              f"({len(probes)} base probes + {len(num_ids)} numeral tokens)")

    # ── 4. merge closure ──
    parent = {}
    for pair in tj["model"]["merges"]:
        l, r = pair if isinstance(pair, list) else pair.split(" ", 1)
        parent[l + r] = (l, r)

    def closure(ids):
        keep = set(ids)
        stack = [id2tok[i] for i in keep]
        while stack:
            s = stack.pop()
            if s in parent:
                for p in parent[s]:
                    pid = vocab[p]
                    if pid not in keep:
                        keep.add(pid)
                        stack.append(p)
        return keep

    fixed = char_tokens | byte_ids | forced
    if a.mode == "coverage":
        k = closure(used_at(0) | fixed)
        n_base = n_regular + len(added)
        print(f"\n  coverage keep-set: {len(k) + len(added)} rows "
              f"({100 * (len(k) + len(added)) / n_base:.1f}% of {n_base:,})")
        print("  -> the trim drops out-of-script tokens only; in-language "
              "spelling is fully preserved by construction.")

    # --min-count is a size-vs-generality knob, not a correctness gate: the
    # emission clause pins every token the model actually emits, so correctness
    # is decided by the audio check (transcription identity == 100% on test).
    # The wiki sample stands in for "general text in these 4 languages", so a
    # lower min-count keeps more of the base model's spelling ability on words
    # FLEURS never uses. The "without" column shows how many emitted tokens the
    # wiki frequency sample alone would have dropped -- i.e. what the dump buys.
    if emission_ids:
        print("  min-count   vocab   emitted-token coverage   (without dump)")
        for m in (1, 2, 5, 10, 20, 50):
            k_yes = closure(used_at(m, True) | fixed)
            k_no = closure(used_at(m, False) | fixed)
            cov = 100.0 * len(emission_ids & k_yes) / len(emission_ids)
            lost = len(emission_ids - k_no)
            print(f"  {m:>9}  {len(k_yes) + len(added):>7}   {cov:6.2f}%"
                  f"                  {lost:>6} dropped")
        print("  -> pick the largest min-count that still passes audio verification;"
              "\n     lower = smaller vocabulary, more retained base behaviour.")
    else:
        print("  min-count   vocab")
        for m in (1, 2, 5, 10, 20, 50):
            k = closure(used_at(m) | fixed)
            print(f"  {m:>9}  {len(k) + len(added):>7}")
        print("  (no --emissions: emitted tokens are NOT pinned. Run "
              "scripts/dump_base_transcripts.py first.)")
    if a.sweep_only:
        return
    keep = closure(used | fixed)
    print(f"keep: {len(keep)} regular tokens after closure "
          f"(used {len(used)}, char {len(char_tokens)}, byte {len(byte_ids)}, forced {len(forced)})")

    # ── new id layout: kept regular (original order), all added tokens, extra rows ──
    import torch
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(base_dir)
    extra = [cfg.timestamp_token_id] if getattr(cfg, "timestamp_token_id", None) else []
    old_ids = sorted(keep) + sorted(added) + extra
    remap = {o: n for n, o in enumerate(old_ids)}
    n_new = len(old_ids)
    print(f"new vocab: {n_new} rows (was {cfg.text_config.vocab_size}) "
          f"→ {100 * n_new / cfg.text_config.vocab_size:.1f}%")

    # ── trimmed tokenizer.json ──
    new_vocab = {id2tok[o]: remap[o] for o in sorted(keep)}
    new_merges = []
    for pair in tj["model"]["merges"]:
        l, r = pair if isinstance(pair, list) else pair.split(" ", 1)
        if l + r in new_vocab:
            assert l in new_vocab and r in new_vocab, f"closure broken at {l!r}+{r!r}"
            new_merges.append(pair)
    tj["model"]["vocab"] = new_vocab
    tj["model"]["merges"] = new_merges
    for t in tj["added_tokens"]:
        t["id"] = remap[t["id"]]

    # ── trimmed model ──
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model = AutoModelForMultimodalLM.from_pretrained(base_dir, dtype=torch.bfloat16,
                                                     device_map="cpu")
    emb = model.get_input_embeddings().weight
    tied = model.lm_head.weight.data_ptr() == emb.data_ptr()
    assert tied, "expected tied embed_tokens/lm_head"
    rows = emb.data[torch.tensor(old_ids)].clone()
    model.resize_token_embeddings(n_new)
    model.get_input_embeddings().weight.data.copy_(rows)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()

    def remap_ids(obj):
        for k, v in list(obj.items()):
            if isinstance(v, dict):
                remap_ids(v)
            elif k.endswith("token_id") and v is not None:
                obj[k] = [remap[x] for x in v] if isinstance(v, list) else remap[v]

    os.makedirs(out, exist_ok=True)
    model.config.text_config.vocab_size = n_new
    model.save_pretrained(out)
    for name in ("config.json", "generation_config.json"):
        p = os.path.join(out, name)
        d = json.load(open(p))
        remap_ids(d)
        if name == "config.json":
            d["text_config"]["vocab_size"] = n_new
        json.dump(d, open(p, "w"), indent=2)

    processor = AutoProcessor.from_pretrained(base_dir)
    processor.save_pretrained(out)
    json.dump(tj, open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({"base": a.base, "new_to_old": old_ids, "languages": list(LANGS),
               "wiki_chars_per_lang": a.wiki_chars, "min_count": a.min_count,
               "corpus": a.corpus},
              open(os.path.join(out, "keep_ids.json"), "w"))
    print(f"wrote {out}")

    verify(base_dir, out, remap, texts, corpus, a)


def verify(base_dir, out, remap, texts, corpus, a):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoConfig

    print("\n── verify ──")
    t_old = Tokenizer.from_file(os.path.join(base_dir, "tokenizer.json"))
    t_new = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))

    # (a) identical segmentation on held-out in-language text
    rng = random.Random(0)
    for lang, lines in texts.items():
        sample = rng.sample(lines, min(2000, len(lines)))
        same = 0
        for o, n, s in zip(t_old.encode_batch(sample, add_special_tokens=False),
                           t_new.encode_batch(sample, add_special_tokens=False), sample):
            same += [remap.get(i, -1) for i in o.ids] == n.ids
            assert t_new.decode(n.ids) == t_old.decode(o.ids), f"decode mismatch: {s[:80]}"
        print(f"  tokenizer {lang}: {100 * same / len(sample):.2f}% identical segmentation on reference text, "
              f"decode round-trip ok")
    o = [t_old.encode(s, add_special_tokens=False).ids for s in corpus]
    n = [t_new.encode(s, add_special_tokens=False).ids for s in corpus]
    same = sum([remap.get(i, -1) for i in x] == y for x, y in zip(o, n))
    if corpus:
        print(f"  tokenizer transcripts: {100 * same / len(corpus):.2f}% identical")

    # (b) foreign scripts: how much is still spellable with whole tokens
    print("  foreign probes (tokens per char; higher = only byte-spellable):")
    for label, probe in FOREIGN_PROBES.items():
        print(f"    {label:9s} old {len(t_old.encode(probe).ids)/len(probe):.2f}"
              f"  new {len(t_new.encode(probe).ids)/len(probe):.2f}")

    # (c) config + processor consistency
    proc = AutoProcessor.from_pretrained(out)
    cfg = AutoConfig.from_pretrained(out)
    assert proc.audio_token_id == cfg.audio_token_id, (proc.audio_token_id, cfg.audio_token_id)
    for s in ("<|im_end|>", "<asr_text>", "<|endoftext|>"):
        assert proc.tokenizer.convert_tokens_to_ids(s) == remap[t_old.token_to_id(s)], s
    print("  special token ids consistent across tokenizer/config/processor")

    # (d) base vs trimmed transcriptions on real audio (greedy)
    rows = []
    for pat in a.verify_audio:
        pat = pat if os.path.isabs(pat) else os.path.join(ROOT, pat)
        # expand globs: os.path.exists() on a "data/*_test.jsonl" pattern is
        # always False, which silently skipped the acceptance check below.
        for p in sorted(glob.glob(pat)):
            with open(p, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f][:a.verify_per_lang]
    if not rows:
        print(f"  (no verify audio matched {a.verify_audio}; "
              "SKIPPED transcription check -- this is the acceptance gate, "
              "re-run with a manifest that exists)")
        return
    print(f"  transcribing {len(rows)} utterances with base and trimmed "
          f"({2 * len(rows)} decodes, ~{len(rows) * 0.6:.0f}s each)")
    import soundfile as sf

    def transcribe(path):
        processor = AutoProcessor.from_pretrained(path)
        model = AutoModelForMultimodalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                         device_map="cuda")
        res = []
        for r in rows:
            audio, _ = sf.read(r["audio"], dtype="float32")
            inp = processor.apply_transcription_request(audio=audio).to("cuda", torch.bfloat16)
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=256, do_sample=False)
            res.append(processor.decode(g[:, inp["input_ids"].shape[1]:])[0])
        del model
        torch.cuda.empty_cache()
        return res

    before, after = transcribe(base_dir), transcribe(out)

    # Split the diffs. A wrong language tag with an identical transcript is not
    # a regression (the trim often *fixes* zh LID), and a punctuation-only
    # difference is not either -- the pipeline ends in TTS. Only a changed
    # transcript counts against acceptance.
    def split(s):
        m = re.match(r"^language\s+(\S+?)<asr_text>(.*)$", s.strip(), re.S)
        return (m.group(1), m.group(2)) if m else (None, s.strip())

    def canon(text, lang):
        text = unicodedata.normalize("NFKC", text).lower()
        out = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat[0] in ("P", "S", "Z"):
                continue
            out.append(ch)
        s = "".join(out)
        return s.replace(" ", "") if lang in ("zh", "ko") else s

    tally = collections.defaultdict(lambda: collections.Counter())
    samples = collections.defaultdict(list)
    for r, b, t in zip(rows, before, after):
        lang = r["lang"]
        lb, tb = split(b)
        lt, tt = split(t)
        c = tally[lang]
        c["n"] += 1
        if b == t:
            c["identical"] += 1
            continue
        if lb != lt and canon(tb, lang) == canon(tt, lang):
            c["lid_only"] += 1
        elif canon(tb, lang) == canon(tt, lang):
            c["punct_only"] += 1
        else:
            c["real"] += 1
            if len(samples[lang]) < 3:
                samples[lang].append((tb, tt))

    print("\n  ── acceptance (transcript content, punctuation/LID ignored) ──")
    tot = collections.Counter()
    for lang in sorted(tally):
        c = tally[lang]
        for k, v in c.items():
            tot[k] += v
        rate = 100 * (c["n"] - c["real"]) / c["n"]
        print(f"  {lang}: {rate:6.2f}% equivalent   "
              f"(identical {c['identical']}, lid-only {c['lid_only']}, "
              f"punct-only {c['punct_only']}, REAL {c['real']}/{c['n']})")
    rate = 100 * (tot["n"] - tot["real"]) / tot["n"]
    print(f"  ALL: {rate:6.2f}% equivalent   "
          f"(REAL regressions {tot['real']}/{tot['n']})")
    verdict = "PASS" if rate >= a.min_equivalent else "FAIL"
    print(f"  gate: >= {a.min_equivalent}% equivalent -> {verdict}")

    if samples:
        print("\n  ── real regressions ──")
        for lang in sorted(samples):
            for tb, tt in samples[lang]:
                print(f"  [{lang}] base: {tb[:100]}")
                print(f"  [{lang}] trim: {tt[:100]}")


if __name__ == "__main__":
    main()
