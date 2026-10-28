#!/usr/bin/env python3
"""Greedy-decode the base ASR model over labelled audio and dump its transcripts.

Why this exists
---------------
`trim_vocab.py` builds its keep-set from FLEURS *reference* transcripts plus a
Wikipedia frequency sample. But the keep-set has to cover the tokens the model
will actually **emit**, and the base model's orthography differs from the human
references (it renders "108" as "108" -> a different token sequence, drops or
adds punctuation, picks synonyms). Those emitted tokens were never in the
keep-set, so after trimming they fall back to byte spelling and the transcript
changes -- 26% of FLEURS dev utterances at --min-count 20.

Passing this file to `trim_vocab.py --corpus` force-keeps every token the base
model emits, because everything in --corpus lands in `transcript_ids`.

Output is a drop-in manifest for --corpus / --verify-audio:
    {"audio", "text", "lang", "duration", "source": "base-emissions"}

Build the keep-set from TRAIN emissions and verify on TEST, so the threshold is
never fitted to the split you report numbers on.

Usage:
    scripts/dump_base_transcripts.py --out data/base_emissions_train.jsonl \\
        data/fleurs/*_train.jsonl
    scripts/dump_base_transcripts.py --max-per-lang 500 --out /tmp/sub.jsonl \\
        data/fleurs/*_train.jsonl
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SAMPLE_RATE = 16000


def load_audio(path):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def read_manifests(patterns, max_seconds):
    rows = []
    for pat in patterns:
        for p in sorted(_glob(pat)):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("duration", 0) <= max_seconds and r.get("audio"):
                        rows.append(r)
    return rows


def _glob(pat):
    import glob as _g

    return _g.glob(pat if os.path.isabs(pat) else os.path.join(ROOT, pat))


def subsample(rows, per_lang, seed):
    """Cap utterances per language; token coverage saturates quickly, so a
    subset gives most of the benefit for proportionally less GPU time."""
    if not per_lang:
        return rows
    by = {}
    for r in rows:
        by.setdefault(r["lang"], []).append(r)
    rng = random.Random(seed)
    out = []
    for lang in sorted(by):
        pool = by[lang]
        rng.shuffle(pool)
        out += pool[:per_lang]
    return out


def patch_left_context(model, n_left):
    """Replace the encoder's per-window attention with block-causal attention.

    The encoder's cu_seqlens already mark window (block) boundaries; instead of
    attending only inside its own block, block j attends to blocks j-n_left..j.
    Nothing attends forward, so every block's output -- at every layer -- is final
    once computed, which is what an encode-while-speaking pipeline can reproduce
    exactly with a per-layer KV cache of past blocks.
    """
    import torch
    import torch.nn.functional as F

    attn_cls = None
    for m in model.modules():
        if type(m).__name__.endswith("Attention") and hasattr(m, "out_proj") and not getattr(m, "is_causal", True):
            attn_cls = type(m)
            break
    if attn_cls is None:
        sys.exit("encoder attention class not found")

    def forward(self, hidden_states, cu_seqlens, max_seqlen=None, **kwargs):
        n, _ = hidden_states.size()
        q = self.q_proj(hidden_states).reshape(n, self.num_heads, -1).transpose(0, 1)
        k = self.k_proj(hidden_states).reshape(n, self.num_heads, -1).transpose(0, 1)
        v = self.v_proj(hidden_states).reshape(n, self.num_heads, -1).transpose(0, 1)
        block = torch.bucketize(torch.arange(n, device=q.device), cu_seqlens[1:].to(q.device), right=True)
        d = block[:, None] - block[None, :]          # query block - key block
        allowed = (d >= 0) & (d <= n_left)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=self.scaling)
        return self.out_proj(out.transpose(0, 1).reshape(n, -1).contiguous())

    attn_cls.forward = forward


def main():
    ap = argparse.ArgumentParser(description="Dump base-model greedy transcripts")
    ap.add_argument("manifests", nargs="+", help="JSONL manifests (globs allowed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-audio-seconds", type=float, default=30)
    ap.add_argument("--max-per-lang", type=int, default=0,
                    help="cap utterances per language (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-window-infer", type=int, default=0,
                    help="override the audio encoder's attention window in mel frames "
                         "(multiple of 100; model default 800 = 8 s). Shorter windows "
                         "make encoder outputs for finished windows final, which is "
                         "what incremental (encode-while-speaking) prefill needs")
    ap.add_argument("--left-context-blocks", type=int, default=-1,
                    help="with --n-window-infer: each window (block) attends to itself and "
                         "the previous N blocks, never to later ones, so a block's output is "
                         "final once computed. -1 = off (independent windows). Needs "
                         "--batch-size 1 (blocks must not span utterances)")
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    rows = subsample(read_manifests(a.manifests, a.max_audio_seconds),
                     a.max_per_lang, a.seed)
    if not rows:
        sys.exit("no rows matched")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    print(f"{len(rows)} utterances; loading {a.base}", flush=True)
    processor = AutoProcessor.from_pretrained(a.base)
    model = AutoModelForMultimodalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if a.n_window_infer:
        hit = [m for m in model.modules() if hasattr(m, "n_window_infer")]
        if not hit:
            sys.exit("no module with n_window_infer")
        for m in hit:
            m.n_window_infer = a.n_window_infer
        print(f"audio encoder n_window_infer -> {a.n_window_infer} ({len(hit)} module)", flush=True)
    if a.left_context_blocks >= 0:
        if not a.n_window_infer or a.batch_size != 1:
            sys.exit("--left-context-blocks needs --n-window-infer and --batch-size 1")
        patch_left_context(model, a.left_context_blocks)
        print(f"audio encoder: block-causal attention, {a.left_context_blocks} blocks of left context",
              flush=True)

    n_empty = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(0, len(rows), a.batch_size):
            chunk = rows[i:i + a.batch_size]
            audios = [load_audio(r["audio"]) for r in chunk]
            inp = processor.apply_transcription_request(
                audio=audios).to(model.device, torch.bfloat16)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=a.max_new_tokens,
                                     do_sample=False)
            new = gen[:, inp["input_ids"].shape[1]:]
            parsed = processor.decode(new, return_format="parsed")
            for r, p in zip(chunk, parsed):
                text = (p["transcription"] or "").strip()
                if not text:
                    n_empty += 1
                fh.write(json.dumps({
                    "audio": r["audio"],
                    "text": text,
                    "lang": r["lang"],
                    "duration": r.get("duration"),
                    "source": "base-emissions",
                    "emitted_language": p.get("language"),
                }, ensure_ascii=False) + "\n")

            if (i // a.batch_size) % 20 == 0:
                done = min(i + a.batch_size, len(rows))
                el = time.time() - t0
                print(f"  {done}/{len(rows)}  {el:.0f}s  "
                      f"eta {el / max(done, 1) * (len(rows) - done):.0f}s",
                      flush=True)

    print(f"wrote {out}  (empty transcripts: {n_empty})")


if __name__ == "__main__":
    main()