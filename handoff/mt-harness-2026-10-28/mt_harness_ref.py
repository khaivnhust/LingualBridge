#!/usr/bin/env python3
"""Byte-for-byte reference of how the on-device S2ST harness talks to the MT model.

Sends exactly what s2st-bench (src/main.cc, src/pipeline.h) sends to llama-server,
streams the reply, splits it into speakable clauses the way the TTS stage does, and
reports the latencies the harness measures. Point it at a llama-server running your
fine-tuned GGUF to see what the device pipeline will see.

  # one sentence
  python3 mt_harness_ref.py --url http://127.0.0.1:8084 --tgt vi --text "We need European leadership."

  # replay the device examples (examples/device_mt_samples.jsonl) through your model
  python3 mt_harness_ref.py --url http://127.0.0.1:8084 --replay examples/device_mt_samples.jsonl

No dependencies beyond the Python standard library.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = json.load(open(os.path.join(HERE, "prompts.json"), encoding="utf-8"))


# ── exactly src/pipeline.h ────────────────────────────────────────────────────
def mt_user_content(source_text: str, tgt: str) -> str:
    """mt_prompt(): per-target instruction + "\\n\\n" + source + "\\n\\n" + anchor.
    One user message, no system message. The anchor is TEXT INSIDE THE USER TURN;
    generation starts after the chat template's assistant header, not after the anchor."""
    return PROMPTS[tgt]["user_content_template"].replace("{source}", source_text)


def request_body(source_text: str, tgt: str, stream: bool = True) -> dict:
    """The JSON the harness POSTs to /v1/chat/completions (main.cc mt_body())."""
    return {
        "messages": [{"role": "user", "content": mt_user_content(source_text, tgt)}],
        "stream": stream,
        "cache_prompt": True,   # the system part is identical per target: KV-cached server-side
        "temperature": 0,       # greedy
        "top_p": 0.6,
        "max_tokens": 256,
    }


def clean_mt_text(raw: str, tgt: str) -> str:
    """clean_mt_text(): drop an echoed anchor, a leading "created" artifact, trim."""
    s = raw
    anchor = PROMPTS[tgt]["anchor"]
    if s.startswith(anchor):
        s = s[len(anchor):]
    if s.startswith("created"):
        s = s[len("created"):]
    return s.strip()


TERMINATORS = set(",;:.!?\n") | {"，", "、", "。", "！", "？", "；", "："}


class ClauseChunker:
    """ClauseChunker: emit a clause at a terminator once the clause is >= min_bytes
    UTF-8 BYTES long (not characters: 14 bytes is ~4-5 CJK characters); never split
    a number ("3.5", "1,000"); force-flush at max_bytes."""

    def __init__(self, min_bytes=14, max_bytes=160):
        self.buf = b""
        self.min, self.max = min_bytes, max_bytes

    def push(self, delta: str):
        self.buf += delta.encode("utf-8")
        out = []
        while True:
            text = self.buf.decode("utf-8", errors="ignore")
            cut = None
            pos = 0
            for idx, ch in enumerate(text):
                pos += len(ch.encode("utf-8"))
                if ch not in TERMINATORS or pos < self.min:
                    continue
                if ch in ".,":
                    prev = text[idx - 1] if idx > 0 else ""
                    nxt = text[idx + 1] if idx + 1 < len(text) else ""
                    if prev.isdigit() and nxt.isdigit():
                        continue
                cut = pos
                break
            if cut is not None:
                out.append(self.buf[:cut].decode("utf-8", errors="ignore"))
                self.buf = self.buf[cut:]
                continue
            if len(self.buf) >= self.max:
                out.append(self.buf.decode("utf-8", errors="ignore"))
                self.buf = b""
            return out

    def flush(self):
        s, self.buf = self.buf.decode("utf-8", errors="ignore"), b""
        return [s] if s else []


# ── one streamed request, timed the way the harness times it ─────────────────
def translate(url: str, source_text: str, tgt: str):
    body = json.dumps(request_body(source_text, tgt)).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    t0 = time.perf_counter()
    first_token = first_clause = None
    full, clauses = "", []
    ch = ClauseChunker()
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {}).get("content") or ""
            if first_token is None:
                first_token = (time.perf_counter() - t0) * 1000
            if not delta:
                continue
            full += delta
            for c in ch.push(delta):
                if first_clause is None:
                    first_clause = (time.perf_counter() - t0) * 1000
                clauses.append(c)
    tail = ch.flush()
    if tail:
        if first_clause is None:
            first_clause = (time.perf_counter() - t0) * 1000
        clauses += tail
    return {
        "tgt": tgt, "source": source_text, "raw": full, "mt_text": clean_mt_text(full, tgt),
        "clauses_to_tts": clauses,
        "first_token_ms": round(first_token or -1), "first_clause_ms": round(first_clause or -1),
        "total_ms": round((time.perf_counter() - t0) * 1000),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8084", help="llama-server serving the MT GGUF")
    ap.add_argument("--tgt", choices=sorted(PROMPTS), help="target language for --text")
    ap.add_argument("--text", help="source text to translate")
    ap.add_argument("--replay", help="JSONL with asr_text/tgt (e.g. examples/device_mt_samples.jsonl)")
    ap.add_argument("--show-request", action="store_true", help="print the exact JSON body and exit")
    a = ap.parse_args()

    if a.show_request:
        print(json.dumps(request_body(a.text or "{source}", a.tgt or "vi"), ensure_ascii=False, indent=2))
        return
    if a.text:
        print(json.dumps(translate(a.url, a.text, a.tgt or "vi"), ensure_ascii=False, indent=2))
        return
    if a.replay:
        for line in open(a.replay, encoding="utf-8"):
            r = json.loads(line)
            out = translate(a.url, r["asr_text"], r["tgt"])
            print(json.dumps({"id": r.get("id"), "tgt": r["tgt"], "source": r["asr_text"],
                              "device_mt_text": r.get("mt_text"), "your_mt_text": out["mt_text"],
                              "raw": out["raw"], "first_token_ms": out["first_token_ms"],
                              "first_clause_ms": out["first_clause_ms"]}, ensure_ascii=False))
        return
    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
