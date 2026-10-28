# MT harness handoff — how the on-device pipeline calls the MT model

For the MT fine-tuner. This package shows **exactly** what the speech-to-speech pipeline sends to the
translation model, what it does with the reply, and how it times it, so a fine-tuned model can be
checked against the real contract before it goes on the phone.

The source of truth is the code (`harness/src/pipeline.h`, `harness/src/main.cc`). `prompts.json` is
generated from it, and `mt_harness_ref.py` reproduces the requests byte-for-byte.
`translator/MT_PROMPT.md` in the main repo is older; where it disagrees with this file, this file is right.

## The pipeline, in one line

```
mic → VAD → ASR (Qwen3-ASR, phone GPU/NPU) → MT (Hy-MT2-1.8B Q6_K, phone NPU, streamed)
        → split into clauses → TTS (Piper, CPU) speaks clause 1 while MT is still decoding
```

The number that matters is **end of speech → first audible word**. MT's share of it is the time to its
**first speakable clause**, not its total time.

## The MT contract

### Request

`POST /v1/chat/completions` to llama-server (the on-device monolith builds the same prompt in-process):

```json
{
  "messages": [{"role": "user", "content": "<user content, below>"}],
  "stream": true,
  "cache_prompt": true,
  "temperature": 0,
  "top_p": 0.6,
  "max_tokens": 256
}
```

- **One user message, no system message.** The GGUF's own chat template (Hy-MT2 / hunyuan) is applied
  by the server.
- **Greedy decoding** (`temperature: 0`, so `top_p` has no effect). Generation stops at the model's EOG
  token or 256 tokens.

### User content

```
<per-target instruction>\n\n<source text>\n\n<anchor>
```

The exact strings for each target are in `prompts.json` (`user_content_template`, with `{source}` as
the placeholder). Run `python3 mt_harness_ref.py --show-request --tgt vi --text "..."` to print a
complete body.

| target | anchor (note the trailing space on three of them) |
|---|---|
| en | `"Pure English translation: "` |
| zh | `"纯中文翻译："` |
| ko | `"순수 한국어 번역: "` |
| vi | `"Bản dịch tiếng Việt thuần túy: "` |

**The anchor is text inside the user turn, not an assistant prefill.** The model starts generating after
the chat template's assistant header, not after the anchor. (`MT_PROMPT.md` says the anchor acts as a
generation prefix; that is not what happens.) If fine-tuning teaches the model to continue from the
anchor, its output must still be the translation only.

Glossary terms are **not** injected (the harness accepts `--terms` but ignores it).

### What the model should output

The translation only: no anchor, no explanation, no quotes, no language label. Before use the
harness applies `clean_mt_text()`:

1. if the output starts with the anchor, drop it;
2. if it starts with `created`, drop that (an artifact of the current un-fine-tuned model);
3. trim whitespace.

A fine-tuned model should never need rules 1–2.

### Streaming and clauses (this is where latency is won or lost)

The streamed output goes through `ClauseChunker` (in `pipeline.h`; `mt_harness_ref.py` has an exact port,
checked identical on 342 streams). A clause is emitted at the first terminator once the clause is
**≥ 14 UTF-8 bytes** (about 4–5 CJK characters, about 14 Latin characters). The terminators are
`, ; : . ! ?` and newline, plus the full-width `， 、 。 ！ ？ ； ：`. A `.` or `,` between two digits never
splits (`3.5`, `1,000`). If no terminator comes, the clause is force-flushed at 160 bytes.

TTS starts speaking as soon as clause 1 exists. Practical consequences:

- **Early, natural punctuation makes speech start sooner.** A long unpunctuated opening delays the
  first audible word until the first terminator, or until 160 bytes.
- The TTS speaks each clause as written, so numbers and symbols should already be spoken-form (the
  instruction asks for that: rules 6–9 in each prompt).

### Prompt caching: keep the instruction byte-stable

Everything before `<source text>` is identical for a given target, so it is cached:
- llama-server `cache_prompt: true` with `-np 1 --cache-ram 256` restores each target's prefix when the
  direction alternates (measured: first token 89 ms median with alternating targets);
- the in-process monolith (`harness/src/mono.cc`, `MtEngine`) prefills the prefix once per target,
  saves its KV state, and restores it in 5–10 ms. It tokenizes the prefix and `source + suffix`
  **separately**, so the boundary falls right after the `\n\n` that precedes the source.

**Any change to the instruction text, even one character, invalidates these caches and changes the
prompt the model was evaluated on.** Fine-tune on these exact bytes.

### Source text the model receives

It comes from the ASR (Qwen3-ASR, fine-tuned Round 6, source language known), so:
- it is punctuated, and numbers are usually digits ("in 2009 and 2010", "11 giờ");
- it can be a **short fragment**, because the VAD splits at pauses ("Tháng Chín", "근데", "for many types.");
- Korean is often **spontaneous and disfluent** (KsponSpeech-style), and Vietnamese ASR is the weakest
  (WER ~19 % on the bench set), so expect noisy Vietnamese input;
- the directions in use are en→vi / en→zh / en→ko and vi/zh/ko→en.

⚠️ Device logs from **before 2026-10-27** have the **first word of every transcript missing** (a harness
bug since fixed). Do not use them as training or evaluation data.

### Deployment target and current numbers

| | |
|---|---|
| model | Hy-MT2-1.8B, Q6_K GGUF |
| device | Snapdragon 8 Elite (OnePlus 13), Hexagon NPU via llama.cpp HTP backend |
| server | `-ngl 99 -c 4096 -np 1 -dev HTP0 --cache-ram 256` |
| NPU weight types | Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, MXFP4 (others fall back to CPU) |
| first token | ~80–100 ms (prefix cached) |
| first clause | ~370–420 ms median |
| decode | ~38 ms/token |

A model that keeps this architecture and quant keeps these numbers. Longer outputs and later
punctuation are what move them.

### Known issues in the current contract

- **Typo in the Korean instruction**, rule 4: `모륰 단어` should be `모르는 단어` ("unknown words"). It is
  left as-is here because this package documents what is deployed. If it is fixed, fix it in
  `pipeline.h` and fine-tune on the fixed bytes at the same time (see prompt caching above).
- The current (un-fine-tuned) model sometimes prefixes `created`; `clean_mt_text()` strips it.

## Verifying a fine-tuned model

```bash
# 1. serve your GGUF (any machine; on the phone add -dev HTP0)
llama-server -m your-mt.gguf --port 8084 -np 1 -c 4096 --cache-ram 256

# 2. one sentence
python3 mt_harness_ref.py --url http://127.0.0.1:8084 --tgt vi --text "We need European leadership and solutions right now."

# 3. replay the 113 real device inputs and compare with the current model's outputs
python3 mt_harness_ref.py --url http://127.0.0.1:8084 --replay examples/device_mt_samples.jsonl > yours.jsonl
```

Each output line shows the source, the current device output, your output, the raw stream (to check
for leaked anchors or explanations), and first-token and first-clause times.
Note: timings on a desktop do not transfer to the phone, but relative changes in output length and
punctuation do.

## Files

| path | what |
|---|---|
| `README.md` | this contract |
| `prompts.json` | exact per-target prompt templates and anchors, generated from `pipeline.h` by `harness/src/mt_prompt_dump.cc` |
| `mt_harness_ref.py` | stdlib-only Python reference: same request, same streaming, same clause chunking, same cleaning, same timings |
| `examples/device_mt_samples.jsonl` | 113 real phone pairs (2026-10-28, ASR Round 6 → current Hy-MT2-1.8B): `asr_text` in, `mt_text` out, with MT timings |
| `harness/src/pipeline.h` | **MT prompt, anchors, clause chunker, output cleaning** (start here) |
| `harness/src/main.cc` | benchmark harness: ASR → MT (streamed) → TTS, latency measurement |
| `harness/src/mono.cc` | whole pipeline in one process; `MtEngine` = per-target prompt-KV save/restore |
| `harness/src/*` | the rest (VAD, HTTP client, TTS, probes); for reference |
| `harness/scripts/` | how the benchmarks were run on the phone; `run_qdc.sh` has the server flags |

Building the C++ harness needs the Android NDK, a llama.cpp build with the Hexagon backend, ONNX Runtime and
espeak-ng. It is **not** needed to verify a model: `mt_harness_ref.py` covers the MT side completely.
