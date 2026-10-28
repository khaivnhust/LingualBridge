# MT: fine-tuning Hy-MT2 for the device

Two documents already cover this; this page says which to read and adds what the phone measurements imply.

| read | for |
|---|---|
| [`handoff/mt-harness-2026-10-28/README.md`](../../handoff/mt-harness-2026-10-28/README.md) | **the contract**: the exact prompt bytes, anchors, sampling, output cleaning, clause chunking, caching. Includes `mt_harness_ref.py` to test a GGUF exactly as the phone calls it, and 113 real ASR→MT pairs from the phone |
| [`docs/mt/MT_DISTILLATION.md`](../mt/MT_DISTILLATION.md) ([vi](../mt/MT_DISTILLATION.vi.md)) | the training plan: trimmed 1.8B student, vocabulary trim to en/vi/zh/ko, recovery fine-tune |

`translator/MT_PROMPT.md` is **older than the handoff README** and wrong about one thing: the anchor is ordinary text in the user turn, not
a generation prefix.

## What the phone numbers mean for MT

- MT owns about 400 ms of the end-of-speech → first-word budget, and that is the **time to the first speakable clause**
  (clause = text up to a terminator once it is ≥ 14 UTF-8 bytes). A translation that puts natural punctuation early
  starts speaking sooner. A model that is the same size but writes longer first clauses is slower in practice.
- Keep the architecture at about 1.8B and a Q6_K / Q4_K / Q8_0 quant: those run on the NPU (about 38 ms/token for Q6_K).
- Train on the **exact prompt bytes** from `handoff/.../prompts.json`. Any change invalidates the per-target prompt cache
  and changes what the model was evaluated on. The Korean instruction has a known typo (`모륰` should be `모르는`); fix it in
  `harness/src/pipeline.h` and in the training data at the same time, or not at all.
- Inputs are ASR output: punctuated, numbers mostly in digits, often **short fragments**, spontaneous Korean,
  and noisy Vietnamese (ASR WER about 19 %). Robustness to those matters more than polish on clean text.
- Don't use phone logs from before 2026-10-27 as data: a harness bug dropped the first word of every transcript.

## Checking a candidate

```bash
llama-server -m candidate.gguf --port 8084 -np 1 -c 4096 --cache-ram 256
python3 handoff/mt-harness-2026-10-28/mt_harness_ref.py --url http://127.0.0.1:8084 \
    --replay handoff/mt-harness-2026-10-28/examples/device_mt_samples.jsonl > candidate.jsonl
```

Then put it on the phone (`docs/guides/device-benchmarking.md`) and rerun `harness/scripts/bench_eos.sh`: the number that decides is
end of speech → first audible word.
