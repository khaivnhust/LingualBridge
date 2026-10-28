# harness: the pipeline on the phone, and how fast it is

C++ for the Snapdragon 8 Elite (OnePlus 13): a **benchmark harness** that measures end of speech → first audible word,
and a **single-process pipeline** (`mono`) that is the product shape. How-to (build, deploy, run):
[`docs/guides/device-benchmarking.md`](../docs/guides/device-benchmarking.md). This README holds **what we found**.

## Contents

| path | what |
|---|---|
| `src/main.cc` | benchmark harness: VAD → ASR (llama-server) → streamed MT (llama-server) → clause → Piper TTS, all timed. `--realtime` replays audio on a wall clock and decodes while it is "spoken" |
| `src/mono.cc` | ASR, MT and TTS in one process, one thread each; speculative ASR; per-target MT prompt-KV save/restore |
| `src/pipeline.h` | text contracts: ASR output cleaning, MT prompts + anchors, MT output cleaning, clause chunker |
| `src/vad.*`, `tts.*`, `http.*`, `stitch.h` | energy VAD, Piper via ONNX Runtime + espeak-ng, minimal HTTP/SSE client, transcript stitcher |
| `src/mt_prompt_dump.cc` | writes the exact MT prompts as JSON (feeds `handoff/`) |
| `run_qdc.sh` | starts both servers with the right flags and runs one measurement |
| `scripts/bench_eos.sh` | **the benchmark**: 80 clips, current best configuration, thermally guarded |
| `scripts/eos_report.py` | summarises a run: latency percentiles + WER/CER against references |
| `scripts/make_bench_data.py`, `build_package.sh`, `setup.sh`, `build_mono.sh` | bench audio, phone package, on-device setup, `mono` build |
| `scripts/android/`, `scripts/root/` | llama.cpp Android builds, deploy; phone max-performance / restore mode |

## Headline

**End of speech → first MT token: median 936 ms, p90 1481 ms, 1 of 113 utterances over 2 s.**
**End of speech → first audible word: median 1530 ms, p90 2410 ms.**
Configuration: ASR audio encoder on the GPU, ASR decoder + MT on the NPU, TTS on the CPU, per-language segmenting policy, Round 6 ASR.

## Results

### End of speech → first MT token (80 clips / 113 utterances, ms from the last speech sample)

| config | median | p90 | max | > 2 s | WER en / vi · CER zh / ko |
|---|---|---|---|---|---|
| A: one-shot, model detects language (R1) | 1436 | 2230 | 3690 | 20 | 6.65 / 18.95 / 10.00 / 12.69 |
| B: one-shot, language prefilled (R1) | 1283 | 2119 | 3484 | 19 | 6.65 / 18.95 / 7.42 / 13.14 |
| C: decode while speaking, 100 ms pause cuts | 1000 | 1412 | 2063 | 1 | 21.2 / 36.0 / 30.0 / 22.1 ❌ |
| E: one-shot started at 120 ms of silence | 1224 | 1987 | 4754 | 11 | = B |
| F: cuts only at ≥ 250 ms pauses, segments ≥ 2.5 s | 1105 | 1789 | 2210 | 4 | 7.16 / 18.95 / 12.90 / 13.14 |
| B6: one-shot, Round 6 | 1297 | 2082 | 3475 | 17 | 6.14 / 18.95 / 3.87 / 11.14 |
| G: per language (F for en/vi/ko, E for zh), R6, ASR on GPU | 1155 | 1897 | 2364 | 5 | 6.39 / 18.95 / 3.87 / 11.14 |
| **Gs: G + decoder & MT on NPU, encoder on GPU** | **936** | **1481** | 2872 | **1** | 5.63 / 31.4* / 4.19 / 10.91 |

\* One 0.7 s fragment hallucinated a sentence; only 4 Vietnamese transcripts differ from G.

### Where the time goes

| piece | cost |
|---|---|
| ASR encode + prefill | about 350 ms + 110 ms per audio second |
| ASR decode | 28 ms/token (NPU) · 37–56 ms/token (GPU); prefilling `language X<asr_text>` saves about 4 tokens |
| MT first token / first clause | about 80–100 ms / about 370–420 ms (Hy-MT2-1.8B Q6_K, NPU, prompt cached) |
| MT decode | about 38 ms/token |
| streaming MT vs waiting for the full sentence | −1.16 s |
| HTTP overhead per call | ASR about 14 ms, MT about 5 ms |
| MT prompt KV restore on a direction change (`mono`) | 5–10 ms |

## Verdicts

| question | answer |
|---|---|
| decode while speaking with short pause cuts? | ❌ meets the target but hallucinates short segments (accuracy ×3 worse) |
| one-shot, started inside the VAD hangover? | ✅ free: same transcripts, fewer misses |
| cut at real pauses only? | ✅ for en/vi/ko; ❌ for zh (+5.5 CER) |
| ASR audio encoder on the NPU? | ❌ broken transcripts at every quant |
| ASR decoder on the NPU? | ✅ with cooling: −220 ms median, no failures. ❌ **uncooled under constant load**: the NPU sat above 100 °C and produced NaN for 10–20 consecutive requests. Next: NaN guard (re-run on GPU) and a cooled soak test |
| one MT slot per target language? | ❌ slower; one slot + `--cache-ram 256` restores each target's prompt (89 ms first token when alternating) |
| `--cache-ram` default on ASR? | ❌ grew to 4.4 GB and thrashed the phone; use `--cache-ram 0` |
| OpenCL + NPU in one process (`mono`)? | ✅ same timings as two servers, 606 MB total. Running ASR and MT concurrently saves 1.3 s per pair, but the NPU decode slows the GPU encode by 64 % while they overlap |
| mic / speaker from a native binary? | ✅ AAudio, no app. Input is low-latency; output needs 48 kHz for the fast path. **Room noise floor about −32 dBFS defeats the energy VAD**, so a neural VAD is needed |

## Harness bugs fixed (on-device numbers from before 2026-10-27 are affected)

1. `asr_ms` held end of speech → first voice in one-shot runs; true ASR time was never recorded.
2. ASR text cleaning **dropped the first word of every transcript** (`language English<asr_text>` has no space).
3. The phone used `/v1/audio/transcriptions`, which adds an English instruction the model was never evaluated with;
   it now uses `/completion` with the evaluated prompt.
4. The ASR prompt cache skipped audio encoding on repeated clips (about 0.6 s too fast); ASR sends `cache_prompt:false`.

## Known limits

- `mono` is a skeleton: file replay only (no mic yet), one-shot policy only (no per-language segmenting yet).
- Korean TTS (`ko_KR-kss-medium`) is garbled: it needs pygoruut phonemes, which Piper 1.2 ignores. Korean latency is still valid.
- The four Piper voices are the previous set, not the ones `docs/history/TTS_PLAN.md` selected.
- Bench audio: zh/vi come from Common Voice (read speech). There is no spontaneous zh/vi set.
