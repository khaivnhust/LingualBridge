# Architecture

## The pipeline

```
           ┌──────────── phone (Snapdragon 8 Elite) ─────────────────────────────────────┐
 mic ──▶ VAD ──▶ ASR ───────────────────▶ MT (streamed) ──▶ clause ──▶ TTS ──▶ speaker
       (energy)  audio encoder  decoder     Hy-MT2-1.8B       chunker    Piper VITS
                 Adreno GPU     Hexagon NPU Hexagon NPU                   CPU (ONNX Runtime)
```

What a user feels is **end of speech → first audible word**. Every design choice below serves that.

| stage | what runs | where | notes |
|---|---|---|---|
| VAD | energy VAD, adaptive noise floor (`harness/src/vad.cc`) | CPU | 250 ms hangover. **Too weak for noisy rooms** (measured noise floor about −32 dBFS); a neural VAD (Silero) is the planned replacement |
| ASR | Qwen3-ASR-1.7B Round 6, Q4_K_M text model + Q8_0 audio encoder (mmproj) | encoder: GPU (OpenCL) · decoder: NPU (HTP) or GPU | the source language is known, so the prompt prefills `language <Name><asr_text>` and skips about 4 decode steps |
| MT | Hy-MT2-1.8B Q6_K | NPU (HTP) | per-target-language prompt + anchor; the instruction part is KV-cached per target |
| TTS | Piper medium VITS | CPU | one clause at a time while MT keeps decoding |

### Why this split

- **The decoder is fastest on the NPU** (about 28 ms/token, against about 37 on the GPU). **The audio encoder is broken on the NPU**:
  it emits truncated or looping transcripts at every quant level, so it stays on the GPU. Running it on the CPU works too but
  competes with TTS.
- **MT on the NPU** is about 31 % faster than on the GPU, and the NPU maps its weights without
  holding them in host RAM (~0.6 GB RSS for a 1.8 GB model).
- **One llama-server slot per model** (`-np 1`). Four default slots cost 47 % more memory and 21 % more time.
- **`--cache-ram 0` on ASR** (it never reuses a prompt; the 8 GiB default host cache grew to 4.4 GB and
  drove the phone into swap thrash). **`--cache-ram 256` on MT**: it restores each target's prompt when the
  translation direction alternates.

## Latency, end of speech → first word

1. **VAD**: the utterance is only "over" after the hangover. The speculative mode starts ASR after
   120 ms of silence, inside the hangover, and drops the result if speech resumes.
2. **ASR**: about 350 ms + about 110 ms per second of audio to encode and prefill, then about 28–56 ms per output token.
   Long utterances are decode-bound. The per-language policy cuts at pauses of at least 250 ms while the user is
   still speaking (en/vi/ko), so only the last segment is decoded after end of speech. Not for zh, where
   pause cuts cost about 5 CER.
3. **MT**: first token about 80–100 ms with the prompt cached; **first speakable clause** about 370–420 ms.
   Streaming MT alone was worth −1.16 s.
4. **TTS**: the first clause is synthesised while MT keeps decoding.

## Two ways to run it

| | `harness/src/main.cc` (harness) | `harness/src/mono.cc` (single process) |
|---|---|---|
| shape | two `llama-server` processes over localhost HTTP + in-process TTS | ASR, MT and TTS in one binary, each on its own thread |
| use | benchmarking; every configuration in `harness/README.md` came from it | the product shape; overlaps ASR of the next utterance with MT/TTS of the previous one |
| MT prompt cache | server-side (`cache_prompt`) | explicit per-target KV state save/restore (5–10 ms) |
| status | complete | skeleton: one-shot policy only, file replay (no mic yet) |

HTTP overhead is negligible (about 14 ms per ASR call, about 5 ms per MT call), so the two give the same per-call
timings. OpenCL and HTP coexist fine in one process (measured: same timings, 606 MB).

## Text contracts between stages

`harness/src/pipeline.h` holds the ASR output cleaning, the MT prompt and anchors, MT output cleaning
and the clause chunker. The MT side is documented for the fine-tuner in
`handoff/mt-harness-2026-10-28/README.md`.
