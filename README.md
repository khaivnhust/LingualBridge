# LingualBridge

On-device **speech-to-speech translation** for **English ↔ Vietnamese, Chinese and Korean**, running
fully offline on a Snapdragon 8 Elite phone (OnePlus 13).

```
mic → VAD → ASR (Qwen3-ASR-1.7B, fine-tuned) → MT (Hy-MT2-1.8B) → TTS (Piper) → speaker
          phone GPU / NPU                       phone NPU           phone CPU
```

The goal is to start speaking the translation **under 2 seconds after the user stops talking**.

## Where things stand (2026-10-28)

| | |
|---|---|
| ASR model | Round 6: Qwen3-ASR-1.7B, vocabulary-trimmed, quantization-aware fine-tuned on conversational + Common Voice data; ships as a 1.43 GB Q4_K_M GGUF |
| Best device configuration | ASR audio encoder on GPU, ASR decoder + MT on NPU, TTS on CPU |
| End of speech → first MT token | **median 936 ms, p90 1481 ms, 1 of 113 utterances over 2 s** |
| End of speech → first audible word | median 1530 ms, p90 2410 ms |
| Weakest link | Vietnamese ASR (WER ~19 % on the bench set); no spontaneous Vietnamese training data yet |

Results and verdicts: [`ASR/README.md`](ASR/README.md) (accuracy) and [`harness/README.md`](harness/README.md) (latency, device). Full history: [`docs/history/WORKLOG.md`](docs/history/WORKLOG.md).

## Start here

| you want to… | read |
|---|---|
| set up a machine and clone the repo | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) |
| understand how the pipeline fits together | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| train or evaluate the ASR model | [`docs/guides/asr-training.md`](docs/guides/asr-training.md) |
| run the pipeline / benchmarks on the phone | [`docs/guides/device-benchmarking.md`](docs/guides/device-benchmarking.md) |
| fine-tune the MT model | [`docs/guides/mt-finetuning.md`](docs/guides/mt-finetuning.md) |
| contribute (rules, what not to commit, lessons learned) | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |

## Repository map

| path | what |
|---|---|
| `ASR/` | ASR fine-tuning: trainer (`src/`), data prep / eval / scoring tools (`scripts/`), the shipped recipe (`configs/round6-qat.json`); **results + verdicts in its README** |
| `harness/` | on-device C++: benchmark harness (`main.cc`), single-process pipeline (`mono.cc`), shared text plumbing (`pipeline.h`), build / deploy / phone scripts; **results + verdicts in its README** |
| `translator/` | the original host (PC) pipeline in Python and its tests |
| `handoff/mt-harness-2026-10-28/` | self-contained MT contract package for the MT fine-tuner |
| `patches/` | local llama.cpp fixes the GGUF export needs |
| `llama.cpp/`, `OpenCL-*` | git submodules (pinned) |
| `docs/` | guides, MT training plan (`mt/`), project history (`history/`) |

Datasets, model weights, checkpoints and phone packages are **not** in git; the guides say where they come from.
