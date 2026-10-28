# Getting started

## 1. Clone

```bash
git clone --recurse-submodules git@github.com:zVNoob/LingualBridge.git
cd LingualBridge
git -C llama.cpp apply ../patches/llama.cpp-conversion.patch   # GGUF-export fixes, see patches/README.md
```

Already cloned without submodules? `git submodule update --init`.

> ⚠️ **Paths.** `ASR/` and `harness/` work from wherever you clone: build scripts find the repo root
> themselves, and `ASR/configs/*.json` paths are relative to `ASR/` (run the trainer from there). The older
> `translator/` code (`pipeline.py`, `test_piper_tts.py`) still has absolute paths from the original
> workstation; edit them before use.

## 2. Host machine (training, evaluation, building)

What the project was developed on:

| | |
|---|---|
| OS | Linux x86-64 |
| GPU | NVIDIA, 12 GB (RTX 5070 Ti Laptop); training fits in 12 GB with the recipes in `ASR/configs/` |
| RAM | ≥ 32 GB (QAT keeps fp32 master weights in host RAM, ~16 GB) |
| Disk | ~100 GB free for datasets + checkpoints |
| Python | 3.14, packages installed user-wide |

Python packages (versions that are known to work together):

```bash
pip install --user torch==2.14 transformers==5.17.0 peft==0.21.2 bitsandbytes==0.50.2 \
    datasets==5.1 accelerate tensorboard==2.21 jiwer sacrebleu librosa soundfile scipy sentencepiece
```

Always export this before running the trainer or the dump/eval tools (torch 2.14 otherwise tries to
JIT a Triton kernel that needs Python dev headers):

```bash
export TORCH_DISABLE_NATIVE_JIT=1
```

Host llama.cpp builds (for GGUF conversion, quantization and host-side GGUF evaluation):

```bash
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build llama.cpp/build -j
# optional GPU eval build (needs the Vulkan SDK):
cmake -S llama.cpp -B llama.cpp/build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build llama.cpp/build-vulkan -j
```

`GGML_VK_VISIBLE_DEVICES` takes a device **index** (check `llama-server --list-devices`), not a name. A
wrong value silently falls back to CPU. Confirm with `nvidia-smi` that the GPU is actually busy.

## 3. Models (not in git)

| model | where it comes from |
|---|---|
| ASR base | Hugging Face `Qwen/Qwen3-ASR-1.7B-hf` (native transformers ≥ 5 format) |
| ASR audio encoder for llama.cpp | `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` from ggml-org's Qwen3-ASR GGUF release; used unchanged |
| ASR shipped text model | built by us: trim vocab → fine-tune → convert → Q4_K_M (see the ASR guide); current = Round 6 |
| MT | `hy-mt2-1.8b-q6_k.gguf` (Hy-MT2-1.8B, Q6_K). Its download source was not recorded; copy it from the phone (`/data/local/tmp/s2st-qdc/model/`) or ask the team. The 7B sibling is `unsloth/Hy-MT2-7B-GGUF`. |
| TTS | Piper medium voices `en_US-lessac`, `vi_VN-vais1000`, `zh_CN-huayan`, `ko_KR-kss` (the Korean one is broken under Piper 1.2; see `docs/history/TTS_PLAN.md`) |

## 4. Datasets (not in git)

Each has a prep script in `ASR/scripts/` that writes JSONL manifests
(`{"audio","text","lang","duration","source","id"}`) plus 16 kHz WAVs under `ASR/data/`:

| dataset | script | used for |
|---|---|---|
| FLEURS (en/vi/zh/ko) | `prepare_fleurs.py` | read-speech train + test |
| KsponSpeech (ko, spontaneous) | `prepare_ksponspeech.py` | conversational Korean; **license not cleared for redistribution** |
| VoxPopuli (en) | `prepare_voxpopuli.py` | semi-spontaneous English |
| Common Voice 22 (vi/zh/ko) | `prepare_commonvoice.py` | zh training, vi/zh/ko eval; keep `--exclude-split-sentences` on |
| Round 3 / OneVoice sample | `prepare_round3.py` | out-of-domain eval; numbers are spelled out, so score with `numwords.py` |

Large downloads: HuggingFace is fast; OpenSLR mirrors are slow per connection, so use `aria2c -c -x16`.

## 5. Phone (only for on-device work)

See [`guides/device-benchmarking.md`](guides/device-benchmarking.md). You need: a rooted Snapdragon 8 Elite phone,
`adb`, the Android NDK (r30 was used), and the Hexagon SDK 6.6 for the NPU backend.
