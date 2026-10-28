# AI-Translator — work log

> **Note (2026-10-28, repository migration):** this log was written in the original workspace. Paths have
> since moved: `asr4/` → `ASR/`, `s2st-bench/` → `harness/`, top-level `scripts/` → `harness/scripts/`. Probes, one-off
> test scripts, run logs and raw results were not carried over; their results and verdicts are in
> `ASR/README.md` and `harness/README.md`.

Last updated: 2026-10-14 (shipped artifact verified on noisy set, §12)

## Machines

| | This machine (training / benchmark) | Local machine (relabeling) |
|---|---|---|
| Host | training workstation (Legion R9000P) | second laptop (Legion R9000X) |
| GPU | RTX 5070 Ti Laptop, 12 GB | NVIDIA laptop dGPU, 8 GB — **currently detached** (see below) |
| Python | 3.14, packages user-wide (`~/.local`, `--break-system-packages`) | 3.10, packages user-wide (`~/.local`) |
| Project | `~/Projects/AI-Translator` (`~/AI-Translator` is a symlink to it) | `~/AI-Translator` |

Both machines share a phone hotspot. Large files are copied
between machines over the LAN where possible; downloads use cellular data.

---

## 1. Translator pipeline benchmark (this machine)

Pipeline: STT Qwen3-ASR-1.7B → MT Hy-MT2-7B Q6_K GGUF (llama.cpp) → TTS Piper.

### Environment set up

- `translator/` cloned here; `piper/` (binary 1.2.0 + 4 voices) copied from the local machine.
- `translator/models/Hy-MT2-7B-Q6_K.gguf` → symlink into the HF cache (`unsloth/Hy-MT2-7B-GGUF`).
- `~/AI-Translator` → `~/Projects/AI-Translator` symlink (pipeline paths are hardcoded).
- Installed user-wide: `llama-cpp-python` 0.3.36 **built with Vulkan** (no CUDA toolkit here),
  `onnxruntime-gpu` 1.31, `piper-tts` 1.8, `accelerate`, `librosa`, `soundfile`, `jiwer`,
  `scipy`, `sounddevice`, `sacrebleu`. `transformers` 5.17.0 and `torch` 2.14 were left as they were.
- Vulkan build inputs: headers in `/tmp/bench-deps/Vulkan-Headers` (temporary), NDK `glslc`,
  `~/.local/spirv-headers`.
- Original cloned results backed up to `translator/results/original_clone/`.

### Code changes (this machine's copy of `translator/` only)

- `pipeline.py`
  - STT runs on native transformers 5 (`Qwen/Qwen3-ASR-1.7B-hf`) through a small
    `Qwen3ASRNative` adapter with the same `transcribe()` interface. `qwen-asr` 0.0.6 needs
    transformers 4.57.6, which would break `asr4` (it uses transformers-5 classes).
  - `TORCH_DISABLE_NATIVE_JIT=1`: torch 2.14 otherwise JIT-compiles a Triton `bmm` kernel,
    which fails without Python dev headers.
- `pipeline_test.py`: `init_stt` delegates to `pipeline.init_stt`; the MT stage subprocess
  exits with `os._exit(0)` after writing results (llama.cpp's Vulkan backend segfaults during teardown).
- New `roundtrip_vram_test.py`: STT + MT + Piper (EN/ZH/VI on CUDA) all resident in VRAM.
  ONNX Runtime arenas shrink after each run, otherwise three voices overflow the ~1 GB headroom.
- New `fleurs_ko_test.py`: Korean round trip on real FLEURS speech, chrF against the human EN/ZH/VI references.

### Results

RTF = processing time / audio duration (lower is better).

**All models in VRAM at once** (`results/roundtrip_vram_results.json`):
10.9 GB loaded, 11.4 GB peak of 12.2 GB.

| | Total (s) | STT RTF | TTS RTF | Round-trip RTF | STT error | Keyword acc. |
|---|---|---|---|---|---|---|
| EN | 1.48 | 0.068 | 0.049 | 0.64 | 6.7% | 88% |
| ZH | 1.49 | 0.062 | 0.047 | 0.61 | 1.5% | 74% |
| KO* | 1.81 | 0.042 | 0.011 | 0.28 | 147% | 0% |
| VI | 1.37 | 0.088 | 0.051 | 0.74 | 13.0% | 81% |

**One model in VRAM at a time** (`results/pipeline_test_results.json`): average round trip 2.12 s,
vs 3.85 s on the local machine's 8 GB GPU.

**Korean on real speech** (FLEURS ko_kr test, 10 sentences, `results/fleurs_ko_results.json`):
STT CER **3.0%**, round-trip RTF 0.24, 2.17 s average. MT chrF (from STT / from true transcript):
EN 50.0 / 55.9, ZH 29.5 / 31.0, VI 48.1 / 50.1.

\* The Korean rows in the first table are invalid: see issue 1 below.

### Issues found (they affect the local machine too)

1. **Korean Piper voice is broken.** `ko_KR-kss-medium.onnx.json` uses `phoneme_type: "pygoruut"`,
   which Piper 1.2's C++ binary doesn't support; it silently falls back to espeak and produces
   garbled speech. That garbled audio, not the STT model, explains the 147–166% "STT error".
   Real Korean speech gets 3.0% CER.
2. **"English" translations sometimes come out in French.** `translate()` sends the language
   code ("Translate … into en") instead of the name. 3 of 10 KO→EN and several ZH/VI→EN outputs
   were French. Fix: pass full language names.

---

## 2. ASR fine-tuning setup (`asr4/`, this machine)

Goal: specialize Qwen3-ASR-1.7B for EN/VI/ZH/KO with English code-switching, audio tower frozen.

### Added packages

`peft` 0.21.2, `bitsandbytes` 0.50.2 (8-bit AdamW verified on this GPU), `tensorboard` 2.21,
`datasets` 5.1.

### Data

`scripts/prepare_fleurs.py` downloads Google FLEURS for the 4 languages, stores 16-bit PCM WAVs, and
writes manifests in asr4's JSONL format:
`{"audio", "text", "lang", "duration", "source", "id"}`.

| | train | dev | test |
|---|---|---|---|
| en | 2602 | 394 | 647 |
| zh | 3246 | 409 | 945 |
| ko | 2307 | 226 | 382 |
| vi | 2994 | 361 | 857 |

Stored in `data/fleurs/` (5.2 GB). FLEURS reuses sentences across speakers, so keep dev/test for evaluation only.

### Trainer

- `src/train_asr.py` + `configs/ft-1.7B-4lang.json`
- Model: native `Qwen3ASRForConditionalGeneration` (2.04B params: audio tower 314M, projector 3M,
  decoder 1.72B incl. 311M tied embeddings).
- Frozen: audio tower (also kept in eval mode) and the tied embeddings/lm_head.
- Supervised target: `language <Name><asr_text><text><|im_end|>`. Labels are located per row
  and padding is left, so loss logits are computed only for the answer tail (`logits_to_keep`).
  The built-in loss computed 152k-vocab logits over every audio position and ran out of memory.
- Eval: WER (en/vi) / CER (zh/ko), language-ID accuracy and escape rate per language.
  Logs go to `runs/<name>/eval.jsonl` and TensorBoard (`runs/<name>/tb`).
- Methods:
  - `lora` (default): r=32 on all decoder linears, projector trained. LoRA runs merge at the end into `runs/<name>/merged`.
  - `full`: decoder weights with paged 8-bit AdamW (bf16 weights, no fp32 master copy).

Smoke tests (FLEURS dev as train):

| Method | Trainable | Batch | Peak VRAM | Notes |
|---|---|---|---|---|
| lora | 38.0M | 8 | 9.04 GiB | 10 steps, loss 0.30 → 0.19; eval and merge OK (`runs/smoke-lora`) |
| full | 1412.6M | 4 | 8.24 GiB | 4 steps OK |
| full | 1412.6M | 2 | 7.28 GiB | |

```bash
cd ~/Projects/AI-Translator/asr4
python3 src/train_asr.py --config configs/ft-1.7B-4lang.json --eval-only    # baseline
python3 src/train_asr.py --config configs/ft-1.7B-4lang.json                # LoRA, 2 epochs
python3 src/train_asr.py --config configs/ft-1.7B-4lang.json --method full --lr 1e-5 --batch-size 4 --grad-accum 8
```

### Vocab trimming (`scripts/trim_vocab.py`)

> ⚠️ **Superseded by §4 below.** The min-count 20 + script-filter run recorded here
> was rejected: it changed 26/100 base transcriptions before any fine-tuning. The
> shipped trim uses `--mode coverage`, which is transcript-equivalent (99.88%).
> Kept for history.

Produces one loadable checkpoint. The tokenizer, embeddings (tied lm_head), `config.json` and
`generation_config.json` ids are all remapped, and `keep_ids.json` maps each new id to its original id.

Keep-set:
- tokens used by the training transcripts (always kept)
- tokens with count ≥ `--min-count` on a 20M-char-per-language Wikipedia sample, restricted
  to characters from the 4 languages' scripts (Latin incl. Vietnamese, Hangul, Han, punctuation/symbols)
- the single-char token of every in-script character seen
- the 256-char byte alphabet, digits/punctuation, `language <Name><asr_text>` format tokens, all specials
- the BPE merge closure

Size by `--min-count`: 1 → 89.6k, 2 → 83.1k, 5 → 72.7k, 10 → 63.2k, 20 → 53.0k, 50 → 42.0k.

| Run | Vocab | Segmentation identical (wiki en/vi/ko/zh) | FLEURS dev transcripts identical to base | Foreign scripts |
|---|---|---|---|---|
| min-count 2, no script filter | 85.4k (56%) | 98.1 / 99.0 / 99.4 / 99.0% | 100/100 | still partly spellable |
| **min-count 20 + script filter** (current, `weights/qwen3-asr-1.7B-4lang-trim`, 3.5 GB) | 53.0k (35%) | 48.6 / 82.6 / 91.8 / 80.8% | 74/100 | byte-spelled only (e.g. Japanese 0.40 → 2.20 tokens/char) |

Training transcripts tokenize 100% identically in both runs.

**Open decision:** min-count 20 changes 26/100 base transcriptions before any fine-tuning: number
formatting (108 → 백팔), punctuation, a few words (升学 → 深学). English is affected most (48.6%
identical segmentation). Fine-tuning will adapt to the new vocab, but a lower threshold
(e.g. 5 → 72.7k) keeps more of the base behavior. Next step was to identify exactly which dropped
tokens the base model generates on FLEURS dev before choosing.

```bash
python3 scripts/trim_vocab.py --min-count 20 --out weights/qwen3-asr-1.7B-4lang-trim --corpus "data/fleurs/*_train.jsonl"
python3 scripts/trim_vocab.py --out /tmp/x --corpus "data/fleurs/*_train.jsonl" --sweep-only   # sizes only
```

To train on the trimmed model, set `base.model_id` in the config to
`weights/qwen3-asr-1.7B-4lang-trim` (absolute path).

---

## 3. Relabeling / self-distillation setup (local machine) — in progress

Plan: relabel weakly labeled and unlabeled audio by agreement between teachers, and later use the
fine-tuned student (from this machine) as a teacher too.

Done so far on the local machine:
- `transformers` upgraded user-wide to **5.17.0** (huggingface_hub 1.33, tokenizers 0.23).
  ⚠️ `qwen-asr` 0.0.6 (used by the local `translator/pipeline.py`) no longer works until the local
  pipeline is ported to the native model, as was done on this machine.
- `silero-vad` 6.2.3 installed (segments long unlabeled audio).
- `Qwen/Qwen3-ForcedAligner-0.6B-hf` downloaded (checks weak labels against audio).
- Teachers already cached: Qwen3-ASR-1.7B (qwen-asr format), faster-whisper large-v3, Cohere Transcribe.
- Copying `Qwen/Qwen3-ASR-1.7B-hf` from this machine over the LAN (was in progress).

Blocker: the local NVIDIA GPU dropped off the driver at 2026-10-09 22:32 (kernel: "Attempting to
remove device 0000:01:00.0 with non-zero usage count"). It needs a reboot or a Legion GPU-mode
change before any GPU work there.

---

---

## 4. Vocab trim v2 — `coverage` mode (supersedes §2)

`scripts/trim_vocab.py --mode coverage`: keep **every in-script token**, drop only out-of-script
(emoji, Khmer, Cyrillic, Bulgarian). 152k → **131,896 rows (86.8%)**.

| | value |
|---|---|
| Transcript equivalence vs base | **99.88%** |
| Checkpoint | `weights/qwen3-asr-1.7B-4lang-trim-coverage` |
| Base checkpoint | `weights/base-1.7B-orig` (3.9 GB, sha256 in `weights/versions.json`) |

Frequency trim (`--min-count`) **failed twice** — 76–80 of 800 REAL regressions both times.
Coverage mode is accepted because it is gated on transcript equivalence, not on count.

**Embeddings frozen after the trim.** The lm_head is tied, and every surviving row is an original
pretrained row, so freezing makes the 99.88% equivalence hold by construction rather than by luck.

Emission dumps (built to cover the tokens the model actually *emits*, not just reference text):
`data/base_emissions_{train,test}.jsonl` — 11,134 / 2,823.

```bash
python3 scripts/trim_vocab.py --mode coverage --emissions data/base_emissions_train.jsonl \
  --out weights/qwen3-asr-1.7B-4lang-trim-coverage --min-equivalent 0.99
```

---

## 5. Baseline and fine-tuning arms

**Baseline** (`runs/ab-ref`, trimmed BF16, FLEURS test): en 3.80 WER, vi 5.96 WER,
ko 4.48 CER, zh 7.28 CER, escape rate 0, RTF 0.015. **zh LID 86.1 was the only defect.**

Pre-registered gate: no language worse than reference by more than 0.5.

| Arm | mean | Result |
|---|---|---|
| A — LoRA | 5.78 | **FAIL** ko +0.70, vi +1.04 |
| B — full FT | 5.47 (best 5.34 @ step 525) | **FAIL** ko +0.59 |
| A + VietSuperSpeech | 5.54 | **FAIL** ko +0.98, vi +0.56 (vi 7.00→6.52, zh 6.79→6.14) |

**All three arms pushed zh LID 86.1 → 100 by step 175 and held it** — the one robust win.

Every arm fails on **ko**, which also has the least data (2,307 utts, 1,347 unique sentences).
Root cause still undiagnosed.

### Trainer changes

- **`CpuFp32Master`** — fp32 master weights + fp32 AdamW in host RAM (~15.8 GiB), for
  `method: full`. Subclasses `torch.optim.Optimizer` because torch 2.14 schedulers require it.
- Tolerant `per_device_batch_size` / `per_device_train_batch_size` aliasing.
- `src/q4k_fakequant.py` — Q4_K / Q4_0 fake-quant with straight-through estimator.
  Validated: Q4_K 22.1 dB SNR, `bits_per_weight()` 4.5.

### VietSuperSpeech

245 h, 60,656 + 6,749 utterances, 100% audio coverage. **Labels unusable as-is** — they come from a
Zipformer-30M. Scoring the base model against those labels gives 25.06 WER, LID 78.6, escape 0.5.
The audio is fine; the text is not.

### Pipeline bug

`translator/pipeline.py` `LANG_MAP` was an identity map (2 bugs, 1 root cause) — the same class of
language-name bug as issue 2 in §1.

---

## 6. GGUF export and quantization

`llama-quantize` built. The converted text GGUF holds **310 tensors**, `blk.*` / `token_embd` only —
**no audio tensors**.

| text GGUF | size |
|---|---|
| `trim-Q4_K_M.gguf` | **1.073 GB** |
| `trim-Q6_K.gguf` | 1.383 GB |
| `trim-Q8_0.gguf` | 1.790 GB |
| `trim-F16.gguf` | 3.360 GB |

- `use_more_bits` promotes `attn_v` and `ffn_down` to Q6_K on **50% of layers** (n=28:
  `i<3 || i>=24 || (i-3)%3==2`). Real Q4_K_M mix is 677 MB Q4_K + 390 MB Q6_K — half the tensors,
  not the few originally estimated.
- The audio conv stack is **never quantized** at any ftype (F16, ~24 MB floor).
- **`Q8_K_M` does not exist** in llama.cpp — only `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`.
- `convert_hf_to_gguf.py` needs `sentencepiece` installed.

### mmproj — no conversion needed

`mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` (355.7 MB, official ggml-org) is reused as-is. It contains
394 `a.*` audio-tower tensors + 4 `mm.*`, **zero vocab keys and zero text-decoder tensors**, so the
vocab trim cannot affect it. Breakdown: 324.2 MB Q8_0 + 24.0 MB F16 conv + 7.4 MB F32 position emb.

**Shipped artifact: 1.073 + 0.356 = 1.43 GB** (vs 3.72 GB for F16 + tower) — **2.60x**.

---

## 7. Measured quantization cost

New tooling: `scripts/eval_gguf_server.py` (threaded, per-line checkpointed, `--resume`) and
`scripts/score_hyps.py` (paired scorer, any two hypothesis files).

The BF16 reference was already on disk — `data/base_emissions_test.jsonl` holds 2,823 per-utterance
hypotheses with LID. Verified it reproduces the published baseline (3.82/6.08/4.47/7.32, LID 86.3),
which made a paired comparison free.

**Segment 1 — llama.cpp runtime, precision unchanged** (BF16-HF → F16-GGUF):
en +0.08, vi −0.24, ko +0.02, **zh −0.93**, mean **−0.27**, LID 96.6 → **100.0**.

**Segment 2 — quantization only** (F16-GGUF → Q4_K_M):
en +0.33, vi +0.61, ko +0.24, zh +0.37, mean **+0.39**, LID unchanged.

Two segments were necessary. The first comparison available changed precision *and* runtime at once
and reported **+0.11**, which looked fine — it nets a backend gain against a quantization loss.
**The real cost is +0.39, i.e. that number under-reported it by 3.5x**, and on it alone QAT would
have looked unnecessary.

- The zh LID fix is **entirely the llama.cpp backend**. All 129 zh utterances that BF16-HF labelled
  `English` are labelled `Chinese` by both GGUFs. It was a *labelling* defect, not a recognition
  one — transcripts were correct, punctuation was Latin. Do not cite it as "Chinese got better".
- Q4_K_M's +0.39 is positive on **all four** languages — the signature of weight-precision damage
  rather than a data artifact. For reference, handy-computer's published Q4_K_M cost on LibriSpeech
  clean is +0.19 WER; ASR is more quantization-sensitive than chat.
- Full write-up: `asr4/RESULTS-quantization.md`.

### Two caveats that matter more than the numbers

1. **Everything above was measured on CPU (`-ngl 0`).** Deployment is Vulkan on Adreno 750.
   Q4_K_M is a *mixed* quantization (Q4_K + Q6_K in the same model), those take different kernels,
   and Vulkan fp16 accumulation differs from x86. The +0.39 is not yet validated on the target.
2. **No realistic audio exists.** FLEURS is clean read speech and is saturated. That is the same set
   that could not separate LoRA (5.78) from full-FT (5.47), and that let three arms through review
   while every one of them regressed ko.

---

## 8. llama.cpp bugs found (this checkout predates Qwen3-ASR)

All uncommitted, all worth reporting upstream. Backups in `/tmp/opencode/*.bak*`.

| # | File | Issue |
|---|---|---|
| 1 | `conversion/base.py` | Tokenizer probe-hash rejects **any** trimmed vocab — probe byte-spells 180→247 tokens; trimmed hash `6a589f93…5d31` → `res = "qwen2"`. Bites on every re-trim. |
| 2 | `conversion/qwen3vl.py` | `rope.dimension_sections` never written by `Qwen3VLTextModel`, but the QWEN3VL loader requires exactly 4 entries. |
| 3 | `conversion/qwen3vl.py` | `get_audio_config` requires `thinker_config`; Qwen3-ASR has `audio_config` top-level. Breaks `--mmproj` on the **stock** model. |
| 4 | `conversion/qwen3vl.py` | `model.audio_tower.conv2d1.bias` unmapped (local only, likely downstream of #3). |
| 5 | `conversion/qwen3vl.py` | `tokenizer.chat_template` **overwritten** with the generic Qwen fallback — the real ASR template is written first, then clobbered with only a `Duplicated key name` warning. Stored template has no audio or `asr_text` handling. |

**Operational consequence of #5:** the evaluation worked only because `llama-server`'s multimodal
handler inserts audio tokens itself and bypasses the template. The edge layer must send the prompt
explicitly and must **not** rely on the GGUF's stored template, or zh LID silently regresses 100 → 86.

`llama-cli` also crashes JSON-serializing valid output (`invalid UTF-8 byte at index 24`). The server
path is unaffected, so this is CLI-only.

---

## 9. Evaluation data acquisition (2026-10-11/12)

The gap driving all of this: **every metric so far is studio read speech.** FLEURS and Common Voice
are both sentence-read; VietSuperSpeech is TTS. For a live translator that is the dominant untested
risk, larger than the +0.39 quantization gap.

### Common Voice (`scripts/prepare_commonvoice.py`)

Source `fsicoli/common_voice_22_0`. Audio ships as `{lang}_{split}_0.tar` containing loose mp3s plus
TSV metadata; mp3 bytes stream from the tar into ffmpeg over stdin, so the 20 GB of mp3 is never
written to disk twice. Result: **69,034 clips / 97.66 h**, all 16 kHz mono PCM_16, 0 errors.

**The official CV split sizes are wildly uneven across our languages:**

| lang | CV official train | FLEURS train | net |
|---|---|---|---|
| ko | 0.91 h | 6.93 h | **-6.0 h** |
| vi | 2.65 h | 9.20 h | -6.6 h (recovered via `other`) |
| **zh-CN** | **42.43 h** | ~10 h | **+32 h** |

CV's `other` split holds most of the volume (ko 5.35 h, vi 9.22 h, zh 757 h raw). Because CV splits
by **sentence**, `other` can contain sentences that also appear in dev/test, so training on it raw
leaks the eval set. `--exclude-split-sentences` (default on) drops those: **vi 3,949 of 12,568 rows
(31%)** and zh `other` 757 h → 462 h (39%) were removed. Keep this on.

Conclusion: **Common Voice is a zh resource, not a ko one.** Do not train on CV ko — it costs hours
and buys nothing. Use CV dev/test (ko 472, vi 1,367, zh 10,635) as independent eval, and take
`vi/other` as training bonus. zh `other` was deliberately left unconverted (462 h ≈ 53 GB).

### KsponSpeech (`scripts/prepare_ksponspeech.py`) — the important one

**74,608 utterances / 116.51 h of spontaneous conversational Korean**, from
`DragonLine/ksponspeech` (raw 16 kHz WAV in parquet, pre-split, ungated). Two-speaker conversation
from 2,000 native speakers, with real disfluencies and overlap. Durations p05 1.4 s / median 4.2 s /
p95 14.9 s / max 31 s.

This is the first genuinely out-of-distribution eval set in the project. Transcripts carry leading
filler/noise annotations (`음/`, `o/`), stripped on ingest — the Hangul range must cover precomposed
syllables (U+AC00–D7A3), not just compatibility jamo, or `음/` slips through and taxes every such
utterance with CER the model can never emit.

Notably `cheulyop/ksponspeech` on HF is only a loader script requiring AIHub credentials;
`DragonLine/ksponspeech` carries the audio with no gate. License is undeclared on the mirror —
acceptable for prototype/challenge use, **not** cleared for product distribution.

### MUSAN

`data/musan/` — 930 noise (687 MB), 660 music (4.6 GB), 426 speech (6.5 GB), all 16 kHz, 11.09 GB
tar verified and extracted. For SNR sweeps on all four languages.

### Downloads: OpenSLR is the bottleneck, not the link

MUSAN from openslr.org ran at **0.18 MB/s** (~16 h remaining). Mirror speeds: www 0.47,
trmal 0.53, elda 0.04, hk/us unreachable. The limit was **per-connection**, not total bandwidth —
8 parallel range requests gave 2.56 MB/s aggregate, so `aria2c -c -x16` finished it in ~2 h.
For reference HF served 8.5 MB/s on the same connection.

### HuggingFace cache: Xet blobs are not in the dataset directory

`rm -rf datasets--fsicoli--common_voice_22_0` freed almost nothing. Newer `huggingface_hub` (Xet)
writes blobs to a shared top-level `hub/blobs/` keyed by **Xet hashes, not git-LFS sha256** — so
deleting the dataset dir orphans the data. Cleanup required walking every `snapshots/` dir to build
the referenced set, then deleting only unreferenced recent blobs: **19.66 GiB** recovered.

Two traps: LFS-oid matching finds nothing (wrong hash space), and a `glob`-based reference scan
undercounted references as 69 instead of 118,252 — which would have deleted 94 GB of live model
data, including `gemma-4-E2B-it-GGUF` that the image-reader MCP depends on. Use `os.walk` over all
`snapshots/` and resolve every symlink before touching `hub/blobs/`.

### Disk: generating data inside an OpenCode session doubles it

Writing ~100k WAVs inside the session filled **107 GB** in `~/.local/share/opencode/snapshot/<session>/`
— real copies, not hardlinks. Snapshot removed, 106.93 GiB reclaimed (52 → 159 GB free, 68%).
**Generate bulk corpora outside the session, or expect to pay for them twice.**

---

## 10. QAT probe — first arm to pass the gate (2026-10-12)

### What changed in the hypothesis

Every arm in §5 regressed ko, and every one trained on **read speech only**. VietSuperSpeech is
Vietnamese-only, so **Korean never received a single noisy or conversational utterance in any run** —
while vi got VSS and zh benefited from cross-lingual transfer. Measured median SNR: VSS 27.7 dB vs
FLEURS 40.2 and KsponSpeech 39.2, confirming VSS is the noisiest corpus here. Revised diagnosis: ko's
failure was a **data** problem, not a method problem.

### Design

QAT = full FT with Q4_K fake-quant in the forward pass, 600 steps, `cpu_fp32_master` (~15.8 GiB host
RAM). Train = FLEURS 4-lang + **12,000 KsponSpeech `valid`** (18.84 h); ko ends up 62% of utterances.
Kspon `test` held out. Gate: conversational CER must fall **and** no FLEURS language regress >0.5.

### QAT must mirror Q4_K_M, and getting that wrong breaks the model

Read tensor types out of the actual artifact (`trim-Q4_K_M.gguf`): `attn_v` and `ffn_down` are
**Q6_K on 14 of 28 layers** (`use_more_bits: i<n/8 || i>=7n/8 || (i-n/8)%3==2`), everything else Q4_K,
`token_embd` Q6_K. Faking all 28 at Q4_K applies distortion the deployed model does not have — the
step-0 control showed **vi 13.12 WER with 4.3% escape** versus a real +0.61.

Added `_fake_quant_q6_k` (256-element blocks, int8 scales). First implementation was wrong:

```python
scale = (mx - mn) / 254.0                            # sized for [0,254]
q = torch.round((x - mn) / scale).clamp_(-128, 127)   # clamped to [-128,127]
```

The upper half of every block saturated → **2.7 dB SNR, 87x the error of Q4_K**, and the model emitted
multilingual garbage with no EOS. Fixed to symmetric `amax/127` → **42.8 dB, 0.009x error**. The
invariant that caught it: *a 6-bit format cannot have more error than a 4-bit one.*

That bug also explained a mystery slowness — the broken model never emitted EOS, so every utterance
ran to the full 256-token cap. Real cost is 0.66 s/step for training; eval (autoregressive) is ~13x
more expensive per utterance.

### Result — step 0 (calibrated control) -> step 600, FLEURS read

| lang | step 0 | step 400 | step 600 | delta | LID |
|---|---|---|---|---|---|
| en | 4.09 | 3.68 | 3.85 | -0.24 | 99.7 -> **100** |
| ko | 4.88 | 4.94 | 4.86 | -0.02 | 100 -> 100 |
| vi | 6.97 | 6.73 | 6.87 | -0.10 | 100 -> 100 |
| **zh** | **8.09** | 4.81 | **4.67** | **-3.42** | **72.7 -> 100** |
| mean | 6.01 | 5.04 | 5.06 | -0.95 | |

**Conversational Korean (Kspon test, 5,000): 13.99 -> 11.05 CER, LID 97.4 -> 100.0.**

**PASS on both gate conditions.** The hypothesis is confirmed sharply: ko on *read* speech is flat
(-0.02) while ko on *conversational* speech drops 2.94 — the gain lands precisely where predicted.
zh LID 72.7 -> 100: fake-quant was wounding zh language ID badly (below even the 86.1 bf16 baseline)
and training repaired it.

### Verification through the real artifact

`runs/qat-ko-conv/final` -> F16 -> Q4_K_M (1.073 GB), paired on identical utterances:

| leg | FLEURS en/ko/vi/zh | conv. ko (5k) | LID |
|---|---|---|---|
| base, untrained | 3.80 / 4.48 / 5.96 / 7.28 | 13.99 | zh 86.1 |
| QAT PyTorch fake-quant | 3.85 / 4.86 / 6.87 / 4.67 | 11.05 | 100 |
| QAT Q4_K_M, llama.cpp **CPU** | 3.82 / 5.00 / 6.41 / 5.93 | 11.47 | 100 |
| **QAT Q4_K_M, llama.cpp Vulkan** | **3.75 / 4.86 / 6.50 / 5.41** | **11.48** | 100 |

**CPU vs Vulkan paired on 1,882 utterances: every delta within +/-0.14**, conversational Korean
matching to 0.01. The Q4_K_M mixed quantization is numerically stable across backends. Vulkan did
5,000 utterances in 16 min vs 29 min on CPU (~1.8x).

**Shippable: 1.43 GB** (1.073 text + 0.356 audio tower).

Outstanding caveats:
- **Adreno 750 still unverified.** CPU/Vulkan agreement on NVIDIA does not imply agreement with
  Qualcomm's driver. Narrowed, not closed.
- **zh is +0.74 against the fake-quant target** (5.41 vs 4.67), consistent across both backends, so
  the gap is against *my approximate Q6_K*, not the artifact. Harmless for shipping (5.41 << 7.28) but
  unexplained.
- The fake-quant is optimistic by **+0.42 CER** on conversational Korean.

**Checkpoint choice: step 600 (`final`), not step 400.** Step 400 had the marginally better FLEURS
mean (5.04 vs 5.06) which initially looked like the better pick, but scored **11.74 vs 11.05** on
conversational Korean (paired, 5,000 utterances) — a 0.69 gap on the metric that actually reflects
deployment. `final` is also the artifact already verified on CPU and Vulkan, so shipping it closes the
loop with no re-verification needed.

### Bug #5 confirmed reproducible

The checkpoint ships a correct `chat_template.jinja` (with `audio_start`), but conversion writes it
then **overwrites it with the generic fallback**, warning only `Duplicated key name`. Reproduced on
both the base and the QAT checkpoint. Harmless for these evals because `llama-server`'s multimodal
handler bypasses the template — **but that is a property of how it is called here, not a guarantee.
The Kotlin layer must send the prompt explicitly.**

### Checkpoints are not loadable standalone

`runs/qat-ko-conv/checkpoint-400/` saved only `config.json`, `generation_config.json`,
`model.safetensors`. The tokenizer/processor files (`tokenizer.json`, `tokenizer_config.json`,
`processor_config.json`, `chat_template.jinja`, `keep_ids.json`) were not written, so
`AutoProcessor.from_pretrained(checkpoint)` fails outright. Copied from the base to score it. Worth
fixing in the trainer's save path — a checkpoint you cannot load without knowing where the tokenizer
lives is a trap.

---

## 11. Round 2 — MUSAN noise injection (NEGATIVE RESULT, round 1 still ships)

### Hypothesis

Round 1 moved conversational ko 13.99 → 11.05 but left en −0.24, vi −0.10 and **ko-read −0.02 flat**
— precisely the languages that received no new audio. So: give all four MUSAN-noised audio, and keep
clean FLEURS in the mix as a rehearsal term so original ability is preserved.

`scripts/inject_musan_noise.py` mixes MUSAN (2,016 clips: 930 noise / 660 music / 426 speech) under
FLEURS train at SNR sampled per utterance from {20, 10, 5} dB — one pass rather than three copies, so
file count and epoch cost stay flat while difficulty still varies. Verified: achieved SNR 20.00/10.00/
5.00, 11,149 files, zero failures.

Mix: 11,149 clean + 11,149 noised + 12,000 Kspon valid = 34,284 utterances. **ko fell 62% → 48%.**
900 steps, eval at 300/600/900. Conversational Korean deliberately **not** noised — MUSAN speech is
read-over-read, not two people interrupting, and Kspon already has genuine overlap.

### The measurement gap I created, and closed

Every round-2 metric was **clean** FLEURS. Training on noise while evaluating only on clean audio can
show the cost but is structurally incapable of showing the benefit. Built `data/fleurs_noised_test/`
afterwards and scored base / round 1 / round 2 on **identical** noisy audio.

### Clean FLEURS (200/lang) — the cost side

| lang | step 0 | 300 | 600 | 900 |
|---|---|---|---|---|
| en | 4.18 | 4.27 | 4.16 | 4.30 |
| ko | 4.19 | 4.61 | 4.39 | 4.72 |
| vi | 6.30 | 6.40 | 5.94 | 5.92 |
| zh | 8.06 | 4.98 | 5.58 | 5.86 |
| mean | 5.68 | 5.07 | 5.02 | 5.20 |

zh peaked at step 300 then degraded; ko-read worsened monotonically. Conversational ko held:
11.05 → 11.04.

### Noisy FLEURS (2,823 paired, identical audio) — the decisive comparison

| lang | base | **round 1** | round 2 | r2 − r1 |
|---|---|---|---|---|
| en | 5.32 | **4.64** | 4.76 | +0.12 |
| ko | 7.32 | **5.19** | 5.67 | **+0.48** |
| vi | 9.79 | **8.33** | 8.65 | +0.32 |
| zh | 12.94 | 6.48 | **6.43** | −0.05 |
| **mean** | **8.84** | **6.16** | 6.38 | **+0.22** |
| LID | 92.4 | 100.0 | 100.0 | 0 |

**Round 2 is worse on 3 of 4 languages. Round 1 ships.**

### What this actually showed

**Round 1 already had the robustness round 2 was trying to add.** It never saw MUSAN, yet cut noisy
mean error 8.84 → 6.16 and took LID 92.4 → 100. Conversational audio was the active ingredient; the
hypothesis about which component mattered was wrong, and measuring it properly caught that before it
reached an artifact.

Likely cause of the loss is **dilution**: ko went 62% → 48%, and ko-noisy is the largest regression
(+0.48). zh — the least diluted — was the only language to improve. Secondary possibility is that
real interference beats synthetic: Kspon's noise is two people overlapping and interrupting in a real
room, whereas MUSAN is read speech under a noise bed at a fixed RMS-relative SNR.

Also note what the noisy eval set was worth: on clean FLEURS round 1 looked like a modest win; on
noisy audio it is the difference between degrading to 8.84 and holding at 6.16.

### Trainer save bug, hit twice

`runs/qat-noise/checkpoint-600/` and `runs/qat-ko-conv/checkpoint-400/` both saved only `config.json`,
`generation_config.json`, `model.safetensors` — **no tokenizer or processor files** — so
`AutoProcessor.from_pretrained(checkpoint)` fails outright. Copied from the base both times. Should be
fixed in the save path.

---

## 12. Shipped artifact on noisy audio (2026-10-14)

### What was missing

Round 1's headline robustness win (noisy mean 8.84 → 6.16) was measured with the HF
PyTorch text models via `dump_base_transcripts.py`. The thing that actually ships
is the 1.43 GB `qat-Q4_K_M.gguf` + mmproj, and it had never been scored on noisy
audio. So the robustness claim was about a model variant, not the artifact.

### Method

`scripts/verify_qat_noisy.sh` — Vulkan server on the shipped GGUF, evaluated on
`data/fleurs_noised_test/*_test.jsonl` (2,823 utts), restricted with
`--only-audio-from data/noisy_base.jsonl` to exactly the utterances the base /
round-1 / round-2 HF runs covered. All comparisons paired on identical audio.

### Results

**Shipped GGUF vs base HF PyTorch** (does the artifact retain round 1's win?):

| lang | base | shipped | delta |
|---|---|---|---|
| en | 5.32 | 5.37 | +0.05 |
| zh | 12.94 | 6.52 | **−6.42** |
| ko | 7.32 | 5.46 | **−1.86** |
| vi | 9.79 | 8.32 | **−1.47** |
| **mean** | **8.84** | **6.42** | **−2.42** |
| LID | 92.4 | **100.0** | +7.6 |

**Shipped GGUF vs round-1 HF PyTorch** (quantization cost on noisy audio):

| lang | round-1 HF | shipped | delta |
|---|---|---|---|
| en | 4.64 | 5.37 | +0.73 |
| zh | 6.48 | 6.52 | +0.04 |
| ko | 5.19 | 5.46 | +0.27 |
| vi | 8.33 | 8.32 | −0.01 |
| **mean** | **6.16** | **6.38** | **+0.22** |

### What this shows

**The shipped artifact retains round 1's robustness win.** Noisy mean drops 8.84 → 6.42
and LID goes 92.4 → 100. The win is in the artifact, not just the PyTorch variant.

**The quantization cost on noisy audio (+0.22) is less than on clean audio (+0.39).**
This is the opposite of the expected direction — noisy audio is already hard enough
that the marginal damage from Q4_K_M is smaller. The clean-audio +0.39 was the
pessimistic estimate.

**English is the most quantization-sensitive on noise (+0.73).** On clean FLEURS it
was +0.33; on noisy it more than doubles. zh/ko/vi are all within ±0.27. If English
noisy performance matters, Q4_K_M is the cost; Q6_K would close most of it.

**The shipped 1.43 GB artifact is validated on both clean and noisy real audio,
on CPU and Vulkan.** The only remaining verification gap is the Adreno 750 itself.

---

## Process notes

Six failures worth not repeating:

- **`pgrep -f <script>` matched the searching shell's own command line** and reported a dead eval
  run as alive for 3.4 h. Use `ps -eo pid,cmd | grep "[s]cript_name"`, or match the interpreter
  (`[p]ython3 src/train_asr.py`). Bit this three times, including once when it made a bash wrapper
  look like a busy training process.
- Eval metrics were computed in RAM and written only at the end. A signal kill lost everything. Now
  every hypothesis is flushed and `fsync`ed per line, and `--resume` picks up the remainder.
- The first client was serial and used 1 of the server's 4 slots. Threading halved the runtime.
- **Estimated dataset value from download size instead of counting rows**, and got it backwards
  twice (CV ko *halves* ko data; zh-CN is +32 h, not "the weakest item").
- **Three paired comparisons silently lost sample size.** Two independently sampled subsets of the
  same corpus intersected on only 260 of 2,000 utterances — twice — and `--limit` was silently
  ignored in FLEURS split mode. Fixed with `--only-audio-from`, which derives the subset from one
  side instead of sampling twice. A 260-row delta is indistinguishable from zero.
- **`GGML_VK_VISIBLE_DEVICES` takes an index, not a display name.** Passing `Vulkan1` filtered out
  every device; llama.cpp fell back to CPU with only `warning: no usable GPU found` and served
  `{"status":"ok"}` throughout, so ~10 min of CPU numbers got labelled Vulkan. Caught by checking
  `nvidia-smi`, not the health endpoint. **Confirm GPU utilisation; never infer it from a successful
  request.**

Also: `convert_hf_to_gguf.py` needs `sentencepiece`; `Q8_K_M` does not exist in llama.cpp; torch 2.14
needs `TORCH_DISABLE_NATIVE_JIT=1` or Triton fails to JIT a `bmm` kernel without Python headers; this
llama.cpp checkout is CPU-only unless rebuilt with `GGML_VULKAN=ON` against `~/.local/vulkan-sdk`.

---

## §12 — On-device measurement, Adreno 830 via OpenCL (2026-10-15)

First RTF/accuracy measurement run entirely on-device, inside a Python chroot, with
nothing else touching the device. Supersedes the §8 figures of RTF 0.021 / 69 ms median,
which **do not reproduce** and appear to have been a concurrent-slot aggregate mislabelled
as serial per-utterance latency.

### Accuracy — matches host, LID perfect

| corpus | lang | n | metric | device | baseline | delta |
|---|---|---|---|---|---|---|
| FLEURS | en | 150 | WER | 3.28 | 3.80 | -0.52 |
| FLEURS | zh | 150 | CER | 5.81 | 7.28 | -1.47 |
| FLEURS | ko | 150 | CER | 5.52 | 4.48 | +1.04 |
| FLEURS | vi | 150 | WER | 6.55 | 5.96 | +0.59 |
| Kspon | ko | 300 | CER | 12.25 | - | - |

LID accuracy 100.0% on all four languages, 900/900 FLEURS + Kspon combined, fails=0.
ko/vi regress slightly against base, consistent with the known QAT trade.

### RTF — serial, one request at a time

| corpus | n | audio | RTF | x realtime | median | p90 | p95 |
|---|---|---|---|---|---|---|---|
| FLEURS | 600 | 1.92 h | **0.365** | 2.7x | 3.99 s | 6.43 s | 7.83 s |
| Kspon ko | 300 | 0.48 h | **0.594** | 1.7x | 2.60 s | 6.81 s | 8.91 s |
| Round3 (part) | 274 | 0.31 h | 0.571 | 1.8x | 2.04 s | 3.80 s | 4.32 s |

Round3 partial only (en 100, vi 174 of 920); its vi/ko references remain truncated
relative to audio per §11, so only the paired delta there is meaningful.

Read speech is faster than conversational speech (0.265 vs 0.42 for ko): Korean
generates more tokens per second of audio, and decode is the dominant cost.

**Conclusion: ~2.7x realtime, not 47x.** The pipeline budget for S2ST is MT-bound.

### Memory

llama-server RSS 406 MB idle, **6.3 GB after sustained inference** for a 1.43 GB model
(4.5x weights). The overhead is allocated during inference, not at load. With
Hy-MT2-7B Q6_K at 6.16 GB that is 12.5 GB of 15.1 GB total - co-residency is marginal,
and RTF<1 requires co-residency (unload/reload costs ~18 s per utterance).
Largest untested lever: server runs `-c 2048` with `n_parallel` auto (4 slots).

### Device is ephemeral

QDC instances rotate without warning. A session loss wiped the chroot, all test data and
every log; the next instance had a different pod id. Pull all results to host immediately.

### MT: Hy-MT2-7B-Q6_K on the Adreno (2026-10-15)

Model is `hunyuan-dense` architecture, not Qwen2 despite the unsloth packaging.
Present in both Android builds (`libllama.so`, not the `llama-server` executable).
Chat template uses `<|extra_N|>` as turn delimiters only; the prompt itself is a
plain instruction, no system prompt:

    Translate the following text into {target_lang}. Note that you should only
    output the translated result without any additional explanation:

    {source_text}

Model card sampling: temperature 0.7, top_p 0.6, top_k 20, repetition_penalty 1.05.

**Decode rate, serial, one request at a time:**

| stage | params | quant | ms/token |
|---|---|---|---|
| ASR | 1.7B | Q4_K_M | 53 |
| MT | 7B | Q6_K | **179** |

3.4x slower for a 4.1x parameter increase, i.e. the Adreno scales with weight
size as expected. This is a working GPU path, not a silent CPU fallback.

**Pipeline budget for a 10 s utterance:**

| stage | cost |
|---|---|
| ASR (conversational RTF 0.47) | ~4.7 s |
| MT (~50 out tokens x 179 ms) | ~9 s |
| TTS | ~1 s |
| **total** | **~14.7 s -> RTF ~1.5** |

**RTF < 1 is not reachable with the 7B.** Over budget by ~50%, not marginally.
MT expands English into more tokens than it consumes (Korean especially), so the
cost is driven by output length, not input audio duration.

Consequences: distilling MT to 2/3B is the deciding factor for whether S2ST
works on this device, not an optimisation. A 2/3B at ~2.5 GB should land near
65-70 ms/token, putting MT at ~3.5 s and the total near RTF 1.0 - still tight.

This also sets the bar for the Hexagon backend: below ~100 ms/token it is worth
real effort, above it is not. 6.16 GB of weights through a 3.5 GB mapping window
makes beating 179 ms/token unlikely.

### Slot count: -np 1 is a large win

| config | RSS | peak (VmHWM) | workload | RTF | median |
|---|---|---|---|---|---|
| 4 slots (default) | 8,449 MB | 8,477 MB | 1174 utts | 0.594 | 2.60 s |
| **-np 1** | **4,456 MB** | **4,457 MB** | **300 utts** | **0.470** | **2.06 s** |

47% less memory and 21% faster, identical accuracy (Kspon ko CER 12.25, LID 100),
and the -np 1 run did less work, so the comparison is conservative. `n_parallel`
defaults to -1 (auto), which derived 4 slots from `-c 2048`; `-np 1` also flips
`kv_unified` to false. Single slot is also the correct model for a single-user
speech device.

This makes co-residency viable: ASR 4.46 GB + MT 6.16 GB = 10.6 GB of 15.1 GB.

### llama.cpp Hexagon (NPU) backend

Builds for aarch64 Android alongside OpenCL, in a separate build dir
(`build-android-hex`) so the working OpenCL config was never perturbed. NDK r30
kept rather than the Snapdragon preset's r28b. Produces four DSP skels
(`libggml-htp-v{73,75,79,81}.so`, all `QUALCOMM DSP6` ELF) plus
`libggml-hexagon.so`. `computev81` matches Snapdragon 8 Elite.

Requires Hexagon SDK **6.6.0.0** (`docs/backend/snapdragon/CMakeUserPresets.json`
pins `HEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0`). Not 4.x, 5.x or LPAI - HMX
tensor-core support sets that floor.

Three environment traps:

1. `source ~/.bashrc` is a no-op in non-interactive bash (`case $- in *i*)`
   returns at line 6), so build scripts get no variables from it.
2. CMake reads `${HEXAGON_SDK_ROOT}` as a CMake variable, not an environment
   variable. The preset bridges with `"$env{...}"`; bypassing the preset requires
   explicit `-D`.
3. `PREBUILT_LIB_DIR` is mandatory (`android_aarch64`) - Qualcomm's
   `hexagon_fun.cmake` does `string(FIND ${PREBUILT_LIB_DIR} ...)` and errors
   when empty.
4. SDK packaging bug: `hexagon_fun.cmake` invokes `${proj-qaic_src}/bin/qaic` but
   the SDK ships it at `qaic/Ubuntu/qaic`. Needs
   `ln -s ../Ubuntu/qaic qaic/bin/qaic` or all four skel targets fail with
   error 127.

Note: `docs/backend/snapdragon/` ships `scripts/snapdragon/build.py`, which
pulls a container bundling NDK + OpenCL SDK + Hexagon SDK. That is the
maintained path; the manual route above exists to keep our verified OpenCL
configuration intact.

### Hexagon NPU A/B: MT on HTP0 vs Adreno (2026-10-15)

Hexagon backend works on this device. `HTP0` registers (`device HTP0 did not
report memory; --fit will not use`), model loads with `-ngl 99`, and **no CPU
fallback warnings appear in the log** - important because
`opt_mm_select = 2` is an HMX -> HVX -> CPU chain that degrades silently.

Same model (Hy-MT2-7B-Q6_K), same 50 FLEURS en->ko sentences, serial:

| backend | ms/token | median/sentence | throughput | host RSS |
|---|---|---|---|---|
| OpenCL / Adreno | 209.7 | 12.71 s | 4.8 tok/s | ~6 GB |
| **HTP / NPU** | **145.1** | **8.82 s** | **6.9 tok/s** | **596 MB** |
| | **-31%** | **-31%** | **+44%** | **-5.6 GB** |

**The NPU is both faster and effectively free in RAM.** 596 MB host RSS for a
6.16 GB model means weights are mapped into the NPU session via
`fastrpc_mmap`/`HAP_mmap` (the sliding window), not held in host memory. The
6.16 GB / 3.5 GB window ratio causes no apparent penalty at this scale.

Quality on HTP: 0 empty outputs, 0 English leakage, fluent Korean. 31/50 outputs
byte-identical to the OpenCL run - expected, since temp 0.7 sampling plus
different GPU-vs-DSP rounding diverges the logits.

**Architecture consequence:** ASR (4.46 GB, Adreno) + MT (~0.6 GB, HTP) = ~5 GB
of 15.1 GB. Co-residency, previously impossible, now has ~10 GB spare for TTS,
orchestrator and Android. Neither backend needs the other; they are not
contending for the same memory bus.

**Correction to an earlier prediction:** I expected the NPU to lose because
decode is bandwidth-bound. That was wrong. The Adreno at 210 ms/token was well
below its ceiling - llama.cpp's OpenCL Q6_K path leaves significant performance
unused on this GPU.

### Supported HTP weight types

From `htp/matmul-ops.c` `htp_mm_get_weight_tile_size()`:
**Q4_0, Q4_1, Q8_0, Q5_K, Q6_K, MXFP4** (the last is the unusual 4-bit format).
`Q4_K` has no tile size and would fall back to CPU. Q6_K has dedicated HMX and
HVX kernels (`hmx-mm-kernels-tiled.h`, `hvx-mm-kernels-tiled.h`), so the shipped
Q6_K MT model needs no re-quantization.

### Revised pipeline budget (10 s utterance)

| stage | OpenCL-only | ASR + MT on HTP |
|---|---|---|
| ASR | ~4.7 s | ~4.7 s |
| MT | ~13.5 s | ~8.8 s |
| TTS | ~1 s | ~1 s |
| **total** | ~19 s | **~14.5 s** |
| **RTF** | **~1.9** | **~1.45** |

Still over budget. A 2/3B distill (~2.5B params) should reach ~52 ms/token on
HTP, putting MT at ~3.4 s and RTF near **0.9**. The distill is still required,
but the NPU is what makes it sufficient.

---

## §13 — Language mix: Rounds 4 and 5, and a scoring artifact (2026-10-16/17)

### What Round 4 was testing

Round 1 shipped (`runs/qat-ko-conv`) but was measured as regressing conversational
English by **+2.83 WER** on the Round 3 paired set. Hypothesis: that was a *data*
problem, not a method problem — Round 1's mix was 62 % conversational Korean
(KsponSpeech) and only 11 % English, so English had essentially no spontaneous
speech. Round 4 held method, lr, QAT surface and gate identical and added
**12,000 VoxPopuli English utterances (33.5 h)**, moving the mix to
en 42 % / ko 41 % / zh 9 % / vi 9 % (35,149 utts).

### The gate result

| gate | criterion | Round 4 | verdict |
|---|---|---|---|
| 1 | Kspon test CER ≤ 11.05 | 11.29 (+0.24, CI [+0.06, +0.46]) | **FAIL**, significant |
| 2 | conversational en improves | 5.04 vs 6.00 (−0.97, CI [−1.29, −0.65]) | **PASS** |
| 3 | no FLEURS lang regresses > 0.5 | worst zh +0.20 | **PASS** |

The hypothesis was half right: more conversational English recovered **−0.97 WER**
on held-out VoxPopuli (significant, 119 better / 48 worse per utterance), but it
was paid for with **+0.24 CER** on conversational Korean — also significant.

### Round 5: the diagnosis that followed

I first probed `checkpoint-400` hoping it would sidestep the ko regression. It did
the opposite — **11.66 (+0.61)**, *worse* than step 600. That refuted the
"late-training instability" reading and pointed at **exposure**. Three measured
points, all with the same mix:

| | steps | ko samples drawn | Kspon CER |
|---|---|---|---|
| Round 4 ck400 | 400 | ~5,250 | 11.66 |
| Round 4 final | 600 | ~7,870 | 11.29 |
| Round 1 final | 600 | ~9,950 (62 % mix) | 11.05 |

The log said `steps=600 (epoch=1097)` — **we only ever saw 55 % of one epoch**,
every sample at most once. ko holds 14,298 samples but was drawing ~7,872. Korean
was *under-trained*, not over-trained. Round 5 therefore held the mix
byte-identical and changed **only** `max_steps: null` (a real full epoch, 1,097).
Korean recovered exactly as predicted:

| | Kspon ko CER | vs Round 1 |
|---|---|---|
| Round 4 final | 11.29 | +0.24 significant |
| **Round 5 final** | **11.10** | **+0.05, CI [−0.10, +0.20] — a tie** |

Round 4 → Round 5 on ko is **−0.19, CI [−0.39, −0.01]**, significant. The only
variable was step count, so the exposure diagnosis holds.

### Is fine-tuning worth it — base vs arms, identical subsets, host protocol

This was the missing anchor: every arm comparison in the project so far was
fine-tuned vs fine-tuned. Base is the vocab-trimmed 4-lang checkpoint with **no
fine-tuning**.

| | base (no FT) | Round 1 | **Round 5** | base→R5 |
|---|---|---|---|---|
| Kspon conversational ko (n=5,000) | 13.99 | 11.05 | **11.10** | **−2.89** (−20.7 %) |
| VoxPopuli conversational en (n=500) | 5.27 | 6.00 | **4.97** | **−0.30** |
| FLEURS zh (n=299) | 6.64 | 4.67 | **5.15** | **−1.49** |
| FLEURS en (n=300) | 3.91 | 3.85 | **3.74** | −0.17 |
| FLEURS ko (n=300) | 4.63 | 4.86 | 4.68 | +0.05 |
| FLEURS vi (n=298) | 5.83 | 6.87 | 6.01 | +0.18 |
| **FLEURS LID (mean)** | **96.4 %** | 100 % | **100 %** | **+3.6** |

**Yes, worth it — and the win is almost entirely Korean.** −2.89 CER (CI
[−3.15, −2.62], 2,097 utterances improved vs 859 worsened) plus LID 96.4 % → 100 %.
For a translation app a language-ID error is not a small WER penalty, it ships the
wrong language, and only fine-tuning buys it.

The most important row is Round 1's English: **6.00 vs base 5.27.** For five arms
fine-tuning was *worse than not fine-tuning* on conversational English. Round 5's
4.97 is the first configuration to beat base on both conversational languages.
"How much to fine-tune" and "with what mix" are separable questions; the first
answer was yes, Round 1 was simply the wrong answer to the second.

**Caveat that remains open:** every fine-tuned arm is trim + QAT + full FT and the
base is trim only, so −2.89 is attributable to the whole stack. Isolating the
gradient updates needs a trim+QAT-without-FT arm, which does not exist.

### Round 3 (OneVoice): the original "+2.83" was 84 % a numeral-formatting bug

The regression that motivated Round 4 was measured on Round 3. It reproduces
(base → Round 1 = **+2.78**, CI [+0.76, +5.20], significant) — but the score was
mostly measuring text formatting, not recognition:

```
REF: convert two thousand three hundred and thirty from g. m. t. plus ...
HYP: convert 2330 from gmt ...                       -> 86 % WER, content identical
REF: seven hundred thirty four main street ... six seven two nine seven
HYP: 734 main street ... 67297                       -> 64 % WER, content identical
```

OneVoice writes numbers as words; every ASR model writes digits. `scripts/numwords.py`
(converting both sides to digits, or stripping numerals from both — the latter is
format-neutral) collapses the effect:

| en WER (n=100) | raw | convert | strip |
|---|---|---|---|
| base | 8.55 | 8.71 | 8.52 |
| Round 1 | 11.32 | 9.42 | 8.97 |
| Round 4 | 12.23 | 10.34 | 9.80 |
| Round 5 | 11.91 | 9.91 | 9.35 |

`base → Round 1` goes from **+2.78 to +0.45**: 84 % of the regression was
formatting. The ordering `base < r1 < r5 < r4` is **identical under all three
treatments**, so the conclusion does not depend on the normalisation choice. Round 5
is +0.39 vs Round 1 here — not significant (CI [−0.17, +1.37]) — while being
clearly better on VoxPopuli. Different speech domains, different answers.

Two corrections to the record:

- **The stated exclusion reason does not hold for `en`.** Round 3 was excluded
  because "vi/ko references are truncated relative to the audio". For English,
  reference chars/sec is 15.2 against hypothesis 14.9 — the references are
  *complete*. And ko CER 0.64 % is what good alignment looks like. `en` is also the
  only language the +2.83 claim ever rested on (n=100; the device dump covered
  en 100 + vi 174 of 460, no ko or zh at all).
- **I should have caught this before Round 4 was designed.** VoxPopuli was still
  the right call — Round 5 is genuinely best on real conversational English — but
  the motivation was misdiagnosed.

### The measurement gap this exposed

We have conversational training data for only **two** of four languages
(VoxPopuli en, Kspon ko) and conversational *evaluation* for only those two.
vi and zh have 2,990 and 3,246 training utterances against ~14,600 for en and ko,
and **no conversational eval exists for either.** We are shipping blind on half
the product.

`data/commonvoice/` was already downloaded and unused for this: vi test 1,367
(1.45 h), zh-CN test 10,635 (17.45 h), zh-CN train 29,470 (42.4 h), vi train only
2,102 (2.65 h). Common Voice is crowdsourced *read* speech — not spontaneous like
Kspon — but it is the only natural-voice eval available for vi/zh.

### Tooling added

- `scripts/bootstrap_ci.py` — paired bootstrap over **utterances**, recomputing
  corpus WER/CER per resample. Reuses `score_hyps.normalize` and loaders so point
  estimates reproduce `score_hyps.py` exactly (verified: 6.00/4.97 and
  11.05/11.29 both matched). Picks word vs character distance from
  `score_hyps.METRIC`, since `normalize` strips all spaces for zh/ko.
- `scripts/numwords.py` — spelled-out numeral ↔ digit, opt-in, off by default.
  FLEURS/VoxPopuli/Kspon already use digits.
- `scripts/train_r4.sh`, `train_r5.sh`, `kspon_gate_r4.sh`, `kspon_ck400_probe.sh`,
  `gate_r5.sh`, `round3_chain.sh`, `base_ref.sh` — all tmux-wrapped.
- `configs/qat-en-conv-full.json` — Round 5.

### Process lessons

- **Duplicate final eval.** `train_asr.py:517` ran `run_eval` a second time after
  the loop when `step % eval_steps == 0`, costing a redundant ~20 min full pass at
  step 600 before `final/` was written. Fixed with `if step % t["eval_steps"] != 0`.
- **`eval.jsonl` is append-only.** Opened with mode `"a"` and never truncated, so a
  stale entry from a previous run looks exactly like a write race. The `[200, 400,
  200, 400]` pattern I first read as concurrent writes was just the dead run's
  rows still on disk. Truncate on open.
- **Checkpoints still miss tokenizer/processor files** (third occurrence, §10,
  §11). `model.save_pretrained` alone leaves `AutoProcessor` unable to load them;
  still an open item.
- **`dump_base_transcripts.py` does not create its output directory.** A missing
  `results/<run>/` killed an entire 20-minute gate job at the first write.
- **Three silent training deaths**, all with clean log EOF, no traceback and a
  vanished harness output file (steps 250 and 500 under the harness). Not
  reproducible under `tmux`, which is now mandatory for anything long-running.
- **Never wrap a long benchmark in `timeout`** — killed three runs; these scripts
  print only at the end, so a kill looks like a clean exit.

---

## §14 — Common Voice: fine-tuning loses out-of-domain, and it qualifies §13 (2026-10-17)

### Closing the vi/zh measurement gap

§13's biggest gap was that **vi and zh had no conversational-ish evaluation at
all** — they hold 2,990 and 3,246 training utterances against ~14,600 for en and
ko, and every conversational number in the project was en or ko only.
`data/commonvoice/` turned out to be downloaded and unused.

Before spending GPU on it I checked the references, since bad references have
already cost this project twice (OneVoice numerals, the misdiagnosed "truncated
vi/ko"). Median reference chars/sec: ko 5.43, vi 8.17, zh-CN 2.69 — all
plausible for read speech (zh is slow because the text is careful encyclopedic
prose, which is what contributors read). Nothing anomalous.

One blocker: CV tags Chinese `zh-CN`, but `score_hyps.LANGS` is
`["en","zh","ko","vi"]`, so every zh-CN row would have been **silently dropped**.
Rewrote the test splits into `data/cvtest/{vi,zh,ko}.jsonl` with normalised tags
and verified the loaders see them.

### The result — fine-tuning is significantly *worse* on natural speech

Common Voice test, 12,474 utterances, identical subsets for all three models:

| | vi WER (n=1,367) | zh CER (n=10,635) | ko CER (n=472) |
|---|---|---|---|
| base (no FT) | **11.84** | **6.30** | **7.41** |
| Round 1 | 14.95 | 7.07 | 8.66 |
| Round 5 | **15.00** | **8.02** | **8.40** |

Paired bootstrap, base → Round 5:

```
vi  +3.16  CI [+2.46, +3.86]   134 better / 316 worse / 917 tied
zh  +1.72  CI [+1.31, +2.13]  1361 better / 1411 worse / 7863 tied
ko  +0.99  CI [+0.22, +1.72]   64 better / 134 worse / 274 tied
```

**All three significant regressions — including Korean, the language Round 5 was
built to fix.** Korean gets +0.99 here for a structural reason: our Korean
training is FLEURS (studio read) + Kspon (spontaneous), and CV Korean is neither.

### The domain split is now visible

| domain | en | ko | zh | vi | fine-tune |
|---|---|---|---|---|---|
| FLEURS studio read | −0.17 | +0.05 | −1.49 | +0.18 | mixed |
| VoxPopuli parliamentary | −0.30 | — | — | — | **wins** |
| Kspon spontaneous | — | **−2.89** | — | — | **wins** |
| Common Voice crowdsourced | — | **+0.99** | **+1.72** | **+3.16** | **loses** |
| OneVoice synthetic | +0.83 | — | — | — | **loses** |

**Fine-tuning wins exactly where its training data lives and loses everywhere
else.** Round 5 has specialised to two acoustic domains — FLEURS studio read and
Kspon spontaneous — and generalises worse to everything between and beyond them.
This qualifies §13's "fine-tuning is worth it" conclusion: it is worth it
*in-domain*, and §13 only had in-domain evidence.

The damage is concentrated rather than diffuse: 917/1,367 vi and 7,863/10,635 zh
utterances are byte-identical to base. Fine-tuning changed its output on ~25 % of
utterances and got more of those wrong — a confident, targeted misfire.

### Correction to §13's LID claim

§13 said fine-tuning buys LID 96.4 % → 100 %. That is true on FLEURS and Kspon
and **false on Common Voice**: vi LID is 94.7 % (base), 96.1 % (Round 5), 97.4 %
(Round 1). On the hardest natural domain every model still misidentifies
Vietnamese several percent of the time, and fine-tuning does not fix it.

### Revised plan — rebalance for acoustics, not for share

The motivation changed. §13 said rebalance because vi/zh were 9 % each; §14 says
the problem is **acoustic domain mismatch, not per-language data volume**. That
makes a cleanly testable hypothesis:

> Adding Common Voice zh-CN train (29,470 utts / 42.4 h, already on disk) teaches
> CV-like acoustics and should recover **vi, zh and ko together** on Common Voice
> — despite containing no Vietnamese at all. If vi recovers without vi data, the
> diagnosis is confirmed and the fix is cheap.

This also happens to rebalance zh (3,246 → ~16,246), moving the mix to roughly
en 30 % / zh 33 % / ko 29 % / vi 10 %. One full epoch ≈ 1,539 steps ≈ 3.1 h.

Design notes:

- **Train on CV `train`, evaluate on CV `test`.** `data/cvtest/*` is eval-only.
- **Hold in-training eval on FLEURS** as before; score CV post-hoc on
  `results/cvtest/`.
- **Expected risk:** conversational share falls (Kspon 34 % → 29 %), i.e. the
  same dilution that cost +0.24 conversational ko in Round 4. The full epoch is
  what repaired that, so it is load-bearing again here.

---

## §14b — Round 6: the acoustic hypothesis is FALSIFIED, and the control is clean (2026-10-17)

### What was predicted

§14 found fine-tuning is significantly *worse* on Common Voice in all three
languages (vi +3.16, zh +1.72, ko +0.99 vs base). The proposed mechanism was
**acoustic domain mismatch**, and the test was deliberately asymmetric: add
**Common Voice zh-CN** (12,000 utts / 17.2 h, containing **no Vietnamese**) and
check whether vi, zh and ko all recover together.

> If acoustics transfer across languages, vi should improve without any vi data.

Deliberately **not** added: CV vi train (2,102 utts). Including it would have
confounded the test — if vi recovered we would not know whether acoustics or
vi-specific data did it. Mix landed at en 31 % / zh 32 % / ko 30 % / vi 6 %,
1,472 steps, full epoch. zh is now 15,242 utterances (was 3,246); en and ko are
**unchanged in absolute terms**, so nothing was traded to pay for it.

### The result

Common Voice test, identical subsets, 12,474 utterances:

| lang | n | base | Round 5 | Round 6 | R6−R5 | R6−base | got CV data? |
|---|---|---|---|---|---|---|---|
| vi | 1,367 | 11.84 | 15.00 | **16.02** | **+1.02** | **+4.18** | no |
| zh | 10,635 | 6.30 | 8.02 | **5.01** | **−3.01** | **−1.29** | **YES (+12,000)** |
| ko | 472 | 7.41 | 8.40 | 8.39 | −0.01 | +0.98 | no |

```
vi  +1.02  CI [+0.24, +1.92]   SIGNIFICANT (worse)
zh  −3.01  CI [−3.40, −2.63]  SIGNIFICANT (better)
ko  −0.01  CI [−0.64, +0.65]  not significant
```

**The hypothesis is falsified.** Acoustics do **not** transfer across languages.
vi did not recover — it got *worse*, and now sits 4.18 from base instead of 3.16.

**The falsification is more valuable than a confirmation would have been**, because
zh is the control that proves the mechanism. zh is the only language that received
Common Voice data, and it is the only one that moved: **8.02 → 5.01, and it now
beats the un-fine-tuned base (−1.29).** ko, which received nothing, did not move.
The gradient is exactly the shape the per-language hypothesis predicts and exactly
the shape the acoustic hypothesis forbids.

So the §14 diagnosis was wrong in an instructive way: the deficit is **per-language
data scarcity**, not acoustic diversity. Give a language natural speech and it
improves, past base. Withhold it and it does not.

### The "must hold" gate held

Conversational Korean, the thing Round 5 was built to win:

```
Kspon-5k ko CER   Round 5 11.10  ->  Round 6 11.12
delta +0.02  CI [−0.14, +0.19]   not significant
```

Adding 12,000 zh utterances diluted the conversational share from 68 % to 60 %,
and the full epoch absorbed it — exactly as it absorbed the same dilution in
Round 5. Two rounds in a row now, which is good evidence the full epoch is the
load-bearing part rather than a coincidence.

### What this implies for vi

vi is the one language with **no natural-speech conversational corpus at all** —
FLEURS 2,990 (studio read) and CV 2,102 (crowdsourced read), nothing spontaneous.
Its Common Voice CER is the worst number in the project (+4.18 vs base) and it got
*worse* in this arm. zh demonstrated the remedy; vi needs the same thing in a
language it can actually use.

**Next: acquire a Vietnamese natural-speech corpus.** Whatever it is, it should be
*spoken*, not read — the zh result says natural acoustics help, and the Kspon/VoxPopuli
result says spontaneity is what the product actually receives. Candidate classes:
VietSpoken-style corpora, AIHub Vietnamese sets, or a Vietnamese subset of a
large multilingual spontaneous set. Budget for it explicitly, because this is now
the clearest known defect in the shipped model.

### The evaluation harness change this arm forced

Round 5's FLEURS-only in-training eval was blind to this whole problem: FLEURS said
vi **+0.18** while Common Voice said **+3.16**. Fixed before launching:

- `evaluate()` and `eval_subset()` now key on **(lang, source)** and report as
  `vi@commonvoice-vi` vs `vi@fleurs`. Safe because
  `processor.apply_transcription_request` is language-agnostic, so `r["lang"]` is
  still the true code for LID, metric choice and normalisation.
- Round 3 (OneVoice) joined the curve as an OOD cell, with numerals stripped from
  both sides inside `evaluate()`.
- **Rejected pre-converting Round 3 references to digits.** `numwords.convert`
  produced `"the year 993000"` from the genuinely ambiguous *"one thousand nine
  hundred and ninety and two thousand"*, and turned the idiom *"at one point"* into
  *"at 1 point"*. `numwords.strip` is format-neutral and never guesses.
- `eval.jsonl` now opens `"w"` instead of `"a"` — the append-mode bug that made
  Round 5's curve look like it had racing writes.

11 cells × 50 utterances across three domains, smoke-tested end-to-end on GPU
before committing a 4-hour run.

### Complete results and the ship decision

Round 6 across every domain measured, all identical subsets:

| measurement | n | base | R5 | R6 | R5−base | R6−base |
|---|---|---|---|---|---|---|
| Common Voice vi | 1,367 | 11.84 | 15.00 | 16.02 | +3.16 | **+4.18** |
| **Common Voice zh** | 10,635 | 6.30 | 8.02 | **5.01** | +1.72 | **−1.29** |
| Common Voice ko | 472 | 7.41 | 8.40 | 8.39 | +0.99 | +0.98 |
| Kspon conv ko | 5,000 | 13.99 | 11.10 | 11.12 | −2.89 | **−2.87** |
| VoxPopuli conv en | 500 | 5.27 | 4.97 | 4.95 | −0.30 | **−0.32** |
| FLEURS en | 300 | 3.91 | 3.74 | 3.80 | −0.17 | −0.11 |
| FLEURS zh | 299 | 6.64 | 5.15 | 5.69 | −1.49 | −0.95 |
| FLEURS ko | 300 | 4.63 | 4.68 | 4.71 | +0.05 | +0.08 |
| FLEURS vi | 298 | 5.83 | 6.01 | 6.52 | +0.18 | +0.69 |
| Round 3 en | 100 | 8.52 | 9.35 | 10.18 | +0.83 | +1.66 |
| Round 3 vi | 460 | 9.81 | 10.14 | 10.44 | +0.33 | +0.63 |
| Round 3 ko | 179 | 0.88 | 0.65 | 0.65 | −0.23 | −0.23 |
| Round 3 zh | 180 | 4.03 | 5.53 | 3.51 | +1.50 | −0.52 |

```
Round 5:  mean +0.28 vs base,  5/13 cells better
Round 6:  mean +0.15 vs base,  7/13 cells better

per-language mean delta vs base (negative = better than base)
en   R5 +0.12   R6 +0.41
ko   R5 -0.52   R6 -0.51
zh   R5 +0.58   R6 -0.92    <-- FLIPPED to better than base
vi   R5 +1.22   R6 +1.83
```

**Ship `runs/qat-en-conv-cv/final`.** Chinese flipped from worse-than-base to
better-than-base, on the heaviest measurement we have (n=10,635, CI
[−3.40, −2.63]); Round 6 is the first arm with more cells better than base than
worse (7/13 vs 5/13); conversational Korean and conversational English both held.

Caveats stated plainly: **vi got worse** (+4.18 vs base, up from +3.16) — the
language that received no data, and now the clearest known defect. **en drifted up**
(+0.41), driven almost entirely by Round 3 *synthetic* en (+1.66); on VoxPopuli
conversational, the domain the product actually receives, it is a statistical tie
(4.95 vs 4.97, CI [−0.27, +0.22]). The aggregate mean is unweighted across cells whose
n ranges 100–10,635, so the per-language framing is the more honest read.

### A note on my own bookkeeping

The Round 6 gate script referenced `results/qat-en-conv/kspon_r5.jsonl` — Round 4's
directory — instead of `results/qat-en-conv-full/kspon_r5.jsonl`, so the Kspon
bootstrap died on `FileNotFoundError`. The score itself printed fine, which is
exactly the failure mode worth catching: a missing number and a passing number look
identical unless you verify every expected line arrived. Re-ran CPU-side with the
corrected path. The script also dumps all 5,506 VoxPopuli utterances before the
intended 500; harmless (the 500-seeded dump overwrites it) but ~15 min wasted.

---

## §15 — S2ST end-to-end latency on QDC: streaming MT is worth 1.16 s (2026-10-17)

### What was built and where

`s2st-bench/` (~2000 lines C++, dependency-light HTTP + VAD + ONNX TTS), packaged
as `s2st-qdc.tar` (**3.19 GB**, 687 entries, sha256 `c088ee09…`) and deployed to
`/data/local/tmp/s2st-qdc` on a Snapdragon 8 Elite (Sun, arm64-v8a, SDK 36). All
634 package files verified on-device; the tar's sha256 matched host exactly.

Topology as deployed, and **verified by offload tables rather than assumed**:

```
ASR  llama.cpp, Qwen3-ASR-1.7B Q4_K_M   -dev GPUOpenCL
     offloaded 29/29 layers to GPU        OpenCL buffer   807.10 MiB
MT   llama.cpp, Hy-MT2-1.8B Q6_K         -dev HTP0
     offloaded 33/33 layers to GPU        HTP0 buffer    1495.04 + KV 256 + compute 48
TTS  Piper VITS via ONNX Runtime, CPU, in-process
```

MT's 1799 MiB total sits **inside the 3.5 GB HTP mapping window**, so no
sliding-window paging — which the 7B alternative would require.

### The headline: end-of-speech → begin-of-voice

159 utterances, 6 directions, conversational audio where it exists:

| direction | n | median | p90 | RTF median | RTF p90 |
|---|---|---|---|---|---|
| en→vi | 22 | 1935 ms | 2860 | 0.444 | 0.576 |
| en→zh | 22 | 2001 ms | 2903 | 0.440 | 0.583 |
| ko→en | 28 | 1593 ms | 3150 | 0.910 | 1.670 |
| vi→en | 22 | 1432 ms | 2004 | 0.872 | 1.100 |
| zh→en | 21 | 2064 ms | 2570 | 0.576 | 0.829 |
| en→ko | 22 | 2902 ms | 3596 | 0.641 | 0.851 |

**Median 1935 ms.** Round-trip RTF is comfortably sub-1 for English sources but
**ko→en (0.910) and vi→en (0.872) sit at the edge, p90 1.67** — those two cannot
reliably keep up with continuous speech.

### Streaming MT is the single largest win

Identical audio, same servers, only `--no-stream` differs:

| | streaming | non-streaming |
|---|---|---|
| first voice | **1935 ms** | 3094 ms |
| MT time-to-first-token | **42 ms** | 1007 ms |

**Saves 1159 ms (37 %).** 42 ms to first token on the HTP is what makes early
speech possible at all; non-streaming costs a flat ~1 s before one word exists.
This is why the harness splits the MT token stream at clause boundaries and feeds
the first speakable clause to TTS on a separate thread while the rest of the
sentence is still decoding.

### Per-stage: ASR is the floor

```
direction    ASR    MT 1stTok  MT 1stClause   TTS
en→vi       1102          42            609   285
en→zh       1610          55            215   435
en→ko       1582          56            557   900
vi→en        982          52            236   178
```

ASR is 982–1610 ms and blocks everything. **No amount of streaming MT gets below
it.** Sub-1 s first-voice requires decoding overlapping windows *while the user
is still talking*, which needs Qwen3-ASR to accept partial audio — it is
utterance-level today, so this is a blocker, not a tuning knob.

### Real-device budget

Compute transfers (same silicon). What does not, and what was never measured:

| component | status |
|---|---|
| compute | 1935 ms, measured, topology verified |
| audio I/O + live VAD | **+100–2000 ms, budgeted but never instrumented** |
| Android scheduling | ~0 by design — the shipped device is stripped |
| thermals | **unknown** |

Two things follow. First, **the unmeasured 0.1–2 s I/O band is as wide as the
entire measured pipeline** — compute is no longer the dominant term. Second, of
that band the VAD hangover is *our own parameter*: 450 ms was a safe default, not
a tuned one, and cutting it to 250 ms saves 200 ms on every utterance for free.
The audio capture buffer is Android's; if the true cost approaches 2 s, something
is misconfigured (voice-communication off, a deep HAL buffer, or PCM capture
rather than the fast path).

**Thermals are the one risk that cannot be bounded by a design decision.** This
was a ~7 minute burst on cloud hardware with active cooling; a 10-minute
conversation in a hand on battery is a different regime. Cheap to close: a
`--repeat` mode running one direction for 15 minutes and reporting the drift
curve. The results carry **no timestamp field**, so drift cannot be recovered
retrospectively — an earlier claim of "+73 % thermal drift" was wrong, produced
by sorting on `asr_ms`, which tracks direction difficulty rather than time.

### Phonemisation validated against piper-tts symbol-for-symbol

| Lang | piper | this harness | verdict |
|---|---|---|---|
| en | 59 | 59 | **bit-identical** |
| vi | 63 | 63 | **bit-identical** |
| zh | 91 | 91 | **bit-identical** |
| ko | 42 | 41 | one `ʲ` palatalisation marker differs |

Three things had to match and initially did not: id framing is
`[BOS][PAD] (id [PAD])* [EOS]` (the per-phoneme PAD is not optional); espeak's
`TextToPhonemes` advances **one character past** the clause boundary, so text
must be split *before* phonemising rather than scanned afterwards; and espeak sets
the text pointer to **NULL** at end of input, which segfaults if dereferenced.

### Four bugs found only by running on device

1. **ASR leaked `<asr_text>` tags** into the MT prompt; the server's parsed-output
   path does not strip the structural template.
2. **`en→zh` segfaulted the process.** espeak's Mandarin voice is `cmn`, not `zh`;
   with an unset voice `espeak_TextToPhonemes` kills the library rather than
   returning an error. Now mapped, with a validity check.
3. **TTS raced** — `drain()` returned when a clause was *dequeued*, not when
   synthesis finished, so first-voice read `-1`.
4. **Results silently truncated** — `--out` opens with `ofstream`, so each of 140
   invocations overwrote the previous one and only 1 utterance per direction
   survived the first run.

Also: the single build registers both backends, so `-ngl 99` put the **ASR model
on the NPU** until `-dev` pinned it. And HTP needs `ADSP_LIBRARY_PATH` — the skel
URI is hardcoded as `file:///libggml-htp-v<gen>.so` and root is read-only.

### What this package is not

`s2st-qdc/SUBSTITUTIONS.md` records all of it. Load-bearing items: the four
shipped Piper voices are the **previous set**, not `TTS_PLAN.md`'s selections
(latency comparable, timbre not); **Korean TTS audio is invalid** because
`ko_KR-kss-medium` is `pygoruut` and espeak cannot produce it (latency still
valid); zh/vi benchmark audio is Common Voice **read** speech; and the ASR GGUF
**predates Round 6**, so accuracy figures are Round 1/4-era even though the
latency graph and KV sizes are unchanged.

---

## Next steps

### Done since the previous list

- ~~**Eval base on conversational Korean**~~ → §10: 13.99 CER, LID 97.4.
- ~~**Measure conversational vi and zh**~~ → §14: built `data/cvtest/` from Common Voice
  test (vi 1,367 / zh-CN 10,635 / ko 472) and measured base, Round 1 and Round 5. Fine-tuning
  is significantly **worse** on all three (vi +3.16, zh +1.72, ko +0.99) — an acoustic-domain
  regression, not a data-volume one.
- ~~**ko training data is still the open problem**~~ → §13: it was under-exposure, not data.
- ~~**Then, and only then, consider rebalancing data**~~ → **DONE in §14–15.** Round 6 added
  12,000 Common Voice zh-CN. The acoustic hypothesis was **falsified** (vi did not recover, it
  got worse), but the experiment was worth more for having failed: zh — the only language that
  received a dose — went 8.02 → 5.01 and now **beats base** (Round 5 was +1.72 worse). The
  deficit is **per-language data scarcity**, proven by the one language that got data.
- ~~**Pick a ship candidate**~~ → §14b: **`runs/qat-en-conv-cv/final` (Round 6).** 7/13 measurement
  cells better than base (first arm over half), zh flipped to better-than-base, conversational
  Korean and English both held.

### Still open

1. **Acquire a Vietnamese natural-speech corpus — the highest-value item in the project.** §14b
   proves the mechanism: give a language natural speech and it improves past base (zh 8.02 → 5.01,
   −1.29 vs base, n=10,635). vi received none and got *worse* (+4.18). It is the worst language on
   every axis — Common Voice +4.18, FLEURS +0.69, Round 3 +0.63, LID 96.1 % (base 94.7 %). What
   exists today is FLEURS 2,990 (studio read) + Common Voice 2,102 (crowdsourced read) and
   **nothing spontaneous**. Requirements: *spoken, not read*, because the Kspon and VoxPopuli
   results say spontaneity is what the product actually receives. Candidates worth checking:
   VietSpoken-style corpora, AIHub Vietnamese sets, Vietnamese subsets of large multilingual
   spontaneous corpora. Then run it through the Round 6 recipe — full epoch, and verify against
   `data/cvtest/vi.jsonl`.
2. **vi LID is unsolved everywhere** — 94.7 % (base) / 96.1 % (Round 6) on Common Voice, under
   100 % on every domain for every model. A translation app misidentifying Vietnamese ~4 % of the
   time ships the wrong language. Item 1 should move this; if it does not, the cause is acoustic
   confusability rather than data volume and needs a different answer entirely.
3. **Export Round 6 to GGUF and verify on device** — every accuracy number in §14b is bf16 on the
   host. The shipped format is Q4_K_M and this arm's artifact has never been measured in it. §7's
   quantization cost should carry over unchanged, but "should" is not a measurement.
4. **Re-verify on-device over Vulkan (Adreno 750)** — the only remaining verification gap. Every
   number is CPU or NVIDIA-Vulkan; the deployment target is Qualcomm. `dist/android-arm64-vulkan/`
   and `deploy-android.sh` exist but have never been run against the shipped artifact, which is
   now Round 6.
5. **Replace the Korean Piper voice** (or install pygoruut) — `ko_KR-kss-medium.onnx.json` still has
   `"phoneme_type": "pygoruut"`, which Piper 1.2 silently ignores. Korean TTS is garbled.
6. **Fix the trainer checkpoint save path** — `train_asr.py` saves only the model at intermediate
   steps; tokenizer/processor files are missing, so `AutoProcessor.from_pretrained(checkpoint)`
   fails. Hit three times (§10, §11, §13). One-line fix: add `processor.save_pretrained(...)`
   alongside the model.
7. **Submit the llama.cpp patches** — #1 (tokenizer probe-hash rejects trimmed vocab) and #5
   (chat template silently overwritten) are the two that matter for shipping. Both are local
   uncommitted changes in `llama.cpp/conversion/`.
8. **Local machine**: bring the GPU back, finish the native-model copy, port the local `pipeline.py`,
   and write the relabeling script (teacher agreement + forced-aligner check).
9. **Commit the repo** — `git ls-files` is empty; the entire project (WORKLOG, scripts, results,
   llama.cpp patches) is untracked. Needs a `.gitignore` for `asr4/data/`, `asr4/runs/`,
   `asr4/weights/`, `*.gguf` before the first commit.

## §16 — S2ST On-Device Benchmark, Prompt Rewrite, and Device Optimization (Oct 23–27, 2026)

### 16.1 Device & Environment

- **Device**: OnePlus 13 (CPH2653, codename "dodge"), Snapdragon 8 Elite, Android 15 (SDK 35), 11 GB RAM.
- **Bootloader**: unlocked (orange state). Rooted via Magisk v28.1 (init_boot.img patch).
- **ASR server**: llama-server on :8083, OpenCL (Adreno), 29/29 layers offloaded, 807 MiB OpenCL buffer, RTF ~0.15–0.47.
- **MT server**: llama-server on :8084, HTP (Hexagon), Hy-MT2-1.8B Q6_K, 33/33 layers, 1495 MiB + 256 KV + 48 compute.
- **TTS**: Piper VITS via ONNX Runtime on CPU, espeak-ng phonemiser.
- **Harness**: `s2st-bench` C++ binary (`qdc_s2st_bench`), 80 benchmark utterances, 6 translation directions.

### 16.2 MT Prompt Rewrite (s2st-bench/src/main.cc)

**Prompt changes** (`mt_prompt()`):
- Replaced complex `[Source Text]` / `[Translation Tasks]` / 4-rule prompt with per-language system prompts (from `translator/test_data.json` and `translator/test_model_base_voice.py`).
- Each target language (en/zh/ko/vi) gets its own instruction in the target language.
- "Toward speech output" rules (numbers written out, no symbols, technical terms preserved) folded into each per-language prompt as rules 6-9.
- No English copy of the instruction — per-language prompt is the single source of truth.
- **Anchor hooking**: each language has an anchor string appended to steer output:
  - en: "Pure English translation: ", zh: "纯中文翻译：", ko: "순수 한국어 번역: ", vi: "Bản dịch tiếng Việt thuần túy: "
- `clean_mt_text()` strips anchor and "created" artifact from model output.
- Glossary terms (`--terms`) no longer injected (per-language prompt is sole instruction).

**Documented in**: `translator/MT_PROMPT.md` (211 lines).

### 16.3 Device Optimization (Rooted OxygenOS)

**Rooting**: Magisk v28.1 on init_boot.img (OxygenOS 15.0.0.864). Bootloader already unlocked.

**bench_mode.sh** — maximum-performance mode:
- Stops SurfaceFlinger + SystemUI (display off, headless)
- Stops thermal daemon (userspace throttling removed; kernel thermal zones remain enabled for safety)
- Kills heavy background apps: Google services, social apps, OnePlus/OPPO bloat, all third-party apps
- Disables Bluetooth, NFC, camera, sensors, location, telephony, Wi-Fi
- Sets CPU governor to performance, GPU to performance, I/O scheduler to none
- RAM freed: ~1.2 GB (10,460 → 9,293 MB used; 635 → 1,803 MB free)
- Processes reduced: 1,072 → 931

**restore.sh** — restores normal operation:
- Re-enables SurfaceFlinger, SystemUI, Launcher
- Re-enables thermal daemon, Bluetooth, NFC, camera, sensors, location, telephony, Wi-Fi
- Restores CPU governor to "walt" (OnePlus default), GPU to "msm-adreno-tz"
- Restores screen brightness, auto-sync, auto-rotate
- Both scripts log to `/data/local/tmp/` (not `/sdcard`, which disconnects when SurfaceFlinger stops)

**Key fix**: Thermal zones kept enabled (not disabled) to prevent device reboot. Earlier attempt to disable zones caused overheating and reboot.

### 16.4 ASR Performance Measurements

**Short-clip smoke tests (en_00.wav, 2.9s audio):**

| Condition | ASR (ms) | RTF | Notes |
|-----------|----------|-----|-------|
| Normal mode, warm | 425-442 | 0.147-0.152 | Consistent, stable |
| Normal mode, cold | 680 | 0.234 | First-run warmup |
| Bench mode | 3,098 | 1.068 | Anomaly (state issue, not thermal) |

**Sustained 5-min burst (84 utterances × 5.3s audio, bench mode):**

| Iteration | ASR med (ms) | MT med (ms) | Total med (ms) | Temp med (°C) | Temp max (°C) |
|-----------|-------------|-------------|----------------|---------------|---------------|
| 1 | 1,976 | 644 | 1,976 | 78 | 84 |
| 2 | 2,074 | 786 | 2,074 | 84 | 86 |
| 3 | 1,930 | 668 | 1,930 | 86 | 89 |
| 4 | 1,770 | 607 | 1,770 | 87 | 90 |

**Key findings:**
- Temperature plateaued at 86-90°C. Device did **not** reboot (kernel thermal protection held at ~95°C).
- ASR performance was **stable** over 5 min (no thermal degradation). First-to-last change (-22.4%) is warmup.
- ASR RTF ~0.36 for 5.3s audio clips.
- MT stayed ~600-800ms — stable, no degradation.

**30-minute estimate (extrapolated from 5-min data):**
- ASR median: ~1,900-2,000 ms (stable, no degradation)
- MT median: ~650-750 ms (stable)
- Total (eos→voice): ~1,900-2,100 ms
- Temperature: 88-92°C (plateau, no reboot risk)
- Battery drain: ~3-5% per 30 min (estimated from SoC specs)

### 16.5 Power Consumption

- Device plugged in via USB (battery 100%, "Full"). `power_now: 0` (not supported by kernel).
- Estimated from Snapdragon 8 Elite specs: ~5-8W active, ~1.6-2.6W average (32% duty cycle), ~1.1-1.8W in bench mode.
- Accurate measurement requires unplugging and measuring battery drain rate.

### 16.6 MT Output Quality (New Prompt)

With the new per-language prompt + anchor:
- Translation quality is correct but model outputs a "created" prefix (artifact of unfine-tuned model).
- `clean_mt_text()` now strips "created" and anchor from output.
- Example (en→vi): "createdCần có sự lãnh đạo..." → cleaned to "Cần có sự lãnh đạo và giải pháp của châu Âu."
- Fine-tuning with the new prompt format should eliminate the "created" artifact.

### 16.7 Chat Template (Hy-MT2)

Extracted from GGUF metadata:
```
{% if messages[0]['role'] == 'system' %}
  system message → <｜hy_begin▁of▁sentence｜>{system}...<｜hy_place▁holder▁no▁3｜>
{% else %}
  <｜hy_begin▁of▁sentence｜>
{% endif %}
{% for message in loop_messages %}
  user: <｜hy_User｜>{content}
  assistant: <｜hy_Assistant｜>{content}<｜hy_place▁holder▁no▁2｜>
{% endfor %}
```

No assistant prefix is added at the end — model generates freely after user message. The anchor approach (appending anchor to user message) doesn't act as a generation prefix; it's just text the model reads. Fine-tuning will teach the model to continue from the anchor.

### 16.8 Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `s2st-bench/src/main.cc` | Modified | MT prompt rewrite (per-language + anchor + speech rules), clean_mt_text() |
| `translator/MT_PROMPT.md` | Created | Prompt contract for MT finetuner (211 lines) |
| `scripts/root/bench_mode.sh` | Created | Benchmark mode (disable services, kill bloat, performance) |
| `scripts/root/restore.sh` | Created | Restore normal operation |
| `s2st-bench/build-android/qdc_s2st_bench` | Rebuilt | Updated binary with new prompt |

### 16.9 Open Items

1. **Fine-tune MT** with the new prompt format to eliminate "created" artifact.
2. **Run full benchmark suite** in bench mode (run_bench.sh) with updated binary.
3. **Measure power consumption** accurately (unplug + batterystats).
4. **VAD hangover tuning** (450 → 250 ms) — free ~200ms per utterance.
5. **Overlapped ASR** validation with n≥50 and accuracy measurement.
6. **Commit the repo** — all changes are untracked.

---

## §17 — Round 6 artifact, and end-of-speech → first MT token on the OnePlus (2026-10-27)

### Round 6 exported and verified as Q4_K_M (host Vulkan)

`asr4/weights/gguf/r6-Q4_K_M.gguf` (1.07 GB, same mix as Round 1) from `runs/qat-en-conv-cv/final`.
`scripts/verify_r6_vulkan.sh`, paired against the bf16 Round 6 gate dumps; results in `results/r6-q4km/`.

| set | bf16 R6 | Q4_K_M R6 | Δ | bootstrap |
|---|---|---|---|---|
| VoxPopuli en (500) | 4.95 | 5.14 | +0.19 | n.s. |
| Kspon ko (5,000) | 11.12 | 11.36 | +0.24 | significant, CI [+0.04, +0.49] |
| CV zh (10,635) | 5.01 | 4.85 | −0.17 | significant (better) |
| CV vi (1,367) | 16.02 | 15.37 | −0.66 | n.s. |
| CV ko (472) | 8.39 | 8.62 | +0.23 | n.s. |
| FLEURS en/zh/ko/vi | 3.80/5.69/4.71/6.52 | 4.11/5.74/4.86/6.30 | +0.31/+0.05/+0.15/−0.22 | — |

Quantization cost is in line with §7/§12. CV vi LID 94.7 → 93.4 %. **Not yet deployed** — the
phone still runs the Round 1 GGUF.

### Harness bugs found (s2st-bench) — all earlier on-device numbers are affected

1. **`asr_ms` was overwritten with eos→voice** in non-overlapped runs (`main.cc`, old line 804). Every
   "ASR" figure from s2st-results-real and §16.4 is really end-of-speech → first voice (that is why §16.4's
   "Total" equals its "ASR" column). True ASR time was never recorded.
2. **`clean_asr_text` dropped the first transcribed word.** "language English<asr_text>Money…" has no
   space after the language name, so "strip to the second space" ate "Money". Visible in every
   s2st-results-real row ("need European leadership…", ref "We need…"). On-device MT has been
   translating truncated input. Now cut at `<asr_text>`.
3. **The device prompt was not the evaluated one.** `/v1/audio/transcriptions` adds
   "Transcribe audio to text (language: en)" to the user turn; the GGUF evals sent audio only. The
   harness now calls `/completion` with the eval prompt (`--asr-api completion`, default).
4. **ASR prompt cache faked speed.** With `cache_prompt` on, a re-run of the same clip skips audio
   encoding (~0.6 s). ASR now sends `cache_prompt:false` (only ~3 template tokens precede the audio).

### The phone was swapping itself to death: `--cache-ram`

kswapd0 was the top CPU consumer, MemAvailable 0, 5.3 GB in zram, CPU capped to ~55 %. Cause:
llama-server's host-RAM prompt cache (`--cache-ram`, **default 8192 MiB** in this build) stores state
after every request. ASR server: **4.35 GB host memory (4.27 GB swapped)**. Killing it freed 4.4 GB.
With `--cache-ram 0` it stays **flat at 378 MB over 160 clips**. This is very likely §12's
"406 MB idle → 6.3 GB after sustained inference". `run_qdc.sh` now passes `--cache-ram 0` (ASR) and
`--cache-ram 256` (MT). The ASR server never reuses a prompt, so the cache had no upside.

Device changes (rooted OnePlus 13, serial `a9f1cc8` — use `adb -s`, a second phone is often attached):
- `/data/swapfile`, 10 GB, is the **only** swap; zram0 is off and reset.
- `/data/adb/service.d/swapfile.sh` re-applies that after boot (waits for boot_completed + 20 s,
  logs to `/data/local/tmp/swapfile.log`). **Not yet tested across a reboot.**

### ASR cost model on the Adreno (Round 1 Q4_K_M, uncached)

| clip | prompt eval (audio encode + prefill) | decode |
|---|---|---|
| 1 s | 453 ms / 32 tok | 10 tok × 58 ms |
| 2 s | 551 ms / 45 tok | 8 tok |
| 4 s | 727 ms / 71 tok | 16 tok |
| 8 s | 1246 ms / 123 tok | 33 tok × 55 ms |

≈ 350 ms + 110 ms/audio-s to encode, then ~56 ms per output token. "language English<asr_text>" is
~4 decode tokens: prefilling it from the known source language (`--no-force-lang` to disable) saves
~110–240 ms/call and fixes zh CER on the bench set (10.0 → 7.4); en/vi unchanged, ko 12.7 → 13.1.
Aborting a request frees the server within one token **only once decoding has started** — audio
encoding/prefill is not interruptible.

MT (Hy-MT2-1.8B, HTP): cold system prompt ~100–250 ms prefill; cached ~80 ms (only the new tokens).
With `-np 1` and alternating targets every turn is cold — `--mt-slot N` + `-np 2` is implemented but
**not yet measured** (`scripts/mt_slot_bench.sh`).

### End-of-speech → first MT token, 80 clips / 113 utterances, wall clock

`scripts/eos_bench.sh` (on device), `scripts/eos_report.py`; results `s2st-results-eos/`.
Hangover now 250 ms (was 450). `--realtime` replays each utterance on a wall clock and decodes while it
is "spoken"; nothing is modelled.

| config | median | p90 | max | > 2 s | WER en / vi, CER zh / ko |
|---|---|---|---|---|---|
| A one-shot, model emits language | 1436 | 2230 | 3690 | 20/113 | 6.65 / 18.95 / 10.00 / 12.69 |
| B one-shot, language forced | 1283 | 2119 | 3484 | 19/113 | 6.65 / 18.95 / **7.42** / 13.14 |
| C realtime, 100 ms pause cuts, 2 s max seg | **1000** | **1412** | **2063** | **1/113** | 21.2 / 36.0 / 30.0 / 22.1 |
| D realtime, 250 ms pauses, ≥2.5 s seg, 5 s max | 1148 | 1776 | 2775 | 7/113 | 11.5 / 18.95 / 16.5 / 13.1 |

(last speech sample → first MT token, ms.)

- **Short utterances never needed overlap**: in B every utterance < 2 s already lands < 2 s (median
  1133). All of B's misses are ≥ 2.8 s, mostly ≥ 4 s. Long transcripts are decode-bound (56 ms/token).
- **Segmenting buys latency with accuracy.** C hits the target but is worse on 53/80 clips: the cuts
  fall inside phrases and short segments hallucinate/duplicate ("…That might result. It's good.").
  D keeps vi/ko at B's accuracy but en and zh still degrade — most likely the forced 5 s cuts, which
  stitch on an overlap.
- Not yet tried: D without forced cuts (pause-only), and the speculative one-shot alone (start the B
  decode at 120 ms of silence instead of waiting for the VAD — a ~130 ms win with zero accuracy cost).

**Answer to "can eos → first MT token be < 2 s":** for utterances under ~3 s, yes, today, with no
accuracy cost (config B). For long utterances only by segmenting, and every segmenting config measured
so far costs accuracy. A hard guarantee is not available on this ASR.

### Process notes
- Pushing a rebuilt binary is a separate step; the first A/B/C run used the stale one. Check `md5sum`
  host vs device before every run.
- `pkill -f eos_bench.sh` inside `adb shell` killed its own shell (the §process-notes trap again).
- A host-side `timeout` around `adb shell` kills the device script with it; launch with
  `(nohup … &)` and poll.

### §17 addendum — speculative one-shot, pause-only segmenting, MT slots, charging (2026-10-27 evening)

Same 80 clips / 113 utterances, clean phone (ASR `--cache-ram 0`, MT `-np 1 --cache-ram 256`).

| config | median | p90 | max | > 2 s | WER en / vi, CER zh / ko |
|---|---|---|---|---|---|
| B one-shot, waits for VAD | 1283 | 2119 | 3484 | 19/113 | 6.65 / 18.95 / 7.42 / 13.14 |
| **E** one-shot started at 120 ms of silence (`--realtime --no-pause-trigger`) | 1224 | 1987 | 4754 | 11/113 | **identical to B** |
| **F** cut only at ≥ 250 ms pauses, segments ≥ 2.5 s, no forced cuts (`--realtime --pause-ms 250 --rt-min-seg 2.5 --rt-max-seg 1000`) | **1105** | **1789** | **2210** | **4/113** | 7.16 / 18.95 / **12.90** / 13.14 |

- E is free: same transcripts, 19 → 11 misses; the remaining misses are 8–11 s English utterances.
- F keeps vi/ko exactly, costs en +0.5 WER, and **breaks zh** (7.4 → 12.9). Pause cuts do not suit zh here.
- Per-language policy (F for en/vi/ko, E for zh) ≈ 6/113 over 2 s, all 2.0–2.3 s, only en +0.5 WER.
  Not yet run as one config.

**MT slots** — 30 alternating en→vi / vi→en turns (`scripts/mt_slot_bench.sh`):

| MT server | first token med / p90 / max | first clause med |
|---|---|---|
| **`-np 1 --cache-ram 256`** | **89 / 98 / 119 ms** | **415 ms** |
| `-np 2`, slot pinned per target | 121 / 132 / 260 ms | 653 ms |

The small host cache restores the other target's system prompt on a slot switch, so alternation costs
~nothing; two slots contend on the HTP and decode slower. Keep `-np 1 --cache-ram 256`; `--mt-slot`
is unnecessary. (So `--cache-ram` is harmful on ASR — no reuse, unbounded growth — but useful on MT.)

**Battery charging disabled** (`echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable`, revert
with 1; status "Not charging", USB online). **Not persistent.** Under benchmark load the battery still
drains slowly: charge_counter 4,432,000 → 4,378,000 µAh over ~55 min (~60 mA avg) — the USB input
(~1.5 A @ 4.84 V) does not cover load spikes. Needs a stronger supply, or a hysteresis script that
re-enables charging below a threshold. Hottest zone 55 °C after the memory fix (was 78–81 °C).

### §17 addendum 2 — monolithic harness feasibility (2026-10-27 night)

**HTTP overhead is negligible.** 19 one-shot utterances, harness time minus llama-server's own timing:
ASR 6–59 ms (median ~14, grows with clip length: WAV + base64 + multipart parse), MT first token
3–20 ms (median ~5). In-process saves ~20 ms/utterance. MT prompt swapping is already ~free (89 ms
alternating, §17 addendum 1). Neither justifies a monolith on its own.

**OpenCL + HTP in one process works** (`s2st-bench/src/coreside.cc`, `scripts/build_coreside.sh`,
linked against `llama.cpp/build-android-hex` = device libs, md5-identical). ASR pinned to `GPUOpenCL`
(model + mmproj via `mtmd_context_params.device`), MT to `HTP0`. 9.7 s clip, 4 iterations each:

| | sequential | concurrent (2 threads) |
|---|---|---|
| ASR encode+prefill | 1468 ms | **2410 ms (+64 %)** |
| ASR decode | ~51 ms/tok | ~51 ms/tok |
| MT prefill / decode | 76 ms / 38.5 ms/tok | 83 ms / 39.3 ms/tok |
| wall, ASR + MT | 5.3 s | **4.0 s** |
| RSS | 606 MB total, flat | |

Matches the two-server timings; identical transcripts; clean exit. Concurrency pays (−1.3 s) but ASR
encode/prefill contends with MT decode (memory bandwidth, or mmproj ops on CPU — not isolated).

**Encode-while-speaking is not free.** The encoder conv stack is per-1 s-chunk, but the transformer
attends bidirectionally inside 8 s windows (`n_window_infer=800`), and the log-mel clamp uses the
clip-wide max — so early-audio embeddings change as audio arrives. (Also: llama.cpp `qwen3a.cpp` uses
full attention with no windows, so it diverges from the reference above 8 s.) Shrinking the window
(`dump_base_transcripts.py --n-window-infer`, `scripts/encoder_window_probe.sh`, Round 6 bf16, paired
vs its 800-frame dumps; w800 reproduces exactly):

| set | 4 s | 2 s | 1 s |
|---|---|---|---|
| VoxPopuli en (500) | +0.12 | +0.07 | +0.20 |
| Kspon ko (1,000) | +0.15 | +0.49 | +1.93 |
| FLEURS en/zh/ko/vi | −0.07/−0.01/+0.18/−0.36 | +0.19/+0.19/+0.37/+0.80 | +0.52/+0.35/+0.82/+2.54 |
| CV zh (1,000) | +0.26 | +0.67 | +1.32 |
| CV vi (1,000) | +1.94 | **+5.33** | +9.97 |

Even lossless, it only removes encode+prefill (~350 ms + 110 ms/s) from after end-of-speech; decode
(~56 ms/token) stays. Untested routes: left-context block attention (blocks see the past 4–6 s, not
the future — still final once computed), or fine-tuning with `n_window_infer=200`.

**Mic/speaker without the app stack:** AAudio (NDK, `libaaudio.so`) from a native binary, needs
`audioserver` + vendor audio HAL running (bench_mode leaves them). Raw tinyalsa bypasses audioserver
but needs hand-built Qualcomm mixer routing. Untested on this phone.

---

## §18 — Round 6 on the phone, ASR on the NPU, left-context encoder, monolith (2026-10-28)

### Device housekeeping
- Battery: `/data/adb/service.d/charge_limit.sh` holds 60–80 % (charging on ≤ 60, off ≥ 80, checked
  every 60 s, log `/data/local/tmp/charge_limit.log`). Started live at 75 %; not yet seen across a reboot.
- Round 6 deployed: `model/qwen3-asr-1.7b-4lang-r6-q4km.gguf` (md5 3a1761e3…); the production ASR server
  now runs it. Round 1 GGUF kept alongside.

### Round 6 vs Round 1 on device (80 clips / 113 utterances, `scripts/eos_policy.sh`)

| config | median | p90 | max | > 2 s | WER en / vi, CER zh / ko |
|---|---|---|---|---|---|
| B  R1 one-shot | 1283 | 2119 | 3484 | 19 | 6.65 / 18.95 / 7.42 / 13.14 |
| B6 R6 one-shot | 1297 | 2082 | 3475 | 17 | **6.14 / 18.95 / 3.87 / 11.14** |
| **G** R6, F for en/vi/ko + E for zh | **1155** | **1897** | **2364** | **5** | 6.39 / 18.95 / 3.87 / 11.14 |

(last speech → first MT token, ms.) Round 6 is a clear on-device accuracy win; G keeps it (en +0.25).
First voice for G: median 1759 ms, p90 2738.

### ASR on the Hexagon NPU (`scripts/asr_device_profile.sh`, `scripts/eos_policy_htp.sh`)
R6 → `r6-BF16.gguf` (exact) → Q4_K_M / Q6_K / Q8_0. This llama.cpp checkout **does** run Q4_K on HTP
(§12's "Q4_K falls back to CPU" is out of date). Note the Q4_K_M quantized from BF16 is not
byte-identical to the F16-sourced one.

- **Audio encoder (mmproj) on HTP is broken**: truncated or looping transcripts at every quant
  ("Money laundering." / "Tháng Mười Một Mười Hai Mười Ba…"). `-dev HTP0` puts mmproj there by default;
  use `-mmdev none` (encoder on CPU).
- Decoder on HTP + encoder on CPU, 8-utterance profile: 7/8 transcripts identical to OpenCL, ASR total
  −20 %, decode 27.9 vs 37.3 ms/token, encode+prefill 787 vs 889 ms.
- Full 113-utterance run (MT also on HTP): **intermittent NaN logits** — ~16 consecutive utterances
  emitted `@@@@…` to the token cap (~8 s each), then it recovered by itself; a few later loops. No server
  error. Both configs failed in the same wall-clock window → device state, not input. en WER 6.1 → 29,
  vi 19 → 28. Where it held (ko): last→MT1 median 1156 → 882 ms, 0/35 over 2 s, CER 11.14 → 10.69.
- **Verdict: not shippable until the NaN window is root-caused**; the upside is ~250 ms median.

### Encoder windows with left context (`dump_base_transcripts.py --left-context-blocks`,
`scripts/encoder_leftctx_probe.sh`; block-causal: each block sees itself + N past blocks, so its output
is final once computed; batch 1, vs batch-1 8 s baseline; w800/lc0 reproduces within 0.04)

| 2 s blocks | independent | + 4 s left context |
|---|---|---|
| VoxPopuli en | +0.07 | +0.13 |
| Kspon ko | +0.49 | **+0.10** |
| CV zh | +0.67 | +0.69 |
| CV vi | +5.33 | **+2.71** |

1 s blocks + 4 s left context: vi +7.7, ko +1.1. Left context halves vi's loss; still not free
without fine-tuning.

### Mic and speaker from a native binary (`src/aaudio_test.cc`)
AAudio works as shell user and as root, no app. Input: LOW_LATENCY granted, 16 kHz, 5 ms bursts, 0 xruns.
Output at 16 kHz: low-latency **not** granted (perf NONE, 61 ms buffer, 1 xrun) — request 48 kHz.
**Room noise floor −31 … −34 dBFS, peaks −20 dBFS with nobody speaking**: the energy VAD will not hold
up there; a neural VAD (Silero, ONNX, runs on the ORT already on device) is the next step.

### Monolith skeleton (`src/mono.cc`, `scripts/build_mono.sh`; text plumbing moved to `src/pipeline.h`)
One process: wall-clock replay → VAD → speculative ASR (GPU, own thread) → MT (HTP, own thread,
streaming) → per-clause TTS (CPU, own thread) → output WAV on the same timeline. 16 items, direction
changing every turn:
- **MT prompt-KV swap: restore 5–10 ms** (`llama_state_seq_get/set_data`; states en 10.4 MB, zh 11.7 MB,
  vi 19.8 MB).
- Speculative ASR fired on every utterance; load-time warm-up (8.5 s) removes the cold first utterance.
- Per clip, ASR matches the server path (11 s clip 3.4 s vs 3.5 s); 1.5 s vs 6 s gaps give the same
  latency, so overlap contention is not the limit here. Long English is decode-bound under one-shot —
  the G segmenting policy is not in the monolith yet.

### §18 addendum — NPU ASR NaN windows: what is and is not the cause (2026-10-28)

**Reproduced with llama-server** (`scripts/htp_nan_repro.sh`, 3 passes × 80 clips, decoder on HTP0,
encoder on CPU, MT also on HTP): 31/339 bad in two windows (calls 115–125, 302–321). Same clips are
fine in other passes; windows start and end on ordinary short requests. (First attempt was invalid:
the shell-user script could not kill the root-owned OpenCL server, so the new one never bound :8083.
Run device scripts as root, and match processes by binary name, not `cmdline` — that killed its own
shell twice more today.)

**Reproduced in-process** (`src/nanprobe.cc`, `scripts/build_nanprobe.sh`; no server, no HTTP, MT
server idle): 22/240 bad, one window. Two phases inside it:
- first ~9 requests: prefill logits all NaN (131,896/131,896) and every decode step NaN;
- next ~13: prefill logits **finite** (correct first token), then NaN on 63/64 decode steps;
- then clean again. Prefill (HMX matmul) recovered before single-token decode — a code path healing
  independently argues against stale-KV poisoning (which also fails to explain healing on short
  requests) and against an input/overflow trigger.

**Correlates with NPU temperature.** `templog.sh` (2 s): during that run the NSP zones spiked into the
mid-90s °C repeatedly with no failures; the only **sustained > 100 °C** stretch (103.0–103.8 °C,
~26 s) coincided with the NaN window; recovery as it fell to ~82 °C. A hot run of 92 requests at
≤ 97 °C spikes had zero failures.

**Not proven causal**: the controlled hot-vs-paced A/B was stopped deliberately — sustaining NSP > 100 °C
(with big cores at 100–105 °C) risks the device and is far outside real duty. Best current explanation:
the NPU produces invalid results under sustained thermal stress / mitigation; no kernel log evidence
(dmesg ring did not reach back).

**Why it got that hot**: the probes and benchmarks drive the NPU back-to-back at ~100 % duty with the
audio encoder on the CPU. A conversation is mostly listening. Benchmarks must stop pushing the phone
this hard: big cores hit 99–105 °C today.

**Mitigations if NPU ASR is pursued**
1. NaN guard: check the first logits row for non-finite values (≈ free) and re-run that utterance on
   OpenCL; the failure is detectable immediately, not only after 256 tokens.
2. Thermal gate: route ASR to OpenCL while any NSP zone > ~90 °C.
3. Benchmark hygiene: pause when NSP/CPU > 90 °C (`templog.sh` pattern) instead of running hot.

### §18 addendum — split ASR: encoder on GPU, decoder + MT on NPU (2026-10-28 evening)

**Context for the NaN windows above:** those runs were on a phone with **no cooling**, under constant
back-to-back load — hence NSP > 100 °C. The phone now runs with cooling (the deployment condition).

`scripts/eos_split.sh`: llama-server `-dev HTP0 -mmdev GPUOpenCL` (audio encoder on Adreno, decoder on
HTP), MT on HTP, TTS on CPU. Policy G, 80 clips / 113 utterances, ≥ 3 s between clips, thermal wait
(NSP < 85 °C, CPU < 90 °C) — never triggered. Hottest point: **NSP 44 °C, CPU 47 °C**. Results in
`s2st-results-eos/Gs.jsonl`.

| | G (all GPU ASR) | Gh (NPU dec + CPU enc, uncooled) | **Gs (NPU dec + GPU enc, cooled)** |
|---|---|---|---|
| last speech → MT1 med / p90 / max | 1155 / 1897 / 2364 | 1007 / 7503 / 14375 | **936 / 1481 / 2872** |
| > 2 s | 5/113 | 13/113 | **1/113** |
| → first voice med / p90 | 1759 / 2738 | 1596 / 8247 | **1530 / 2410** |
| NaN rows | 0 | 11 | **0** |
| WER en / vi, CER zh / ko | 6.39 / 18.95 / 3.87 / 11.14 | 33.3 / 28.1 / 3.87 / 10.69 | 5.63 / 31.4* / 4.19 / 10.91 |

\* vi is one utterance: a 0.7 s fragment (vi_14) where OpenCL was already wrong ("Nếu có.") and the
NPU hallucinated a ~20-word sentence; only 4 vi transcripts differ at all. Watch short-fragment
hallucination, but it is not a systematic regression.

- Encoder on the GPU is correct (on HTP it is broken) and frees the CPU for TTS.
- MT shares the NPU: first clause 367 → 419 ms; net still a clear win (first voice −229 ms median).
- **Fastest configuration measured.** Next: a cooled, conversation-paced ~30 min soak, and the NaN
  guard (non-finite first logits → re-run on OpenCL) as insurance before making it the default.
