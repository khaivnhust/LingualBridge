# MT training: trimmed 1.8B student for on-device translation

Standalone guide for producing a small multilingual translation model that runs
on a Snapdragon Android device (Adreno GPU / Hexagon HTP) inside a
speech-to-speech pipeline.

Written to be followed by someone with no prior context on this project.

**Status: the base model has been chosen and measured on target hardware. What
remains is vocabulary trimming plus a training pass to recover from it.**

---

## 0. What you are building

Take `Hy-MT2-1.8B` (measured on the target device at **34.7 ms/token** on the
Hexagon NPU), **trim its vocabulary** to en/vi/zh/ko so it physically cannot emit
foreign-language tokens, then **fine-tune to recover** from the trim. Optionally
drop the knowledge we do not need.

Two components, only one of which is optional:

| component | status | rationale |
|---|---|---|
| **vocabulary trim** | **REQUIRED** | structural guarantee: the output head cannot emit tokens from the ~29 languages we do not ship. Not a quality improvement - a hard constraint. |
| **knowledge drop** | recommended | removes instruction-mode features (terminology, style, structured-data) and non-target languages from behaviour. Improves fit; not required for correctness. |

The trim **cannot** be done without a training pass afterwards - removing tokens
from a tied embedding changes the output distribution and the model must be
re-adapted. So a fine-tune happens regardless. What is optional is only how much
beyond recovery you also try to strip out.

### Measured results, target device

All serial, one request at a time, on a Snapdragon 8 Elite reference device:

| stage | backend | result |
|---|---|---|
| ASR Qwen3-ASR-1.7B Q4_K_M | Adreno / OpenCL | RTF **0.365** (FLEURS), **0.594** (Kspon conversational) |
| MT Hy-MT2-7B Q6_K | Adreno / OpenCL | 209.7 ms/token |
| MT Hy-MT2-7B Q6_K | Hexagon HTP | **130.3** ms/token, 7.72 s median |
| MT Hy-MT2-1.8B Q6_K | Hexagon HTP | **34.7** ms/token, 1.57 s median |

MT-1.8B quality on 267 parallel FLEURS sentences, 40 per direction:

| direction | BLEU | chrF | | direction | BLEU | chrF |
|---|---|---|---|---|---|---|
| en->ko | 39.61 | 27.56 | | ko->en | 26.41 | 51.03 |
| en->vi | 36.41 | 51.71 | | vi->en | 27.94 | 52.73 |
| en->zh | 42.18 | 29.48 | | zh->en | 29.70 | 53.39 |
| | | | | **mean** | **33.71** | **44.32** |

Pipeline budget with the 1.8B: ASR 4.7 s + MT 1.6 s + TTS 1.0 s = **~7.3 s for a
10 s utterance, RTF ~0.73**. Under budget with margin. ASR is now the dominant
cost.

Latency per direction on Flores was 48-55 ms/token, against a target of <= 50.

> **Note on chrF across scripts.** Korean and Chinese have no whitespace word
> boundaries, so character n-gram overlap scores paraphrase variation harshly -
> two valid renderings can score near zero. **BLEU is the comparable metric
> across our four languages; chrF is only meaningful within a script.** That is
> why en->ko shows chrF 27.56 while en->vi shows 51.71 at similar BLEU.

## 1. Requirements

- Cloud GPU, **A100 40 GB or better**. Teacher generation is the heaviest step;
  see section 3.
- 200 GB+ disk, 64 GB+ RAM
- Python 3.10+ with PyTorch, `transformers`, `datasets`, `accelerate`, `peft`
- `huggingface_hub`
- `llama.cpp` built and running, to serve the teacher

## 2. Why not a local GPU

Full fine-tuning a 3B model with Adam needs weights + gradients alone at
**12.0 GB**. A card with 11.9 GB does not fit, and CPU-offloading the optimizer
makes it worse rather than better: it moves memory from the scarce resource to
the CPU, and each step then costs 0.3-1.0 s of PCIe transfer while the GPU
sits idle. At 40 GB you should not need offload. **If you find yourself needing
it, your instance is too small** - assert on step time rather than accept it.

## 3. Choosing the teacher

The student is `Hy-MT2-1.8B`. The teacher is a separate decision, and with a
48-96 GB card teacher size is not a constraint.

### Verified available

| model | repo | size | FLORES XCOMET | % of Gemini 3.1 Pro |
|---|---|---|---|---|
| **Hy-MT2-30B-A3B** | `tencent/Hy-MT2-30B-A3B-GGUF` | **18.2 GB** Q4_K_M / 32.0 GB Q8_0 | **87.47** | **98.6%** |
| Hy-MT2-7B | `unsloth/Hy-MT2-7B-GGUF` | 4.62 GB Q4_K_M / 6.16 GB Q6_K | 86.89 | 97.9% |
| Hy-MT2-1.8B | `unsloth/Hy-MT2-1.8B-GGUF` | 1.47 GB Q6_K | 79.77 | 89.9% |
| Index-Translate-35B-A3B | `IndexTeam/Index-Translate-35B-A3B-preview-GGUF` | 21.7 GB Q4_K_M / 29.2 GB Q6_K | - | - |
| Index-Translate-9B | `IndexTeam/Index-Translate-9B-GGUF` | 5.78 GB Q4_K_M | - | - |
| Index-Translate-2B | `IndexTeam/Index-Translate-2B-GGUF` | - | - | - |
| MADLAD-400 3B MT | `cstr/madlad400-3b-mt-GGUF` | - | - | - |

### Recommended: Hy-MT2-30B-A3B

Same family as the student base. That matters more than the raw quality gap:

1. **No cross-family artefact transfer.** Published work on low-resource
   distillation documents a "sudden stop" pathology in NLLB-200-3.3B that
   *"is faithfully inherited by the student model"*, and recommends
   *"selecting a teacher of the same decoder-only family"*. Same-family
   distillation sidesteps this.
2. **Same prompt format** - already validated on device, no re-engineering.
3. **98.6% of Gemini 3.1 Pro**, on exactly the languages we ship.
4. **Fits comfortably** - 18.2 GB on a 48 GB card leaves room for batching.

**MoE note:** the 30B-A3B is ~3B *active* parameters despite 30B total. Teacher
inference is roughly 3B-class speed, not 30B. Generating a full synthetic corpus
is far cheaper than the parameter count suggests.

### The one serious alternative

Index-Translate-35B-A3B scores higher on some benchmarks, and dramatically so on
minority-language pairs:

| model | FLORES minor-pair | off-target |
|---|---|---|
| Index-Translate-35B-A3B | 0.8168 | 2.4% |
| Index-Translate-2B | 0.7377 | 4.2% |
| Hy-MT2-7B | 0.4626 | 35.7% |
| Hy-MT2-1.8B | 0.3163 | 54.2% |

That off-target rate for Hy-MT2 looks alarming, but **"minor pair" most likely
excludes vi/ko** - they are well resourced, and the 1.8B was measured producing
clean Korean with zero English leakage. Do not act on this table without
checking.

**Settle it empirically:** run both teachers over the 267 parallel sentences and
compare against our references. That is an hour of work and removes the guess.
If Index-Translate wins by a lot, note that it changes the student base too, and
the prompt must be re-validated.

### Adjudicator

Not for bulk generation - for **spot-checking ~200 sentences against one
commercial API** (Google Translate, DeepL, Microsoft). If the teacher disagrees
with the reference API on 5% of sentences you know which to trust. Cheap
insurance at small n, and it does not scale with corpus size.

MADLAD-400 3B is a useful **second opinion** when a teacher output looks wrong.

### Teacher serving and prompt

```bash
./llama-server -m Hy-MT2-30B-A3B-Q4_K_M.gguf --host 127.0.0.1 --port 8084 -ngl 99 -c 2048 -np 1
```

The chat template carries no role tokens; `<|extra_N|>` are turn delimiters, not
language tags. **There is no system prompt:**

```
Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}
```

`target_lang` is the **full English language name**: `English`, `Vietnamese`,
`Chinese`, `Korean`.

**For generating training targets use `temperature 0.0`.** The model card's 0.7 is
for interactive generation; 0.7-sampled labels bake sampling noise into every
example you train on. Keep 0.7 only when measuring the deployed configuration's
latency.

## 4. Vocabulary trimming (REQUIRED)

### Why

The requirement is structural. After trimming, the model **physically cannot
emit a token from any language outside our four**. That is a guarantee, not a
statistical tendency - it holds regardless of what the model was asked to do.

Hy-MT2-1.8B has a 120,818-token vocabulary, tied embeddings, hidden 2048:

```
body  : 32 layers x 54.5M = 1.74B params  (88%)
vocab : 120,818 x 2,048  = 0.247B params  (12%)
```

At Q6_K the tied embedding is ~200 MB of a 1.47 GB file, so **trimming saves only
about 10% of size.** The value is the correctness guarantee, not the bytes.
Depth pruning is the size lever if you ever need it.

### Sizing - measure, never guess

**Trim before training**, and size the vocabulary from **measured token usage on
the training corpus**, not from a guess. The ASR work established this rule the
hard way.

Keep:

- every subword token that appears in any of en/vi/zh/ko **in the training
  corpus**, for both source and target sides
- the instruction template and special tokens
- punctuation, digits, whitespace markers that survive normalisation

A 32k-48k target is a reasonable starting estimate; measure before committing.

### What trimming does not guarantee

Trimmed, the model still holds all **four of our** languages. Asked to translate
English to Korean it can still emit Vietnamese. Trimming removes *foreign*
tokens, not cross-language confusion among retained languages.

For a hard guarantee there, add either:

- **per-direction logit masking** at inference - restrict sampling to the target
  language's token subset. Cheap, exact, composes with the trim.
- or a **language tag** in the output head.

Given ASR measured LID at 100% on all four languages, confusion should be rare -
but "rare" is not "impossible", and the mask is cheap. **Measure before
building it.**

## 5. Knowledge drop (recommended, optional)

Beyond recovery, you can push the model toward the narrow task we actually
deploy: four languages, plain sentence translation, no instruction modes
(terminology control, style transfer, structured-data handling), no other 29
languages.

Two ways to do it:

- **Behavioural:** train on a narrow task mix until the unused modes stop
  appearing. Cheap.
- **Structural:** after measuring that instruction-mode and foreign-language
  tokens are unused, remove them in a second trim pass.

Measure before the structural pass. The first trim is the one that matters; the
second is where you can easily damage the model for no measurable gain.

## 6. Data

### 6.1 You need parallel text, not speech transcripts

If your background is ASR: a large speech corpus does **not** give you MT data.
Transcribed speech is monolingual. ~50k monolingual sentences is roughly 1-2% of
what an MT run needs.

### 6.2 Training pairs

```bash
hf download Helsinki-NLP/opus-100 --repo-type dataset \
  --include "en-vi/*" "en-zh/*" "en-ko/*" --local-dir ./data/opus100
```

Confirmed present: `en-vi/`, `en-zh/`, `en-ko/`, each with
`train/validation/test-00000-of-00001.parquet`.

**Check whether `en-zh` is Simplified or a mix before trusting zh numbers.**

### 6.3 Held-out eval - the parallel FLEURS set

267 sentence-aligned sentences across en/vi/zh/ko, derived from FLEURS test
splits by shared sentence id. Ships in the device test package as
`data/mt/fleurs_parallel.jsonl`:

```json
{"id": 1660,
 "en": "Romanticism had a large element of cultural determinism...",
 "ko": "낭만주의는 괴테나 피히테, 슐레겔과 같은 작가들로부터...",
 "vi": "Chủ nghĩa lãng mạn có một yếu tố lớn về quyết định văn hóa...",
 "zh": "浪漫主义具有很强的文化决定论色彩。..."}
```

**Never train on this.** Measured baseline to beat: mean **BLEU 33.71**,
**chrF 44.32**.

`facebook/flores` devtest (1012 sentences/language) is the standard external
benchmark, but the repo is **gated** - requires an account and accepting terms.
`Helsinki-NLP/flores-200` is likewise 401.

### 6.4 Synthetic pairs in all six directions

Because GPU cost is not the binding constraint, generate synthetic pairs
**directly in all six directions**, not English-pivot only. Translate each
monolingual sentence into the other three languages, giving a genuine
four-language corpus. This directly serves the non-pivot directions, which are
otherwise the weakest (`ko->en` BLEU 26.41 against `en->zh` 42.18).

Use a **decoder-only teacher**, not an encoder-decoder one - published work finds
decoder-only teachers outperform on this task.

Keep synthetic data separate from real pairs and cap it around 50%; synthetic
targets inherit the teacher's errors and can amplify them.

## 7. Training

### 7.1 Memory

At 40 GB a 2B fits with room: weights 4 GB + gradients 4 GB + Adam fp32 16 GB +
fp32 master 8 GB + activations ~5 GB = ~37 GB.

### 7.2 Full-parameter, not LoRA

**LoRA is contraindicated for translation.** Two independent 2026 papers:

> *"the use of LoRA is not recommended, as it yields limited performance
> improvements"*

> *"full-parameter fine-tuning and decoder-only teachers outperform LoRA and
> encoder-decoder approaches"*

This project's own ASR arms agree - LoRA (+0.70) and full FT (+0.59) both failed
the gate while QAT-against-the-deployment-surface passed. LoRA rank made no
difference across three values in the published ablation.

### 7.3 Settings

| setting | value | why |
|---|---|---|
| precision | bf16 | ~12 GB less than fp32 |
| optimizer | AdamW fp32 | 8-bit Adam if the instance is smaller |
| method | **full-parameter** | see 7.2 |
| seq len | 512 | Flores outputs run ~65 tokens |
| loss mask | target tokens only | prompt tokens to -100 |
| schedule | cosine, 3% warmup | |
| effective batch | 256-512 via grad accumulation | |

### 7.4 Two mistakes that quietly ruin distillation

1. **Train with a different prompt than you deploy with.** The teacher saw the
   section 3.3 instruction; if the student trains on bare text it learns a
   different I/O contract. Keep the template in one shared constant.
2. **Average your eval across languages.** English will look good and hide a
   broken Vietnamese. Always report per direction.

## 8. Quantize and deploy

```bash
./llama-quantize ./trimmed-student-f16.gguf ./student-Q6_K.gguf Q6_K
```

**Supported HTP weight types: Q4_0, Q4_1, Q8_0, Q5_K, Q6_K, MXFP4.**
**`Q4_K` is NOT supported** - no HTP tile size, falls back to CPU *silently*.
Check your quant before shipping.

**Ship Q6_K.** Host memory is quant-independent on the HTP - a 6.16 GB model used
only 596-609 MB RSS because weights are mapped into the NPU session rather than
held in host RAM - so there is no memory pressure to downgrade. Q4_0 is the
*weakest* 4-bit format; use IQ4_NL or Q5_K if you must go lower. At 1.47 GB the
1.8B already sits **inside** the 3.5 GB NPU mapping window.

## 9. Evaluate

1. **Parallel FLEURS, all six directions**, BLEU and chrF **per direction**.
   Baseline to beat: mean BLEU 33.71, chrF 44.32.
2. **Teacher gap** - run the teacher on the same inputs.
3. **End-to-end** - ASR output -> student MT on real recorded audio. Text-only
   eval misses domain shift from ASR errors.
4. **Device** - ms/token on the HTP, median and p95 over 50+ sentences.
   Target <= 50 ms/token; untrimmed 1.8B measured 34.7.
5. **RTF** - ASR + MT + TTS on 10 s utterances. Must be < 1.0; 1.8B projected 0.73.
6. **Illegal-token check** - confirm the trimmed model cannot emit a
   non-target-language token. This is the whole point of the trim; test it.

Steps 4-6 **must** run on the target device. Host numbers mislead: quant support
differs per backend, and a quant that runs fast on an RTX can be CPU-bound on
an NPU.

## 10. Gotchas

| symptom | cause |
|---|---|
| student fine in en, poor in vi | base weak in vi. Not fixable by more training |
| student emits bare text, ignores instruction | prompt mismatch train vs deploy |
| targets noisy or inconsistent | teacher ran at 0.7 instead of 0.0 |
| device ms/token far worse than host | quant is Q4_K, or an op fell back to CPU |
| MT log has no `HTP` line | skels not found; `ADSP_LIBRARY_PATH` must point at `lib/` |
| run collapses partway | OOM. Assert step time, watch for offload stalls |
| everything fine then the device disappears | cloud instances rotate without warning |
| chrF looks broken for ko/zh | expected - script mixing. Compare BLEU across scripts |

### On-device environment notes

- The device has **no Python** outside a chroot, and **no curl/wget** - only `nc`.
- **DNS:** the Debian rootfs ships `1.1.1.1`, which is not routable on every
  network. Symptom is `Temporary failure in name resolution` despite working
  TCP egress. `8.8.8.8` usually works. `scripts/setup.sh` now tests and fixes it.
- On-device model download runs ~17 MB/s, versus ~7 MB/s pushing over adb.
  Fetch on-device when the device has network.
- Score **on-device** with the included stdlib BLEU/chrF++ implementation and
  pull only the small metric JSONs. Verify it first: self-test asserts
  identical-to-reference scores exactly 100.00 in every direction.

## 11. Order of work

1. Download OPUS-100; confirm sentence counts
2. **Measure token usage** on the corpus to size the trim
3. Fetch the teacher (Hy-MT2-30B-A3B Q4_K_M) and generate targets at
   `temperature 0.0`, all six directions
4. Trim 120,818 -> N
5. Full-parameter fine-tune to recover
6. Optionally: knowledge-drop pass
7. Quantize Q6_K, verify tensor types and that illegal tokens are unreachable
8. Evaluate on the parallel set, per direction
9. Measure ms/token, illegal-token check, and RTF on device
10. If RTF >= 1.0, the lever is a smaller model or shorter outputs - **not**
    lower quantisation, since memory is already free on the NPU