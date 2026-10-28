# ASR: Qwen3-ASR-1.7B for en / vi / zh / ko

Fine-tuning and evaluation of the speech recogniser that ships on the phone. How-to (commands, setup):
[`docs/guides/asr-training.md`](../docs/guides/asr-training.md). This README holds **what we found**.

## Contents

| path | what |
|---|---|
| `src/train_asr.py` | trainer: full or LoRA fine-tune, audio tower frozen, QAT, per-(language, source) eval |
| `src/q4k_fakequant.py` | Q4_K / Q6_K fake quantization (straight-through) that mirrors llama.cpp's Q4_K_M mix |
| `configs/round6-qat.json` | **the recipe of the shipped model** (Round 6), with its hypothesis and gate written in |
| `scripts/prepare_*.py` | dataset → JSONL manifest + 16 kHz WAV (FLEURS, KsponSpeech, VoxPopuli, Common Voice, Round 3) |
| `scripts/trim_vocab.py` (+ `vocab_plan.py`) | vocabulary trim to the four languages (`--mode coverage`) |
| `scripts/dump_base_transcripts.py` | greedy transcripts from a HF checkpoint (`--n-window-infer`, `--left-context-blocks` for encoder experiments) |
| `scripts/eval_gguf_server.py` | transcripts through a llama-server (GGUF), resumable |
| `scripts/score_hyps.py`, `bootstrap_ci.py`, `numwords.py` | WER/CER with paired deltas, paired bootstrap CI, numeral normalisation |

Data, weights, checkpoints and per-utterance transcripts are not in git (see `docs/GETTING_STARTED.md`).

## Shipped model: Round 6

`round6-qat.json`: trimmed vocab (152k → 131.9k tokens, 99.88 % transcript-equivalent, embeddings frozen),
full fine-tune with Q4_K_M-mirroring QAT, lr 1e-5, **one full epoch** (1,473 steps) on
FLEURS ×4 + KsponSpeech 12k + VoxPopuli 12k + Common Voice zh 12k (en 31 % / zh 32 % / ko 30 % / vi 6 %).
Exported as **Q4_K_M, 1.07 GB + 0.36 GB audio encoder**.

### Accuracy (WER en/vi, CER zh/ko; identical subsets; bf16 on host)

| set | base | Round 1 | Round 5 | **Round 6** |
|---|---|---|---|---|
| KsponSpeech conversational ko (5,000) | 13.99 | 11.05 | 11.10 | **11.12** |
| VoxPopuli en (500) | 5.27 | 6.00 | 4.97 | **4.95** |
| Common Voice zh (10,635) | 6.30 | 7.07 | 8.02 | **5.01** |
| Common Voice ko (472) | 7.41 | 8.66 | 8.40 | 8.39 |
| Common Voice vi (1,367) | **11.84** | 14.95 | 15.00 | 16.02 |
| FLEURS en / zh / ko / vi | 3.91 / 6.64 / 4.63 / 5.83 | 3.85 / 4.67 / 4.86 / 6.87 | 3.74 / 5.15 / 4.68 / 6.01 | 3.80 / 5.69 / 4.71 / 6.52 |
| language ID, FLEURS | 96.4 % | 100 % | 100 % | 100 % |

On the phone (113 utterances, Q4_K_M): Round 6 vs Round 1 → zh CER 7.42 → **3.87**, ko 13.14 → **11.14**,
en WER 6.65 → **6.14**, vi unchanged (18.95).

### Quantization cost (Round 6, bf16 → Q4_K_M, paired)

VoxPopuli +0.19 (n.s.), Kspon +0.24 (significant), CV zh −0.17 (better), CV vi −0.66 (n.s.), CV ko +0.23 (n.s.),
FLEURS +0.31 / +0.05 / +0.15 / −0.22. Earlier rounds: +0.22 (noisy audio) to +0.39 (clean read).
GGUF on CPU and on GPU agree within ±0.14.

## Verdicts: what each experiment taught us

| experiment | verdict |
|---|---|
| frequency vocab trim (`--min-count`) | ❌ changed about 10 % of base transcripts twice. Coverage trim ✅ |
| LoRA vs full fine-tune on FLEURS only (arms A/B) | ❌ both regressed Korean; the method was not the problem |
| + VietSuperSpeech | ❌ its labels come from a 30M model (25 % WER against our base); audio fine, text unusable |
| Round 1: QAT + 12k KsponSpeech | ✅ conversational ko 13.99 → 11.05, zh LID 86 → 100 %. **Data, not method** |
| Round 2: + MUSAN noise | ❌ worse than Round 1 on noisy audio; Round 1 was already robust |
| Round 4: + 12k VoxPopuli en | ½: en fixed (−0.97), ko regressed (+0.24), because only 55 % of an epoch was trained |
| Round 5: same mix, **full epoch** | ✅ ko recovered; first config to beat base on both conversational sets |
| Round 6: + Common Voice zh (testing "acoustics transfer across languages") | hypothesis ❌ (vi got worse without vi data); zh ✅ (8.02 → 5.01, beats base) |
| "Round 1 regressed English by +2.8" | 84 % was numerals spelled out in the references. Score Round 3 with `numwords.py` |
| encoder in 2 s windows (encode while speaking) | ❌ CV vi +5.3; with 4 s left context +2.7; ko +0.1. Not free without training |

**The rule these add up to:** a language improves only when **it** gets natural speech; acoustics do not transfer;
dilution is absorbed by training a full epoch; fine-tuning wins in-domain and loses out-of-domain.

## Open problems

1. **Vietnamese** is worse than base on Common Voice (+4.2) and has no spontaneous training corpus. Highest value next step.
2. Vietnamese language ID is still about 96 % on Common Voice; it should improve with item 1.
3. Intermediate `checkpoint-*` folders lack tokenizer files (copy them from the base).
4. llama.cpp conversion overwrites the ASR chat template; always send the prompt explicitly (`patches/README.md`).
