# ASR: train, evaluate, export

How the shipped ASR (Round 6) was built, and how to run the next round. Everything is under `ASR/`.
Read [`ASR/README.md`](../../ASR/README.md) first: it has the results and the verdict of every experiment so far.

## The recipe in one paragraph

Start from `Qwen/Qwen3-ASR-1.7B-hf`. **Trim the vocabulary** to en/vi/zh/ko (keeping every in-script token,
so transcripts stay 99.88 % identical to base), and keep the embeddings frozen. **Full fine-tune** the decoder (the audio tower stays frozen) with
**quantization-aware training**: the forward pass fake-quantizes weights exactly as the shipped Q4_K_M will (Q6_K on
`v_proj`/`down_proj` in the layers llama.cpp promotes, Q4_K elsewhere). Train **one full epoch** on a mix of read and
**conversational** speech, then convert to GGUF and quantize to Q4_K_M.

## 1. Vocabulary trim (once per base model)

```bash
cd ASR
python3 scripts/trim_vocab.py --mode coverage --emissions data/base_emissions_train.jsonl \
    --out weights/qwen3-asr-1.7B-4lang-trim-coverage --min-equivalent 0.99
```

Do not use frequency trimming (`--min-count`): it failed twice (about 10 % of real transcripts changed).

## 2. Train

The config is the experiment. Copy `configs/round6-qat.json` (the shipped Round 6 recipe) and change one thing.
The configs carry `_comment` / `_hypothesis` / `_gate` keys; fill them in **before** training.

```bash
export TORCH_DISABLE_NATIVE_JIT=1
tmux new -s train 'python3 -u src/train_asr.py --config configs/<yours>.json --method full > runs/<name>/train.log 2>&1'
```

- `optim: cpu_fp32_master` keeps fp32 master weights and AdamW state in **host RAM (~16–21 GB)**. Do not start
  other big jobs during the run.
- **Always run long jobs in tmux.** Three runs died silently under other harnesses.
- **Train a full epoch** (`max_steps: null`). Korean regressed at 55 % of an epoch and recovered with the full one.
- In-training eval is a tripwire (50 utterances per `(language, source)` cell, in `runs/<name>/eval.jsonl`). Decide on the post-hoc gate,
  not on this.
- Known trainer bug: intermediate `checkpoint-*` folders lack tokenizer/processor files. Copy them from the base
  before loading one with `AutoProcessor`.

Data mix lessons (WORKLOG §13–14b): a language improves only when **it** gets natural speech. Acoustics do not
transfer across languages. Dilution can be absorbed by a full epoch. The next high-value item is a **spontaneous
Vietnamese corpus**.

## 3. Evaluate (host, bf16)

```bash
# transcripts for a manifest (or several), greedy
python3 -u scripts/dump_base_transcripts.py data/cvtest/vi.jsonl --base runs/<name>/final --out results/<name>/cv.jsonl
# score against references; --base gives a paired delta against another dump
python3 scripts/score_hyps.py --hyps results/<name>/cv.jsonl --base results/round6/cv.jsonl --ref-manifest data/cvtest/vi.jsonl
# is the difference real? paired bootstrap over utterances
python3 scripts/bootstrap_ci.py --hyp-a results/round6/cv.jsonl --hyp-b results/<name>/cv.jsonl \
    --ref-manifest data/cvtest/vi.jsonl --lang vi --iters 10000
```

The gate used for every round: Common Voice test vi/zh/ko, Kspon test 5k, VoxPopuli held-out 500, FLEURS 300/lang,
and Round 3 (numerals stripped). Dump each with `dump_base_transcripts.py`, score with `score_hyps.py`, and compare
against the previous round **and** base on identical subsets. Put the outcome in `ASR/README.md`; outputs and
logs stay out of git.

Rules learned the hard way:
- **Pair on identical audio.** Use `--only-audio-from` / the same manifests; two independent samples overlapped on
  only 260 of 2,000 utterances.
- **Check the references.** Round 3 spells out numbers ("two thousand"); 84 % of one "regression" was formatting.
  Use `scripts/numwords.py` (strip mode).
- Common Voice tags Chinese `zh-CN`. The scorer expects `zh`, so rewrite the tags (see `data/cvtest/`), or rows vanish silently.

## 4. Export and verify the shippable artifact

```bash
cd ../llama.cpp      # with patches/llama.cpp-conversion.patch applied
python3 convert_hf_to_gguf.py ../ASR/runs/<name>/final --outtype bf16 --outfile ../ASR/weights/gguf/<name>-BF16.gguf
./build/bin/llama-quantize ../ASR/weights/gguf/<name>-BF16.gguf ../ASR/weights/gguf/<name>-Q4_K_M.gguf Q4_K_M
```

- The audio encoder is the stock `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`; the vocabulary trim does not touch it.
- Conversion warns `Duplicated key name 'tokenizer.chat_template'`: the stored template is wrong (bug #5).
  The pipeline sends the ASR prompt explicitly, so never rely on the GGUF's template.
- Verify the **quantized file**, not the PyTorch model: start a (Vulkan) llama-server on the GGUF + mmproj, then
  `scripts/eval_gguf_server.py --manifest <set> --only-audio-from <bf16 dump> --url http://127.0.0.1:<port>/v1/chat/completions`
  and `score_hyps.py --base <bf16 dump>` for the paired quantization cost. Expect about +0.2–0.4.
