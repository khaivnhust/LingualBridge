#!/usr/bin/env python3
"""Fine-tune Qwen3-ASR (native transformers>=5) with the audio tower frozen.

Target sequence per utterance (what the model generates at inference):

    language <Name><asr_text><transcript><|im_end|>

Only those tokens are supervised. Labels are located per row from the last
<|im_start|> (the assistant turn), so they are correct under left or right
padding.

Methods (configs/ft-1.7B-4lang.json → "method"):
    lora  LoRA adapters on every decoder linear; projector optionally trained
    full  decoder weights trained directly with 8-bit paged AdamW

Usage:
    python3 src/train_asr.py --config configs/ft-1.7B-4lang.json
    python3 src/train_asr.py --config ... --max-steps 20 --eval-per-lang 5   # smoke
    python3 src/train_asr.py --config ... --eval-only                        # baseline
"""
import argparse, json, math, os, random, re, sys, time, unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import numwords

# torch 2.14 routes bmm to Triton kernels that need Python dev headers to build
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_RATE = 16000
LANG_NAMES = {"en": "English", "zh": "Chinese", "ko": "Korean", "vi": "Vietnamese"}
NAME_TO_CODE = {v.lower(): k for k, v in LANG_NAMES.items()}


# ── Data ──────────────────────────────────────────────────────────────────────

def read_manifests(paths, max_seconds):
    rows = []
    for p in paths:
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["lang"] not in LANG_NAMES:
                    raise ValueError(f"{p}: unsupported lang {r['lang']!r}")
                if r.get("duration", 0) <= max_seconds and r["text"].strip():
                    rows.append(r)
    return rows


def load_audio(path):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


class Collator:
    """Builds chat-formatted inputs + labels for a list of manifest rows."""

    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.header = tok.encode("assistant\n", add_special_tokens=False)
        # Left padding puts every answer at the right edge, so the loss only
        # needs logits for the last few positions (see causal_lm_loss).
        tok.padding_side = "left"

    def __call__(self, rows):
        convs, audios = [], []
        for r in rows:
            audios.append(load_audio(r["audio"]))
            target = f"language {LANG_NAMES[r['lang']]}<asr_text>{r['text'].strip()}"
            convs.append([
                {"role": "user", "content": [{"type": "audio", "audio": "<audio>"}]},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ])
        texts = self.processor.apply_chat_template(convs, tokenize=False)
        batch = self.processor(text=texts, audio=audios, padding=True, return_tensors="pt")

        ids, mask = batch["input_ids"], batch["attention_mask"]
        labels = torch.full_like(ids, -100)
        h = len(self.header)
        for i in range(ids.shape[0]):
            starts = (ids[i] == self.im_start).nonzero().flatten()
            s = int(starts[-1]) + 1
            if ids[i, s:s + h].tolist() != self.header:
                raise RuntimeError("assistant header not found; chat template changed?")
            s += h
            ends = ((ids[i] == self.im_end) & (torch.arange(ids.shape[1]) >= s)).nonzero().flatten()
            e = int(ends[0]) + 1  # supervise through <|im_end|>
            labels[i, s:e] = ids[i, s:e]
        labels[mask == 0] = -100
        batch["labels"] = labels
        return batch


def causal_lm_loss(model, batch):
    """Next-token loss computing logits only over the supervised tail.

    The model's built-in loss materializes [batch, seq, vocab] logits for every
    position, audio included (8 x ~400 x 152k fp32 is several GiB). With left
    padding the labels sit in the last `keep` positions, so only those are
    projected through lm_head.
    """
    labels = batch.pop("labels")
    first = int((labels != -100).float().argmax(dim=1).min())
    keep = labels.shape[1] - first + 1  # +1: the position predicting the first label
    out = model(**batch, logits_to_keep=keep)
    logits = out.logits[:, :-1].float()
    target = labels[:, first:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
    return loss, int((target != -100).sum())


# ── Model ─────────────────────────────────────────────────────────────────────

def count(params):
    return sum(p.numel() for p in params) / 1e6


def build_model(cfg, device):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    base = cfg["base"]["model_id"]
    processor = AutoProcessor.from_pretrained(base)
    model = AutoModelForMultimodalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map=device)

    fr = cfg["freeze"]
    for p in model.parameters():
        p.requires_grad = cfg["method"] == "full"
    if fr["audio_tower"]:
        model.model.audio_tower.requires_grad_(False)
    if fr.get("embeddings", True):
        model.model.language_model.embed_tokens.requires_grad_(False)  # tied lm_head
    model.model.multi_modal_projector.requires_grad_(not fr.get("projector", False))

    if cfg["method"] == "lora":
        from peft import LoraConfig, get_peft_model

        lc = cfg["lora"]
        modules_to_save = [] if fr.get("projector", False) else ["multi_modal_projector"]
        model = get_peft_model(model, LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], modules_to_save=modules_to_save,
            bias="none"))
    elif cfg["method"] != "full":
        raise ValueError(f"unknown method {cfg['method']!r}")

    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    if cfg.get("qat", {}).get("enabled"):
        enable_qat(model, cfg)

    trainable = [p for p in model.parameters() if p.requires_grad]
    tower = unwrap(model).model.audio_tower
    print(f"params: total {count(model.parameters()):.1f}M | trainable {count(trainable):.1f}M"
          f" | audio_tower trainable {count(p for p in tower.parameters() if p.requires_grad):.1f}M")
    return model, processor


def unwrap(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def enable_qat(model, cfg):
    """Quantize decoder linear weights in the forward pass, STE on gradients.

    Rationale: the shipped artifact is Q4_K_M, so the thing we optimise must be
    the quantized model, not a bf16 model that only *becomes* Q4 at conversion
    time. Measured cost of that quantisation on conversational Korean is +0.33
    CER (results/kspon_seg2.json), so the quantized surface is a fair target.

    The patch stays installed for eval too: evaluating the bf16 weights would
    measure a model nobody deploys.

    Only Linear weights whose in_features is a multiple of 32 are patched; the
    Q4_K quantiser assumes the block size divides the row.
    """
    from q4k_fakequant import q4_k_fake_quant, q6_k_fake_quant

    q = cfg.get("qat", {})
    targets = q.get("target_modules", [])
    qscales = q.get("quantise_scales", False)
    promoted = set(q.get("q6_modules", ["v_proj", "down_proj"]))
    n_layers = q.get("n_layers", 28)
    patched, skipped = 0, 0
    by_type = {"Q4_K": 0, "Q6_K": 0}

    def uses_more_bits(i):
        # llama.cpp llama-quant.cpp: i < n/8 || i >= 7n/8 || (i - n/8)%3 == 2
        return i < n_layers // 8 or i >= 7 * n_layers // 8 \
            or (i - n_layers // 8) % 3 == 2

    for name, mod in unwrap(model).named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not mod.weight.requires_grad:
            continue
        if not any(t in name for t in targets):
            continue
        if mod.weight.shape[-1] % 32 != 0:
            skipped += 1
            continue

        leaf = name.rsplit(".", 1)[-1]
        parts = name.split(".")
        layer_i = next((int(p) for p in parts if p.isdigit()), -1)
        kind = "Q6_K" if (leaf in promoted and 0 <= layer_i < n_layers
                          and uses_more_bits(layer_i)) else "Q4_K"
        by_type[kind] += 1

        if kind == "Q6_K":
            def fwd(x, _m=mod):
                return torch.nn.functional.linear(
                    x, q6_k_fake_quant(_m.weight), _m.bias)
        elif qscales:
            def fwd(x, _m=mod):
                return torch.nn.functional.linear(
                    x, q4_k_fake_quant(_m.weight, True), _m.bias)
        else:
            def fwd(x, _m=mod):
                return torch.nn.functional.linear(
                    x, q4_k_fake_quant(_m.weight), _m.bias)
        mod.forward = fwd
        patched += 1

    if qscales:
        print("QAT: quantising scales too (scales live in bf16 at lr 1e-5)")
    print(f"QAT: patched {patched} Linear layers "
          f"(Q4_K {by_type['Q4_K']}, Q6_K {by_type['Q6_K']})"
          + (f" ({skipped} skipped, in_features%32!=0)" if skipped else ""))
    return patched


def set_train_mode(model, cfg):
    model.train()
    if cfg["freeze"]["audio_tower"]:
        unwrap(model).model.audio_tower.eval()  # no dropout in a frozen tower


class CpuFp32Master(torch.optim.Optimizer):
    """fp32 master weights + fp32 AdamW moments, held in host RAM.

    Full FT of the 1.7B decoder on a 12 GB card cannot hold an fp32 master
    copy on the GPU next to bf16 weights, bf16 grads and activations. bf16
    alone is not a substitute: one ULP at |w|~0.02 is ~7.8e-5, so an
    lr=1e-5 Adam update quantises straight back to the same value and most
    steps are silently lost.

    So: weights and grads stay on the GPU (fast forward/backward) and only the
    optimiser arithmetic moves to RAM. Costs one bf16 copy each way per
    optimiser step (~5.7 GB, ~0.23 s); 697 steps is under 3 minutes total.
    RAM needed is 12 bytes per trainable param (fp32 master + fp32 m + fp32 v)
    ~= 15.8 GiB, which fits the ~25 GB free here.

    Subclasses Optimizer because torch 2.14's LRScheduler raises
    TypeError unless isinstance(optimizer, Optimizer) -- duck-typing
    param_groups is not enough.
    """

    def __init__(self, named_params, lr, weight_decay):
        self.tracked = [(n, p) for n, p in named_params if p.requires_grad]
        super().__init__([p for _, p in self.tracked],
                         dict(lr=lr, weight_decay=weight_decay))
        self.master = {n: p.detach().float().cpu().clone() for n, p in self.tracked}
        self.opt = torch.optim.AdamW(
            list(self.master.values()), lr=lr, weight_decay=weight_decay)
        # The scheduler must read/write the real CPU optimizer's groups, not
        # the throwaway ones Optimizer.__init__ built from the GPU params.
        self.param_groups = self.opt.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        grads = [(n, p) for n, p in self.tracked if p.grad is not None]
        if not grads:
            return  # LambdaLR probes step() at construction; grads are None
        for name, p in grads:
            self.master[name].grad = p.grad.detach().float().cpu()
        self.opt.step()
        for name, p in self.tracked:
            p.data.copy_(self.master[name].to(device=p.device, dtype=p.dtype))

    def zero_grad(self, set_to_none=True):
        for _, p in self.tracked:
            p.grad = None
        for m in self.master.values():
            m.grad = None


def make_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    t = cfg["train"]
    if cfg["method"] == "full":
        mode = t.get("optim", "cpu_fp32_master")
        n = sum(p.numel() for p in params)
        if mode == "cpu_fp32_master":
            named = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]
            print(f"optim=cpu_fp32_master trainable={n / 1e6:.1f}M  "
                  f"needs ~{12 * n / 2**30:.1f} GiB host RAM for fp32 master+m+v",
                  flush=True)
            return CpuFp32Master(named, lr=t["lr"], weight_decay=t["weight_decay"])
        if mode == "paged8bit":
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(
                params, lr=t["lr"], weight_decay=t["weight_decay"])
        raise ValueError(f"unknown optim mode {mode!r}")
    return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"], fused=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def normalize(text, lang):
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " "
                   for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "") if lang in ("zh", "ko") else text


@torch.no_grad()
def evaluate(model, processor, rows, batch_size=8, max_new_tokens=256):
    import jiwer

    m = unwrap(model) if not hasattr(model, "generate") else model
    model.eval()
    per_lang = defaultdict(lambda: {"ref": [], "hyp": [], "lid_ok": 0, "escape": 0, "n": 0})
    t0, audio_s = time.time(), 0.0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        audios = [load_audio(r["audio"]) for r in chunk]
        audio_s += sum(len(a) for a in audios) / SAMPLE_RATE
        inputs = processor.apply_transcription_request(audio=audios).to(
            m.device, torch.bfloat16)
        out = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        for r, p in zip(chunk, processor.decode(gen, return_format="parsed")):
            d = per_lang[(r["lang"], r.get("source", "?"))]
            d["n"] += 1
            pred_lang = NAME_TO_CODE.get((p["language"] or "").lower())
            d["lid_ok"] += pred_lang == r["lang"]
            d["escape"] += pred_lang is None
            ref, hyp = r["text"], p["transcription"]
            if str(r.get("source", "")).startswith("round3"):
                # OneVoice writes numbers as words ("two thousand three hundred
                # and thirty") while every ASR model emits digits ("2330");
                # scored as-is that charges ~2.3 WER for formatting on identical
                # content. Strip numerals from BOTH sides -- format-neutral, and
                # unlike numwords.convert it never guesses a value ("one thousand
                # nine hundred and ninety and two thousand" is genuinely
                # ambiguous) nor mangles idioms ("at one point" is not a number).
                ref, hyp = numwords.strip(ref), numwords.strip(hyp)
            d["ref"].append(normalize(ref, r["lang"]))
            d["hyp"].append(normalize(hyp, r["lang"]))
    elapsed = time.time() - t0

    res = {}
    for (lang, src), d in sorted(per_lang.items()):
        metric = "cer" if lang in ("zh", "ko") else "wer"
        fn = jiwer.cer if metric == "cer" else jiwer.wer
        # Report per (lang, domain) so a mixed eval set stays legible and every
        # number remains comparable to a like-for-like post-hoc score.
        res[f"{lang}@{src}"] = {"n": d["n"], metric: round(fn(d["ref"], d["hyp"]) * 100, 2),
                                "lid_acc": round(100 * d["lid_ok"] / d["n"], 1),
                                "escape_rate": round(100 * d["escape"] / d["n"], 1)}
    res["_rtf"] = round(elapsed / audio_s, 4)
    return res


def eval_subset(rows, per_lang, seed):
    """Sample up to `per_lang` utterances per (lang, source) cell.

    Keyed on (lang, source) rather than lang alone: an eval set may mix domains
    (FLEURS studio read, Common Voice crowdsourced, OneVoice synthetic), and
    grouping on lang alone would blend them into one number that is comparable
    to nothing. The prompt built by processor.apply_transcription_request is
    language-agnostic, so the extra key costs nothing -- r["lang"] is still the
    true code for LID checking, metric selection and normalization.
    """
    by = defaultdict(list)
    for r in rows:
        by[(r["lang"], r.get("source", "?"))].append(r)
    rng = random.Random(seed)
    out = []
    for cell in sorted(by):
        rng.shuffle(by[cell])
        out += by[cell][:per_lang]
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fine-tune Qwen3-ASR with a frozen audio tower")
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["lora", "full"])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--eval-per-lang", type=int)
    ap.add_argument("--eval-steps", type=int)
    ap.add_argument("--run-dir")
    ap.add_argument("--train", help="comma-separated manifests (overrides config)")
    ap.add_argument("--eval", help="comma-separated manifests (overrides config)")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the base model and exit")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()

    cfg_path = a.config if os.path.isabs(a.config) else os.path.join(ROOT, a.config)
    with open(cfg_path) as f:
        cfg = json.load(f)
    t = cfg["train"]
    if a.method: cfg["method"] = a.method
    if a.max_steps: t["max_steps"] = a.max_steps
    if a.lr: t["lr"] = a.lr
    if a.batch_size: t["per_device_batch_size"] = a.batch_size
    if a.grad_accum: t["gradient_accumulation_steps"] = a.grad_accum
    if a.eval_steps: t["eval_steps"] = a.eval_steps
    if a.eval_per_lang: cfg["data"]["eval_per_lang"] = a.eval_per_lang
    if a.train: cfg["data"]["train"] = a.train.split(",")
    if a.eval: cfg["data"]["eval"] = a.eval.split(",")
    # Tolerate the HF-style spelling. configs/student-4lang.json uses it and
    # would otherwise KeyError; a CLI --batch-size already set by this point,
    # so the explicit flag still wins.
    if "per_device_batch_size" not in t and "per_device_train_batch_size" in t:
        t["per_device_batch_size"] = t["per_device_train_batch_size"]
    run_dir = a.run_dir or cfg["output"]["run_dir"]
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    random.seed(t["seed"]); np.random.seed(t["seed"]); torch.manual_seed(t["seed"])
    device = "cuda"

    eval_rows = eval_subset(read_manifests(cfg["data"]["eval"], cfg["data"]["max_audio_seconds"]),
                            cfg["data"]["eval_per_lang"], t["seed"])
    model, processor = build_model(cfg, device)

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run_dir, "tb"))

    def run_eval(step):
        res = evaluate(model, processor, eval_rows)
        print(f"[eval step {step}] {json.dumps(res, ensure_ascii=False)}", flush=True)
        for lang, v in res.items():
            if lang.startswith("_"):
                continue
            for k, x in v.items():
                if k != "n":
                    tb.add_scalar(f"eval/{lang}_{k}", x, step)
        with open(os.path.join(run_dir, "eval.jsonl"), "w") as f:
            f.write(json.dumps({"step": step, **res}, ensure_ascii=False) + "\n")
        return res

    if a.eval_only:
        run_eval(0)
        return

    train_rows = read_manifests(cfg["data"]["train"], cfg["data"]["max_audio_seconds"])
    langs = defaultdict(int)
    for r in train_rows:
        langs[r["lang"]] += 1
    print(f"train: {len(train_rows)} utterances {dict(langs)} | eval: {len(eval_rows)}")

    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    collate = Collator(processor)
    loader = DataLoader(train_rows, batch_size=t["per_device_batch_size"], shuffle=True,
                        collate_fn=collate, num_workers=4, drop_last=True,
                        persistent_workers=True)
    accum = t["gradient_accumulation_steps"]
    steps_per_epoch = len(loader) // accum
    total = t["max_steps"] or steps_per_epoch * t["epochs"]
    optim = make_optimizer(model, cfg)
    sched = get_scheduler(t["lr_scheduler_type"], optim,
                          num_warmup_steps=math.ceil(total * t["warmup_ratio"]),
                          num_training_steps=total)
    print(f"method={cfg['method']} steps={total} (epoch={steps_per_epoch}) "
          f"effective batch={t['per_device_batch_size'] * accum}")

    torch.cuda.reset_peak_memory_stats()
    step, micro, t_last, tok_count, loss_acc = 0, 0, time.time(), 0, 0.0
    set_train_mode(model, cfg)
    while step < total:
        for batch in loader:
            batch = batch.to(device)
            batch["input_features"] = batch["input_features"].to(torch.bfloat16)
            loss, n_tgt = causal_lm_loss(model, batch)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
            tok_count += n_tgt
            micro += 1
            if micro % accum:
                continue

            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], t["max_grad_norm"])
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            step += 1

            if step % t["log_steps"] == 0 or step == 1:
                dt = time.time() - t_last
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"step {step}/{total} loss {loss_acc:.4f} gnorm {float(gn):.2f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {tok_count / dt:.0f} tgt-tok/s "
                      f"peak {mem:.2f} GiB", flush=True)
                tb.add_scalar("train/loss", loss_acc, step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                tb.add_scalar("train/peak_gib", mem, step)
                t_last, tok_count = time.time(), 0
            loss_acc = 0.0

            if step % t["eval_steps"] == 0:
                run_eval(step)
                set_train_mode(model, cfg)
            if step % t["save_steps"] == 0:
                model.save_pretrained(os.path.join(run_dir, f"checkpoint-{step}"))
            if step >= total:
                break

    # Final eval. Skipped when the loop already evaluated this step -- step 600 is
    # divisible by eval_steps, so without this guard step 600 is evaluated twice,
    # costing a redundant ~20 min full pass before `final/` is written.
    if step % t["eval_steps"] != 0:
        run_eval(step)
    final = os.path.join(run_dir, "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"saved {final}")

    if cfg["method"] == "lora" and cfg["output"].get("merge_at_end") and not a.no_merge:
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        path = os.path.join(run_dir, "merged")
        merged.save_pretrained(path)
        processor.save_pretrained(path)
        print(f"saved merged model {path}")


if __name__ == "__main__":
    main()
