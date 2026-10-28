#!/usr/bin/env python3
import json, sys, unicodedata, re, argparse, os, time
from llama_cpp import Llama

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_GGUF = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Hy-MT2-7B-GGUF/snapshots/818a655121a71debd854df6ae252f94380e28d61/Hy-MT2-7B-Q6_K.gguf")

parser = argparse.ArgumentParser()
parser.add_argument("--gguf", default=DEFAULT_GGUF, help="Path to GGUF file")
parser.add_argument("--lang", nargs="*", default=["EN","ZH","KR","VN"])
parser.add_argument("--n-gpu-layers", type=int, default=0, help="Number of GPU layers (0=CPU only)")
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--top-p", type=float, default=0.6)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--repeat-penalty", type=float, default=1.05)
parser.add_argument("--chat-template", action="store_true", help="Use chat completion API (for models like Gemma 4)")
args = parser.parse_args()

print(f"Loading GGUF: {args.gguf}")
t0 = time.time()
llm = Llama(
    model_path=args.gguf,
    n_ctx=2048,
    n_gpu_layers=args.n_gpu_layers,
    n_batch=512,
    use_kvcache=False,
    verbose=False,
)
print(f"Model loaded in {time.time()-t0:.1f}s")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

PROMPTS = data["prompts"]
ANCHORS = data["anchors"]
TEST_CASES = data["test_cases"]


def build_prompt(text, lang):
    return (f"Translate the following text into {lang}. "
            f"Note that you should only output the translated result "
            f"without any additional explanation:\n\n{text}")


def clean_output(text):
    end_pat = r"<\|?im[_e]+_?end>?|<turn\|>|<\|?eot_id\|?>|<eos>|\|eos\|>|<\|?startoftext\|?>|<\|?extra_\d+\|?>"
    first_end = re.search(end_pat, text)
    if first_end:
        text = text[:first_end.start()]
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def translate(text, lang):
    prompt = build_prompt(text, lang)
    t0 = time.time()
    if args.chat_template:
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
        )
        raw = result["choices"][0]["message"]["content"].strip()
    else:
        output = llm(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
            stop=["</s>"],
        )
        raw = output["choices"][0]["text"].strip()
    elapsed = time.time() - t0
    cleaned = clean_output(raw)
    return cleaned, elapsed


def strip_diacritics(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def has_script(text, script):
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            name = unicodedata.name(ch, "")
            if script == "CJK" and ("CJK" in name or "HIRAGANA" in name or "KATAKANA" in name):
                return True
            if script == "HANGUL" and "HANGUL" in name:
                return True
    return False


def has_vietnamese(text):
    vn_chars = set("ăâđêôơư")
    for ch in text.lower():
        if ch in vn_chars:
            return True
    if strip_diacritics(text) != text:
        for ch in text:
            name = unicodedata.name(ch, "")
            if "LATIN" in name and unicodedata.category(ch).startswith("L"):
                nfkd = unicodedata.normalize("NFKD", ch)
                if len(nfkd) > 1:
                    return True
    return False


def score_output(output, lang, case):
    details = []
    output_lower = output.lower()
    output_stripped = strip_diacritics(output_lower)
    kw_found = 0
    kw_total = len(case["keywords"])
    for kw in case["keywords"]:
        kw_l = kw.lower()
        if kw_l in output_lower or kw_l in output_stripped:
            kw_found += 1
        else:
            details.append(f"  missing keyword: {kw}")
    kw_score = kw_found / kw_total * 100 if kw_total else 100

    purity = 100
    if lang == "EN":
        if has_script(output[:len(output)//2], "CJK") or has_script(output[:len(output)//2], "HANGUL"):
            purity -= 40
            details.append("  CONTAMINATION: non-English script in first half of EN output")
        elif has_script(output, "CJK") or has_script(output, "HANGUL"):
            purity -= 20
            details.append("  CONTAMINATION: non-English script found in EN output")
    elif lang == "ZH":
        if has_script(output, "HANGUL"):
            purity -= 40
            details.append("  CONTAMINATION: Korean chars found in ZH output")
        if has_vietnamese(output):
            purity -= 20
            details.append("  CONTAMINATION: Vietnamese diacritics in ZH output")
    elif lang == "KR":
        if has_script(output, "CJK"):
            cjk_count = sum(1 for ch in output if unicodedata.category(ch).startswith("L") and "CJK" in unicodedata.name(ch, ""))
            hangul_count = sum(1 for ch in output if "HANGUL" in unicodedata.name(ch, ""))
            if cjk_count > hangul_count:
                purity -= 40
                details.append("  CONTAMINATION: significant CJK (Chinese/Japanese) in KR output")
            elif has_script(output[:len(output)//3], "CJK"):
                purity -= 30
                details.append("  CONTAMINATION: CJK chars in opening portion of KR output")
    elif lang == "VN":
        if has_script(output, "CJK") or has_script(output, "HANGUL"):
            purity -= 40
            details.append("  CONTAMINATION: CJK/Hangul found in VN output")

    return kw_score, purity, details


total_cases = 0
total_kw = 0
total_purity = 0
all_results = []
total_time = 0

for lang in args.lang:
    if lang not in TEST_CASES:
        print(f"Skipping unknown lang: {lang}")
        continue
    cases = TEST_CASES[lang]
    print(f"\n========== {lang} ({len(cases)} cases) ==========")
    for i, case in enumerate(cases):
        total_cases += 1
        output, elapsed = translate(case["input"], lang)
        total_time += elapsed
        kw, purity, details = score_output(output, lang, case)
        total_kw += kw
        total_purity += purity
        passed = kw >= 60 and purity == 100
        partial = (kw >= 30 or purity >= 60) and not passed
        status = "PASS" if passed else "PARTIAL" if partial else "FAIL"
        inp = case["input"][:50]
        print(f"  [{status}] #{i+1} input={inp}")
        print(f"    output: {output[:150]}")
        print(f"    keyword={kw:.0f}% purity={purity:.0f}% time={elapsed:.2f}s")
        for d in details:
            print(d)
        all_results.append({
            "lang": lang, "input": case["input"], "output": output,
            "kw_score": kw, "purity": purity, "status": status,
            "latency_s": round(elapsed, 3)
        })

overall_kw = total_kw / total_cases if total_cases else 0
overall_purity = total_purity / total_cases if total_cases else 0
overall_pass = sum(1 for r in all_results if r["status"] == "PASS")
overall_accuracy = (overall_kw * 0.5 + overall_purity * 0.3 + (overall_pass / total_cases * 100) * 0.2) if total_cases else 0
avg_latency = total_time / total_cases if total_cases else 0

print(f"\n{'='*50}")
print(f"  OVERALL ACCURACY SCORE: {overall_accuracy:.1f}/100")
print(f"{'='*50}")
print(f"  Components: keyword={overall_kw:.1f}% purity={overall_purity:.1f}% pass_rate={overall_pass}/{total_cases} ({overall_pass/total_cases*100:.0f}%)")
print(f"  Avg latency: {avg_latency:.2f}s/case (total {total_time:.1f}s for {total_cases} cases)")
print(f"  Formula: 0.5*kw + 0.3*purity + 0.2*pass_rate")
print()

by_lang = {}
for r in all_results:
    by_lang.setdefault(r["lang"], []).append(r)
for lang, results in by_lang.items():
    avg_kw = sum(r["kw_score"] for r in results) / len(results)
    avg_p = sum(r["purity"] for r in results) / len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    lang_acc = avg_kw * 0.5 + avg_p * 0.3 + (passed / len(results) * 100) * 0.2
    lang_lat = sum(r["latency_s"] for r in results) / len(results)
    print(f"  {lang}: accuracy={lang_acc:.1f} kw={avg_kw:.0f}% purity={avg_p:.0f}% pass={passed}/{len(results)} latency={lang_lat:.2f}s")

os.makedirs(RESULTS_DIR, exist_ok=True)
model_short = "Hy_MT2_7B_Q6_K_GGUF"
RESULTS_PATH = os.path.join(RESULTS_DIR, f"accuracy_{model_short}.json")
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")
