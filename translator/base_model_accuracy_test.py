#!/usr/bin/env python3
import torch, json, sys, unicodedata, re, argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tencent/Hy-MT2-7B")
parser.add_argument("--lang", nargs="*", default=["EN","ZH","KR","VN"])
args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading model:", args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
print("Model loaded.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

PROMPTS = data["prompts"]
ANCHORS = data["anchors"]
TEST_CASES = data["test_cases"]

TOK_START = "<" + "%s" + ">"
TOK_END = "<" + "%s" + ">"

def build_raw_prompt(text, lang):
    prompt = PROMPTS[lang]
    anchor = ANCHORS[lang]
    ml = args.model.lower()

    if "gemma" in ml:
        return (f"<|turn>system\n{prompt}<turn|>\n"
                f"<|turn>user\n{text}<turn|>\n"
                f"<|turn>model\n{anchor}")
    elif "hy-mt2" in ml or "hunyuan" in ml:
        instruction = f"Translate the following text into {lang}. Note that you should only output the translated result without any additional explanation:\n\n{text}"
        return instruction
    elif "seed-x" in ml or "seed_x" in ml:
        lang_abbr = {"EN": "en", "ZH": "zh", "KR": "ko", "VN": "vi"}.get(lang, "en")
        return f"Translate the following text into {lang}:\n\n{text} <{lang_abbr}>"
    elif "qwen" in ml:
        t0 = "im_start"
        t1 = "im_end"
        return (f"<{t0}>system\n{prompt}<{t1}>\n"
                f"<{t0}>user\n{text}<{t1}>\n"
                f"<{t0}>assistant\n{anchor}")
    elif "llama" in ml:
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{anchor}")
    else:
        return f"System: {prompt}\nUser: {text}\nAssistant: {anchor}"


def translate(text, lang):
    ml = args.model.lower()
    if "hy-mt2" in ml or "hunyuan" in ml:
        messages = [{"role": "user", "content": f"Translate the following text into {lang}. Note that you should only output the translated result without any additional explanation:\n\n{text}"}]
        enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    else:
        raw = build_raw_prompt(text, lang)
        enc = tokenizer(raw, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.6,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    gen = out[0][enc.input_ids.shape[1]:]
    decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return clean_output(decoded)


def clean_output(text):
    end_pat = r"<\|?im[_e]+_?end>?|<turn\|>|<\|?eot_id\|?>|<eos>|\|eos\|>|<\|?startoftext\|?>|<\|?extra_\d+\|?>"
    first_end = re.search(end_pat, text)
    if first_end:
        text = text[:first_end.start()]
    text = re.sub(r"<\|?im[_e]+_?start>?", "", text)
    text = re.sub(r"<\|turn>system", "", text)
    text = re.sub(r"<\|turn>user", "", text)
    text = re.sub(r"<\|turn>model", "", text)
    text = re.sub(r"<\|?begin_of_text\|?>", "", text)
    text = re.sub(r"<\|?start_header_id\|?>", "", text)
    text = re.sub(r"<\|?end_header_id\|?>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


total_cases = 0
total_kw = 0
total_purity = 0
all_results = []

for lang in args.lang:
    if lang not in TEST_CASES:
        print(f"Skipping unknown lang: {lang}")
        continue
    cases = TEST_CASES[lang]
    print(f"\n========== {lang} ({len(cases)} cases) ==========")
    for i, case in enumerate(cases):
        total_cases += 1
        output = translate(case["input"], lang)
        kw, purity, details = score_output(output, lang, case)
        total_kw += kw
        total_purity += purity
        passed = kw >= 60 and purity == 100
        partial = (kw >= 30 or purity >= 60) and not passed
        status = "PASS" if passed else "PARTIAL" if partial else "FAIL"
        inp = case["input"][:50]
        print(f"  [{status}] #{i+1} input={inp}")
        print(f"    output: {output[:150]}")
        print(f"    keyword={kw:.0f}% purity={purity:.0f}%")
        for d in details:
            print(d)
        all_results.append({
            "lang": lang, "input": case["input"], "output": output,
            "kw_score": kw, "purity": purity, "status": status
        })

overall_kw = total_kw / total_cases if total_cases else 0
overall_purity = total_purity / total_cases if total_cases else 0
overall_pass = sum(1 for r in all_results if r["status"] == "PASS")
overall_accuracy = (overall_kw * 0.5 + overall_purity * 0.3 + (overall_pass / total_cases * 100) * 0.2) if total_cases else 0

print(f"\n{'='*50}")
print(f"  OVERALL ACCURACY SCORE: {overall_accuracy:.1f}/100")
print(f"{'='*50}")
print(f"  Components: keyword={overall_kw:.1f}% purity={overall_purity:.1f}% pass_rate={overall_pass}/{total_cases} ({overall_pass/total_cases*100:.0f}%)")
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
    print(f"  {lang}: accuracy={lang_acc:.1f} kw={avg_kw:.0f}% purity={avg_p:.0f}% pass={passed}/{len(results)}")

os.makedirs(RESULTS_DIR, exist_ok=True)
model_short = args.model.split("/")[-1].replace("-", "_")
RESULTS_PATH = os.path.join(RESULTS_DIR, f"accuracy_{model_short}.json")
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")
