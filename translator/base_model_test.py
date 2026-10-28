#!/bin/python3
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Cấu hình nén 4-bit để ép toàn bộ mô hình nằm trọn trong VRAM GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading model...")
model_name = "tencent/Hy-MT2-7B"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

target = sys.argv[1] if len(sys.argv) > 1 else "VN"

LANG_PROMPTS = {
    "EN": """You are a localized translation chip designed ONLY to output English.

STRICT LAWS:
1. Translate the user's text literally and accurately into English.
2. Output ONLY the English text. Never include explanations or notes.
3. NEVER mix Chinese characters (like 今天, 是, 的), Korean (한국어), Vietnamese (tiếng Việt), or any other foreign script in the output.
4. If you do not know a word, translate its meaning into English anyway. Do not keep foreign script.
5. Use natural, fluent English that sounds like a native speaker.""",

    "ZH": """你是一个专门用于输出中文的本地化翻译芯片。

严格法则：
1. 将用户的文本逐字逐句准确翻译成中文。
2. 只输出中文文本。绝不包含解释或注释。
3. 绝不在输出中混入英文、韩文（한국어）、越南语（tiếng Việt）或任何其他外文字符。
4. 如果遇到不认识的词，必须将其含义翻译成中文。不要保留外文。
5. 遇到专有名词或技术术语（如 GPU、API、Transformer）时，保留英文原文，但其余内容必须使用中文。
6. 使用自然流畅的中文，像母语者一样表达。""",

    "KR": """당신은 한국어만 출력하도록 설계된 현지화 번역 칩입니다.

엄격한 규칙:
1. 사용자의 텍스트를 정확하게 한국어로 번역합니다.
2. 한국어 텍스트만 출력합니다. 설명이나 메모를 포함하지 않습니다.
3. 출력에 중국어(예: 今天, 是, 的), 영어, 베트남어(tiếng Việt) 또는 기타 외국 문자를 절대 혼입하지 않습니다.
4. 모르는 단어가 있으면 의미를 한국어로 번역합니다. 외국 문자를 그대로 두지 않습니다.
5. 전문용어나 기술 용어(예: GPU, API, Transformer)는 영어 원어를 유지하되, 나머지는 반드시 한국어로 작성합니다.
6. 원어민이 말하는 것처럼 자연스럽고 유창한 한국어를 사용합니다.""",

    "VN": """Bạn là con chip dịch thuật được bản địa hóa CHỈ để xuất ra tiếng Việt.

NGUYÊN TẮC NGHIÊM NGẶT:
1. Dịch văn bản của người dùng một cách đúng nghĩa đen và chính xác sang tiếng Việt.
2. CHỈ xuất ra văn bản tiếng Việt. Không bao giờ giải thích hay ghi chú.
3. KHÔNG BAO GIỜ pha trộn ký tự Trung Quốc (như 今天, 是, của), tiếng Hàn (한국어), tiếng Anh, hay bất kỳ ký tự nước ngoài nào trong đầu ra.
4. Nếu không biết từ nào, phải dịch nghĩa sang tiếng Việt. Không giữ nguyên chữ nước ngoài.
5. Đối với thuật ngữ chuyên ngành hoặc kỹ thuật (ví dụ: GPU, API, Transformer), giữ nguyên tiếng Anh, nhưng phần còn lại phải viết bằng tiếng Việt.
6. Sử dụng tiếng Việt tự nhiên, trôi chảy như người bản xứ."""
}

prompt = LANG_PROMPTS.get(target, LANG_PROMPTS["VN"])

prefix_messages = [{"role": "system", "content": prompt}]
cumulative_user_speech = ""
previous_words = []

print(f"\n🤖 System ready! Translating to {target}. Type '!reset' to clear history.\n")

while True:
    try:
        user_input = input("> ").strip()
        if not user_input: continue

        if user_input.lower() == '!reset':
            cumulative_user_speech = ""
            previous_words = []
            print("✨ History cleared.\n")
            continue

        cumulative_user_speech = f"{cumulative_user_speech} {user_input}".strip() if cumulative_user_speech else user_input
        
        # --- KỸ THUẬT ASSISTANT FORCING TRỰC TIẾP TRÊN RAW PROMPT ---
        # Điền sẵn cụm mồi hệ thống để neo giữ không gian ngôn ngữ đích cho mô hình 4-bit
        forcing_anchors = {
            "EN": "Pure English translation: ",
            "ZH": "纯中文翻译：",
            "KR": "순수 한국어 번역: ",
            "VN": "Bản dịch tiếng Việt thuần túy: "
        }
        forcing_anchor = forcing_anchors.get(target, forcing_anchors["VN"])
        model_lower = model_name.lower()
        
        if "gemma-4" in model_lower or "gemma4" in model_lower:
            # Gemma 4 sử dụng cấu trúc <|turn>role ... <turn|> và role 'model' thay cho 'assistant'
            raw_prompt = (f"<|turn>system\n{prompt}<turn|>\n"
                	  f"<|turn>user\n{cumulative_user_speech}<turn|>\n"
                	  f"<|turn>model\n{forcing_anchor}")
            stop_tokens = ["<turn|>", "<|turn>", "<|endoftext|>"]
            
        elif "gemma-2" in model_lower or "gemma2" in model_lower:
            system_section = f"<start_of_turn>user\n{prompt}\n\n{cumulative_user_speech}<end_of_turn>\n"
            raw_prompt = (system_section + f"<start_of_turn>model\n{forcing_anchor}")
            stop_tokens = ["<end_of_turn>", "eos"]

        elif "hy-mt2" in model_lower or "hunyuan" in model_lower:
            messages = [{"role": "user", "content": f"Translate the following text into {target}. Note that you should only output the translated result without any additional explanation:\n\n{cumulative_user_speech}"}]
            full_encoding = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            raw_prompt = None
            stop_tokens = ["<|eos|>"]

        elif "seed-x" in model_lower or "seed_x" in model_lower:
            lang_abbr = {"EN": "en", "ZH": "zh", "KR": "ko", "VN": "vi"}.get(target, "en")
            raw_prompt = f"Translate the following text into {target}:\n\n{cumulative_user_speech} <{lang_abbr}>"
            stop_tokens = ["</s>", "eos"]

        elif "qwen" in model_lower:
            raw_prompt = (f"<|im_start|>system\n{prompt}<|im_end|>\n"
            		  f"<|im_start|>user\n{cumulative_user_speech}<|im_end|>\n"
            		  f"<|im_start|>assistant\n{forcing_anchor}")
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
            
        elif "llama-3" in model_lower or "llama3" in model_lower:
            raw_prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>"
            		  f"<|start_header_id|>user<|start_header_id|>\n\n{cumulative_user_speech}<|eot_id|>"
            		  f"<|start_header_id|>assistant<|end_header_id|>\n\n{forcing_anchor}")
            stop_tokens = ["<|eot_id|>", "<|begin_of_text|>"]
        else:
            raw_prompt = f"System: {prompt}\nUser: {cumulative_user_speech}\nAssistant: "
            stop_tokens = ["\n", "User:"]
        #raw_prompt = f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{cumulative_user_speech}<|im_end|>\n<|im_start|>assistant\n{forcing_anchor}"
        
        if raw_prompt is not None:
            full_encoding = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=full_encoding.input_ids,          
                attention_mask=full_encoding.attention_mask,     
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
            )
        
        # Cắt bỏ toàn bộ phần input bao gồm cả cụm từ mồi forcing_anchor
        input_len = full_encoding.input_ids.shape[1]
        generated_tokens = output_ids[0][input_len:]
        full_translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # --- THUẬT TOÁN WORD-LEVEL DIFF ---
        current_words = full_translation.split()
        
        match_index = 0
        if previous_words:
            for i in range(len(current_words)):
                if current_words[i:i+len(previous_words)] == previous_words:
                    match_index = i + len(previous_words)
                    break
                elif current_words[i] in previous_words[-2:] if len(previous_words) >= 2 else previous_words:
                    match_index = i + 1

        latest_words = current_words[match_index:]
        latest_translation = " ".join(latest_words)
        latest_translation = latest_translation.lstrip(",.?!;:- ")
        
        previous_words = current_words
        
        if latest_translation:
            print(f"👉 {latest_translation}")
        
    except KeyboardInterrupt:
        break

