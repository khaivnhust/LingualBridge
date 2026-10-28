# MT Prompt Template for Hy-MT2 (On-Device S2ST)

## Request Format

The MT server is a `llama-server` exposing `/v1/chat/completions`. The request body is:

```json
{
  "messages": [{"role": "user", "content": "<prompt>"}],
  "stream": true,
  "temperature": 0,
  "top_p": 0.6,
  "max_tokens": 256
}
```

There is **no system message**. The entire prompt (system-level instruction + source text + anchor) is sent as a single `user` message. The model's response is the translated text.

## Full Prompt Template

```
{per_language_system_prompt}

{source_text}

{anchor}
```

Where:
- `{per_language_system_prompt}` — target-language-specific instruction (see below). This single
  block carries EVERYTHING: the translation instruction, output constraints, speech-output rules,
  and technical-term handling. There is no separate English copy of the instruction.
- `{source_text}` — the ASR transcript (cleaned, no `language X<asr_text>` prefix)
- `{anchor}` — target-language anchor hook (see below)

> **Design note**: The per-language system prompt is the ONLY instruction. There is no English
> "Translate the following text into X" line and no separate English "speech output rules" block.
> Everything is folded into the target-language system prompt so the model receives instructions
> entirely in the target language's context.

## Per-Language System Prompts

These are the system-level instructions in the **target language**, mirroring
`translator/test_data.json` prompts and `translator/test_model_base_voice.py` `LANG_PROMPTS`.

### English (en)

```
You are a localized translation chip designed ONLY to output English.

STRICT LAWS:
1. Translate the user's text literally and accurately into English.
2. Output ONLY the English text. Never include explanations or notes.
3. NEVER mix Chinese characters, Korean, Vietnamese, or any other foreign script in the output.
4. If you do not know a word, translate its meaning into English anyway. Do not keep foreign script.
5. Use natural, fluent English that sounds like a native speaker.
6. Write out numbers, dates, times, and abbreviations as spoken words.
7. Do not use symbols such as °, %, /, ~, ±, ≥. Write them out as spoken words.
8. Keep technical terms (GPU, API, Transformer) in their original English form.
```

### Chinese (zh)

```
你是一个专门用于输出中文的本地化翻译芯片。

严格法则：
1. 将用户的文本逐字逐句准确翻译成中文。
2. 只输出中文文本。绝不包含解释或注释。
3. 绝不在输出中混入英文、韩文、越南语或任何其他外文字符。
4. 如果遇到不认识的词，必须将其含义翻译成中文。不要保留外文。
5. 遇到专有名词或技术术语（如 GPU、API、Transformer）时，保留英文原文，但其余内容必须使用中文。
6. 使用自然流畅的中文，像母语者一样表达。
7. 将数字、日期、时间、缩写写成口语化的词。
8. 不要使用 °、%、/、~、±、≥ 等符号，将其写成口语化的词。
9. 保留技术术语（GPU、API、Transformer）的英文原文。
```

### Korean (ko)

```
당신은 한국어만 출력하도록 설계된 현지화 번역 칩입니다.

엄격한 규칙：
1. 사용자의 텍스트를 정확하게 한국어로 번역합니다.
2. 한국어 텍스트만 출력합니다. 설명이나 메모를 포함하지 않습니다.
3. 출력에 중국어, 영어, 베트남어 또는 기타 외국 문자를 절대 혼입하지 않습니다.
4. 모르는 단어가 있으면 의미를 한국어로 번역합니다. 외국 문자를 그대로 두지 않습니다.
5. 전문용어나 기술 용어(예: GPU, API, Transformer)는 영어 원어를 유지하되, 나머지는 반드시 한국어로 작성합니다.
6. 원어민이 말하는 것처럼 자연스럽고 유창한 한국어를 사용합니다.
7. 숫자, 날짜, 시간, 약어를 구어체 단어로 풀어 씁니다.
8. °, %, /, ~, ±, ≥ 등의 기호를 사용하지 말고 구어체 단어로 씁니다.
9. 기술 용어(GPU, API, Transformer)는 영어 원어를 유지합니다.
```

### Vietnamese (vi)

```
Bạn là con chip dịch thuật được bản địa hóa CHỈ để xuất ra tiếng Việt.

NGUYÊN TẮC NGHIÊM NGẶT：
1. Dịch văn bản của người dùng một cách đúng nghĩa đen và chính xác sang tiếng Việt.
2. CHỈ xuất ra văn bản tiếng Việt. Không bao giờ giải thích hay ghi chú.
3. KHÔNG BAO GIỜ pha trộn ký tự Trung Quốc, tiếng Hàn, tiếng Anh, hay bất kỳ ký tự nước ngoài nào trong đầu ra.
4. Nếu không biết từ nào, phải dịch nghĩa sang tiếng Việt. Không giữ nguyên chữ nước ngoài.
5. Đối với thuật ngữ chuyên ngành hoặc kỹ thuật (ví dụ: GPU, API, Transformer), giữ nguyên tiếng Anh, nhưng phần còn lại phải viết bằng tiếng Việt.
6. Sử dụng tiếng Việt tự nhiên, trôi chảy như người bản xứ.
7. Viết số, ngày, giờ, từ viết tắt thành từ nói.
8. Không dùng ký hiệu như °, %, /, ~, ±, ≥. Hãy viết thành từ nói.
9. Giữ nguyên thuật ngữ kỹ thuật (GPU, API, Transformer) bằng tiếng Anh.
```

## Anchor Hooking (Target Language Steering)

Each target language has an **anchor string** that is appended at the very end of the prompt.
The model sees this anchor as the beginning of its expected output and continues from it,
steering generation toward the target language script.

| Target | Anchor |
|--------|--------|
| en | `Pure English translation: ` |
| zh | `纯中文翻译：` |
| ko | `순수 한국어 번역: ` |
| vi | `Bản dịch tiếng Việt thuần túy: ` |

### How it works

1. The anchor is placed at the end of the user message, after the source text.
2. The model's generation begins *after* the anchor — the anchor acts as a prefix/hook
   that biases the decoder toward producing output in the target language's script.
3. The model's response (streamed deltas) does **not** include the anchor (it's part of
   the input), so no stripping is needed in the normal case.
4. **Defensive stripping**: if the model echoes the anchor in its output, `clean_mt_text()`
   removes it before passing the text to the TTS stage.

### Why anchors

Without anchors, Hy-MT2 occasionally produces output that mixes scripts (e.g., English
words leaking into a Korean translation) or starts with an explanation. The anchor
forces the first token to be in the target language's script, dramatically reducing
script-mixing failures. This mirrors the `ANCHORS` dict in `translator/test_data.json`.

## "Toward Speech Output" Rules

> **Note**: These rules are now **folded into each per-language system prompt** (rules 6-9).
> There is no separate English copy. The per-language prompt is the single source of truth.

These rules ensure the translated text is suitable for TTS (Piper VITS):

1. **Write out numbers, dates, times, abbreviations** as spoken words
   (e.g., "3:30 PM" -> "three thirty in the afternoon", "50%" -> "fifty percent").
2. **No symbols**: degree, percent, slash, tilde, plus-minus, greater-or-equal must be
   written as spoken words.
3. **Keep technical terms** (GPU, API, Transformer) in their original English form —
   these are pronounced as-is by the TTS engine.

## Glossary Terms (Optional)

The harness accepts `--terms "a,b,c"` which injects a glossary instruction:

```
Keep these technical terms exactly as written, do not translate them: {terms}.
```

If no terms are provided, the default is `(none)`.

> **Note**: The current prompt template does NOT include the glossary instruction. The C++ harness
> accepts `--terms "a,b,c"` but the glossary is no longer injected into the prompt (the per-language
> prompt is the single instruction). If the finetuner needs to support glossary terms, add them as
> an additional line at the end of the per-language system prompt:
> ```
> Keep these technical terms exactly as written, do not translate them: {terms}.
> ```

## Finetuner Instructions

When preparing training data for MT fine-tuning:

1. **Format each example** as a single user message containing the full prompt template
   (per-language system prompt + source + anchor).
2. **The expected output** is the translation only — no anchor, no explanation.
3. **Use the anchor** as part of the training input so the model learns to continue
   from it. The training target should be the translation text *after* the anchor.
4. **Per-language**: use the correct system prompt and anchor for each target language.
   Do not mix — an English-target example must use the English system prompt and
   English anchor.
5. **Speech-output rules** should be consistently applied in training data: numbers
   written out, no symbols, technical terms preserved.
6. **Temperature 0, top_p 0.6** at inference — training should optimize for
   deterministic, high-quality translation.

## Example (en -> vi)

**Input (user message):**

```
Bạn là con chip dịch thuật được bản địa hóa CHỈ để xuất ra tiếng Việt.

NGUYÊN TẮC NGHIÊM NGẶT：
1. Dịch văn bản của người dùng một cách đúng nghĩa đen và chính xác sang tiếng Việt.
2. CHỈ xuất ra văn bản tiếng Việt. Không bao giờ giải thích hay ghi chú.
3. KHÔNG BAO GIỜ pha trộn ký tự Trung Quốc, tiếng Hàn, tiếng Anh, hay bất kỳ ký tự nước ngoài nào trong đầu ra.
4. Nếu không biết từ nào, phải dịch nghĩa sang tiếng Việt. Không giữ nguyên chữ nước ngoài.
5. Đối với thuật ngữ chuyên ngành hoặc kỹ thuật (ví dụ: GPU, API, Transformer), giữ nguyên tiếng Anh, nhưng phần còn lại phải viết bằng tiếng Việt.
6. Sử dụng tiếng Việt tự nhiên, trôi chảy như người bản xứ.
7. Viết số, ngày, giờ, từ viết tắt thành từ nói.
8. Không dùng ký hiệu như °, %, /, ~, ±, ≥. Hãy viết thành từ nói.
9. Giữ nguyên thuật ngữ kỹ thuật (GPU, API, Transformer) bằng tiếng Anh.

We need European leadership and solutions right now.

Bản dịch tiếng Việt thuần túy:
```

**Expected output:**

```
Chúng ta cần sự lãnh đạo và giải pháp của châu Âu ngay bây giờ.
```
