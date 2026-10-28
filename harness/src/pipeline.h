// pipeline.h -- text plumbing shared by the benchmark harness (main.cc) and the
// monolithic pipeline (mono.cc): clause chunking for streaming TTS, ASR output
// cleaning, and the per-target-language MT prompt contract (translator/MT_PROMPT.md).
#pragma once
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ── clause chunker ───────────────────────────────────────────────────────────
// Operates on UTF-8 code points, not bytes: '，' and '。' are multi-byte, so
// `case '，':` would be a multi-character constant (and in practice silently
// dead). We decode, test the code point, and slice on byte offsets.
inline uint32_t utf8_next(const std::string & s, size_t & i) {
    if (i >= s.size()) return 0;
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { ++i; return c; }
    int n = ((c & 0xE0) == 0xC0) ? 2 : ((c & 0xF0) == 0xE0) ? 3 : ((c & 0xF8) == 0xF0) ? 4 : 1;
    uint32_t cp = (uint32_t)(c & (0xFF >> (n + 1)));
    ++i;
    for (int k = 1; k < n && i < s.size(); ++k)
        cp = (cp << 6) | ((uint32_t)(unsigned char)s[i++] & 0x3F);
    return cp;
}

inline uint32_t utf8_peek(const std::string & s, size_t i) {
    size_t j = i;
    return utf8_next(s, j);
}

inline uint32_t utf8_prev(const std::string & s, size_t i) {
    if (i == 0) return 0;
    size_t j = i - 1;
    while (j > 0 && ((unsigned char)s[j] & 0xC0) == 0x80) --j;
    return utf8_peek(s, j);
}

class ClauseChunker {
public:
    explicit ClauseChunker(int min_chars = 14, int max_chars = 160)
        : min_(min_chars), max_(max_chars) {}

    bool push(const std::string & delta, std::string * out) {
        buf_ += delta;
        size_t i = 0;
        while (i < buf_.size()) {
            const size_t start = i;
            const uint32_t cp = utf8_next(buf_, i);
            if (!term(cp)) continue;
            if (i < (size_t)min_) continue;
            if (cp == '.' || cp == ',') {
                // never split inside a number: 3.5, 1,000
                const uint32_t a = utf8_prev(buf_, start), b = utf8_peek(buf_, i);
                if (a >= '0' && a <= '9' && b >= '0' && b <= '9') continue;
            }
            *out = buf_.substr(0, i);
            buf_.erase(0, i);
            return true;
        }
        if ((int)buf_.size() >= max_) { *out = buf_; buf_.clear(); return true; }
        return false;
    }

    bool flush(std::string * out) {
        if (buf_.empty()) return false;
        *out = buf_;
        buf_.clear();
        return true;
    }

private:
    static bool term(uint32_t cp) {
        switch (cp) {
            case ',': case ';': case ':': case '.': case '!': case '?': case '\n':
                return true;
            case 0xFF0C:  // ，
            case 0x3001:  // 、
            case 0x3002:  // 。
            case 0xFF01:  // ！
            case 0xFF1F:  // ？
            case 0xFF1B:  // ；
            case 0xFF1A:  // ：
                return true;
            default:
                return false;
        }
    }
    std::string buf_;
    int min_, max_;
};

// Qwen3-ASR emits a structured sequence:
//     <|im_start|>assistant\nlanguage English<asr_text>The answer<|im_end|>
// The server's parsed-output path does not strip the structural tags, so feed
// MT the transcript only. Leaving them in degrades translation and shows up as
// literal "<asr_text>" in the synthesised speech.
inline std::string clean_asr_text(const std::string & raw) {
    std::string s = raw;
    // "language English<asr_text>..." -- cut at the tag itself. There is no
    // space between the language name and the transcript, so the old
    // "drop up to the second space" rule ate the first transcribed word.
    const size_t at = s.find("<asr_text>");
    if (at != std::string::npos) s.erase(0, at + std::strlen("<asr_text>"));
    else if (s.compare(0, 9, "language ") == 0 && s.find(' ', 9) == std::string::npos)
        s.clear();   // language header only, no transcript
    static const char * kTags[] = {"<asr_text>", "<|im_start|>", "<|im_end|>",
                                   "<|endoftext|>", "<|audio_bos|>", "<|audio_eos|>"};
    for (const char * t : kTags) {
        const size_t n = std::strlen(t);
        size_t p = 0;
        while ((p = s.find(t, p)) != std::string::npos) s.erase(p, n);
    }
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline const std::map<std::string, std::string> & lang_names() {
    static const std::map<std::string, std::string> m = {
        {"en", "English"}, {"zh", "Chinese"}, {"ko", "Korean"}, {"vi", "Vietnamese"}};
    return m;
}

// Per-language instruction (target language) + anchor hook. Mirrors the
// prompts/anchors in translator/test_data.json and translator/test_model_base_voice.py
// so the on-device MT prompt matches the translator/test/benchmark scripts.
inline const std::map<std::string, std::string> & mt_lang_prompts() {
    static const std::map<std::string, std::string> m = {
        {"en", "You are a localized translation chip designed ONLY to output English.\n"
               "\n"
               "STRICT LAWS:\n"
               "1. Translate the user's text literally and accurately into English.\n"
               "2. Output ONLY the English text. Never include explanations or notes.\n"
               "3. NEVER mix Chinese characters, Korean, Vietnamese, or any other foreign script in the output.\n"
               "4. If you do not know a word, translate its meaning into English anyway. Do not keep foreign script.\n"
               "5. Use natural, fluent English that sounds like a native speaker.\n"
               "6. Write out numbers, dates, times, and abbreviations as spoken words.\n"
               "7. Do not use symbols such as °, %, /, ~, ±, ≥. Write them out as spoken words.\n"
               "8. Keep technical terms (GPU, API, Transformer) in their original English form."},
        {"zh", "你是一个专门用于输出中文的本地化翻译芯片。\n"
               "\n"
               "严格法则：\n"
               "1. 将用户的文本逐字逐句准确翻译成中文。\n"
               "2. 只输出中文文本。绝不包含解释或注释。\n"
               "3. 绝不在输出中混入英文、韩文、越南语或任何其他外文字符。\n"
               "4. 如遇到不认识的词，必须将其含义翻译成中文。不要保留外文。\n"
               "5. 遇到专名词或技术术语（如 GPU、API、Transformer）时，保留英文原文，但其余内容必须使用中文。\n"
               "6. 使用自然流畅的中文，像母语者一样表达。\n"
               "7. 将数字、日期、时间、缩写写成口语化的词。\n"
               "8. 不要使用 °、%、/、~、±、≥ 等符号，将其写成口语化的词。\n"
               "9. 保留技术术语（GPU、API、Transformer）的英文原文。"},
        {"ko", "당신은 한국어만 출력하도록 설계된 현지화 번역 칩입니다.\n"
               "\n"
               "엄격한 규칙:\n"
               "1. 사용자의 텍스트를 정확하게 한국어로 번역합니다.\n"
               "2. 한국어 텍스트만 출력합니다. 설명이나 메모를 포함하지 않습니다.\n"
               "3. 출력에 중국어, 영어, 베트남어 또는 기타 외국 문자를 절대 혼입하지 않습니다.\n"
               "4. 모륰 단어가 있으면 의미를 한국어로 번역합니다. 외국 문자를 그대로 두지 않습니다.\n"
               "5. 전문용어나 기술 용어(예: GPU, API, Transformer)는 영어 원어를 유지하되, 나머지는 반드시 한국어로 작성합니다.\n"
               "6. 원어민이 말하는 것처럼 자연스럽고 유창한 한국어를 사용합니다.\n"
               "7. 숫자, 날짜, 시간, 약어를 구어체 단어로 풀어 씁니다.\n"
               "8. °, %, /, ~, ±, ≥ 등의 기호를 사용하지 말고 구어체 단어로 씁니다.\n"
               "9. 기술 용어(GPU, API, Transformer)는 영어 원어를 유지합니다."},
        {"vi", "Bạn là con chip dịch thuật được bản địa hóa CHỈ để xuất ra tiếng Việt.\n"
               "\n"
               "NGUYÊN TẮC NGHIÊM NGẶT:\n"
               "1. Dịch văn bản của người dùng một cách đúng nghĩa đen và chính xác sang tiếng Việt.\n"
               "2. CHỈ xuất ra văn bản tiếng Việt. Không bao giờ giải thích hay ghi chú.\n"
               "3. KHÔNG BAO GIỜ pha trộn ký tự Trung Quốc, tiếng Hàn, tiếng Anh, hay bất kỳ ký tự nước ngoài nào trong đầu ra.\n"
               "4. Nếu không biết từ nào, phải dịch nghĩa sang tiếng Việt. Không giữ nguyên chữ nước ngoài.\n"
               "5. Đối với thuật ngữ chuyên ngành hoặc kỹ thuật (ví dụ: GPU, API, Transformer), giữ nguyên tiếng Anh, nhưng phần còn lại phải viết bằng tiếng Việt.\n"
               "6. Sử dụng tiếng Việt tự nhiên, trôi chảy như người bản xứ.\n"
               "7. Viết số, ngày, giờ, từ viết tắt thành từ nói.\n"
               "8. Không dùng ký hiệu như °, %, /, ~, ±, ≥. Hãy viết thành từ nói.\n"
               "9. Giữ nguyên thuật ngữ kỹ thuật (GPU, API, Transformer) bằng tiếng Anh."},
    };
    return m;
}
inline const std::map<std::string, std::string> & mt_anchors() {
    static const std::map<std::string, std::string> m = {
        {"en", "Pure English translation: "},
        {"zh", "纯中文翻译："},
        {"ko", "순수 한국어 번역: "},
        {"vi", "Bản dịch tiếng Việt thuần túy: "}
    };
    return m;
}

inline std::string mt_prompt(const std::string & src, const std::string & tgt_code,
                             const std::vector<std::string> & terms) {
    const auto & prompts = mt_lang_prompts();
    const auto & anchors = mt_anchors();
    const std::string & sys = prompts.count(tgt_code) ? prompts.at(tgt_code) : prompts.at("en");
    const std::string & anchor = anchors.count(tgt_code) ? anchors.at(tgt_code) : anchors.at("en");
    std::ostringstream o;
    o << sys << "\n\n"
      << src << "\n\n"
      << anchor;
    return o.str();
}

// Strip the anchor hook (and any leading/trailing whitespace) from MT output.
inline std::string clean_mt_text(const std::string & raw, const std::string & tgt_code) {
    const auto & anchors = mt_anchors();
    std::string s = raw;
    const std::string & anchor = anchors.count(tgt_code) ? anchors.at(tgt_code) : anchors.at("en");
    // Remove anchor if it appears at the start (model echoed it).
    if (s.compare(0, anchor.size(), anchor) == 0)
        s.erase(0, anchor.size());
    // Defensive: strip leading "created" artifact from unfine-tuned model.
    const std::string created = "created";
    if (s.compare(0, created.size(), created) == 0)
        s.erase(0, created.size());
    // Trim
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}



