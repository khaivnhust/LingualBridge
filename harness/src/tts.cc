#include "tts.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef S2ST_WITH_ESPEAK
#include <cctype>
#include <espeak-ng/speak_lib.h>

namespace {
inline std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
// remove "(en)"-style language-switch markers, as piper does
inline void strip_lang_switches(std::string * s) {
    std::string o;
    bool in = false;
    for (char c : *s) {
        if (c == '(') { in = true; continue; }
        if (c == ')' && in) { in = false; continue; }
        if (!in) o += c;
    }
    *s = o;
}
inline int utf8_len(unsigned char c) {    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}
inline bool is_terminator_cp(uint32_t cp) {
    switch (cp) {
        case ',': case ';': case ':': case '.': case '!': case '?':
        case 0xFF0C: case 0x3001: case 0x3002: case 0xFF01: case 0xFF1F:
        case 0xFF1B: case 0xFF1A:
            return true;
        default:
            return false;
    }
}
inline uint32_t u8_next(const std::string & s, size_t & i) {
    if (i >= s.size()) return 0;
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { ++i; return c; }
    const int n = ((c & 0xE0) == 0xC0) ? 2 : ((c & 0xF0) == 0xE0) ? 3
                 : ((c & 0xF8) == 0xF0) ? 4 : 1;
    uint32_t cp = (uint32_t)(c & (0xFF >> (n + 1)));
    ++i;
    for (int k = 1; k < n && i < s.size(); ++k)
        cp = (cp << 6) | ((uint32_t)(unsigned char)s[i++] & 0x3F);
    return cp;
}
} // namespace

// Our language codes -> espeak-ng voice names. Verified against espeak 1.52:
// "zh" is not a voice (the Mandarin voice is "cmn"), and espeak dies inside
// espeak_TextToPhonemes when phonemising with an unset voice, so this mapping
// is a correctness requirement, not a nicety.
const char * espeak_voice_for(const std::string & lang) {
    if (lang == "zh") return "cmn";
    if (lang == "en") return "en-us";
    if (lang == "ko") return "ko";
    if (lang == "vi") return "vi";
    return lang.c_str();
}

// Split text into (clause-body, terminator) pairs on ASCII and CJK punctuation.
// Does not split decimals ("3.5") or thousands separators ("1,000").
std::vector<std::pair<std::string, std::string>> split_clauses(const std::string & text) {
    std::vector<std::pair<std::string, std::string>> out;
    std::string cur;
    size_t i = 0;
    while (i < text.size()) {
        const size_t start = i;
        const uint32_t cp = u8_next(text, i);
        if (is_terminator_cp(cp)) {
            const size_t clen = i - start;
            bool split = true;
            if ((cp == '.' || cp == ',') && clen == 1) {
                const bool prev_d = start > 0 && std::isdigit((unsigned char)text[start - 1]);
                const bool next_d = i < text.size() && std::isdigit((unsigned char)text[i]);
                if (prev_d && next_d) split = false;
            }
            if (split && !trim(cur).empty()) {
                out.push_back({trim(cur), text.substr(start, clen)});
                cur.clear();
            } else if (split) {
                cur.clear();
            } else {
                cur.append(text, start, clen);
            }
            continue;
        }
        cur.append(text, start, i - start);
    }
    if (!trim(cur).empty()) out.push_back({trim(cur), ""});
    return out;
}
#endif

#include <onnxruntime_cxx_api.h>

namespace s2st {
namespace {
using clk = std::chrono::steady_clock;
long long ms_since(const clk::time_point & t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
}
constexpr int kSampleRate = 22050;
} // namespace

struct Tts::Impl {
    Ort::Env             env{ORT_LOGGING_LEVEL_ERROR, "s2st"};
    Ort::SessionOptions  so;
    std::unique_ptr<Ort::Session> sess;
    Ort::MemoryInfo      mem{Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)};

    Impl() {
        so.SetIntraOpNumThreads(4);
        so.SetInterOpNumThreads(1);
        // VITS is one long conv stack; arena growth causes jitter in the
        // first-sample number, which is the metric we care about.
        so.EnableCpuMemArena();
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    }
};

Tts::Tts() : impl_(new Impl) {}
Tts::~Tts() = default;

bool Tts::espeak_available() {
#ifdef S2ST_WITH_ESPEAK
    return true;
#else
    return false;
#endif
}

void Tts::set_speaker_scale(double length_penalty, double noise_scale, double noise_w) {
    length_scale_ = length_penalty;
    noise_scale_  = noise_scale;
    noise_w_scale_ = noise_w;
}

bool Tts::load_phoneme_map(const std::string & path) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string j = ss.str();

    auto grab = [&](const std::string & key, std::vector<std::pair<std::string, int64_t>> * out) {
        size_t k = j.find("\"" + key + "\"");
        if (k == std::string::npos) return false;
        size_t ob = j.find('{', k);
        size_t oe = j.find('}', ob);
        if (ob == std::string::npos || oe == std::string::npos) return false;
        std::string body = j.substr(ob + 1, oe - ob - 1);
        size_t p = 0;
        while (p < body.size()) {
            size_t q1 = body.find('"', p);
            if (q1 == std::string::npos) break;
            size_t q2 = body.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            size_t colon = body.find(':', q2);
            if (colon == std::string::npos) break;
            size_t val = body.find_first_of("0123456789", colon);
            if (val == std::string::npos) break;
            out->push_back({body.substr(q1 + 1, q2 - q1 - 1), std::strtoll(body.c_str() + val, nullptr, 10)});
            size_t next = body.find(',', val);
            if (next == std::string::npos) break;
            p = next + 1;
        }
        return true;
    };

    pmap_.clear();
    if (!grab("phoneme_id_map", &pmap_)) return false;
    id_of_.clear();
    for (const auto & kv : pmap_) if (!kv.first.empty()) id_of_[kv.first] = kv.second;
    // Piper special symbols. Verified against every shipped voice's map.
    auto sym_id = [&](const char * s, int64_t dflt) {
        auto it = id_of_.find(s);
        return it == id_of_.end() ? dflt : it->second;
    };
    id_pad_ = sym_id("_", 0);
    id_bos_ = sym_id("^", 1);
    id_eos_ = sym_id("$", 2);
    return !pmap_.empty();
}

std::vector<int64_t> Tts::phonemize(const std::string & text, const std::string & lang,
                                    int * n_unmapped) {
    std::vector<int64_t> ids;
    if (n_unmapped) *n_unmapped = 0;
#ifdef S2ST_WITH_ESPEAK
    if (id_of_.empty() || !inited_) return ids;
    // espeak voice names are NOT the same as our language codes. Passing "zh"
    // makes espeak_SetVoiceByName fail and the next espeak_TextToPhonemes call
    // dies inside the library -- an invalid voice is fatal, not an error we can
    // report afterwards.
    const char * ev = espeak_voice_for(lang);
    if (espeak_SetVoiceByName(ev) != EE_OK) {
        if (n_unmapped) *n_unmapped = -1;      // signals "voice not found"
        return ids;
    }

    // Split into clauses FIRST, then phonemise each clause separately.
    //
    // Calling espeak_TextToPhonemes on the whole string does not work: it
    // returns phonemes for one clause but advances the input pointer one
    // character PAST the boundary -- for "The conveyor motor has failed,
    // please stop..." it consumes through the 'p' of "please", so any attempt
    // to recover the terminator by scanning backwards finds a letter instead.
    // Piper avoids this with a custom C wrapper (espeakbridge.get_phonemes).
    // Splitting ourselves removes the pointer ambiguity entirely and produces
    // the same symbols: phonemes + terminator (+ a space after , : ;).
    for (const auto & cl : split_clauses(text)) {
        const std::string & body = cl.first;
        const std::string & term = cl.second;
        if (body.empty()) continue;
        const void * tp = body.c_str();
        std::string ph;
        while (tp && *static_cast<const char *>(tp)) {
            const char * got = espeak_TextToPhonemes(&tp, espeakCHARS_UTF8,
                                                     espeakPHONEMES_IPA);
            if (!got) break;
            if (got[0]) ph += got;
        }
        std::string s = ph;
        strip_lang_switches(&s);
        s += term;
        if (term == "," || term == ":" || term == ";") s += " ";
        for (size_t k = 0; k < s.size();) {
            const int n = utf8_len((unsigned char)s[k]);
            auto it = id_of_.find(s.substr(k, n));
            if (it == id_of_.end()) { if (n_unmapped) ++*n_unmapped; }
            else ids.push_back(it->second);
            k += n;
        }
    }
#endif
    return ids;
}

std::unique_ptr<Tts> Tts::create(const std::string & onnx_path,
                                 const std::string & phoneme_map_path,
                                 int num_threads,
                                 const std::string & espeak_data_path) {
    auto t = std::unique_ptr<Tts>(new Tts());
    t->data_path_ = espeak_data_path;
    t->impl_->so.SetIntraOpNumThreads(num_threads);
    Ort::SessionOptions & so = t->impl_->so;
    t->impl_->sess = std::make_unique<Ort::Session>(t->impl_->env, onnx_path.c_str(), so);
    if (!phoneme_map_path.empty()) t->load_phoneme_map(phoneme_map_path);
#ifdef S2ST_WITH_ESPEAK
    // Must happen before any phonemise() call, and must have the data path.
    espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 50,
                      t->data_path_.empty() ? nullptr : t->data_path_.c_str(), 0);
    t->inited_ = true;
#endif
    return t;
}

TtsResult Tts::synth_ids(const std::vector<int64_t> & ids) {
    TtsResult r;
    if (!impl_->sess) { r.error = "no session"; return r; }
    if (ids.empty())   { r.error = "empty phoneme sequence"; return r; }

    auto t0 = clk::now();

    // Piper framing, verified against piper/phoneme_ids.py::phonemes_to_ids:
    //     ^  _            BOS  PAD
    //     <id> _          each phoneme followed by PAD
    //     $               EOS
    // Omitting the trailing PAD per phoneme produces plausible-looking audio
    // that is audibly wrong, so this must match exactly.
    std::vector<int64_t> seq;
    seq.reserve(ids.size() * 2 + 3);
    seq.push_back(id_bos_);
    seq.push_back(id_pad_);
    for (int64_t id : ids) {
        seq.push_back(id);
        seq.push_back(id_pad_);
    }
    seq.push_back(id_eos_);

    const int64_t B = 1;
    std::vector<int64_t> input_shape{B, (int64_t)seq.size()};
    std::vector<int64_t> len_shape{B};
    std::vector<float> scales{(float)noise_scale_, (float)length_scale_, (float)noise_w_scale_};

    std::vector<int64_t> in_buf  = seq;
    std::vector<int64_t> len_buf = {(int64_t)seq.size()};

    std::array<int64_t, 2> in_dims{B, (int64_t)seq.size()};
    std::array<int64_t, 1> len_dims{B};
    std::array<int64_t, 1> sc_dims{3};

    // Ort::MemoryInfo is move-only; reference the member rather than copying.
    Ort::MemoryInfo & mi = impl_->mem;
    std::vector<Ort::Value> in;
    try {
        in.push_back(Ort::Value::CreateTensor<int64_t>(mi, in_buf.data(), in_buf.size(),
                                                        in_dims.data(), in_dims.size()));
        in.push_back(Ort::Value::CreateTensor<int64_t>(mi, len_buf.data(), len_buf.size(),
                                                        len_dims.data(), len_dims.size()));
        in.push_back(Ort::Value::CreateTensor<float>(mi, scales.data(), scales.size(),
                                                      sc_dims.data(), sc_dims.size()));
    } catch (const Ort::Exception & e) {
        r.error = std::string("input tensor: ") + e.what();
        return r;
    }

    const char * in_names[]  = {"input", "input_lengths", "scales"};
    const char * out_names[] = {"output"};

    r.t_prep_ms = ms_since(t0);
    auto t1 = clk::now();
    std::vector<Ort::Value> out;
    try {
        out = impl_->sess->Run(Ort::RunOptions{nullptr}, in_names, in.data(), in.size(),
                               out_names, 1);
    } catch (const Ort::Exception & e) {
        r.error = std::string("Run: ") + e.what();
        return r;
    }
    r.t_first_ms = ms_since(t1);
    r.t_run_ms   = ms_since(t1);

    if (out.empty() || !out[0].IsTensor()) { r.error = "bad output"; return r; }
    auto info = out[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    size_t total = (shape.size() == 4)
                       ? (size_t)shape[0] * shape[1] * shape[2] * shape[3]
                       : (size_t)shape.back();
    const float * p = out[0].GetTensorData<float>();

    r.n_samples   = (int)total;
    r.duration_s  = total / (float)kSampleRate;
    r.pcm.assign(p, p + total);

    // first audible sample (content offset within the clip)
    float peak = 1e-6f;
    for (size_t i = 0; i < total; ++i) peak = std::max(peak, std::fabs(p[i]));
    const float thresh = std::max(1e-3f, 0.02f * peak);
    for (size_t i = 0; i < total; ++i) {
        if (std::fabs(p[i]) > thresh) { r.first_voice_ix = (int)i; break; }
    }
    r.ok = true;
    return r;
}

TtsResult Tts::synth_text(const std::string & text, const std::string & lang) {
#ifdef S2ST_WITH_ESPEAK
    if (pmap_.empty()) { TtsResult r; r.error = "no phoneme map loaded"; return r; }
    if (!inited_) { TtsResult r; r.error = "espeak-ng not initialised"; return r; }

    auto t0 = clk::now();
    int unmapped = 0;
    std::vector<int64_t> ids = phonemize(text, lang, &unmapped);
    if (ids.empty()) {
        TtsResult r;
        r.error = (unmapped < 0) ? ("espeak has no voice for '" + lang +
                                   "' (expected e.g. en-us/cmn/ko/vi)")
                                  : "phonemise produced no ids (" + std::to_string(unmapped) +
                                    " unmapped)";
        return r;
    }
    TtsResult r = synth_ids(ids);
    r.t_prep_ms = ms_since(t0);
    return r;
#else
    (void)text; (void)lang;
    TtsResult r;
    r.error = "built without espeak-ng (use IDS mode)";
    return r;
#endif
}

} // namespace s2st
