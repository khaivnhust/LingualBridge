// qdc_s2st_bench.cc -- round-trip RTF and end-of-speech -> begin-of-voice latency
// for the LingualBridge speech-to-speech pipeline, targeted at Snapdragon 8 Elite.
//
// Topology under test (fixed by the deployment target, not by convenience):
//     ASR : llama.cpp, Qwen3-ASR-1.7B Q4_K_M, Adreno GPU via OpenCL   [blocking]
//     MT  : llama.cpp, Hy-MT2-1.8B Q6_K,     Hexagon HTP (DSP)         [streaming]
//     TTS : Piper VITS via ONNX Runtime, CPU                           [per clause]
//
// The number that matters for a live conversation is not round-trip RTF, it is
// how long after the user stops talking the first audible word comes out. With
// MT at ~34.7 ms/token, waiting for a complete sentence before synthesising
// throws away most of the available budget -- so MT is streamed and the first
// speakable clause is handed to TTS *while the rest of the sentence is still
// decoding*. TTS runs on its own thread so synthesising clause 1 does not stall
// the MT stream that is still producing clause 2.
//
// Both llama.cpp stages are driven as local HTTP servers (the pattern already
// proven on QDC in asr4/scripts/bench_mt.py), which keeps each backend isolated
// in its own process: OpenCL and HTP in one process is not something we have
// measured, and localhost HTTP costs ~1 ms against a multi-second budget.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "http.h"
#include "stitch.h"
#include "tts.h"
#include "vad.h"

using namespace s2st;
using clk = std::chrono::steady_clock;

static long long ms_since(const clk::time_point & t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
}
static clk::time_point g_epoch;   // wall anchor for all timestamps

static double rel_ms(const clk::time_point & t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t - g_epoch).count() / 1000.0;
}

// ── WAV I/O ──────────────────────────────────────────────────────────────────
struct Wav {
    int                rate = 0;
    std::vector<float> pcm;
};

static bool wav_read(const std::string & path, Wav * out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (all.size() < 44 || all.compare(0, 4, "RIFF") != 0) return false;
    auto rd32 = [&](size_t a) {
        return (uint32_t)(uint8_t)all[a] | ((uint32_t)(uint8_t)all[a + 1] << 8) |
               ((uint32_t)(uint8_t)all[a + 2] << 16) | ((uint32_t)(uint8_t)all[a + 3] << 24);
    };
    auto rd16 = [&](size_t a) { return (uint16_t)((uint8_t)all[a] | ((uint8_t)all[a + 1] << 8)); };

    size_t pos = 12;
    int channels = 1, bits = 16;
    while (pos + 8 <= all.size()) {
        std::string id = all.substr(pos, 4);
        uint32_t csz = rd32(pos + 4);
        size_t   d   = pos + 8;
        if (id == "fmt ") {
            channels = rd16(d + 2);
            out->rate = (int)rd32(d + 4);
            bits = rd16(d + 14);
        } else if (id == "data") {
            size_t n = std::min((size_t)csz, all.size() - d);
            if (bits == 16) {
                out->pcm.resize(n / 2);
                for (size_t i = 0; i < out->pcm.size(); ++i)
                    out->pcm[i] = (float)(int16_t)rd16(d + i * 2) / 32768.0f;
            } else if (bits == 32 && channels == 1) {
                out->pcm.resize(n / 4);
                for (size_t i = 0; i < out->pcm.size(); ++i) {
                    float v;
                    std::memcpy(&v, all.data() + d + i * 4, 4);
                    out->pcm[i] = v;
                }
            } else {
                return false;
            }
            break;
        }
        pos = d + csz + (csz & 1);
    }
    (void)channels;
    return out->rate > 0 && !out->pcm.empty();
}

static void w32(std::string & o, uint32_t v) {
    o += (char)(v & 0xff); o += (char)((v >> 8) & 0xff);
    o += (char)((v >> 16) & 0xff); o += (char)((v >> 24) & 0xff);
}
static void w16(std::string & o, uint16_t v) {
    o += (char)(v & 0xff); o += (char)((v >> 8) & 0xff);
}

static std::string wav_encode_16k(const std::vector<float> & pcm, int rate = 16000) {
    std::string o;
    const uint32_t db = (uint32_t)(pcm.size() * 2);
    o += "RIFF"; w32(o, 36 + db); o += "WAVE";
    o += "fmt "; w32(o, 16); w16(o, 1); w16(o, 1); w32(o, rate);
    w32(o, (uint32_t)(rate * 2)); w16(o, 2); w16(o, 16);
    o += "data"; w32(o, db);
    for (float s : pcm) {
        float c = std::max(-1.f, std::min(1.f, s));
        w16(o, (uint16_t)(int16_t)(c * 32767.f));
    }
    return o;
}

static std::string base64(const std::string & in) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t v = ((uint8_t)in[i] << 16) | ((uint8_t)in[i + 1] << 8) | (uint8_t)in[i + 2];
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += T[v & 63];
    }
    if (i + 1 == in.size()) {
        uint32_t v = (uint8_t)in[i] << 16;
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += "==";
    } else if (i + 2 == in.size()) {
        uint32_t v = ((uint8_t)in[i] << 16) | ((uint8_t)in[i + 1] << 8);
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += '=';
    }
    return o;
}

static bool wav_write(const std::string & path, const std::vector<float> & pcm, int rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t db = (uint32_t)(pcm.size() * 2);
    auto p32 = [&](uint32_t v) { f.write((char *)&v, 4); };
    auto p16 = [&](uint16_t v) { f.write((char *)&v, 2); };
    f.write("RIFF", 4); p32(36 + db); f.write("WAVE", 4);
    f.write("fmt ", 4); p32(16); p16(1); p16(1); p32((uint32_t)rate);
    p32((uint32_t)(rate * 2)); p16(2); p16(16);
    f.write("data", 4); p32(db);
    for (float s : pcm) {
        float c = std::max(-1.f, std::min(1.f, s));
        int16_t v = (int16_t)(c * 32767.f);
        f.write((char *)&v, 2);
    }
    return true;
}

// ── TTS worker: runs clauses on its own thread so MT streaming is not stalled ─
class TtsRunner {
public:
    TtsRunner(Tts * tts, const std::map<std::string, std::vector<int64_t>> * ids, std::string lang)
        : tts_(tts), ids_(ids), lang_(std::move(lang)) {
        if (tts_) {
            worker_ = std::thread([this] {
                for (;;) {
                    std::string c;
                    {
                        std::unique_lock<std::mutex> lk(m_);
                        cv_.wait(lk, [this] { return !q_.empty() || stop_; });
                        if (q_.empty() && stop_) return;
                        c = q_.front();
                        q_.pop_front();
                        ++inflight_;
                    }
                    auto t_run0 = clk::now();
                    TtsResult r = synth(c);
                    {
                        std::lock_guard<std::mutex> lk2(m_);
                        --inflight_;
                        cv_.notify_all();
                    }
                    if (!r.ok) continue;
                    // VITS returns the whole clip in one call: the audio becomes
                    // playable the instant Run() returns. That instant is the
                    // latency; first_voice_ix is a content offset only.
                    double audible_at = rel_ms(clk::now());
                    ++done_;
                    if (first_voice_ms_ < 0) { first_voice_ms_ = audible_at; tts_first_ms_ = r.t_run_ms + r.t_prep_ms; }
                    if (first_voice_offset_ms_ < 0) first_voice_offset_ms_ = r.first_voice_ms();
                    tts_prep_ms_ += std::max(0LL, r.t_prep_ms);
                    tts_run_ms_  += std::max(0LL, r.t_run_ms);
                    audio_.insert(audio_.end(), r.pcm.begin(), r.pcm.end());
                }
            });
        }
    }

    ~TtsRunner() {
        if (worker_.joinable()) {
            { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
            cv_.notify_all();
            worker_.join();
        }
    }

    void submit(const std::string & clause) {
        if (!tts_) return;
        { std::lock_guard<std::mutex> lk(m_); q_.push_back(clause); }
        cv_.notify_one();
    }

    void drain() {
        if (!worker_.joinable()) return;
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return q_.empty() && inflight_ == 0; });
    }

    bool available() const { return tts_ != nullptr; }
    long long first_voice_ms() const {
        return first_voice_ms_ < 0 ? -1 : (long long)llround(first_voice_ms_);
    }
    float first_voice_offset_ms() const {
        return first_voice_offset_ms_ < 0 ? 0.f : (float)first_voice_offset_ms_;
    }
    long long tts_prep_ms() const { return tts_prep_ms_; }
    // duration of the FIRST synthesised clause only -- the summed
    // tts_run_ms covers every clause, and later ones do not gate voice onset
    long long tts_first_chunk_ms() const { return tts_first_ms_ < 0 ? 0 : tts_first_ms_; }
    long long tts_first_ms_ = -1;   // duration of the FIRST clause only
    long long tts_run_ms() const { return tts_run_ms_; }
    int         done() const { return done_; }
    const std::vector<float> & audio() const { return audio_; }
    std::string last_error;

private:
    TtsResult synth(const std::string & clause) {
        if (Tts::espeak_available()) {
            TtsResult r = tts_->synth_text(clause, lang_);
            if (!r.ok && last_error.empty()) last_error = r.error;
            return r;
        }
        if (ids_) {
            auto it = ids_->find(clause);
            if (it == ids_->end()) {
                TtsResult r;
                r.error = "clause absent from --tts-ids table";
                if (last_error.empty()) last_error = r.error;
                return r;
            }
            return tts_->synth_ids(it->second);
        }
        TtsResult r;
        r.error = "no espeak-ng on device and no --tts-ids table";
        if (last_error.empty()) last_error = r.error;
        return r;
    }

    Tts * tts_ = nullptr;
    const std::map<std::string, std::vector<int64_t>> * ids_ = nullptr;
    std::string lang_;
    std::thread worker_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    bool stop_ = false;
    int  inflight_ = 0;
    std::vector<float> audio_;
    int  done_ = 0;
    long long tts_prep_ms_ = 0, tts_run_ms_ = 0;
    double first_voice_ms_ = -1, first_voice_offset_ms_ = -1;
};

#include "pipeline.h"

// clause -> phoneme ids, for runs without espeak-ng on device
static bool load_ids_table(const std::string & path,
                           std::map<std::string, std::vector<int64_t>> * out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string text = line.substr(0, tab);
        std::string ids  = line.substr(tab + 1);
        std::vector<int64_t> v;
        std::stringstream ss(ids);
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoll(tok));
        (*out)[text] = v;
    }
    return !out->empty();
}

struct Result {
    int    utt = 0;
    double audio_s = 0, t_start_ms = 0, t_last_ms = 0, t_detect_ms = 0, vad_penalty_ms = 0;
    long long asr_ms = 0, mt_first_token_ms = -1, mt_first_clause_ms = -1;
    long long eos_to_voice_ms = -1, eos_to_voice_from_last_ms = -1, total_ms = 0;
    long long tts_prep_ms = 0, tts_run_ms = 0;
    float    tts_lead_offset_ms = 0;
    double rtf = 0;
    int    n_chunks = 0, tts_chunks = 0, n_predecode = 0, model_ms = 0, stitch_segments = 0;
    long long tail_ms = 0, pred_ms = 0; double tail_s = 0;
    // End-of-speech -> first MT token, from the VAD decision and from the last
    // speech sample. In --realtime these are wall-clock on the replay timeline.
    long long detect_to_mt1_ms = -1, last_to_mt1_ms = -1;
    long long last_to_asr_done_ms = -1;   // realtime: transcript final, from last speech
    int    n_cancel = 0, n_pause_trig = 0;
    std::string asr_text, mt_text;
};

static void usage() {
    std::cout <<
R"(qdc_s2st_bench -- round-trip RTF and end-of-speech->begin-of-voice latency

  --audio FILE.wav        16 kHz mono input (required)
  --asr-port N            llama-server ASR port (OpenCL/GPU)  [default 8083]
  --mt-port N             llama-server MT port (HTP/NPU)      [default 8084]
  --src LANG              source language code (en|zh|ko|vi)   [default en]
  --tgt LANG              target language code               [default vi]
  --tts ONNX              Piper voice .onnx
  --tts-map JSON          Piper phoneme_id_map .onnx.json
  --tts-ids FILE          "text<TAB>id,id,..." table, when espeak-ng is absent
  --espeak-data DIR       directory containing espeak-ng-data/ (REQUIRED on
                          Android -- espeak has no compiled-in default there and
                          initialising without it segfaults on first phonemise)
  --tts-threads N         ORT intra-op threads               [default 4]
  --hangover-ms N         VAD hangover                      [default 250]
  --asr-api API           completion: /completion with the eval-verified prompt
                          (user turn = audio only, as asr4/scripts/eval_gguf_server.py)
                          transcribe: /v1/audio/transcriptions (adds an English
                          instruction the model was never evaluated with)
                                                             [default completion]
  --no-force-lang         let the model emit "language X<asr_text>" itself instead
                          of prefilling it from --src (costs ~4 decode tokens/call)
  --realtime              replay each utterance on a wall clock and decode
                          overlapping ASR windows WHILE it is being "spoken".
                          All latencies are then measured, not modelled.
                          Segments are cut at pauses, so they concatenate
                          without overlap; a long pause-free stretch gets a
                          forced cut and its next segment re-decodes an overlap.
  --pause-ms N            silence that counts as a cut point          [default 100]
  --pause-rel X           "silence" = frame energy < X * the utterance's
                          75th-percentile frame energy                [default 0.15]
  --rt-min-seg S          shortest segment worth a decode             [default 0.6]
  --rt-max-seg S          force a cut after S s with no pause         [default 2.0]
  --rt-overlap S          overlap re-decoded after a forced cut       [default 0.5]
  --spec-ms N             start the final decode after N ms of silence,
                          inside the hangover (0 = wait for the VAD)  [default 120]
  --no-pause-trigger      no decoding while speaking (tail only)
  --mt-slot N             pin MT requests to server slot N (with -np >= 2, one
                          slot per target language keeps each prompt cached)
  --mt-warm               prefill the MT system prompt once before the first
                          utterance, so the cold-prompt cost is not charged to it
  --pred-window N         incremental overlapped ASR: decode a sliding window of
                          N seconds (transcripts are stitched)        [default 2.0]
  --pred-step N           advance the window by N seconds (set once, enables)
                          [default 1.0]. Needs step >= cost(window) or the
                          queue grows: device cost is ~750 ms + 150 ms/audio-s.
  --chunk-min N           min chars before a clause is speakable   [default 14]
  --chunk-max N           force-flush a clause at this length    [default 160]
  --terms "a,b,c"         glossary terms for the MT prompt
  --out FILE.jsonl        per-utterance results              [default results.jsonl]
  --save-tts DIR          write synthesised audio per utterance
  --warmup N              discard first N utterances         [default 1]
  --limit N               stop after N utterances            [default 0 = all]
  --no-stream             disable MT streaming (A/B the chunking win)
)";
}

int main(int argc, char ** argv) {
    g_epoch = clk::now();

    std::string audio_path, out_path = "results.jsonl", save_dir;
    std::string tts_onnx, tts_map, tts_ids_path, terms_s, espeak_data;
    int asr_port = 8083, mt_port = 8084, tts_threads = 4, warmup = 1, limit = 0;
    std::string src = "en", tgt = "vi";
    VadConfig vcfg;
    int chunk_min = 14, chunk_max = 160;
    double pred_every_s = 0.0, pred_win_s = 2.0, pred_step_s = 1.0;
    bool stream = true;
    std::string asr_api = "completion";
    bool force_lang = true, realtime = false, pause_trigger = true, mt_warm = false;
    double rt_min_seg_s = 0.6, rt_max_seg_s = 2.0, rt_overlap_s = 0.5, pause_rel = 0.15;
    int spec_ms = 120, pause_ms = 100, mt_slot = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--audio")       audio_path = next();
        else if (a == "--out")         out_path = next();
        else if (a == "--save-tts")    save_dir = next();
        else if (a == "--asr-port")    asr_port = std::stoi(next());
        else if (a == "--mt-port")     mt_port = std::stoi(next());
        else if (a == "--src")         src = next();
        else if (a == "--tgt")         tgt = next();
        else if (a == "--tts")         tts_onnx = next();
        else if (a == "--tts-map")     tts_map = next();
        else if (a == "--tts-ids")     tts_ids_path = next();
        else if (a == "--espeak-data")  espeak_data = next();
        else if (a == "--tts-threads") tts_threads = std::stoi(next());
        else if (a == "--hangover-ms") vcfg.hangover_ms = std::stoi(next());
        else if (a == "--pred-window")  pred_win_s = std::atof(next().c_str());
        else if (a == "--pred-step")   { pred_every_s = 1.0; pred_step_s = std::atof(next().c_str()); }
        else if (a == "--chunk-min")  chunk_min = std::stoi(next());
        else if (a == "--chunk-max")  chunk_max = std::stoi(next());
        else if (a == "--terms")       terms_s = next();
        else if (a == "--warmup")      warmup = std::stoi(next());
        else if (a == "--limit")       limit = std::stoi(next());
        else if (a == "--no-stream")   stream = false;
        else if (a == "--asr-api")     asr_api = next();
        else if (a == "--no-force-lang") force_lang = false;
        else if (a == "--realtime")    realtime = true;
        else if (a == "--rt-min-seg")  rt_min_seg_s = std::atof(next().c_str());
        else if (a == "--rt-max-seg")  rt_max_seg_s = std::atof(next().c_str());
        else if (a == "--rt-overlap")  rt_overlap_s = std::atof(next().c_str());
        else if (a == "--spec-ms")     spec_ms = std::stoi(next());
        else if (a == "--pause-ms")    pause_ms = std::stoi(next());
        else if (a == "--pause-rel")   pause_rel = std::atof(next().c_str());
        else if (a == "--no-pause-trigger") pause_trigger = false;
        else if (a == "--mt-slot")     mt_slot = std::stoi(next());
        else if (a == "--mt-warm")     mt_warm = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
    }
    if (audio_path.empty()) { usage(); return 2; }

    std::vector<std::string> terms;
    { std::stringstream ss(terms_s); std::string t;
      while (std::getline(ss, t, ',')) if (!t.empty()) terms.push_back(t); }

    Wav w;
    if (!wav_read(audio_path, &w)) { std::cerr << "cannot read " << audio_path << "\n"; return 1; }
    std::printf("# input %s  %.1f s @ %d Hz\n", audio_path.c_str(),
                w.pcm.size() / (double)w.rate, w.rate);

    std::unique_ptr<Tts> tts;
    std::map<std::string, std::vector<int64_t>> ids_table;
    const std::map<std::string, std::vector<int64_t>> * ids_ptr = nullptr;
    if (!tts_onnx.empty()) {
        tts = Tts::create(tts_onnx, tts_map, tts_threads, espeak_data);
        std::printf("# TTS %s  espeak=%s\n", tts_onnx.c_str(),
                    Tts::espeak_available() ? "yes" : "NO (need --tts-ids)");
        if (!tts_ids_path.empty() && load_ids_table(tts_ids_path, &ids_table)) ids_ptr = &ids_table;
    } else {
        std::printf("# TTS disabled -- eos_to_voice will be -1\n");
    }

    // ── VAD pass ──────────────────────────────────────────────────────────────
    Vad vad(vcfg);
    std::vector<VadEvent> events;
    {
        const size_t blk = 16000;
        for (size_t o = 0; o < w.pcm.size(); o += blk) {
            size_t n = std::min(blk, w.pcm.size() - o);
            std::vector<float> part(w.pcm.begin() + o, w.pcm.begin() + o + n);
            auto ev = vad.feed(part, 1000.0 * o / w.rate);
            events.insert(events.end(), ev.begin(), ev.end());
        }
        auto ev = vad.flush(1000.0 * w.pcm.size() / w.rate);
        events.insert(events.end(), ev.begin(), ev.end());
    }
    std::printf("# VAD: %d utterances, hangover %d ms, floor %.2e\n",
                (int)events.size(), vcfg.hangover_ms, vad.noise_floor());
    if (events.empty()) return 1;

    HttpOptions asr_opt, mt_opt;
    asr_opt.port = asr_port;
    mt_opt.port  = mt_port;

    // The media marker is random per server process unless pinned with
    // LLAMA_MEDIA_MARKER, so ask the server rather than hardcoding it.
    std::string media_marker;
    if (asr_api == "completion") {
        media_marker = json_get_string(http_get("/props", asr_opt).body, "media_marker");
        if (media_marker.empty()) { std::cerr << "ASR /props gave no media_marker\n"; return 1; }
        if (force_lang && !lang_names().count(src)) { std::cerr << "unknown --src " << src << "\n"; return 1; }
    }
    std::printf("# ASR api=%s force_lang=%d  realtime=%d pause=%dms min/max seg %.1f/%.1fs "
                "overlap %.1fs spec=%dms segmenting=%d\n",
                asr_api.c_str(), (int)force_lang, (int)realtime, pause_ms, rt_min_seg_s,
                rt_max_seg_s, rt_overlap_s, spec_ms, (int)pause_trigger);

    auto mt_body = [&](const std::string & text, bool streaming, int max_tokens) {
        return "{\"messages\":[{\"role\":\"user\",\"content\":\"" +
               json_escape(mt_prompt(text, tgt, terms)) + "\"}],"
               "\"stream\":" + std::string(streaming ? "true" : "false") +
               ",\"cache_prompt\":true" +
               (mt_slot >= 0 ? ",\"id_slot\":" + std::to_string(mt_slot) : std::string()) +
               ",\"temperature\":0,\"top_p\":0.6,\"max_tokens\":" + std::to_string(max_tokens) + "}";
    };
    if (mt_warm) {
        // Everything up to the source text is identical across utterances with
        // the same target, so one throwaway request leaves it in the KV cache.
        auto t0 = clk::now();
        http_post_json("/v1/chat/completions", mt_body("", false, 1), mt_opt);
        std::printf("# MT prompt warm-up %lld ms\n", ms_since(t0));
    }

    std::vector<Result> results;
    std::ofstream jout(out_path);
    int n_fail = 0;

    for (size_t i = 0; i < events.size(); ++i) {
        const VadEvent & e = events[i];
        if (warmup > 0 && (int)i < warmup) continue;
        if (limit > 0 && (int)results.size() >= limit) break;

        Result r;
        r.utt = e.index;
        std::vector<float> pcm((const float *)e.audio.data(),
                               (const float *)e.audio.data() + e.audio.size() / sizeof(float));
        r.audio_s = pcm.size() / 16000.0;
        r.t_start_ms = e.t_start_ms;
        r.t_last_ms = e.t_last_ms;
        r.t_detect_ms = e.t_detect_ms;
        r.vad_penalty_ms = e.t_detect_ms - e.t_last_ms;

        std::printf("\n[%d] %.2fs  speech %.0f-%.0f ms  detect %.0f  (+%.0f penalty)\n",
                    r.utt, r.audio_s, e.t_start_ms, e.t_last_ms, e.t_detect_ms, r.vad_penalty_ms);

        // Everything below is timed from the VAD decision instant -- except in
        // --realtime, where the clock starts at the first speech sample and the
        // VAD decision is a point on that timeline (t_origin + detect_at).
        const clk::time_point t_detect = clk::now();
        clk::time_point t_detect_wall = t_detect;
        double last_ms = 0.0;   // realtime: last speech sample, ms after t_origin
        clk::time_point t_origin = t_detect;

        // ── ASR (blocking, OpenCL/GPU) ────────────────────────────────────────
        auto asr_request = [&](const std::vector<float> & seg, long long * took_ms,
                               std::string * text,
                               const std::atomic<bool> * cancel = nullptr) -> int {
            HttpOptions o = asr_opt;
            o.cancel = cancel;
            auto t0 = clk::now();
            HttpResponse resp;
            if (asr_api == "completion") {
                // Same turn structure the GGUF accuracy evals used: user turn is
                // the audio alone. Prefilling "language X<asr_text>" from the
                // known source language skips ~4 decode steps (~56 ms each).
                std::string p = "<|im_start|>user\n" + media_marker +
                                "<|im_end|>\n<|im_start|>assistant\n";
                if (force_lang) p += "language " + lang_names().at(src) + "<asr_text>";
                resp = http_post_json("/completion",
                    "{\"prompt\":{\"prompt_string\":\"" + json_escape(p) +
                    "\",\"multimodal_data\":[\"" + base64(wav_encode_16k(seg)) + "\"]},"
                    // cache_prompt off: an identical clip (a re-run of the same
                    // file) would otherwise skip audio encoding entirely and
                    // report ~0.6 s too fast. Only ~3 template tokens precede the
                    // audio, so honest timing costs nothing in deployment.
                    "\"n_predict\":256,\"temperature\":0,\"cache_prompt\":false}", o);
            } else {
                resp = http_post_multipart(
                    "/v1/audio/transcriptions",
                    {{"model", "asr"}, {"response_format", "json"}, {"language", src}},
                    "file", "audio.wav", "audio/wav", wav_encode_16k(seg), o);
            }
            if (took_ms) *took_ms = ms_since(t0);
            if (cancel && cancel->load()) return -2;
            if (resp.status != 200) return resp.status;
            if (text) *text = clean_asr_text(json_get_string(
                resp.body, asr_api == "completion" ? "content" : "text"));
            return 200;
        };
        // Timeline state for the overlapped-ASR emulation (unused when off).
        double busy_until = 0.0, total_ms = 0.0;
        long long tail_ms = 0;
        if (realtime) {
            // ── Real-time replay, overlapped ASR ────────────────────────────
            // Sample k of the utterance is "spoken" at t_origin + k/16 ms, and
            // ASR may only see audio that has been spoken. The decoder works
            // while the user talks; at end of speech only the uncovered tail is
            // left. Nothing here is modelled: every latency is a clock reading.
            //
            // Windows are adaptive rather than fixed: whenever the decoder is
            // idle and >= step of new audio exists, decode [committed - overlap,
            // now]. A fixed [t-W, t] schedule loses coverage whenever a decode
            // overruns its slot; this one just takes a longer window next time.
            struct Call {
                std::thread th;
                std::atomic<bool> done{false};
                int status = 0; long long took = 0; std::string text;
                double a0 = 0, a1 = 0, t_start = 0, t_end = 0;
            };
            const double D = pcm.size() / 16.0;
            const double detect_at = D + r.vad_penalty_ms;
            const double spec_at = (spec_ms > 0 && spec_ms < r.vad_penalty_ms) ? D + spec_ms : detect_at;
            last_ms = D;

            // 20 ms frame energies. A gap between words never drops to the
            // VAD's noise-floor threshold (measured: zero pauses found in 113
            // utterances), so "quiet" is relative to this utterance's own level.
            const int fr = 320;
            std::vector<double> rms(pcm.size() / fr + 1, 0.0);
            for (size_t f = 0; f * fr < pcm.size(); ++f) {
                double acc = 0; size_t n = std::min((size_t)fr, pcm.size() - f * fr);
                for (size_t k = 0; k < n; ++k) acc += (double)pcm[f * fr + k] * pcm[f * fr + k];
                rms[f] = std::sqrt(acc / n);
            }
            std::vector<double> srt(rms);
            std::sort(srt.begin(), srt.end());
            const double level = srt[srt.size() * 3 / 4];   // typical voiced frame
            const double thr = std::max(vad.noise_floor() * vcfg.offset_ratio, pause_rel * level);
            const double loud = std::max(vad.noise_floor() * vcfg.onset_ratio, vcfg.absolute_floor);
            auto has_speech = [&](double lo, double hi) {
                for (int f = (int)(lo / 20.0); f < (int)(hi / 20.0) && f < (int)rms.size(); ++f)
                    if (rms[f] > loud) return true;
                return false;
            };
            const int pause_frames = std::max(1, pause_ms / 20);
            // Latest pause (>= pause_ms of quiet) inside (lo, hi]; returns the
            // middle of the quiet run, or -1. Cutting there means neither side
            // holds a partial word, so segments simply concatenate.
            auto latest_pause = [&](double lo, double hi) -> double {
                int run = 0;
                for (int f = (int)(hi / 20.0) - 1; f >= 0 && f * 20.0 > lo; --f) {
                    if (f < (int)rms.size() && rms[f] < thr) {
                        if (++run >= pause_frames) {
                            int end = f + run;   // first frame after the run so far
                            while (end < (int)rms.size() && end * 20.0 < hi && rms[end] < thr) ++end;
                            return (f + end) * 10.0;
                        }
                    } else {
                        run = 0;
                    }
                }
                return -1.0;
            };
            auto quietest = [&](double lo, double hi) {   // fallback cut point
                int best = (int)(hi / 20.0) - 1;
                for (int f = best; f >= 0 && f * 20.0 >= lo; --f)
                    if (f < (int)rms.size() && rms[f] < rms[best]) best = f;
                return best * 20.0 + 10.0;
            };

            StreamStitcher stitcher(src == "zh");
            double committed = 0.0;     // audio [0, committed) has text
            bool overlap_next = false;  // last cut was forced mid-speech
            bool cur_overlap = false;
            std::unique_ptr<Call> cur;
            t_origin = clk::now();
            t_detect_wall = t_origin + std::chrono::microseconds((long long)(detect_at * 1000));
            auto now_rel = [&] {
                return std::chrono::duration<double, std::milli>(clk::now() - t_origin).count();
            };
            auto launch = [&](double a1, bool forced) {
                const double a0 = overlap_next ? std::max(0.0, committed - rt_overlap_s * 1000.0) : committed;
                cur.reset(new Call);
                Call * c = cur.get();
                c->a0 = a0; c->a1 = a1; c->t_start = now_rel();
                cur_overlap = overlap_next;
                overlap_next = forced;
                committed = a1;   // claimed; the text arrives when the call returns
                std::vector<float> seg(pcm.begin() + (size_t)(a0 * 16.0),
                                       pcm.begin() + std::min(pcm.size(), (size_t)(a1 * 16.0)));
                c->th = std::thread([&asr_request, &now_rel, c, seg = std::move(seg)] {
                    c->status = asr_request(seg, &c->took, &c->text);
                    c->t_end = now_rel();
                    c->done = true;
                });
            };
            auto reap = [&]() -> double {
                cur->th.join();
                const double t_end = cur->t_end;
                if (cur->status == 200) {
                    if (cur_overlap) stitcher.add(cur->text);   // align on the re-decoded overlap
                    else stitcher.append(cur->text);           // cut at silence: just concatenate
                } else {
                    ++n_fail;
                }
                if (cur->a1 >= D - 1.0) { tail_ms = cur->took; r.tail_s = (cur->a1 - cur->a0) / 1000.0; }
                std::printf("      seg %6.0f-%6.0f ms  sent %6.0f  back %6.0f  (%lld ms)%s\n",
                            cur->a0, cur->a1, cur->t_start, t_end, cur->took,
                            cur_overlap ? "  [overlap]" : "");
                ++r.n_predecode;
                cur.reset();
                return t_end;
            };

            double t_asr_done = -1.0;
            bool tail = false;
            for (;;) {
                const double t = now_rel();
                if (cur && cur->done) {
                    const double te = reap();
                    if (tail) { t_asr_done = te; break; }
                }
                if (!tail && t >= spec_at) {
                    // Enough silence to commit: whatever is in flight covers audio
                    // we need, so let it finish, then decode the remainder.
                    double te = now_rel();
                    if (cur) te = reap();
                    // A sliver after the last cut costs a full call (~0.7 s
                    // fixed) for no words; skip it unless it holds speech.
                    if (committed >= D - 1.0 || !has_speech(committed, D)) { t_asr_done = te; break; }
                    tail = true;
                    launch(D, false);
                    continue;
                }
                if (!tail && !cur && pause_trigger) {
                    const double avail = std::min(t, D);
                    const double cut = latest_pause(committed + rt_min_seg_s * 1000.0, avail);
                    if (cut > 0) {
                        ++r.n_pause_trig;
                        launch(cut, false);
                    } else if (avail - committed >= rt_max_seg_s * 1000.0) {
                        launch(quietest(avail - 400.0, avail), true);
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            r.stitch_segments = (int)stitcher.segments();
            r.asr_text = stitcher.text();
            r.last_to_asr_done_ms = (long long)llround(t_asr_done - D);
            r.asr_ms = r.last_to_asr_done_ms;   // ASR on the critical path, from last speech
            r.tail_ms = tail_ms;
            // MT may not start before the VAD has committed the utterance.
            std::this_thread::sleep_until(t_detect_wall);
        } else if (pred_every_s <= 0.0) {
            long long took = 0; std::string txt;
            int st = asr_request(pcm, &took, &txt);
            r.asr_ms = took;
            if (st != 200) { std::printf("    ASR HTTP %d\n", st); ++n_fail; continue; }
            r.asr_text = txt;
        } else {
            // ── Incremental overlapped ASR ──────────────────────────────────
            // Decode a sliding WINDOW of audio, not audio-so-far. Each trigger
            // covers [t-W, t]; overlapping regions are merged by prefix match.
            // This is O(D) of GPU work instead of O(D^2/step), which is what
            // sank the first emulation: at 0.5 s triggers the redundant
            // re-decoding cost more than it saved.
            //
            // The decoder can only stay ahead if step >= cost(window). Measured
            // on device: asr_ms ~= 750 + 150 * audio_s, so a 2 s window costs
            // 1.05 s and needs a >= 1.05 s step or the queue grows without bound.
            total_ms = pcm.size() / 16000.0 * 1000.0;
            const double win_ms = pred_win_s * 1000.0;
            const double step_ms = pred_step_s * 1000.0;
            StreamStitcher stitcher(src == "ko" || src == "zh");
            std::vector<double> started, finished;
            double seg_end_ms = 0.0;      // audio right edge already decoded

            // first trigger must wait for a full window of audio to exist
            for (double t = win_ms; t <= total_ms; t += step_ms) {
                const double a = std::max(0.0, t - win_ms);
                const size_t i0 = (size_t)(a / 1000.0 * 16000.0);
                const size_t i1 = (size_t)(t / 1000.0 * 16000.0);
                if (i1 <= i0) continue;
                long long took = 0; std::string txt;
                if (asr_request(std::vector<float>(pcm.begin() + i0, pcm.begin() + i1),
                                &took, &txt) != 200)
                    continue;
                double st = t;                       // trigger time (audio exists)
                if (!finished.empty() && finished.back() > st)
                    st = finished.back();            // server serialises
                started.push_back(st);
                finished.push_back(st + (double)took);
                if (st + (double)took <= total_ms) {  // finished before end-of-speech
                    stitcher.add(txt);
                    seg_end_ms = t;
                }
            }
            r.n_predecode = (int)started.size();
            for (size_t i = 0; i < started.size(); ++i)
                if (finished[i] > total_ms) { busy_until = finished[i]; break; }

            // Final decode covers the tail back to the window overlap, so the
            // same audio is transcribed twice and the stitcher can align it.
            const size_t j_end = pcm.size();
            const size_t j_start = (size_t)(std::max(0.0, seg_end_ms + step_ms - win_ms)
                                           / 1000.0 * 16000.0);
            std::string tail_text;
            if (j_end > j_start) {
                asr_request(std::vector<float>(pcm.begin() + j_start, pcm.begin() + j_end),
                            &tail_ms, &tail_text);
                stitcher.add(tail_text);
            }
            r.asr_ms = (long long)llround(std::max(0.0, busy_until - total_ms)) + tail_ms;
            r.tail_ms = tail_ms;
            r.tail_s = (j_end - j_start) / 16000.0;
            r.pred_ms = (long long)llround(std::max(0.0, busy_until - total_ms));
            r.stitch_segments = (int)stitcher.segments();
            r.asr_text = stitcher.text();
        }
        std::printf("    ASR %6lld ms  \"%s\"\n", r.asr_ms, r.asr_text.substr(0, 70).c_str());

        // ── MT (streaming, HTP/NPU) → clause queue → TTS (CPU) ────────────────
        TtsRunner runner(tts.get(), ids_ptr, tgt);
        const std::string body = mt_body(r.asr_text, stream, 256);

        ClauseChunker chunker(chunk_min, chunk_max);
        auto t_mt0 = clk::now();
        long long mt_first_token_ms = -1, mt_first_clause_ms = -1;
        clk::time_point t_mt_first{};

        if (stream) {
            std::string full;
            auto on_delta = [&](const std::string & payload) -> bool {
                if (mt_first_token_ms < 0) { mt_first_token_ms = ms_since(t_mt0); t_mt_first = clk::now(); }
                std::string d = json_get_string(payload, "content");
                if (d.empty()) return true;
                full += d;
                std::string chunk;
                while (chunker.push(d, &chunk)) {
                    d.clear();
                    if (mt_first_clause_ms < 0) mt_first_clause_ms = ms_since(t_mt0);
                    ++r.n_chunks;
                    runner.submit(chunk);
                    if (!d.empty()) chunker.push(d, &chunk);
                }
                return true;
            };
            http_post_stream("/v1/chat/completions", body, mt_opt, on_delta);
            std::string tail;
            if (chunker.flush(&tail)) {
                if (mt_first_clause_ms < 0) mt_first_clause_ms = ms_since(t_mt0);
                ++r.n_chunks;
                runner.submit(tail);
            }
            r.mt_text = clean_mt_text(full, tgt);
            std::printf("    MT  first token %lld ms, first clause %lld ms, total %lld ms\n",
                        mt_first_token_ms, mt_first_clause_ms, ms_since(t_mt0));
        } else {
            HttpResponse resp = http_post_json("/v1/chat/completions", body, mt_opt);
            r.mt_text = clean_mt_text(json_get_string(resp.body, "content"), tgt);
            mt_first_token_ms = mt_first_clause_ms = ms_since(t_mt0);
            t_mt_first = clk::now();
            r.n_chunks = 1;
            runner.submit(r.mt_text);
            std::printf("    MT  (non-streaming) %lld ms\n", mt_first_token_ms);
        }

        runner.drain();
        const double detect_rel = rel_ms(t_detect_wall);
        const long long fv_abs  = runner.first_voice_ms();
        if (pred_every_s > 0.0 && !realtime) {
            // MODELLING, not wall-clock. In file mode every pre-decode is issued
            // AFTER the utterance ends, so the real elapsed time contains work
            // that in a live system would have happened while the user was still
            // talking. Charging that to the user would make overlap look worse
            // than doing nothing.
            //
            // first_voice = (pre-decode still draining at EOS)
            //             + (decode of audio not yet covered)
            //             + MT to first clause
            //             + TTS of that clause
            const double asr_wait = std::max(0.0, busy_until - total_ms)
                                  + (double)tail_ms;
            const double mt_wait = (r.mt_first_clause_ms >= 0)
                                 ? (double)r.mt_first_clause_ms : 0.0;
            const double tts_wait = (r.tts_chunks > 0)
                                  ? (double)runner.tts_first_chunk_ms() : 0.0;
            r.eos_to_voice_ms = (long long)llround(asr_wait + mt_wait + tts_wait);
            r.asr_ms = (long long)llround(asr_wait);
            r.model_ms = 1;
        } else if (fv_abs >= 0) {
            // (asr_ms used to be overwritten with this value, which made every
            // "ASR" figure from non-overlapped runs really eos->voice)
            r.eos_to_voice_ms = (long long)llround(fv_abs - detect_rel);
        }
        if (mt_first_token_ms >= 0) {
            if (pred_every_s > 0.0 && !realtime)   // modelled timeline, as above
                r.detect_to_mt1_ms = r.asr_ms + mt_first_token_ms;
            else
                r.detect_to_mt1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_mt_first - t_detect_wall).count();
            r.last_to_mt1_ms = r.detect_to_mt1_ms + (long long)llround(r.vad_penalty_ms);
        }
        if (r.eos_to_voice_ms >= 0)
            r.eos_to_voice_from_last_ms =
                r.eos_to_voice_ms + (long long)llround(r.vad_penalty_ms);
        r.tts_chunks = runner.done();
        r.tts_prep_ms = runner.tts_prep_ms();
        r.tts_run_ms  = runner.tts_run_ms();
        r.tts_lead_offset_ms = runner.first_voice_offset_ms();
        r.mt_first_token_ms = mt_first_token_ms;
        r.mt_first_clause_ms = mt_first_clause_ms;
        r.total_ms = ms_since(t_detect_wall);
        r.rtf = r.audio_s > 0 ? r.total_ms / (1000.0 * r.audio_s) : 0;

        if (!runner.last_error.empty())
            std::printf("    TTS %s\n", runner.last_error.c_str());

        std::printf("    >>> eos->voice %lld ms   (from last speech %lld)   rtf %.3f   "
                    "clauses %d  tts %d\n",
                    r.eos_to_voice_ms, r.eos_to_voice_from_last_ms, r.rtf,
                    r.n_chunks, r.tts_chunks);

        if (!save_dir.empty() && !runner.audio().empty()) {
            std::string p = save_dir + "/utt" + std::to_string(r.utt) + ".wav";
            wav_write(p, runner.audio(), 22050);
        }

        results.push_back(r);
        if (jout.is_open()) {
            jout << "{\"utt\":" << r.utt
                 << ",\"audio_s\":" << r.audio_s
                 << ",\"vad_penalty_ms\":" << r.vad_penalty_ms
                 << ",\"asr_ms\":" << r.asr_ms
                 << ",\"mt_first_token_ms\":" << r.mt_first_token_ms
                 << ",\"mt_first_clause_ms\":" << r.mt_first_clause_ms
                 << ",\"detect_to_mt1_ms\":" << r.detect_to_mt1_ms
                 << ",\"last_to_mt1_ms\":" << r.last_to_mt1_ms
                 << ",\"last_to_asr_done_ms\":" << r.last_to_asr_done_ms
                 << ",\"n_cancel\":" << r.n_cancel
                 << ",\"n_pause_trig\":" << r.n_pause_trig
                 << ",\"realtime\":" << (realtime ? 1 : 0)
                 << ",\"eos_to_voice_ms\":" << r.eos_to_voice_ms
                 << ",\"eos_to_voice_from_last_speech_ms\":" << r.eos_to_voice_from_last_ms
                 << ",\"total_ms\":" << r.total_ms
                 << ",\"rtf\":" << r.rtf
                 << ",\"n_predecode\":" << r.n_predecode
                 << ",\"stitch_segments\":" << r.stitch_segments
                 << ",\"tail_s\":" << r.tail_s
                 << ",\"tail_ms\":" << r.tail_ms
                 << ",\"n_chunks\":" << r.n_chunks
                 << ",\"tts_chunks\":" << r.tts_chunks
                 << ",\"tts_prep_ms\":" << r.tts_prep_ms
                 << ",\"tts_run_ms\":" << r.tts_run_ms
                 << ",\"tts_lead_offset_ms\":" << r.tts_lead_offset_ms
                 << ",\"asr_text\":\"" << json_escape(r.asr_text) << "\""
                 << ",\"mt_text\":\"" << json_escape(r.mt_text) << "\"}\n";
            jout.flush();
        }
    }

    auto q = [](std::vector<double> v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[(size_t)(p * (v.size() - 1))];
    };
    std::vector<double> lat, latl, rtf, asr, vadp, tok, cl, m1, m1l;
    for (auto & r : results) {
        if (r.eos_to_voice_ms >= 0) { lat.push_back(r.eos_to_voice_ms); latl.push_back(r.eos_to_voice_from_last_ms); }
        rtf.push_back(r.rtf);
        asr.push_back(r.asr_ms);
        vadp.push_back(r.vad_penalty_ms);
        if (r.mt_first_token_ms >= 0) tok.push_back(r.mt_first_token_ms);
        if (r.mt_first_clause_ms >= 0) cl.push_back(r.mt_first_clause_ms);
        if (r.last_to_mt1_ms >= 0) { m1.push_back(r.detect_to_mt1_ms); m1l.push_back(r.last_to_mt1_ms); }
    }

    std::printf("\n================ summary (%d utterances, %d failed) ================\n",
                (int)results.size(), n_fail);
    if (!lat.empty()) {
        std::printf("  END-OF-SPEECH -> BEGIN OF VOICE   median %7.0f ms    p90 %7.0f ms\n",
                    q(lat, 0.5), q(lat, 0.9));
        std::printf("    incl. VAD detection penalty       median %7.0f ms\n", q(latl, 0.5));
        std::printf("  VAD detection penalty              median %7.0f ms\n", q(vadp, 0.5));
    } else {
        std::printf("  END-OF-SPEECH -> BEGIN OF VOICE   unavailable (TTS disabled or no ids)\n");
    }
    if (!m1l.empty()) {
        std::printf("  LAST SPEECH -> FIRST MT TOKEN      median %7.0f ms    p90 %7.0f ms    max %7.0f ms\n",
                    q(m1l, 0.5), q(m1l, 0.9), q(m1l, 1.0));
        std::printf("  VAD DECISION -> FIRST MT TOKEN     median %7.0f ms    p90 %7.0f ms\n",
                    q(m1, 0.5), q(m1, 0.9));
    }
    std::printf("  ROUND-TRIP RTF                     median %7.3f\n", q(rtf, 0.5));
    std::printf("  ASR stage                          median %7.0f ms\n", q(asr, 0.5));
    if (!tok.empty()) std::printf("  MT first token                     median %7.0f ms\n", q(tok, 0.5));
    if (!cl.empty())  std::printf("  MT first clause                    median %7.0f ms\n", q(cl, 0.5));
    std::printf("  results -> %s\n", out_path.c_str());
    return 0;
}
