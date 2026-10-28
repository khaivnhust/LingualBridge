// mono.cc -- the whole S2ST pipeline in ONE process (skeleton).
//
//   capture (wall-clock replay) -> VAD -> ASR  [Adreno/OpenCL, own thread]
//                                      -> MT   [Hexagon/HTP, own thread, streaming]
//                                      -> TTS  [CPU/ORT, own thread, per clause]
//
// What the two-server harness could not do, and this does:
//   * ASR of utterance N+1 overlaps MT/TTS of utterance N (measured: -1.3 s per
//     pair, measured with a one-process co-residency probe), with each stage on its own thread and device.
//   * Speculative ASR: decode starts after --spec-ms of silence, inside the VAD
//     hangover; if speech resumes the result is discarded (WORKLOG §17, config E:
//     same transcripts, 19 -> 11 of 113 utterances over 2 s).
//   * Explicit MT prompt-KV swapping: the KV state of each target language's
//     system prompt is saved once (llama_state_seq_get_data) and restored when
//     the direction changes, so language pair/direction can change per utterance.
//
// Input is a script of "wav src tgt" lines, replayed back to back on a wall clock
// with --gap-ms of silence between them. Output: per-utterance JSONL timings
// (all relative to the last speech sample) and the synthesised speech as a WAV
// laid out on the same timeline.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "http.h"
#include "pipeline.h"
#include "tts.h"
#include "vad.h"

using namespace s2st;
using clk = std::chrono::steady_clock;

static clk::time_point g_t0;
static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now() - g_t0).count(); }

// ── small thread-safe queue ──────────────────────────────────────────────────
template <class T> class Queue {
public:
    void push(T v) { { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(v)); } cv_.notify_all(); }
    bool pop(T * out) {   // false once closed and drained
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        *out = std::move(q_.front()); q_.pop_front();
        return true;
    }
    void close() { { std::lock_guard<std::mutex> l(m_); closed_ = true; } cv_.notify_all(); }
private:
    std::mutex m_; std::condition_variable cv_; std::deque<T> q_; bool closed_ = false;
};

static bool read_wav16k(const std::string & path, std::vector<float> * out) {
    std::ifstream f(path, std::ios::binary);
    std::string a((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (a.size() < 44 || a.compare(0, 4, "RIFF")) return false;
    for (size_t p = 12; p + 8 <= a.size();) {
        uint32_t sz; std::memcpy(&sz, &a[p + 4], 4);
        if (a.compare(p, 4, "data") == 0) {
            size_t n = std::min<size_t>(sz, a.size() - p - 8) / 2;
            out->resize(n);
            for (size_t i = 0; i < n; ++i) { int16_t s; std::memcpy(&s, &a[p + 8 + 2 * i], 2); (*out)[i] = s / 32768.f; }
            return true;
        }
        p += 8 + sz + (sz & 1);
    }
    return false;
}

static ggml_backend_dev_t dev(const char * name) {
    ggml_backend_dev_t d = ggml_backend_dev_by_name(name);
    if (!d) { std::fprintf(stderr, "no device %s\n", name); std::exit(1); }
    return d;
}

// ── ASR: Qwen3-ASR on the Adreno ─────────────────────────────────────────────
struct AsrJob {
    int uid = 0; std::string src; std::vector<float> audio;
    std::atomic<bool> stale{false}, done{false};
    std::string text; double t_start = -1, t_done = -1;
};

class AsrEngine {
public:
    bool load(const char * gguf, const char * mmproj, int n_threads) {
        ggml_backend_dev_t ds[] = {dev("GPUOpenCL"), nullptr};
        auto mp = llama_model_default_params(); mp.devices = ds; mp.n_gpu_layers = 99;
        model_ = llama_model_load_from_file(gguf, mp);
        if (!model_) return false;
        auto cp = llama_context_default_params(); cp.n_ctx = 2048; cp.n_batch = 2048;
        cp.n_threads = cp.n_threads_batch = n_threads;
        ctx_ = llama_init_from_model(model_, cp);
        voc_ = llama_model_get_vocab(model_);
        auto xp = mtmd_context_params_default();
        xp.use_gpu = true; xp.device = dev("GPUOpenCL"); xp.print_timings = false; xp.n_threads = n_threads;
        mt_ = mtmd_init_from_file(mmproj, model_, xp);
        return ctx_ && mt_;
    }
    // Same prompt as the GGUF evals, with the known source language prefilled
    // (saves ~4 decode steps; WORKLOG §17).
    std::string run(const std::vector<float> & pcm, const std::string & src) {
        llama_memory_clear(llama_get_memory(ctx_), true);
        std::string p = std::string("<|im_start|>user\n") + mtmd_default_marker() +
                        "<|im_end|>\n<|im_start|>assistant\nlanguage " + lang_names().at(src) + "<asr_text>";
        mtmd_bitmap * bm = mtmd_bitmap_init_from_audio(pcm.size(), pcm.data());
        mtmd_input_text txt{p.c_str(), p.size(), false, true};
        mtmd_input_chunks * ch = mtmd_input_chunks_init();
        const mtmd_bitmap * bms[] = {bm};
        std::string out;
        llama_pos n_past = 0;
        if (mtmd_tokenize(mt_, ch, &txt, bms, 1) == 0 &&
            mtmd_helper_eval_chunks(mt_, ctx_, ch, 0, 0, llama_n_batch(ctx_), true, &n_past) == 0) {
            llama_sampler * s = llama_sampler_init_greedy();
            for (int i = 0; i < 256; ++i) {
                llama_token t = llama_sampler_sample(s, ctx_, -1);
                if (llama_vocab_is_eog(voc_, t)) break;
                char buf[256]; int n = llama_token_to_piece(voc_, t, buf, sizeof buf, 0, true);
                if (n > 0) out.append(buf, n);
                if (llama_decode(ctx_, llama_batch_get_one(&t, 1))) break;
            }
            llama_sampler_free(s);
        }
        mtmd_input_chunks_free(ch);
        mtmd_bitmap_free(bm);
        return clean_asr_text(out);
    }
    ~AsrEngine() { if (mt_) mtmd_free(mt_); if (ctx_) llama_free(ctx_); if (model_) llama_model_free(model_); }
private:
    llama_model * model_ = nullptr; llama_context * ctx_ = nullptr; mtmd_context * mt_ = nullptr;
    const llama_vocab * voc_ = nullptr;
};

// ── MT: Hy-MT2 on the Hexagon NPU, with per-target prompt-KV swapping ────────
class MtEngine {
public:
    bool load(const char * gguf, int n_threads) {
        ggml_backend_dev_t ds[] = {dev("HTP0"), nullptr};
        auto mp = llama_model_default_params(); mp.devices = ds; mp.n_gpu_layers = 99;
        model_ = llama_model_load_from_file(gguf, mp);
        if (!model_) return false;
        auto cp = llama_context_default_params(); cp.n_ctx = 4096; cp.n_batch = 512;
        cp.n_threads = cp.n_threads_batch = n_threads;
        ctx_ = llama_init_from_model(model_, cp);
        voc_ = llama_model_get_vocab(model_);
        return ctx_ != nullptr;
    }

    // Everything before the source text is fixed per target language. Render the
    // chat template around a placeholder and split there, so the prefix can be
    // prefilled once and its KV state reused.
    void split_prompt(const std::string & tgt, std::string * pre, std::string * post) {
        static const std::string PH = "\x01SRC\x01";
        const std::string user = mt_prompt(PH, tgt, {});
        llama_chat_message m{"user", user.c_str()};
        std::vector<char> buf(user.size() * 2 + 1024);
        int n = llama_chat_apply_template(llama_model_chat_template(model_, nullptr), &m, 1, true,
                                          buf.data(), buf.size());
        std::string all(buf.data(), std::max(0, n));
        size_t k = all.find(PH);
        *pre = all.substr(0, k);
        *post = all.substr(k + PH.size());
    }

    // Make the KV cache hold exactly tgt's prompt prefix. Returns ms spent and
    // how it got there: "keep" (same target, trim the old suffix), "restore"
    // (saved state copied back), or "prefill" (first use of this target).
    double prepare(const std::string & tgt, std::string * how) {
        const double t0 = now_ms();
        llama_memory_t mem = llama_get_memory(ctx_);
        if (tgt == cur_) {
            llama_memory_seq_rm(mem, 0, prefix_len_[tgt], -1);
            *how = "keep";
        } else {
            llama_memory_clear(mem, true);
            auto it = saved_.find(tgt);
            if (it != saved_.end() && llama_state_seq_set_data(ctx_, it->second.data(), it->second.size(), 0) > 0) {
                *how = "restore";
            } else {
                std::string pre, post;
                split_prompt(tgt, &pre, &post);
                suffix_tmpl_[tgt] = post;
                std::vector<llama_token> tok(pre.size() + 16);
                int n = llama_tokenize(voc_, pre.c_str(), pre.size(), tok.data(), tok.size(), true, true);
                tok.resize(std::max(0, n));
                for (size_t i = 0; i < tok.size(); i += 512) {
                    int m = (int)std::min<size_t>(512, tok.size() - i);
                    llama_decode(ctx_, llama_batch_get_one(tok.data() + i, m));
                }
                prefix_len_[tgt] = (int)tok.size();
                std::vector<uint8_t> st(llama_state_seq_get_size(ctx_, 0));
                st.resize(llama_state_seq_get_data(ctx_, st.data(), st.size(), 0));
                saved_[tgt] = std::move(st);
                *how = "prefill";
            }
            cur_ = tgt;
        }
        return now_ms() - t0;
    }

    // Stream a translation; on_piece gets each detokenised piece as it decodes.
    std::string translate(const std::string & src_text, const std::string & tgt,
                          const std::function<void(const std::string &)> & on_piece) {
        const std::string s = src_text + suffix_tmpl_[tgt];
        std::vector<llama_token> tok(s.size() + 16);
        int n = llama_tokenize(voc_, s.c_str(), s.size(), tok.data(), tok.size(), false, true);
        tok.resize(std::max(0, n));
        llama_decode(ctx_, llama_batch_get_one(tok.data(), (int)tok.size()));
        std::string out;
        llama_sampler * smp = llama_sampler_init_greedy();
        for (int i = 0; i < 256; ++i) {
            llama_token t = llama_sampler_sample(smp, ctx_, -1);
            if (llama_vocab_is_eog(voc_, t)) break;
            char buf[256]; int k = llama_token_to_piece(voc_, t, buf, sizeof buf, 0, false);
            if (k > 0) { std::string p(buf, k); out += p; on_piece(p); }
            if (llama_decode(ctx_, llama_batch_get_one(&t, 1))) break;
        }
        llama_sampler_free(smp);
        return clean_mt_text(out, tgt);
    }
    size_t saved_bytes(const std::string & tgt) const { auto it = saved_.find(tgt); return it == saved_.end() ? 0 : it->second.size(); }
    ~MtEngine() { if (ctx_) llama_free(ctx_); if (model_) llama_model_free(model_); }
private:
    llama_model * model_ = nullptr; llama_context * ctx_ = nullptr; const llama_vocab * voc_ = nullptr;
    std::string cur_;
    std::map<std::string, int> prefix_len_;
    std::map<std::string, std::string> suffix_tmpl_;
    std::map<std::string, std::vector<uint8_t>> saved_;
};

// ── one utterance through the pipeline ───────────────────────────────────────
struct Utt {
    int uid = 0; std::string src, tgt;
    double audio_s = 0, t_last = 0, t_detect = 0;
    std::shared_ptr<AsrJob> job;
    bool speculative = false; int n_stale = 0;
    // filled downstream (absolute ms on g_t0)
    double t_mt_start = -1, t_mt_first = -1, t_clause1 = -1, t_voice1 = -1, mt_swap_ms = 0;
    std::string mt_how, mt_text;
};

struct Clause { std::shared_ptr<Utt> u; std::string text; bool last = false; };

int main(int argc, char ** argv) {
    std::string script, asr_gguf, mmproj, mt_gguf, voices_dir = "voices", espeak = ".";
    std::string out_jsonl = "mono.jsonl", out_wav = "mono_out.wav";
    int spec_ms = 120, gap_ms = 1500, n_threads = 4;
    VadConfig vcfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if      (a == "--script")  script = next();
        else if (a == "--asr")     asr_gguf = next();
        else if (a == "--mmproj")  mmproj = next();
        else if (a == "--mt")      mt_gguf = next();
        else if (a == "--voices")  voices_dir = next();
        else if (a == "--espeak-data") espeak = next();
        else if (a == "--out")     out_jsonl = next();
        else if (a == "--out-wav") out_wav = next();
        else if (a == "--spec-ms") spec_ms = std::stoi(next());
        else if (a == "--gap-ms")  gap_ms = std::stoi(next());
        else if (a == "--hangover-ms") vcfg.hangover_ms = std::stoi(next());
        else if (a == "--threads") n_threads = std::stoi(next());
    }
    if (script.empty() || asr_gguf.empty() || mmproj.empty() || mt_gguf.empty()) {
        std::fprintf(stderr, "usage: %s --script FILE(wav src tgt per line) --asr GGUF --mmproj GGUF --mt GGUF\n"
                             "  [--voices DIR] [--espeak-data DIR] [--out JSONL] [--out-wav WAV]\n"
                             "  [--spec-ms 120] [--gap-ms 1500] [--hangover-ms 250] [--threads 4]\n", argv[0]);
        return 2;
    }

    // timeline: every script item back to back, --gap-ms of silence between
    struct Seg { size_t begin, end; std::string src, tgt; };
    std::vector<float> timeline;
    std::vector<Seg> segs;
    {
        std::ifstream f(script);
        std::string wav, src, tgt;
        while (f >> wav >> src >> tgt) {
            std::vector<float> pcm;
            if (!read_wav16k(wav, &pcm)) { std::fprintf(stderr, "bad wav %s\n", wav.c_str()); return 1; }
            segs.push_back({timeline.size(), timeline.size() + pcm.size(), src, tgt});
            timeline.insert(timeline.end(), pcm.begin(), pcm.end());
            timeline.insert(timeline.end(), (size_t)gap_ms * 16, 0.f);
        }
    }
    auto seg_at = [&](size_t sample) -> const Seg & {
        for (auto & s : segs) if (sample < s.end + (size_t)gap_ms * 16) return s;
        return segs.back();
    };

    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr); }, nullptr);
    AsrEngine asr;
    MtEngine mt;
    if (!asr.load(asr_gguf.c_str(), mmproj.c_str(), n_threads)) { std::fprintf(stderr, "ASR load failed\n"); return 1; }
    if (!mt.load(mt_gguf.c_str(), n_threads)) { std::fprintf(stderr, "MT load failed\n"); return 1; }
    std::map<std::string, std::unique_ptr<Tts>> tts;
    auto voice_for = [&](const std::string & lang) -> Tts * {
        static const std::map<std::string, std::string> v = {
            {"en", "en_US-lessac-medium"}, {"vi", "vi_VN-vais1000-medium"},
            {"zh", "zh_CN-huayan-medium"}, {"ko", "ko_KR-kss-medium"}};
        auto & p = tts[lang];
        if (!p) { const std::string b = voices_dir + "/" + v.at(lang); p = Tts::create(b + ".onnx", b + ".onnx.json", n_threads, espeak); }
        return p.get();
    };
    for (auto & s : segs) voice_for(s.tgt);   // load voices up front, not on the clock
    { std::string how; for (auto & s : segs) mt.prepare(s.tgt, &how); }   // warm every target's prefix
    {
        // First inference on a fresh Adreno context compiles kernels (~4 s); pay it
        // here, not on the first utterance. One short clip per source language,
        // and one translation, so every code path has run once.
        std::map<std::string, bool> seen;
        const double tw = now_ms();
        for (auto & s : segs) {
            if (seen[s.src]) continue;
            seen[s.src] = true;
            asr.run(std::vector<float>(timeline.begin() + s.begin,
                                       timeline.begin() + std::min(s.end, s.begin + 16000)), s.src);
        }
        std::string how;
        mt.prepare(segs[0].tgt, &how);
        mt.translate("warm up", segs[0].tgt, [](const std::string &) {});
        voice_for(segs[0].tgt)->synth_text("warm up", segs[0].tgt);
        std::printf("warm-up %.0f ms\n", now_ms() - tw);
    }
    std::printf("loaded: %zu items, %.1f s timeline, voices %zu\n", segs.size(), timeline.size() / 16000.0, tts.size());

    Queue<std::shared_ptr<AsrJob>> asr_q;
    Queue<std::shared_ptr<Utt>> mt_q;
    Queue<Clause> tts_q;
    std::vector<std::shared_ptr<Utt>> done;
    std::mutex done_m;
    std::vector<float> out_audio;   // 22.05 kHz, laid out on the timeline
    double play_until_ms = 0;

    g_t0 = clk::now();

    // ASR thread: one GPU, jobs in order; a job made stale before it starts is skipped
    std::thread asr_th([&] {
        std::shared_ptr<AsrJob> j;
        while (asr_q.pop(&j)) {
            if (j->stale) { j->done = true; continue; }
            j->t_start = now_ms();
            j->text = asr.run(j->audio, j->src);
            j->t_done = now_ms();
            j->done = true;
        }
    });

    // MT thread: utterances in order; waits for its transcript, swaps the prompt
    // KV to the target, streams clauses to TTS
    std::thread mt_th([&] {
        std::shared_ptr<Utt> u;
        while (mt_q.pop(&u)) {
            while (!u->job->done) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            u->t_mt_start = now_ms();
            u->mt_swap_ms = mt.prepare(u->tgt, &u->mt_how);
            ClauseChunker ch;
            std::string c;
            u->mt_text = mt.translate(u->job->text, u->tgt, [&](const std::string & piece) {
                if (u->t_mt_first < 0) u->t_mt_first = now_ms();
                std::string d = piece;
                while (ch.push(d, &c)) {   // push("") rescans for a further clause
                    d.clear();
                    if (u->t_clause1 < 0) u->t_clause1 = now_ms();
                    tts_q.push({u, clean_mt_text(c, u->tgt), false});
                }
            });
            if (ch.flush(&c)) { if (u->t_clause1 < 0) u->t_clause1 = now_ms(); tts_q.push({u, clean_mt_text(c, u->tgt), false}); }
            tts_q.push({u, "", true});
        }
        tts_q.close();
    });

    // TTS thread: synthesise each clause, queue it on the output timeline
    std::thread tts_th([&] {
        Clause c;
        while (tts_q.pop(&c)) {
            if (c.last) { std::lock_guard<std::mutex> l(done_m); done.push_back(c.u); continue; }
            if (c.text.empty()) continue;
            TtsResult r = voice_for(c.u->tgt)->synth_text(c.text, c.u->tgt);
            if (!r.ok) continue;
            const double t = now_ms();
            if (c.u->t_voice1 < 0) c.u->t_voice1 = t;
            // a speaker plays clauses back to back from when each one exists
            const double start = std::max(t, play_until_ms);
            const size_t at = (size_t)(start * 22.05);
            if (out_audio.size() < at + r.pcm.size()) out_audio.resize(at + r.pcm.size(), 0.f);
            std::copy(r.pcm.begin(), r.pcm.end(), out_audio.begin() + at);
            play_until_ms = start + r.pcm.size() / 22.05;
        }
    });

    // capture: replay the timeline in 20 ms frames on the wall clock
    Vad vad(vcfg);
    const size_t frame = 320;
    int uid = 0;
    std::shared_ptr<AsrJob> spec;          // speculative job for the open utterance
    double spec_last = 0; int n_stale = 0;
    for (size_t o = 0; o < timeline.size(); o += frame) {
        const double t_frame_end = (o + frame) / 16.0;
        while (now_ms() < t_frame_end) std::this_thread::sleep_for(std::chrono::microseconds(500));
        std::vector<float> part(timeline.begin() + o, timeline.begin() + std::min(timeline.size(), o + frame));
        auto ev = vad.feed(part, o / 16.0);

        if (vad.speaking()) {
            const int sil = vad.silence_ms();
            if (!spec && spec_ms > 0 && sil >= spec_ms) {
                // quiet long enough: start decoding now, inside the hangover
                spec = std::make_shared<AsrJob>();
                spec->uid = uid; spec->src = seg_at(o).src; spec->audio = vad.open_audio();
                spec_last = t_frame_end - sil;
                asr_q.push(spec);
            } else if (spec && sil == 0) {
                spec->stale = true; spec.reset(); ++n_stale;   // speech resumed
            }
        }
        for (auto & e : ev) {
            auto u = std::make_shared<Utt>();
            u->uid = uid++;
            const Seg & s = seg_at((size_t)(e.t_start_ms * 16));
            u->src = s.src; u->tgt = s.tgt;
            u->audio_s = e.audio.size() / sizeof(float) / 16000.0;
            u->t_last = e.t_last_ms; u->t_detect = e.t_detect_ms; u->n_stale = n_stale;
            if (spec && !spec->stale) {
                u->job = spec; u->speculative = true;
            } else {
                u->job = std::make_shared<AsrJob>();
                u->job->uid = u->uid; u->job->src = u->src;
                u->job->audio.assign((const float *)e.audio.data(), (const float *)e.audio.data() + e.audio.size() / sizeof(float));
                asr_q.push(u->job);
            }
            spec.reset(); n_stale = 0;
            mt_q.push(u);
        }
    }
    asr_q.close(); mt_q.close();
    asr_th.join(); mt_th.join(); tts_th.join();
    (void)spec_last;

    // report
    std::ofstream jo(out_jsonl);
    std::vector<double> m1;
    std::sort(done.begin(), done.end(), [](auto & a, auto & b) { return a->uid < b->uid; });
    for (auto & u : done) {
        auto rel = [&](double t) { return t < 0 ? -1.0 : t - u->t_last; };
        jo << "{\"uid\":" << u->uid << ",\"src\":\"" << u->src << "\",\"tgt\":\"" << u->tgt << "\""
           << ",\"audio_s\":" << u->audio_s << ",\"vad_ms\":" << (u->t_detect - u->t_last)
           << ",\"speculative\":" << (u->speculative ? 1 : 0) << ",\"n_stale\":" << u->n_stale
           << ",\"asr_wait_ms\":" << rel(u->job->t_start) << ",\"asr_done_ms\":" << rel(u->job->t_done)
           << ",\"mt_how\":\"" << u->mt_how << "\",\"mt_swap_ms\":" << u->mt_swap_ms
           << ",\"mt1_ms\":" << rel(u->t_mt_first) << ",\"clause1_ms\":" << rel(u->t_clause1)
           << ",\"voice1_ms\":" << rel(u->t_voice1)
           << ",\"asr_text\":\"" << json_escape(u->job->text) << "\",\"mt_text\":\"" << json_escape(u->mt_text) << "\"}\n";
        if (u->t_mt_first >= 0) m1.push_back(rel(u->t_mt_first));
        std::printf("[%2d] %s->%s %4.1fs  spec=%d  asr_done %5.0f  swap %-7s %4.0f  mt1 %5.0f  voice1 %5.0f  | %.50s\n",
                    u->uid, u->src.c_str(), u->tgt.c_str(), u->audio_s, (int)u->speculative,
                    rel(u->job->t_done), u->mt_how.c_str(), u->mt_swap_ms, rel(u->t_mt_first),
                    rel(u->t_voice1), u->job->text.c_str());
    }
    if (!m1.empty()) {
        std::sort(m1.begin(), m1.end());
        std::printf("last speech -> first MT token: median %.0f  p90 %.0f  max %.0f  (>2s: %d/%zu)\n",
                    m1[m1.size() / 2], m1[m1.size() * 9 / 10], m1.back(),
                    (int)std::count_if(m1.begin(), m1.end(), [](double x) { return x > 2000; }), m1.size());
    }
    // output speech on the same clock
    {
        std::ofstream f(out_wav, std::ios::binary);
        auto w32 = [&](uint32_t v) { f.write((char *)&v, 4); };
        auto w16 = [&](uint16_t v) { f.write((char *)&v, 2); };
        uint32_t db = out_audio.size() * 2;
        f.write("RIFF", 4); w32(36 + db); f.write("WAVE", 4); f.write("fmt ", 4); w32(16); w16(1); w16(1);
        w32(22050); w32(44100); w16(2); w16(16); f.write("data", 4); w32(db);
        for (float s : out_audio) { int16_t v = (int16_t)(std::max(-1.f, std::min(1.f, s)) * 32767); f.write((char *)&v, 2); }
    }
    for (auto & v : tts) std::printf("MT prompt state %s: %zu KB\n", v.first.c_str(), mt.saved_bytes(v.first) / 1024);
    llama_backend_free();
    return 0;
}
