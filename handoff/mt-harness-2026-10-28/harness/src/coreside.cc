// coreside.cc -- can ASR (Adreno/OpenCL) and MT (Hexagon/HTP) live in ONE process?
//
// The harness runs two llama-server processes because OpenCL + HTP in a single
// process was never measured, and a monolithic binary depends on it. This loads
// both models through libllama/libmtmd with each pinned to its device, then:
//   1. runs ASR and MT back to back (compare to the two-server numbers), and
//   2. runs them CONCURRENTLY on two threads -- the overlap a pipelined
//      monolith needs (ASR of utterance N+1 while MT of N is decoding).
// Reports wall times, prefill/decode split, and VmRSS at each stage.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string k;
    long v = -1;
    while (f >> k) {
        if (k == "VmRSS:") { f >> v; break; }
    }
    return v;
}

static bool read_wav16k(const std::string & path, std::vector<float> * out) {
    std::ifstream f(path, std::ios::binary);
    std::string a((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (a.size() < 44 || a.compare(0, 4, "RIFF")) return false;
    size_t p = 12;
    while (p + 8 <= a.size()) {
        uint32_t sz; std::memcpy(&sz, &a[p + 4], 4);
        if (a.compare(p, 4, "data") == 0) {
            size_t n = std::min<size_t>(sz, a.size() - p - 8) / 2;
            out->resize(n);
            for (size_t i = 0; i < n; ++i) {
                int16_t s; std::memcpy(&s, &a[p + 8 + 2 * i], 2);
                (*out)[i] = s / 32768.0f;
            }
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

struct Gen { std::string text; double prefill_ms = 0, first_ms = 0, total_ms = 0; int n_tok = 0; };

// greedy continuation from whatever is already in the KV cache
static void decode_greedy(llama_context * ctx, const llama_vocab * voc, int max_tok, Gen * g,
                          clk::time_point t0) {
    llama_sampler * s = llama_sampler_init_greedy();
    for (int i = 0; i < max_tok; ++i) {
        llama_token t = llama_sampler_sample(s, ctx, -1);
        if (i == 0) g->first_ms = ms(t0, clk::now());
        if (llama_vocab_is_eog(voc, t)) break;
        char buf[256];
        int n = llama_token_to_piece(voc, t, buf, sizeof buf, 0, true);
        if (n > 0) g->text.append(buf, n);
        ++g->n_tok;
        if (llama_decode(ctx, llama_batch_get_one(&t, 1))) break;
    }
    llama_sampler_free(s);
    g->total_ms = ms(t0, clk::now());
}

struct Asr {
    llama_model * model = nullptr; llama_context * ctx = nullptr; mtmd_context * mt = nullptr;
    const llama_vocab * voc = nullptr;
    Gen run(const std::vector<float> & pcm, const std::string & lang) {
        Gen g;
        auto t0 = clk::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        std::string p = std::string("<|im_start|>user\n") + mtmd_default_marker() +
                        "<|im_end|>\n<|im_start|>assistant\nlanguage " + lang + "<asr_text>";
        mtmd_bitmap * bm = mtmd_bitmap_init_from_audio(pcm.size(), pcm.data());
        mtmd_input_text txt{p.c_str(), p.size(), false, true};
        mtmd_input_chunks * ch = mtmd_input_chunks_init();
        const mtmd_bitmap * bms[] = {bm};
        mtmd_tokenize(mt, ch, &txt, bms, 1);
        llama_pos n_past = 0;
        mtmd_helper_eval_chunks(mt, ctx, ch, 0, 0, llama_n_batch(ctx), true, &n_past);
        g.prefill_ms = ms(t0, clk::now());
        mtmd_input_chunks_free(ch);
        mtmd_bitmap_free(bm);
        decode_greedy(ctx, voc, 128, &g, t0);
        return g;
    }
};

struct Mt {
    llama_model * model = nullptr; llama_context * ctx = nullptr; const llama_vocab * voc = nullptr;
    Gen run(const std::string & src, const std::string & tgt_name) {
        Gen g;
        auto t0 = clk::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        std::string user = "Translate the following text into " + tgt_name +
            ". Note that you should only output the translated result without any additional "
            "explanation:\n\n" + src;
        llama_chat_message m{"user", user.c_str()};
        std::vector<char> buf(user.size() * 2 + 1024);
        int n = llama_chat_apply_template(llama_model_chat_template(model, nullptr), &m, 1, true,
                                          buf.data(), buf.size());
        std::vector<llama_token> tok(n + 16);
        int nt = llama_tokenize(voc, buf.data(), n, tok.data(), tok.size(), true, true);
        tok.resize(nt);
        llama_decode(ctx, llama_batch_get_one(tok.data(), nt));
        g.prefill_ms = ms(t0, clk::now());
        decode_greedy(ctx, voc, 64, &g, t0);
        return g;
    }
};

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s ASR.gguf MMPROJ.gguf MT.gguf clip.wav [iters]\n", argv[0]);
        return 2;
    }
    const int iters = argc > 5 ? std::atoi(argv[5]) : 4;
    std::vector<float> pcm;
    if (!read_wav16k(argv[4], &pcm)) { std::fprintf(stderr, "bad wav\n"); return 1; }

    llama_backend_init();
    std::printf("devices:");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        std::printf(" %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
    std::printf("\nrss start %ld MB\n", rss_kb() / 1024);

    // ── ASR on the Adreno ──
    Asr asr;
    {
        ggml_backend_dev_t ds[] = {dev("GPUOpenCL"), nullptr};
        auto mp = llama_model_default_params();
        mp.devices = ds; mp.n_gpu_layers = 99;
        asr.model = llama_model_load_from_file(argv[1], mp);
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048; cp.n_batch = 2048;
        asr.ctx = llama_init_from_model(asr.model, cp);
        asr.voc = llama_model_get_vocab(asr.model);
        auto xp = mtmd_context_params_default();
        xp.use_gpu = true; xp.device = dev("GPUOpenCL"); xp.print_timings = false;
        asr.mt = mtmd_init_from_file(argv[2], asr.model, xp);
        if (!asr.model || !asr.ctx || !asr.mt) { std::fprintf(stderr, "ASR load failed\n"); return 1; }
    }
    std::printf("rss after ASR load %ld MB\n", rss_kb() / 1024);

    // ── MT on the Hexagon NPU ──
    Mt mt;
    {
        ggml_backend_dev_t ds[] = {dev("HTP0"), nullptr};
        auto mp = llama_model_default_params();
        mp.devices = ds; mp.n_gpu_layers = 99;
        mt.model = llama_model_load_from_file(argv[3], mp);
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096;
        mt.ctx = llama_init_from_model(mt.model, cp);
        mt.voc = llama_model_get_vocab(mt.model);
        if (!mt.model || !mt.ctx) { std::fprintf(stderr, "MT load failed\n"); return 1; }
    }
    std::printf("rss after MT load %ld MB\n", rss_kb() / 1024);

    auto show = [](const char * tag, const Gen & g) {
        std::printf("  %-14s prefill %6.0f  first %6.0f  total %6.0f ms  %2d tok  %5.1f ms/tok  | %.70s\n",
                    tag, g.prefill_ms, g.first_ms, g.total_ms, g.n_tok,
                    g.n_tok > 1 ? (g.total_ms - g.first_ms) / (g.n_tok - 1) : 0.0, g.text.c_str());
    };

    std::printf("\n== sequential (ASR then MT), %d iters ==\n", iters);
    std::string asr_text;
    for (int i = 0; i < iters; ++i) {
        Gen a = asr.run(pcm, "English");
        Gen m = mt.run(a.text, "Vietnamese");
        show("ASR", a); show("MT", m);
        asr_text = a.text;
    }
    std::printf("rss after sequential %ld MB\n", rss_kb() / 1024);

    std::printf("\n== concurrent (ASR and MT on two threads), %d iters ==\n", iters);
    for (int i = 0; i < iters; ++i) {
        Gen a, m;
        auto t0 = clk::now();
        std::thread ta([&] { a = asr.run(pcm, "English"); });
        std::thread tm([&] { m = mt.run(asr_text, "Vietnamese"); });
        ta.join(); tm.join();
        show("ASR (conc)", a); show("MT (conc)", m);
        std::printf("  wall for both %6.0f ms\n", ms(t0, clk::now()));
    }
    std::printf("rss end %ld MB\n", rss_kb() / 1024);

    mtmd_free(asr.mt);
    llama_free(asr.ctx); llama_model_free(asr.model);
    llama_free(mt.ctx);  llama_model_free(mt.model);
    llama_backend_free();
    std::printf("clean exit\n");
    return 0;
}
