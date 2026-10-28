// nanprobe.cc -- root-causing NaN windows of Qwen3-ASR on the Hexagon NPU (WORKLOG §18).
//
// Symptom (llama-server, decoder on HTP0, audio encoder on CPU): every ~100-200
// requests, a run of 10-20 consecutive requests emit NaN logits ("@@@..." to the
// token cap), then it recovers by itself. Same clips are fine in other passes.
//
// This runs the same ASR in-process over a clip list, several passes, and checks
// the logits directly, under one variable at a time:
//   --clear-data   zero the KV memory between requests (llama_memory_clear(.., true));
//                  default frees the cells but keeps their contents, as llama-server does
//   --no-fa        disable flash attention
//   --sleep-ms N   pause between requests (keeps the NPU cooler)
// Each line carries a wall-clock time and the hottest NSP (NPU) thermal zone.
// Prints one line per request: index, clip, NaN-at-first-token, n_prompt, n_gen.
#include <algorithm>
#include <chrono>
#include <ctime>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

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

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s ASR.gguf MMPROJ.gguf CLIPLIST [passes=3] [--clear-data] [--no-fa] [--sleep-ms N]\n", argv[0]);
        return 2;
    }
    int passes = 3, sleep_ms = 0; bool clear_data = false, no_fa = false;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--clear-data")) clear_data = true;
        else if (!std::strcmp(argv[i], "--no-fa")) no_fa = true;
        else if (!std::strcmp(argv[i], "--sleep-ms") && i + 1 < argc) sleep_ms = std::atoi(argv[++i]);
        else passes = std::atoi(argv[i]);
    }
    struct Clip { std::string name, lang; std::vector<float> pcm; };
    std::vector<Clip> clips;
    {
        std::ifstream f(argv[3]);
        std::string path, lang;
        while (f >> path >> lang) {
            Clip c; c.name = path; c.lang = lang;
            if (read_wav16k(path, &c.pcm)) clips.push_back(std::move(c));
        }
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_WARN) std::fputs(t, stderr); }, nullptr);
    ggml_backend_dev_t htp = ggml_backend_dev_by_name("HTP0");
    ggml_backend_dev_t ds[] = {htp, nullptr};
    auto mp = llama_model_default_params(); mp.devices = ds; mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params(); cp.n_ctx = 2048; cp.n_batch = 2048;
    if (no_fa) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * voc = llama_model_get_vocab(model);
    auto xp = mtmd_context_params_default();
    xp.use_gpu = false; xp.print_timings = false;   // audio encoder on CPU (on HTP it is broken)
    mtmd_context * mt = mtmd_init_from_file(argv[2], model, xp);
    if (!model || !ctx || !mt) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::printf("# clips %zu passes %d clear_data %d flash_attn %s\n", clips.size(), passes,
                (int)clear_data, no_fa ? "off" : "auto");

    // hottest NSP zone, read the same way templog.sh does
    std::vector<std::string> nsp_zones;
    for (int z = 0; z < 200; ++z) {
        std::ifstream t("/sys/class/thermal/thermal_zone" + std::to_string(z) + "/type");
        std::string ty;
        if (t >> ty && ty.rfind("nsp", 0) == 0)
            nsp_zones.push_back("/sys/class/thermal/thermal_zone" + std::to_string(z) + "/temp");
    }
    auto nsp_max = [&]() {
        long m = 0;
        for (auto & z : nsp_zones) { std::ifstream f(z); long v = 0; if (f >> v) m = std::max(m, v); }
        return m / 1000.0;
    };
    auto clock = []() {
        char b[16]; std::time_t t = std::time(nullptr); std::strftime(b, sizeof b, "%H:%M:%S", std::localtime(&t));
        return std::string(b);
    };

    const char * names[] = {"English", "Vietnamese", "Chinese", "Korean"};
    const char * codes[] = {"en", "vi", "zh", "ko"};
    int idx = 0, n_bad = 0;
    for (int pass = 1; pass <= passes; ++pass) {
        for (auto & c : clips) {
            const char * lname = "English";
            for (int k = 0; k < 4; ++k) if (c.lang == codes[k]) lname = names[k];
            llama_memory_clear(llama_get_memory(ctx), clear_data);
            std::string p = std::string("<|im_start|>user\n") + mtmd_default_marker() +
                            "<|im_end|>\n<|im_start|>assistant\nlanguage " + lname + "<asr_text>";
            mtmd_bitmap * bm = mtmd_bitmap_init_from_audio(c.pcm.size(), c.pcm.data());
            mtmd_input_text txt{p.c_str(), p.size(), false, true};
            mtmd_input_chunks * ch = mtmd_input_chunks_init();
            const mtmd_bitmap * bms[] = {bm};
            mtmd_tokenize(mt, ch, &txt, bms, 1);
            const size_t n_prompt = mtmd_helper_get_n_tokens(ch);
            llama_pos n_past = 0;
            mtmd_helper_eval_chunks(mt, ctx, ch, 0, 0, llama_n_batch(ctx), true, &n_past);
            const float * lg = llama_get_logits_ith(ctx, -1);
            int n_nan = 0;
            for (int v = 0; v < llama_vocab_n_tokens(voc); ++v) if (!std::isfinite(lg[v])) ++n_nan;
            // a short greedy decode, to see whether decode steps go bad too
            llama_sampler * s = llama_sampler_init_greedy();
            int n_gen = 0, n_nan_steps = 0;
            std::string out;
            for (; n_gen < 64; ++n_gen) {
                const float * l = llama_get_logits_ith(ctx, -1);
                if (!std::isfinite(l[0])) ++n_nan_steps;
                llama_token t = llama_sampler_sample(s, ctx, -1);
                if (llama_vocab_is_eog(voc, t)) break;
                char buf[128]; int k = llama_token_to_piece(voc, t, buf, sizeof buf, 0, false);
                if (k > 0) out.append(buf, k);
                if (llama_decode(ctx, llama_batch_get_one(&t, 1))) break;
            }
            llama_sampler_free(s);
            mtmd_input_chunks_free(ch);
            mtmd_bitmap_free(bm);
            const bool bad = n_nan > 0 || n_nan_steps > 0;
            n_bad += bad;
            if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            std::printf("%s nsp %5.1f %4d p%d %-26s %s prompt %4zu gen %3d nan_logits %6d nan_steps %3d | %.40s\n",
                        clock().c_str(), nsp_max(), ++idx, pass, c.name.c_str(), bad ? "BAD" : "ok ", n_prompt, n_gen, n_nan,
                        n_nan_steps, out.c_str());
            std::fflush(stdout);
        }
    }
    std::printf("# done: %d bad of %d\n", n_bad, idx);
    mtmd_free(mt); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
