// aaudio_test.cc -- can a native binary (no app, no Java) use the mic and speaker?
//
// Records N seconds through AAudio (NDK, talks to audioserver), prints the level
// every 100 ms -- the room's noise floor is what the energy VAD has to sit above --
// saves a WAV, then plays the recording back through the speaker. Reports what the
// stack actually granted (rate, format, burst, buffer, sharing/perf mode), since a
// low-latency request can silently come back as a deep-buffer stream.
//
//   aaudio_test [seconds=5] [out.wav=/data/local/tmp/aaudio_rec.wav] [--no-play]
#include <aaudio/AAudio.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static const char * perf_name(aaudio_performance_mode_t m) {
    return m == AAUDIO_PERFORMANCE_MODE_LOW_LATENCY ? "LOW_LATENCY"
         : m == AAUDIO_PERFORMANCE_MODE_POWER_SAVING ? "POWER_SAVING" : "NONE";
}

static AAudioStream * open(aaudio_direction_t dir, int rate) {
    AAudioStreamBuilder * b = nullptr;
    if (AAudio_createStreamBuilder(&b) != AAUDIO_OK) return nullptr;
    AAudioStreamBuilder_setDirection(b, dir);
    AAudioStreamBuilder_setSampleRate(b, rate);
    AAudioStreamBuilder_setChannelCount(b, 1);
    AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_I16);
    AAudioStreamBuilder_setPerformanceMode(b, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(b, AAUDIO_SHARING_MODE_SHARED);
    if (dir == AAUDIO_DIRECTION_INPUT)
        AAudioStreamBuilder_setInputPreset(b, AAUDIO_INPUT_PRESET_VOICE_RECOGNITION);
    else
        AAudioStreamBuilder_setUsage(b, AAUDIO_USAGE_MEDIA);
    AAudioStream * s = nullptr;
    aaudio_result_t r = AAudioStreamBuilder_openStream(b, &s);
    AAudioStreamBuilder_delete(b);
    if (r != AAUDIO_OK) {
        std::fprintf(stderr, "open %s failed: %s\n", dir == AAUDIO_DIRECTION_INPUT ? "input" : "output",
                     AAudio_convertResultToText(r));
        return nullptr;
    }
    std::printf("%-6s rate %d  ch %d  fmt %d  burst %d frames  buffer %d/%d frames  perf %s  sharing %s\n",
                dir == AAUDIO_DIRECTION_INPUT ? "input" : "output",
                AAudioStream_getSampleRate(s), AAudioStream_getChannelCount(s),
                AAudioStream_getFormat(s), AAudioStream_getFramesPerBurst(s),
                AAudioStream_getBufferSizeInFrames(s), AAudioStream_getBufferCapacityInFrames(s),
                perf_name(AAudioStream_getPerformanceMode(s)),
                AAudioStream_getSharingMode(s) == AAUDIO_SHARING_MODE_EXCLUSIVE ? "EXCLUSIVE" : "SHARED");
    return s;
}

static void wav_write(const std::string & p, const std::vector<int16_t> & x, int rate) {
    std::ofstream f(p, std::ios::binary);
    auto w32 = [&](uint32_t v) { f.write((char *)&v, 4); };
    auto w16 = [&](uint16_t v) { f.write((char *)&v, 2); };
    uint32_t db = x.size() * 2;
    f.write("RIFF", 4); w32(36 + db); f.write("WAVE", 4); f.write("fmt ", 4); w32(16); w16(1); w16(1);
    w32(rate); w32(rate * 2); w16(2); w16(16); f.write("data", 4); w32(db);
    f.write((const char *)x.data(), db);
}

int main(int argc, char ** argv) {
    double secs = argc > 1 ? std::atof(argv[1]) : 5.0;
    std::string out = argc > 2 ? argv[2] : "/data/local/tmp/aaudio_rec.wav";
    bool play = !(argc > 3 && std::strcmp(argv[3], "--no-play") == 0);
    const int rate = 16000;   // what the ASR wants; resampling (if any) is audioserver's job

    AAudioStream * in = open(AAUDIO_DIRECTION_INPUT, rate);
    if (!in) return 1;
    const int got = AAudioStream_getSampleRate(in);
    std::vector<int16_t> pcm((size_t)(secs * got));
    if (AAudioStream_requestStart(in) != AAUDIO_OK) { std::fprintf(stderr, "input start failed\n"); return 1; }

    auto t0 = std::chrono::steady_clock::now();
    size_t n = 0;
    const size_t hop = got / 10;   // 100 ms
    std::vector<double> lvl;
    long first_read_ms = -1;
    while (n < pcm.size()) {
        size_t want = std::min(hop, pcm.size() - n);
        aaudio_result_t r = AAudioStream_read(in, pcm.data() + n, want, 1000000000LL);
        if (r < 0) { std::fprintf(stderr, "read: %s\n", AAudio_convertResultToText(r)); break; }
        if (first_read_ms < 0 && r > 0)
            first_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
        n += r;
        if (n % hop == 0 && n > 0) {
            double acc = 0;
            for (size_t i = n - hop; i < n; ++i) acc += (double)pcm[i] * pcm[i];
            double db = 20 * std::log10(std::sqrt(acc / hop) / 32768.0 + 1e-9);
            lvl.push_back(db);
            int bars = std::max(0, (int)((db + 80) / 2));
            std::printf("  %5.1fs %6.1f dBFS %s\n", n / (double)got, db, std::string(bars, '#').c_str());
        }
    }
    AAudioStream_requestStop(in);
    std::printf("input xruns %d  first data after %ld ms\n", AAudioStream_getXRunCount(in), first_read_ms);
    AAudioStream_close(in);
    pcm.resize(n);
    wav_write(out, pcm, got);

    if (!lvl.empty()) {
        std::vector<double> s(lvl);
        std::sort(s.begin(), s.end());
        std::printf("level dBFS: min %.1f  p10 %.1f  median %.1f  p90 %.1f  max %.1f   -> %s\n",
                    s.front(), s[s.size() / 10], s[s.size() / 2], s[s.size() * 9 / 10], s.back(), out.c_str());
        std::printf("noise floor (p10) %.1f dBFS = %.2e linear rms; VAD absolute_floor is 3.0e-4 (%.1f dBFS)\n",
                    s[s.size() / 10], std::pow(10, s[s.size() / 10] / 20), 20 * std::log10(3.0e-4));
    }

    if (play) {
        AAudioStream * o = open(AAUDIO_DIRECTION_OUTPUT, got);
        if (!o) return 1;
        AAudioStream_requestStart(o);
        size_t w = 0;
        while (w < pcm.size()) {
            aaudio_result_t r = AAudioStream_write(o, pcm.data() + w, std::min<size_t>(hop, pcm.size() - w),
                                                   1000000000LL);
            if (r < 0) { std::fprintf(stderr, "write: %s\n", AAudio_convertResultToText(r)); break; }
            w += r;
        }
        // let the queued buffer drain before stopping
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        AAudioStream_requestStop(o);
        std::printf("played %.1f s, output xruns %d\n", w / (double)got, AAudioStream_getXRunCount(o));
        AAudioStream_close(o);
    }
    return 0;
}
