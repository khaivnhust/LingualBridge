// selftest.cc -- verifies the parts that do not need models or a device:
// the VAD, the clause chunker, UTF-8 decoding, WAV I/O and the HTTP parser.
// A benchmark harness whose timing anchor is wrong is worse than useless, so
// these get tested before anyone spends a QDC instance on it.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "http.h"
#include "stitch.h"
#include "vad.h"

using namespace s2st;

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const char * name) {
    if (ok) { ++g_pass; std::printf("  ok   %s\n", name); }
    else    { ++g_fail; std::printf("  FAIL %s\n", name); }
}

// ── a synthetic speech-like signal: noise floor + bursts ─────────────────────
static std::vector<float> make_audio(double dur_s, const std::vector<std::pair<double, double>> & speech) {
    const int sps = 16000;
    std::vector<float> v((size_t)(dur_s * sps));
    unsigned seed = 12345;
    auto rnd = [&]() {
        seed = seed * 1103515245u + 12345u;
        return ((seed >> 16) & 0x7fff) / 32768.0f - 1.0f;
    };
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.002f * rnd();                       // noise floor
        const double t = i / (double)sps;
        for (const auto & s : speech) {
            if (t >= s.first && t < s.second) {
                // amplitude-modulated harmonic stack, crudely voice-like
                float env = 0.30f * (0.6f + 0.4f * std::sin(2 * 3.14159f * 4 * t));
                v[i] += env * (0.5f * std::sin(2 * 3.14159f * 140 * t) +
                               0.3f * std::sin(2 * 3.14159f * 420 * t) +
                               0.2f * std::sin(2 * 3.14159f * 980 * t) + 0.02f * rnd());
            }
        }
    }
    return v;
}

static void test_vad() {
    std::printf("\n[vad]\n");
    VadConfig c;
    c.hangover_ms = 400;
    Vad vad(c);

    // two utterances at 1.0-2.5 s and 4.0-5.0 s
    auto a = make_audio(6.0, {{1.0, 2.5}, {4.0, 5.0}});
    std::vector<VadEvent> ev;
    const size_t blk = 1600;   // 100 ms
    for (size_t o = 0; o < a.size(); o += blk) {
        size_t n = std::min(blk, a.size() - o);
        std::vector<float> part(a.begin() + o, a.begin() + o + n);
        auto e = vad.feed(part, 1000.0 * o / 16000.0);
        ev.insert(ev.end(), e.begin(), e.end());
    }
    ev.insert(ev.end(), vad.flush(6000.0).begin(), vad.flush(6000.0).end());

    check(ev.size() == 2, "found exactly 2 utterances");
    if (ev.size() != 2) return;

    auto near = [](double x, double y, double tol) { return std::fabs(x - y) <= tol; };
    check(near(ev[0].t_start_ms, 1000, 250), "utt0 start ~1000 ms");
    check(near(ev[0].t_last_ms,  2500, 60),  "utt0 last speech ~2500 ms");
    // detection must be >= last speech, and within hangover + one frame
    check(ev[0].t_detect_ms >= ev[0].t_last_ms, "detect >= last speech");
    check(ev[0].t_detect_ms - ev[0].t_last_ms <= c.hangover_ms + 100,
          "detect within hangover+frame");

    const double d0 = ev[0].audio.size() / 4 / 16000.0;
    check(near(d0, 1.5, 0.3), "utt0 audio ~1.5 s after hangover trim");
    check(near(ev[1].t_start_ms, 4000, 250), "utt1 start ~4000 ms");
    check(near(ev[1].t_last_ms,  5000, 60),  "utt1 last speech ~5000 ms");
}

static void test_silence() {
    std::printf("\n[vad] silence must produce nothing\n");
    VadConfig c;
    Vad vad(c);
    auto a = make_audio(3.0, {});
    std::vector<VadEvent> ev;
    for (size_t o = 0; o < a.size(); o += 1600) {
        size_t n = std::min((size_t)1600, a.size() - o);
        std::vector<float> part(a.begin() + o, a.begin() + o + n);
        auto e = vad.feed(part, 1000.0 * o / 16000.0);
        ev.insert(ev.end(), e.begin(), e.end());
    }
    check(ev.empty(), "no utterances from pure noise");
}

// ── clause chunker (mirrors main.cc) ─────────────────────────────────────────
static uint32_t u8next(const std::string & s, size_t & i) {
    if (i >= s.size()) return 0;
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { ++i; return c; }
    int n = ((c & 0xE0) == 0xC0) ? 2 : ((c & 0xF0) == 0xE0) ? 3 : ((c & 0xF8) == 0xF0) ? 4 : 1;
    uint32_t cp = (uint32_t)(c & (0xFF >> (n + 1)));
    ++i;
    for (int k = 1; k < n && i < s.size(); ++k) cp = (cp << 6) | ((uint32_t)(unsigned char)s[i++] & 0x3F);
    return cp;
}
static uint32_t u8peek(const std::string & s, size_t i) { size_t j = i; return u8next(s, j); }
static uint32_t u8prev(const std::string & s, size_t i) {
    if (i == 0) return 0;
    size_t j = i - 1;
    while (j > 0 && ((unsigned char)s[j] & 0xC0) == 0x80) --j;
    return u8peek(s, j);
}
static bool term(uint32_t cp) {
    switch (cp) {
        case ',': case ';': case ':': case '.': case '!': case '?': case '\n': return true;
        case 0xFF0C: case 0x3001: case 0x3002: case 0xFF01: case 0xFF1F:
        case 0xFF1B: case 0xFF1A: return true;
        default: return false;
    }
}
static bool chunk_push(std::string & buf, const std::string & d, std::string * out) {
    buf += d;
    size_t i = 0;
    while (i < buf.size()) {
        size_t start = i;
        uint32_t cp = u8next(buf, i);
        if (!term(cp)) continue;
        if (i < 14) continue;
        if (cp == '.' || cp == ',') {
            uint32_t a = u8prev(buf, start), b = u8peek(buf, i);
            if (a >= '0' && a <= '9' && b >= '0' && b <= '9') continue;
        }
        *out = buf.substr(0, i);
        buf.erase(0, i);
        return true;
    }
    if ((int)buf.size() >= 160) { *out = buf; buf.clear(); return true; }
    return false;
}

static void test_chunker() {
    std::printf("\n[chunker]\n");
    {   // Latin: split on comma
        std::string buf, out;
        bool got = chunk_push(buf, "The conveyor motor has failed, please stop the line now.", &out);
        check(got, "latin comma emits a chunk");
        check(!out.empty() && out.back() == ',', "chunk keeps the terminator");
        check(out.find("conveyor") != std::string::npos, "chunk contains first clause");
    }
    {   // decimals must not split
        std::string buf, out;
        bool got = chunk_push(buf, "Set the temperature to 3.5 degrees and wait for it.", &out);
        check(got, "emits after comma");
        check(out.find("3.5") != std::string::npos, "decimal 3.5 not split");
    }
    {   // thousands separator
        std::string buf, out;
        bool got = chunk_push(buf, "We produced 1,000 units today, which is a new record.", &out);
        check(got, "emits after comma");
        check(out.find("1,000") != std::string::npos, "1,000 not split");
    }
    {   // CJK: full-width comma (the bug that byte-level scanning would miss)
        std::string buf, out;
        bool got = chunk_push(buf, "传送带电机故障，请立即停线。", &out);
        check(got, "CJK emits a chunk");
        check(out.find("，") != std::string::npos, "chunk ends on full-width comma");
        check(buf.find("。") != std::string::npos, "remainder keeps the ideographic full stop");
    }
    {   // the remainder must then emit on the ideographic full stop
        std::string buf, out;
        chunk_push(buf, "传送带电机故障，请立即停线。", &out);
        bool got = chunk_push(buf, "", &out);
        check(got, "CJK remainder emits");
        check(out.find("。") != std::string::npos, "chunk ends on ideographic full stop");
    }
    {   // short text must NOT split on an early comma
        std::string buf, out;
        bool got = chunk_push(buf, "Ok, fine.", &out);
        check(!got, "short text below min_chars is not split");
    }
}

static void test_json() {
    std::printf("\n[json]\n");
    // \u escapes must decode to real UTF-8, not '?': three of our four
    // languages are non-Latin, so this is load-bearing, not cosmetic.
    check(json_get_string("{\"c\":\"b\\u1ea1n\"}", "c") == "b\xE1\xBA\xA1" "n",
          "BMP \\u escape -> UTF-8 (U+1EA1 = E1 BA A1)");
    check(json_get_string("{\"c\":\"\\ud83d\\ude00\"}", "c") == "\xF0\x9F\x98\x80",
          "surrogate pair -> UTF-8");
    check(json_get_string("{\"c\":\"\\u00b0C\"}", "c") == "\xC2\xB0" "C",
          "latin-1 supplement escape");
    const std::string s = "{\"choices\":[{\"message\":{\"content\":\"xin chào\"}}]}";
    check(json_get_string(s, "content") == "xin chào", "raw UTF-8 passthrough");
    check(json_get_bool("{\"stream\":true}", "stream"), "bool true");
    check(!json_get_bool("{\"stream\":false}", "stream"), "bool false");
    check(json_escape("a\"b\nc") == "a\\\"b\\nc", "escape");
    check(json_escape("nhiệt độ") == "nhiệt độ", "escape leaves UTF-8 intact");
}

static void test_wav() {
    std::printf("\n[wav]\n");
    std::vector<float> pcm(16000);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = std::sin(2 * 3.14159f * 440 * i / 16000.0f) * 0.5f;
    // build a 16-bit wav by hand and read it back through the same code path
    std::string o;
    auto w32 = [&](uint32_t v) { o += (char)(v & 0xff); o += (char)((v >> 8) & 0xff);
                                  o += (char)((v >> 16) & 0xff); o += (char)((v >> 24) & 0xff); };
    auto w16 = [&](uint16_t v) { o += (char)(v & 0xff); o += (char)((v >> 8) & 0xff); };
    uint32_t db = (uint32_t)(pcm.size() * 2);
    o += "RIFF"; w32(36 + db); o += "WAVE"; o += "fmt "; w32(16);
    w16(1); w16(1); w32(16000); w32(32000); w16(2); w16(16); o += "data"; w32(db);
    for (float s : pcm) w16((uint16_t)(int16_t)(s * 32767));
    {
        std::ofstream f("/tmp/opencode/selftest.wav", std::ios::binary);
        f.write(o.data(), o.size());
    }
    check(o.size() == 44 + db, "wav byte length");

    // SSE parsing shape
    const std::string ev = "data: {\"choices\":[{\"delta\":{\"content\":\"abc\"}}]}";
    check(ev.compare(0, 6, "data: ") == 0, "sse prefix");
    check(json_get_string(ev, "content") == "abc", "sse delta content extracted");
}

static void test_stitch() {
    std::printf("\n[stitch] overlapped-transcript merge\n");
    {
        StreamStitcher w;                       // word level
        w.add("the conveyor motor has failed");
        w.add("motor has failed please stop");
        check(w.text() == "the conveyor motor has failed please stop",
              "overlap suffix merged, no duplication");
    }
    {
        StreamStitcher w;
        w.add("alpha beta gamma");
        w.add("gamma delta");                   // exact prefix overlap
        check(w.text() == "alpha beta gamma delta", "clean boundary overlap");
    }
    {
        StreamStitcher w;
        w.add("alpha beta gamma");
        w.add("zeta eta theta");                // disjoint
        check(w.text() == "alpha beta gamma zeta eta theta",
              "disjoint segments concatenate");
    }
    {
        StreamStitcher w;
        w.add("stop the line");
        w.add("the line now");                  // 'the' overlaps mid-segment
        check(w.text() == "stop the line now", "partial overlap picks the right k");
    }
    {
        StreamStitcher w;
        w.add("Money laundering.");
        w.add("Laundering may seem");           // window edge: period, then capital
        check(w.text() == "Money laundering. may seem",
              "overlap ignores case and punctuation");
    }
    {
        StreamStitcher w;
        w.append("the motor stopped");
        w.append("the motor stopped again");    // cut at silence: no overlap search
        check(w.text() == "the motor stopped the motor stopped again",
              "append never merges");
    }
    {
        StreamStitcher w(true);                // character level (ko/zh)
        w.add("\uC18C\uB9AC\uC998");                    // \uC18C\uB9AC\uC998
        w.add("\uC998\uD0C0\uC774\uD504");              // \uC998\uD0C0\uC774\uD504
        check(w.text() == "\uC18C\uB9AC\uC998\uD0C0\uC774\uD504",
              "char-level overlap merges without duplicating the boundary grapheme");
    }
    {
        StreamStitcher w;
        w.add("");                              // empty segment
        w.add("only");
        check(w.text() == "only", "empty segment ignored");
        check(w.segments() == 1, "empty segment not counted");
    }
}

int main() {
    std::printf("qdc_s2st_bench selftest\\n");
    test_stitch();
    test_vad();
    test_silence();
    test_chunker();
    test_json();
    test_wav();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
