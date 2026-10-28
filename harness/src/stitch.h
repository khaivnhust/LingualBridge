// stitch.h -- transcript stitching for overlapped / sliding-window ASR.
//
// A real overlapped decoder transcribes overlapping windows and must combine
// them. Decoding audio-so-far instead (what the first emulation did) avoids
// stitching entirely, but costs O(D^2/step) of wasted GPU time -- for a 5 s
// utterance at 1 s triggers that is ~4 s of redundant decoding.
//
// The merge here is the standard conservative one: find the largest k such that
// the last k committed WORDS equal the first k words of the incoming segment,
// drop them, and append the remainder. It never invents text and never drops
// committed words, so a failure mode is a duplicated or dropped boundary word,
// not a hallucinated sentence.
//
// Header-only so it can be unit-tested without a model or a device.
#pragma once
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace s2st {

// Split on ASCII/Unicode whitespace. Syllable scripts (ko, zh, ja) have no
// spaces, so this degenerates to one token per clause for those languages --
// see note in `add`.
inline std::vector<std::string> split_words(const std::string & s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && !(s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r')) ++j;
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

// Character-level variant for scripts without word delimiters. Character k
// overlap is the honest analogue of word overlap when the ASR emits one
// continuous run of graphemes per clause.
inline std::vector<std::string> split_chars(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        size_t n = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0) ? 2 : ((c & 0xF0) == 0xE0) ? 3
                 : ((c & 0xF8) == 0xF0) ? 4 : 1;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

class StreamStitcher {
public:
    // `by_char` selects character-level overlap. Use it for ko/zh, where
    // espeak emits a single unbroken phoneme run and word splitting would
    // produce one enormous "word" that overlaps everything.
    explicit StreamStitcher(bool by_char = false) : by_char_(by_char) {}

    // Segment cut at silence: nothing is shared with the previous one.
    void append(const std::string & segment) {
        std::vector<std::string> seg = tok(segment);
        if (seg.empty()) return;
        for (auto & w : seg) committed_.push_back(w);
        ++segments_;
    }

    void add(const std::string & segment) {
        if (segment.empty()) return;
        std::vector<std::string> seg = tok(segment);
        if (seg.empty()) return;
        size_t k = overlap(seg);
        for (size_t i = k; i < seg.size(); ++i) committed_.push_back(seg[i]);
        ++segments_;
    }

    std::string text() const {
        std::string out;
        for (size_t i = 0; i < committed_.size(); ++i) {
            if (i) out += by_char_ ? "" : " ";
            out += committed_[i];
        }
        return out;
    }

    size_t segments() const { return segments_; }
    size_t tokens() const { return committed_.size(); }
    // Tokens appended by the most recent segment: how much NEW text each decode
    // contributed. Falls to 0 when a segment is wholly contained in the overlap,
    // which is the signal that the window is too short.
    size_t last_gain() const { return last_gain_; }
    void   set_last_gain(size_t g) { last_gain_ = g; }

private:
    std::vector<std::string> tok(const std::string & s) const {
        return by_char_ ? split_chars(s) : split_words(s);
    }

    // Case and ASCII punctuation are ignored when matching: a window that ends
    // mid-sentence gets a period and the next one a capital ("laundering." /
    // "Laundering"), and exact matching then finds no overlap at all.
    static std::string key(const std::string & w) {
        std::string o;
        for (unsigned char c : w)
            if (!(c < 0x80 && std::ispunct(c))) o += (char)std::tolower(c);
        return o;
    }

    // largest k with committed[-k:] == seg[:k]
    size_t overlap(const std::vector<std::string> & seg) const {
        size_t maxk = std::min(committed_.size(), seg.size());
        for (size_t k = maxk; k > 0; --k) {
            bool eq = true;
            for (size_t i = 0; i < k; ++i)
                if (key(committed_[committed_.size() - k + i]) != key(seg[i])) { eq = false; break; }
            if (eq) return k;
        }
        return 0;
    }

    bool by_char_;
    std::vector<std::string> committed_;
    size_t segments_ = 0, last_gain_ = 0;
};

} // namespace s2st
