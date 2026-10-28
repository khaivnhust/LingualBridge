// vad.h -- energy VAD with an adaptive noise floor.
//
// The metric this harness exists to measure is "end of speech -> begin of
// voice". That is only well defined relative to *when the system decided the
// user stopped talking*, which is necessarily later than the last speech sample
// (you cannot know someone has finished mid-word). So every event carries both:
//
//   t_last_speech_ms : timestamp of the final speech-containing sample
//   t_detect_ms      : when hangover expired and the utterance was handed off
//
// and the harness reports BOTH latencies. The difference is the VAD's own
// detection penalty, which is real user-facing latency and must not be hidden
// inside an aggregate.
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace s2st {

struct VadConfig {
    int      frame_ms          = 20;      // analysis frame
    double   onset_ratio       = 3.0;     // speech when rms > floor * onset_ratio
    double   offset_ratio      = 1.8;     // silence when rms < floor * offset_ratio
    double   noise_floor_alpha = 0.002;   // EMA rate when in silence
    int      onset_frames      = 3;       // consecutive speech frames to open
    int      hangover_ms       = 250;     // silence before an utterance is closed
                                          // (450 was a safe guess; every ms of it
                                          // is user-facing latency)
    int      min_utterance_ms  = 300;     // reject clicks/breaths
    double   absolute_floor    = 3.0e-4;  // below this we are in true silence
};

struct VadEvent {
    int         index        = 0;
    double      t_start_ms   = 0;   // first speech sample
    double      t_last_ms    = 0;   // last speech sample
    double      t_detect_ms  = 0;   // hangover expired -> handed to ASR
    std::string audio;              // 16 kHz mono float32 samples
};

class Vad {
public:
    explicit Vad(const VadConfig & cfg = {}) : cfg_(cfg) {}

    // Feed samples, get any completed utterances. Samples are 16 kHz mono float.
    // `t0_ms` is the timestamp of the first sample in this block.
    std::vector<VadEvent> feed(const std::vector<float> & samples, double t0_ms);

    // Force-close any open utterance (end of file / user interrupt).
    std::vector<VadEvent> flush(double t_end_ms);

    double noise_floor() const { return floor_; }
    const VadConfig & config() const { return cfg_; }

    // Live state of the open utterance, for decoding speculatively before the
    // hangover expires: how long it has been quiet, and the audio so far with
    // that quiet tail removed.
    bool   speaking() const { return speaking_; }
    int    silence_ms() const { return speaking_ ? run_off_ * cfg_.frame_ms : 0; }
    std::vector<float> open_audio() const {
        const size_t tail = (size_t)silence_ms() * 16;
        return std::vector<float>(buf_.begin(), buf_.end() - std::min(tail, buf_.size()));
    }

private:
    VadConfig cfg_;
    double    floor_       = 1e-3;
    int       run_on_      = 0;   // consecutive speech frames
    int       run_off_     = 0;   // consecutive silence frames
    bool      speaking_    = false;
    int       idx_         = 0;
    double    t_start_     = 0;
    double    t_last_      = 0;
    std::vector<float> buf_;
};

} // namespace s2st
