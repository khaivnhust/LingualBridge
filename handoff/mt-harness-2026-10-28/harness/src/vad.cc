#include "vad.h"

#include <cmath>

namespace s2st {

std::vector<VadEvent> Vad::feed(const std::vector<float> & samples, double t0_ms) {
    std::vector<VadEvent> out;
    const int sps = 16000;
    const int frame = cfg_.frame_ms * sps / 1000;

    size_t off = 0;
    while (off < samples.size()) {
        size_t n = std::min((size_t)frame, samples.size() - off);
        double f_start = t0_ms + 1000.0 * off / sps;
        double f_end   = t0_ms + 1000.0 * (off + n) / sps;

        double acc = 0;
        for (size_t i = 0; i < n; ++i) acc += (double)samples[off + i] * samples[off + i];
        double rms = std::sqrt(acc / (double)n);

        const bool loud = rms > std::max(floor_ * cfg_.onset_ratio, cfg_.absolute_floor);
        const bool quiet = rms < floor_ * cfg_.offset_ratio;

        if (loud) {
            ++run_on_;
            run_off_ = 0;
            if (!speaking_ && run_on_ >= cfg_.onset_frames) {
                speaking_ = true;
                // back-date the start to when the run actually began
                t_start_ = f_end - (double)(run_on_ - 1) * cfg_.frame_ms;
                buf_.clear();
            }
        } else {
            run_on_ = 0;
            if (!speaking_) {
                // track the noise floor only while silent
                floor_ = (1.0 - cfg_.noise_floor_alpha) * floor_ + cfg_.noise_floor_alpha * rms;
                if (floor_ < 1e-5) floor_ = 1e-5;
            }
        }

        // Buffer EVERY frame while the utterance is open, not just the loud
        // ones. Pauses inside a sentence carry meaning ("yes ... the motor is
        // broken"), so dropping low-energy frames splices the utterance
        // together and shifts it out of alignment with the source timeline.
        // The trailing hangover is removed once, on close.
        if (speaking_) {
            buf_.insert(buf_.end(), samples.begin() + off, samples.begin() + off + n);
            if (loud) t_last_ = f_end;
        }

        if (speaking_ && !loud) {
            ++run_off_;
            if (quiet && run_off_ * cfg_.frame_ms >= cfg_.hangover_ms) {
                size_t hang = (size_t)(cfg_.hangover_ms * sps / 1000);
                if (buf_.size() > hang) buf_.resize(buf_.size() - hang);
                if (buf_.size() > (size_t)(cfg_.min_utterance_ms * sps / 1000)) {
                    VadEvent e;
                    e.index       = idx_++;
                    e.t_start_ms  = t_start_;
                    e.t_last_ms   = t_last_;
                    e.t_detect_ms = f_end;
                    e.audio.assign((const char *)buf_.data(), buf_.size() * sizeof(float));
                    out.push_back(std::move(e));
                }
                buf_.clear();
                speaking_ = false;
            }
        }
        off += n;
    }
    return out;
}

std::vector<VadEvent> Vad::flush(double t_end_ms) {
    std::vector<VadEvent> out;
    if (!speaking_ || buf_.empty()) return out;
    const int sps = 16000;
    size_t keep = buf_.size() > (size_t)(cfg_.hangover_ms * sps / 1000)
                      ? buf_.size() - (size_t)(cfg_.hangover_ms * sps / 1000)
                      : 0;
    if (keep > (size_t)(cfg_.min_utterance_ms * sps / 1000)) {
        buf_.resize(keep);
        VadEvent e;
        e.index       = idx_++;
        e.t_start_ms  = t_start_;
        e.t_last_ms   = t_last_;
        e.t_detect_ms = t_end_ms;
        e.audio.assign((const char *)buf_.data(), buf_.size() * sizeof(float));
        out.push_back(std::move(e));
    }
    buf_.clear();
    speaking_ = false;
    return out;
}

} // namespace s2st
