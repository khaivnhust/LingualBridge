// tts.h -- Piper VITS via ONNX Runtime (CPU).
//
// Signature verified against piper/voices/en_US-*.onnx with onnxruntime 1.31:
//   in : input int64 [batch, phonemes]
//        input_lengths int64 [batch]
//        scales float [3]        (noise / length / noise-w scale)
//   out: output float [batch, 1, 1, samples]   @ 22050 Hz
//
// Phonemisation is the awkward part on Android: Piper normally drives espeak-ng,
// which is not present on the QDC rootfs. Two modes are supported and the
// active one is recorded in the results, because they have very different
// latencies and we should not silently compare one against the other:
//
//   TEXT   -- espeak-ng linked in at build time; phonemes computed in-process
//   IDS    -- phoneme id sequence supplied per chunk (pre-computed offline)
//
// "Begin of voice" is the first sample exceeding `--onset-thresh`, not sample 0:
// VITS output opens with near-silence, and synthesising fast is worthless if the
// first audible sound still arrives late.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace s2st {

struct TtsResult {
    bool        ok            = false;
    std::string error;
    long long   t_prep_ms     = -1;   // phonemise / build inputs
    long long   t_run_ms      = -1;   // ORT Run()
    long long   t_first_ms    = -1;   // when Run() returned (audio now exists)
    int         first_voice_ix= -1;   // first sample above onset threshold
    int         n_samples     = 0;
    float       duration_s   = 0.f;
    std::vector<float> pcm;           // the waveform itself

    // Piper/VITS produces the whole utterance inside one Run() call, so the
    // waveform does not exist until Run() returns -- there is no earlier moment
    // at which "the first sample" could have been delivered. first_voice_ix is
    // therefore a *content* offset (how far into the clip the first audible
    // sound sits, worth trimming before playback), not a timing saving. The
    // latency is t_run_ms. Treating first_voice_ix as a timing win would
    // understate end-of-speech -> begin-of-voice by up to that offset.
    float first_voice_ms() const {
        return first_voice_ix >= 0 ? 1000.f * first_voice_ix / 22050.f : 0.f;
    }
};

class Tts {
public:
    // voice_dir must contain the .onnx and its .onnx.json phoneme_id_map.
    // espeak_data_path must be the directory CONTAINING espeak-ng-data/ (Android
    // has no compiled-in default, and initialising espeak without its data
    // segfaults on the first espeak_TextToPhonemes call).
    static std::unique_ptr<Tts> create(const std::string & onnx_path,
                                       const std::string & phoneme_map_path,
                                       int num_threads = 4,
                                       const std::string & espeak_data_path = "");
    ~Tts();

    // ids: phoneme ids in [1,256] as Piper encodes them.
    TtsResult synth_ids(const std::vector<int64_t> & ids);

    // text: requires espeak-ng at build time.
    TtsResult synth_text(const std::string & text, const std::string & lang);

    static bool espeak_available();

    // Load a Piper phoneme_id_map JSON into symbol->id tables.
    bool load_phoneme_map(const std::string & path);

    // Directory that contains espeak-ng-data/ (required on Android).
    void set_espeak_data_path(const std::string & p) { data_path_ = p; }

    const std::vector<int64_t> & last_pad() const { return pad_; }
    void   set_speaker_scale(double length_penalty, double noise_scale, double noise_w);

    // Exposed for validation against piper-tts on the host.
    std::vector<int64_t> phonemize(const std::string & text, const std::string & lang,
                                   int * n_unmapped);
private:
    Tts();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<int64_t> pad_;
    std::vector<std::pair<std::string, int64_t>> pmap_;
    std::unordered_map<std::string, int64_t> id_of_;
    int64_t id_bos_ = 1, id_eos_ = 2, id_pad_ = 0;
    bool     inited_ = false;
    std::string data_path_;
    double length_scale_ = 1.0, noise_scale_ = 0.667, noise_w_scale_ = 0.8;
};

} // namespace s2st

// Split text into (clause-body, terminator) pairs. UTF-8 aware; does not split
// decimals or thousands separators. Declared here because the host-side
// validation tool uses it too.
std::vector<std::pair<std::string, std::string>> split_clauses(const std::string & text);
