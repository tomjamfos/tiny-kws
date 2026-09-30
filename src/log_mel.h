#pragma once

#include "fft.h"

#include <cstddef>
#include <vector>

namespace kws {

// Constants, not parameters: this is the feature layout the model is trained on.

inline constexpr std::size_t kSampleRate = 16000;
inline constexpr std::size_t kNyquistHz = kSampleRate / 2;

// Clips are 1 second long after padding/truncation.
inline constexpr std::size_t kClipSamples = kSampleRate;

inline constexpr std::size_t kFrameSamples = 480; // 30 ms
inline constexpr std::size_t kHopSamples = 160;   // 10 ms

// Clip must yield at least one frame.
static_assert(kFrameSamples <= kClipSamples);

// Non-centred framing: each frame starts at a multiple of kHopSamples,
// with no extra padding at the start or end of the clip.
inline constexpr std::size_t kLastFrameStart = kClipSamples - kFrameSamples;
inline constexpr std::size_t kNumFrames = kLastFrameStart / kHopSamples + 1;

// End of last frame must be flush with end of clip,
// otherwise trailing samples are silently dropped.
static_assert(kLastFrameStart % kHopSamples == 0);

// Smallest power of 2 >= kFrameSamples, as the FFT is radix-2.
inline constexpr std::size_t kFftSize = 512;

static_assert(kFrameSamples <= kFftSize);

inline constexpr std::size_t kNumFreqBins = kFftSize / 2 + 1; // 0 Hz to Nyquist
inline constexpr double kFreqBinWidthHz = // FFT frequency resolution
    static_cast<double>(kSampleRate) / static_cast<double>(kFftSize);

inline constexpr std::size_t kNumMelBins = 32;
inline constexpr double kMelMinHz = 20.0;
inline constexpr double kMelMaxHz = 8000.0;

static_assert(kMelMinHz < kMelMaxHz);
static_assert(kMelMaxHz <= static_cast<double>(kNyquistHz));

// Lower bound on mel power before the log,
// so silence gives ln(1e-6) ≈ -13.8 rather than -inf.
inline constexpr float kLogMelFloor = 1e-6f;

// Dense HTK scale mel filterbank (no normalisation),
// kNumMelBins rows of kNumFreqBins weights, row-major.
std::vector<float> mel_filterbank();

class LogMel {
public:
    LogMel();

    // input: kClipSamples samples in [-1, 1), all finite.
    // process() does no padding or truncation.
    // output: kNumMelBins x kNumFrames log-mel features, row-major.
    // One LogMel can serve concurrent calls, given separate output buffers.
    void process(const float* input, float* output) const noexcept;

private:
    Fft fft_;
    std::vector<float> window_;     // Periodic Hann window across kFrameSamples
    std::vector<float> filterbank_; // kNumMelBins x kNumFreqBins filterbank, row-major
};

} // namespace kws
