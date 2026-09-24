#pragma once

#include <cstddef>
#include <vector>

namespace kws {

// Constants, not parameters: this is the feature layout the model is trained on.

inline constexpr std::size_t kSampleRate = 16000;
inline constexpr std::size_t kNyquistHz = kSampleRate / 2;

// Clips are 1 second long after padding/truncation
inline constexpr std::size_t kClipSamples = kSampleRate;

inline constexpr std::size_t kFrameSamples = 480; // 30 ms
inline constexpr std::size_t kHopSamples = 160;   // 10 ms

static_assert(kFrameSamples <= kClipSamples);

inline constexpr std::size_t kLastFrameStart = kClipSamples - kFrameSamples;
inline constexpr std::size_t kNumFrames = kLastFrameStart / kHopSamples + 1;

// End of last frame must be flush with end of clip,
// otherwise trailing samples are silently dropped.
static_assert(kLastFrameStart % kHopSamples == 0);

inline constexpr std::size_t kFftSize = 512;

static_assert(kFrameSamples <= kFftSize);

inline constexpr std::size_t kNumFreqBins = kFftSize / 2 + 1; // 0 Hz to Nyquist
inline constexpr double kFreqBinWidthHz = // Fft frequency resolution
    static_cast<double>(kSampleRate) / static_cast<double>(kFftSize);

inline constexpr std::size_t kNumMelBins = 32;
inline constexpr double kMelMinHz = 20.0;
inline constexpr double kMelMaxHz = 8000.0;

static_assert(kMelMinHz < kMelMaxHz);
static_assert(kMelMaxHz <= static_cast<double>(kNyquistHz));

// Dense HTK scale mel filterbank (no normalisation),
// kNumMelBins rows of kNumFreqBins weights, row-major.
std::vector<float> mel_filterbank();

} // namespace kws
