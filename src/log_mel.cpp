#include "log_mel.h"
#include "math_constants.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace kws {
namespace {

std::vector<float> hann_window() {

    std::vector<float> window(kFrameSamples);
    for (std::size_t n = 0; n < kFrameSamples; ++n) {

        // Periodic, so divide by kFrameSamples not kFrameSamples - 1.
        const double phase =
            2.0 * kPi * static_cast<double>(n) / static_cast<double>(kFrameSamples);
        window[n] = static_cast<float>(0.5 - 0.5 * std::cos(phase));
    }
    return window;
}

// HTK mel scale
double hz_to_mel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }

double mel_to_hz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

} // namespace

std::vector<float> mel_filterbank() {

    // The filters are triangles with corners on a shared grid of edges.
    // Each mel bin's filter peaks on its own edge, while its base runs
    // from the peak of the filter below to the peak of the filter above.
    // As the first and last filters have no neighbour below and above,
    // an edge is added at each end, at kMelMinHz and kMelMaxHz.
    constexpr std::size_t num_edges = kNumMelBins + 2;
    double edges_hz[num_edges] = {0.0};

    const double mel_min = hz_to_mel(kMelMinHz), mel_max = hz_to_mel(kMelMaxHz);
    const double mel_step = (mel_max - mel_min) / static_cast<double>(num_edges - 1);

    // Assign endpoints exactly, avoiding the Hz to mel to Hz round trip,
    // as its rounding error would leave a small nonzero weight in
    // the last element (highest mel filter, Nyquist bin).
    edges_hz[0] = kMelMinHz;
    edges_hz[num_edges - 1] = kMelMaxHz;

    // Inner edges: uniform in mel, so in Hz the filters widen with frequency.
    for (std::size_t edge = 1; edge < num_edges - 1; ++edge) {
        edges_hz[edge] = mel_to_hz(mel_min + static_cast<double>(edge) * mel_step);
    }

    std::vector<float> filterbank(kNumMelBins * kNumFreqBins);
    for (std::size_t mel_bin = 0; mel_bin < kNumMelBins; ++mel_bin) {

        const double low_edge = edges_hz[mel_bin];
        const double centre_edge = edges_hz[mel_bin + 1]; // Where the filter peaks
        const double high_edge = edges_hz[mel_bin + 2];

        for (std::size_t freq_bin = 0; freq_bin < kNumFreqBins; ++freq_bin) {

            const double hz = static_cast<double>(freq_bin) * kFreqBinWidthHz;

            // 0 at low_edge, 1 at centre_edge.
            const double rising = (hz - low_edge) / (centre_edge - low_edge);

            // 1 at centre_edge, 0 at high_edge.
            const double falling = (high_edge - hz) / (high_edge - centre_edge);

            // The filter's weight at this bin's frequency: take the minimum of
            // rising and falling so the slope flips sign at their crossing,
            // centre_edge. Clamped at 0.0 so the filter has support only
            // between low_edge and high_edge. Row major (mel_bin, freq_bin).
            filterbank[mel_bin * kNumFreqBins + freq_bin] =
                static_cast<float>(std::max(0.0, std::min(rising, falling)));
        }
    }
    return filterbank;
}

LogMel::LogMel()
    : fft_(kFftSize), window_(hann_window()), filterbank_(mel_filterbank()) {}

void LogMel::process(const float* input, float* output) const noexcept {

    // Scratch buffers on the stack, so concurrent process() calls never share them.
    std::complex<float> fft_buffer[kFftSize];
    float power_spectrum[kNumFreqBins];

    for (std::size_t frame = 0; frame < kNumFrames; ++frame) {

        // Slide the frame along input by one hop.
        const float* frame_input = input + frame * kHopSamples;

        // Copy the (real-valued) current frame
        // into fft_buffer and apply the Hann window.
        for (std::size_t n = 0; n < kFrameSamples; ++n) {
            fft_buffer[n] = {frame_input[n] * window_[n], 0.0f};
        }

        // Zero-pad to kFftSize after each frame's samples, as fft_.forward()
        // on the previous frame overwrote all of fft_buffer, including the padding.
        for (std::size_t n = kFrameSamples; n < kFftSize; ++n) {
            fft_buffer[n] = {0.0f, 0.0f};
        }

        fft_.forward(fft_buffer);

        // Power spectrum, 0 Hz to Nyquist only,
        // as real input gives a mirrored upper half.
        for (std::size_t freq_bin = 0; freq_bin < kNumFreqBins; ++freq_bin) {
            const float real = fft_buffer[freq_bin].real();
            const float imaginary = fft_buffer[freq_bin].imag();
            power_spectrum[freq_bin] = real * real + imaginary * imaginary;
        }

        for (std::size_t mel_bin = 0; mel_bin < kNumMelBins; ++mel_bin) {

            // Weights of the mel bin's filter.
            const float* filter = filterbank_.data() + mel_bin * kNumFreqBins;

            // Dot product of the mel bin's filter with the frame's power spectrum.
            float mel_power = 0.0f;
            for (std::size_t freq_bin = 0; freq_bin < kNumFreqBins; ++freq_bin) {
                mel_power += power_spectrum[freq_bin] * filter[freq_bin];
            }

            // Floored natural log of mel_power. Output is row-major.
            // mel_power is the first argument, so NaN propagates as it does
            // through torch.clamp.
            output[mel_bin * kNumFrames + frame] =
                std::log(std::max(mel_power, kLogMelFloor));
        }
    }
}

} // namespace kws
