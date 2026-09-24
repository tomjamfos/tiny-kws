#include "log_mel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace kws {
namespace {

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
    for (std::size_t edge = 1; edge + 1 < num_edges; ++edge) {
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

} // namespace kws
