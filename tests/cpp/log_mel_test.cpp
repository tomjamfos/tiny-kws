#include "log_mel.h"
#include "math_constants.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace kws::test {

TEST_CASE("LogMel process silent input", "[log_mel]") {

    const std::vector<float> input(kClipSamples, 0.0f);

    // Fill output buffer with zeros, so any element process()
    // fails to write also fails the log_floor equality check.
    std::vector<float> output(kNumMelBins * kNumFrames, 0.0f);

    const LogMel frontend;
    frontend.process(input.data(), output.data());

    const float log_floor = std::log(kLogMelFloor);
    for (std::size_t n = 0; n < output.size(); ++n) {
        CAPTURE(n);

        // process() clamps mel power to kLogMelFloor before the log,
        // so silence gives log(kLogMelFloor) in every bin.
        // Safe to compare with exact equality.
        REQUIRE(output[n] == log_floor);
    }
}

TEST_CASE("LogMel process impulse at sample 7280", "[log_mel]") {

    constexpr std::size_t impulse_idx = 7280;
    constexpr double amplitude = 0.5;

    // Frame i spans [i * kHopSamples, i * kHopSamples + kFrameSamples).
    constexpr std::size_t first_frame = 43;
    constexpr std::size_t last_frame = 45;

    std::vector<float> input(kClipSamples, 0.0f);
    input[impulse_idx] = static_cast<float>(amplitude);

    std::vector<float> output(kNumMelBins * kNumFrames, 0.0f);

    const LogMel frontend;
    frontend.process(input.data(), output.data());

    const std::vector<float> filterbank = mel_filterbank();
    const float log_floor = std::log(kLogMelFloor);

    for (std::size_t mel_bin = 0; mel_bin < kNumMelBins; ++mel_bin) {

        // Compute sum of all the weights of mel_bin's filter.
        double filter_sum = 0.0;
        for (std::size_t freq_bin = 0; freq_bin < kNumFreqBins; ++freq_bin) {
            filter_sum +=
                static_cast<double>(filterbank[mel_bin * kNumFreqBins + freq_bin]);
        }

        for (std::size_t frame = 0; frame < kNumFrames; ++frame) {
            CAPTURE(mel_bin, frame);

            const float actual = output[mel_bin * kNumFrames + frame];

            // Frames that don't contain the impulse are silent, so just
            // check that every bin is at the log floor and move to next frame.
            if (frame < first_frame || frame > last_frame) {
                REQUIRE(actual == log_floor);
                continue;
            }

            // Impulse's offset within this frame.
            const std::size_t position = impulse_idx - frame * kHopSamples;

            // Windowed impulse amplitude.
            const double phase = 2.0 * kPi * static_cast<double>(position) /
                                 static_cast<double>(kFrameSamples);
            const double peak = amplitude * (0.5 - 0.5 * std::cos(phase));

            // An impulse has equal power at every frequency,
            // so each filter just scales it by the sum of its weights.
            const double expected = std::log(peak * peak * filter_sum);

            REQUIRE(std::abs(static_cast<double>(actual) - expected) < 1e-5);
        }
    }
}

TEST_CASE("LogMel process cosine at 440, 720 and 7500 Hz", "[log_mel]") {

    struct Tone {
        double hz;
        std::size_t mel_bin; // expected peak bin
    };
    constexpr Tone tones[] = {{440.0, 5}, {720.0, 8}, {7500.0, 31}};
    constexpr double amplitude = 0.5;

    const LogMel frontend;
    for (const Tone& tone : tones) {

        // Compute kClipSamples of a pure cosine at tone.hz.
        std::vector<float> input(kClipSamples);
        for (std::size_t n = 0; n < kClipSamples; ++n) {
            const double phase = 2.0 * kPi * tone.hz * static_cast<double>(n) /
                                 static_cast<double>(kSampleRate);
            input[n] = static_cast<float>(amplitude * std::cos(phase));
        }

        std::vector<float> output(kNumMelBins * kNumFrames, 0.0f);
        frontend.process(input.data(), output.data());

        for (std::size_t frame = 0; frame < kNumFrames; ++frame) {

            // Argmax over frame's mel bins.
            std::size_t peak_bin = 0;
            for (std::size_t mel_bin = 1; mel_bin < kNumMelBins; ++mel_bin) {
                if (output[mel_bin * kNumFrames + frame] >
                    output[peak_bin * kNumFrames + frame]) {
                    peak_bin = mel_bin;
                }
            }

            CAPTURE(tone.hz, frame);
            REQUIRE(peak_bin == tone.mel_bin);
        }
    }
}

TEST_CASE("mel_filterbank filter support", "[log_mel]") {

    const std::vector<float> filterbank = mel_filterbank();
    REQUIRE(filterbank.size() == kNumMelBins * kNumFreqBins);

    std::size_t prev_first_freq_bin = 0, prev_last_freq_bin = 0;
    for (std::size_t mel_bin = 0; mel_bin < kNumMelBins; ++mel_bin) {
        CAPTURE(mel_bin);

        const float* filter = filterbank.data() + mel_bin * kNumFreqBins;

        // Find the filter's support, its first and last non-zero weights.
        // first_freq_bin stays at kNumFreqBins if every weight is zero.
        std::size_t first_freq_bin = kNumFreqBins, last_freq_bin = 0;
        for (std::size_t freq_bin = 0; freq_bin < kNumFreqBins; ++freq_bin) {
            if (filter[freq_bin] != 0.0f) {
                if (first_freq_bin == kNumFreqBins) {
                    first_freq_bin = freq_bin;
                }
                last_freq_bin = freq_bin;
            }
        }

        REQUIRE(first_freq_bin < kNumFreqBins);    // at least one non-zero weight
        REQUIRE(first_freq_bin > 0);               // no weight at 0 Hz
        REQUIRE(last_freq_bin < kNumFreqBins - 1); // no weight at Nyquist

        // All weights in the support should be in (0, 1].
        for (std::size_t freq_bin = first_freq_bin; freq_bin <= last_freq_bin;
             ++freq_bin) {
            CAPTURE(freq_bin);
            REQUIRE(filter[freq_bin] > 0.0f);
            REQUIRE(filter[freq_bin] <= 1.0f);
        }

        // Each filter's support should start and end higher than the previous filter's.
        if (mel_bin > 0) {
            REQUIRE(first_freq_bin > prev_first_freq_bin);
            REQUIRE(last_freq_bin > prev_last_freq_bin);
        }
        prev_first_freq_bin = first_freq_bin;
        prev_last_freq_bin = last_freq_bin;
    }
}

TEST_CASE("mel_filterbank weights sum to 1 between first and last peaks", "[log_mel]") {

    const std::vector<float> filterbank = mel_filterbank();

    const auto peak_freq_bin = [&filterbank](std::size_t mel_bin) {
        const float* filter = filterbank.data() + mel_bin * kNumFreqBins;

        // Argmax over mel_bin's frequency bin weights.
        std::size_t argmax = 0;
        for (std::size_t freq_bin = 1; freq_bin < kNumFreqBins; ++freq_bin) {
            if (filter[freq_bin] > filter[argmax]) {
                argmax = freq_bin;
            }
        }
        return argmax;
    };

    const std::size_t first_peak_bin = peak_freq_bin(0);
    const std::size_t last_peak_bin = peak_freq_bin(kNumMelBins - 1);
    REQUIRE(first_peak_bin + 1 < last_peak_bin);

    for (std::size_t freq_bin = first_peak_bin + 1; freq_bin < last_peak_bin;
         ++freq_bin) {
        CAPTURE(freq_bin);

        // Sum the weight of freq_bin in every mel bin.
        double weight_sum = 0.0;
        for (std::size_t mel_bin = 0; mel_bin < kNumMelBins; ++mel_bin) {
            weight_sum +=
                static_cast<double>(filterbank[mel_bin * kNumFreqBins + freq_bin]);
        }

        // Between the centres of adjacent filters,
        // one falls as the other rises, summing to 1.
        REQUIRE(std::abs(1.0 - weight_sum) < 1e-6);
    }
}

} // namespace kws::test
