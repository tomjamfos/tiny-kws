#include "fft.h"
#include "naive_dft.h"
#include "signals.h"
#include "spectrum_error.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kws::test {

namespace {
constexpr std::size_t kN = 8;
constexpr double kTolerance = 1e-5;
} // namespace

TEST_CASE("Fft forward impulse at index 0", "[fft]") {

    const Fft fft(kN);
    const std::vector<std::complex<double>> expected(kN, {1.0, 0.0});
    auto actual = to_complex_float(impulse(kN, 0));
    fft.forward(actual.data());

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("Fft forward impulse at index 3", "[fft]") {

    constexpr std::size_t delay = 3;
    const Fft fft(kN);
    std::vector<std::complex<double>> expected(kN);
    for (std::size_t k = 0; k < kN; ++k) {
        const double angle = -2.0 * kPi * static_cast<double>(k) *
                             static_cast<double>(delay) / static_cast<double>(kN);
        expected[k] = std::polar(1.0, angle);
    }
    auto actual = to_complex_float(impulse(kN, delay));
    fft.forward(actual.data());

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("Fft forward constant input", "[fft]") {

    constexpr double dc = 0.5;
    const Fft fft(kN);
    std::vector<std::complex<double>> expected(kN);
    expected[0] = dc * static_cast<double>(kN);
    auto actual = to_complex_float(std::vector<double>(kN, dc));
    fft.forward(actual.data());

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("Fft forward cosine at bin 2", "[fft]") {

    constexpr std::size_t bin = 2;
    const Fft fft(kN);
    std::vector<std::complex<double>> expected(kN);
    expected[bin] = std::polar(static_cast<double>(kN) / 2.0, 0.0);
    expected[kN - bin] = std::conj(expected[bin]);
    auto actual = to_complex_float(tone(kN, bin, 0.0));
    fft.forward(actual.data());

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("Fft forward sine at bin 3", "[fft]") {

    constexpr std::size_t bin = 3;
    constexpr double phase = -kPi / 2.0;
    const Fft fft(kN);
    std::vector<std::complex<double>> expected(kN);
    expected[bin] = std::polar(static_cast<double>(kN) / 2.0, phase);
    expected[kN - bin] = std::conj(expected[bin]);
    auto actual = to_complex_float(tone(kN, bin, phase));
    fft.forward(actual.data());

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("Fft forward matches NaiveDft on noise (real values)", "[fft]") {

    constexpr std::size_t sizes[] = {2, 8, 512};
    constexpr std::uint32_t seed = 42;
    for (const std::size_t n : sizes) {
        CAPTURE(n);

        const Fft fft(n);
        const NaiveDft dft(n);
        const std::vector<double> x = noise_real(n, seed);
        const std::vector<std::complex<double>> expected = dft.forward(x);
        auto actual = to_complex_float(x);
        fft.forward(actual.data());

        REQUIRE(scaled_max_error(actual, expected) < kTolerance);
    }
}

TEST_CASE("Fft forward matches NaiveDft on noise (complex)", "[fft]") {

    constexpr std::size_t sizes[] = {2, 8, 512};
    constexpr std::uint32_t seed = 42;
    for (const std::size_t n : sizes) {
        CAPTURE(n);

        const Fft fft(n);
        const NaiveDft dft(n);
        const std::vector<std::complex<double>> z = noise_complex(n, seed);

        std::vector<double> real(n);
        std::vector<double> imaginary(n);
        for (std::size_t i = 0; i < n; ++i) {
            real[i] = z[i].real();
            imaginary[i] = z[i].imag();
        }

        const auto real_spectrum = dft.forward(real);
        const auto imaginary_spectrum = dft.forward(imaginary);
        std::vector<std::complex<double>> expected(n);
        for (std::size_t k = 0; k < n; ++k) {
            expected[k] = real_spectrum[k] +
                          std::complex<double>(0.0, 1.0) * imaginary_spectrum[k];
        }

        auto actual = to_complex_float(z);
        fft.forward(actual.data());

        REQUIRE(scaled_max_error(actual, expected) < kTolerance);
    }
}

TEST_CASE("Fft throws on invalid size", "[fft]") {

    REQUIRE_THROWS_AS(Fft(0), std::invalid_argument);
    REQUIRE_THROWS_AS(Fft(1), std::invalid_argument);
    REQUIRE_THROWS_AS(Fft(12), std::invalid_argument);

    const Fft fft(kN);
    REQUIRE(fft.size() == kN);
}

} // namespace kws::test
