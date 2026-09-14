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
constexpr double kTolerance = 1e-12;
} // namespace

TEST_CASE("NaiveDft forward impulse at index 0", "[naive_dft]") {

    const NaiveDft dft(kN);
    const std::vector<std::complex<double>> expected(kN, {1.0, 0.0});
    const auto actual = dft.forward(impulse(kN, 0));

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("NaiveDft forward impulse at index 3", "[naive_dft]") {

    constexpr std::size_t delay = 3;
    const NaiveDft dft(kN);
    std::vector<std::complex<double>> expected(kN);
    for (std::size_t k = 0; k < kN; ++k) {
        const double angle = -2.0 * kPi * static_cast<double>(k) *
                             static_cast<double>(delay) / static_cast<double>(kN);
        expected[k] = std::polar(1.0, angle);
    }
    const auto actual = dft.forward(impulse(kN, delay));

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("NaiveDft forward constant input", "[naive_dft]") {

    constexpr double dc = 0.5;
    const NaiveDft dft(kN);
    std::vector<std::complex<double>> expected(kN);
    expected[0] = dc * static_cast<double>(kN);
    const auto actual = dft.forward(std::vector<double>(kN, dc));

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("NaiveDft forward cosine at bin 2", "[naive_dft]") {

    constexpr std::size_t bin = 2;
    const NaiveDft dft(kN);
    std::vector<std::complex<double>> expected(kN);
    expected[bin] = std::polar(static_cast<double>(kN) / 2.0, 0.0);
    expected[kN - bin] = std::conj(expected[bin]);
    const auto actual = dft.forward(tone(kN, bin, 0.0));

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("NaiveDft forward sine at bin 3", "[naive_dft]") {

    constexpr std::size_t bin = 3;
    constexpr double phase = -kPi / 2.0;
    const NaiveDft dft(kN);
    std::vector<std::complex<double>> expected(kN);
    expected[bin] = std::polar(static_cast<double>(kN) / 2.0, phase);
    expected[kN - bin] = std::conj(expected[bin]);
    const auto actual = dft.forward(tone(kN, bin, phase));

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("NaiveDft forward and backward matches input", "[naive_dft]") {

    constexpr std::uint32_t seed = 42;
    const NaiveDft dft(kN);
    const std::vector<double> x = noise_real(kN, seed);
    std::vector<std::complex<double>> expected(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        expected[i] = x[i];
    }
    const auto actual = dft.backward(dft.forward(x));

    REQUIRE(scaled_max_error(actual, expected) < kTolerance);
}

TEST_CASE("NaiveDft throws when size is invalid", "[naive_dft]") {

    REQUIRE_THROWS_AS(NaiveDft(0), std::invalid_argument);
    REQUIRE_THROWS_AS(NaiveDft(1), std::invalid_argument);
    REQUIRE_THROWS_AS(NaiveDft(6), std::invalid_argument);

    const NaiveDft dft(kN);
    REQUIRE_THROWS_AS(dft.forward(std::vector<double>(kN - 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(dft.backward(std::vector<std::complex<double>>(kN + 1)),
                      std::invalid_argument);
}

} // namespace kws::test
