#include "signals.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace kws::test {

std::vector<double> impulse(std::size_t n, std::size_t delay_index) {

    if (delay_index >= n) {
        throw std::invalid_argument(
            "impulse: delay_index should be less than n: " + std::to_string(n) +
            ", got " + std::to_string(delay_index));
    }
    std::vector<double> buffer(n);
    buffer[delay_index] = 1.0;
    return buffer;
}

std::vector<double> tone(std::size_t n, std::size_t bin, double phase) {

    if (bin >= n) {
        throw std::invalid_argument("tone: bin should be less than n: " +
                                    std::to_string(n) + ", got " + std::to_string(bin));
    }
    std::vector<double> buffer(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(bin) *
                             static_cast<double>(i) / static_cast<double>(n);
        buffer[i] = std::cos(angle + phase);
    }
    return buffer;
}

std::vector<double> noise_real(std::size_t n, std::uint32_t seed) {

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> buffer(n);
    for (std::size_t i = 0; i < n; ++i) {
        buffer[i] = dist(rng);
    }
    return buffer;
}

std::vector<std::complex<double>> noise_complex(std::size_t n, std::uint32_t seed) {

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::complex<double>> buffer(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double real = dist(rng);
        const double imaginary = dist(rng);
        buffer[i] = {real, imaginary};
    }
    return buffer;
}

std::vector<std::complex<float>> to_complex_float(const std::vector<double>& x) {

    std::vector<std::complex<float>> buffer(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        buffer[i] = {static_cast<float>(x[i]), 0.0f};
    }
    return buffer;
}

} // namespace kws::test
