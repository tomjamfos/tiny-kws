#include "naive_dft.h"
#include "math_constants.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace kws::test {

NaiveDft::NaiveDft(std::size_t n) : n_(n) {

    if (n < 2 || (n & (n - 1)) != 0) {
        throw std::invalid_argument(
            "NaiveDft: n should be at least 2, and a power of 2, got " +
            std::to_string(n));
    }
    inverse_n_ = 1.0 / static_cast<double>(n_);

    twiddles_.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        const double angle =
            2.0 * kPi * static_cast<double>(i) / static_cast<double>(n_);
        twiddles_[i] = {std::cos(angle), -std::sin(angle)};
    }
}

std::vector<std::complex<double>>
NaiveDft::forward(const std::vector<double>& x) const {

    if (x.size() != n_) {
        throw std::invalid_argument(
            "NaiveDft: x should match n: " + std::to_string(n_) + ", got " +
            std::to_string(x.size()));
    }
    std::vector<std::complex<double>> result(n_);
    for (std::size_t k = 0; k < n_; ++k) {

        std::complex<double> bin_value{0.0, 0.0};
        for (std::size_t m = 0; m < n_; ++m) {
            bin_value += x[m] * twiddles_[(k * m) % n_];
        }
        result[k] = bin_value;
    }
    return result;
}

std::vector<std::complex<double>>
NaiveDft::backward(const std::vector<std::complex<double>>& spectrum) const {

    if (spectrum.size() != n_) {
        throw std::invalid_argument(
            "NaiveDft: spectrum should match n: " + std::to_string(n_) + ", got " +
            std::to_string(spectrum.size()));
    }
    std::vector<std::complex<double>> result(n_);
    for (std::size_t m = 0; m < n_; ++m) {

        std::complex<double> sample_value{0.0, 0.0};
        for (std::size_t k = 0; k < n_; ++k) {
            sample_value += spectrum[k] * std::conj(twiddles_[(k * m) % n_]);
        }
        result[m] = inverse_n_ * sample_value;
    }
    return result;
}

} // namespace kws::test
