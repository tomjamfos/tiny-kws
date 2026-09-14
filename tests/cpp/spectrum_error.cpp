#include "spectrum_error.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace kws::test {

double scaled_max_error(const std::vector<std::complex<double>>& actual,
                        const std::vector<std::complex<double>>& expected) {

    if (actual.size() != expected.size()) {
        throw std::invalid_argument("scaled_max_error: size mismatch between actual: " +
                                    std::to_string(actual.size()) + " and expected: " +
                                    std::to_string(expected.size()));
    }
    double max_error = 0.0, max_magnitude = 0.0;
    for (std::size_t k = 0; k < expected.size(); ++k) {
        max_error = std::max(max_error, std::abs(actual[k] - expected[k]));
        max_magnitude = std::max(max_magnitude, std::abs(expected[k]));
    }
    if (max_magnitude == 0.0) {
        throw std::invalid_argument(
            "scaled_max_error: expected spectrum contains no non-zero elements");
    }
    return max_error / max_magnitude;
}

double scaled_max_error(const std::vector<std::complex<float>>& actual,
                        const std::vector<std::complex<double>>& expected) {

    std::vector<std::complex<double>> actual_double(actual.size());
    for (std::size_t k = 0; k < actual.size(); ++k) {
        actual_double[k] = actual[k];
    }
    return scaled_max_error(actual_double, expected);
}

} // namespace kws::test
