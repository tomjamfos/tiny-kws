#pragma once

#include <complex>
#include <vector>

namespace kws::test {

double scaled_max_error(const std::vector<std::complex<double>>& actual,
                        const std::vector<std::complex<double>>& expected);

double scaled_max_error(const std::vector<std::complex<float>>& actual,
                        const std::vector<std::complex<double>>& expected);
} // namespace kws::test
