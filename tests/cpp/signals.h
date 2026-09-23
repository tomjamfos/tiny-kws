#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kws::test {

inline constexpr double kPi = 3.14159265358979323846;

std::vector<double> impulse(std::size_t n, std::size_t delay_index);

std::vector<double> tone(std::size_t n, std::size_t bin, double phase);

std::vector<double> noise_real(std::size_t n, std::uint32_t seed);

std::vector<std::complex<double>> noise_complex(std::size_t n, std::uint32_t seed);

std::vector<std::complex<float>> to_complex_float(const std::vector<double>& x);
std::vector<std::complex<float>> to_complex_float(const std::vector<std::complex<double>>& x);

} // namespace kws::test
