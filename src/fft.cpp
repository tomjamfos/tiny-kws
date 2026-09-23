#include "fft.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace kws {

namespace {

constexpr double kPi = 3.14159265358979323846;

// std::complex operator* compiled by clang or GCC incurs a NaN
// test and runtime helper call per multiply in order to handle
// infinite operands. forward() requires finite input.
inline std::complex<float> mul(std::complex<float> a, std::complex<float> b) noexcept {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
} // namespace

Fft::Fft(std::size_t n) : n_(n) {

    if (n < 2 || (n & (n - 1)) != 0) {
        throw std::invalid_argument(
            "Fft: n should be at least 2, and a power of 2, got " + std::to_string(n));
    }

    // std::log2 isn't guaranteed exact, so count bits instead.
    std::size_t bits = 0;
    while ((static_cast<std::size_t>(1) << bits) < n_) {
        ++bits;
    }

    // Precompute the bit-reversal permutation.
    bit_reversal_.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        std::size_t reversed = 0;
        for (std::size_t bit = 0; bit < bits; ++bit) {
            reversed = (reversed << 1) | ((i >> bit) & static_cast<std::size_t>(1));
        }
        bit_reversal_[i] = reversed;
    }

    // Twiddles computed in double and rounded to float only when stored.
    twiddles_.resize(n_ / 2);
    for (std::size_t k = 0; k < n_ / 2; ++k) {
        const double angle =
            2.0 * kPi * static_cast<double>(k) / static_cast<double>(n_);
        twiddles_[k] = {static_cast<float>(std::cos(angle)),
                        static_cast<float>(-std::sin(angle))};
    }
}

void Fft::forward(std::complex<float>* x) const noexcept {

    for (std::size_t i = 0; i < n_; ++i) {
        std::size_t j = bit_reversal_[i];

        // Only swap elements of each pair once.
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }

    // Each stage merges pairs of adjacent DFTs into DFTs twice the size.
    // The first half of each block holds the even-indexed sub-DFT and the
    // second half the odd-indexed. twiddles_ holds the nth roots of unity,
    // which each stage steps through in increments of stride.
    for (std::size_t block_size = 2; block_size <= n_; block_size <<= 1) {
        const std::size_t half_block_size = block_size / 2;
        const std::size_t stride = n_ / block_size;

        for (std::size_t block_start = 0; block_start < n_; block_start += block_size) {

            for (std::size_t j = 0; j < half_block_size; ++j) {

                const std::complex<float> even = x[block_start + j];
                const std::complex<float> rotated_odd =
                    mul(twiddles_[j * stride], x[block_start + half_block_size + j]);
                x[block_start + j] = even + rotated_odd;
                x[block_start + half_block_size + j] = even - rotated_odd;
            }
        }
    }
}

} // namespace kws
