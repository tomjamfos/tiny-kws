#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace kws {

// Iterative radix-2 decimation-in-time FFT over a complex-valued float32 buffer.
// The transform is unnormalised, with twiddle factors exp(-i*2*pi*k/n),
// matching numpy.fft.fft and torch.fft.fft. Arithmetic is float32 throughout.
// Twiddle factors are computed in double at construction and rounded to float once.
class Fft {
public:
    // n must be a power of two and >= 2
    explicit Fft(std::size_t n);

    std::size_t size() const noexcept { return n_; }

    // In-place transform of x[0, n).
    // x must point to at least size() elements, all of which are finite.
    void forward(std::complex<float>* x) const noexcept;

private:
    std::size_t n_;                             // size of the FFT
    std::vector<std::complex<float>> twiddles_; // exp(-i*2*pi*k/n), k in [0, n/2)
    std::vector<std::size_t> bit_reversal_;     // decimation-in-time input order table
};
} // namespace kws
