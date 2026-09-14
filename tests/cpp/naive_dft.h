#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace kws::test {

class NaiveDft {

public:
    explicit NaiveDft(std::size_t n);

    std::size_t size() const noexcept { return n_; }

    std::vector<std::complex<double>> forward(const std::vector<double>& x) const;

    std::vector<std::complex<double>>
    backward(const std::vector<std::complex<double>>& spectrum) const;

private:
    std::size_t n_;
    double inverse_n_;
    std::vector<std::complex<double>> twiddles_;
};
} // namespace kws::test
