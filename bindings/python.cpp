#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "log_mel.h"

namespace py = pybind11;

namespace {

py::array_t<float> mel_filterbank() {
    const std::vector<float> filterbank = kws::mel_filterbank();
    return py::array_t<float>({static_cast<py::ssize_t>(kws::kNumMelBins),
                               static_cast<py::ssize_t>(kws::kNumFreqBins)},
                              filterbank.data());
}

} // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "tiny_kws native frontend";
    m.def("mel_filterbank", &mel_filterbank,
          "Dense HTK mel filterbank, no normalisation.\n"
          "shape: (32, 257), mel bins x FFT bins (0 Hz to Nyquist), dtype: float32");
}
