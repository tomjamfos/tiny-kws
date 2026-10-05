#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "log_mel.h"
#include "wav.h"

namespace py = pybind11;

namespace {

py::array_t<float> load_wav(const std::filesystem::path& path) {
    const std::vector<float> samples = kws::load_wav(path.string());
    return py::array_t<float>(static_cast<py::ssize_t>(samples.size()), samples.data());
}

py::array_t<float> mel_filterbank() {
    const std::vector<float> filterbank = kws::mel_filterbank();
    return py::array_t<float>({static_cast<py::ssize_t>(kws::kNumMelBins),
                               static_cast<py::ssize_t>(kws::kNumFreqBins)},
                              filterbank.data());
}

py::array_t<float> log_mel(const py::array& samples) {

    // Integer PCM would convert without the 1/32768 scaling, so reject it here.
    if (samples.dtype().kind() != 'f') {
        throw py::type_error("log_mel: expected a floating point array, got dtype " +
                             py::str(samples.dtype()).cast<std::string>());
    }

    // Any float width or memory layout becomes contiguous float32.
    const auto clip =
        py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(samples);
    if (!clip) {
        throw py::error_already_set();
    }

    if (clip.ndim() != 1) {
        throw std::invalid_argument(
            "log_mel: expected a one-dimensional array, got shape " +
            py::str(clip.attr("shape")).cast<std::string>());
    }

    // Clips shorter than one second are zero-padded, longer clips throw.
    const std::vector<float> input =
        kws::pad_clip(clip.data(), static_cast<std::size_t>(clip.shape(0)));

    // Uninitialised (kNumMelBins, kNumFrames) array.
    // process() writes every element.
    py::array_t<float> output({static_cast<py::ssize_t>(kws::kNumMelBins),
                               static_cast<py::ssize_t>(kws::kNumFrames)});
    static const kws::LogMel frontend;
    frontend.process(input.data(), output.mutable_data());
    return output;
}

} // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "tiny_kws native frontend";
    m.def("load_wav", &load_wav, py::arg("path"),
          "Samples of a wav file, float in [-1, 1).\n"
          "path: mono 16-bit PCM sampled at 16 kHz.\n"
          "Invalid formats raise ValueError, unreadable files raise RuntimeError.\n"
          "shape: (num_samples,), dtype: float32");
    m.def("mel_filterbank", &mel_filterbank,
          "Dense HTK mel filterbank, no normalisation.\n"
          "shape: (32, 257), mel bins x FFT bins (0 Hz to Nyquist), dtype: float32");
    m.def("log_mel", &log_mel, py::arg("clip"),
          "Log-mel features for one clip.\n"
          "clip: one-dimensional, at most 16000 samples at 16 kHz, float in [-1, 1).\n"
          "Shorter clips are zero-padded at the end, longer clips raise ValueError.\n"
          "shape: (32, 98), mel bins x frames, dtype: float32");
}
