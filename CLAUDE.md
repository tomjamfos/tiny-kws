# Tiny Keyword Spotting
A lightweight keyword spotting classifier trained on Google Speech Commands v0.02
(12-class: ten target words, unknown, silence), exported to ONNX and run from a standalone C++ binary with no
PyTorch dependency at inference time. Log-mel features are computed by a hand-written, dependency-free C++ frontend,
used for both training (via pybind11) and inference.
Inference is CPU-only, with a Raspberry Pi 3 (ARM Cortex-A53, 1 GB RAM) as the performance floor.

## Repo layout
- `src/` -> C++ frontend: load wavs, FFT, log-mel.
- `tiny_kws/` -> Python package: dataset, model, training, ONNX export.
- `bindings/` -> Pybind11 module, installed as `tiny_kws._native`.
- `cli/` -> Standalone C++ inference binary (ONNX Runtime).
- `scripts/` -> One-off tooling: benchmarks, quantisation.
- `tests/` -> `tests/python/` for pytest (including frontend/torchaudio comparison tests), `tests/cpp/` for Catch2.
- `models/` -> Exported `.onnx` files (committed).
- `data/` -> Datasets (gitignored).
- `notebooks/` -> Experiments and results plots (outputs committed).
- `pyproject.toml` -> Scikit-build-core, builds the extension via CMake.
- `CMakeLists.txt` -> Builds `src/`, `bindings/`, `cli/`.

## Data
Google Speech Commands v0.02, at `data/datasets/speech-commands-v0.02` (gitignored, may be a symlink).
Source: `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`, extracted into that directory.
Structure: 35 word folders, `_background_noise_/` (longer recordings for silence and augmentation),
and `validation_list.txt` / `testing_list.txt` (official splits).
- Classes, in label order -> 0: `down`, 1: `go`, 2: `left`, 3: `no`, 4: `off`, 5: `on`, 6: `right`,
  7: `stop`, 8: `up`, 9: `yes`, 10: `unknown` (all other 25 words), 11: `silence`
  (one-second crops of `_background_noise_/`).
- Splits -> `validation_list.txt` and `testing_list.txt` are validation and test, everything else is training.
  The split is by speaker, so never re-split.
- Balance -> training uses a random subset of `unknown` clips, about as many as a single target word has.
  `silence` is generated to a similar count.
  Validation and test keep every `unknown` clip, and generate `silence` with a fixed seed.

## Frontend
Log-mel features, largely matching `torchaudio.transforms.MelSpectrogram` defaults (deviations noted).
- Input: 16kHz mono int16 PCM, divided by 32768 to float in `[-1, 1)`,
  shorter clips zero-padded at the end up to 16000 samples, longer clips rejected.
- Frames: 30ms (480 samples), 10ms (160 samples) hop, no centring. Produces 98 frames per clip.
  Framed by window length, which differs from torchaudio. It frames by `n_fft` (512) and centres the
  window in each frame, giving 97 frames offset by 16 samples. To compare, zero-pad the reference
  input by 16 samples at each end, 16 = (512 - 480) / 2.
- Window: Hann, periodic.
- FFT: 512-point, hand-rolled radix-2 complex, frame zero-padded.
- Spectrum: power.
- Mel: 32 bins, HTK scale, no filter normalisation, 20 to 8000 Hz (`f_min=20` differs from default).
- Log: `ln(max(x, 1e-6))`, applied separately since MelSpectrogram outputs linear power.
- Output: 32 (mel bins) by 98 (time frames), float32.
- torchaudio is the numerical reference for the frontend. Fix the C++ to match it, not the reverse.
  Framing is the one deliberate exception.

## Build
- Python 3.12, managed with uv. `pyproject.toml` uses scikit-build-core to build the pybind11 extension via CMake.
- C++17, CMake with Ninja. Developed on Apple clang / arm64, must stay portable to GCC on Linux (aarch64).
- Dependencies flow one way -> `src/` depends only on the C++ standard library. `bindings/` adds pybind11,
  `cli/` adds ONNX Runtime. Nothing in `src/` or `cli/` may depend on Python or torch.
- ONNX Runtime (for `cli/`) and Catch2 (for `tests/cpp/`) are fetched by CMake at configure time, not installed system-wide.
- Dependencies -> base: numpy, onnxruntime. `train` extra: torch, torchaudio, onnx.
  `dev` group: pybind11, pytest, ruff, plus train. `notebooks` group: jupyterlab, matplotlib.
  Torch is only imported inside training and export modules, never at package top level.
- Commands -> `uv sync` builds the Python package and extension. For C++ development:
  `cmake -B build/dev -G Ninja -DPython_EXECUTABLE=$PWD/.venv/bin/python -DTINY_KWS_BUILD_TESTS=ON -DTINY_KWS_BUILD_CLI=ON`,
  then `cmake --build build/dev` and `ctest --test-dir build/dev`.
  Changing the Python or any CMake option needs a clean configure (`rm -rf build/dev` first).
- CMake options (all default OFF) -> `TINY_KWS_BUILD_TESTS`, `TINY_KWS_BUILD_CLI`, `TINY_KWS_WERROR`, `TINY_KWS_SANITIZE`.

## Workflow
- Repo owner writes all repo content by default. 
  Make concrete code proposals in conversation, but do not write to files unless asked.
- Git is read-only (e.g. `status`, `log`, `diff`, `show`). 
  Do not run anything that writes, including staging, unless asked.
- Propose code in small logical steps at a time, avoid generating whole files or complete implementations.
- Point out bugs, style issues and design concerns directly, as soon as they're spotted.
  Formatting is not a review topic — clang-format handles it on save.
- Keep code proposals readable for a competent reader. 
  Advanced or otherwise non-obvious optimisations are welcome providing they represent real gain and are fully explained.
- Do not install packages or dependencies unless asked.
