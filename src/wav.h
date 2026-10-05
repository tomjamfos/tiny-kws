#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace kws {

// Decode the bytes of a RIFF WAVE file to float samples in [-1, 1), scaled by 1/32768.
// Clip length has no upper limit. Chunks other than fmt and data are skipped,
// and anything after the data chunk is ignored.
// Throws std::invalid_argument if the input isn't 16-bit PCM mono at
// 16 kHz, is malformed, or contains no samples.
std::vector<float> decode_wav(const std::byte* input, std::size_t num_bytes);

// Read the file at path into memory and decode it with decode_wav.
// Throws std::runtime_error if the file can't be opened or read,
// or std::invalid_argument if its contents can't be decoded.
std::vector<float> load_wav(const std::string& path);
} // namespace kws
