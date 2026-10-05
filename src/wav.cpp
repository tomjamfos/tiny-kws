#include "wav.h"
#include "log_mel.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

namespace kws {

namespace {

// fmt chunk format tag for integer PCM.
constexpr std::uint16_t kPcmFormatTag = 0x0001;

// Reciprocal of full-scale int16, converting samples to [-1, 1). Exact in float.
constexpr float kPcm16ToFloatFactor = 1.0f / 32768.0f;

// Integers in a wav file are little-endian.
// Shifting bytes into place doesn't depend on host byte order.
std::uint16_t read_u16(const std::byte* data) {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[0]) |
                                      static_cast<std::uint16_t>(data[1]) << 8);
}
std::uint32_t read_u32(const std::byte* data) {
    return static_cast<std::uint32_t>(data[0]) |
           static_cast<std::uint32_t>(data[1]) << 8 |
           static_cast<std::uint32_t>(data[2]) << 16 |
           static_cast<std::uint32_t>(data[3]) << 24;
}

std::int16_t read_i16(const std::byte* data) {
    return static_cast<std::int16_t>(read_u16(data));
}

// Compare a chunk ID with an expected ID, e.g. "fmt ".
// IDs are four bytes with no terminator, so they can't be compared as C strings.
bool chunk_id_matches(const std::byte* data, const char* id) {
    return std::memcmp(data, id, 4) == 0;
}

} // namespace

std::vector<float> decode_wav(const std::byte* input, std::size_t num_bytes) {

    // A chunk header is a 4-byte ID followed by a 4-byte body size.
    constexpr std::size_t chunk_header_size = 8;

    // The RIFF chunk header is followed by the 4-byte "WAVE" form type.
    constexpr std::size_t riff_header_size = chunk_header_size + 4;
    if (num_bytes < riff_header_size || !chunk_id_matches(input, "RIFF") ||
        !chunk_id_matches(input + chunk_header_size, "WAVE")) {
        throw std::invalid_argument("decode_wav: invalid RIFF/WAVE header");
    }
    std::size_t offset = riff_header_size;

    bool valid_fmt = false;
    while (offset + chunk_header_size <= num_bytes) {

        const std::byte* subchunk = input + offset;

        const std::size_t body_size = read_u32(subchunk + 4);
        if (body_size > num_bytes - offset - chunk_header_size) {
            throw std::invalid_argument(
                "decode_wav: chunk body extends past end of input");
        }
        const std::byte* body = subchunk + chunk_header_size;

        if (chunk_id_matches(subchunk, "fmt ")) {

            if (body_size < 16) {
                throw std::invalid_argument(
                    "decode_wav: fmt chunk shorter than 16 bytes");
            }

            const std::uint16_t format_tag = read_u16(body);
            const std::uint16_t num_channels = read_u16(body + 2);
            const std::uint32_t sample_rate = read_u32(body + 4);
            const std::uint16_t bits_per_sample = read_u16(body + 14);

            if (format_tag != kPcmFormatTag || num_channels != 1 ||
                sample_rate != kSampleRate || bits_per_sample != 16) {

                throw std::invalid_argument(
                    "decode_wav: expected 16-bit PCM mono with a sample rate of " +
                    std::to_string(kSampleRate) + " Hz, got format tag " +
                    std::to_string(format_tag) + ", " + std::to_string(num_channels) +
                    " channels, " + std::to_string(sample_rate) + " Hz sample rate, " +
                    std::to_string(bits_per_sample) + "-bit");
            }

            valid_fmt = true;
        } else if (chunk_id_matches(subchunk, "data")) {
            if (!valid_fmt) {
                throw std::invalid_argument(
                    "decode_wav: no fmt chunk before data chunk");
            }

            if (body_size % 2 != 0) {
                throw std::invalid_argument(
                    "decode_wav: data chunk size isn't a whole number of samples");
            }

            const std::size_t num_samples = body_size / 2;
            if (num_samples == 0) {
                throw std::invalid_argument(
                    "decode_wav: data chunk contains no samples");
            }

            // Decode each 2-byte sample and rescale from [-32768, 32767] to [-1, 1).
            std::vector<float> output(num_samples);
            for (std::size_t n = 0; n < num_samples; ++n) {
                const std::int16_t sample = read_i16(body + 2 * n);
                output[n] = static_cast<float>(sample) * kPcm16ToFloatFactor;
            }
            return output;
        }

        // Chunks start on even offsets, so an odd-sized body
        // is followed by a pad byte that body_size doesn't count.
        offset += chunk_header_size + body_size + (body_size % 2);
    }
    throw std::invalid_argument("decode_wav: missing data chunk");
}

std::vector<float> load_wav(const std::string& path) {

    // Open the file in binary mode, with the get position at the end.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("load_wav: cannot open " + path);
    }

    // The get position is at the end, so its offset is the file size in bytes.
    const std::streamoff num_bytes = file.tellg();
    file.seekg(0);

    // Read the whole file in one call. read() sets failbit when it
    // extracts fewer bytes than requested, so a short read throws.
    std::vector<std::byte> file_bytes(static_cast<std::size_t>(num_bytes));
    if (!file.read(reinterpret_cast<char*>(file_bytes.data()),
                   static_cast<std::streamsize>(file_bytes.size()))) {
        throw std::runtime_error("load_wav: cannot read " + path);
    }

    return decode_wav(file_bytes.data(), file_bytes.size());
}

} // namespace kws
