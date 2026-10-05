#include "wav.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kws::test {

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;

namespace {

void append_u16(std::vector<std::byte>& wav, std::uint16_t value) {
    wav.push_back(static_cast<std::byte>(value & 0xFF));
    wav.push_back(static_cast<std::byte>(value >> 8));
}

void append_u32(std::vector<std::byte>& wav, std::uint32_t value) {
    append_u16(wav, static_cast<std::uint16_t>(value & 0xFFFF));
    append_u16(wav, static_cast<std::uint16_t>(value >> 16));
}

void append_chunk_id(std::vector<std::byte>& wav, const char* id) {
    for (std::size_t i = 0; i < 4; ++i) {
        wav.push_back(static_cast<std::byte>(id[i]));
    }
}

void append_data_chunk(std::vector<std::byte>& wav,
                       const std::vector<std::int16_t>& samples) {
    append_chunk_id(wav, "data");
    append_u32(wav, static_cast<std::uint32_t>(samples.size() * 2));
    for (std::size_t n = 0; n < samples.size(); ++n) {
        append_u16(wav, static_cast<std::uint16_t>(samples[n]));
    }
}

struct WavFormat {
    std::uint16_t format_tag = 1; // integer PCM
    std::uint16_t num_channels = 1;
    std::uint32_t sample_rate = 16000;
    std::uint16_t bits_per_sample = 16;
};

void append_fmt_chunk(std::vector<std::byte>& wav, const WavFormat& fmt) {
    const std::uint16_t block_align =
        static_cast<std::uint16_t>(fmt.num_channels * fmt.bits_per_sample / 8);
    append_chunk_id(wav, "fmt "); // chunk ID
    append_u32(wav, 16);          // body size
    append_u16(wav, fmt.format_tag);
    append_u16(wav, fmt.num_channels);
    append_u32(wav, fmt.sample_rate);
    append_u32(wav, fmt.sample_rate * block_align); // byte rate
    append_u16(wav, block_align);
    append_u16(wav, fmt.bits_per_sample);
}

void append_riff_header(std::vector<std::byte>& wav) {
    append_chunk_id(wav, "RIFF");
    append_u32(wav, 0);
    append_chunk_id(wav, "WAVE");
}

std::vector<std::byte> minimal_wav(const WavFormat& fmt,
                                   const std::vector<std::int16_t>& samples) {
    std::vector<std::byte> wav;
    append_riff_header(wav);
    append_fmt_chunk(wav, fmt);
    append_data_chunk(wav, samples);
    return wav;
}

const std::vector<std::int16_t> kPcmEdges = {0, 1, -1, 32767, -32768};

} // namespace

TEST_CASE("decode_wav int16 edge values", "[wav]") {

    std::vector<std::byte> wav = minimal_wav(WavFormat{}, kPcmEdges);
    std::vector<float> output = decode_wav(wav.data(), wav.size());

    REQUIRE(output.size() == kPcmEdges.size());
    for (std::size_t n = 0; n < kPcmEdges.size(); ++n) {
        CAPTURE(n);

        REQUIRE(output[n] == static_cast<float>(kPcmEdges[n]) / 32768.0f);
    }
}

TEST_CASE("decode_wav skips unknown chunks", "[wav]") {

    std::vector<std::byte> wav;
    append_riff_header(wav);
    append_fmt_chunk(wav, WavFormat{});
    append_chunk_id(wav, "LIST");

    SECTION("even-sized body") {
        append_u32(wav, 4);
        for (std::size_t i = 0; i < 4; ++i) {
            wav.push_back(std::byte{0});
        }
        append_data_chunk(wav, kPcmEdges);
        std::vector<std::byte> expected = minimal_wav(WavFormat{}, kPcmEdges);
        REQUIRE(decode_wav(wav.data(), wav.size()) ==
                decode_wav(expected.data(), expected.size()));
    }

    SECTION("odd-sized body with a pad byte") {
        append_u32(wav, 3);
        for (std::size_t i = 0; i < 3; ++i) {
            wav.push_back(std::byte{0});
        }
        wav.push_back(std::byte{0}); // pad byte, not counted in the size
        append_data_chunk(wav, kPcmEdges);
        std::vector<std::byte> expected = minimal_wav(WavFormat{}, kPcmEdges);
        REQUIRE(decode_wav(wav.data(), wav.size()) ==
                decode_wav(expected.data(), expected.size()));
    }
}

TEST_CASE("decode_wav throws on malformed input", "[wav]") {

    SECTION("empty input") {
        REQUIRE_THROWS_MATCHES(decode_wav(nullptr, 0), std::invalid_argument,
                               MessageMatches(ContainsSubstring("RIFF/WAVE header")));
    }

    SECTION("incomplete RIFF header") {
        std::vector<std::byte> wav;
        append_riff_header(wav);
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size() - 1),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("RIFF/WAVE header")));
    }

    SECTION("RIFF chunk ID mismatch") {
        std::vector<std::byte> wav = minimal_wav(WavFormat{}, kPcmEdges);
        wav[0] = std::byte{'X'};
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size()),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("RIFF/WAVE header")));
    }

    SECTION("WAVE form type mismatch") {
        std::vector<std::byte> wav = minimal_wav(WavFormat{}, kPcmEdges);
        wav[8] = std::byte{'X'};
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size()),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("RIFF/WAVE header")));
    }

    SECTION("fmt chunk shorter than 16 bytes") {
        std::vector<std::byte> wav;
        append_riff_header(wav);
        append_chunk_id(wav, "fmt ");
        append_u32(wav, 14); // short by 2
        for (std::size_t i = 0; i < 14; ++i) {
            wav.push_back(std::byte{0});
        }
        append_data_chunk(wav, kPcmEdges);
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size()),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("shorter than 16")));
    }

    SECTION("missing data chunk") {
        std::vector<std::byte> wav;
        append_riff_header(wav);
        append_fmt_chunk(wav, WavFormat{});
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size()),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("missing data chunk")));
    }

    SECTION("no fmt chunk before data chunk") {
        std::vector<std::byte> wav;
        append_riff_header(wav);
        append_data_chunk(wav, kPcmEdges);
        append_fmt_chunk(wav, WavFormat{});
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size()),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("no fmt chunk")));
    }

    SECTION("data chunk body extends past end of input") {
        std::vector<std::byte> wav = minimal_wav(WavFormat{}, kPcmEdges);
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size() - 1),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("extends past end")));
    }

    SECTION("data chunk contains no samples") {
        std::vector<std::byte> wav = minimal_wav(WavFormat{}, {});
        REQUIRE_THROWS_MATCHES(
            decode_wav(wav.data(), wav.size()), std::invalid_argument,
            MessageMatches(ContainsSubstring("contains no samples")));
    }

    SECTION("unknown chunk body extends past end of input") {
        std::vector<std::byte> wav;
        append_riff_header(wav);
        append_chunk_id(wav, "LIST");
        append_u32(wav, 0xFFFFFFFF); // 2^32 - 1: the largest value uint32_t can hold.
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size()),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("extends past end")));
    }

    SECTION("data chunk size isn't a whole number of samples") {
        std::vector<std::byte> wav;
        append_riff_header(wav);
        append_fmt_chunk(wav, WavFormat{});
        append_chunk_id(wav, "data");
        append_u32(wav, 3);
        for (std::size_t i = 0; i < 3; ++i) {
            wav.push_back(std::byte{0});
        }
        REQUIRE_THROWS_MATCHES(
            decode_wav(wav.data(), wav.size()), std::invalid_argument,
            MessageMatches(ContainsSubstring("isn't a whole number of samples")));
    }
}

TEST_CASE("decode_wav throws on unsupported formats", "[wav]") {
    const WavFormat formats[] = {
        {3, 1, 16000, 16}, // float format tag
        {1, 2, 16000, 16}, // stereo
        {1, 1, 8000, 16},  // 8 kHz sample rate
        {1, 1, 16000, 8}   // 8-bit
    };
    for (const WavFormat& fmt : formats) {
        CAPTURE(fmt.format_tag, fmt.num_channels, fmt.sample_rate, fmt.bits_per_sample);
        std::vector<std::byte> wav = minimal_wav(fmt, kPcmEdges);
        REQUIRE_THROWS_MATCHES(decode_wav(wav.data(), wav.size()),
                               std::invalid_argument,
                               MessageMatches(ContainsSubstring("16-bit PCM mono")));
    }
}

TEST_CASE("load_wav throws on a missing file", "[wav]") {
    REQUIRE_THROWS_AS(load_wav("missing_file.wav"), std::runtime_error);
}

} // namespace kws::test
