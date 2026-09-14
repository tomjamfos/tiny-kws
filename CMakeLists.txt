cmake_minimum_required(VERSION 3.20)
project(tiny_kws LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TINY_KWS_BUILD_TESTS "Build C++ tests (fetches Catch2)" OFF)
option(TINY_KWS_BUILD_CLI "Build inference CLI (fetches ONNX Runtime)" OFF)
option(TINY_KWS_WERROR "Treat warnings as errors" OFF)
option(TINY_KWS_SANITIZE "Enable address and UB sanitisers" OFF)

# Warnings. -ffast-math is deliberately absent: it reorders and approximates float arithmetic,
# and the frontend is verified to agree with torchaudio to about 1e-4.
add_library(tiny_kws_warnings INTERFACE)
target_compile_options(tiny_kws_warnings INTERFACE
    -Wall -Wextra -Wpedantic
    -Wconversion -Wsign-conversion
    -Wshadow -Wold-style-cast -Wnon-virtual-dtor
    $<$<BOOL:${TINY_KWS_WERROR}>:-Werror>
)
if(TINY_KWS_SANITIZE)
    target_compile_options(tiny_kws_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(tiny_kws_warnings INTERFACE -fsanitize=address,undefined)
endif()

add_subdirectory(src)
add_subdirectory(bindings)
if(TINY_KWS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
if(TINY_KWS_BUILD_CLI)
    add_subdirectory(cli)
endif()
