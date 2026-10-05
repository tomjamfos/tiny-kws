include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.10.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# Reference implementations the tests compare against. Test-only, never part of tiny_kws_core.
add_library(tiny_kws_test_support STATIC
    naive_dft.cpp
    signals.cpp
    spectrum_error.cpp
)
target_include_directories(tiny_kws_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tiny_kws_test_support
    PUBLIC tiny_kws_core
    PRIVATE tiny_kws_warnings
)

add_executable(tiny_kws_tests
    fft_test.cpp
    log_mel_test.cpp
    naive_dft_test.cpp
    wav_test.cpp
)
target_link_libraries(tiny_kws_tests PRIVATE
    tiny_kws_test_support # brings in tiny_kws_core
    tiny_kws_warnings
    Catch2::Catch2WithMain
)

# One ctest entry per TEST_CASE, discovered by running the binary after it links.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(tiny_kws_tests)
