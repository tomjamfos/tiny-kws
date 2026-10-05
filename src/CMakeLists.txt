add_library(tiny_kws_core STATIC
    fft.cpp
    log_mel.cpp
    wav.cpp
)
target_include_directories(tiny_kws_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tiny_kws_core PRIVATE tiny_kws_warnings)
