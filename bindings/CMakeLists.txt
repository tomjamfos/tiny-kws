find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native python.cpp)
target_link_libraries(_native PRIVATE tiny_kws_core tiny_kws_warnings)
install(TARGETS _native DESTINATION tiny_kws)
