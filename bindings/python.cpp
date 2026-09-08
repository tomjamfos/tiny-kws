#include <pybind11/pybind11.h>

PYBIND11_MODULE(_native, m) { m.doc() = "tiny_kws native frontend"; }
