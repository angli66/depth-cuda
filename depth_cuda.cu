#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "disparity_method.h"

namespace py = pybind11;

static bool initiated = false;

template <class T>
Mat2d<T> ndarray2Mat2d(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    auto ptr = static_cast<T *>(buf.ptr);
    Mat2d<T> new_arr(buf.shape[0], buf.shape[1], ptr);
    return new_arr;
}

template <class T>
py::array_t<T> Mat2d2ndarray(Mat2d<T> arr) {
    py::str NO_COPY; // Magic to let pybind create array without copying
    py::array_t<T> new_arr = py::array({arr.rows(), arr.cols()}, arr.data(), NO_COPY);
    return new_arr;
}

void init() {
    init_disparity_method(10, 120);
    initiated = true;
}

py::array_t<uint8_t> compute_disparity(py::array_t<uint8_t> left_ndarray, py::array_t<uint8_t> right_ndarray) {
    if (initiated == false) { throw std::runtime_error("init() must be called before calling compute_disparity()"); }
    if (MAX_DISPARITY != 128) { throw std::runtime_error("MAX_DISPARITY must be 128"); }
    if (PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8) { throw std::runtime_error("PATH_AGGREGATION must be 4 or 8"); }

    Mat2d<uint8_t> left = ndarray2Mat2d<uint8_t>(left_ndarray);
    Mat2d<uint8_t> right = ndarray2Mat2d<uint8_t>(right_ndarray);

    if (left.rows() != right.rows() || left.cols() != right.cols()) { throw std::runtime_error("Both images must have the same dimensions"); }
    if (left.rows() % 4 != 0 || left.cols() % 4 != 0) { throw std::runtime_error("Image width and height must be divisible by 4"); }

    Mat2d<uint8_t> disp = compute_disparity_method(left, right);

    py::array_t<uint8_t> disp_ndarray = Mat2d2ndarray<uint8_t>(disp);

    return disp_ndarray;
}

void finish() {
    finish_disparity_method();
}

PYBIND11_MODULE(depth_cuda, m) {
    m.def("init", &init);
    m.def("compute_disparity", &compute_disparity, py::return_value_policy::take_ownership);
    m.def("finish", &finish);
}