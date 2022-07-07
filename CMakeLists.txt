cmake_minimum_required(VERSION 3.6)
project(depth_cuda)

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ../3rd_party/pybind11/include
    )

link_directories(
    /usr/local/cuda/lib64
    )
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
  )

cuda_add_library(depth_cuda SHARED
  depth_cuda.cu median_filter.cu hamming_cost.cu disparity_method.cu costs.cu)

target_link_libraries(depth_cuda
  ${OpenCV_LIBS}
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(depth_cuda PROPERTIES PREFIX "")
