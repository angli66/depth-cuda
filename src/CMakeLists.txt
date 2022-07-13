cmake_minimum_required(VERSION 3.6)
project(engine)

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -lineinfo
  -gencode=arch=compute_75,code=sm_75
  -gencode=arch=compute_80,code=sm_80
  -gencode=arch=compute_86,code=sm_86
  -ftemplate-depth=1024
)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ../3rd_party/pybind11/include
)

link_directories(
  /usr/local/cuda/lib64
)

cuda_add_library(engine SHARED
  engine.cu costs.cu hamming_cost.cu median_filter.cu camera.cu depth_method.cu
)

target_link_libraries(engine
  ${OpenCV_LIBS}
  ${PYTHON_LIBRARIES}
  cudart
)

set_target_properties(engine PROPERTIES PREFIX "")
