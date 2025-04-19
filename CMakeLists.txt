cmake_minimum_required(VERSION 3.18)
project(EndoscopicDenoiser LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_ARCHITECTURES 75)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
    add_definitions(-DDEBUG_CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_NVCC_FLAGS "${CMAKE_NVCC_FLAGS} -Wno-deprecated-gpu-targets -diag-suppress 611 --expt-relaxed-constexpr")

set(CPP_SOURCES
    src/main.cpp
)

set(CUDA_SOURCES
    src/filters/temporal_median_filter.cu
    src/filters/spatiotemporal_bilateral_filter.cu
    src/filters/optical_flow_blending.cu
    src/filters/deep_denoiser.cu
)

cuda_add_library(cuda_filters STATIC ${CUDA_SOURCES})

add_executable(denoiser ${CPP_SOURCES})

target_link_libraries(denoiser
    cuda_filters
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_static_LIBRARY}
)