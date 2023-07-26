cmake_minimum_required(VERSION 3.12)

project(softmax)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++14)
add_definitions(-O2 -pthread)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb ")

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter -g)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_87,code=sm_87")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

include_directories(include)
find_package(CUDA REQUIRED)
cuda_add_executable(softmax1 src/softmax1.cu)
#target_compile_features(softmax1 PUBLIC cuda_std_14)
target_link_libraries(softmax1 cudart)

