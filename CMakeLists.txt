cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(yadif)

find_package(Threads)
find_package(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_86,code=sm_86)
    # -O3 -gencode arch=compute_62,code=sm_62) for tegra


SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# TARGETS --------------------------------------------------------------------------------------------------------------
cuda_add_executable(${PROJECT_NAME} main.cpp yadif.cpp yadif_cuda.cu)
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
