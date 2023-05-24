cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(yadif)

set(DECKLINK_SDK_INCLUDE_DIR /home/meysam/BlackMagic/Blackmagic_DeckLink_SDK_12.4.2/Linux/include)


find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_86,code=sm_86)
    # -O3 -gencode arch=compute_62,code=sm_62) for tegra

# Add DeckLink API
SET(PLATFORM_SOURCES
${DECKLINK_SDK_INCLUDE_DIR}/DeckLinkAPI.h
${DECKLINK_SDK_INCLUDE_DIR}/DeckLinkAPIDispatch.cpp
)
LIST(APPEND PLATFORM_LIBRARIES dl pthread)

SET(APP_HEADERS
        capture_delegate.h
        Config.h
        bmcapture.h
        )

SET(APP_SOURCES
        main.cpp
        Config.cpp
        capture_delegate.cpp
        bmcapture.cpp
        ${APP_HEADERS}
        )




SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# TARGETS --------------------------------------------------------------------------------------------------------------
cuda_add_executable(${PROJECT_NAME} main.cpp yadif.cpp yadif_cuda.cu ${APP_SOURCES} ${PLATFORM_SOURCES} )
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")
target_include_directories(${PROJECT_NAME} PUBLIC   ${CUDA_INCLUDE_DIRS} 
                                                    ${DECKLINK_SDK_INCLUDE_DIR}
                                                    ${OpenCV_INCLUDE_DIRS}  )

target_link_libraries(${PROJECT_NAME}   ${CUDA_LIBRARIES} 
                                        ${CMAKE_THREAD_LIBS_INIT}  
                                        ${PLATFORM_LIBRARIES}
                                        ${OpenCV_LIBS} )
