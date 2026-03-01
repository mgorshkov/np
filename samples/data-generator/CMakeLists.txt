cmake_minimum_required(VERSION 3.18.0)

set(DATA_GENERATOR data_generator)

project(${DATA_GENERATOR})

# C++ 20
set(CMAKE_CXX_STANDARD 20)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 9.1.0
)
FetchContent_MakeAvailable(fmt)

find_library(NP_LIBRARY
    NAMES np libnp
    PATHS
        ../../build
        /usr/local/lib
        /usr/lib
    NO_DEFAULT_PATH
)
if(NOT NP_LIBRARY)
    message(FATAL_ERROR "NP library not found! Build parent np first.")
endif()

message(STATUS "Found NP: ${NP_LIBRARY}")

include_directories(../../include)

add_executable(${DATA_GENERATOR} main.cpp)

target_link_libraries(
    ${DATA_GENERATOR}
    ${PTHREAD}
    ${NP_LIBRARY}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    fmt::fmt
)

install(
    TARGETS ${DATA_GENERATOR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${DATA_GENERATOR}
)
