cmake_minimum_required(VERSION 3.18.0)

# Project name
set(PROJECT_NAME np)

# Set up project
project(${PROJECT_NAME} VERSION 0.1.9 DESCRIPTION "⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU" LANGUAGES CXX)

# Tune up various optimizations
# Enable CUDA support
option(ENABLE_CUDA "Enable CUDA optimizations" ON)
# Set up options for basic compile optimizations
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)
# AVX-512 sub-features (Ice Lake+ only). ENABLE_AVX512 alone enables the base
# set (F/DQ/BW/VL) so the library runs on any AVX-512 CPU (e.g. Cascade Lake).
# VBMI/VBMI2 are NOT present on Cascade Lake; enabling them without runtime
# dispatch support crashes with SIGILL on older AVX-512 CPUs.
option(ENABLE_AVX512_VBMI "Enable AVX512-VBMI (byte permutes; Ice Lake+) optimizations" OFF)
option(ENABLE_AVX512_VBMI2 "Enable AVX512-VBMI2 (byte compress/expand; Ice Lake+) optimizations" OFF)

option(ENABLE_AMX "Enable AMX (Advanced Matrix Extensions) optimizations" OFF)
# Set up options for OpenMP support
option(ENABLE_OPENMP "Enable OpenMP" ON)

include(FetchContent)

FetchContent_Declare(
    cmake
    GIT_REPOSITORY https://github.com/mgorshkov/cmake.git
    GIT_TAG main
)
FetchContent_MakeAvailable(cmake)

# Apply global cmake options
include(${cmake_SOURCE_DIR}/Config.cmake)

file(GLOB HEADER_FILES "include/np/*.hpp")

add_subdirectory(src)

# Apply target-dependent cmake options
include(${cmake_SOURCE_DIR}/Target.cmake)

target_include_directories(${PROJECT_NAME} PUBLIC include)

# The CUDA toolkit may be detected with an older host compiler than
# CMAKE_CXX_COMPILER (nvcc 12.x supports gcc <= 13, while the project may be
# built with a newer gcc). The CUDA toolkit then contributes the host
# compiler's libstdc++ directory to every link line, where it shadows the
# libstdc++ of the C++ compiler that compiled the C++ objects and breaks the
# link (e.g. undefined __cxa_call_terminate when gcc 13's runtime is picked for
# gcc 15 objects). Prepend the C++ compiler's own libstdc++ directory so the
# matching runtime wins; when both compilers are the same this is a no-op.
if(ENABLE_CUDA AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE NP_LIBSTDCXX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(IS_ABSOLUTE "${NP_LIBSTDCXX_PATH}")
        get_filename_component(NP_LIBSTDCXX_DIR "${NP_LIBSTDCXX_PATH}" DIRECTORY)
        add_link_options("-L${NP_LIBSTDCXX_DIR}")
    endif()
    unset(NP_LIBSTDCXX_PATH)
    unset(NP_LIBSTDCXX_DIR)
endif()

set(MAKE_DOC OFF)

# Make unit tests and sample if it's the main project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(MAKE_UNIT_TESTS ON)
else()
    set(MAKE_UNIT_TESTS OFF)
endif()

if (MAKE_UNIT_TESTS)
    message(STATUS "${PROJECT_NAME}: Enable testing")
    add_subdirectory(unit_tests)
    include (CTest)
    enable_testing()
endif()

if (MAKE_DOC)
    message(STATUS "${PROJECT_NAME}: Enable doxygen")
    add_subdirectory(doc)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/np
    DESTINATION include
)
