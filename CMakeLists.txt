cmake_minimum_required(VERSION 3.18.0)

# Project name
set(NP np)

# Set up project
project(${NP} VERSION 0.1.3 DESCRIPTION "C++ Numpy Implementation" LANGUAGES CXX CUDA)

# C++ 20
set(CMAKE_CXX_STANDARD 20)

# Enable CUDA language
enable_language(CUDA)

# CUDA specific settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # RTX 20xx/30xx/40xx
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
# Avoid "style of line directive is a GCC" nvcc warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-pedantic")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Default build type is Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR "bin" CACHE PATH "user executables (bin)")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
    add_definitions(-DNOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 9.1.0
)
FetchContent_MakeAvailable(fmt)

# Set up Open MP
option(ENABLE_OPENMP "Enable OpenMP" OFF)
if (ENABLE_OPENMP)
    message("OpenMP ON")
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

# Set up vector optimizations
option(ENABLE_SSE2 "Enable SSE2 optimizations" OFF)
option(ENABLE_SSE3 "Enable SSE3 optimizations" OFF)
option(ENABLE_AVX "Enable AVX optimizations" OFF)
option(ENABLE_AVX2 "Enable AVX2 optimizations" OFF)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (ENABLE_AVX512)
        add_compile_definitions(ENABLE_AVX512)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX512 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX512 optimization ON")
            add_compile_options(-O3 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX512 optimization ON")
            add_compile_options(/O2 /arch:AVX512)
        endif()
    endif()
    if (ENABLE_AVX2)
        add_compile_definitions(ENABLE_AVX2)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX2 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx2)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX2 optimization ON")
            add_compile_options(-O3 -msse4.2 -axAVX,core-avx2)
            add_compile_options(-axAVX -xAVX)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX2 optimization ON")
            add_compile_options(/O2 /arch:AVX2)
        endif()
    endif()
    if (ENABLE_AVX)
        add_compile_definitions(ENABLE_AVX)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, AVX optimization ON")
            add_compile_options(-O3 -ftree-vectorize -mavx)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, AVX optimization ON")
            add_compile_options(-Ofast -axAVX -xAVX)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, AVX optimization ON")
            add_compile_options(/O2 /arch:AVX)
        endif()
    endif()
    if (ENABLE_SSE3)
        add_compile_definitions(ENABLE_SSE3)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, SSE3 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -msse3)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            message("Intel compiler, SSE3 optimization ON")
            add_compile_options(-Ofast -axSSE3 -xSSE3)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, SSE3 optimization ON")
            add_compile_options(/O2 /arch:SSE2) # VC doesn't support /arch:SSE3
        endif()
    endif()
    if (ENABLE_SSE2)
        add_compile_definitions(ENABLE_SSE2)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message("GNU or Clang compiler, SSE2 optimization ON")
            add_compile_options(-O3 -ftree-vectorize -msse2)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            message("Intel compiler, SSE2 optimization ON")
            add_compile_options(-Ofast -axSSE2 -xSSE2)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message("MSVC compiler, SSE2 optimization ON")
            if (NOT CMAKE_CL_64)
                add_compile_options(/O2 /arch:SSE2)
            endif()
        endif()
    endif()
endif()

file(GLOB HEADER_FILES "include/np/*.hpp")
file(GLOB CUDA_SOURCE_FILES "src/cuda/*.cu")

# Create the library
add_library(${NP} ${CUDA_SOURCE_FILES})

target_include_directories(${NP} PUBLIC include)

set_target_properties(${NP} PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Add CUDA dependencies
target_link_libraries(${NP} INTERFACE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    fmt::fmt
)

# Add compile features
target_compile_features(${NP} INTERFACE cxx_std_20)

# Add compiler options
target_compile_options(${NP} INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

set(MAKE_DOC OFF)

# Make unit tests and sample if it's the main project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(MAKE_UNIT_TESTS ON)
else()
    set(MAKE_UNIT_TESTS OFF)
endif()

if (MAKE_UNIT_TESTS)
    message(STATUS "Enable testing in ${NP}")
    add_subdirectory(unit_tests)
    include (CTest)
    enable_testing()
endif()

if (MAKE_DOC)
    message(STATUS "Enable doxygen in ${NP}")
    add_subdirectory(doc)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/np
    DESTINATION include
)
