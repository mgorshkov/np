cmake_minimum_required(VERSION 3.18.0)

# Project name
set(PROJECT_NAME np)

# Set up project
project(${PROJECT_NAME} VERSION 0.1.5 DESCRIPTION "⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU" LANGUAGES CXX)

# Tune up various optimizations
# Enable CUDA support
option(ENABLE_CUDA "Enable CUDA optimizations" ON)
# Set up options for basic compile optimizations
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)
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

option(MAKE_DOC "Build Doxygen documentation" ON)

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
