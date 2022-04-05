cmake_minimum_required(VERSION 3.13.0)

set(NP np)

project(${NP} VERSION 0.0.1 DESCRIPTION "C++ Numpy Implementation")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_library(${NP} INTERFACE)

target_include_directories(${NP} INTERFACE $<BUILD_INTERFACE:include>)

# Make unit tests and doc if it's the main project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(MAKE_UNIT_TESTS, "Make unit tests" ON)
    option(MAKE_DOC, "Make doc" ON)
else()
    option(MAKE_UNIT_TESTS, "Make unit tests" OFF)
    option(MAKE_DOC, "Make doc" OFF)
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