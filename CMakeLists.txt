cmake_minimum_required(VERSION 3.13.0)

set(NP np)

project(${NP} VERSION 0.0.4 DESCRIPTION "C++ Numpy Implementation")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (MSVC)
    add_compile_options(/W4 /WX)
    add_definitions(-DNOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_library(${NP} INTERFACE)

target_include_directories(${NP} INTERFACE $<BUILD_INTERFACE:include>)

# Make unit tests and doc if it's the main project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(MAKE_UNIT_TESTS ON)
    set(MAKE_DOC ON)
    set(MAKE_SAMPLE ON)
else()
    set(MAKE_UNIT_TESTS OFF)
    set(MAKE_DOC OFF)
    set(MAKE_SAMPLE OFF)
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

if (MAKE_SAMPLE)
    message(STATUS "Enable samples build in ${NP}")
    add_subdirectory(samples/monte-carlo)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/np
    DESTINATION include
)