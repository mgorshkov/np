cmake_minimum_required(VERSION 3.13.0)

set(NP_UNIT_TESTS np_unit_tests)

project(${NP_UNIT_TESTS} VERSION 0.0.5 DESCRIPTION "C++ Numpy Unit Tests")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR "bin" CACHE PATH "user executables (bin)")
endif()

if (MSVC)
    add_compile_options(/W4 /WX)
	add_definitions(-DNOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_definitions(-DTEST_DATA_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

include_directories(../include include ${googletest_SOURCE_DIR}/googletest/include)

add_executable(${NP_UNIT_TESTS})

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR} "src/*.cpp")
target_sources(${NP_UNIT_TESTS} PUBLIC ${SOURCES})

add_test(NAME ${NP_UNIT_TESTS} COMMAND tests)

target_link_libraries(
    ${NP_UNIT_TESTS}
    gtest
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${NP_UNIT_TESTS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${NP_UNIT_TESTS}
)