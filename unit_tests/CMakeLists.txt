cmake_minimum_required(VERSION 3.13.0)

set(NP_UNIT_TESTS np_unit_tests)

project(${NP_UNIT_TESTS} VERSION 0.0.7 DESCRIPTION "C++ Numpy Unit Tests")

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