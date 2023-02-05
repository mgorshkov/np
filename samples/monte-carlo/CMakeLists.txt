cmake_minimum_required(VERSION 3.13.0)

set(MONTE_CARLO monte_carlo)

project(${MONTE_CARLO})

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

include_directories(../../include)

add_executable(${MONTE_CARLO})

target_sources(${MONTE_CARLO} PUBLIC main.cpp)

target_link_libraries(
    ${MONTE_CARLO}
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${MONTE_CARLO}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${MONTE_CARLO}
)
