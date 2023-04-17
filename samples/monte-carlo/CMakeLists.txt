cmake_minimum_required(VERSION 3.13.0)

set(MONTE_CARLO monte_carlo)

project(${MONTE_CARLO})

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
