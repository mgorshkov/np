cmake_minimum_required(VERSION 3.13.0)

set(DATA_GENERATOR data_generator)

project(${DATA_GENERATOR})

include_directories(../../include)

add_executable(${DATA_GENERATOR})

target_sources(${DATA_GENERATOR} PUBLIC main.cpp)

target_link_libraries(
    ${DATA_GENERATOR}
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${DATA_GENERATOR}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${DATA_GENERATOR}
)
