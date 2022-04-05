cmake_minimum_required(VERSION 3.13.0)

project(doc)

find_package(Doxygen REQUIRED)

if (DOXYGEN_FOUND)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "{CMAKE_SOURCE_DIR}/README.md")
    doxygen_add_docs(doc ${CMAKE_SOURCE_DIR}/../include COMMENT "Generating documentation")
else()
    message(STATUS, "No doxygen found")
endif()
