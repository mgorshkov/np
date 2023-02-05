cmake_minimum_required(VERSION 3.13.0)

project(np)

find_package(Doxygen REQUIRED)

if (DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "np")
    set(DOXYGEN_WARN_LOGFILE ${CMAKE_CURRENT_BINARY_DIR}/warnings.log)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/../README.md)
    set(DOXYGEN_EXCLUDE_PATTERNS "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
    set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYFILE_OUT ${DOXYGEN_OUTPUT_DIR}/Doxyfile)
    set(DOXYGEN_HTML_TIMESTAMP YES)
    set(DOXYGEN_HTML_DYNAMIC_SECTIONS YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_TREEVIEW_WIDTH 280)
    set(DOXYGEN_UML_LOOK YES)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_DOT_IMAGE_FORMAT svg)
    set(DOXYGEN_INTERACTIVE_SVG YES)

    if(NOT DEFINED CMAKE_INSTALL_DATAROOTDIR)
        set(CMAKE_INSTALL_DATAROOTDIR "share" CACHE PATH "read-only architecture-independent data root (share)")
    endif()

    doxygen_add_docs(doc ${CMAKE_CURRENT_SOURCE_DIR}/../README.md 
        ${CMAKE_CURRENT_SOURCE_DIR}/../CONTRIBUTING.md
        ${CMAKE_CURRENT_SOURCE_DIR}/../COPYING.md
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../samples
        WORKING_DIRECTORY ${PROJECT_DIR}
        COMMENT "Generating documentation")

    # Ensure that documentation is built before installing it
    install(CODE "execute_process(
        COMMAND ${CMAKE_COMMAND} --build \"${CMAKE_CURRENT_BINARY_DIR}\" --target doc
        WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\"
        )"
    )

    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}
        COMPONENT doc
    )

else()
    message(STATUS, "No doxygen found")
endif()
