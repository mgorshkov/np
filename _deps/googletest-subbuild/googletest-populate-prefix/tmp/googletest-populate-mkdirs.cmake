# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/mike/git/np/_deps/googletest-src")
  file(MAKE_DIRECTORY "/home/mike/git/np/_deps/googletest-src")
endif()
file(MAKE_DIRECTORY
  "/home/mike/git/np/_deps/googletest-build"
  "/home/mike/git/np/_deps/googletest-subbuild/googletest-populate-prefix"
  "/home/mike/git/np/_deps/googletest-subbuild/googletest-populate-prefix/tmp"
  "/home/mike/git/np/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
  "/home/mike/git/np/_deps/googletest-subbuild/googletest-populate-prefix/src"
  "/home/mike/git/np/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/mike/git/np/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/mike/git/np/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
