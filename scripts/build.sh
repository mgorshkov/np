#!/bin/bash
#
# build.sh — Build the np library and create a redistributable package.
#
# This script performs a full build pipeline:
#   1. Configure and compile the project with CMake (Release mode).
#   2. Assemble a package directory containing headers, libraries, samples,
#      unit tests, and supporting files.
#   3. Compress the package into a .tgz archive.
#
# The package version and name are read from build.properties at the project
# root.  The final archive is written to:
#   <project-root>/package/<package-name>-<version>.tgz
#
# Usage:
#   ./scripts/build.sh [additional CMake arguments...]
#
# Examples:
#   ./scripts/build.sh
#   ./scripts/build.sh -DBUILD_CUDA=ON

set -euo pipefail

# Resolve the project root directory (parent of the scripts/ folder).
ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

# Load version and package-name metadata.
source ${ROOT_DIR}/build.properties

# ------------------------------------------------------------------
# Paths derived from build.properties
# ------------------------------------------------------------------
PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}
PACKAGE_TAR=${PACKAGE_PATH}.tgz

echo "PACKAGE_TAR: $PACKAGE_TAR"

# ------------------------------------------------------------------
# build_package — Configure and compile the project with CMake.
#
# The build directory is wiped first to guarantee a clean release build.
# All extra arguments passed to this script are forwarded to cmake.
# ------------------------------------------------------------------
function build_package() {
    rm -rf ${ROOT_DIR}/build || return 1
    mkdir -p ${ROOT_DIR}/build || return 1
    cd ${ROOT_DIR}/build || return 1
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PACKAGE_PATH} "$@" || return 1
    cmake --build . --config Release -j
}

# ------------------------------------------------------------------
# copy — Copy source/header/data files from a project sub-directory
#        into the corresponding location under the package tree.
#
# Only files with recognised extensions (.hpp, .cpp, .md, .csv, .npy,
# .sh, .txt, .a, .lib) are copied.
#
# Arguments:
#   $1 — Source path relative to the project root.
#   $2 — Destination path under PACKAGE_PATH (defaults to $1).
# ------------------------------------------------------------------
function copy() {
    local path="$1"
    local dest="${2:-$PACKAGE_PATH/$path}"

    mkdir -p $dest

    find $path -maxdepth 1 -type f -regex ".*\.\(hpp\|cpp\|md\|csv\|npy\|sh\|txt\|a\|lib\)$" -exec cp {} $dest \;
}

# ------------------------------------------------------------------
# create_package — Assemble the package directory tree.
#
# For each folder listed in FOLDERS the copy() function is called,
# which replicates the project layout under PACKAGE_PATH.  A symlink
# without the version suffix is also created for convenience (e.g.
# package/np -> package/np-0.1.5).
# ------------------------------------------------------------------
function create_package() {
    rm -rf ${PACKAGE_PATH}
    rm -f ${PACKAGE_TAR}
    mkdir -p ${PACKAGE_PATH}
    cd ${PACKAGE_ROOT} || return 1
    rm -f ${PACKAGE_NAME}
    ln -s ${PACKAGE_FULLNAME} ${PACKAGE_NAME}
    cd ${ROOT_DIR} || return 1

    FOLDERS=(
        .
        include
        include/np
        include/np/internal
        include/np/ndarray
        include/np/ndarray/dynamic
        include/np/ndarray/dynamic/internal
        include/np/ndarray/internal
        include/np/ndarray/static
        include/np/ndarray/static/internal
        samples
        samples/data-generator
        samples/least-squares
        samples/monte-carlo
        scripts
        unit_tests
        unit_tests/include
        unit_tests/src
        unit_tests/test_data
    )
    for folder in "${FOLDERS[@]}"; do
        copy $folder
    done
    mkdir -p $PACKAGE_PATH/lib
    copy build/libnp.* $PACKAGE_PATH/lib

    return 0
}

# ------------------------------------------------------------------
# zip_package — Compress the package directory into a .tgz archive.
# ------------------------------------------------------------------
function zip_package() {
    rm -f ${PACKAGE_TAR} || return 1
    tar zcf ${PACKAGE_TAR} -C "$(dirname ${PACKAGE_PATH})" "$(basename ${PACKAGE_PATH})"

    return 0
}

# ------------------------------------------------------------------
# main — Run the full pipeline: build → package → compress.
#
# If build_package fails the script exits early without creating a
# partial package.
# ------------------------------------------------------------------
function main() {
    build_package || return 1
    create_package || return 1
    zip_package
}

main
