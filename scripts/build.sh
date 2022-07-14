#!/bin/bash

ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

source ${ROOT_DIR}/build.properties

PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}
PACKAGE_TAR=${PACKAGE_PATH}.tgz

echo "PACKAGE_TAR: $PACKAGE_TAR"

function copy() {
    local path="$1"
    local dest=$PACKAGE_PATH/$path

    mkdir -p $dest

    find -E $path -maxdepth 1 -type f -regex ".*.(hpp|cpp|md|csv|npy|sh|txt)$" -exec cp {} $dest \;
}

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

    return 0
}

function zip_package() {
    rm -f ${PACKAGE_TAR} || return 1
    tar zcf ${PACKAGE_TAR} -C "$(dirname ${PACKAGE_PATH})" "$(basename ${PACKAGE_PATH})"

    return 0
}

function main() {
    create_package || return 1
    zip_package
}

main