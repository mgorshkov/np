#!/bin/bash

ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

source ${ROOT_DIR}/build.properties

PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}.tgz
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}
URL=https://mgorshkov.jfrog.io/artifactory/default-generic-local/np/$PACKAGE_FULLNAME

curl -T $PACKAGE_PATH -u$USERNAME:$PASSWORD "$URL"
