#!/bin/bash
#
# deploy.sh — Upload the built package archive to the JFrog Artifactory
# repository.
#
# This script reads credentials (USERNAME, PASSWORD) from environment
# variables and uploads the .tgz archive produced by build.sh to the
# configured Artifactory server.
#
# Prerequisites:
#   - build.sh must have been run successfully first.
#   - The environment variables USERNAME and PASSWORD must be set.
#
# Usage:
#   USERNAME=<user> PASSWORD=<pass> ./scripts/deploy.sh
#
# Examples:
#   USERNAME=deploy PASSWORD=s3cret ./scripts/deploy.sh

ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

# Load version and package-name metadata.
source ${ROOT_DIR}/build.properties

PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}.tgz
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}

# Artifactory upload URL for the np package.
URL=https://mgorshkov.jfrog.io/artifactory/default-generic-local/np/$PACKAGE_FULLNAME

# Upload the package using HTTP PUT with basic auth.
curl -T $PACKAGE_PATH -u$USERNAME:$PASSWORD "$URL"
