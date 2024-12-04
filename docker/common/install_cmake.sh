#!/bin/bash

set -ex

ARCH=$(uname -m)
CMAKE_VERSION="3.30.2"
GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

PARSED_CMAKE_VERSION=$(echo $CMAKE_VERSION | sed 's/\.[0-9]*$//')
CMAKE_FILE_NAME="cmake-${CMAKE_VERSION}-linux-${ARCH}"
RELEASE_URL_CMAKE=${GITHUB_URL}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_FILE_NAME}.tar.gz
wget --no-verbose ${RELEASE_URL_CMAKE} -P /tmp
tar -xf /tmp/${CMAKE_FILE_NAME}.tar.gz -C /usr/local/
ln -s /usr/local/${CMAKE_FILE_NAME} /usr/local/cmake

echo 'export PATH=$PATH:/usr/local/cmake/bin' >> "${ENV}"
