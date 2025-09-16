#!/bin/bash

set -ex

ARCH=$(uname -m)
CMAKE_VERSION="4.0.3"
GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

PARSED_CMAKE_VERSION=$(echo $CMAKE_VERSION | sed 's/\.[0-9]*$//')
CMAKE_FILE_NAME="cmake-${CMAKE_VERSION}-linux-${ARCH}"
RELEASE_URL_CMAKE=${GITHUB_URL}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_FILE_NAME}.tar.gz
wget --retry-connrefused --timeout=180 --tries=10 --continue ${RELEASE_URL_CMAKE} -P /tmp
tar -xf /tmp/${CMAKE_FILE_NAME}.tar.gz -C /usr/local/
ln -s /usr/local/${CMAKE_FILE_NAME} /usr/local/cmake

# Clean up temporary files
rm -rf /tmp/${CMAKE_FILE_NAME}.tar.gz
rm -rf /usr/local/${CMAKE_FILE_NAME}/doc
rm -rf /usr/local/${CMAKE_FILE_NAME}/man
rm -rf /usr/local/${CMAKE_FILE_NAME}/share/aclocal
rm -rf /usr/local/${CMAKE_FILE_NAME}/share/bash-completion
rm -rf /usr/local/${CMAKE_FILE_NAME}/share/emacs
rm -rf /usr/local/${CMAKE_FILE_NAME}/share/vim

echo 'export PATH=/usr/local/cmake/bin:$PATH' >> "${ENV}"
