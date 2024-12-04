#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

ARCH=$(uname -m)
CCACHE_VERSION="4.8.3"
SYSTEM_ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')

if [[ $ARCH == *"x86_64"* ]] && [[ $SYSTEM_ID == *"centos"* ]]; then
  curl -L ${GITHUB_URL}/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-${ARCH}.tar.xz | xz -d | tar -x -C /tmp/
  cp /tmp/ccache-${CCACHE_VERSION}-linux-x86_64/ccache /usr/bin/ccache
fi
