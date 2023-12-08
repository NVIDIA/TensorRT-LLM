#!/bin/bash

set -ex

CCACHE_VERSION="4.8.3"
curl -L https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz | xz -d | tar -x -C /tmp/
cp /tmp/ccache-${CCACHE_VERSION}-linux-x86_64/ccache /usr/bin/ccache
