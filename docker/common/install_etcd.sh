#!/bin/bash

set -ex

ETCD_VER=v3.5.21

# choose either URL
DOWNLOAD_URL=https://storage.googleapis.com/etcd

# Detect CPU architecture
ARCH=$(uname -m)

# Map common architecture names to their standard forms
if [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
elif [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
fi

# Use a temporary location for downloading files
TMP_DIR=$(mktemp -d)
ETCD_TAR=${TMP_DIR}/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz

# Download etcd binaries
curl -L ${DOWNLOAD_URL}/${ETCD_VER}/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz -o ${ETCD_TAR}

# Extract binaries to /usr/local/bin directly to avoid unnecessary copying
tar xzvf ${ETCD_TAR} -C /usr/local/bin --strip-components=1

# Cleanup temporary files and directories
rm -rf ${TMP_DIR}
