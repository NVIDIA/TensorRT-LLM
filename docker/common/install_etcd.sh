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

rm -f /tmp/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz
rm -rf /tmp/etcd-download-test && mkdir -p /tmp/etcd-download-test

curl -L ${DOWNLOAD_URL}/${ETCD_VER}/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz -o /tmp/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz
tar xzvf /tmp/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz -C /tmp/etcd-download-test --strip-components=1
rm -f /tmp/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz

mv /tmp/etcd-download-test/* /usr/local/bin/

rm -rf /tmp/etcd-download-test
