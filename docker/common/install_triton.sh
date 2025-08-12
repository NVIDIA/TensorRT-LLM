#!/bin/bash

set -ex

install_boost() {
  # Install boost version >= 1.78 for boost::span
  # Current libboost-dev apt packages are < 1.78, so install from tar.gz
  wget -O /tmp/boost.tar.gz --timeout=180 --tries=3 https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
    && tar xzf /tmp/boost.tar.gz -C /tmp \
    && mv /tmp/boost_1_80_0/boost /usr/include/boost \
    && rm -rf /tmp/boost_1_80_0 /tmp/boost.tar.gz
}

install_triton_deps() {
  apt-get update \
    && apt-get install -y --no-install-recommends \
      pigz \
      libxml2-dev \
      libre2-dev \
      libnuma-dev \
      python3-build \
      libb64-dev \
      libarchive-dev \
    && install_boost \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
  # Copy /tmp/libdcgm.so* files back to /usr/lib/<arch>-linux-gnu/
  if [ -d /usr/lib/x86_64-linux-gnu ]; then
    cp -f /tmp/libdcgm.so* /usr/lib/x86_64-linux-gnu/ || true
  elif [ -d /usr/lib/aarch64-linux-gnu ]; then
    cp -f /tmp/libdcgm.so* /usr/lib/aarch64-linux-gnu/ || true
  else
    echo "Target /usr/lib directory for architecture not found, skipping libdcgm.so* copy"
  fi
}

# Install Triton only if base image is Ubuntu
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [ "$ID" == "ubuntu" ]; then
  install_triton_deps
else
  rm -rf /opt/tritonserver
  echo "Skip Triton installation for non-Ubuntu base image"
fi
