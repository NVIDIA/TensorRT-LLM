#!/bin/bash

set -ex

install_boost() {
  # Install boost version >= 1.78 for boost::span
  # Current libboost-dev apt packages are < 1.78, so install from tar.gz
  wget -O /tmp/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz && (cd /tmp && tar xzf boost.tar.gz) && mv /tmp/boost_1_80_0/boost /usr/include/boost
  rm -rf /tmp/boost_1_80_0
  rm -rf /tmp/boost.tar.gz
}

install_triton_deps() {
  apt-get update && apt-get install -y \
    pigz \
    libxml2-dev \
    libre2-dev \
    libnuma-dev \
    python3-build \
    libb64-dev \
    libarchive-dev \
    datacenter-gpu-manager=1:3.3.6

  install_boost
}

# Install Triton only if base image in Ubuntu
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [ "$INSTALL_TRITON" == "1" ]; then
  if [ "$ID" == "ubuntu" ]; then
    install_triton_deps
  else
    rm -rf /opt/tritonserver
    echo "Skip Triton installation for non-Ubuntu base image"
  fi
else
  echo "Skip Triton installation when INSTALL_TRITON is set to 0"
  rm -rf /opt/tritonserver
fi
