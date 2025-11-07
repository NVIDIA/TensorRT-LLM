#!/bin/bash

set -ex

CUDA_VER="13"

if [ -n "${GITHUB_MIRROR}" ]; then
  BOOST_URL="${GITHUB_MIRROR%github-go-remote}sw-dl-triton-generic-local/triton/ci-cd/binaries/boost/1.80.0/boost_1_80_0.tar.gz"
else
  BOOST_URL="https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz"
fi

install_boost() {
  # Install boost version >= 1.78 for boost::span
  # Current libboost-dev apt packages are < 1.78, so install from tar.gz
  wget --retry-connrefused --timeout=180 --tries=10 --continue -O /tmp/boost.tar.gz ${BOOST_URL} \
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
      datacenter-gpu-manager-4-cuda${CUDA_VER} \
    && install_boost \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
}

# Install Triton only if base image is Ubuntu
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [ "$ID" == "ubuntu" ]; then
  install_triton_deps
else
  rm -rf /opt/tritonserver
  echo "Skip Triton installation for non-Ubuntu base image"
fi
