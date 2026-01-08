#!/bin/bash

set -ex

CUDA_VER="13"

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
