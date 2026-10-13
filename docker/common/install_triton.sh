#!/bin/bash

set -ex

CUDA_VER="13"

setup_cuda_apt_repo() {
  # datacenter-gpu-manager-4-cuda* is published in the NVIDIA CUDA apt repo,
  # which is not configured in every base image.
  local ubuntu_ver distro arch keyring
  ubuntu_ver=$(grep -oP '(?<=^VERSION_ID=).+' /etc/os-release | tr -d '".')
  distro="ubuntu${ubuntu_ver}"
  case "$(uname -m)" in
    x86_64) arch="x86_64" ;;
    aarch64) arch="sbsa" ;;
    *) echo "Unsupported architecture $(uname -m)"; exit 1 ;;
  esac
  keyring="cuda-keyring_1.1-1_all.deb"
  wget --no-verbose --retry-connrefused --timeout=180 --tries=10 -O "/tmp/${keyring}" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/${keyring}"
  dpkg -i "/tmp/${keyring}"
  rm -f "/tmp/${keyring}"
}

install_triton_deps() {
  setup_cuda_apt_repo
  apt-get update \
    && apt-get install -y --no-install-recommends \
      pigz \
      libxml2-dev \
      libre2-dev \
      libnuma-dev \
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
