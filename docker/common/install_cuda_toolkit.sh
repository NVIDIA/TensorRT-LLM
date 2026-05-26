#!/bin/bash

set -ex

# This script is used for reinstalling CUDA on Rocky Linux 8 or Ubuntu with the run file.
# The 26.04 base image ships CUDA 13.2.x, but we intentionally downgrade the
# toolkit (nvcc / cudart / nvrtc) to the 26.02 stack version 13.1.1, while the
# higher-level libraries (TensorRT, cuBLAS, NCCL, cuDNN) are kept at their
# 26.04 versions via install_tensorrt.sh.
CUDA_VER="13.1.1_590.48.01"
CUDA_VER_SHORT="${CUDA_VER%_*}"

NVCC_VERSION_OUTPUT=$(nvcc --version)
OLD_CUDA_VER=$(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1)
echo "The version of pre-installed CUDA is ${OLD_CUDA_VER}."

check_cuda_version() {
    if [ -n "$CUDA_VERSION" ] && [ -n "$CUDA_DRIVER_VERSION" ]; then
        CUDA_VERSION_SHORT=$(echo "$CUDA_VERSION" | cut -d'.' -f1-3)
        ENV_CUDA_VER="${CUDA_VERSION_SHORT}_${CUDA_DRIVER_VERSION}"
        if [ "$ENV_CUDA_VER" = "$CUDA_VER" ]; then
            echo "CUDA version matches ($ENV_CUDA_VER), skipping reinstallation"
            return 0
        fi
    fi
    return 1
}

reinstall_rockylinux_cuda() {
    dnf -y install epel-release
    dnf remove -y "cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    rm -rf /usr/local/cuda-${OLD_CUDA_VER}
    wget --retry-connrefused --timeout=180 --tries=10 --continue https://developer.download.nvidia.com/compute/cuda/${CUDA_VER_SHORT}/local_installers/cuda_${CUDA_VER}_linux.run
    sh cuda_${CUDA_VER}_linux.run --silent --override --toolkit
    rm -f cuda_${CUDA_VER}_linux.run
}

reinstall_ubuntu_cuda() {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    # Remove existing CUDA toolkit and related libraries. `|| true` keeps the
    # script resilient when a pattern matches no installed package.
    apt-get remove --purge -y "cuda*" "libcublas*" "libcufft*" "libcufile*" "libcurand*" \
        "libcusolver*" "libcusparse*" "libnpp*" "libnvjpeg*" "nsight*" "libnvvm*" \
        "gds-tools*" || true
    apt-get autoremove --purge -y || true
    rm -rf /usr/local/cuda-${OLD_CUDA_VER}
    # Install prerequisites for the .run installer.
    apt-get install -y --no-install-recommends wget ca-certificates build-essential
    wget --retry-connrefused --timeout=180 --tries=10 --continue https://developer.download.nvidia.com/compute/cuda/${CUDA_VER_SHORT}/local_installers/cuda_${CUDA_VER}_linux.run
    sh cuda_${CUDA_VER}_linux.run --silent --override --toolkit
    rm -f cuda_${CUDA_VER}_linux.run
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  rocky)
    if check_cuda_version; then
        echo "CUDA version matches ($CUDA_VER), skipping reinstallation"
        exit 0
    fi
    echo "Reinstall CUDA for RockyLinux 8..."
    reinstall_rockylinux_cuda
    ;;
  ubuntu)
    if check_cuda_version; then
        echo "CUDA version matches ($CUDA_VER), skipping reinstallation"
        exit 0
    fi
    echo "Reinstall CUDA for Ubuntu..."
    reinstall_ubuntu_cuda
    ;;
  *)
    echo "Skip for other OS..."
    ;;
esac
