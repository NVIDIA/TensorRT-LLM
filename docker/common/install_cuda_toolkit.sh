#!/bin/bash

set -ex

# Reinstall CUDA from the run file on the base images that do not already ship CUDA_VER. On
# most NGC PyTorch bumps nvcr.io/nvidia/cuda has no rockylinux/ubuntu tag yet for the CUDA the
# NGC PyTorch image carries, so those images start from an older public tag (docker/Makefile)
# and this script replaces their CUDA with CUDA_VER. Bump CUDA_VER with the NGC PyTorch image:
# the check below skips the reinstall once the base image already ships it, which is what keeps
# this a no-op on NGC PyTorch.
CUDA_VER="13.4.1"

# The aarch64 run file carries an _sbsa suffix.
case "$(uname -m)" in
  aarch64) CUDA_RUN_FILE="cuda_${CUDA_VER}_linux_sbsa.run" ;;
  *)       CUDA_RUN_FILE="cuda_${CUDA_VER}_linux.run" ;;
esac

NVCC_VERSION_OUTPUT=$(nvcc --version)
OLD_CUDA_VER=$(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1)
echo "The version of pre-installed CUDA is ${OLD_CUDA_VER}."

# Match on the toolkit version alone: that is all the run file replaces. A driver-version
# check works on neither base image family: the NGC CUDA images do not set CUDA_DRIVER_VERSION
# at all, and the NGC PyTorch one reports an internal build that is usually never published,
# so it would never match and would wipe a perfectly good toolkit.
check_cuda_toolkit_version() {
    if [ -n "$CUDA_VERSION" ] && [ "$(echo "$CUDA_VERSION" | cut -d'.' -f1-3)" = "$CUDA_VER" ]; then
        return 0
    fi
    return 1
}

reinstall_rockylinux_cuda() {
    dnf -y install epel-release
    dnf remove -y "cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    rm -rf /usr/local/cuda-${OLD_CUDA_VER}
    wget --retry-connrefused --timeout=180 --tries=10 --continue https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/${CUDA_RUN_FILE}
    sh ${CUDA_RUN_FILE} --silent --override --toolkit
    rm -f ${CUDA_RUN_FILE}
}

reinstall_ubuntu_cuda() {
    apt-get update
    # install_cuda_libs.sh runs next and puts cuDNN/NCCL/cuBLAS/NVRTC back, along with the
    # cuda-keyring and the cuda-compat holding libcuda.so.1 this purge also takes out.
    apt-get remove --purge -y --allow-change-held-packages \
        "cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" \
        "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    apt-get autoremove -y
    rm -rf /usr/local/cuda-${OLD_CUDA_VER}
    wget --retry-connrefused --timeout=180 --tries=10 --continue https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/${CUDA_RUN_FILE}
    sh ${CUDA_RUN_FILE} --silent --override --toolkit
    rm -f ${CUDA_RUN_FILE}
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  rocky)
    if check_cuda_toolkit_version; then
        echo "CUDA version matches ($CUDA_VER), skipping reinstallation"
        exit 0
    fi
    echo "Reinstall CUDA for RockyLinux 8..."
    reinstall_rockylinux_cuda
    ;;
  ubuntu)
    if check_cuda_toolkit_version; then
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
