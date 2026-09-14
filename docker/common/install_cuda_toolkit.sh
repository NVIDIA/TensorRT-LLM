#!/bin/bash

set -ex

# Reinstall CUDA from the run file on the non-DLFW base images. On most DLFW bumps
# nvcr.io/nvidia/cuda has no rockylinux/ubuntu tag yet for the CUDA the DLFW image carries,
# so those images start from an older public tag (docker/Makefile) and this script replaces
# their CUDA with CUDA_VER. Bump CUDA_VER with the DLFW image: the checks below skip the
# reinstall once the base image already ships it, which is what keeps this a no-op on DLFW.
#
# NB: newer CUDA releases dropped the driver version from the run file name
# (cuda_<ver>_linux.run, was cuda_<ver>_<driver>_linux.run), so keep the two versions in
# separate variables.
CUDA_VER="13.4.1"
CUDA_DRIVER_VER="615.71.09"

# The aarch64 run file carries an _sbsa suffix.
case "$(uname -m)" in
  aarch64) CUDA_RUN_FILE="cuda_${CUDA_VER}_linux_sbsa.run" ;;
  *)       CUDA_RUN_FILE="cuda_${CUDA_VER}_linux.run" ;;
esac

NVCC_VERSION_OUTPUT=$(nvcc --version)
OLD_CUDA_VER=$(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1)
echo "The version of pre-installed CUDA is ${OLD_CUDA_VER}."

check_cuda_version() {
    if [ -n "$CUDA_VERSION" ] && [ -n "$CUDA_DRIVER_VERSION" ]; then
        CUDA_VERSION_SHORT=$(echo "$CUDA_VERSION" | cut -d'.' -f1-3)
        if [ "$CUDA_VERSION_SHORT" = "$CUDA_VER" ] && [ "$CUDA_DRIVER_VERSION" = "$CUDA_DRIVER_VER" ]; then
            echo "CUDA version matches (${CUDA_VERSION_SHORT}_${CUDA_DRIVER_VERSION}), skipping reinstallation"
            return 0
        fi
    fi
    return 1
}

# The DLFW image is Ubuntu based, and its CUDA_DRIVER_VERSION is an internal build that is
# usually not published, so the Rocky check above would miss and wipe a good toolkit. Match
# on the toolkit version alone: that is all the run file replaces.
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
    # cuda-keyring this purge also takes out.
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
    if check_cuda_version; then
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
