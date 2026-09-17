#!/bin/bash

set -ex

# This script is used for reinstalling CUDA on Rocky Linux 8 with the run file.
# CUDA version is usually aligned with the latest NGC CUDA image tag.
# Only use when public CUDA image is not ready.
#
# Bring Rocky up to the CUDA the DLFW 26.08 image carries. The Rocky base image
# (nvcr.io/nvidia/cuda:13.3.1-devel-rockylinux8, docker/Makefile) is older -- nvcr.io
# publishes no 13.4 tag yet -- so this reinstall does run there; check_cuda_version
# below only skips it once the base image itself ships this version.
#
# NB: 13.4.1 dropped the driver version from the run file name -- it is
# cuda_13.4.1_linux.run, whereas every release up to 13.3.1 spelled it
# cuda_<ver>_<driver>_linux.run. Keep the toolkit version and the driver version
# separate so the file name does not depend on the latter.
CUDA_VER="13.4.1"
CUDA_DRIVER_VER="615.71.09"
CUDA_RUN_FILE="cuda_${CUDA_VER}_linux.run"

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

reinstall_rockylinux_cuda() {
    dnf -y install epel-release
    dnf remove -y "cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    rm -rf /usr/local/cuda-${OLD_CUDA_VER}
    wget --retry-connrefused --timeout=180 --tries=10 --continue https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/${CUDA_RUN_FILE}
    sh ${CUDA_RUN_FILE} --silent --override --toolkit
    rm -f ${CUDA_RUN_FILE}
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
  *)
    echo "Skip for other OS..."
    ;;
esac
