#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# Align with the pre-installed cuDNN / cuBLAS / NCCL versions from
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-06.html#rel-26-06
CUDA_VER="13.3" # 13.3.0
# Keep the installation for cuDNN if users want to install PyTorch with source codes.
# PyTorch 2.x can compile with cuDNN v9.
CUDNN_VER="9.23.0.39-1"
# NGC PyTorch 26.06 retains the CUDA 13.2 build of NCCL 2.30.4.
NCCL_VER="2.30.4-1+cuda13.2"
CUBLAS_VER="13.5.1.27-1"
# Align with the pre-installed CUDA / NVCC / NVRTC versions from
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
NVRTC_VER="13.3.33-1"
CUDA_RUNTIME="13.3.29-1"
CUDA_DRIVER_VERSION="610.43.02-1.el8"

for i in "$@"; do
    case $i in
        --CUDA_VER=?*) CUDA_VER="${i#*=}";;
        --CUDNN_VER=?*) CUDNN_VER="${i#*=}";;
        --NCCL_VER=?*) NCCL_VER="${i#*=}";;
        --CUBLAS_VER=?*) CUBLAS_VER="${i#*=}";;
        *) ;;
    esac
    shift
done

NVCC_VERSION_OUTPUT=$(nvcc --version)
if [[ $(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1) != ${CUDA_VER} ]]; then
  echo "The version of pre-installed CUDA is not equal to ${CUDA_VER}."
fi

install_ubuntu_requirements() {
    apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates
    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi

    # this file exists in cuda base image, and has conflicts with cuda-keyring with the following error, so we need to remove it first:
    # E: Conflicting values set for option Signed-By regarding
    # source https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/ /: /usr/share/keyrings/cuda-archive-keyring.gpg !=
    rm -f /etc/apt/sources.list.d/cuda.list

    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    apt-get update
    if [[ $(apt list --installed | grep libcudnn9) ]]; then
      apt-get remove --purge -y libcudnn9*
    fi
    if [[ $(apt list --installed | grep libnccl) ]]; then
      apt-get remove --purge -y --allow-change-held-packages libnccl*
    fi
    if [[ $(apt list --installed | grep libcublas) ]]; then
      apt-get remove --purge -y --allow-change-held-packages libcublas*
    fi
    if [[ $(apt list --installed | grep cuda-nvrtc-dev) ]]; then
      apt-get remove --purge -y --allow-change-held-packages cuda-nvrtc-dev*
    fi

    NVRTC_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    CUBLAS_MAJOR_VER=$(echo $CUBLAS_VER | cut -d. -f1)
    CUBLAS_PACKAGE="libcublas-${NVRTC_CUDA_VERSION}"
    CUBLAS_DEV_PACKAGE="libcublas-dev-${NVRTC_CUDA_VERSION}"

    apt-get install -y --no-install-recommends \
        libcudnn9-cuda-13=${CUDNN_VER} \
        libcudnn9-dev-cuda-13=${CUDNN_VER} \
        libcudnn9-headers-cuda-13=${CUDNN_VER} \
        libnccl2=${NCCL_VER} \
        libnccl-dev=${NCCL_VER} \
        ${CUBLAS_PACKAGE}=${CUBLAS_VER} \
        ${CUBLAS_DEV_PACKAGE}=${CUBLAS_VER} \
        cuda-nvrtc-dev-${NVRTC_CUDA_VERSION}=${NVRTC_VER}

    apt-get clean
    rm -rf /var/lib/apt/lists/*

    # cublas >= 13.4.1.2 installs headers to /usr/include/libcublas/<major>/
    # instead of /usr/local/cuda/include/. Symlink them back for build compatibility.
    CUBLAS_HDR_DIR="/usr/include/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_HDR_DIR}" ]; then
        for hdr in "${CUBLAS_HDR_DIR}"/*.h; do
            ln -sf "${hdr}" "/usr/local/cuda/include/$(basename ${hdr})"
        done
    fi

    # cublas >= 13.4.1.2 installs .so files to /usr/lib/<arch>-linux-gnu/libcublas/<major>/
    # instead of /usr/local/cuda/lib64/. Symlink them so LD_LIBRARY_PATH=/usr/local/cuda/lib64 finds them.
    ARCH=$(uname -m)
    CUBLAS_LIB_DIR="/usr/lib/${ARCH}-linux-gnu/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_LIB_DIR}" ]; then
        for lib in "${CUBLAS_LIB_DIR}"/libcublas*.so*; do
            [ -e "${lib}" ] || continue
            target="/usr/local/cuda/lib64/$(basename ${lib})"
            [ -e "${target}" ] || ln -sf "${lib}" "${target}"
        done
    fi
    ldconfig
}

install_rockylinux_requirements() {
    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    CUBLAS_MAJOR_VER=$(echo $CUBLAS_VER | cut -d. -f1)
    CUBLAS_PACKAGE="libcublas-${CUBLAS_CUDA_VERSION}"
    CUBLAS_DEV_PACKAGE="libcublas-devel-${CUBLAS_CUDA_VERSION}"

    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ];then ARCH1="x86_64" && ARCH2="x64" && ARCH3=$ARCH1;fi
    if [ "$ARCH" = "aarch64" ];then ARCH1="aarch64" && ARCH2="aarch64sbsa" && ARCH3="sbsa";fi

    # Download and install packages
    for pkg in \
        "libnccl-${NCCL_VER}.${ARCH1}" \
        "libnccl-devel-${NCCL_VER}.${ARCH1}" \
        "cuda-compat-${CUBLAS_CUDA_VERSION}-${CUDA_DRIVER_VERSION}.${ARCH1}" \
        "cuda-toolkit-${CUBLAS_CUDA_VERSION}-config-common-${CUDA_RUNTIME}.noarch" \
        "cuda-toolkit-13-config-common-${CUDA_RUNTIME}.noarch" \
        "cuda-toolkit-config-common-${CUDA_RUNTIME}.noarch" \
        "${CUBLAS_PACKAGE}-${CUBLAS_VER}.${ARCH1}" \
        "${CUBLAS_DEV_PACKAGE}-${CUBLAS_VER}.${ARCH1}"; do
        wget --retry-connrefused --timeout=180 --tries=10 --continue "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${ARCH3}/${pkg}.rpm"
    done

    # Remove old packages
    dnf remove -y "libnccl*"

    # Install new packages
    dnf -y install \
        libnccl-${NCCL_VER}.${ARCH1}.rpm \
        libnccl-devel-${NCCL_VER}.${ARCH1}.rpm \
        cuda-compat-${CUBLAS_CUDA_VERSION}-${CUDA_DRIVER_VERSION}.${ARCH1}.rpm \
        cuda-toolkit-${CUBLAS_CUDA_VERSION}-config-common-${CUDA_RUNTIME}.noarch.rpm \
        cuda-toolkit-13-config-common-${CUDA_RUNTIME}.noarch.rpm \
        cuda-toolkit-config-common-${CUDA_RUNTIME}.noarch.rpm \
        ${CUBLAS_PACKAGE}-${CUBLAS_VER}.${ARCH1}.rpm \
        ${CUBLAS_DEV_PACKAGE}-${CUBLAS_VER}.${ARCH1}.rpm

    # Clean up
    rm -f *.rpm
    dnf clean all
    nvcc --version

    # cublas >= 13.4.1.2 installs headers to /usr/include/libcublas/<major>/
    # instead of /usr/local/cuda/include/. Symlink them back for build compatibility.
    CUBLAS_HDR_DIR="/usr/include/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_HDR_DIR}" ]; then
        for hdr in "${CUBLAS_HDR_DIR}"/*.h; do
            ln -sf "${hdr}" "/usr/local/cuda/include/$(basename ${hdr})"
        done
    fi

    # cublas >= 13.4.1.2 installs .so files to /usr/lib64/libcublas/<major>/
    # instead of /usr/local/cuda/lib64/. Symlink them so LD_LIBRARY_PATH=/usr/local/cuda/lib64 finds them.
    CUBLAS_LIB_DIR="/usr/lib64/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_LIB_DIR}" ]; then
        for lib in "${CUBLAS_LIB_DIR}"/libcublas*.so*; do
            [ -e "${lib}" ] || continue
            target="/usr/local/cuda/lib64/$(basename ${lib})"
            [ -e "${target}" ] || ln -sf "${lib}" "${target}"
        done
    fi
    ldconfig
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu_requirements
    ;;
  rocky)
    install_rockylinux_requirements
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
