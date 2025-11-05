#!/bin/bash

set -ex

TRT_VER="10.13.2.6"
# Align with the pre-installed cuDNN / cuBLAS / NCCL versions from
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html#rel-25-08
CUDA_VER="13.0" # 13.0.0
# Keep the installation for cuDNN if users want to install PyTorch with source codes.
# PyTorch 2.x can compile with cuDNN v9.
CUDNN_VER="9.12.0.46-1"
# NCCL version 2.26.x used in the NGC PyTorch 25.05 image but has a performance regression issue.
# Use NCCL version 2.27.5 which has the fixes.
NCCL_VER="2.27.7-1+cuda13.0"
# Use cuBLAS version 13.0.0.19 instead.
CUBLAS_VER="13.0.0.19-1"
# Align with the pre-installed CUDA / NVCC / NVRTC versions from
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
NVRTC_VER="13.0.48-1"
CUDA_RUNTIME="13.0.48-1"
CUDA_DRIVER_VERSION="580.65.06-1.el8"

for i in "$@"; do
    case $i in
        --TRT_VER=?*) TRT_VER="${i#*=}";;
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

    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    NVRTC_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')

    apt-get install -y --no-install-recommends \
        libcudnn9-cuda-13=${CUDNN_VER} \
        libcudnn9-dev-cuda-13=${CUDNN_VER} \
        libcudnn9-headers-cuda-13=${CUDNN_VER} \
        libnccl2=${NCCL_VER} \
        libnccl-dev=${NCCL_VER} \
        libcublas-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} \
        libcublas-dev-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} \
        cuda-nvrtc-dev-${NVRTC_CUDA_VERSION}=${NVRTC_VER}

    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

install_rockylinux_requirements() {
    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')

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
        "libcublas-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}" \
        "libcublas-devel-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}"; do
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
        libcublas-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}.rpm \
        libcublas-devel-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}.rpm

    # Clean up
    rm -f *.rpm
    dnf clean all
    nvcc --version
}

install_tensorrt() {
    PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")
    TRT_CUDA_VERSION=${CUDA_VER}
    TRT_VER_SHORT=$(echo $TRT_VER | cut -d. -f1-3)

    if [ -z "$RELEASE_URL_TRT" ];then
        ARCH=${TRT_TARGETARCH}
        if [ -z "$ARCH" ];then ARCH=$(uname -m);fi
        if [ "$ARCH" = "arm64" ];then ARCH="aarch64";fi
        if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
        RELEASE_URL_TRT="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TRT_VER_SHORT}/tars/TensorRT-${TRT_VER}.Linux.${ARCH}-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz"
    fi

    wget --retry-connrefused --timeout=180 --tries=10 --continue ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar
    tar -xf /tmp/TensorRT.tar -C /usr/local/
    mv /usr/local/TensorRT-${TRT_VER} /usr/local/tensorrt
    pip3 install --no-cache-dir /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
    rm -rf /tmp/TensorRT.tar
    echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> "${ENV}"

    rm -f /usr/local/tensorrt/lib/libnvinfer_vc_plugin_static.a \
          /usr/local/tensorrt/lib/libnvinfer_plugin_static.a \
          /usr/local/tensorrt/lib/libnvinfer_static.a \
          /usr/local/tensorrt/lib/libnvinfer_dispatch_static.a \
          /usr/local/tensorrt/lib/libnvinfer_lean_static.a \
          /usr/local/tensorrt/lib/libnvonnxparser_static.a \
          /usr/local/tensorrt/lib/libnvinfer_builder_resource_win.so.*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu_requirements
    install_tensorrt
    ;;
  rocky)
    install_rockylinux_requirements
    install_tensorrt
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
