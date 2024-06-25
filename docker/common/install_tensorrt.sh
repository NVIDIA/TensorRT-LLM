#!/bin/bash

set -ex

TRT_VER="10.1.0.27"
# Align with the pre-installed cuDNN / cuBLAS / NCCL versions from
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.html#rel-24-05
CUDA_VER="12.4" # 12.4.1
# Keep the installation for cuDNN if users want to install PyTorch with source codes.
# PyTorch 2.3.x can compile with cuDNN v9.
CUDNN_VER="9.1.0.70-1"
NCCL_VER="2.21.5-1+cuda12.4"
CUBLAS_VER="12.4.5.8-1"
# Align with the pre-installed CUDA / NVCC / NVRTC versions from
# https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html
NVRTC_VER="12.4.127-1"

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
  exit 1
fi

install_ubuntu_requirements() {
    apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates
    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${ARCH}/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb

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
    apt-get install -y --no-install-recommends libcudnn9-cuda-12=${CUDNN_VER} libcudnn9-dev-cuda-12=${CUDNN_VER}
    apt-get install -y --no-install-recommends libnccl2=${NCCL_VER} libnccl-dev=${NCCL_VER}
    apt-get install -y --no-install-recommends libcublas-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} libcublas-dev-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER}
    # NVRTC static library doesn't exist in NGC PyTorch container.
    NVRTC_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    apt-get install -y --no-install-recommends cuda-nvrtc-dev-${NVRTC_CUDA_VERSION}=${NVRTC_VER}
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

install_centos_requirements() {
    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    yum -y update
    yum -y install epel-release
    yum remove -y libnccl* && yum -y install libnccl-${NCCL_VER} libnccl-devel-${NCCL_VER}
    yum remove -y libcublas* && yum -y install libcublas-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER} libcublas-devel-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}
    yum clean all
}

install_tensorrt() {
    PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")
    TRT_CUDA_VERSION="12.4"

    if [ -z "$RELEASE_URL_TRT" ];then
        ARCH=${TRT_TARGETARCH}
        if [ -z "$ARCH" ];then ARCH=$(uname -m);fi
        if [ "$ARCH" = "arm64" ];then ARCH="aarch64";fi
        if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
        if [ "$ARCH" = "x86_64" ];then DIR_NAME="x64-agnostic"; else DIR_NAME=${ARCH};fi
        if [ "$ARCH" = "aarch64" ];then OS1="Ubuntu22_04" && OS2="Ubuntu-22.04" && OS="ubuntu-22.04"; else OS1="Linux" && OS2="Linux" && OS="linux";fi
        RELEASE_URL_TRT=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-${TRT_VER}.${OS2}.${ARCH}-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz
    fi
    wget --no-verbose ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar
    tar -xf /tmp/TensorRT.tar -C /usr/local/
    mv /usr/local/TensorRT-${TRT_VER} /usr/local/tensorrt
    pip3 install /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
    rm -rf /tmp/TensorRT.tar
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt/lib' >> "${ENV}"
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu_requirements
    install_tensorrt
    ;;
  centos)
    install_centos_requirements
    install_tensorrt
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
