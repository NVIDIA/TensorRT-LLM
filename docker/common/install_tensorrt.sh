#!/bin/bash

set -ex

install_ubuntu_requirements() {
    CUDNN_VERSION="8"
    apt-get update
    apt-get install -y --no-install-recommends libcudnn${CUDNN_VERSION} libcudnn${CUDNN_VERSION}-dev libnccl-dev
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

install_centos_requirements() {
    CUDNN_VERSION="8"
    yum -y update
    yum -y install epel-release
    yum -y install libcudnn${CUDNN_VERSION} libcudnn${CUDNN_VERSION}-devel libnccl-devel
    yum clean all
}

install_tensorrt() {
    TENSOR_RT_VERSION="9.1.0.4"
    CUDA_VERSION="12.2"

    PY_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")

    if [ -z "$RELEASE_URL_TRT" ];then
        ARCH=${TRT_TARGETARCH}
        if [ -z "$ARCH" ];then ARCH=$(uname -m);fi
        if [ "$ARCH" = "arm64" ];then ARCH="aarch64";fi
        if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
        if [ "$ARCH" = "x86_64" ];then DIR_NAME="x64-agnostic"; else DIR_NAME=${ARCH};fi
        if [ "$ARCH" = "aarch64" ];then OS="ubuntu-22.04"; else OS="linux";fi
        RELEASE_URL_TRT=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.1.0/tars/tensorrt-${TENSOR_RT_VERSION}.${OS}.${ARCH}-gnu.cuda-${CUDA_VERSION}.tar.gz;
    fi
    wget --no-verbose ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar
    tar -xf /tmp/TensorRT.tar -C /usr/local/
    mv /usr/local/TensorRT-${TENSOR_RT_VERSION} /usr/local/tensorrt
    pip install /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
    rm -rf /tmp/TensorRT.tar
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
