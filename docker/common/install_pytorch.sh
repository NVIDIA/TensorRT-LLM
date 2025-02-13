#!/bin/bash

set -ex

# Use latest stable version from https://pypi.org/project/torch/#history
# and closest to the version specified in
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-01.html#rel-25-01
TORCH_VERSION="2.6.0"
SYSTEM_ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')

prepare_environment() {
    if [[ $SYSTEM_ID == *"ubuntu"* ]]; then
      apt-get update && apt-get -y install ninja-build
      apt-get clean && rm -rf /var/lib/apt/lists/*
    elif [[ $SYSTEM_ID == *"rocky"* ]]; then
      dnf makecache --refresh && dnf install -y ninja-build && dnf clean all
    else
      echo "This system type cannot be supported..."
      exit 1
    fi
}

install_from_source() {
    if [[ $SYSTEM_ID == *"rocky"* ]]; then
      VERSION_ID=$(grep -oP '(?<=^VERSION_ID=).+' /etc/os-release | tr -d '"')
      if [[ $VERSION_ID == "7" ]]; then
        echo "Installation from PyTorch source codes cannot be supported..."
        exit 1
      fi
    fi
    prepare_environment $1

    export _GLIBCXX_USE_CXX11_ABI=$1

    export TORCH_CUDA_ARCH_LIST="8.0;9.0"
    export PYTORCH_BUILD_VERSION=${TORCH_VERSION}
    export PYTORCH_BUILD_NUMBER=0
    pip3 uninstall -y torch
    cd /tmp
    git clone --depth 1 --branch v${TORCH_VERSION} https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync && git submodule update --init --recursive
    pip3 install -r requirements.txt
    python3 setup.py install
    cd /tmp && rm -rf /tmp/pytorch

    # Get torchvision version by dry run
    TORCHVISION_VERSION=$(pip3 install --dry-run torch==${TORCH_VERSION} torchvision | grep "Would install" | tr ' ' '\n' | grep torchvision | cut -d "-" -f 2)

    export PYTORCH_VERSION=${PYTORCH_BUILD_VERSION}
    export FORCE_CUDA=1
    export BUILD_VERSION=${TORCHVISION_VERSION}
    pip3 uninstall -y torchvision
    cd /tmp
    git clone --depth 1 --branch v${TORCHVISION_VERSION} https://github.com/pytorch/vision
    cd vision
    python3 setup.py install
    cd /tmp && rm -rf /tmp/vision
}

install_from_pypi() {
    pip3 uninstall -y torch torchvision
    pip3 install torch==${TORCH_VERSION} torchvision
}

case "$1" in
  "skip")
    ;;
  "pypi")
    install_from_pypi
    ;;
  "src_cxx11_abi")
    install_from_source 1
    ;;
  "src_non_cxx11_abi")
    install_from_source 0
    ;;
  *)
    echo "Incorrect input argument..."
    exit 1
    ;;
esac
