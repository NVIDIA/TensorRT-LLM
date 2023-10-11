#!/bin/bash

set -ex

# Only verified under ubuntu.
install_from_source() {
    TORCH_VERSION="2.0.1"
    export _GLIBCXX_USE_CXX11_ABI=1
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"

    apt-get update && apt-get -y install ninja-build
    apt-get clean && rm -rf /var/lib/apt/lists/*
    cd /tmp
    git clone --depth 1 --branch v$TORCH_VERSION https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync && git submodule update --init --recursive
    pip install -r requirements.txt
    python setup.py install
    cd /tmp && rm -rf /tmp/pytorch
}

if [ "$1" == "1" ]; then
    install_from_source
fi
