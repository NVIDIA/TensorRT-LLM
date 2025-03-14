#!/usr/bin/env bash

### This script targets to build tensorrt_llm .so libs and wheel in TRT dev container.
# The motivation is to ease in-tree development and debugging when either TRT or tensorrt_llm code is changed.
# Tested in following containers, other TRT container may or may not work.
#      main-native-x86_64-ubuntu22.04-cuda12.0
# Steps:
# 1. cd <tensorrt_llm_root>
# 2. Launch container: git trt runc -w -i main-native-x86_64-ubuntu22.04-cuda12.0  -v -H --mounts `pwd`:/workspace/tensorrt_llm,`realpath ~/.local/`:$HOME/.local
# 3. Inside container, run: cd /workspace/tensorrt_llm && bash /workspace/tensorrt_llm/scripts/build_wheel_trt_dev_container.sh

if [ -z "${TRT_CONTAINER_IMAGE_TAG}" ]; then
    echo ${TRT_CONTAINER_IMAGE_TAG}
    echo "This script are supposed to be run in TRT dev container, TRT_CONTAINER_IMAGE_TAG is not set, are you sure this is in TRT dev container?"
    exit 1
else
    echo "Building tensorrt_llm wheel in TRT dev container: ${TRT_CONTAINER_IMAGE_TAG}, pwd:$(pwd)"
fi
set -e

# install deps
pip install -r requirements.txt

ARCH=$(uname -m)
if [ ${ARCH} = aarch64 ]; then
    DIR_NAME=aarch64sbsa
elif [ ${ARCH} = x86_64 ]; then
    DIR_NAME=x64
fi

## WAR: download NCCL to build/ directory, before TRT container has NCCL by default
mkdir -p build
NCCL_VERSION=nccl_2.15.3-1+cuda12.0_${ARCH}
if [ ! -e build/${NCCL_VERSION} ]; then
    echo "NCCL not found, downloading to build/ directory"
    pushd build
    wget http://cuda-repo/release-candidates/Libraries/NCCL/v2.15/NightlyBuilds/stable/20221009_NCCL2.15.3/CUDA12.0-r525_cl-31908123/txz/agnostic/${DIR_NAME}/${NCCL_VERSION}.txz  && \
    xz -d ${NCCL_VERSION}.txz && \
    tar xvf ${NCCL_VERSION}.tar
    popd
fi
export NCCL_INSTALL_DIR=$(realpath build/${NCCL_VERSION})

# Otherwise, cmake can not detect NVCC compiler ID correctly
export PATH=$CUDA_INSTALL_DIR/bin:$PATH

# build decoding lib
rm -rf cpp/build && mkdir -p cpp/build && \
    pushd cpp/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DPYTHON_PATH=python3 \
    -DTRT_LIB_DIR=$HOME/trt/build/${ARCH}-gnu/ -DTRT_INCLUDE_DIR=$HOME/trt/include/ \
    -DCUDNN_ROOT_DIR=$CUDNN_INSTALL_DIR \
    -DNCCL_LIB_DIR=${NCCL_INSTALL_DIR}/lib -DNCCL_INCLUDE_DIR=${NCCL_INSTALL_DIR}/include \
    .. &&\
    make -j"$(grep -c ^processor /proc/cpuinfo)" && popd

# copy decoding lib
rm -rf tensorrt_llm/libs && mkdir -p tensorrt_llm/libs && \
    cp cpp/build/tensorrt_llm/thop/libth_common.so tensorrt_llm/libs/libth_common.so && \
    cp cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so

# build wheel
mkdir -p build && python3 setup.py --quiet bdist_wheel --dist-dir build
