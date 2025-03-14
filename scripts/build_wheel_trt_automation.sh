#!/usr/bin/env bash

### This script targets to build tensorrt_llm .so libs and wheel in TRT automation pipeline.
# bash scripts/build_wheel_trt_automation.sh <TENSORRT_ROOT_DIR>
set -e

TRT_ROOT=$1

# install deps
pip install -r requirements.txt

## WAR: download NCCL to build/ directory, before TRT container has NCCL by default
ARCH=$(uname -m)
if [ ${ARCH} = aarch64]; then
    DIR_NAME=aarch64sbsa
elif [ ${ARCH} = x86_64]
    DIR_NAME=x64
fi

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

# build decoding lib, use 4 jobs to avoid being killed due to OOM
rm -rf cpp/build && mkdir -p cpp/build && \
    pushd cpp/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DPYTHON_PATH=python3 \
    -DTRT_LIB_DIR=$TRT_ROOT/targets/${ARCH}-linux-gnu/lib -DTRT_INCLUDE_DIR=$TRT_ROOT/include/ \
    -DCUDNN_ROOT_DIR=$CUDNN_INSTALL_DIR \
    -DNCCL_LIB_DIR=${NCCL_INSTALL_DIR}/lib -DNCCL_INCLUDE_DIR=${NCCL_INSTALL_DIR}/include \
    .. &&\
    make -j4 && popd

# copy decoding lib
rm -rf tensorrt_llm/libs && mkdir -p tensorrt_llm/libs && \
    cp cpp/build/tensorrt_llm/thop/libth_common.so tensorrt_llm/libs/libth_common.so && \
    cp cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so

# build wheel
mkdir -p build && python3 setup.py --quiet bdist_wheel --dist-dir build
