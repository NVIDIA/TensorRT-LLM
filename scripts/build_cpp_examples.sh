#!/usr/bin/env bash
set -e

BUILD_DIR="examples/cpp/executor/build"
TRT_DIR="/usr/local/tensorrt"

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
pushd ${BUILD_DIR}

cmake .. -DTRT_LIB_DIR=${TRT_DIR}/lib -DTRT_INCLUDE_DIR=${TRT_DIR}/include
make -j

popd
