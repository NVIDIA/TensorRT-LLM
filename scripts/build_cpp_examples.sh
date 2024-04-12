#!/usr/bin/env bash
set -e

BUILD_DIR="examples/cpp/executor/build"

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
pushd ${BUILD_DIR}

cmake ..
make -j

popd
