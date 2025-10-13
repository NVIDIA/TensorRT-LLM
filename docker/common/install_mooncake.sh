#!/bin/bash
set -ex

MOONCAKE_VERSION="v0.3.5"
MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"

git clone ${MOONCAKE_VERSION} ${MOONCAKE_REPO}
cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DBUILD_SHARED_LIBS=ON
make -j
make install
