#!/bin/bash
set -ex

MOONCAKE_VERSION="v0.3.6.post1"
MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
MOONCAKE_INSTALL_PATH="/usr/local/Mooncake"

git clone --depth 1 -b ${MOONCAKE_VERSION} ${MOONCAKE_REPO}
cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${MOONCAKE_INSTALL_PATH}
make -j
make install

echo "export LD_LIBRARY_PATH=${MOONCAKE_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
