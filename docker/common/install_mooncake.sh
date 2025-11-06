#!/bin/bash
set -ex

MOONCAKE_VERSION="v0.3.6.post1"
MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
MOONCAKE_INSTALL_PATH="/usr/local"

git clone --depth 1 -b ${MOONCAKE_VERSION} ${MOONCAKE_REPO}

tar -czf mooncake-${MOONCAKE_VERSION}.tar.gz Mooncake
mkdir -p /third-party-source
mv mooncake-${MOONCAKE_VERSION}.tar.gz /third-party-source/

cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DBUILD_SHARED_LIBS=ON
make -j
make install

cd ../..
rm -rf Mooncake

echo "export LD_LIBRARY_PATH=${MOONCAKE_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
