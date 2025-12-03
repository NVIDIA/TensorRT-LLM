#!/bin/bash
set -ex

MOONCAKE_VERSION="v0.3.7.post2"
MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
MOONCAKE_INSTALL_PATH="/usr/local/Mooncake"

apt-get update

# https://kvcache-ai.github.io/Mooncake/getting_started/build.html
# libboost-all-dev is removed because it will install a duplicated MPI library
# triton also installed boost so the requirement is already met
apt-get install -y --no-install-recommends \
    build-essential \
    libibverbs-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libnuma-dev \
    libunwind-dev \
    libssl-dev \
    libyaml-cpp-dev \
    libcurl4-openssl-dev \
    libhiredis-dev \
    pkg-config \
    patchelf

mkdir -p /third-party-source

git clone --depth 1 https://github.com/alibaba/yalantinglibs.git
tar -czf /third-party-source/yalantinglibs.tar.gz yalantinglibs
cd yalantinglibs
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
make -j
make install
cd ../..
rm -rf yalantinglibs

git clone --depth 1 -b ${MOONCAKE_VERSION} ${MOONCAKE_REPO}
tar -czf /third-party-source/Mooncake-${MOONCAKE_VERSION}.tar.gz Mooncake
cd Mooncake
git submodule update --init --recursive --depth 1
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DBUILD_SHARED_LIBS=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=${MOONCAKE_INSTALL_PATH}
make -j
make install
cd ../..
rm -rf Mooncake

echo "export LD_LIBRARY_PATH=${MOONCAKE_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
