#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

MOONCAKE_VERSION="v0.3.7.post2"
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

YALANTINGLIBS_VERSION="v1.0.0"
curl -L ${GITHUB_URL}/alibaba/yalantinglibs/archive/refs/tags/${YALANTINGLIBS_VERSION}.tar.gz -o yalantinglibs-${YALANTINGLIBS_VERSION}.tar.gz
tar -xzf yalantinglibs-${YALANTINGLIBS_VERSION}.tar.gz
mv yalantinglibs-${YALANTINGLIBS_VERSION} yalantinglibs
cd yalantinglibs
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
make -j
make install
cd ../..
rm -rf yalantinglibs

curl -L ${GITHUB_URL}/kvcache-ai/Mooncake/archive/refs/tags/${MOONCAKE_VERSION}.tar.gz -o Mooncake-${MOONCAKE_VERSION}.tar.gz
tar -xzf Mooncake-${MOONCAKE_VERSION}.tar.gz
mv Mooncake-${MOONCAKE_VERSION} Mooncake
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
