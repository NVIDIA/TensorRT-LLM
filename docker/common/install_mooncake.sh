#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

MOONCAKE_VERSION="0.3.7.post2"
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

YALANTINGLIBS_VERSION="0.5.5"
curl -L ${GITHUB_URL}/alibaba/yalantinglibs/archive/refs/tags/${YALANTINGLIBS_VERSION}.tar.gz -o yalantinglibs-${YALANTINGLIBS_VERSION}.tar.gz
tar -xzf yalantinglibs-${YALANTINGLIBS_VERSION}.tar.gz
# Auto-detect extracted directory name
YALANTINGLIBS_DIR=$(tar -tzf yalantinglibs-${YALANTINGLIBS_VERSION}.tar.gz | head -1 | cut -f1 -d"/")
mv ${YALANTINGLIBS_DIR} yalantinglibs
cd yalantinglibs
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
make -j
make install
cd ../..
rm -rf yalantinglibs

curl -L ${GITHUB_URL}/kvcache-ai/Mooncake/archive/refs/tags/v${MOONCAKE_VERSION}.tar.gz -o Mooncake-${MOONCAKE_VERSION}.tar.gz
tar -xzf Mooncake-${MOONCAKE_VERSION}.tar.gz
# Auto-detect extracted directory name
MOONCAKE_DIR=$(tar -tzf Mooncake-${MOONCAKE_VERSION}.tar.gz | head -1 | cut -f1 -d"/")
mv ${MOONCAKE_DIR} Mooncake
cd Mooncake

# Manually download pybind11 submodule (since tarball doesn't include submodules)
PYBIND11_VERSION="v2.13.6"
mkdir -p extern
curl -L ${GITHUB_URL}/pybind/pybind11/archive/refs/tags/${PYBIND11_VERSION}.tar.gz -o pybind11.tar.gz
tar -xzf pybind11.tar.gz
PYBIND11_DIR=$(tar -tzf pybind11.tar.gz | head -1 | cut -f1 -d"/")
mv ${PYBIND11_DIR} extern/pybind11
rm pybind11.tar.gz

mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DBUILD_SHARED_LIBS=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=${MOONCAKE_INSTALL_PATH}
make -j
make install
cd ../..
rm -rf Mooncake

echo "export LD_LIBRARY_PATH=${MOONCAKE_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
