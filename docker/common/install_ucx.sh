#!/bin/bash
set -ex

UCX_VERSION="v1.22.x"
UCX_COMMIT="8a6b06fb880accbb933a79cda893883872c68d9d"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
GITHUB_URL="${GITHUB_MIRROR:-https://github.com}"
UCX_REPO="${GITHUB_URL}/openucx/ucx.git"

mkdir -p /third-party-source

rm -rf ${UCX_INSTALL_PATH}
git clone -b ${UCX_VERSION} ${UCX_REPO}
cd ucx
git checkout ${UCX_COMMIT}
cd ..
tar -czf /third-party-source/ucx-${UCX_VERSION}.tar.gz ucx
cd ucx
./autogen.sh
./contrib/configure-release       \
  --prefix=${UCX_INSTALL_PATH}    \
  --enable-shared                 \
  --disable-static                \
  --disable-doxygen-doc           \
  --enable-optimizations          \
  --enable-cma                    \
  --enable-devel-headers          \
  --with-cuda=${CUDA_PATH}        \
  --with-verbs                    \
  --with-dm                       \
  --enable-mt
make install -j$(nproc)
cd ..
rm -rf ucx  # Remove UCX source to save space
echo "export LD_LIBRARY_PATH=${UCX_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
