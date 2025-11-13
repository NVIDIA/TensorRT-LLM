#!/bin/bash
set -ex

UCX_VERSION="v1.20.x"
GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"

mkdir -p /third-party-source

rm -rf ${UCX_INSTALL_PATH}
curl -L ${GITHUB_URL}/openucx/ucx/archive/refs/heads/${UCX_VERSION}.tar.gz -o ucx-${UCX_VERSION}.tar.gz
cp ucx-${UCX_VERSION}.tar.gz /third-party-source/
tar -xzf ucx-${UCX_VERSION}.tar.gz
mv ucx-${UCX_VERSION} ucx
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
