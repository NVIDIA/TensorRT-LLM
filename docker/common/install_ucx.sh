#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
UCX_VERSION="v1.19.0"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/openucx/ucx.git"

if [ ! -d ${UCX_INSTALL_PATH} ]; then
  git clone --depth 1 -b ${UCX_VERSION} ${UCX_REPO}
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
fi
