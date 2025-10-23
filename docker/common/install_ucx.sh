#!/bin/bash
set -ex

UCX_VERSION="v1.19.x"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/openucx/ucx.git"

if [ -n "${GITHUB_MIRROR}" ]; then
  export UCX_REPO="https://gitlab-master.nvidia.com/ftp/GitHubSync/ucx.git"
fi

rm -rf ${UCX_INSTALL_PATH}
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
