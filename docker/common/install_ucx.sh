#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
  GITHUB_URL=${GITHUB_MIRROR}
fi

UCX_VERSION="v1.21.x"
UCX_COMMIT="167a4c6a311d9a42e30a37dcc01b8a3e73ea2826"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_ARCHIVE="${GITHUB_URL}/openucx/ucx/archive/${UCX_COMMIT}.tar.gz"

mkdir -p /third-party-source

rm -rf ${UCX_INSTALL_PATH}
curl -L ${UCX_ARCHIVE} | tar -zx
mv ucx-${UCX_COMMIT} ucx
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
