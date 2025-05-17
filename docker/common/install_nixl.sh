#!/bin/bash
set -ex

GITHUB_URL="https://github.com"

UCX_VERSION="v1.18.1"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"

if [ ! -d ${UCX_INSTALL_PATH} ]; then
  git clone --depth 1 -b ${UCX_VERSION} https://github.com/openucx/ucx.git
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

NIXL_VERSION="0.2.0"
NIXL_REPO="${GITHUB_URL}/ai-dynamo/nixl.git"

ARCH_NAME="x86_64-linux-gnu"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
  ARCH_NAME="aarch64-linux-gnu"
  EXTRA_NIXL_ARGS="-Ddisable_gds_backend=true"
fi

if [ $ARCH_NAME != "x86_64-linux-gnu" ]; then
  echo "The NIXL backend is temporarily unavailable on the aarch64 platform. Exiting script."
  exit 0
fi

pip3 install --no-cache-dir meson ninja pybind11
git clone --depth 1 -b ${NIXL_VERSION} ${NIXL_REPO}
cd nixl
meson setup builddir -Ducx_path=${UCX_INSTALL_PATH} -Dstatic_plugins=UCX ${EXTRA_NIXL_ARGS}
cd builddir && ninja install
cd ../..
rm -rf nixl*  # Remove NIXL source tree to save space

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
