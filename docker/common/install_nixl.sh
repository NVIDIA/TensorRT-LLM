#!/bin/bash
set -ex

GITHUB_URL="https://github.com"

UCX_VERSION="v1.18.1"
UCX_INSTALL_PATH="/usr/local/ucx/"

NIXL_VERSION="0.2.0"

UCX_REPO="https://github.com/openucx/ucx.git"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"

UCX_MIRROR="https://gitlab-master.nvidia.com/ftp/GitHubSync/ucx.git"
NIXL_MIRROR="https://gitlab-master.nvidia.com/ftp/GitHubSync/nixl.git"

if [ -n "${GITHUB_MIRROR}" ]; then
  UCX_REPO=${UCX_MIRROR}
  NIXL_REPO=${NIXL_MIRROR}
fi

if [ ! -d ${UCX_INSTALL_PATH} ]; then
  git clone --depth 1 -b ${UCX_VERSION} ${UCX_REPO}
  cd ucx
  ./autogen.sh
  ./contrib/configure-release --prefix=${UCX_INSTALL_PATH}
  make install -j$(nproc)
  cd ..
  rm -rf ucx  # Remove UCX source to save space
  echo "export LD_LIBRARY_PATH=${UCX_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
fi

ARCH_NAME="x86_64-linux-gnu"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
  ARCH_NAME="aarch64-linux-gnu"
  EXTRA_NIXL_ARGS="-Ddisable_gds_backend=true"
fi

pip3 install --no-cache-dir meson ninja pybind11
git clone --depth 1 -b ${NIXL_VERSION} ${NIXL_REPO}
cd nixl
meson setup builddir -Ducx_path=${UCX_INSTALL_PATH} ${EXTRA_NIXL_ARGS}
cd builddir && ninja install
cd ../..
rm -rf nixl  # Remove NIXL source tree to save space

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
