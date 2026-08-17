#!/bin/bash
set -ex

# Authenticate the github.com clone below; no-op when no token is available.
source "$(dirname "${BASH_SOURCE[0]}")/github_auth.sh"

UCX_VERSION="v1.22.x"
UCX_COMMIT="8a6b06fb880accbb933a79cda893883872c68d9d"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/openucx/ucx.git"

mkdir -p /third-party-source

rm -rf ${UCX_INSTALL_PATH}

# Fetch just the pinned commit rather than cloning the whole history
rm -rf ucx
git init -q ucx
git -C ucx remote add origin ${UCX_REPO}
git -C ucx fetch -q --depth 1 origin ${UCX_COMMIT}
git -C ucx checkout -q FETCH_HEAD

tar -czf /third-party-source/ucx-${UCX_VERSION}.tar.gz ucx
cd ucx
# Pull external/gpunetio shallow, for the same reason: autogen.sh below does a
# full-history `git submodule update --init` on it. With the submodule already
# at the recorded commit that call does nothing.
git submodule update --init --depth 1
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
