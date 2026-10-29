#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Authenticate the github.com clone below; no-op when no token is available.
source "${SCRIPT_DIR}/github_auth.sh"

GITHUB_URL="https://github.com"
UCX_INSTALL_PATH="/usr/local/ucx/"

# Interm build, minimum supported libfabric version the NIXL LIBFABRIC plugin works with, retained for backward compatibility.
LIBFABRIC_VERSION="v1.21.0"
LIBFABRIC_INSTALL_PATH="/tmp/libfabric"
wget "https://github.com/ofiwg/libfabric/releases/download/${LIBFABRIC_VERSION}/libfabric-${LIBFABRIC_VERSION#v}.tar.bz2" -O libfabric.tar.bz2
tar xjf libfabric.tar.bz2 && rm libfabric.tar.bz2
cd libfabric-*
./autogen.sh
./configure --prefix=${LIBFABRIC_INSTALL_PATH} \
            --disable-verbs \
            --disable-psm3 \
            --disable-opx \
            --disable-usnic \
            --enable-efa \
            --with-cuda=/usr/local/cuda \
            --enable-cuda-dlopen \
            --with-gdrcopy \
            --enable-gdrcopy-dlopen
make -j${NPROC:-$(nproc)}
make install
cd ../
rm -rf libfabric*

# NGC PyTorch's default Open MPI comes from HPC-X and is exposed at /usr/local/mpi.
# Reuse the hwloc bundled with that MPI installation for the EFA plugin.
HWLOC_INSTALL_PATH="/usr/local/mpi"
if [[ -n "${LIBFABRIC_INSTALL_PATH}" &&
      -f "${HWLOC_INSTALL_PATH}/include/hwloc.h" &&
      -f "${HWLOC_INSTALL_PATH}/lib/libhwloc.so" &&
      -f "${HWLOC_INSTALL_PATH}/lib/pkgconfig/hwloc.pc" ]]; then
    export PKG_CONFIG_PATH="${HWLOC_INSTALL_PATH}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
fi
CUDA_PATH="/usr/local/cuda"
NIXL_VERSION="v1.5.2"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

ARCH_NAME="x86_64-linux-gnu"
GDS_PATH="$CUDA_PATH/targets/x86_64-linux"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
  ARCH_NAME="aarch64-linux-gnu"
  GDS_PATH="$CUDA_PATH/targets/sbsa-linux"
fi

if [ -n "${GITHUB_MIRROR}" ]; then
  export PIP_INDEX_URL="https://artifactory.pdx.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
fi
pip3 install meson ninja pybind11 setuptools

git clone --depth 1 -b ${NIXL_VERSION} ${NIXL_REPO}
cd nixl

CUDA_SO_PATH=$(find "/usr/local" -name "libcuda.so.1" 2>/dev/null | head -n1)

if [[ -z "$CUDA_SO_PATH" ]]; then
    echo "libcuda.so.1 not found"
    exit 1
fi

CUDA_SO_PATH=$(dirname $CUDA_SO_PATH)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_SO_PATH
# An empty libfabric_path uses optional system detection; no match skips the plugin.
meson setup builddir \
    -Ducx_path=$UCX_INSTALL_PATH \
    -Dlibfabric_path="${LIBFABRIC_INSTALL_PATH}" \
    -Dcudapath_lib="$CUDA_PATH/lib64" \
    -Dcudapath_inc="$CUDA_PATH/include" \
    -Dgds_path="$GDS_PATH" \
    -Dinstall_headers=true \
    -Ddisable_plugins=POSIX \
    -Dbuild_tests=false \
    -Dbuild_examples=false \
    --buildtype=release

cd builddir && ninja install
cd ../..
rm -rf nixl*  # Remove NIXL source tree to save space
rm -rf /tmp/libfabric # Remove libfabric interim build
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH

# Consumers import `nixl`, but the build above installs the backend as
# `nixl_cu13`. Install the dispatching shim with --no-deps: the backend it would
# otherwise pull from PyPI bundles a second UCX, which segfaults alongside the
# one torch already links.
pip3 install --no-deps "nixl==${NIXL_VERSION#v}"

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
# ninja installs the bindings outside site-packages, so the shim needs PYTHONPATH. This takes
# precedence over the PyPI backend that requirements-dev.txt installs into the test
# environments, so keep that pin in step with NIXL_VERSION above -- otherwise the wheel is
# installed and silently never used.
# Prepend only once: ${ENV} is sourced again by every nested shell.
cat >> "${ENV}" <<'EOF'
case ":${PYTHONPATH:-}:" in
  *":/opt/nvidia/nvda_nixl/lib/python3/dist-packages:"*) ;;
  *) export PYTHONPATH="/opt/nvidia/nvda_nixl/lib/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
EOF
