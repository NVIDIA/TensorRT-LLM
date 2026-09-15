#!/bin/bash
set -ex

# Authenticate the github.com clone below; no-op when no token is available.
source "$(dirname "${BASH_SOURCE[0]}")/github_auth.sh"

GITHUB_URL="https://github.com"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
NIXL_VERSION="v1.4.0"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

ARCH_NAME="x86_64-linux-gnu"
GDS_PATH="$CUDA_PATH/targets/x86_64-linux"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
  ARCH_NAME="aarch64-linux-gnu"
  GDS_PATH="$CUDA_PATH/targets/sbsa-linux"
fi

if [ -n "${GITHUB_MIRROR}" ]; then
  export PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
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
meson setup builddir \
    -Ducx_path=$UCX_INSTALL_PATH \
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
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH

# Consumers import `nixl`, but the build above installs the backend as
# `nixl_cu13`. Install the dispatching shim with --no-deps: the backend it would
# otherwise pull from PyPI bundles a second UCX, which segfaults alongside the
# one torch already links.
pip3 install --no-deps "nixl==${NIXL_VERSION#v}"

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
# ninja installs the bindings outside site-packages, so the shim needs PYTHONPATH.
echo "export PYTHONPATH=/opt/nvidia/nvda_nixl/lib/python3/dist-packages\${PYTHONPATH:+:\$PYTHONPATH}" >> "${ENV}"
