#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
NIXL_VERSION="0.9.0"
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
pip3 install --no-cache-dir meson ninja pybind11 setuptools

curl -L ${GITHUB_URL}/ai-dynamo/nixl/archive/refs/tags/${NIXL_VERSION}.tar.gz -o nixl-${NIXL_VERSION}.tar.gz
tar -xzf nixl-${NIXL_VERSION}.tar.gz
mv nixl-${NIXL_VERSION} nixl
cd nixl

# Remove POSIX backend compilation from meson.build
sed -i "/^subdir('posix')/d" src/plugins/meson.build

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
    --buildtype=release

cd builddir && ninja install
cd ../..
rm -rf nixl*  # Remove NIXL source tree to save space
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
