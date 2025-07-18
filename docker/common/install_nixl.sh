#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
NIXL_VERSION="0.3.1"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"

ARCH_NAME="x86_64-linux-gnu"
GDS_PATH="$CUDA_PATH/targets/x86_64-linux"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
  ARCH_NAME="aarch64-linux-gnu"
  GDS_PATH="$CUDA_PATH/targets/sbsa-linux"
fi

pip3 install --no-cache-dir meson ninja pybind11
git clone --depth 1 -b ${NIXL_VERSION} ${NIXL_REPO}
cd nixl

cuda_path=$(find / -name "libcuda.so.1" 2>/dev/null | head -n1)
if [[ -z "$cuda_path" ]]; then
    echo "libcuda.so.1 not found "
    exit 1
fi

ln -sf $cuda_path $CUDA_PATH/lib64/libcuda.so.1

meson setup builddir \
    -Ducx_path=$UCX_INSTALL_PATH \
    -Dcudapath_lib="$CUDA_PATH/lib64" \
    -Dcudapath_inc="$CUDA_PATH/include" \
    -Dgds_path="$GDS_PATH" \
    -Dinstall_headers=true \
    -Dstatic_plugins=UCX

cd builddir && ninja install
cd ../..
rm -rf nixl*  # Remove NIXL source tree to save space
rm  $CUDA_PATH/lib64/libcuda.so.1

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
