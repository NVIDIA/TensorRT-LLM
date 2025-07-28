#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
NIXL_VERSION="0.4.1"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

ARCH_NAME="x86_64-linux-gnu"
GDS_PATH="$CUDA_PATH/targets/x86_64-linux"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
  ARCH_NAME="aarch64-linux-gnu"
  GDS_PATH="$CUDA_PATH/targets/sbsa-linux"
fi

pip3 install --no-cache-dir meson ninja pybind11
git clone --depth 1 -b ${NIXL_VERSION} ${NIXL_REPO}
cd nixl

cuda_so_path=$(find "/usr/local" -path "*/cuda*/compat/lib.real/libcuda.so.1" 2>/dev/null | head -n1)
if [[ -z "$cuda_so_path" ]]; then
    echo "libcuda.so.1 not found "
    exit 1
fi

echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$cuda_so_path" >> "${ENV}"

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

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$OLD_LD_LIBRARY_PATH" >> "${ENV}"
