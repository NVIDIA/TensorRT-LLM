#!/bin/bash
set -ex

GITHUB_URL="https://github.com"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
NIXL_VERSION="0.9.0"
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
pip3 install --no-cache-dir meson ninja pybind11 setuptools

git clone --depth 1 -b ${NIXL_VERSION} ${NIXL_REPO}
cd nixl

# Remove POSIX backend compilation from meson.build
sed -i -E "/^[[:space:]]*subdir\(['\"]posix['\"]\)/d" src/plugins/meson.build
DISABLE_GPUNETIO="${DISABLE_NIXL_GPUNETIO:-0}"
if [[ "${DISABLE_GPUNETIO}" != "1" ]] && [[ -d /opt/mellanox/doca/include ]] &&
   ! grep -Rqs "need_mcst" /opt/mellanox/doca/include; then
  DISABLE_GPUNETIO=1
fi
if [[ "${DISABLE_GPUNETIO}" == "1" ]]; then
  # NIXL 0.9.0's GPUNETIO plugin is not compatible with DOCA 3.3 headers
  # (doca_gpu_dev_verbs_qp::need_mcst was removed). TRT-LLM's RDMA DOCA path
  # uses its own extension, so keep NIXL on the stable UCX/GDS plugins.
  sed -i -E "/^[[:space:]]*subdir\(['\"]gpunetio['\"]\)/d" src/plugins/meson.build
  if grep -nE "^[[:space:]]*subdir\(['\"]gpunetio['\"]\)" src/plugins/meson.build; then
    echo "Failed to disable NIXL GPUNETIO plugin" >&2
    exit 1
  fi
  echo "NIXL GPUNETIO plugin disabled"
fi

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
