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
  export UV_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
  export UV_HTTP_TIMEOUT=120
fi
pip3 install meson ninja pybind11 setuptools

git clone --depth 1 -b ${NIXL_VERSION} ${NIXL_REPO}
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

# NIXL auto-builds the LIBFABRIC plugin when meson's dependency('libfabric')
# resolves. Prefer the AWS EFA libfabric when present (ships the optimized
# "efa" provider); otherwise fall back to pkg-config detection of the distro
# libfabric installed by install_libfabric_efa.sh.
LIBFABRIC_OPTS=()
if [ -d "/opt/amazon/efa" ] && [ -f "/opt/amazon/efa/include/rdma/fabric.h" ]; then
  LIBFABRIC_OPTS+=("-Dlibfabric_path=/opt/amazon/efa")
fi

meson setup builddir \
    -Ducx_path=$UCX_INSTALL_PATH \
    "${LIBFABRIC_OPTS[@]}" \
    -Dcudapath_lib="$CUDA_PATH/lib64" \
    -Dcudapath_inc="$CUDA_PATH/include" \
    -Dgds_path="$GDS_PATH" \
    -Dinstall_headers=true \
    --buildtype=release

cd builddir && ninja install
cd ../..

# Verify the LIBFABRIC plugin actually built. The TRT-LLM perf submit sets
# TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC on EFA nodes, and NixlTransferAgent
# asserts in getPluginParams() if the plugin is missing — fail the image
# build now instead of crashing executors at runtime.
PLUGIN_DIR="/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}/plugins"
if ! ls "${PLUGIN_DIR}"/libplugin_LIBFABRIC* >/dev/null 2>&1; then
  echo "[install_nixl] ERROR: LIBFABRIC plugin not built under ${PLUGIN_DIR}." >&2
  echo "[install_nixl] Ensure install_libfabric_efa.sh ran first and that" >&2
  echo "[install_nixl] libfabric headers are visible to meson." >&2
  exit 1
fi

rm -rf nixl*  # Remove NIXL source tree to save space
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
