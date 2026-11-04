#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Authenticate the github.com clone below; no-op when no token is available.
source "${SCRIPT_DIR}/github_auth.sh"

# Build-only ABI baseline; runtime libfabric is provided separately.
LIBFABRIC_BUILD_VERSION="v1.21.0"
# Source-build fallback from NIXL v1.5.2's manylinux dependency set.
HWLOC_VERSION=2.12.2
NIXL_VERSION="v1.5.2"

GITHUB_URL="${GITHUB_MIRROR:-https://github.com}"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
CUDA_PATH="/usr/local/cuda"
UCX_INSTALL_PATH="/usr/local/ucx/"
HWLOC_INSTALL_PATH="/usr/local/mpi"

# Build against the oldest supported libfabric ABI, independently of the runtime EFA installation.
build_libfabric_for_nixl() {
    wget --tries=3 --waitretry=5 --timeout=30 --read-timeout=60 \
        "${GITHUB_URL}/ofiwg/libfabric/releases/download/${LIBFABRIC_BUILD_VERSION}/libfabric-${LIBFABRIC_BUILD_VERSION#v}.tar.bz2" -O libfabric.tar.bz2
    tar xjf libfabric.tar.bz2 && rm libfabric.tar.bz2
    cd "libfabric-${LIBFABRIC_BUILD_VERSION#v}"
    ./autogen.sh
    ./configure --prefix="${LIBFABRIC_BUILD_PATH}" \
                --disable-verbs \
                --disable-psm3 \
                --disable-opx \
                --disable-usnic \
                --disable-rstream \
                --enable-efa \
                --with-cuda="$CUDA_PATH" \
                --enable-cuda-dlopen \
                --with-gdrcopy \
                --enable-gdrcopy-dlopen
    make -j"${NPROC:-$(nproc)}"
    make install
    cd ../
    rm -rf "libfabric-${LIBFABRIC_BUILD_VERSION#v}"
}

prepare_hwloc() {
    # NGC PyTorch's default Open MPI comes from HPC-X and is exposed at /usr/local/mpi.
    # Reuse the hwloc bundled with that MPI installation for the EFA plugin.
    if [[ -f "${HWLOC_INSTALL_PATH}/include/hwloc.h" &&
          -f "${HWLOC_INSTALL_PATH}/lib/libhwloc.so" &&
          -f "${HWLOC_INSTALL_PATH}/lib/pkgconfig/hwloc.pc" ]]; then
        export PKG_CONFIG_PATH="${HWLOC_INSTALL_PATH}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    fi
    # The LIBFABRIC plugin requires hwloc >= 2.3. For the current base images:
    # - NGC PyTorch: reuse bundled HPC-X hwloc 2.7.1 without rebuilding.
    # - Non-NGC SBSA Ubuntu 24.04: reuse system hwloc 2.10.0 without rebuilding.
    # - Rocky 8: system hwloc 2.2.0 is too old, so build the fallback below.
    # Missing hwloc development files also trigger this fallback on any image.
    if ! pkg-config --atleast-version=2.3 hwloc; then
        curl --connect-timeout 30 --max-time 300 --retry 3 --retry-delay 2 \
            -fsSL "https://download.open-mpi.org/release/hwloc/v${HWLOC_VERSION%.*}/hwloc-${HWLOC_VERSION}.tar.gz" \
            -o hwloc.tar.gz
        tar -xzf hwloc.tar.gz
        cd "hwloc-${HWLOC_VERSION}"
        # Match NIXL's manylinux dependency; retain it for runtime as well as compilation.
        ./configure --prefix=/usr/local --libdir=/usr/local/lib --disable-nvml
        make -j"${NPROC:-$(nproc)}"
        make install
        cd ..
        rm -rf "hwloc-${HWLOC_VERSION}" hwloc.tar.gz
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
        # On Rocky 8, the system hwloc RPM and its /usr/lib64 files remain installed.
        # Prefer the new /usr/local/lib/libhwloc.so.15 through the loader cache.
        # This changes default hwloc selection for other consumers, including Open MPI.
        printf '%s\n' /usr/local/lib > /etc/ld.so.conf.d/000_nixl_hwloc.conf
        ldconfig
    fi
}

build_and_install_nixl() {
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
    # This build-only prefix is removed on exit; runtime libfabric comes from the image or host.
    meson setup builddir \
        -Ducx_path=$UCX_INSTALL_PATH \
        -Dlibfabric_path="${LIBFABRIC_BUILD_PATH}" \
        -Dcudapath_lib="$CUDA_PATH/lib64" \
        -Dcudapath_inc="$CUDA_PATH/include" \
        -Dgds_path="$GDS_PATH" \
        -Dinstall_headers=true \
        -Ddisable_plugins=POSIX \
        -Dbuild_tests=false \
        -Dbuild_examples=false \
        --buildtype=release

    cd builddir && ninja install
    # Meson uses dist-packages on Ubuntu and versioned site-packages on Rocky.
    NIXL_PYTHON_PATH=$(meson introspect --installed | python3 -c '
import json, pathlib, sys
paths = {str(pathlib.Path(p).parent.parent) for p in json.load(sys.stdin).values()
         if pathlib.Path(p).parent.name == "nixl_cu13" and pathlib.Path(p).name.startswith("_bindings.")}
assert len(paths) == 1, paths
print(paths.pop())
    ')
    cd ../..
    rm -rf nixl  # Remove NIXL source tree to save space
    export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH

    # Consumers import `nixl`, but the build above installs the backend as
    # `nixl_cu13`. Install the dispatching shim with --no-deps: the backend it would
    # otherwise pull from PyPI bundles a second UCX, which segfaults alongside the
    # one torch already links.
    pip3 install --no-deps "nixl==${NIXL_VERSION#v}"

    echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
    # The bindings are outside Python's default search path, so the shim needs PYTHONPATH. This takes
    # precedence over the PyPI backend that requirements-dev.txt installs into the test
    # environments, so keep that pin in step with NIXL_VERSION above -- otherwise the wheel is
    # installed and silently never used.
    # Prepend only once: ${ENV} is sourced again by every nested shell.
    cat >> "${ENV}" <<EOF
case ":\${PYTHONPATH:-}:" in
  *":${NIXL_PYTHON_PATH}:"*) ;;
  *) export PYTHONPATH="${NIXL_PYTHON_PATH}\${PYTHONPATH:+:\${PYTHONPATH}}" ;;
esac
EOF
}

# Keep build-only libfabric until NIXL is installed; hwloc is retained outside this workspace.
WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT
LIBFABRIC_BUILD_PATH="${WORK_DIR}/libfabric"
cd "${WORK_DIR}"

build_libfabric_for_nixl
prepare_hwloc
build_and_install_nixl
