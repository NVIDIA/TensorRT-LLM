#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENMPI_SOURCE_ARCHIVE=/opt/hpcx/sources/openmpi4-gitclone.tar.gz
OPENMPI_PREFIX=/opt/hpcx/ompi4
# OpenMPI upstream commit 0c86834a68c8691088570925e33b854758e1e400.
OPENMPI_PATCH="${SCRIPT_DIR}/patches/openmpi/0c86834a-request-add-wait-sync-memory-barriers.diff"

if [[ "$(uname -m)" != "aarch64" || ! -d "${OPENMPI_PREFIX}" ]]; then
    echo "Skipping the OpenMPI wait-sync backport outside ARM64 HPC-X images"
    exit 0
fi

if [[ ! -f "${OPENMPI_SOURCE_ARCHIVE}" ]]; then
    echo "HPC-X OpenMPI source archive not found: ${OPENMPI_SOURCE_ARCHIVE}" >&2
    exit 1
fi

source_dir="$(mktemp -d /tmp/openmpi-wait-sync.XXXXXX)"
trap 'rm -rf "${source_dir}"' EXIT

tar -xzf "${OPENMPI_SOURCE_ARCHIVE}" --strip-components=1 -C "${source_dir}"
cd "${source_dir}"
git apply --check --no-index "${OPENMPI_PATCH}"
git apply --no-index "${OPENMPI_PATCH}"

unset PMIX_VERSION
./configure \
    --prefix="${OPENMPI_PREFIX}" \
    --with-libevent=internal \
    --enable-mpi1-compatibility \
    --without-xpmem \
    --with-cuda=/usr/local/cuda \
    --with-slurm \
    --with-platform=contrib/platform/mellanox/optimized \
    --with-hcoll=/opt/hpcx/hcoll \
    --with-ucx=/opt/hpcx/ucx \
    --with-ucc=/opt/hpcx/ucc
make -j"$(nproc)"
make install

"${OPENMPI_PREFIX}/bin/ompi_info" --version
