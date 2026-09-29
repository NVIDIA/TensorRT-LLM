#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENMPI_SOURCE_ARCHIVE=/opt/hpcx/sources/openmpi5-gitclone.tar.gz
OPENMPI_PREFIX=/opt/hpcx/ompi5
# Backport of OpenMPI v5.0.x commit d054029e8a9eb60f887a7c69e2c284d88152650b
# against HPC-X 2.50's OpenMPI v5.0.10rc2-gb99be7132e source archive.
OPENMPI_PATCH="${SCRIPT_DIR}/patches/openmpi/d054029e-request-add-wait-sync-memory-barriers.diff"

case "$(uname -m)" in
    aarch64|x86_64) ;;
    *)
        echo "Skipping the OpenMPI wait-sync backport on an unsupported architecture"
        exit 0
        ;;
esac

active_openmpi_prefix="$(readlink -f /usr/local/mpi 2>/dev/null || true)"
if [[ "${active_openmpi_prefix}" != "${OPENMPI_PREFIX}" ]]; then
    echo "Skipping the OpenMPI 5 wait-sync backport: active prefix is ${active_openmpi_prefix:-unknown}"
    exit 0
fi

if [[ ! -f "${OPENMPI_SOURCE_ARCHIVE}" ]]; then
    echo "HPC-X OpenMPI source archive not found: ${OPENMPI_SOURCE_ARCHIVE}" >&2
    exit 1
fi

source_dir="$(mktemp -d /tmp/openmpi-wait-sync.XXXXXX)"
trap 'rm -rf "${source_dir}"' EXIT

# HPC-X relocates OpenMPI after building it and gives its ELF objects relative
# RUNPATHs.  A plain in-place `make install` replaces those objects with ones
# that have no RUNPATH, allowing the loader to satisfy OpenMPI 5 dependencies
# from the sibling OpenMPI 4 prefix.  Preserve the package's path-specific
# RUNPATHs and restore them after installing the patched build.
rpath_manifest="${source_dir}/hpcx-openmpi5-rpaths.tsv"
while IFS= read -r -d '' elf; do
    if rpath="$(patchelf --print-rpath "${elf}" 2>/dev/null)" && [[ -n "${rpath}" ]]; then
        printf '%s\t%s\n' "${elf#${OPENMPI_PREFIX}/}" "${rpath}" >> "${rpath_manifest}"
    fi
done < <(find "${OPENMPI_PREFIX}" -type f -print0)

if [[ ! -s "${rpath_manifest}" ]]; then
    echo "No HPC-X OpenMPI 5 RUNPATHs were found to preserve" >&2
    exit 1
fi

tar -xzf "${OPENMPI_SOURCE_ARCHIVE}" --strip-components=1 -C "${source_dir}"
cd "${source_dir}"
if git apply --reverse --check --no-index "${OPENMPI_PATCH}"; then
    echo "OpenMPI source already contains the wait-sync barriers"
else
    git apply --check --no-index "${OPENMPI_PATCH}"
    git apply --no-index "${OPENMPI_PATCH}"
fi

unset PMIX_VERSION
# HPC-X 2.50's relocated ucx.pc still points at the absent ucx/mt prefix.
# Prefer the explicit --with-ucx prefix below over that stale metadata.
ucx_USE_PKG_CONFIG=0 ./configure \
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

while IFS=$'\t' read -r relative_path rpath; do
    installed_elf="${OPENMPI_PREFIX}/${relative_path}"
    if [[ ! -f "${installed_elf}" ]]; then
        echo "OpenMPI install did not reproduce ${relative_path}" >&2
        exit 1
    fi
    patchelf --set-rpath "${rpath}" "${installed_elf}"
done < "${rpath_manifest}"

pml_ucx="${OPENMPI_PREFIX}/lib/openmpi/mca_pml_ucx.so"
pml_libmpi="$(ldd "${pml_ucx}" | awk '$1 ~ /^libmpi\.so/ {print $3; exit}')"
if [[ -z "${pml_libmpi}" ]] \
    || [[ "$(readlink -f "${pml_libmpi}")" != "$(readlink -f "${OPENMPI_PREFIX}/lib/libmpi.so.40")" ]]; then
    echo "Patched OpenMPI 5 UCX PML does not resolve its matching libmpi" >&2
    ldd "${pml_ucx}" >&2
    exit 1
fi

"${OPENMPI_PREFIX}/bin/ompi_info" --version
