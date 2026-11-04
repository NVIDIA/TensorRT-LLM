#!/bin/bash

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

# Replace this image's CUDA forward-compatibility driver with the one from another image.
#
# Usage: install_cuda_compat.sh <source /usr/local>
#
# <source /usr/local> is the /usr/local tree of the image that provides the compat driver. Only
# the driver libraries under /usr/local/cuda-<major.minor>/compat (lib.real, and lib.real-nvgpu
# where present) are replaced; the CUDA toolkit is untouched, so the source image must ship the
# same CUDA major.minor as this one. The base image's shell init (/etc/shinit_v2) checks and
# selects the compat driver from that directory at container start, so nothing else changes.

set -euo pipefail

SRC_USR_LOCAL="${1:?usage: install_cuda_compat.sh <source /usr/local>}"

CUDA_DIR="$(readlink -f /usr/local/cuda)"
CUDA_DIR_NAME="$(basename "${CUDA_DIR}")"
DST_COMPAT="${CUDA_DIR}/compat"
SRC_COMPAT="${SRC_USR_LOCAL}/${CUDA_DIR_NAME}/compat"

if ! ls "${SRC_COMPAT}"/lib.real/libcuda.so.* >/dev/null 2>&1; then
    echo "No ${CUDA_DIR_NAME}/compat/lib.real/libcuda.so.* in the compat source image; it must" \
         "ship the same CUDA version as this image (${CUDA_DIR_NAME})." >&2
    exit 1
fi
if ! ls "${DST_COMPAT}"/lib.real/libcuda.so.* >/dev/null 2>&1; then
    echo "This image has no ${DST_COMPAT}/lib.real/libcuda.so.*; nothing to replace." >&2
    exit 1
fi

echo "Replacing CUDA compat driver $(basename "$(ls "${DST_COMPAT}"/lib.real/libcuda.so.*.* | head -1)")" \
     "with $(basename "$(ls "${SRC_COMPAT}"/lib.real/libcuda.so.*.* | head -1)"):"
for entry in lib.real lib.real-nvgpu; do
    rm -rf "${DST_COMPAT:?}/${entry}"
    if [ -d "${SRC_COMPAT}/${entry}" ]; then
        cp -a "${SRC_COMPAT}/${entry}" "${DST_COMPAT}/${entry}"
    fi
done
# The shell init skips its compat check on hosts with a `.<driver>.<host>.checked` marker; drop
# markers recorded against the replaced driver so the check runs against this one.
find "${DST_COMPAT}" -maxdepth 1 -name '.*.checked' -delete

ldconfig
test -e "${DST_COMPAT}/lib.real/libcuda.so.1"
ls -l "${DST_COMPAT}"/lib.real/libcuda.so.*
