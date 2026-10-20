#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prepare the FA4 (flash_attn.cute) copy that MiniMax-M3's FP8 dense prefill uses on SM107.
#
# Usage: bash setup_fa4.sh /absolute/path/to/new/fa4-dir
#
# Copies the installed flash_attn package (flash-attn-4 4.0.0b19) into a new directory and applies
# 3rdparty/patches/fa4_fp8_rescale0.patch: for FP8 queries the forward rescales the row max on every increase
# (rescale_threshold 0.0 instead of 4.0), so P stays within E4M3 range. Prepend the directory to
# PYTHONPATH in every worker. Without the change the dense prefill keeps fmha_sm100 and logs
# "FA4 lacks the FP8 rescale_threshold 0.0 change".
set -euo pipefail

readonly BASE_SHA256="7d5f2baf8b9c9e28aade1591f6fd9353e16956a7b9b12e8bd00577fdab7f61e3"
readonly PATCHED_SHA256="418c2d40442b7e336b5877bb19d55205b7582e548c8536b2a64e82430156f190"
readonly PATCH_SHA256="66988cec21ec806ec04f6d6a8105ba71d3dd8b77d87376aa7a0a5616321c5d8d"

fa4_destination="${1:?Supply a new directory for the patched FA4 copy}"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -e "${fa4_destination}" ]]; then
    echo "Destination already exists: ${fa4_destination}" >&2
    exit 1
fi
package_directory="$(python3 -c 'import os, flash_attn; print(os.path.dirname(flash_attn.__file__))')"
target="flash_attn/cute/flash_fwd_sm100.py"
actual_sha256="$(sha256sum "${package_directory}/cute/flash_fwd_sm100.py" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${BASE_SHA256}" ]]; then
    echo "Unexpected ${target} SHA256 ${actual_sha256}; expected ${BASE_SHA256} (flash-attn-4 4.0.0b19)" >&2
    exit 1
fi

patch_file="${script_directory}/../../../3rdparty/patches/fa4_fp8_rescale0.patch"
actual_sha256="$(sha256sum "${patch_file}" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${PATCH_SHA256}" ]]; then
    echo "Unexpected patch SHA256 ${actual_sha256}; expected ${PATCH_SHA256}" >&2
    exit 1
fi

mkdir -p "${fa4_destination}"
cp -R "${package_directory}" "${fa4_destination}/flash_attn"
find "${fa4_destination}/flash_attn" -name __pycache__ -prune -exec rm -rf {} +
patch -d "${fa4_destination}" -p1 --forward --quiet < "${patch_file}"
actual_sha256="$(sha256sum "${fa4_destination}/${target}" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${PATCHED_SHA256}" ]]; then
    echo "Unexpected patched SHA256 ${actual_sha256}; expected ${PATCHED_SHA256}" >&2
    exit 1
fi
echo "FA4 is ready; prepend ${fa4_destination} to PYTHONPATH in every worker."
