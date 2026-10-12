#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prepare the native MiniMax MSA checkout used by TRTLLM_MSA_NVDEV_PREFILL=1.
#
# Usage: bash setup_msa.sh /absolute/path/to/new/msa-checkout
#
# The checkout is MiniMax-AI/MSA at a pinned nv_dev commit plus
# msa_q8kv4_prefill_sm107.patch.gz, which extends the Q8KV4 (NVFP4 KV) prefill
# wrapper with SM107 support, strided cache planes, device global scales and
# four-token V-scale groups, and waits on the upstream grid before the first
# global load when launched with programmatic dependent launch. The FP8 (Q8KV8)
# prefill wrapper is used unmodified.
set -euo pipefail

readonly MSA_REPOSITORY="https://github.com/MiniMax-AI/MSA.git"
readonly MSA_COMMIT="c8e1013a16a708d2a7656805d5fc26f2b87715d1"
readonly CUTLASS_COMMIT="dc45f979ae336a235da1676b311f35efeb30149a"
readonly PATCH_SHA256="e92aad52e8e5b42fedee4a24ff36ce9e81277c62cb6fcf228f1b4d8d72a28554"

msa_destination="${1:?Supply a new directory for the MSA checkout}"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -e "${msa_destination}" ]]; then
    echo "Destination already exists: ${msa_destination}" >&2
    exit 1
fi
patch_file="$(mktemp "${TMPDIR:-/tmp}/msa-patch.XXXXXX")"
trap 'rm -f "${patch_file}"' EXIT
gzip -dc "${script_directory}/msa_q8kv4_prefill_sm107.patch.gz" > "${patch_file}"
actual_sha256="$(sha256sum "${patch_file}" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${PATCH_SHA256}" ]]; then
    echo "Unexpected patch SHA256 ${actual_sha256}; expected ${PATCH_SHA256}" >&2
    exit 1
fi

git clone "${MSA_REPOSITORY}" "${msa_destination}"
git -C "${msa_destination}" checkout --detach "${MSA_COMMIT}"
git -C "${msa_destination}" apply --check "${patch_file}"
git -C "${msa_destination}" apply "${patch_file}"
git -C "${msa_destination}" submodule update --init third_party/cutlass
actual_cutlass="$(git -C "${msa_destination}/third_party/cutlass" rev-parse HEAD)"
if [[ "${actual_cutlass}" != "${CUTLASS_COMMIT}" ]]; then
    echo "Unexpected CUTLASS commit ${actual_cutlass}; expected ${CUTLASS_COMMIT}" >&2
    exit 1
fi
echo "MSA is ready at ${msa_destination}; set MSA_ROOT to this path and source runtime_env.sh."
