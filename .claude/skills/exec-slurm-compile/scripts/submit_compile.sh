#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# submit_compile.sh — Submit a TensorRT-LLM compile job to a SLURM cluster.
#
# This is a TEMPLATE. The agent should copy and customize it per the user's
# environment before executing. Key variables to set:
#   container_image  — path to .sqsh container image
#   partition        — SLURM partition name
#   account          — SLURM account
#   jobname          — descriptive job name
#   mount_dir        — top-level directory to bind-mount into the container
#   scripts_dir      — directory containing compile.slurm and compile.sh
#   repo_dir         — path to the TensorRT-LLM repo to compile
#
# Usage: bash submit_compile.sh

set -euo pipefail

# ── User configuration (EDIT THESE) ─────────────────────────────────────
container_image="<CONTAINER_IMAGE_PATH>"   # e.g., /path/to/image.sqsh
partition="<PARTITION>"                     # e.g., batch
account="<ACCOUNT>"                        # e.g., my_account
jobname="<JOB_NAME>"                       # e.g., trtllm-compile.username

mount_dir="<MOUNT_DIR>"                    # e.g., /shared/users
scripts_dir="<SCRIPTS_DIR>"               # directory containing compile.slurm & compile.sh
repo_dir="<REPO_DIR>"                      # path to TensorRT-LLM repo
# ─────────────────────────────────────────────────────────────────────────

# Optional: extra flags for build_wheel.py (appended after repo_dir)
extra_build_args=()

echo "Submitting compile job..."
echo "  Container: ${container_image}"
echo "  Repo:      ${repo_dir}"
echo "  Partition:  ${partition} / Account: ${account}"

sbatch \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gres=gpu:4 \
    --partition="${partition}" \
    --account="${account}" \
    --job-name="${jobname}" \
    "${scripts_dir}/compile.slurm" \
    "${container_image}" "${mount_dir}" "${scripts_dir}" "${repo_dir}" \
    "${extra_build_args[@]}"
