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

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH -o log/compile.sbatch.%j.out
#
# compile.slurm — SLURM batch script that launches a container and runs compile.sh.
# Usage: sbatch [sbatch_flags...] compile.slurm <container_image> <mount_dir> <scripts_dir> <repo_dir> [build_wheel_args...]

container_image=$1
mount_dir=$2
scripts_dir=$3
repo_dir=$4
shift 4  # remaining args forwarded to compile.sh

logdir=${scripts_dir}/log
mkdir -p "${logdir}"

srun \
    --container-image="${container_image}" \
    --container-mounts="${mount_dir}:${mount_dir}" \
    --no-container-mount-home \
    --mpi=pmix \
    bash "${scripts_dir}/compile.sh" "${repo_dir}" "$@" \
    2>&1 | tee "${logdir}/compile_${SLURM_JOB_ID}.srun.log"
