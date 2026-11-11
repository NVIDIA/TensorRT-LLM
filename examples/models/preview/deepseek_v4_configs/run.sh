#!/bin/bash
# Submits one point as a Slurm job.
#   ./run.sh <point dir> [--smoke] [--dry-run]
#     --smoke     functional check: 5-minute client window, at most 512 sessions unless overridden
#     --dry-run   print the sbatch command and the job's srun commands, render the configs; submit nothing
#   env: CLUSTER_ENV   cluster settings file (default: cluster.env next to this script)
#        CONCURRENCY   client sessions instead of the point's (for a smaller smoke run)
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euo pipefail
pkg=$(cd "$(dirname "$0")" && pwd)
point='' smoke=0 dry=0
for arg in "$@"; do
    case $arg in
        --smoke) smoke=1 ;;
        --dry-run) dry=1 ;;
        -*) echo "run.sh: unknown option $arg" >&2; exit 2 ;;
        *) point=$arg ;;
    esac
done
if [ ! -f "$point/point.env" ]; then
    sed -n '2,7s/^# \{0,1\}//p' "$0" >&2
    exit 2
fi
point=$(cd "$point" && pwd)
cluster_env=$(realpath "${CLUSTER_ENV:-$pkg/cluster.env}")
concurrency=${CONCURRENCY:-}
source "$cluster_env"
source "$point/point.env"
if [ -z "$concurrency" ]; then
    concurrency=$CONCURRENCY
    if ((smoke && concurrency > 512)); then concurrency=512; fi
fi
repo=$(git -C "$pkg" rev-parse --show-toplevel)
source_commit=$(git -C "$repo" rev-parse HEAD)
# The direct parent is a compatible build when this commit changes only this config package.
runtime_parent=$(git -C "$repo" rev-parse HEAD^)
if [ -n "$(git -C "$repo" diff --name-only "$runtime_parent" HEAD -- . \
    ':(exclude)examples/models/preview/deepseek_v4_configs')" ]; then
    runtime_parent=$source_commit
fi

ctx_per_node=${#CTX_GPU_ORDER[@]}
nodes=$((GEN_SERVERS * GEN_NODES_PER_SERVER + (CTX_SERVERS + ctx_per_node - 1) / ctx_per_node))
if [ "$CLIENT_NODE" = dedicated ]; then nodes=$((nodes + 1)); fi
name=$(basename "$point")
time_limit=$TIME_LIMIT suffix=''
if ((smoke)); then time_limit=$TIME_LIMIT_SMOKE suffix=-smoke; fi
run_dir=$WORK_DIR/$(date +%Y%m%d-%H%M%S)-$name$suffix

sbatch_args=(--job-name="dsv41-$name" --partition="$PARTITION" --account="$ACCOUNT"
    --time="$time_limit" --nodes="$nodes" --ntasks-per-node=4 --gres=gpu:4 --exclusive --mem=0
    --output="$run_dir/slurm-%j.out")
if [ "$USE_SEGMENT" = 1 ]; then sbatch_args+=(--segment="$nodes"); fi
sbatch_args+=("${SBATCH_EXTRA[@]}")
export PKG_DIR=$pkg CLUSTER_ENV=$cluster_env POINT_DIR=$point RUN_DIR=$run_dir SMOKE=$smoke \
    CLIENT_CONCURRENCY=$concurrency SOURCE_COMMIT=$source_commit RUNTIME_PARENT=$runtime_parent
if [ "$concurrency" != "$CONCURRENCY" ]; then
    echo "run.sh: concurrency $concurrency instead of the point's configured $CONCURRENCY"
fi

if ((dry)); then
    echo "sbatch ${sbatch_args[*]} $pkg/slurm/launch.sbatch"
    RUN_DIR=$(mktemp -d) DRY_RUN=1 SLURM_JOB_ID=dryrun DRY_NODES=$(printf 'node%02d ' $(seq "$nodes")) \
        exec bash "$pkg/slurm/launch.sbatch"
fi
if [[ -z "$IMAGE" || "$IMAGE" == *'<'* || "$IMAGE" == *'>'* ]]; then
    echo "run.sh: set IMAGE to the image built from this checkout (see README, Build the runtime)" >&2
    exit 1
fi
if [[ "$IMAGE" == /* && ! -f "$IMAGE" ]]; then
    echo "run.sh: image file $IMAGE not found" >&2
    exit 1
fi
[ -f "$MODEL_PATH/config.json" ] || { echo "run.sh: checkpoint config missing in $MODEL_PATH" >&2; exit 1; }
mkdir -p "$run_dir"
sbatch "${sbatch_args[@]}" "$pkg/slurm/launch.sbatch"
echo "run directory: $run_dir"
