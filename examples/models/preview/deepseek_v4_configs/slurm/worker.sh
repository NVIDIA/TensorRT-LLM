#!/bin/bash
# One rank of a context or generation server. slurm/launch.sbatch runs it with srun inside the
# container, one task per GPU; rank 0 also serves HTTP on <port>.
#   usage: worker.sh <run dir> ctx|gen <this server's GPUs on the node, in rank order> <port> <rank-0 host>
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
run_dir=$1 role=$2 gpus=$3 port=$4 master=$5
source "$run_dir/cluster.env"
source "$run_dir/point.env"
# The node-local copy that slurm/stage.sh made, when STAGE_DIR is set.
model=$MODEL_PATH
if [ -n "${STAGE_DIR:-}" ]; then model=$STAGE_DIR/$(basename "$MODEL_PATH"); fi

# Rank i drives the i-th GPU of the list. The order is part of the measured placement.
export CUDA_VISIBLE_DEVICES=$gpus
# KV transfer (NIXL over UCX) inside the NVL72 needs cuda_ipc, and an `rc` transport breaks it.
# Set here, inside the container, because some sites preset UCX_TLS in the container environment.
export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp
# torch.distributed rendezvous of this server, as measured: one port per server.
export MASTER_ADDR=$master MASTER_PORT=$((port - 3000))
# JIT and autotuner caches: node-local for CUDA and Triton; the rest in CACHE_DIR (shared by runs,
# so only the first start-up compiles and tunes) or, unset, in the run directory.
cache=${CACHE_DIR:-$run_dir/cache}
mkdir -p "$cache"
export CUDA_CACHE_PATH=/tmp/cuda_cache_dsv41 TRITON_CACHE_DIR=/tmp/triton_cache_dsv41 \
    CUTE_DSL_CACHE_DIR=$cache/cute_dsl DG_JIT_CACHE_DIR=$cache/deep_gemm \
    TLLM_AUTOTUNER_CACHE_PATH=$cache/autotuner_$role.json

if [ "$role" = ctx ]; then role_env=("${CTX_ENV[@]}"); else role_env=("${GEN_ENV[@]}"); fi
for kv in "${FABRIC_ENV[@]}" "${WORKER_ENV[@]}" "${role_env[@]}"; do export "$kv"; done
if [ -n "$FORCE_ACCEPTED_TOKENS" ]; then
    export TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=$FORCE_ACCEPTED_TOKENS
fi

# Host memory from the CPU NUMA nodes only. Context ranks, and the serve frontends their rank 0
# starts, also run on the CPUs of their GPU's NUMA node.
bind=(numactl -m 0,1)
if [ "$role" = ctx ]; then
    IFS=, read -ra list <<< "$gpus"
    gpu=${list[$SLURM_LOCALID]}
    numa=$(nvidia-smi topo -C -i "$gpu" | sed -n 's/.*closest CPU: *\([0-9][0-9]*\).*/\1/p' | head -1) || true
    if [ -n "$numa" ]; then
        bind+=(--cpunodebind="$numa")
    else
        echo "worker.sh: no NUMA node found for GPU $gpu; CPUs not bound" >&2
    fi
fi

ulimit -n "$(ulimit -Hn)" 2> /dev/null || true
exec "${bind[@]}" trtllm-llmapi-launch \
    trtllm-serve "$model" --host 0.0.0.0 --port "$port" --config "$run_dir/$role.yaml"
