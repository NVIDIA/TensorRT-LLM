#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 ("golden prairie") e2e sanity — full model on 16 GPUs (4x GB300).
#
# Preferred: submit via examples/kimi_k3/run_kimi.py, which resolves all
# inputs from ~/.config/kimi-bringup.ini (see kimi-bringup.ini.example).
# Direct sbatch use requires exporting the inputs yourself:
#
#   KIMI_K3_MODEL_DIR=/path/to/goldenprairie-final-weights_vv1 \
#   IMAGE=/path/to/trtllm-pytorch-26.05-...sbsa...sqsh \
#   sbatch --export=ALL examples/kimi_k3/sanity_kimi_k3.sbatch
#
# Quick 4-GPU pipeline-only variant (single tray, 4/93 layers, ~10 min):
# add  -N1  and  KIMI_K3_TP=4 KIMI_K3_NUM_LAYERS_OVERRIDE=4  to the
# environment.
#
#SBATCH --job-name=k3-sanity
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --output=k3-sanity-%j.log

set -euo pipefail

# Required inputs — no defaults; submit via examples/kimi_k3/run_kimi.py
# (reads ~/.config/kimi-bringup.ini) or export these before sbatch.
: "${KIMI_K3_MODEL_DIR:?set via run_kimi.py or export (goldenprairie-final-weights_vv1 checkpoint)}"
: "${IMAGE:?set via run_kimi.py or export (path to the 26.05 sbsa .sqsh)}"
KIMI_K3_CACHE_DIR=${KIMI_K3_CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
export KIMI_K3_MODEL_DIR KIMI_K3_CACHE_DIR
REPO=${REPO:-$SLURM_SUBMIT_DIR}

MOUNTS="$REPO:$REPO:rw,$KIMI_K3_MODEL_DIR:$KIMI_K3_MODEL_DIR:ro"
MOUNTS+=",$KIMI_K3_CACHE_DIR:$KIMI_K3_CACHE_DIR:rw"

srun --mpi=pmix \
    --container-image="$IMAGE" \
    --container-mounts="$MOUNTS" \
    bash -c "
        set -x
        ulimit -n 65536

        # --- Kimi K3 runtime env (every rank, before any python import) ---
        CACHE=\$KIMI_K3_CACHE_DIR
        # Import path: the repo itself (the editable install records the
        # build-time mount point). The fused MoE default is the in-tree
        # TRTLLM-Gen SiTU op (KIMI_K3_FUSED_MOE=native) — fully
        # self-contained, no external kernel collection needed.
        export PYTHONPATH=$REPO:\${PYTHONPATH:-}
        # Persistent caches for stock flashinfer JIT artifacts (non-MoE ops).
        export FLASHINFER_WORKSPACE_BASE=\$CACHE/fi_workspace
        export FLASHINFER_CUBIN_DIR=\$CACHE/fi_cubins
        export HF_HOME=\${HF_HOME:-\$CACHE/hf_home}
        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\" \"\$TRITON_CACHE_DIR\"
        export KIMI_K3_CKPT=\$KIMI_K3_MODEL_DIR
        export KIMI_K3_FUSED_MOE=\${KIMI_K3_FUSED_MOE:-native}
        export PATH='$REPO/.venv-3.12/bin':\$PATH

        export KIMI_K3_TP=${KIMI_K3_TP:-16}
        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python \
            '$REPO/examples/kimi_k3/sanity_kimi_k3.py'
    "
