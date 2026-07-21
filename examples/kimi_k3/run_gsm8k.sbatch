#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 ("golden prairie") GSM8K eval on 16 GPUs (4x GB300 trays).
#
# Submit from the repo root (paths default to tali's workspace; override
# KIMI_K3_* / IMAGE via the environment for other checkouts):
#
#   KIMI_K3_OPT_WORK_DIR=/path/to/exisiting_optimization_work \
#   KIMI_K3_MODEL_DIR=/path/to/goldenprairie-final-weights_vv1 \
#   IMAGE=/path/to/trtllm-pytorch-26.05-...sbsa...sqsh \
#   sbatch --export=ALL examples/kimi_k3/run_gsm8k.sbatch
#
# REPO defaults to the submit dir (must be the built TensorRT-LLM checkout
# with .venv-3.12). Partition/account/time are cluster defaults — override
# on the sbatch command line as needed.
#
#SBATCH --job-name=k3-gsm8k
#SBATCH --partition=batch_long
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=k3-gsm8k-%j.log

set -euo pipefail

# Paths — defaults point at tali's golden-prairie workspace; override via
# the environment for other checkouts.
KIMI_K3_OPT_WORK_DIR=${KIMI_K3_OPT_WORK_DIR:-/home/tali/tali/goldenprairie/exisiting_optimization_work}
KIMI_K3_MODEL_DIR=${KIMI_K3_MODEL_DIR:-/home/tali/tali/goldenprairie/goldenprairie-final-weights_vv1}
IMAGE=${IMAGE:-/home/tali/tali/trtllm-pytorch-26.05-py3-sbsa-ubuntu24.04-trt10.16.1.11-skip-tritondevel-202607151440-16194.sqsh}
export KIMI_K3_OPT_WORK_DIR KIMI_K3_MODEL_DIR
REPO=${REPO:-$SLURM_SUBMIT_DIR}

# Mount every input at its host path so the env vars are valid inside too.
MOUNTS="$REPO:$REPO:rw,$KIMI_K3_OPT_WORK_DIR:$KIMI_K3_OPT_WORK_DIR:rw"
MOUNTS+=",$KIMI_K3_MODEL_DIR:$KIMI_K3_MODEL_DIR:ro"

srun --mpi=pmix \
    --container-image="$IMAGE" \
    --container-mounts="$MOUNTS" \
    bash -c "
        set -x
        ulimit -n 65536

        # --- Kimi K3 runtime env (every rank, before any python import) ---
        SNAP=\$KIMI_K3_OPT_WORK_DIR/trtllmgen_MOE
        CACHE=\$KIMI_K3_OPT_WORK_DIR/.kimi_k3_cache
        # Import paths: the repo itself (the editable install records the
        # build-time mount point) and the private flashinfer snapshot with
        # the SiTU MXFP4 MoE cubins (must shadow the stock flashinfer).
        export PYTHONPATH=$REPO:\$SNAP:\${PYTHONPATH:-}
        export FLASHINFER_PRIVATE_CUBIN_DIR=\$SNAP/local_cubins/20260617_v0613rc1_situ_v0611_barrierfix
        # Persistent caches (first flashinfer JIT build ~10 min, then reused).
        export FLASHINFER_WORKSPACE_BASE=\$CACHE/fi_workspace
        export FLASHINFER_CUBIN_DIR=\$CACHE/fi_cubins
        export HF_HOME=\${HF_HOME:-\$CACHE/hf_home}
        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\" \"\$TRITON_CACHE_DIR\"
        export KIMI_K3_CKPT=\$KIMI_K3_MODEL_DIR
        export KIMI_K3_FUSED_MOE=\${KIMI_K3_FUSED_MOE:-1}
        export PATH='$REPO/.venv-3.12/bin':\$PATH

        # trtllm-eval is a console script whose shebang records the
        # build-time venv path — run it through python explicitly so it
        # works regardless of where the repo is mounted.
        exec trtllm-llmapi-launch python '$REPO/.venv-3.12/bin/trtllm-eval' \
            --model \"\$KIMI_K3_CKPT\" \
            --backend pytorch \
            --tp_size ${KIMI_K3_TP:-16} \
            --max_batch_size ${KIMI_K3_MAX_BATCH_SIZE:-32} \
            --max_seq_len ${KIMI_K3_MAX_SEQ_LEN:-8192} \
            --max_num_tokens ${KIMI_K3_MAX_NUM_TOKENS:-8192} \
            --kv_cache_free_gpu_memory_fraction 0.25 \
            --disable_kv_cache_reuse \
            --trust_remote_code \
            --config '$REPO/examples/kimi_k3/eval_extra_llm_options.yaml' \
            gsm8k ${GSM8K_EXTRA_ARGS:-}
    "
