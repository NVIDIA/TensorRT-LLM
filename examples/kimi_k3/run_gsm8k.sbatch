#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 ("golden prairie") GSM8K eval on 16 GPUs (4x GB300 trays).
#
# Preferred: submit via examples/kimi_k3/run_kimi.py, which resolves all
# inputs from ~/.config/kimi-bringup.ini (see kimi-bringup.ini.example).
# Direct sbatch use requires exporting the inputs yourself:
#
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

# Required inputs — no defaults; submit via examples/kimi_k3/run_kimi.py
# (reads ~/.config/kimi-bringup.ini) or export these before sbatch.
: "${KIMI_K3_MODEL_DIR:?set via run_kimi.py or export (goldenprairie-final-weights_vv1 checkpoint)}"
: "${IMAGE:?set via run_kimi.py or export (path to the 26.05 sbsa .sqsh)}"
KIMI_K3_CACHE_DIR=${KIMI_K3_CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
export KIMI_K3_MODEL_DIR KIMI_K3_CACHE_DIR
REPO=${REPO:-$SLURM_SUBMIT_DIR}

# Mount every input at its host path so the env vars are valid inside too.
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
        # self-contained, no external kernel collection needed. To run the
        # private-snapshot flashinfer path (KIMI_K3_FUSED_MOE=1 — still
        # needed for SA spec dec until in-tree cubins cover the bs<=8
        # shapes), restore
        #   SNAP=\$KIMI_K3_OPT_WORK_DIR/trtllmgen_MOE
        #   export PYTHONPATH=$REPO:\$SNAP:...
        #   export FLASHINFER_PRIVATE_CUBIN_DIR=\$SNAP/local_cubins/20260617_v0613rc1_situ_v0611_barrierfix
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

        # trtllm-eval is a console script whose shebang records the
        # build-time venv path — run it through python explicitly so it
        # works regardless of where the repo is mounted.
        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python '$REPO/.venv-3.12/bin/trtllm-eval' \
            --model \"\$KIMI_K3_CKPT\" \
            --backend pytorch \
            --tp_size ${KIMI_K3_TP:-16} \
            --max_batch_size ${KIMI_K3_MAX_BATCH_SIZE:-32} \
            --max_seq_len ${KIMI_K3_MAX_SEQ_LEN:-8192} \
            --max_num_tokens ${KIMI_K3_MAX_NUM_TOKENS:-8192} \
            --kv_cache_free_gpu_memory_fraction 0.25 \
            --disable_kv_cache_reuse \
            --trust_remote_code \
            --config '${KIMI_K3_EVAL_CONFIG:-$REPO/examples/kimi_k3/eval_extra_llm_options_dep16.yaml}' \
            gsm8k ${GSM8K_EXTRA_ARGS:-}
    "
