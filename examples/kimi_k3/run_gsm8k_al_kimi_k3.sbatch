#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 + DSpark: GSM8K acceptance-length (AL) measurement.
#
# Sibling of run_eval_kimi_k3.sbatch. That script measures ACCURACY on the
# lm-eval default workload; this one measures ACCEPTANCE on the workload the
# DSpark drafter was distilled against. They are not interchangeable — the
# workload alone moves AL by ~1.8, which is larger than any code change we
# have measured on this path, so an AL quoted without its prompt recipe is
# meaningless.
#
#   sbatch examples/kimi_k3/run_gsm8k_al_kimi_k3.sbatch \
#       --model /path/to/kimi-k3-checkpoint \
#       --drafter /path/to/dspark-drafter \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# --parallel tep8 (default, 2 nodes) or tep16 (submit with --nodes=4
# --segment=4; the sbatch CLI overrides the directives below).
#
# TEP, not DEP: enable_attention_dp replicates every non-routed weight on
# every rank, which does not fit a 184 GB GB200 at any EP width and needs a
# 288 GB part even at EP16. Attention-TP shards those weights instead.
#
# GB300 is sm_103, NOT sm_100. A build whose CMAKE_CUDA_ARCHITECTURES lacks
# 103 loads and initializes fine, then dies at the first MLA context attention
# with "Deepseek should be supported by fmha in context part"
# (attentionOp.cpp) after the dispatcher logs "Fall back to unfused MHA ...
# in sm_103". Build with -a "100-real;103-real".
#
# ---------------------------------------------------------------------------
# The AL recipe, and why each knob is here
# ---------------------------------------------------------------------------
# --apply_chat_template  K3 ships no jinja template, so this reaches the
#                        checkpoint's own encoder (trust_remote_code), which
#                        opens a thinking channel and defaults to
#                        thinking_effort="max". SGLang's serving path calls
#                        the same tokenizer entry point, so both sides get the
#                        same prompt structure.
# --num_fewshot 0        The drafter was distilled on chat traffic — one
#                        question per request. The gsm8k task yaml pins 5
#                        shots, which is out of distribution for it.
# --max_output_length    Thinking output runs ~240 tokens per question; the
#                        GSM8K task default of 256 truncates half the corpus
#                        inside the thinking block.
# TLLM_EVAL_SPEC_STATS=1 Prints the AL/AR summary this script exists for.
# KIMI_K3_ROUTER_BF16=0  With attention-DP off the MoE router gate would
#                        otherwise take its bf16 fast path, which can flip
#                        borderline expert picks.
#
# ---------------------------------------------------------------------------
# Reading the output, and comparing it to SGLang
# ---------------------------------------------------------------------------
# The AL this prints is TRT-LLM's convention: total decoded tokens / total
# decode iterations, and the denominator INCLUDES the context forward (which
# emits one token). SGLang's token-weighted acc_len divides by verify steps
# only. To compare, use the AR line instead:
#
#     acc_len = 1 + K * AR                (K = max_draft_len = 7)
#
# Measured by this script against the radixark Kimi-K3-DSpark drafter,
# 200 samples:
#
#   target   layout   AL      AR      acc_len
#   MXFP4    TEP8     5.920   72.2%   6.05
#   MXFP4    TEP16    5.927   72.3%   6.05
#   NVFP4    TEP16    5.916   72.0%   6.04
#
# NVFP4 needs --config eval_extra_llm_options_dspark_nvfp4.yaml. The three
# agree to 0.5pt of AR, so neither the expert-parallel width nor the routed
# quantization moves acceptance; treat a run outside 71-73% as a regression
# rather than as variance. SGLang reports acc_len 6.09 on the MXFP4 pair, and
# the residual sits almost entirely on the first draft position.
#
# Accuracy is NOT eval-grade on this workload: at 0 shots the model stops
# emitting "#### <answer>", so the gsm8k strict filter reads ~0. Use
# run_eval_kimi_k3.sbatch, or the TestKimiK3DSpark integration test (which
# adds a format-pinning system prompt), when the score is what you want.
#
# Worktree submits: export REPO=<worktree>, EXTRA_MOUNTS="$MAIN:$MAIN:rw",
# and submit with --export=ALL (mirrors run_eval_kimi_k3.sbatch).
#
#SBATCH --job-name=kimi-k3-gsm8k-al
#SBATCH --partition=gb300
#SBATCH --account=${account}
#SBATCH --nodes=2
#SBATCH --segment=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=kimi-k3-gsm8k-al-%j.log

set -euo pipefail

CLEANUP_FILES=()
trap '[ ${#CLEANUP_FILES[@]} -eq 0 ] || rm -f "${CLEANUP_FILES[@]}"' EXIT

usage() {
    echo "Usage: sbatch $0 --model PATH --drafter PATH --image PATH" \
         "[--parallel tep8|tep16] [--num-samples N] [--config PATH]"
}

MODEL=""
DRAFTER=""
CONTAINER_IMAGE=""
PARALLEL=tep8
NUM_SAMPLES=200
# DSpark LLM-options template. Override for a checkpoint whose MoE
# backend differs from the MXFP4 default -- the NVFP4 export needs
# moe_config.backend: TRTLLM, which AUTO does not select.
EVAL_TEMPLATE=""

require_value() {
    [[ $2 -ge 2 ]] || { echo "error: $1 requires a value" >&2; usage >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        require_value "$1" $#; MODEL=$2; shift 2 ;;
        --drafter)      require_value "$1" $#; DRAFTER=$2; shift 2 ;;
        --image)        require_value "$1" $#; CONTAINER_IMAGE=$2; shift 2 ;;
        --parallel)     require_value "$1" $#; PARALLEL=$2; shift 2 ;;
        --num-samples)  require_value "$1" $#; NUM_SAMPLES=$2; shift 2 ;;
        --config)       require_value "$1" $#; EVAL_TEMPLATE=$2; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$MODEL" ]]           || { echo "error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$DRAFTER" ]]         || { echo "error: --drafter is required" >&2; usage >&2; exit 2; }
[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]]           || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$DRAFTER" ]]         || { echo "error: drafter path does not exist: $DRAFTER" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }
[[ -z "$EVAL_TEMPLATE" || -e "$EVAL_TEMPLATE" ]] || { echo "error: config path does not exist: $EVAL_TEMPLATE" >&2; exit 2; }

EVAL_TEMPLATE=${EVAL_TEMPLATE:-$REPO/examples/kimi_k3/eval_extra_llm_options_dspark.yaml}

case "$PARALLEL" in
    tep8)  TP_SIZE=8  ;;
    tep16) TP_SIZE=16 ;;
    *) echo "error: unknown --parallel: $PARALLEL (expected tep8 or tep16)" >&2; exit 2 ;;
esac

REPO=${REPO:-$SLURM_SUBMIT_DIR}
VENV=${TRTLLM_VENV:-$REPO/.venv-3.12}

# Per-job copy of the DSpark template with the drafter path and the parallel
# layout substituted. It must live under $REPO: that is the only path
# guaranteed to be container-mounted identically on every node.
EVAL_CONFIG=$REPO/examples/kimi_k3/.eval_al_runtime.$SLURM_JOB_ID.yaml
CLEANUP_FILES+=("$EVAL_CONFIG")
# Escape sed replacement metacharacters so arbitrary drafter paths substitute
# literally.
DRAFTER_ESC=$(printf '%s' "$DRAFTER" | sed 's/[&|\\]/\\&/g')
sed -e "s|speculative_model: .*|speculative_model: $DRAFTER_ESC|" \
    -e "s|^tensor_parallel_size: .*|tensor_parallel_size: $TP_SIZE|" \
    -e "s|^moe_expert_parallel_size: .*|moe_expert_parallel_size: $TP_SIZE|" \
    -e "s|^enable_attention_dp: .*|enable_attention_dp: false|" \
    "$EVAL_TEMPLATE" > "$EVAL_CONFIG"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$DRAFTER:$DRAFTER:ro"
if [ -n "${EXTRA_MOUNTS:-}" ]; then
    MOUNTS+=",$EXTRA_MOUNTS"
fi

srun --mpi=pmix \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    bash -c "
        set -x
        ulimit -n 65536
        export PYTHONNOUSERSITE=1
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        # DeepGEMM JIT cache: a shared cache lets two ranks race the
        # temporary-file rename and abort on a null JIT runtime
        # (jit/compiler.hpp), which hangs the job rather than failing it.
        export DG_JIT_CACHE_DIR=/tmp/deep-gemm-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$DG_JIT_CACHE_DIR\"
        export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}
        export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}
        export HF_HOME=\"${HF_HOME:-\$HOME/.cache/huggingface}\"
        export PATH=\"$VENV/bin:\$PATH\"
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"

        export TLLM_EVAL_PARTIAL_SCORES_EVERY=\"\${TLLM_EVAL_PARTIAL_SCORES_EVERY:-100}\"
        # The point of this script.
        export TLLM_EVAL_SPEC_STATS=1
        # See the header: keeps the router gate off its bf16 fast path.
        export KIMI_K3_ROUTER_BF16=\"\${KIMI_K3_ROUTER_BF16:-0}\"

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
            '$VENV/bin/trtllm-eval' \
            --model \"$MODEL\" \
            --backend pytorch \
            --tp_size $TP_SIZE \
            --max_batch_size 8 \
            --max_seq_len 8192 \
            --max_num_tokens 4096 \
            --kv_cache_free_gpu_memory_fraction 0.20 \
            --disable_kv_cache_reuse \
            --trust_remote_code \
            --config '$EVAL_CONFIG' \
            gsm8k \
            --apply_chat_template \
            --num_fewshot 0 \
            --max_input_length 2048 \
            --max_output_length 4096 \
            --num_samples $NUM_SAMPLES
    "
