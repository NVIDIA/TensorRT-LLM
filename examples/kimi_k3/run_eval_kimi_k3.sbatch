#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 accuracy evaluation (GSM8K or MMMU) on 16 NVIDIA Blackwell GPUs.
#
# Submit the job with the checkpoint and container paths:
#
#   sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
#       --model /path/to/kimi-k3-checkpoint \
#       --image /path/to/tensorrt-llm-container.sqsh \
#       [--task gsm8k|mmmu] [--parallel dep|tep]
#
# --task selects the benchmark (default: gsm8k, identical behavior to the
# former run_gsm8k_kimi_k3.sbatch).
#
# --parallel selects the 16-GPU attention layout (default: dep).
#
#   * dep: attention data-parallel + MoE EP16 (DEP16) — the layout the
#     checked-in eval YAMLs pin and the reference scores were measured with.
#   * tep: attention tensor-parallel + MoE EP16 (TEP16). Rewrites a per-job
#     copy of the selected YAML with enable_attention_dp: false. Every rank
#     then serves the same global batch instead of its own, so in the default
#     mode max_batch_size is raised 32 -> 128 to recover eval concurrency
#     (the spec-dec modes keep their own batch limits). Note: with
#     attention-DP off the MoE router gate defaults to its bf16 fast path,
#     which can flip borderline expert picks; export KIMI_K3_ROUTER_BF16=0
#     for apples-to-apples accuracy comparison against DEP16 references.
#
# --task mmmu evaluates the multimodal path (KimiK3ForConditionalGeneration
# with the vision tower; point --model at a K3 VL checkpoint). It passes
# --post_process_fn kimi_k3_mmmu so the MMMU letter answer is read from K3's
# <|open|>response<|sep|> channel, and keeps a 16384-token generation budget
# via --preserve_caller_max_tokens: lm-eval's MMMU default of 512 tokens
# truncates the chain-of-thought before the response channel opens, silently
# degrading the score. --max_seq_len is raised to 24576 (8192 input + 16384
# output) for this task; explicitly passed trtllm-eval CLI flags override the
# --config YAML's max_seq_len.
#
# MMMU downloads its images from the HF hub on first use, so compute nodes
# need network access — or pre-fetch the dataset into a shared cache and
# export HF_HOME before submitting (sbatch propagates the environment):
#
#   HF_HOME=/shared/hf_home python3 -c "from huggingface_hub import \
#       snapshot_download; snapshot_download('MMMU/MMMU', repo_type='dataset')"
#   export HF_HOME=/shared/hf_home
#
# Pass --sa to evaluate with suffix-automaton speculative decoding
# (selects eval_extra_llm_options_sa.yaml and the SA-required
# max_batch_size of 8).
#
# Pass --reuse to additionally enable KV-cache block reuse
# (eval_extra_llm_options_reuse.yaml); chunked prefill is enabled by
# default.
#
# Pass --dflash /path/to/drafter to evaluate with DFlash speculative
# decoding (selects eval_extra_llm_options_dflash.yaml with the drafter
# path substituted in). With a synthetic drafter
# (make_synthetic_dflash_drafter.py) this is a wiring smoke only.
#
# --sa, --reuse, and --dflash are mutually exclusive.
#
# REPO defaults to the submit directory (override by exporting REPO), and
# must contain the built TensorRT-LLM checkout. Provide scheduler-specific
# options on the sbatch command line as needed.
#
#SBATCH --job-name=kimi-k3-eval
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=kimi-k3-eval-%j.log

set -euo pipefail

# Per-job runtime config copies (the --dflash and --parallel tep rewrites) are
# removed when this batch script exits. Every branch that writes one appends
# it here; a single trap covers them all because a second bare `trap ... EXIT`
# would *replace* the first, not add to it.
CLEANUP_FILES=()
trap '[ ${#CLEANUP_FILES[@]} -eq 0 ] || rm -f "${CLEANUP_FILES[@]}"' EXIT

usage() {
    echo "Usage: sbatch $0 --model PATH --image PATH [--task gsm8k|mmmu] [--parallel dep|tep] [--sa | --reuse | --dflash DRAFTER_PATH]"
}

MODEL=""
CONTAINER_IMAGE=""
TASK=gsm8k
PARALLEL=dep
MODE=default
SPEC_STATS_DEFAULT=0
DFLASH_DRAFTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sa|--reuse)
            [[ "$MODE" == default ]] || { echo "error: --sa, --reuse, and --dflash are mutually exclusive" >&2; usage >&2; exit 2; }
            MODE=${1#--}
            shift
            ;;
        --dflash)
            [[ $# -ge 2 ]] || { echo "error: --dflash requires a drafter checkpoint path" >&2; usage >&2; exit 2; }
            [[ "$MODE" == default ]] || { echo "error: --sa, --reuse, and --dflash are mutually exclusive" >&2; usage >&2; exit 2; }
            MODE=dflash
            DFLASH_DRAFTER=$2
            shift 2
            ;;
        --dflash=*)
            [[ "$MODE" == default ]] || { echo "error: --sa, --reuse, and --dflash are mutually exclusive" >&2; usage >&2; exit 2; }
            MODE=dflash
            DFLASH_DRAFTER=${1#*=}
            shift
            ;;
        --task)
            [[ $# -ge 2 ]] || { echo "error: --task requires a value" >&2; usage >&2; exit 2; }
            TASK=$2
            shift 2
            ;;
        --task=*)
            TASK=${1#*=}
            shift
            ;;
        --parallel)
            [[ $# -ge 2 ]] || { echo "error: --parallel requires a value" >&2; usage >&2; exit 2; }
            PARALLEL=$2
            shift 2
            ;;
        --parallel=*)
            PARALLEL=${1#*=}
            shift
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; usage >&2; exit 2; }
            MODEL=$2
            shift 2
            ;;
        --model=*)
            MODEL=${1#*=}
            shift
            ;;
        --image)
            [[ $# -ge 2 ]] || { echo "error: --image requires a value" >&2; usage >&2; exit 2; }
            CONTAINER_IMAGE=$2
            shift 2
            ;;
        --image=*)
            CONTAINER_IMAGE=${1#*=}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }
case "$TASK" in
    gsm8k|mmmu) ;;
    *) echo "error: unknown --task: $TASK (expected gsm8k or mmmu)" >&2; usage >&2; exit 2 ;;
esac
case "$PARALLEL" in
    dep|tep) ;;
    *) echo "error: unknown --parallel: $PARALLEL (expected dep or tep)" >&2; usage >&2; exit 2 ;;
esac

REPO=${REPO:-$SLURM_SUBMIT_DIR}
# Virtual environment created by build_wheel.py; export TRTLLM_VENV to point
# at .venv-<major>.<minor> when the container ships another Python version.
VENV=${TRTLLM_VENV:-$REPO/.venv-3.12}

# Per-mode LLM options.
# - sa: SA speculative decoding needs its own LLM options (eager, plain-EP;
#   see the yaml) and max_batch_size <= 8 (SpeculativeState buffers).
# - reuse: KV-cache block reuse; drops --disable_kv_cache_reuse so the
#   yaml's enable_block_reuse takes effect.
# - dflash: DFlash speculative decoding against the --dflash drafter; the
#   drafter path is substituted into a per-job copy of the DFlash yaml
#   (the checked-in file carries a placeholder). Smaller warmup/activation
#   footprint and KV budget: drafter weights, capture buffer, and DFlash
#   context-KV slots leave less headroom than the SA config assumes
#   (0.25/8192 OOM'd in warmup on GB300).
MAX_BATCH_SIZE=32
MAX_NUM_TOKENS=8192
KV_FRAC=0.25
KV_REUSE_ARG="--disable_kv_cache_reuse"
case "$MODE" in
    sa)
        EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options_sa.yaml
        MAX_BATCH_SIZE=8
        SPEC_STATS_DEFAULT=1
        ;;
    reuse)
        EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options_reuse.yaml
        KV_REUSE_ARG=""
        ;;
    dflash)
        [[ -e "$DFLASH_DRAFTER" ]] || { echo "error: drafter path does not exist: $DFLASH_DRAFTER" >&2; exit 2; }
        EVAL_CONFIG=$REPO/examples/kimi_k3/.eval_dflash_runtime.$SLURM_JOB_ID.yaml
        # Escape sed replacement metacharacters (\, & and the | delimiter) so
        # arbitrary drafter paths substitute literally.
        DFLASH_ESC=$(printf '%s' "$DFLASH_DRAFTER" | sed 's/[&|\\]/\\&/g')
        sed "s|speculative_model: .*|speculative_model: $DFLASH_ESC|" \
            "$REPO/examples/kimi_k3/eval_extra_llm_options_dflash.yaml" > "$EVAL_CONFIG"
        CLEANUP_FILES+=("$EVAL_CONFIG")
        MAX_BATCH_SIZE=8
        MAX_NUM_TOKENS=4096
        KV_FRAC=0.20
        SPEC_STATS_DEFAULT=1
        ;;
    *)
        EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options.yaml
        ;;
esac

# Per-task eval settings (see the header for the mmmu rationale). The task
# subcommand string is expanded into the container command below, so keep it
# on a single line. Per-sample MMMU outputs land next to the job log.
MAX_SEQ_LEN=8192
case "$TASK" in
    gsm8k)
        TASK_CMD="gsm8k"
        ;;
    mmmu)
        MAX_SEQ_LEN=24576
        TASK_CMD="mmmu --post_process_fn kimi_k3_mmmu --max_input_length 8192 --max_output_length 16384 --preserve_caller_max_tokens --output_path '$REPO/kimi-k3-mmmu-results-$SLURM_JOB_ID'"
        ;;
esac

# Per-parallel-layout adjustments (see the header). The TEP rewrite happens on
# a per-job copy so the checked-in YAMLs stay canonical for DEP16. In the
# default mode the batch bump also rewrites the YAML's own max_batch_size and
# the nested cuda_graph_config max_batch_size so graph coverage follows.
if [[ "$PARALLEL" == tep ]]; then
    # The per-job copy must live under $REPO: that is the only path guaranteed
    # to be container-mounted at the same location on every rank
    # ($SLURM_SUBMIT_DIR only coincides with it by default, and a node-local
    # mktemp file would be invisible to the other nodes). Removed by the
    # CLEANUP_FILES exit trap; also gitignored in case the job dies before
    # the trap fires.
    TEP_CONFIG=$REPO/examples/kimi_k3/.eval_tep_runtime.$SLURM_JOB_ID.yaml
    CLEANUP_FILES+=("$TEP_CONFIG")
    if [[ "$MODE" == default ]]; then
        MAX_BATCH_SIZE=128
        sed -e 's/^enable_attention_dp: true$/enable_attention_dp: false/' \
            -e "s/^max_batch_size: .*/max_batch_size: $MAX_BATCH_SIZE/" \
            -e "s/^  max_batch_size: .*/  max_batch_size: $MAX_BATCH_SIZE/" \
            "$EVAL_CONFIG" > "$TEP_CONFIG"
    else
        sed -e 's/^enable_attention_dp: true$/enable_attention_dp: false/' \
            "$EVAL_CONFIG" > "$TEP_CONFIG"
    fi
    EVAL_CONFIG=$TEP_CONFIG
fi

# Mount every input at its host path so the paths are valid inside too.
# EXTRA_MOUNTS (comma-separated src:dst:flags) lets git-worktree submits
# also mount the built main checkout that the worktree's artifact
# symlinks resolve into (submit with --export=ALL).
MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro"
[[ -n "$DFLASH_DRAFTER" ]] && MOUNTS="$MOUNTS,$DFLASH_DRAFTER:$DFLASH_DRAFTER:ro"
# Optional extra container mounts (srun --container-mounts syntax). Needed
# e.g. when $REPO is a git worktree whose build artifacts (.venv-3.12,
# bindings *.so) are symlinks into the main checkout.
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

        # Ignore \$HOME/.local site-packages, which --container-mount-home
        # leaks into the container: a user-site torch there shadows the
        # container's torch and breaks bindings symbol resolution
        # (libth_common.so: undefined symbol).
        export PYTHONNOUSERSITE=1

        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"

        # Node-local flashinfer cache: the default (\$HOME) is shared NFS and
        # races across ranks during cubin download (stale file handles).
        # These caches are cold on every run (~minutes of JIT per rank).
        export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}

        # Node-local HF remote-code modules cache: all ranks share \$HOME
        # (mounted at /root), and racing mkdir of
        # ~/.cache/huggingface/modules at startup escapes pathlib's
        # exist_ok on NFS (stale attribute cache fails the is_dir()
        # recheck after EEXIST) -> 'FileExistsError: /root/.cache' spam
        # from every rank. Only the trust_remote_code modules move
        # node-local; the dataset/hub cache stays shared for reuse.
        export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}

        # HF hub/datasets cache. Defaults to the shared \$HOME cache provided
        # by --container-mount-home; export HF_HOME before submitting to use
        # a pre-fetched location instead (recommended for the MMMU images —
        # see the header).
        export HF_HOME=\"${HF_HOME:-\$HOME/.cache/huggingface}\"

        # Run trtllm-eval from the in-place installation created in the
        # README Prerequisites section (build_wheel.py creates .venv-3.12).
        export PATH=\"$VENV/bin:\$PATH\"

        # Log a running partial score every N completed responses (0 = off).
        export TLLM_EVAL_PARTIAL_SCORES_EVERY=\"\${TLLM_EVAL_PARTIAL_SCORES_EVERY:-100}\"
        # Cap in-flight requests: may cost some throughput vs submit-all, but yields steady partial scores (early failure signal) instead of one burst at the end (0 = off).
        export TLLM_EVAL_MAX_IN_FLIGHT=\"\${TLLM_EVAL_MAX_IN_FLIGHT:-0}\"
        # Spec-dec acceptance summary (AL/AR) at eval end; default ON for the
        # sa/dflash modes, off otherwise (explicit env still wins either way).
        export TLLM_EVAL_SPEC_STATS=\"\${TLLM_EVAL_SPEC_STATS:-$SPEC_STATS_DEFAULT}\"

        # Import tensorrt_llm from $REPO, not from wherever the venv's
        # in-place install points (for git-worktree submits those differ;
        # without this the job silently tests the main checkout's code).
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
            '$VENV/bin/trtllm-eval' \
            --model \"$MODEL\" \
            --backend pytorch \
            --tp_size 16 \
            --max_batch_size $MAX_BATCH_SIZE \
            --max_seq_len $MAX_SEQ_LEN \
            --max_num_tokens $MAX_NUM_TOKENS \
            --kv_cache_free_gpu_memory_fraction $KV_FRAC \
            $KV_REUSE_ARG \
            --trust_remote_code \
            --config '$EVAL_CONFIG' \
            $TASK_CMD
    "
