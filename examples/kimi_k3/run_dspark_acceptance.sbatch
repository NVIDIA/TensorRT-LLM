#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 DSpark acceptance / speedup A/B on 16 NVIDIA Blackwell GPUs.
#
# One submission runs BOTH legs of the weights-day A/B back to back in the
# same allocation (identical nodes, so the TPOT comparison is clean):
#   (a) spec-off TPOT reference (recorder off),
#   (b) DSpark-on measurement (AL, per-position AR, histogram, and — with a
#       confidence head in the drafter — calibration data).
#
#   sbatch examples/kimi_k3/run_dspark_acceptance.sbatch \
#       --model /path/to/kimi-k3-checkpoint \
#       --drafter /path/to/dspark-drafter \
#       --image /path/to/tensorrt-llm-container.sqsh \
#       --outdir /path/to/results
#
# Optional: --num-prompts N (default 64), --max-tokens N (default 256),
# --confidence-threshold T --confidence-policy P (leave unset for
# calibration runs; see measure_dspark_acceptance.py), --skip-baseline
# (rerun only the DSpark leg, e.g. for a threshold sweep).
#
# Worktree submits: export REPO=<worktree>, EXTRA_MOUNTS="$MAIN:$MAIN:rw",
# and submit with --export=ALL (mirrors run_eval_kimi_k3.sbatch).
#
#SBATCH --job-name=kimi-k3-dspark-accept
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=kimi-k3-dspark-accept-%j.log

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --model PATH --drafter PATH --image PATH [--outdir DIR]" \
         "[--num-prompts N] [--max-tokens N] [--confidence-threshold T]" \
         "[--confidence-policy first_below|cumulative] [--skip-baseline]"
}

MODEL=""
DRAFTER=""
CONTAINER_IMAGE=""
OUTDIR=""
NUM_PROMPTS=64
MAX_TOKENS=256
CONF_THRESHOLD=""
CONF_POLICY=first_below
SKIP_BASELINE=0
# Abort with a usage error when a value-taking option has no value ($1 is the
# option name, $2 the remaining positional-argument count of the caller);
# without this, `set -u` aborts on the unbound $2 instead.
require_value() {
    [[ $2 -ge 2 ]] || { echo "error: $1 requires a value" >&2; usage >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        require_value --model $#; MODEL=$2; shift 2 ;;
        --model=*)      MODEL=${1#*=}; shift ;;
        --drafter)      require_value --drafter $#; DRAFTER=$2; shift 2 ;;
        --drafter=*)    DRAFTER=${1#*=}; shift ;;
        --image)        require_value --image $#; CONTAINER_IMAGE=$2; shift 2 ;;
        --image=*)      CONTAINER_IMAGE=${1#*=}; shift ;;
        --outdir)       require_value --outdir $#; OUTDIR=$2; shift 2 ;;
        --outdir=*)     OUTDIR=${1#*=}; shift ;;
        --num-prompts)  require_value --num-prompts $#; NUM_PROMPTS=$2; shift 2 ;;
        --max-tokens)   require_value --max-tokens $#; MAX_TOKENS=$2; shift 2 ;;
        --confidence-threshold) require_value --confidence-threshold $#; CONF_THRESHOLD=$2; shift 2 ;;
        --confidence-policy)    require_value --confidence-policy $#; CONF_POLICY=$2; shift 2 ;;
        --skip-baseline) SKIP_BASELINE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$MODEL" && -n "$DRAFTER" && -n "$CONTAINER_IMAGE" ]] || { usage >&2; exit 2; }
[[ -e "$MODEL" ]]   || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$DRAFTER" ]] || { echo "error: drafter path does not exist: $DRAFTER" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }

REPO=${REPO:-$SLURM_SUBMIT_DIR}
# Virtual environment created by build_wheel.py; export TRTLLM_VENV to point
# at .venv-<major>.<minor> when the container ships another Python version.
VENV=${TRTLLM_VENV:-$REPO/.venv-3.12}
OUTDIR=${OUTDIR:-$REPO/dspark-accept-$SLURM_JOB_ID}
mkdir -p "$OUTDIR"

CONF_ARGS=""
[[ -n "$CONF_THRESHOLD" ]] && \
    CONF_ARGS="--confidence-threshold $CONF_THRESHOLD --confidence-policy $CONF_POLICY"

# Mount every input at its host path so the paths are valid inside too.
# EXTRA_MOUNTS (comma-separated src:dst:flags) lets git-worktree submits
# also mount the built main checkout that the worktree's artifact
# symlinks resolve into (submit with --export=ALL).
MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$DRAFTER:$DRAFTER:ro,$OUTDIR:$OUTDIR:rw"
if [ -n "${EXTRA_MOUNTS:-}" ]; then
    MOUNTS+=",$EXTRA_MOUNTS"
fi

# User cache (GSM8K dataset under ~/.cache/huggingface):
# --container-mount-home mounts $HOME at /root, but when ~/.cache is a
# symlink onto a shared filesystem its target dead-ends inside the container
# (FileNotFoundError /root/.cache/huggingface). Mount the
# resolved cache root at its host path — the /root/.cache symlink chain
# then works too — and point HF_HOME at it explicitly. flashinfer/triton
# JIT caches stay node-local via the /tmp overrides below.
CACHE_HOST=$(readlink -f "$HOME/.cache" 2>/dev/null || echo "$HOME/.cache")
if [[ -d "$CACHE_HOST" && "$CACHE_HOST" != "$HOME/.cache" ]]; then
    MOUNTS+=",$CACHE_HOST:$CACHE_HOST:rw"
    # Also mount at the symlink's literal target so the in-container
    # /root/.cache -> <target> chain resolves (the target's parent dirs
    # do not otherwise exist in the container).
    CACHE_LINK_TARGET=$(readlink "$HOME/.cache")
    if [[ -n "$CACHE_LINK_TARGET" && "$CACHE_LINK_TARGET" != "$CACHE_HOST" ]]; then
        MOUNTS+=",$CACHE_HOST:$CACHE_LINK_TARGET:rw"
    fi
fi
HF_CACHE_HOST="$CACHE_HOST/huggingface"

run_leg() {
    local leg=$1; shift
    # %q-quote every forwarded argument: they are spliced into the nested
    # bash -c string, where a bare $* would re-split values containing
    # whitespace (e.g. a drafter path with spaces).
    local escaped_args
    printf -v escaped_args ' %q' "$@"
    srun --mpi=pmix \
        --container-image="$CONTAINER_IMAGE" \
        --container-mount-home \
        --container-mounts="$MOUNTS" \
        bash -c "
            set -x
            ulimit -n 65536
            # Node-local JIT caches (shared-NFS races; see run_eval_kimi_k3.sbatch)
            export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
            mkdir -p \"\$TRITON_CACHE_DIR\"
            export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}
            export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}
            export HF_HOME='$HF_CACHE_HOST'
            # Recorder env must be exported HERE (per rank), not inside the
            # python script: trtllm-llmapi-launch pre-spawns the MPI worker
            # ranks, so os.environ changes in the driver never reach the
            # workers where DFlashWorker lives (symptom: no accept-stats
            # files). Empty for the spec-off leg (recorder off = clean TPOT).
            ${LEG_ENV_EXPORT:-true}
            export PATH=\"$VENV/bin:\$PATH\"
            # Import tensorrt_llm from \$REPO, not from wherever the venv's
            # in-place install points (worktree submits differ).
            export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"
            exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
                '$REPO/examples/kimi_k3/measure_dspark_acceptance.py' \
                --model '$MODEL' --tp-size 16 \
                --num-prompts $NUM_PROMPTS --max-tokens $MAX_TOKENS \
                $escaped_args
        " 2>&1 | tee "$OUTDIR/$leg.log"
}

if [[ "$SKIP_BASELINE" -eq 0 ]]; then
    echo "=== leg (a): spec-off TPOT reference ==="
    LEG_ENV_EXPORT="" \
        run_leg spec_off --spec-off --output-json "$OUTDIR/results_spec_off.json"
fi

echo "=== leg (b): DSpark-on measurement ==="
LEG_ENV_EXPORT="export TLLM_DFLASH_ACCEPT_STATS_DIR='$OUTDIR/accept-stats'" \
    run_leg dspark --drafter "$DRAFTER" $CONF_ARGS \
    --stats-dir "$OUTDIR/accept-stats" \
    --output-json "$OUTDIR/results_dspark.json"

# Combined speedup summary (host-side python3, stdlib only)
python3 - "$OUTDIR" <<'EOF'
import json, os, sys
outdir = sys.argv[1]
def load(name):
    p = os.path.join(outdir, name)
    return json.load(open(p)) if os.path.exists(p) else None
ref, spec = load("results_spec_off.json"), load("results_dspark.json")
summary = {"spec_off": ref, "dspark": spec}
if ref and spec and ref.get("tpot_proxy_ms") and spec.get("tpot_proxy_ms"):
    summary["e2e_speedup"] = ref["tpot_proxy_ms"] / spec["tpot_proxy_ms"]
    print(f"E2E speedup (spec-off TPOT / dspark TPOT): {summary['e2e_speedup']:.3f}x")
json.dump(summary, open(os.path.join(outdir, "ab_summary.json"), "w"), indent=2)
print(f"A/B summary: {os.path.join(outdir, 'ab_summary.json')}")
EOF
