#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit the standard Kimi K3 17-point serving sweep (tuned recipes):
#   tep16     c 1 2 4 8 16         4 nodes  1:30
#   tep8      c 1 2 4 8 16         2 nodes  1:30
#   dep16-lo  c 16 32 64 128 256   4 nodes  2:30
#   dep16-hi  c 512 1024           4 nodes  2:30
#
# All four jobs are submitted TOGETHER on purpose: weight loads overlapping
# another job's measurement window depress DEP16 c>=128 points by ~30%
# (loads overlapping loads are harmless). Run from a fresh results folder.
#
# Partition/account come from the perf_sweep.sbatch placeholders — edit
# them there or export SBATCH_PARTITION / SBATCH_ACCOUNT before submitting.
# On Slurm clusters with block topology, export USE_SLURM_SEGMENTS=1 to
# also request one NVLink segment per node group (sbatch --segment).
#
# Usage: submit_perf_sweep.sh --model PATH --image PATH [--repo PATH] \
#            [--jobs "tep16 tep8 dep16-lo dep16-hi"] [--dry-run]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SBATCH_SCRIPT=$SCRIPT_DIR/perf_sweep.sbatch

MODEL=""; IMAGE=""; REPO=""; JOBS="tep16 tep8 dep16-lo dep16-hi"; DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --repo) REPO=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODEL" && -n "$IMAGE" ]] || {
    echo "Usage: $0 --model PATH --image PATH [--repo PATH] [--jobs ...] [--dry-run]" >&2; exit 2; }

REPO_ARGS=()
[[ -n "$REPO" ]] && REPO_ARGS=(--repo "$REPO")

submit() {
    local name=$1 mode=$2 tag=$3 nodes=$4 walltime=$5 segment=$6 concurrencies=$7
    local opts=(--job-name="kimi-k3-sweep-$name" --output="kimi-k3-sweep-$name-%j.log"
                --nodes="$nodes" --time="$walltime")
    [[ -n "$segment" && "${USE_SLURM_SEGMENTS:-0}" == 1 ]] && opts+=(--segment="$segment")
    local args=(--mode "$mode" --model "$MODEL" --image "$IMAGE"
                --concurrencies "$concurrencies" "${REPO_ARGS[@]}")
    [[ -n "$tag" ]] && args+=(--tag "$tag")
    if (( DRY_RUN )); then
        echo "DRY RUN: sbatch ${opts[*]} $SBATCH_SCRIPT ${args[*]}"
    else
        # Checked assignment so a failed sbatch stops the sweep (set -e);
        # echo "$(sbatch ...)" would discard the failure.
        local result
        result=$(sbatch "${opts[@]}" "$SBATCH_SCRIPT" "${args[@]}")
        echo "$name: $result"
    fi
}

for job in $JOBS; do
    case "$job" in
        tep16)    submit tep16    tep16 ""   4 "01:30:00" "4" "1 2 4 8 16" ;;
        tep8)     submit tep8     tep8  ""   2 "01:30:00" "2" "1 2 4 8 16" ;;
        dep16-lo) submit dep16-lo dep16 "lo" 4 "02:30:00" "4" "16 32 64 128 256" ;;
        dep16-hi) submit dep16-hi dep16 "hi" 4 "02:30:00" "4" "512 1024" ;;
        *) echo "error: unknown job: $job" >&2; exit 2 ;;
    esac
done
