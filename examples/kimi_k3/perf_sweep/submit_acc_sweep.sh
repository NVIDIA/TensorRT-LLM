#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit the Kimi K3 GSM8K ACCURACY runs for the three perf-sweep serving
# recipes (see acc_sweep.sbatch; expect ~96.5 +/- 0.5 each):
#   tep16  4 nodes  2:00        tep8  2 nodes  2:00
#   dep16  4 nodes  2:00
#
# Partition/account come from the acc_sweep.sbatch placeholders — edit
# them there or export SBATCH_PARTITION / SBATCH_ACCOUNT before submitting.
# On Slurm clusters with block topology, export USE_SLURM_SEGMENTS=1 to
# also request one NVLink segment per node group (sbatch --segment).
#
# Usage: submit_acc_sweep.sh --model PATH --image PATH [--repo PATH] \
#            [--jobs "tep16 tep8 dep16"] [--dry-run]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SBATCH_SCRIPT=$SCRIPT_DIR/acc_sweep.sbatch

MODEL=""; IMAGE=""; REPO=""; JOBS="tep16 tep8 dep16"; DRY_RUN=0
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
    local mode=$1 nodes=$2 segment=$3
    local opts=(--job-name="kimi-k3-acc-$mode" --output="kimi-k3-acc-$mode-%j.log"
                --nodes="$nodes" --time="02:00:00")
    [[ -n "$segment" && "${USE_SLURM_SEGMENTS:-0}" == 1 ]] && opts+=(--segment="$segment")
    local args=(--mode "$mode" --model "$MODEL" --image "$IMAGE" "${REPO_ARGS[@]}")
    if (( DRY_RUN )); then
        echo "DRY RUN: sbatch ${opts[*]} $SBATCH_SCRIPT ${args[*]}"
    else
        # Checked assignment so a failed sbatch stops the sweep (set -e);
        # echo "$(sbatch ...)" would discard the failure.
        local result
        result=$(sbatch "${opts[@]}" "$SBATCH_SCRIPT" "${args[@]}")
        echo "$mode: $result"
    fi
}

for job in $JOBS; do
    case "$job" in
        tep16) submit tep16 4 "4" ;;
        tep8)  submit tep8  2 "2" ;;
        dep16) submit dep16 4 "4" ;;
        *) echo "error: unknown job: $job (tep16|tep8|dep16)" >&2; exit 2 ;;
    esac
done
