#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run one staircase-v41 entry command and block until it finishes, printing its
# log and exiting with its exit code. This is the command to cache in
# test_command.md; it is self-contained and runnable from the login node.
#
#   bash staircase_v41_run.sh verify "<files>" <probe.py|-> <test-target|->
#   bash staircase_v41_run.sh test   unittest/_torch/staircase/gemm/test_staircase_mxfp8_mxfp8_gemm.py
#   bash staircase_v41_run.sh probe  staircase_v41_mxfp8_guard_probe.py
#   bash staircase_v41_run.sh lint   "<space separated files>"
#   bash staircase_v41_run.sh fmt    "<space separated files>"
#
# Prefer `verify` for a catalog entry: one container start instead of four, and
# it fixes the fmt-before-test ordering that receipt freshness depends on.
#
# Two paths, chosen per invocation:
#
#   fast      a persistent allocation is up  -> srun --jobid --overlap
#   fallback  it is not                      -> sbatch, then poll
#
# The fast path is an optimization and nothing depends on it. The allocation is
# re-checked on every call rather than assumed, because it can expire or be
# preempted at any time, and a stale one fails in a way that looks like the
# entry is broken instead of like the allocation is gone.

set -uo pipefail

REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM
WORK=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41
MODELS=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models
IMAGE=/lustre/fsw/portfolios/coreai/users/fredw/containers/trtllm-pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202608271702-17084.sqsh
JOBID_FILE=$WORK/alloc.jobid

MODE="${1:?usage: $0 <mode> [args...]}"
shift
mkdir -p "$WORK/logs"

usable_alloc() {
    local jid
    [ -r "$JOBID_FILE" ] || return 1
    jid=$(cat "$JOBID_FILE" 2>/dev/null)
    [ -n "$jid" ] || return 1
    squeue -h -j "$jid" -o '%T' 2>/dev/null | grep -qx RUNNING || return 1
    echo "$jid"
}

if jid=$(usable_alloc); then
    LOG=$WORK/logs/entry-alloc${jid}-$(date -u +%H%M%S)-$$.log
    echo "[run] fast path: srun into allocation $jid  ($LOG)"
    srun --jobid="$jid" --overlap --ntasks=1 \
         --container-image="$IMAGE" --container-mount-home \
         --container-mounts="$REPO:$REPO:rw,$MODELS:$MODELS:ro,$WORK:$WORK:rw" \
         bash "$REPO/staircase_v41_body.sh" "$MODE" "$@" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    # A vanished allocation surfaces as an srun launch failure, not as a real
    # verdict. Retry once through the fallback so the caller sees the entry's
    # result rather than a scheduling artifact.
    if [ "$rc" -ne 0 ] && grep -qiE 'Invalid job id|job .* has expired|Unable to allocate|step creation' "$LOG"; then
        echo "[run] allocation $jid is gone mid-command; falling back to sbatch"
        rm -f "$JOBID_FILE"
    else
        echo "[run] exit $rc (fast path)"
        exit $rc
    fi
fi

echo "[run] fallback: sbatch"
jobid=$(sbatch --parsable "$REPO/staircase_v41_entry.sbatch" "$MODE" "$@") || exit 2
echo "[run] submitted $jobid; waiting"
while squeue -h -j "$jobid" -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 10
done
LOG=$WORK/logs/entry-$jobid.log
[ -r "$LOG" ] && cat "$LOG"
rc=$(sacct -j "$jobid" -X -o ExitCode --noheader -P 2>/dev/null | head -1 | cut -d: -f1)
rc=${rc:-1}
echo "[run] exit $rc (sbatch $jobid)"
exit "$rc"
