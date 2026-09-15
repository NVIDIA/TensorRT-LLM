#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run one Goal 1.1 reference-ladder command and block until it finishes,
# printing its log and exiting with its exit code. Self-contained and runnable
# from the login node, so it is the form to cache in test_command.md.
#
#   bash staircase_v41_refladder.sh assets
#   bash staircase_v41_refladder.sh hash    # the 487 GiB shard payload check
#   bash staircase_v41_refladder.sh fmt
#   bash staircase_v41_refladder.sh lint
#   bash staircase_v41_refladder.sh small [prompt_len] [tag]
#   bash staircase_v41_refladder.sh full  [num_prompts] [tag]
#
# Same two paths as staircase_v41_run.sh, and for the same measured reason
# (~90s of every job here is queue wait plus pyxis unpacking the .sqsh):
#
#   fast      a persistent allocation is up  -> srun --jobid --overlap
#   fallback  it is not                      -> sbatch, then poll
#
# The allocation is re-checked on every call rather than assumed, because it
# expires and can be preempted, and a stale one fails in a way that reads like
# the driver is broken instead of like the allocation is gone. Nothing about a
# result depends on which path produced it.
#
# `full` is a four-rank torchrun inside ONE task, which is what
# inference/generate.py expects -- not trtllm-llmapi-launch. This leg never
# imports tensorrt_llm, so the LLM API's MPI rules do not apply to it.

set -uo pipefail

REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM
WORK=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41
MODELS=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models
IMAGE=/lustre/fsw/portfolios/coreai/users/fredw/containers/trtllm-pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202608271702-17084.sqsh
JOBID_FILE=$WORK/alloc.jobid

# No braces in this message: a literal } would close the parameter expansion
# early and the leftover text becomes part of MODE.
MODE="${1:?usage: $0 assets|hash|lint|fmt|small|full [arg] [tag]}"
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
    LOG=$WORK/logs/refladder-alloc${jid}-$(date -u +%H%M%S)-$$.log
    echo "[refladder] fast path: srun into allocation $jid  ($LOG)"
    srun --jobid="$jid" --overlap --ntasks=1 \
         --container-image="$IMAGE" --container-mount-home \
         --container-mounts="$REPO:$REPO:rw,$MODELS:$MODELS:ro,$WORK:$WORK:rw" \
         bash "$REPO/staircase_v41_refladder_body.sh" "$MODE" "$@" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] && grep -qiE 'Invalid job id|job .* has expired|Unable to allocate|step creation' "$LOG"; then
        echo "[refladder] allocation $jid is gone mid-command; falling back to sbatch"
        rm -f "$JOBID_FILE"
    else
        echo "[refladder] exit $rc (fast path)"
        exit $rc
    fi
fi

echo "[refladder] fallback: sbatch"
jobid=$(sbatch --parsable "$REPO/staircase_v41_refladder.sbatch" "$MODE" "$@") || exit 2
echo "[refladder] submitted $jobid; waiting"
while squeue -h -j "$jobid" -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 10
done
LOG=$WORK/logs/refladder-$jobid.log
[ -r "$LOG" ] && cat "$LOG"
rc=$(sacct -j "$jobid" -X -o ExitCode --noheader -P 2>/dev/null | head -1 | cut -d: -f1)
rc=${rc:-1}
echo "[refladder] exit $rc (sbatch $jobid)"
exit "$rc"
