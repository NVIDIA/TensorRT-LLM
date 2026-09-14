#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# A persistent 4-GPU allocation for staircase-v41 entry commands.
#
#   bash staircase_v41_alloc.sh up      # claim one, write $WORK/alloc.jobid
#   bash staircase_v41_alloc.sh status  # is it usable right now?
#   bash staircase_v41_alloc.sh down    # release it
#
# Why: measured on this cluster, ~90s of every entry job is fixed overhead --
# queue wait (0.3-1.1 min) plus pyxis unpacking the .sqsh on a fresh node. A
# held allocation removes the queue wait and lets the second and later
# containers reuse the node's squashfs cache. It does NOT remove the
# `import tensorrt_llm` cost: every `srun` step is still a fresh process, which
# is deliberate -- see the keepalive note below.
#
# Nothing depends on this being up. staircase_v41_run.sh falls back to sbatch
# whenever the allocation is missing, expired, or preempted, so the worst case
# is the old behaviour.

set -uo pipefail

REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM
WORK=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41
IMAGE=/lustre/fsw/portfolios/coreai/users/fredw/containers/trtllm-pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202608271702-17084.sqsh
JOBID_FILE=$WORK/alloc.jobid
# batch caps at 4:00:00 (`sinfo -p batch -o %l`). Asking for more is not
# clamped -- salloc rejects the whole request with "Requested time limit is
# invalid", which reads like a syntax error rather than a policy limit.
HOURS=${STAIRCASE_ALLOC_HOURS:-4}
if [ "$HOURS" -gt 4 ]; then
    echo "[alloc] batch allows at most 4h; clamping $HOURS -> 4"
    HOURS=4
fi

mkdir -p "$WORK/logs"

alloc_state() {
    local jid
    [ -r "$JOBID_FILE" ] || return 1
    jid=$(cat "$JOBID_FILE" 2>/dev/null)
    [ -n "$jid" ] || return 1
    squeue -h -j "$jid" -o '%T' 2>/dev/null | grep -qx RUNNING || return 1
    echo "$jid"
}

case "${1:-status}" in
up)
    if jid=$(alloc_state); then
        echo "[alloc] already up: job $jid"
        exit 0
    fi
    rm -f "$JOBID_FILE"
    LOG=$WORK/logs/alloc-$(date -u +%Y%m%dT%H%M%SZ).log
    echo "[alloc] claiming 4 GPUs for ${HOURS}h -> $LOG"

    # salloc holds the allocation for as long as its command runs, so the
    # command is the keepalive itself. Two things it has to do:
    #
    #   1. Keep the GPUs nominally busy. This cluster's watchdog kills jobs
    #      whose GPUs sit at 0% utilization for about ten minutes, and an
    #      allocation waiting between commands is exactly that.
    #   2. Stay off GPU 0. Entry commands run on GPU 0 (CUDA_VISIBLE_DEVICES=0
    #      in staircase_v41_body.sh). The autotuner picks GEMM tactics by
    #      timing them, and the catalog contracts pin cold-autotuner bits --
    #      a concurrent kernel on the same device could change which tactic
    #      wins and silently move what a receipt certifies. So the keepalive
    #      is pinned to GPUs 1-3 and the measurement device stays clean.
    nohup salloc -N1 --gres=gpu:4 --mem=0 \
        --partition=batch --account=coreai_comparch_trtllm \
        --time=${HOURS}:00:00 --job-name=staircase-v41-alloc --no-shell \
        > "$LOG" 2>&1 &

    for _ in $(seq 1 60); do
        jid=$(squeue -h -u "$USER" -n staircase-v41-alloc -o '%i %T' 2>/dev/null \
              | awk '$2=="RUNNING"{print $1; exit}')
        [ -n "${jid:-}" ] && break
        sleep 5
    done
    if [ -z "${jid:-}" ]; then
        echo "[alloc] FAILED to get a RUNNING allocation in 5 min; see $LOG"
        exit 1
    fi

    # RUNNING in squeue is NOT the same as ready to accept a step. Measured:
    # an srun issued 10s after the state flipped failed with
    #   srun: error: Unable to create step for job <id>: Invalid job id specified
    # which reads like a wrong job id rather than "not ready yet" -- and both
    # the keepalive and the first real command raced in and hit it. Probe with
    # a trivial step until one lands, and only then publish the id.
    ready=""
    for _ in $(seq 1 30); do
        if srun --jobid="$jid" --overlap --ntasks=1 true 2>/dev/null; then ready=1; break; fi
        sleep 5
    done
    if [ -z "$ready" ]; then
        echo "[alloc] job $jid never accepted a step; cancelling"
        scancel "$jid" 2>/dev/null
        exit 1
    fi

    srun --jobid="$jid" --overlap --ntasks=1 \
         --container-image="$IMAGE" --container-mount-home \
         --container-mounts="$REPO:$REPO:rw" \
         bash -c 'CUDA_VISIBLE_DEVICES=1,2,3 python3 - <<PY
import time, torch
xs = [torch.randn(512, 512, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
print(f"[keepalive] holding {len(xs)} device(s), off the measurement GPU", flush=True)
while True:
    for x in xs:
        _ = (x @ x).sum().item()
    time.sleep(20)
PY' >> "$LOG" 2>&1 &

    echo "$jid" > "$JOBID_FILE"
    echo "[alloc] up: job $jid, ${HOURS}h, keepalive on GPUs 1-3"
    ;;

status)
    if jid=$(alloc_state); then
        echo "usable $jid"
        squeue -h -j "$jid" -o '  %i %T %M / %l  %N'
        exit 0
    fi
    echo "unusable (no allocation; commands fall back to sbatch)"
    exit 1
    ;;

down)
    if [ -r "$JOBID_FILE" ]; then
        jid=$(cat "$JOBID_FILE")
        scancel "$jid" 2>/dev/null && echo "[alloc] cancelled $jid"
    fi
    rm -f "$JOBID_FILE"
    ;;

*)
    echo "usage: $0 {up|status|down}" >&2
    exit 2
    ;;
esac
