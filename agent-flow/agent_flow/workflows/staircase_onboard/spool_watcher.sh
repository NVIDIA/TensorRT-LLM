#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The compute half of `environment.kind: session_spool`.
#
# WHY IT EXISTS, IN ONE PARAGRAPH. On a cluster whose Slurm CLI cannot reach
# the controller, jobs go through an MCP broker; the broker's blocking calls
# are aborted after a few minutes of silence and every useful job is longer
# than that, while `slurm_exec_in_job` runs work in a job *step* whose cgroup
# Slurm tears down when the foreground command exits -- so a `nohup`'d child
# does not survive it (measured on this cluster: the log stopped at the first
# banner and never grew). The way out is for the long-running process to be
# the job's OWN main process. So the job runs this loop, and work arrives as
# files on a shared filesystem. A plain Python process with no MCP client --
# the orchestrator -- can then drop a script in `pending/` and poll for the
# matching `.rc`, which hands gate execution back to where the design wanted
# it. It also imports the container ONCE for the whole session instead of
# once per command.
#
# PROTOCOL, which is `agent_flow.runtime.commands.spool`'s and not this
# script's:
#
#   <spool>/pending/<name>.sh   appears (renamed into place, never partial)
#   <spool>/logs/<name>.log     stdout+stderr, grows while running
#   <spool>/done/<name>.rc      the exit code, renamed into place LAST
#
# `.rc` last is the whole ordering contract: a reader that sees it may trust
# the log beside it is complete.
#
# CONCURRENCY, AND THE ONE RULE THAT MATTERS. Work is dispatched concurrently,
# because the orchestrator itself may be one of the jobs -- on a host where a
# login-node background process does not survive, the session is the only
# place with a lifetime measured in hours, and a serial watcher would then
# deadlock: the orchestrator would hold the only slot while blocking on a gate
# that can never start.
#
# But concurrent *GPU* work on one allocation is a different mistake. A gate
# at world size 4 owns all four devices, and two of those at once fail with
# "Executor creation failed due to insufficient GPU memory" on a 284 GB card
# -- which reads as a target defect and is not. So:
#
#   a script whose name begins with `free-` runs immediately and holds nothing
#   every other script serializes on an exclusive lock for its whole run
#
# The orchestrator is submitted as `free-...`. Gates are named `gate-<millis>`
# by the executor and therefore serialize, which is what they want.
#
# Usage, as the container command of a 1-node / N-GPU allocation:
#
#   bash /path/to/spool_watcher.sh /path/to/spool

set -u

SPOOL="${1:?usage: spool_watcher.sh <spool-dir> [poll-seconds]}"
POLL="${2:-3}"
LOCK="$SPOOL/.gpu.lock"

mkdir -p "$SPOOL"/{pending,running,done,logs}
: > "$LOCK"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

{
  echo "host=$(hostname)"
  echo "pid=$$"
  echo "started=$(stamp)"
  echo "spool=$SPOOL"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv 2>&1
} > "$SPOOL/watcher.status"

run_one() {
  local script=$1 name=$2
  {
    echo "=== start $(stamp) on $(hostname) ==="
    echo
  } > "$SPOOL/logs/$name.log"

  if [[ "$name" == free-* ]]; then
    bash "$script" >> "$SPOOL/logs/$name.log" 2>&1
  else
    # 9 is an arbitrary free fd; `flock` holds it for the subshell's lifetime
    # and releases it however the script exits, including a kill.
    ( flock 9; bash "$script" ) 9>"$LOCK" >> "$SPOOL/logs/$name.log" 2>&1
  fi
  local rc=$?

  echo "=== end rc=$rc $(stamp) ===" >> "$SPOOL/logs/$name.log"
  mv "$SPOOL/running/$name.sh" "$SPOOL/done/$name.sh" 2>/dev/null || true
  echo "$rc" > "$SPOOL/done/$name.rc.tmp"
  mv "$SPOOL/done/$name.rc.tmp" "$SPOOL/done/$name.rc"
}

shopt -s nullglob
echo "watcher: ready at $(stamp)" >> "$SPOOL/watcher.status"

while true; do
  for f in "$SPOOL"/pending/*.sh; do
    name=$(basename "$f" .sh)
    # Claim by moving. If two watchers ever race the same spool, only one
    # `mv` wins and the work runs once.
    if mv "$f" "$SPOOL/running/$name.sh" 2>/dev/null; then
      run_one "$SPOOL/running/$name.sh" "$name" &
    fi
  done

  # Written every poll, so a stale mtime means a dead watcher rather than a
  # slow one. Two spellings because two readers exist: the ISO line a human
  # and the shell helpers read, and the file mtime the Python side can stat
  # without parsing anything.
  echo "alive=$(stamp)" > "$SPOOL/watcher.heartbeat.tmp"
  mv "$SPOOL/watcher.heartbeat.tmp" "$SPOOL/watcher.heartbeat"

  # A SEPARATE file from the heartbeat, deliberately: `scripts/spool_submit.sh`
  # pipes the heartbeat straight into `date -d` after stripping `alive=`, so
  # anything else on that line breaks every submitter. This one holds the job
  # id and nothing else.
  #
  # What reads it is `environment.shares_allocation_with_spool`, which asks
  # whether the orchestrator is living inside the very allocation it dispatches
  # work to. If it is, the two die together -- three times in one run on the
  # predecessor project, and twice more here: once when a disagg probe's worker
  # teardown took the step down, and once when an agent's `pkill -9` caught
  # this watcher, which IS the allocation's main process. Each time the
  # orchestrator went with it, holding GPUs it never used.
  #
  # `unknown` outside Slurm rather than an empty file, because the reader
  # treats unknown as no evidence and an empty file as a partial write.
  echo "${SLURM_JOB_ID:-unknown}" > "$SPOOL/watcher.jobid.tmp"
  mv "$SPOOL/watcher.jobid.tmp" "$SPOOL/watcher.jobid"

  # `STOP` is the clean shutdown, checked after the drain so anything already
  # claimed finishes rather than being orphaned mid-gate.
  if [ -f "$SPOOL/STOP" ]; then
    echo "watcher: STOP seen at $(stamp), waiting for running work" >> "$SPOOL/watcher.status"
    wait
    echo "watcher: exiting at $(stamp)" >> "$SPOOL/watcher.status"
    rm -f "$SPOOL/watcher.heartbeat" "$SPOOL/watcher.jobid"
    exit 0
  fi

  sleep "$POLL"
done
