#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The compute half of `environment.kind: session_spool`: become the spool's scheduler.
#
# Run it as the container command of an allocation, so the scheduler IS the
# allocation's main process. A long-running process started inside a
# `slurm_exec_in_job` step does not survive the step's teardown.
#
#   bash spool_watcher.sh <spool-dir> [poll-seconds] [scheduler options...]
#
# Scheduler options (see `infra/substrate/scheduler.py watch --help`):
#   --gpus auto|N|0,1,2,3   devices to hand out (default: CUDA_VISIBLE_DEVICES, else nvidia-smi)
#   --session NAME          the session name in status files and successor requests
#   --default-wall 2h       wall limit, from start, for a GPU job that declares none
#   --requeue 1             times a job lost with its allocation runs again (0 = never)
#
# Jobs a previous start of the allocation (a Slurm requeue) or an ended one
# left in <spool>/running/ are queued again when this scheduler starts, or
# closed out with reason=node_lost when they may not run again.
#
# Jobs are scripts renamed into <spool>/pending/. Each may declare what it
# needs in a header line (`#spool: gpus=2 wall=2h`); without one, a `free-`
# name holds no device and any other name holds the whole node. The protocol
# and the safe-kill rule are documented in `infra/substrate/scheduler.py`.
#
# The scheduler is standard-library Python, so any `python3` in the container
# runs it; set MV2FLOW_SCHEDULER_PYTHON to choose one.

set -u

SPOOL="${1:?usage: spool_watcher.sh <spool-dir> [poll-seconds] [scheduler options...]}"
shift
POLL=3
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  POLL=$1
  shift
fi

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${MV2FLOW_SCHEDULER_PYTHON:-python3}" "$HERE/substrate/scheduler.py" watch "$SPOOL" --poll "$POLL" "$@"
