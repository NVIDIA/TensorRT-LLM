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
# Scheduler options (see `substrate/scheduler.py watch --help`):
#   --gpus auto|N|0,1,2,3   devices to hand out (default: CUDA_VISIBLE_DEVICES, else nvidia-smi)
#   --session NAME          the session name in status files and successor requests
#   --default-wall 2h       wall limit, from start, for a GPU job that declares none
#
# Jobs are scripts renamed into <spool>/pending/. Each may declare what it
# needs in a header line (`#spool: gpus=2 wall=2h`); without one, a `free-`
# name holds no device and any other name holds the whole node. The protocol
# and the safe-kill rule are documented in `substrate/scheduler.py`.
#
# The scheduler is standard-library Python, so any `python3` in the container
# runs it; set MV2FLOW_SCHEDULER_PYTHON to choose one. MV2FLOW_SPOOL_WATCHER=legacy
# runs the previous flock-protocol watcher instead (scripts/spool_watcher_legacy.sh).

set -u

SPOOL="${1:?usage: spool_watcher.sh <spool-dir> [poll-seconds] [scheduler options...]}"
shift
POLL=3
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  POLL=$1
  shift
fi

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ "${MV2FLOW_SPOOL_WATCHER:-scheduler}" = legacy ]; then
  # Escape hatch: the flock-protocol watcher this scheduler replaced.
  exec bash "$HERE/scripts/spool_watcher_legacy.sh" "$SPOOL" "$POLL"
fi
exec "${MV2FLOW_SCHEDULER_PYTHON:-python3}" "$HERE/substrate/scheduler.py" watch "$SPOOL" --poll "$POLL" "$@"
