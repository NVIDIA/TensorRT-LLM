#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run one command in a compute session and WAIT for it: prints the job's log
# and exits with its exit code.
#
#   spool_submit.sh [options] <name> '<shell command>'
#
#   --spool DIR      the session's spool (default $SPOOL, then $STAIRCASE_SPOOL)
#   --gpus N|all     devices the job needs; `all` is the whole node, exclusively
#   --wall 2h        wall-clock limit counted from START (queueing is free); 0 = none
#   --priority P     higher starts first (default 0); whole-node jobs go before slot jobs
#   --mem-gb G       resident-memory limit for the job's processes
#   --owner PREFIX   requester prefix (default $STAIRCASE_JOB_PREFIX); names must start with it
#   --tree DIR       checkout to run against: the working directory AND the first PYTHONPATH
#                    entry, so every MPI rank imports it; the result is refused if any
#                    process imported tensorrt_llm from another checkout
#   --cwd DIR        working directory (default --tree, then $STAIRCASE_REPO, then the current one)
#
# Without --gpus the old convention holds: a `free-` name holds no device, any
# other name holds the whole node, and a command that calls `gpu_slot.sh N`
# holds N devices. Exit code 70 means the session is not running.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${STAIRCASE_PYTHON:-python3}" "$HERE/../substrate/spoolctl.py" submit "$@"
