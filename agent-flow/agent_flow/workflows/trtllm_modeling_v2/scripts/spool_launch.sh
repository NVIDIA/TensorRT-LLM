#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Queue one command in a compute session WITHOUT waiting for it.
#
#   spool_launch.sh [options] <name> '<shell command>'
#
# Same options and protocol as spool_submit.sh. Use it for work whose
# lifetime is longer than the caller's, such as the orchestrator itself
# (queued under a `free-` name so it holds no device while it blocks on
# gates). `spoolctl.py wait <name>` blocks on it later.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${MV2FLOW_PYTHON:-python3}" "$HERE/../substrate/spoolctl.py" launch "$@"
