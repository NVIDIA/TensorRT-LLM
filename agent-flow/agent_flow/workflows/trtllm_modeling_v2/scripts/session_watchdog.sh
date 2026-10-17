#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Block until a compute session needs attention, or until a timeout:
#
#   session_watchdog.sh [--spool DIR]... [--timeout 3000] [--stale 180] [--ending 1800]
#
# Prints `STALE <spool> ...` (the scheduler stopped heart-beating),
# `ENDING <spool> ...` (the allocation ends soon and no successor is
# acknowledged) or `QUARANTINED <spool> ...` (devices held for processes the
# safe-kill rule forbids stopping), or `OK` on timeout, then one summary line
# per session. Spools default to $MV2FLOW_SESSION_SPOOLS (':'-separated).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${MV2FLOW_PYTHON:-python3}" "$HERE/../substrate/watchdog.py" "$@"
