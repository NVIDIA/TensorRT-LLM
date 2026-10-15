#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Inspect or control a compute session's spool:
#
#   spoolctl.sh [--spool DIR] status [--json]   who runs what, what waits and why
#   spoolctl.sh [--spool DIR] cancel <name>     stop a job (the only sanctioned kill)
#   spoolctl.sh [--spool DIR] wait <name>       block on a launched job's receipt
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${MV2FLOW_PYTHON:-python3}" "$HERE/../substrate/spoolctl.py" "$@"
