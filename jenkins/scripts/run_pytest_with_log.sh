#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# run_pytest_with_log.sh — wrap a single pytest invocation with log capture
# and timeout classification.
#
# Usage:
#   bash run_pytest_with_log.sh \
#       <cmd-script>       # bash script containing "cd ... && pytest ..."
#       <out-dir>          # directory for pytest_output.log
#       <timeout-data>     # path to timeout_data.jsonl (appended, dir auto-created)
#       <invocation-idx>   # integer index (unused at runtime, kept for callers)
#       <unfinished>       # path to unfinished_test.txt
#
# Exit code: always equals pytest's own exit code regardless of whether
# post-processing steps succeed.

set -o pipefail  # propagate pipeline failures; we override with PIPESTATUS below

if [ "$#" -ne 5 ]; then
    echo "ERROR: run_pytest_with_log.sh requires exactly 5 arguments." >&2
    echo "Usage: bash run_pytest_with_log.sh <cmd-script> <out-dir> <timeout-data> <invocation-idx> <unfinished>" >&2
    exit 1
fi

CMD_SCRIPT="$1"
OUT_DIR="$2"
TIMEOUT_DATA="$3"
# $4 (invocation-idx) is accepted but not used inside this script
UNFINISHED="$5"

LOG="${OUT_DIR}/pytest_output.log"
CLASSIFY="$(dirname "$0")/classify_timeout.py"

# Ensure the output directory exists before writing the log.
mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Run pytest, capturing stdout+stderr to LOG while still streaming to the
# Jenkins console.  tee's exit code is discarded; we read pytest's via
# PIPESTATUS immediately after the pipeline completes.
# ---------------------------------------------------------------------------
bash "${CMD_SCRIPT}" 2>&1 | tee "${LOG}"
PYTEST_RC=${PIPESTATUS[0]}

# ---------------------------------------------------------------------------
# Post-processing — all steps are best-effort.  A failure here must not
# change PYTEST_RC or prevent the caller from seeing pytest's exit code.
# ---------------------------------------------------------------------------
python3 "${CLASSIFY}" \
    --log        "${LOG}" \
    --out        "${TIMEOUT_DATA}" \
    --unfinished "${UNFINISHED}"
CLASSIFY_RC=$?
if [ "${CLASSIFY_RC}" -ne 0 ]; then
    echo "WARNING: run_pytest_with_log.sh: classify_timeout.py exited with code" \
         "${CLASSIFY_RC}; timed-out tests in this invocation may be reported as" \
         "'unknown' instead of 'pytest_timeout'. Log: ${LOG}" >&2
fi

rm -f "${LOG}" || true

# Always exit with pytest's original exit code.
exit "${PYTEST_RC}"
