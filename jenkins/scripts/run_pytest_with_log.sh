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
#       <log> <timeout-data> <unfinished> -- <command> [args...]
#
# Exit code: always equals pytest's own exit code regardless of whether
# post-processing steps succeed.

set -o pipefail  # propagate pipeline failures; we override with PIPESTATUS below

if [ "$#" -lt 5 ] || [ "$4" != "--" ]; then
    echo "Usage: bash run_pytest_with_log.sh <log> <timeout-data> <unfinished> -- <command> [args...]" >&2
    exit 1
fi

LOG="$1"
TIMEOUT_DATA="$2"
UNFINISHED="$3"
shift 4
CLASSIFY="$(dirname "$0")/classify_timeout.py"

mkdir -p "$(dirname "${LOG}")"

# The normal post-processing path removes this log explicitly.  Keep an EXIT
# trap as well so an early shell exit does not leave a potentially large log
# in the workspace for later artifact collection.
cleanup_log() {
    rm -f "${LOG}" || true
}

cleanup_and_exit() {
    cleanup_log
    trap - EXIT
    exit "$1"
}

trap cleanup_log EXIT
trap 'cleanup_and_exit 129' HUP
trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM

# Run pytest, capturing stdout+stderr to LOG while still streaming to the
# Jenkins console.  tee's exit code is discarded; we read pytest's via
# PIPESTATUS immediately after the pipeline completes.
TRTLLM_TIMEOUT_DATA_FILE="${TIMEOUT_DATA}" "$@" 2>&1 | tee "${LOG}"
PYTEST_RC=${PIPESTATUS[0]}

# Post-processing — all steps are best-effort.  A failure here must not
# change PYTEST_RC or prevent the caller from seeing pytest's exit code.
python3 "${CLASSIFY}" \
    --log        "${LOG}" \
    --out        "${TIMEOUT_DATA}" \
    --unfinished "${UNFINISHED}"
CLASSIFY_RC=$?
if [ "${CLASSIFY_RC}" -ne 0 ]; then
    echo "WARNING: run_pytest_with_log.sh: classify_timeout.py exited with code" \
         "${CLASSIFY_RC}; timed-out tests in this invocation may be reported as" \
         "'terminated_unexpectedly' instead of 'pytest_timeout'. Log: ${LOG}" >&2
fi

cleanup_log

# Always exit with pytest's original exit code.
exit "${PYTEST_RC}"
