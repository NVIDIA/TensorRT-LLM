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

# One pass of progress collection for a Slurm batch stage: when the remote results.xml has changed
# since the last pass that copied it, copy the job's results into the stage results directory, for
# progress_snapshot.sh to tar. Runs only the commands it is given and needs no Artifactory
# credential.
#
# Required env vars:
#   WORKSPACE            the stage's workspace; the commands run from it
#   STAGE_NAME           stage directory name (under WORKSPACE) and log prefix
#   SLURM_SSH_STAT_CMD   shell command that prints the remote results.xml mtime
#                        (run on the compute node via srun --overlap to bypass
#                        the login node's NFS attribute cache)
#   SLURM_SCP_XML_CMD    shell command that SCPs remote results*.xml locally
#   PROGRESS_MTIME_FILE  holds the mtime of the last results copied; a missing file counts as 0
#
# Optional:
#   SLURM_SSH_REFRESH_CACHE_CMD  shell command run on the login node to prime its NFS
#                                cache after the stat detects a change, before SCP
#   SLURM_SCP_UNFINISHED_CMD     shell command that SCPs remote unfinished_test.txt locally
#   SLURM_SSH_LIST_PERF_CMD      shell command that prints remote perf folder paths (one per line)
#   SLURM_SCP_PERF_TEMPLATE      SCP command with PERF_FOLDER_PLACEHOLDER substituted per folder
#
# Each copy is tried up to 3 times, 10 seconds apart. A failed copy of unfinished_test.txt or of a
# perf folder is only logged.
#
# Exit codes:
#   0  new results were copied, and their mtime is in PROGRESS_MTIME_FILE
#   3  the remote results.xml has not changed since the last copy
#   1  the results*.xml copy failed after 3 attempts (the mtime is not recorded, so the next pass
#      tries again), or a required env var is not set

set +e

for _var in WORKSPACE STAGE_NAME SLURM_SSH_STAT_CMD SLURM_SCP_XML_CMD PROGRESS_MTIME_FILE; do
    if [ -z "${!_var}" ]; then
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${_var} is not set"
        exit 1
    fi
done
# The copy commands write to paths relative to the workspace.
cd "$WORKSPACE" || exit 1

# Runs a copy command up to 3 times, 10 seconds apart, and returns 0 once it succeeds. The command
# gets no stdin, so it cannot consume the perf folder list that the loop below reads.
copy_with_retries() {
    local what=$1 cmd=$2 attempt
    for attempt in 1 2 3; do
        eval "$cmd" < /dev/null && return 0
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp ${what} failed (attempt ${attempt}/3)"
        [ "$attempt" -lt 3 ] && sleep 10
    done
    return 1
}

# Capture stderr too (srun/ssh errors) so a stuck mtime=0 can be diagnosed instead of silently
# swallowed; grep still extracts the integer mtime so banners/errors mixed into the output cannot
# pollute the parsed value.
stat_raw=$(eval "$SLURM_SSH_STAT_CMD" 2>&1)
mtime=$(printf '%s\n' "$stat_raw" | grep -oE '^[0-9]+$' | head -1)
if [ -z "$mtime" ]; then
    mtime=0
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: stat cmd returned no numeric mtime, raw output: ${stat_raw}"
fi
last=$(cat "$PROGRESS_MTIME_FILE" 2>/dev/null)
case "$last" in
    "" | *[!0-9]*) last=0 ;;
esac
echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: mtime=${mtime} last=${last}"
[ "$mtime" -le "$last" ] && exit 3

# Prime the login node's NFS cache before SCP: the stat above ran on the compute node (via srun
# --overlap), so the login node may still have a stale view of the directory. Running ls on the
# login node forces a fresh READDIR/GETATTR RPC to the NFS server before the copies read it.
if [ -n "$SLURM_SSH_REFRESH_CACHE_CMD" ]; then
    eval "$SLURM_SSH_REFRESH_CACHE_CMD" 2>/dev/null || true
fi
mkdir -p "${WORKSPACE}/${STAGE_NAME}"
if ! copy_with_retries xml "$SLURM_SCP_XML_CMD"; then
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp xml failed after 3 attempts"
    exit 1
fi
# Fetch unfinished_test.txt for timeout XML generation
if [ -n "$SLURM_SCP_UNFINISHED_CMD" ] \
        && ! copy_with_retries unfinished "$SLURM_SCP_UNFINISHED_CMD"; then
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp unfinished not available, skipping"
fi
# Fetch perf result folders (aggr*/disagg*) one by one
if [ -n "$SLURM_SSH_LIST_PERF_CMD" ] && [ -n "$SLURM_SCP_PERF_TEMPLATE" ]; then
    while IFS= read -r folder; do
        [ -z "$folder" ] && continue
        perf_cmd=${SLURM_SCP_PERF_TEMPLATE//PERF_FOLDER_PLACEHOLDER/$folder}
        if ! copy_with_retries "perf $folder" "$perf_cmd"; then
            echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp perf $folder failed after 3 attempts"
        fi
    done < <(eval "$SLURM_SSH_LIST_PERF_CMD" 2>/dev/null)
fi

if ! echo "$mtime" > "$PROGRESS_MTIME_FILE"; then
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: could not record mtime=${mtime} in ${PROGRESS_MTIME_FILE}"
fi
echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: copied results with mtime=${mtime}"
