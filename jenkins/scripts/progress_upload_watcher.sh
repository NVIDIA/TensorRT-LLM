#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Background progress-upload watcher. Polls for results changes and calls
# progress_upload_snapshot.sh on each update, until PROGRESS_DONE_FILE appears.
#
# Required env vars (WORKSPACE/ART_USER/ART_PASS injected by Jenkins/withCredentials;
# STAGE_NAME/PROGRESS_TAR/PROGRESS_URL exported by the caller sh block):
#   PROGRESS_DONE_FILE  sentinel file; watcher exits when it exists
#   PROGRESS_INTERVAL   poll interval in seconds
#   LABEL_PREFIX        log label prefix, e.g. "checkpoint", "sbatch checkpoint"
#
# SLURM mode (set both to activate):
#   SLURM_SSH_STAT_CMD        shell command that prints the remote results.xml mtime
#                             (run on the compute node via srun --overlap to bypass
#                             the login node's NFS attribute cache)
#   SLURM_SSH_REFRESH_CACHE_CMD  shell command run on the login node to prime its NFS
#                             cache after the stat detects a change, before SCP
#   SLURM_SCP_XML_CMD         shell command that SCPs remote results*.xml locally
#
# SLURM mode optional enrichment (non-fatal; each retried up to 3 times):
#   SLURM_SCP_UNFINISHED_CMD  shell command that SCPs remote unfinished_test.txt locally
#   SLURM_SSH_LIST_PERF_CMD   shell command that prints remote perf folder paths (one per line)
#   SLURM_SCP_PERF_TEMPLATE   SCP command with PERF_FOLDER_PLACEHOLDER substituted per folder
#
# Local mode with mtime guard (set to activate; omit for rerun mode):
#   XML_PATH                  path to local results.xml to stat

set +e
last=0
_poll=0
while [ ! -f "$PROGRESS_DONE_FILE" ]; do
    # Poll the done sentinel frequently so stage teardown does not wait for a
    # full progress-upload interval after the foreground process exits.
    _remaining=$PROGRESS_INTERVAL
    while [ "$_remaining" -gt 0 ] && [ ! -f "$PROGRESS_DONE_FILE" ]; do
        _sleep=5
        [ "$_remaining" -lt "$_sleep" ] && _sleep=$_remaining
        sleep "$_sleep"
        _remaining=$((_remaining - _sleep))
    done
    [ -f "$PROGRESS_DONE_FILE" ] && break
    _poll=$((_poll + 1))

    if [ -n "$SLURM_SSH_STAT_CMD" ]; then
        # SLURM_SSH_STAT_CMD runs ls on the remote dir first to flush the NFS
        # attribute cache, then stats results.xml. Use grep to extract the
        # integer mtime so SSH banners cannot pollute the value.
        m=$(eval "$SLURM_SSH_STAT_CMD" 2>/dev/null | grep -oE '^[0-9]+$' | head -1)
        [ -z "$m" ] && m=0
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: poll #${_poll} mtime=${m} last=${last}"
        [ "$m" -le "$last" ] && continue
        # Prime the login node's NFS cache before SCP: the stat above ran on the
        # compute node (via srun --overlap) so the login node may still have a
        # stale view of the directory.  Running ls on the login node here forces
        # a fresh READDIR/GETATTR RPC to the NFS server before we attempt SCP.
        if [ -n "$SLURM_SSH_REFRESH_CACHE_CMD" ]; then
            eval "$SLURM_SSH_REFRESH_CACHE_CMD" 2>/dev/null || true
        fi
        mkdir -p "${WORKSPACE}/${STAGE_NAME}"
        _scp_ok=0
        for _attempt in 1 2 3; do
            eval "$SLURM_SCP_XML_CMD" && { _scp_ok=1; break; }
            echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp xml failed, retry ${_attempt}/3"
            sleep 10
        done
        if [ "$_scp_ok" -eq 0 ]; then
            echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp failed after 3 attempts; skipping this iteration"
            continue
        fi
        last=$m
        # Fetch unfinished_test.txt for timeout XML generation (retry 3 times)
        if [ -n "$SLURM_SCP_UNFINISHED_CMD" ]; then
            _unfinished_ok=0
            for _attempt in 1 2 3; do
                eval "$SLURM_SCP_UNFINISHED_CMD" && { _unfinished_ok=1; break; }
                echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp unfinished failed (attempt ${_attempt}/3)"
                [ "$_attempt" -lt 3 ] && sleep 10
            done
            [ "$_unfinished_ok" -eq 0 ] && echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp unfinished not available, skipping"
        fi
        # Fetch perf result folders (aggr*/disagg*) one by one (retry 3 times each)
        if [ -n "$SLURM_SSH_LIST_PERF_CMD" ] && [ -n "$SLURM_SCP_PERF_TEMPLATE" ]; then
            while IFS= read -r folder; do
                [ -z "$folder" ] && continue
                _perf_ok=0
                for _attempt in 1 2 3; do
                    eval "${SLURM_SCP_PERF_TEMPLATE//PERF_FOLDER_PLACEHOLDER/$folder}" && { _perf_ok=1; break; }
                    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp perf $folder failed (attempt ${_attempt}/3)"
                    [ "$_attempt" -lt 3 ] && sleep 10
                done
                [ "$_perf_ok" -eq 0 ] && echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp perf $folder failed after 3 attempts"
            done < <(eval "$SLURM_SSH_LIST_PERF_CMD" 2>/dev/null)
        fi
    elif [ -n "$XML_PATH" ]; then
        m=$(stat -c %Y "$XML_PATH" 2>/dev/null || echo 0)
        [ "$m" -le "$last" ] && continue
        last=$m
    else
        # Rerun mode has no mtime source, so invoke the snapshot every interval;
        # progress_upload_snapshot.sh skips unchanged content by hash.
        :
    fi

    LABEL="${LABEL_PREFIX}${m:+ (mtime=$m)}" \
    bash "$(dirname "${BASH_SOURCE[0]}")/progress_upload_snapshot.sh"
done
echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: watcher exiting"
