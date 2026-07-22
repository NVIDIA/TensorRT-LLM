#!/usr/bin/env bash
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
while [ ! -f "$PROGRESS_DONE_FILE" ]; do
    sleep "$PROGRESS_INTERVAL"
    [ -f "$PROGRESS_DONE_FILE" ] && break

    if [ -n "$SLURM_SSH_STAT_CMD" ]; then
        m=$(eval "$SLURM_SSH_STAT_CMD" 2>/dev/null | tr -dc '0-9')
        [ -z "$m" ] && m=0
        [ "$m" -le "$last" ] && continue
        last=$m
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
    fi

    LABEL="${LABEL_PREFIX}${m:+ (mtime=$m)}" \
    bash "$(dirname "${BASH_SOURCE[0]}")/progress_upload_snapshot.sh" || continue
done
echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: watcher exiting"
