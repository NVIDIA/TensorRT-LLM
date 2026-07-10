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
#   SLURM_SSH_STAT_CMD  shell command that prints the remote results.xml mtime
#   SLURM_SCP_XML_CMD   shell command that SCPs remote results*.xml locally
#
# Local mode with mtime guard (set to activate; omit for rerun mode):
#   XML_PATH            path to local results.xml to stat

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
        if ! eval "$SLURM_SCP_XML_CMD"; then
            echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: scp failed; skipping this iteration"
            continue
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
