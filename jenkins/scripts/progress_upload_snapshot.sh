#!/usr/bin/env bash
# Tar the stage results directory and upload a progress snapshot to Artifactory.
# Exits 0 on success, 1 on failure (logs the outcome either way).
#
# Required env vars (WORKSPACE/ART_USER/ART_PASS are injected by Jenkins/withCredentials):
#   STAGE_NAME    stage directory name and log prefix
#   PROGRESS_TAR  tar filename (relative to WORKSPACE)
#   PROGRESS_URL  Artifactory PUT URL
#   LABEL         short description for log messages (e.g. "checkpoint", "run1 final snapshot")

set +e
( cd "$WORKSPACE" && tar -czf "$PROGRESS_TAR" "${STAGE_NAME}/" ) || {
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} tar failed"
    exit 1
}
if curl -fsSL --retry 2 -u "$ART_USER:$ART_PASS" \
        -T "${WORKSPACE}/${PROGRESS_TAR}" "$PROGRESS_URL"; then
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} uploaded"
else
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} upload failed (non-fatal)"
    exit 1
fi
