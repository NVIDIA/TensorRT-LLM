#!/usr/bin/env bash
# Tar the stage results directory and upload a progress snapshot to Artifactory.
# Exits 0 on success, 1 on failure (logs the outcome either way).
#
# Required env vars (WORKSPACE/ART_USER/ART_PASS are injected by Jenkins/withCredentials):
#   STAGE_NAME    stage directory name and log prefix
#   PROGRESS_TAR  tar filename (relative to WORKSPACE)
#   PROGRESS_URL  Artifactory PUT URL
#   LABEL         short description for log messages (e.g. "checkpoint", "run1 final snapshot")
#
# Optional:
#   PROGRESS_HASH_FILE  path to store the content hash of the last successful upload
#                       (default: ${WORKSPACE}/${PROGRESS_TAR}.last_hash)
#                       Set to empty string to disable dedup.
#   TIMEOUT_XML_SCRIPT  path to generate_timeout_xml.py; when set, regenerates
#                       results-timeout.xml from unfinished_test.txt before each upload.

set +e

# --- Generate timeout XML from unfinished_test.txt (final snapshot only) ---
# Skipped during periodic snapshots so incomplete unfinished_test.txt does not produce premature timeout reports.
if [ -n "$FINAL_SNAPSHOT" ] && \
        [ -n "$TIMEOUT_XML_SCRIPT" ] && [ -f "${WORKSPACE}/${STAGE_NAME}/unfinished_test.txt" ]; then
    ( cd "$WORKSPACE" && python3 "$TIMEOUT_XML_SCRIPT" \
        --stage-name "$STAGE_NAME" \
        --test-file-path "unfinished_test.txt" \
        --output-file "${WORKSPACE}/${STAGE_NAME}/results-timeout.xml" ) 2>/dev/null || true
fi

# --- Fix testsuite name in XML files (pytest default -> stage name) ---
find "${WORKSPACE}/${STAGE_NAME}" -name "*.xml" -exec \
    sed -i "s|testsuite name=\"pytest\"|testsuite name=\"${STAGE_NAME}\"|g" {} + 2>/dev/null || true

# --- Content-based dedup: skip upload when stage directory is unchanged ---
HASH_FILE="${PROGRESS_HASH_FILE-${WORKSPACE}/${PROGRESS_TAR}.last_hash}"
if [ -n "$HASH_FILE" ]; then
    current_hash=$(find "${WORKSPACE}/${STAGE_NAME}" -type f | sort \
                   | xargs sha256sum 2>/dev/null | sha256sum | awk '{print $1}')
    last_hash=$(cat "$HASH_FILE" 2>/dev/null || echo "")
    if [ -n "$current_hash" ] && [ "$current_hash" = "$last_hash" ]; then
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} content unchanged, skipping upload"
        exit 0
    fi
fi

# Use --transform to rename results*.xml inside the tar when POST_TAG is set,
# without touching the on-disk files (SCP overwrites them cleanly next iteration).
if [ -n "$POST_TAG" ]; then
    ( cd "$WORKSPACE" && tar -czf "$PROGRESS_TAR" \
        --transform "s|^\(${STAGE_NAME}/results[^/]*\)\.xml$|\1${POST_TAG}.xml|" \
        "${STAGE_NAME}/" ) || {
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} tar failed"
        exit 1
    }
else
    ( cd "$WORKSPACE" && tar -czf "$PROGRESS_TAR" "${STAGE_NAME}/" ) || {
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} tar failed"
        exit 1
    }
fi
rm -f "${WORKSPACE}/${PROGRESS_TAR}.upload_ok" 2>/dev/null || true
if curl -fsSL --retry 2 -u "$ART_USER:$ART_PASS" \
        -T "${WORKSPACE}/${PROGRESS_TAR}" "$PROGRESS_URL"; then
    [ -n "$HASH_FILE" ] && echo "$current_hash" > "$HASH_FILE"
    touch "${WORKSPACE}/${PROGRESS_TAR}.upload_ok" 2>/dev/null || true
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} uploaded"
else
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} upload failed (non-fatal)"
    exit 1
fi
