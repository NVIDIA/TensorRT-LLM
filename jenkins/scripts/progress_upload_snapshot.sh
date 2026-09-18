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
#   TIMEOUT_XML_SCRIPT  path to generate_timeout_xml.py; when set, the final
#                       snapshot merges unfinished lists and timeout JSONL files.

set +e

# --- Generate timeout XML from unfinished lists and timeout data (final snapshot only) ---
# Skipped during periodic snapshots so incomplete unfinished_test.txt does not
# produce premature timeout reports. A stage can have the original list plus
# rerun lists, and a Slurm invocation can have one JSONL file per rank.
if [ -n "$FINAL_SNAPSHOT" ] && \
        [ -n "$TIMEOUT_XML_SCRIPT" ] && [ -d "${WORKSPACE}/${STAGE_NAME}" ]; then
    test_file_args=()
    while IFS= read -r path; do
        test_file_args+=(--test-file-path "$path")
    done < <(find "${WORKSPACE}/${STAGE_NAME}" -type f \
        -name unfinished_test.txt -print | sort)

    timeout_data_args=()
    while IFS= read -r path; do
        timeout_data_args+=(--timeout-data-file "$path")
    done < <(find "${WORKSPACE}/${STAGE_NAME}" -maxdepth 1 -type f \
        -name 'timeout_data*.jsonl' -print | sort)

    if [ "${#test_file_args[@]}" -gt 0 ]; then
        python3 "$TIMEOUT_XML_SCRIPT" \
            --stage-name "$STAGE_NAME" \
            "${test_file_args[@]}" \
            --output-file "${WORKSPACE}/${STAGE_NAME}/results-timeout.xml" \
            "${timeout_data_args[@]}" 2>/dev/null || true
    fi
fi

# --- Fix testsuite name in XML files (pytest default -> stage name) ---
find "${WORKSPACE}/${STAGE_NAME}" -name "*.xml" -exec \
    sed -i "s|testsuite name=\"pytest\"|testsuite name=\"${STAGE_NAME}\"|g" {} + 2>/dev/null || true

# --- Content-based dedup: skip upload when stage directory is unchanged ---
HASH_FILE="${PROGRESS_HASH_FILE-${WORKSPACE}/${PROGRESS_TAR}.last_hash}"
if [ -n "$HASH_FILE" ]; then
    current_hash=$(find "${WORKSPACE}/${STAGE_NAME}" -type f \
                   ! -name 'timeout_data*.jsonl' | sort \
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
        --exclude="${STAGE_NAME}/timeout_data*.jsonl" \
        --transform "s|^\(${STAGE_NAME}/results[^/]*\)\.xml$|\1${POST_TAG}.xml|" \
        "${STAGE_NAME}/" ) || {
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} tar failed"
        exit 1
    }
else
    ( cd "$WORKSPACE" && tar -czf "$PROGRESS_TAR" \
        --exclude="${STAGE_NAME}/timeout_data*.jsonl" \
        "${STAGE_NAME}/" ) || {
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} tar failed"
        exit 1
    }
fi
rm -f "${WORKSPACE}/${PROGRESS_TAR}.upload_ok" 2>/dev/null || true
if curl -fsSL --retry 2 -o /dev/null -u "$ART_USER:$ART_PASS" \
        -T "${WORKSPACE}/${PROGRESS_TAR}" "$PROGRESS_URL"; then
    [ -n "$HASH_FILE" ] && echo "$current_hash" > "$HASH_FILE"
    touch "${WORKSPACE}/${PROGRESS_TAR}.upload_ok" 2>/dev/null || true
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} uploaded"
else
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} upload failed (non-fatal)"
    exit 1
fi
