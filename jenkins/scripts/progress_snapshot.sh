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

# Tar the stage results directory into a progress snapshot for the caller to upload, and record
# the snapshot's content hash. Makes no network request and needs no credential.
#
# Required env vars:
#   WORKSPACE     the stage's workspace
#   STAGE_NAME    stage directory name (under WORKSPACE) and log prefix
#   PROGRESS_TAR  tar filename (relative to WORKSPACE)
#   LABEL         short description for log messages (e.g. "checkpoint", "run1 final snapshot")
#
# Optional:
#   POST_TAG            the attempt's suffix; renames results*.xml inside the tar, not on disk
#   FINAL_SNAPSHOT      non-empty for the snapshot taken when the tests end
#   TIMEOUT_XML_SCRIPT  path to generate_timeout_xml.py; with FINAL_SNAPSHOT, regenerates
#                       results-timeout.xml from unfinished_test.txt
#   PROGRESS_HASH_FILE  path that holds the content hash every upload target last received
#                       (default: ${WORKSPACE}/${PROGRESS_TAR}.last_hash).
#                       Set to empty string to disable dedup.
#                       The script only reads it: it writes the new hash to
#                       ${PROGRESS_HASH_FILE}.pending, and the caller moves that file over
#                       PROGRESS_HASH_FILE once every upload target has the snapshot.
#
# Exit codes:
#   0  the tar is ready, and (with dedup) its content hash is in ${PROGRESS_HASH_FILE}.pending
#   3  the content matches PROGRESS_HASH_FILE, so there is nothing to upload; no tar is written
#   1  the tar or the pending hash could not be written, or a required env var is not set

set +e

for _var in WORKSPACE STAGE_NAME PROGRESS_TAR LABEL; do
    if [ -z "${!_var}" ]; then
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${_var} is not set"
        exit 1
    fi
done

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

# --- Content-based dedup: no snapshot when every upload target already has this content ---
HASH_FILE="${PROGRESS_HASH_FILE-${WORKSPACE}/${PROGRESS_TAR}.last_hash}"
if [ -n "$HASH_FILE" ]; then
    current_hash=$(find "${WORKSPACE}/${STAGE_NAME}" -type f | sort \
                   | xargs sha256sum 2>/dev/null | sha256sum | awk '{print $1}')
    last_hash=$(cat "$HASH_FILE" 2>/dev/null || echo "")
    if [ -n "$current_hash" ] && [ "$current_hash" = "$last_hash" ]; then
        echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} content unchanged, skipping upload"
        exit 3
    fi
fi

# Use --transform to rename results*.xml inside the tar when POST_TAG is set, without touching the
# on-disk files, which the tests and later copies keep overwriting under their own names.
transform=()
if [ -n "$POST_TAG" ]; then
    transform=(--transform "s|^\(${STAGE_NAME}/results[^/]*\)\.xml$|\1${POST_TAG}.xml|")
fi
if ! ( cd "$WORKSPACE" && tar -czf "$PROGRESS_TAR" "${transform[@]}" "${STAGE_NAME}/" ); then
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} tar failed"
    exit 1
fi

if [ -n "$HASH_FILE" ] && ! echo "$current_hash" > "${HASH_FILE}.pending"; then
    echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} could not write ${HASH_FILE}.pending"
    exit 1
fi
echo "[PROGRESS-UPLOAD] ${STAGE_NAME}: ${LABEL} tarball ready"
