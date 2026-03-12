#!/bin/bash

# Copyright 2026, NVIDIA CORPORATION & AFFILIATES
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

# Script to fetch imageKeyToTag.json from the latest L0_PostMerge build for a given branch.
# Usage: ./get_image_key_to_tag.sh <branch_name>
# Output: imageKeyToTag.json content to stdout

set -e

BRANCH_NAME="${1}"

if [[ -z "$BRANCH_NAME" ]]; then
    echo "Usage: $0 <branch_name>" >&2
    exit 1
fi

JENKINS_BASE="https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/${BRANCH_NAME}/job/L0_PostMerge"
ARTIFACTORY_BASE="https://urm.nvidia.com/artifactory/sw-tensorrt-generic-local/llm-artifacts/LLM/${BRANCH_NAME}/L0_PostMerge"

echo "Fetching latest build number from Jenkins for branch: ${BRANCH_NAME}" >&2

# Query Jenkins API for the last successful build number
BUILD_NUMBER=$(curl -fsSL "${JENKINS_BASE}/lastBuild/api/json" | python3 -c "import json,sys; print(json.load(sys.stdin)['number'])" 2>/dev/null)

if [[ -z "$BUILD_NUMBER" ]]; then
    echo "Failed to get last successful build number. Trying last completed build..." >&2
    BUILD_NUMBER=$(curl -fsSL "${JENKINS_BASE}/lastCompletedBuild/api/json" | python3 -c "import json,sys; print(json.load(sys.stdin)['number'])" 2>/dev/null)
fi

if [[ -z "$BUILD_NUMBER" ]]; then
    echo "Error: Could not determine the latest build number from ${JENKINS_BASE}" >&2
    exit 1
fi

echo "Latest build number: ${BUILD_NUMBER}" >&2

while [[ "${BUILD_NUMBER}" -gt 0 ]]; do
    ARTIFACT_URL="${ARTIFACTORY_BASE}/${BUILD_NUMBER}/imageKeyToTag.json"
    echo "Fetching: ${ARTIFACT_URL}" >&2
    HTTP_STATUS=$(curl -sL -o /tmp/imageKeyToTag.json -w "%{http_code}" "${ARTIFACT_URL}")
    if [[ "${HTTP_STATUS}" == "200" ]]; then
        cat /tmp/imageKeyToTag.json
        exit 0
    fi
    echo "Got HTTP ${HTTP_STATUS} for build ${BUILD_NUMBER}, trying build $((BUILD_NUMBER - 1))..." >&2
    BUILD_NUMBER=$((BUILD_NUMBER - 1))
done

echo "Error: Could not find imageKeyToTag.json in any recent build" >&2
exit 1
