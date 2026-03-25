#!/bin/bash

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
