#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Materialize $JOB_WORKSPACE/.coveragerc from coveragerc.template by substituting env placeholders.

set -euo pipefail

: "${TRTLLM_WHEEL_PATH:?TRTLLM_WHEEL_PATH is required}"
: "${TRTLLM_SRC_PATH:?TRTLLM_SRC_PATH is required}"
: "${JOB_WORKSPACE:?JOB_WORKSPACE is required}"
: "${STAGE_NAME:?STAGE_NAME is required}"

TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_FILE="${TEMPLATE_DIR}/coveragerc.template"
OUTPUT_FILE="${JOB_WORKSPACE}/.coveragerc"

if [[ ! -f "${TEMPLATE_FILE}" ]]; then
    echo "error: template not found at ${TEMPLATE_FILE}" >&2
    exit 1
fi

mkdir -p "${JOB_WORKSPACE}"

sed \
    -e "s|@TRTLLM_WHEEL_PATH@|${TRTLLM_WHEEL_PATH}|g" \
    -e "s|@TRTLLM_SRC_PATH@|${TRTLLM_SRC_PATH}|g" \
    -e "s|@JOB_WORKSPACE@|${JOB_WORKSPACE}|g" \
    -e "s|@STAGE_NAME@|${STAGE_NAME}|g" \
    "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"

echo "${OUTPUT_FILE}"
