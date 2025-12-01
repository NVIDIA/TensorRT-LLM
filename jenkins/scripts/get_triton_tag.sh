#!/bin/bash

# Script to get triton short tag from docker/Dockerfile.multi
# Usage: ./get_triton_tag.sh [llm_root_path]
# Output: triton short tag to stdout

set -e

# Default to current directory if no path provided
LLM_ROOT="${1:-.}"

# Default triton tag
TRITON_SHORT_TAG="main"

# Path to Dockerfile.multi
DOCKERFILE_MULTI_PATH="${LLM_ROOT}/docker/Dockerfile.multi"

# Check if Dockerfile.multi exists
if [[ -f "$DOCKERFILE_MULTI_PATH" ]]; then
    # Extract TRITON_BASE_TAG from Dockerfile.multi
    TRITON_BASE_TAG_LINE=$(grep -E '^ARG TRITON_BASE_TAG=' "$DOCKERFILE_MULTI_PATH" | tail -n1)

    if [[ -n "$TRITON_BASE_TAG_LINE" ]]; then
        TRITON_BASE_TAG=$(echo "$TRITON_BASE_TAG_LINE" | cut -d'=' -f2)

        if [[ -n "$TRITON_BASE_TAG" ]]; then
            # Remove -py3 suffix and add r prefix
            TRITON_SHORT_TAG="r${TRITON_BASE_TAG%-py3*}"
            echo "Using triton tag from Dockerfile.multi: $TRITON_SHORT_TAG" >&2
        fi
    fi
else
    echo "Dockerfile.multi not found at $DOCKERFILE_MULTI_PATH" >&2
fi

# Output the triton short tag to stdout
echo "$TRITON_SHORT_TAG"
