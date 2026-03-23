#!/bin/bash
set -e
cd /code/tensorrt_llm
git config --global --add safe.directory /code/tensorrt_llm
git config --global --add safe.directory /home/mihai/workspace/TensorRT-LLM
git config --global --add safe.directory /home/mihai/workspace/trtllm-pr12309
pip install pre-commit 2>&1 | tail -1
CHANGED=$(git diff --name-only main...HEAD)
echo "Changed files: $CHANGED"
pre-commit run --show-diff-on-failure --files $CHANGED 2>&1
