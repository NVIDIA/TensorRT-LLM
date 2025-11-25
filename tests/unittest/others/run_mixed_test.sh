#!/usr/bin/env bash
set -euo pipefail

cd /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others

MODE=${1:-all}

export PYTHONUNBUFFERED=1
export UCX_LOG_LEVEL=info
export TRTLLM_DISABLE_SELECTIVE_CACHE_TRANSFER=1
export TLLM_LOG_LEVEL=INFO

PYTEST_ARGS=(-v -s)
PYTEST_ARGS+=(-k "not nvfp4")

pytest /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others/test_mixed_kv_transceiver.py \
    "${PYTEST_ARGS[@]}" | tee log.log