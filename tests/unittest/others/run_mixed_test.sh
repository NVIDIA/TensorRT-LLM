#!/usr/bin/env bash
set -euo pipefail

cd /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others

MODE=${1:-all}

export PYTHONUNBUFFERED=1
export TRTLLM_ENABLE_KV_CACHE_PRECISION_CONVERSION=1
export TRTLLM_KVCACHE_ENABLE_PRECISION_CONVERSION=1
export UCX_LOG_LEVEL=info
export TRTLLM_DISABLE_SELECTIVE_CACHE_TRANSFER=1
export TLLM_LOG_LEVEL=INFO

PYTEST_ARGS=(-v -s)
case "${MODE}" in
    all)
        ;;
    without-nvfp4)
        PYTEST_ARGS+=(-k "not nvfp4")
        ;;
    *)
        echo "Usage: $0 [all|without-nvfp4]" >&2
        exit 1
        ;;
esac

pytest /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others/test_mixed_kv_transceiver.py \
    "${PYTEST_ARGS[@]}" | tee log.log