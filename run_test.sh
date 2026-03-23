#!/bin/bash
# Run CuTEDSL MoE tests for PR #12309 with proper logging and memory management
# Usage: ./run_test.sh [test_filter]
# Example: ./run_test.sh "test_nvfp4_grouped_gemm_blackwell"
# Example: ./run_test.sh  (runs all CuTEDSL MoE tests)

set -euo pipefail

WORKTREE="/home/mihai/workspace/trtllm-pr12309"
LOGDIR="${WORKTREE}/test_logs"
mkdir -p "${LOGDIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOGDIR}/test_${TIMESTAMP}.log"

TEST_FILTER="${1:-}"
FILTER_ARGS=""
if [[ -n "${TEST_FILTER}" ]]; then
    FILTER_ARGS="-k '${TEST_FILTER}'"
fi

echo "=== Test Run: $(date) ===" | tee "${LOGFILE}"
echo "Filter: ${TEST_FILTER:-<all>}" | tee -a "${LOGFILE}"
echo "Log: ${LOGFILE}" | tee -a "${LOGFILE}"

# Show system state before test
echo "" | tee -a "${LOGFILE}"
echo "=== PRE-TEST SYSTEM STATE ===" | tee -a "${LOGFILE}"
free -h | tee -a "${LOGFILE}"
swapon --show | tee -a "${LOGFILE}"
nvidia-smi 2>/dev/null | tee -a "${LOGFILE}"
echo "" | tee -a "${LOGFILE}"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${WORKTREE}:/workspace/TensorRT-LLM" \
  -v /home/mihai/.cache/huggingface:/root/.cache/huggingface \
  --network host \
  -w /workspace/TensorRT-LLM \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_MODULE_LOADING=LAZY \
  tensorrt_llm/devel:latest \
  bash -c "
    set -euo pipefail
    echo '=== Container started: '\$(date)' ==='
    echo '=== Installing TRT-LLM ==='
    pip install --no-build-isolation -e '.[devel]' 2>&1 | tail -5
    pip install -r requirements-dev.txt 2>&1 | tail -5
    echo '=== Upgrading nvidia-cutlass-dsl to 4.3.5 ==='
    pip install nvidia-cutlass-dsl==4.3.5 2>&1 | tail -3
    echo '=== Install done: '\$(date)' ==='
    echo ''
    echo '=== PYTORCH_CUDA_ALLOC_CONF='\${PYTORCH_CUDA_ALLOC_CONF}' ==='
    echo '=== Running pytest ==='
    echo ''
    cd tests/unittest
    python3 -m pytest \
      _torch/thop/parallel/test_cute_dsl_moe.py \
      ${FILTER_ARGS} \
      -v -s \
      --tb=short \
      --timeout=7200 \
      -p no:cacheprovider \
      2>&1
    echo ''
    echo '=== Tests complete: '\$(date)' ==='
  " 2>&1 | tee -a "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}

# Show system state after test
echo "" | tee -a "${LOGFILE}"
echo "=== POST-TEST SYSTEM STATE ===" | tee -a "${LOGFILE}"
free -h | tee -a "${LOGFILE}"
echo "" | tee -a "${LOGFILE}"
echo "=== Exit code: ${EXIT_CODE} ===" | tee -a "${LOGFILE}"
echo "=== Log saved to: ${LOGFILE} ===" | tee -a "${LOGFILE}"
exit ${EXIT_CODE}
