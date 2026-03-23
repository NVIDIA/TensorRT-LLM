#!/bin/bash
# E2E test: serve a small MoE model on SM121 with NVFP4, verify it generates text
# Tests the full path: model load → autotuner → MoE dispatch → CUTLASS GEMM → generation

set -euo pipefail

WORKTREE="/home/mihai/workspace/trtllm-pr12309"
LOGDIR="${WORKTREE}/test_logs"
mkdir -p "${LOGDIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOGDIR}/e2e_${TIMESTAMP}.log"
MODEL="${1:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
PORT=9123

echo "=== E2E Test: $(date) ===" | tee "${LOGFILE}"
echo "Model: ${MODEL}" | tee -a "${LOGFILE}"
echo "Log: ${LOGFILE}" | tee -a "${LOGFILE}"
free -h | tee -a "${LOGFILE}"

echo "" | tee -a "${LOGFILE}"
echo "=== Starting trtllm-serve in docker ===" | tee -a "${LOGFILE}"

# Run trtllm-serve inside the container
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${WORKTREE}:/workspace/TensorRT-LLM" \
  -v /home/mihai/.cache/huggingface:/root/.cache/huggingface \
  --network host \
  -w /workspace/TensorRT-LLM \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/root/.cache/huggingface \
  --name trtllm-e2e-test \
  tensorrt_llm/devel:latest \
  bash -c "
    set -euo pipefail
    echo '=== Container started: '\$(date)' ==='

    echo '=== Installing TRT-LLM ==='
    pip install --no-build-isolation -e '.[devel]' 2>&1 | tail -3
    echo '=== Install done: '\$(date)' ==='

    echo ''
    echo '=== Starting trtllm-serve on port ${PORT} ==='
    echo '=== Model: ${MODEL} ==='
    echo '=== Backend: pytorch (default AUTO MoE) ==='
    echo ''

    # Start the server in background, log output
    trtllm-serve ${MODEL} \
      --backend pytorch \
      --port ${PORT} \
      --max_batch_size 1 \
      --max_seq_len 256 \
      2>&1 &
    SERVER_PID=\$!

    # Wait for server to be ready (poll health endpoint)
    echo 'Waiting for server to start...'
    for i in \$(seq 1 1200); do
      if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo \"Server ready after \${i}s\"
        break
      fi
      if ! kill -0 \$SERVER_PID 2>/dev/null; then
        echo 'Server process died!'
        wait \$SERVER_PID || true
        exit 1
      fi
      sleep 1
    done

    if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
      echo 'Server failed to start within 1200s'
      kill \$SERVER_PID 2>/dev/null || true
      exit 1
    fi

    echo ''
    echo '=== Running inference test ==='
    RESPONSE=\$(curl -s http://localhost:${PORT}/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{
        \"model\": \"${MODEL}\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 32,
        \"temperature\": 0.0
      }')

    echo \"Response: \${RESPONSE}\"
    echo ''

    # Check if response contains generated text
    if echo \"\${RESPONSE}\" | python3 -c 'import json,sys; d=json.load(sys.stdin); text=d[\"choices\"][0][\"text\"]; print(f\"Generated: {text}\"); assert len(text.strip()) > 0, \"Empty response\"' 2>&1; then
      echo '=== E2E TEST PASSED ==='
    else
      echo '=== E2E TEST FAILED: no valid response ==='
      echo \"Raw response: \${RESPONSE}\"
      kill \$SERVER_PID 2>/dev/null || true
      exit 1
    fi

    # Cleanup
    kill \$SERVER_PID 2>/dev/null || true
    wait \$SERVER_PID 2>/dev/null || true
    echo '=== Done: '\$(date)' ==='
  " 2>&1 | tee -a "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a "${LOGFILE}"
echo "=== Exit code: ${EXIT_CODE} ===" | tee -a "${LOGFILE}"
echo "=== Log saved to: ${LOGFILE} ===" | tee -a "${LOGFILE}"

# Cleanup container if still running
docker rm -f trtllm-e2e-test 2>/dev/null || true

exit ${EXIT_CODE}
