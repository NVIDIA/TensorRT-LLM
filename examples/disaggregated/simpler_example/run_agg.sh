#!/bin/bash

echo current time: $(date)
export TLLM_LOG_LEVEL=INFO
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HOST=localhost
PORT=8000
CONFIG=./agg_config.yaml

echo "Killing existing servers"
AGG_PROCESS_ID=$(pgrep -f "trtllm-serve ${MODEL} --host ${HOST} --port ${PORT} --config ${CONFIG}" || true)
DISAGG_PROCESS_ID=$(pgrep -f "trtllm-serve disaggregated -c ./disagg_config.yaml" || true)

if [ -n "${AGG_PROCESS_ID}" ] || [ -n "${DISAGG_PROCESS_ID}" ]; then
    kill -9 ${AGG_PROCESS_ID} ${DISAGG_PROCESS_ID}
fi

echo "Starting aggregate server"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} trtllm-serve ${MODEL} \
    --host ${HOST} --port ${PORT} \
    --config ${CONFIG} 2>&1 | tee log_agg &

echo "Waiting for the aggregate server to start"
while [ $(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health") -ne 200 ]; do
    sleep 1
done

echo "The aggregate server is ready"

echo "Sending a beam search request to the aggregate server"
curl http://${HOST}:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "NVIDIA is a great company because",
        "use_beam_search": true,
        "best_of": 4,
        "n": 4,
        "max_tokens": 1024,
        "temperature": 0
    }' -w "\n" 2>&1 | tee output_agg.json
