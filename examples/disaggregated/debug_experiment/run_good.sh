#!/bin/bash

echo current time: $(date)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=$1
MODEL_PATH=dummy_path
MODEL_NAME=deterministic-beam-dummy
CUSTOM_MODULE_DIR=$SCRIPT_DIR/deterministic_beam_model

mkdir -p "$LOG_DIR"

echo "Starting context servers"
# Start context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve "$MODEL_PATH" \
    --host localhost --port 8001 \
    --served_model_name "$MODEL_NAME" \
    --custom_module_dirs "$CUSTOM_MODULE_DIR" \
    --config "$SCRIPT_DIR/ctx_config_good.yaml" 2>&1  | tee "$LOG_DIR/log_ctx_good" &

echo "Starting generation server"
# Start generation server
CUDA_VISIBLE_DEVICES=1 trtllm-serve "$MODEL_PATH" \
    --host localhost --port 8002 \
    --served_model_name "$MODEL_NAME" \
    --custom_module_dirs "$CUSTOM_MODULE_DIR" \
    --config "$SCRIPT_DIR/gen_config_good.yaml" 2>&1  | tee "$LOG_DIR/log_gen_good" &

sleep 80
echo "Starting disaggregated server"
# Start disaggregated server
trtllm-serve disaggregated -c "$SCRIPT_DIR/disagg_config.yaml" 2>&1 | tee "$LOG_DIR/log_disagg_good" &

echo "Waiting for the disaggregated server to start"
# wait for the disaggregated server to start

# Health check for the disaggregated server
while [ $(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/health") -ne 200 ]; do
    sleep 1
done

echo "The disaggregated server is ready"

echo "Sending a request to the disaggregated server"
# send a request to the disaggregated server
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deterministic-beam-dummy",
        "prompt": [1, 2, 3],
        "use_beam_search": true,
        "n": 2,
        "best_of": 2,
        "max_tokens": 8,
        "temperature": 0,
        "detokenize": false
    }' -w "\n" 2>&1 | tee "$LOG_DIR/first_output_good.json"

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deterministic-beam-dummy",
        "prompt": [1, 5, 6],
        "use_beam_search": true,
        "n": 2,
        "best_of": 2,
        "max_tokens": 8,
        "temperature": 0,
        "detokenize": false
    }' -w "\n" 2>&1 | tee "$LOG_DIR/second_output_good.json"
