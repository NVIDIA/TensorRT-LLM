#!/bin/bash

echo current time: $(date)
export TLLM_LOG_LEVEL=INFO
echo "Killing existing servers"
export CTX_PROCESS_ID=$(pgrep -f "trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --config ./ctx_config.yaml")
export GEN_PROCESS_ID=$(pgrep -f "trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --config ./gen_config.yaml")
export DISAGG_PROCESS_ID=$(pgrep -f "trtllm-serve disaggregated -c ./disagg_config.yaml")

kill -9 $CTX_PROCESS_ID $GEN_PROCESS_ID $DISAGG_PROCESS_ID

echo "Starting context servers"
# Start context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8001 \
    --config ./ctx_config.yaml 2>&1  | tee log_ctx_0 &

echo "Starting generation server"
# Start generation server
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8002 \
    --config ./gen_config.yaml 2>&1  | tee log_gen_0 &

sleep 80
echo "Starting disaggregated server"
# Start disaggregated server
trtllm-serve disaggregated -c ./disagg_config.yaml 2>&1 | tee log_disagg &

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
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "NVIDIA is a great company because",
        "use_beam_search": true,
        "n": 4,
        "max_tokens": 1024,
        "temperature": 0
    }' -w "\n" 2>&1 | tee output.json

# curl http://localhost:8000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         "prompt": "Some times there is a bug in the context server, and the context server returns 4 choices, but the generation server returns 1 choice.  This is a test of the disaggregated server. We want to see if the disaggregated server can handle this situation and return the correct choice. Please return the correct choice. No mistakes.",
#         "use_beam_search": true,
#         "n": 4,
#         "max_tokens": 1024,
#         "temperature": 0
#     }' -w "\n"

# echo "Request sent"