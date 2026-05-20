#!/bin/bash
echo "Killing existing servers"
export CTX_PROCESS_ID=$(pgrep -f "trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --config ./ctx_config.yaml")
export GEN_PROCESS_ID=$(pgrep -f "trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --config ./gen_config.yaml")
export DISAGG_PROCESS_ID=$(pgrep -f "trtllm-serve disaggregated -c ./disagg_config.yaml")

kill -9 $CTX_PROCESS_ID $GEN_PROCESS_ID $DISAGG_PROCESS_ID
