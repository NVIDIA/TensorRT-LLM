#!/bin/bash
echo "Killing existing servers"
export AGG_PROCESS_ID=$(pgrep -f "trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8000 --config ./agg_config.yaml")
kill -9 $AGG_PROCESS_ID
