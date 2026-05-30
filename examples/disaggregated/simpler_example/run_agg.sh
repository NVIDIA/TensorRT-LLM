#!/bin/bash

echo current time: $(date)
export TLLM_LOG_LEVEL_BY_MODULE=debug:batchmgr
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HOST=localhost
PORT=8000
CONFIG=./agg_config.yaml

./kill_agg.sh
./kill.sh 

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
        "n": 4,
        "max_tokens": 1024,
        "temperature": 0
    }' -w "\n" 2>&1 | tee output_agg.json

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "The layers of the atmosphere are",
        "use_beam_search": true,
        "n": 4,
        "max_tokens": 1024,
        "temperature": 0
    }' -w "\n" 2>&1 | tee atmosphere_output.json

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Home is whenever I am with you",
        "use_beam_search": true,
        "n": 4,
        "max_tokens": 1024,
        "temperature": 0
    }' -w "\n" 


curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Tungsten is a metal",
        "use_beam_search": true,
        "n": 4,
        "max_tokens": 1024,
        "temperature": 0
    }' -w "\n" 


curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": "When war orphan Rin aced the Keju",
    "use_beam_search": true,
    "n": 4,
    "max_tokens": 1024,
    "temperature": 0
}' -w "\n" 