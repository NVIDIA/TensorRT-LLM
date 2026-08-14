#! /usr/bin/env bash

curl http://localhost:8000/v1/responses \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "input": "Where is New York?",
        "max_output_tokens": 16
    }'
