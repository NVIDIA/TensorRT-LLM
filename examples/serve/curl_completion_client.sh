#! /usr/bin/env bash

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Where is New York?",
        "max_tokens": 16,
        "temperature": 0
    }'
