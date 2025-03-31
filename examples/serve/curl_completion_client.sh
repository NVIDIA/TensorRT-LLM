#! /usr/bin/env bash

if [[ -z $BASE_URL ]]; then
    BASE_URL=http://localhost:8000/v1
fi

curl ${BASE_URL}/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": TinyLlama-1.1B-Chat-v1.0,
        "prompt": "Where is New York?",
        "max_tokens": 16,
        "temperature": 0
    }'
