#! /usr/bin/env bash

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": <model_name>,
        "prompt": "Where is New York?",
        "max_tokens": 16,
        "temperature": 0
    }'
