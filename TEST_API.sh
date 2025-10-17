#! /usr/bin/env bash

curl http://localhost:8500/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "messages":[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Give me a random number "}],
        "max_tokens": 16,
        "temperature": 0
    }'
