#! /usr/bin/env bash

genai-perf profile \
    -m TinyLlama-1.1B-Chat-v1.0 \
    --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --service-kind openai \
    --endpoint-type chat \
    --num-prompts 1 \
    --random-seed 123 \
    --synthetic-input-tokens-mean 128 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --request-rate 1 \
    --profile-export-file my_profile_export.json \
    --url localhost:8000 \
    --streaming
