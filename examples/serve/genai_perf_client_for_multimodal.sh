#! /usr/bin/env bash

genai-perf profile \
    -m Qwen2.5-VL-3B-Instruct \
    --tokenizer Qwen/Qwen2.5-VL-3B-Instruct \
    --service-kind openai \
    --endpoint-type vision \
    --random-seed 123 \
    --image-width-mean 512 \
    --image-width-stddev 30 \
    --image-height-mean 512 \
    --image-height-stddev 30 \
    --image-format png \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --request-count 100 \
    --request-rate 10 \
    --profile-export-file my_profile_export.json \
    --url localhost:8000 \
    --streaming
