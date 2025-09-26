#! /usr/bin/env bash

genai-perf profile \
    -m Qwen2.5-VL-3B-Instruct \
    --tokenizer Qwen/Qwen2.5-VL-3B-Instruct \
    --endpoint-type multimodal \
    --random-seed 123 \
    --image-width-mean 64 \
    --image-height-mean 64 \
    --image-format png \
    --synthetic-input-tokens-mean 128 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --request-count 5 \
    --request-rate 1 \
    --profile-export-file my_profile_export.json \
    --url localhost:8000 \
    --streaming
