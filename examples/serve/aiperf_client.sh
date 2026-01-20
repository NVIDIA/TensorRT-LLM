#! /usr/bin/env bash

aiperf profile \
    -m TinyLlama-1.1B-Chat-v1.0 \
    --tokenizer ${AIPERF_TOKENIZER_PATH:-TinyLlama/TinyLlama-1.1B-Chat-v1.0} \
    --endpoint-type chat \
    --random-seed 123 \
    --synthetic-input-tokens-mean 128 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --request-count 100 \
    --request-rate 10 \
    --profile-export-file my_profile_export.json \
    --url localhost:8000 \
    --streaming
