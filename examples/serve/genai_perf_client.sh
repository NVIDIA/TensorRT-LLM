#! /usr/bin/env bash

# Set to TinyLlama/TinyLlama-1.1B-Chat-v1.0 to download from Hugging Face.
# Or set to the path of a local tokenizer directory.
TOKENIZER_PATH_OR_NAME= \
    "/home/scratch.trt_llm_data/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
genai-perf profile \
    -m TinyLlama-1.1B-Chat-v1.0 \
    --tokenizer $TOKENIZER_PATH_OR_NAME \
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
