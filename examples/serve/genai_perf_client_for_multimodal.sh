#! /usr/bin/env bash

# Set to TinyLlama/TinyLlama-1.1B-Chat-v1.0 to download from Hugging Face.
# Or set to the path of a local tokenizer directory.
TOKENIZER_PATH_OR_NAME="/home/scratch.trt_llm_data/llm-models/Qwen2.5-VL-3B-Instruct"
genai-perf profile \
    -m Qwen2.5-VL-3B-Instruct \
    --tokenizer $TOKENIZER_PATH_OR_NAME \
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
