#!/bin/bash
mkdir -p Logs/
CUDA_VISIBLE_DEVICES=0 trtllm-serve encoder llava-hf/llava-v1.6-mistral-7b-hf \
    --host localhost \
    --port 8001 \
    --backend pytorch \
    &> Logs/log_encoder_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve llava-hf/llava-v1.6-mistral-7b-hf \
    --host localhost \
    --port 8002 \
    --backend pytorch \
    --extra_llm_api_options ./extra-llm-api-config.yml \
    &> Logs/log_pd_tp1 &
trtllm-serve disaggregated_mm -c disagg_config.yaml &> Logs/log_disagg_server &
