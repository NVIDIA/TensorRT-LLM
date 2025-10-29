#! /usr/bin/env bash

trtllm-serve \
    deepseek-ai/DeepSeek-R1 \
    --host localhost --port 8000 \
    --max_batch_size 161 --max_num_tokens 1160 \
    --tp_size 8 --ep_size 8 --pp_size 1 \
    --kv_cache_free_gpu_memory_fraction 0.95 \
    --extra_llm_api_options ./extra-llm-api-config.yml \
    --reasoning_parser deepseek-r1
