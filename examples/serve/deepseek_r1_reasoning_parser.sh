#! /usr/bin/env bash

cat >./config.yml <<EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 512
enable_attention_dp: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
stream_interval: 10
moe_config:
    backend: DEEPGEMM
EOF

trtllm-serve \
    deepseek-ai/DeepSeek-R1 \
    --host localhost --port 8000 \
    --trust_remote_code \
    --max_batch_size 1024 --max_num_tokens 8192 \
    --tp_size 8 --ep_size 8 --pp_size 1 \
    --config ./config.yml \
    --reasoning_parser deepseek-r1
