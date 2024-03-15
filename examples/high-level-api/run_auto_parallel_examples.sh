#!/bin/bash
set -ex

PROMPT="Tell a story"
LLAMA_MODEL_DIR=$1
WORLD_SIZE=${2:-2}

dir=$(dirname "$0")

python3 $dir/llm_examples.py --task run_llm_with_auto_parallel \
    --prompt="$PROMPT" \
    --world_size=$WORLD_SIZE \
    --hf_model_dir=$LLAMA_MODEL_DIR

python3 $dir/llm_examples.py --task run_llm_with_auto_parallel_async \
    --prompt="$PROMPT" \
    --world_size=$WORLD_SIZE \
    --hf_model_dir=$LLAMA_MODEL_DIR
