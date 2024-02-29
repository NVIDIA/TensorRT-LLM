#!/bin/bash
set -ex

hf_model_dir=$1
engine_dir=$2

# fake a 1-layer LLaMA model for CI
python3 ../../examples/llama/build.py \
    --use_gemm_plugin \
    --enable_context_fmha \
    --use_gpt_attention_plugin \
    --paged_kv_cache \
    --remove_input_padding \
    --n_layer 1 \
    --dtype float16 \
    --output_dir $engine_dir

cp $hf_model_dir/tokenizer* $engine_dir
