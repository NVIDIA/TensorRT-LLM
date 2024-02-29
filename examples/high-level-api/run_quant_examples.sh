#!/bin/bash
set -ex

PROMPT="Tell a story"
LLAMA_MODEL_DIR=$1


python3 llm_examples.py --task run_llm_with_quantization \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --quant_type="int4_awq"

python3 llm_examples.py --task run_llm_with_quantization \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --quant_type="fp8"
