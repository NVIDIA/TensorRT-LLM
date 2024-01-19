#!/bin/bash
set -ex

PROMPT="Tell a story"
LLAMA_MODEL_DIR=$1

python3 llm_examples.py --task run_llm_from_huggingface_model \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --dump_engine_dir=./tllm.engine.example

python3 llm_examples.py --task run_llm_from_tllm_engine \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --dump_engine_dir=./tllm.engine.example

python3 llm_examples.py --task run_llm_on_tensor_parallel \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR

python3 llm_examples.py --task run_llm_generate_async_example \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR

python3 llm_examples.py --task run_llm_with_quantization \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --dump_engine_dir=./tllm.engine.example \
    --quant_type="int4_awq"
