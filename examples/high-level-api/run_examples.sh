#!/bin/bash
set -ex

PROMPT="Tell a story"
LLAMA_MODEL_DIR=$1
default_engine_dir="./tllm.engine.example"
TMP_ENGINE_DIR="${2:-$default_engine_dir}"

python3 llm_examples.py --task run_llm_from_huggingface_model \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --dump_engine_dir=$TMP_ENGINE_DIR

# TP enabled
python3 llm_examples.py --task run_llm_from_huggingface_model \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --tp_size=2

python3 llm_examples.py --task run_llm_from_tllm_engine \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --dump_engine_dir=$TMP_ENGINE_DIR

python3 llm_examples.py --task run_llm_generate_async_example \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR

# Both TP and streaming enabled
python3 llm_examples.py --task run_llm_generate_async_example \
    --prompt="$PROMPT" \
    --hf_model_dir=$LLAMA_MODEL_DIR \
    --streaming \
    --tp_size=2
