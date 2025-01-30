#!/bin/bash

# Check if the environment variable is set
if [[ -z "${HUGGING_FACE_HUB_TOKEN}" ]]; then
    echo "The environment variable HUGGING_FACE_HUB_TOKEN is not set."
    exit 1
fi

# Get GPU name using nvidia-smi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)

GPU="GH200"

# Check if the GPU is a GH200
if echo "$gpu_name" | grep -q "GH200"; then
    GPU="GH200"
else
    GPU="H100"
fi

echo "Running with ${GPU}."

# Generate context prompts of 16,000 tokens for each user
python3 $(pwd)/../../cpp/prepare_dataset.py \
    --output=$(pwd)/dataset.json \
    --tokenizer=meta-llama/Llama-3.1-70B token-norm-dist \
    --num-requests=20 \
    --input-mean=16000 \
    --output-mean=64 \
    --input-stdev=0 \
    --output-stdev=0

# Build the model
trtllm-bench --workspace $(pwd)/${GPU} \
    --model meta-llama/Llama-3.1-70B \
    build \
    --max_batch_size 16 \
    --max_num_tokens 17800 \
    --max_seq_len 17800 \
    --quantization FP8

# Run the benchmark script
for user_size in $(seq 2 16); do
    echo "Run benchmark with user size = ${user_size}."
    python3 benchmark.py \
        --model_path $(pwd)/${GPU}/meta-llama/Llama-3.1-70B/tp_1_pp_1 \
        --input_dataset_path dataset.json \
        --n ${user_size}
done
