#!/bin/bash

set -e  # Exit on any error

echo "=== LoRA TRT gptManagerBenchmark TP=2 Scenario 1 Profiling ==="
echo "=============================================================="
echo "Scenario 1: Engine WITHOUT LoRA, Requests WITHOUT LoRA"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================
MODEL_CHECKPOINT=/lustre/fsw/portfolios/coreai/users/smor/models/Llama-3.1-8B-Instruct
SOURCE_LORA=/lustre/fsw/portfolios/coreai/users/smor/models/lora_adapter/lora-llama-3.1-nemoguard-8b-topic-control
CPP_LORA=/lustre/fsw/portfolios/coreai/users/smor/models/lora_adapter/lora-llama-3.1-nemoguard-8b-topic-control-cpp
EG_DIR=/lustre/fsw/portfolios/coreai/users/smor/models/lora-eg
TOKENIZER=/lustre/fsw/portfolios/coreai/users/smor/models/Llama-3.1-8B-Instruct

DTYPE=float16
ISL=100
OSL=1000
MAX_BATCH=1
NUM_LAYERS=32
MAX_LORA_RANK=8
NUM_LORA_MODS=7
EOS_ID=2
NUM_REQUESTS=2

# Fixed TP=2
TP=2

echo "Configuration:"
echo "  ISL/OSL: ${ISL}/${OSL}"
echo "  TP Size: ${TP}"
echo "  Requests: ${NUM_REQUESTS}"
echo ""

# ============================================================================
# One-time Setup
# ============================================================================
echo "=== One-time Setup ==="

# Download LoRA adapter (if not exists)
if [ ! -d "$SOURCE_LORA" ]; then
    echo "Downloading LoRA adapter..."
    mkdir -p /lustre/fsw/portfolios/coreai/users/smor/models/lora_adapter
    huggingface-cli download nvidia/llama-3.1-nemoguard-8b-topic-control --local-dir /lustre/fsw/portfolios/coreai/users/smor/models/lora_adapter/lora-llama-3.1-nemoguard-8b-topic-control
else
    echo "LoRA adapter already exists, skipping download"
fi

# Convert LoRA to cpp format (if not exists)
if [ ! -d "$CPP_LORA" ]; then
    echo "Converting LoRA to C++ format..."
    python examples/hf_lora_convert.py \
        -i $SOURCE_LORA \
        --storage-type $DTYPE \
        -o $CPP_LORA
else
    echo "LoRA C++ format already exists, skipping conversion"
fi

# Create base directories
mkdir -p $EG_DIR/data

echo "One-time setup completed!"
echo ""

# ============================================================================
# TP=2 Scenario 1: Engine WITHOUT LoRA, Requests WITHOUT LoRA
# ============================================================================
echo "=== TP=${TP} - Scenario 1: Engine WITHOUT LoRA, Requests WITHOUT LoRA ==="

# TP-specific directories
BASE_DIR="/lustre/fsw/portfolios/coreai/users/smor/models/lora-benchmark-tp${TP}"
CONVERTED_CHECKPOINT="${BASE_DIR}/llama-3.1-8B-ckpt"
ENGINE_NO_LORA="${BASE_DIR}/llama-3.1-8B-engine-no-lora"

mkdir -p ${BASE_DIR}

echo "Converting checkpoint (no LoRA) for TP=${TP}..."
if [ ! -d "${CONVERTED_CHECKPOINT}" ]; then
    python examples/models/core/llama/convert_checkpoint.py \
        --model_dir ${MODEL_CHECKPOINT} \
        --output_dir ${CONVERTED_CHECKPOINT} \
        --dtype ${DTYPE} \
        --tp_size ${TP} \
        --pp_size 1
    echo "Checkpoint conversion completed"
else
    echo "Checkpoint already exists, skipping"
fi

echo "Building engine (no LoRA) for TP=${TP}..."
if [ ! -d "${ENGINE_NO_LORA}" ]; then
    trtllm-build \
        --checkpoint_dir ${CONVERTED_CHECKPOINT} \
        --output_dir ${ENGINE_NO_LORA} \
        --max_batch_size ${MAX_BATCH} \
        --max_input_len ${ISL} \
        --max_seq_len $((${OSL}+${ISL})) \
        --gemm_plugin ${DTYPE} \
        --use_paged_context_fmha enable
    echo "Engine build completed"
else
    echo "Engine already exists, skipping"
fi

echo "Preparing dataset (no LoRA task IDs)..."
if [ ! -f "${EG_DIR}/data/token-norm-dist-no-lora.json" ]; then
    python benchmarks/cpp/prepare_dataset.py \
        --output "${EG_DIR}/data/token-norm-dist-no-lora.json" \
        --tokenizer ${TOKENIZER} \
        token-norm-dist \
        --num-requests ${NUM_REQUESTS} \
        --input-mean ${ISL} --input-stdev 0 \
        --output-mean ${OSL} --output-stdev 0
    echo "Dataset prepared"
else
    echo "Dataset already exists, skipping"
fi

echo "Running nsys profiling for Scenario 1 TP=${TP}..."
mkdir -p ${EG_DIR}/profiles/scenario1-no-lora-engine-tp-${TP}

# Run with nsys profiling
nsys profile \
    --trace=nvtx,cuda,osrt \
    --cuda-graph-trace=graph \
    --force-overwrite=true \
    --output=${EG_DIR}/profiles/scenario1-no-lora-engine-tp-${TP}/gpt_manager_scenario1_tp2 \
    mpirun -n ${TP} --allow-run-as-root --oversubscribe \
        cpp/build/benchmarks/gptManagerBenchmark \
        --engine_dir ${ENGINE_NO_LORA} \
        --type IFB \
        --dataset "${EG_DIR}/data/token-norm-dist-no-lora.json" \
        --kv_cache_free_gpu_mem_fraction 0.70 \
        --log_level info \
        --eos_id ${EOS_ID} \
        --streaming

echo "Scenario 1 profiling completed for TP=${TP}"
echo "Profile saved to: ${EG_DIR}/profiles/scenario1-no-lora-engine-tp-${TP}/gpt_manager_scenario1_tp2.nsys-rep"
echo ""
echo "To view the profile, run:"
echo "nsys-ui ${EG_DIR}/profiles/scenario1-no-lora-engine-tp-${TP}/gpt_manager_scenario1_tp2.nsys-rep"
