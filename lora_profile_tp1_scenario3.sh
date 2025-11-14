#!/bin/bash

set -e  # Exit on any error

echo "=== LoRA TRT gptManagerBenchmark TP=1 Scenario 3 Profiling ==="
echo "=============================================================="
echo "Scenario 3: Engine WITH LoRA, Requests WITH LoRA"
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

# Fixed TP=1
TP=1

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
# TP=1 Scenario 3: Engine WITH LoRA, Requests WITH LoRA
# ============================================================================
echo "=== TP=${TP} - Scenario 3: Engine WITH LoRA, Requests WITH LoRA ==="

# TP-specific directories
BASE_DIR="/lustre/fsw/portfolios/coreai/users/smor/models/lora-benchmark-tp${TP}"
CONVERTED_CHECKPOINT_LORA="${BASE_DIR}/llama-3.1-8B-ckpt-lora"
ENGINE_WITH_LORA="${BASE_DIR}/llama-3.1-8B-engine-with-lora"

mkdir -p ${BASE_DIR}

echo "Converting checkpoint (with LoRA) for TP=${TP}..."
if [ ! -d "${CONVERTED_CHECKPOINT_LORA}" ]; then
    python examples/models/core/llama/convert_checkpoint.py \
        --model_dir ${MODEL_CHECKPOINT} \
        --output_dir ${CONVERTED_CHECKPOINT_LORA} \
        --dtype ${DTYPE} \
        --tp_size ${TP} \
        --pp_size 1
    echo "Checkpoint conversion completed"
else
    echo "Checkpoint already exists, skipping"
fi

echo "Building LoRA-enabled engine for TP=${TP}..."
if [ ! -d "${ENGINE_WITH_LORA}" ]; then
    trtllm-build \
        --checkpoint_dir ${CONVERTED_CHECKPOINT_LORA} \
        --output_dir ${ENGINE_WITH_LORA} \
        --max_batch_size ${MAX_BATCH} \
        --max_input_len ${ISL} \
        --max_seq_len $((${OSL}+${ISL})) \
        --gemm_plugin ${DTYPE} \
        --lora_plugin ${DTYPE} \
        --use_paged_context_fmha enable \
        --lora_target_modules attn_q attn_k attn_v attn_dense mlp_h_to_4h mlp_4h_to_h mlp_gate \
        --max_lora_rank ${MAX_LORA_RANK}
    echo "LoRA engine build completed"
else
    echo "LoRA engine already exists, skipping"
fi

echo "Generating random LoRA weights for TP=${TP}..."
if [ ! -d "${EG_DIR}/loras-tp${TP}" ]; then
    python benchmarks/cpp/utils/generate_rand_loras.py ${CPP_LORA} ${EG_DIR}/loras-tp${TP} 16
    echo "LoRA weights generated"
else
    echo "LoRA weights already exist, skipping"
fi

echo "Preparing dataset (with LoRA task IDs) for TP=${TP}..."
if [ ! -f "${EG_DIR}/data/token-norm-dist-lora-1-tp${TP}.json" ]; then
    python benchmarks/cpp/prepare_dataset.py \
        --output "${EG_DIR}/data/token-norm-dist-lora-1-tp${TP}.json" \
        --rand-task-id 0 0 \
        --tokenizer ${TOKENIZER} \
        token-norm-dist \
        --num-requests ${NUM_REQUESTS} \
        --input-mean ${ISL} --input-stdev 0 \
        --output-mean ${OSL} --output-stdev 0
    echo "LoRA dataset prepared"
else
    echo "LoRA dataset already exists, skipping"
fi

echo "Running nsys profiling for Scenario 3 TP=${TP}..."
mkdir -p ${EG_DIR}/profiles/scenario3-lora-engine-lora-requests-tp-${TP}

# Run with nsys profiling
nsys profile \
    --trace=nvtx,cuda \
    --cuda-graph-trace=graph \
    --force-overwrite=true \
    --output=${EG_DIR}/profiles/scenario3-lora-engine-lora-requests-tp-${TP}/gpt_manager_scenario3_tp1 \
    mpirun -n ${TP} --allow-run-as-root --oversubscribe \
        cpp/build/benchmarks/gptManagerBenchmark \
        --engine_dir ${ENGINE_WITH_LORA} \
        --type IFB \
        --dataset "${EG_DIR}/data/token-norm-dist-lora-1-tp${TP}.json" \
        --lora_host_cache_bytes 8589934592 \
        --lora_num_device_mod_layers $(( 16 * NUM_LAYERS * NUM_LORA_MODS * MAX_LORA_RANK )) \
        --kv_cache_free_gpu_mem_fraction 0.70 \
        --log_level info \
        --eos_id ${EOS_ID} \
        --lora_dir ${EG_DIR}/loras-tp${TP} \
        --streaming

echo "Scenario 3 profiling completed for TP=${TP}"
echo "Profile saved to: ${EG_DIR}/profiles/scenario3-lora-engine-lora-requests-tp-${TP}/gpt_manager_scenario3_tp1.nsys-rep"
echo ""
echo "To view the profile, run:"
echo "nsys-ui ${EG_DIR}/profiles/scenario3-lora-engine-lora-requests-tp-${TP}/gpt_manager_scenario3_tp1.nsys-rep"
