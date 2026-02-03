#!/bin/bash

set -e  # Exit on any error

echo "=== LoRA TRT gptManagerBenchmark Multi-TP Benchmarking ==="
echo "=========================================================="

# ============================================================================
# Environment Setup (Global)
# ============================================================================
MODEL_CHECKPOINT=/lustre/fsw/portfolios/coreai/users/smor/models/Llama-3.1-8B-Instruct
SOURCE_LORA=/tmp/lora_adapter/lora-llama-3.1-nemoguard-8b-topic-control
CPP_LORA=/tmp/lora_adapter/lora-llama-3.1-nemoguard-8b-topic-control-cpp
EG_DIR=/tmp/lora-eg
TOKENIZER=/lustre/fsw/portfolios/coreai/users/smor/models/Llama-3.1-8B-Instruct

DTYPE=float16
ISL=100
OSL=1000
MAX_BATCH=1
NUM_LAYERS=32
MAX_LORA_RANK=8
NUM_LORA_MODS=7
EOS_ID=2
NUM_REQUESTS=20

# TP sizes to test
TP_SIZES=(1 2 4)

echo "Configuration:"
echo "  ISL/OSL: ${ISL}/${OSL}"
echo "  TP Sizes: ${TP_SIZES[@]}"
echo "  Requests: ${NUM_REQUESTS}"
echo ""

# ============================================================================
# One-time Setup
# ============================================================================
echo "=== One-time Setup ==="

# Download LoRA adapter (if not exists)
if [ ! -d "$SOURCE_LORA" ]; then
    echo "Downloading LoRA adapter..."
    mkdir -p /tmp/lora_adapter
    huggingface-cli download nvidia/llama-3.1-nemoguard-8b-topic-control --local-dir /tmp/lora_adapter/lora-llama-3.1-nemoguard-8b-topic-control
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
# Main Loop: For each TP size
# ============================================================================
for TP in ${TP_SIZES[@]}; do
    echo ""
    echo "###################################################################"
    echo "#                     STARTING TP=${TP} BENCHMARK                    #"
    echo "###################################################################"
    echo ""

    # TP-specific directories
    BASE_DIR="/tmp/lora-benchmark-tp${TP}"
    CONVERTED_CHECKPOINT="${BASE_DIR}/llama-3.1-8B-ckpt"
    CONVERTED_CHECKPOINT_LORA="${BASE_DIR}/llama-3.1-8B-ckpt-lora"
    ENGINE_NO_LORA="${BASE_DIR}/llama-3.1-8B-engine-no-lora"
    ENGINE_WITH_LORA="${BASE_DIR}/llama-3.1-8B-engine-with-lora"

    mkdir -p ${BASE_DIR}

    # ========================================================================
    # Scenario 1: Engine WITHOUT LoRA, Requests WITHOUT LoRA
    # ========================================================================
    echo "=== TP=${TP} - Scenario 1: Engine WITHOUT LoRA, Requests WITHOUT LoRA ==="

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

    echo "Running benchmark (no LoRA engine) for TP=${TP}..."
    mkdir -p ${EG_DIR}/logs/scenario1-no-lora-engine-tp-${TP}
    if [ ${TP} -eq 1 ]; then
        # Single GPU - no MPI
        cpp/build/benchmarks/gptManagerBenchmark \
            --engine_dir ${ENGINE_NO_LORA} \
            --type IFB \
            --dataset "${EG_DIR}/data/token-norm-dist-no-lora.json" \
            --kv_cache_free_gpu_mem_fraction 0.70 \
            --log_level info \
            --eos_id ${EOS_ID} \
            --streaming \
            > ${EG_DIR}/logs/scenario1-no-lora-engine-tp-${TP}/output.log 2>&1
    else
        # Multi-GPU with MPI
        mpirun -n ${TP} --allow-run-as-root --oversubscribe \
            --output-filename ${EG_DIR}/logs/scenario1-no-lora-engine-tp-${TP} \
            cpp/build/benchmarks/gptManagerBenchmark \
            --engine_dir ${ENGINE_NO_LORA} \
            --type IFB \
            --dataset "${EG_DIR}/data/token-norm-dist-no-lora.json" \
            --kv_cache_free_gpu_mem_fraction 0.70 \
            --log_level info \
            --eos_id ${EOS_ID} \
            --streaming
    fi
    echo "Scenario 1 completed for TP=${TP}"

    # ========================================================================
    # Scenario 2: Engine WITH LoRA, Requests WITHOUT LoRA
    # ========================================================================
    echo ""
    echo "=== TP=${TP} - Scenario 2: Engine WITH LoRA, Requests WITHOUT LoRA ==="

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

    echo "Running benchmark (LoRA engine, no LoRA requests) for TP=${TP}..."
    mkdir -p ${EG_DIR}/logs/scenario2-lora-engine-no-lora-requests-tp-${TP}
    if [ ${TP} -eq 1 ]; then
        # Single GPU - no MPI
        cpp/build/benchmarks/gptManagerBenchmark \
            --engine_dir ${ENGINE_WITH_LORA} \
            --type IFB \
            --dataset "${EG_DIR}/data/token-norm-dist-no-lora.json" \
            --lora_host_cache_bytes 8589934592 \
            --lora_num_device_mod_layers $(( 32 * NUM_LAYERS * NUM_LORA_MODS * MAX_LORA_RANK )) \
            --kv_cache_free_gpu_mem_fraction 0.70 \
            --log_level info \
            --eos_id ${EOS_ID} \
            --streaming \
            > ${EG_DIR}/logs/scenario2-lora-engine-no-lora-requests-tp-${TP}/output.log 2>&1
    else
        # Multi-GPU with MPI
        mpirun -n ${TP} --allow-run-as-root --oversubscribe \
            --output-filename ${EG_DIR}/logs/scenario2-lora-engine-no-lora-requests-tp-${TP} \
            cpp/build/benchmarks/gptManagerBenchmark \
            --engine_dir ${ENGINE_WITH_LORA} \
            --type IFB \
            --dataset "${EG_DIR}/data/token-norm-dist-no-lora.json" \
            --lora_host_cache_bytes 8589934592 \
            --lora_num_device_mod_layers $(( 32 * NUM_LAYERS * NUM_LORA_MODS * MAX_LORA_RANK )) \
            --kv_cache_free_gpu_mem_fraction 0.70 \
            --log_level info \
            --eos_id ${EOS_ID} \
            --streaming
    fi
    echo "Scenario 2 completed for TP=${TP}"

    # ========================================================================
    # Scenario 3: Engine WITH LoRA, Requests WITH LoRA
    # ========================================================================
    echo ""
    echo "=== TP=${TP} - Scenario 3: Engine WITH LoRA, Requests WITH LoRA ==="

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

    echo "Running benchmark (LoRA engine, LoRA requests) for TP=${TP}..."
    mkdir -p ${EG_DIR}/logs/scenario3-lora-engine-lora-requests-tp-${TP}
    if [ ${TP} -eq 1 ]; then
        # Single GPU - no MPI
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
            --streaming \
            > ${EG_DIR}/logs/scenario3-lora-engine-lora-requests-tp-${TP}/output.log 2>&1
    else
        # Multi-GPU with MPI
        mpirun -n ${TP} --allow-run-as-root --oversubscribe \
            --output-filename ${EG_DIR}/logs/scenario3-lora-engine-lora-requests-tp-${TP} \
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
    fi
    echo "Scenario 3 completed for TP=${TP}"

    echo ""
    echo "###################################################################"
    echo "#                   COMPLETED TP=${TP} BENCHMARK                     #"
    echo "###################################################################"
    echo ""

done

# ============================================================================
# Results Summary
# ============================================================================
echo ""
echo "###################################################################"
echo "#                        RESULTS SUMMARY                         #"
echo "###################################################################"
echo ""

parse_results() {
    local tp=$1
    local scenario=$2
    local log_path="$3"

    echo "TP=${tp} - ${scenario}:"

    if [ ${tp} -eq 1 ]; then
        # Single GPU logs
        if [ -f "${log_path}/output.log" ]; then
            grep -E "(token_throughput|seq_throughput|total_latency|avg_sequence_latency|avg_time_to_first_token|avg_inter_token_latency)" "${log_path}/output.log" 2>/dev/null || echo "  No metrics found"
        else
            echo "  Log file not found"
        fi
    else
        # Multi-GPU MPI logs
        if [ -d "${log_path}" ]; then
            find ${log_path} -name "stdout" -exec grep -E "(token_throughput|seq_throughput|total_latency|avg_sequence_latency|avg_time_to_first_token|avg_inter_token_latency)" {} \; 2>/dev/null || echo "  No metrics found"
        else
            echo "  Log directory not found"
        fi
    fi
    echo ""
}

for TP in ${TP_SIZES[@]}; do
    echo "========== TP=${TP} Results =========="
    parse_results ${TP} "Scenario 1 (No LoRA Engine)" "${EG_DIR}/logs/scenario1-no-lora-engine-tp-${TP}"
    parse_results ${TP} "Scenario 2 (LoRA Engine, No LoRA Requests)" "${EG_DIR}/logs/scenario2-lora-engine-no-lora-requests-tp-${TP}"
    parse_results ${TP} "Scenario 3 (LoRA Engine, LoRA Requests)" "${EG_DIR}/logs/scenario3-lora-engine-lora-requests-tp-${TP}"
done

echo ""
echo "###################################################################"
echo "#                    BENCHMARKING COMPLETE!                      #"
echo "###################################################################"
echo ""
echo "All results saved in: ${EG_DIR}/logs/"
echo "Detailed logs available for each TP and scenario combination."
