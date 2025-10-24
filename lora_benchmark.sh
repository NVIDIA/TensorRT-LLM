#!/bin/bash

set -e  # Exit on any error

# =============================================================================
# LoRA Benchmarking Script for TensorRT-LLM
# Supports TP: 1, 4, 8 with fixed ISL/OSL: 128/128
# =============================================================================

# Configuration
MODEL_CHECKPOINT="/lustre/fsw/portfolios/coreai/users/smor/models/Llama-3.1-8B-Instruct"
TOKENIZER="/lustre/fsw/portfolios/coreai/users/smor/models/Llama-3.1-8B-Instruct"
SOURCE_LORA="/tmp/lora_adapter/lora-llama-3.1-nemoguard-8b-topic-control"

# Fixed parameters for 128:128
DTYPE="float16"
PP=1
MAX_INPUT_LEN=128
MAX_OUTPUT_LEN=128
MAX_SEQ_LEN=$((MAX_INPUT_LEN + MAX_OUTPUT_LEN))  # 256
MAX_BATCH=8
NUM_LAYERS=32
MAX_LORA_RANK=8
NUM_LORA_MODS=7
EOS_ID=2
NUM_REQUESTS=100

# TP configurations to test
TP_SIZES=(1 4 8)
NUM_LORAS=(1 2 4)

# Base directories
BASE_DIR="/tmp/lora-benchmark"
mkdir -p ${BASE_DIR}

echo "=== LoRA Benchmarking Script Started ==="
echo "ISL/OSL: ${MAX_INPUT_LEN}/${MAX_OUTPUT_LEN}"
echo "TP Sizes: ${TP_SIZES[@]}"
echo "Base Directory: ${BASE_DIR}"
echo "============================================="

# Function to run benchmark for specific TP size
run_benchmark_for_tp() {
    local TP=$1
    echo ""
    echo "=== Starting Benchmark for TP=${TP} ==="

    # TP-specific directories
    CONVERTED_CHECKPOINT="${BASE_DIR}/llama-3.1-8B-ckpt-tp${TP}"
    LORA_ENGINE="${BASE_DIR}/llama-3.1-8B-engine-tp${TP}"
    CPP_LORA="${BASE_DIR}/lora-cpp-tp${TP}"
    EG_DIR="${BASE_DIR}/results-tp${TP}"

    mkdir -p ${EG_DIR}/data ${EG_DIR}/logs

    echo "--- Step 1: Converting checkpoint for TP=${TP} ---"
    if [ ! -d "${CONVERTED_CHECKPOINT}" ]; then
        python examples/models/core/llama/convert_checkpoint.py \
            --model_dir ${MODEL_CHECKPOINT} \
            --output_dir ${CONVERTED_CHECKPOINT} \
            --dtype ${DTYPE} \
            --tp_size ${TP} \
            --pp_size ${PP}
        echo "Checkpoint conversion completed for TP=${TP}"
    else
        echo "Checkpoint already exists for TP=${TP}, skipping conversion"
    fi

    echo "--- Step 2: Building TensorRT engine for TP=${TP} ---"
    if [ ! -d "${LORA_ENGINE}" ]; then
        trtllm-build \
            --checkpoint_dir ${CONVERTED_CHECKPOINT} \
            --output_dir ${LORA_ENGINE} \
            --max_batch_size ${MAX_BATCH} \
            --max_input_len ${MAX_INPUT_LEN} \
            --max_seq_len ${MAX_SEQ_LEN} \
            --gemm_plugin ${DTYPE} \
            --lora_plugin ${DTYPE} \
            --use_paged_context_fmha enable \
            --lora_target_modules attn_q attn_k attn_v attn_dense mlp_h_to_4h mlp_4h_to_h mlp_gate \
            --max_lora_rank ${MAX_LORA_RANK} \
            --workers ${TP}
        echo "Engine build completed for TP=${TP}"
    else
        echo "Engine already exists for TP=${TP}, skipping build"
    fi

    echo "--- Step 3: Converting LoRA to C++ format ---"
    if [ ! -d "${CPP_LORA}" ]; then
        python examples/hf_lora_convert.py \
            -i ${SOURCE_LORA} \
            --storage-type ${DTYPE} \
            -o ${CPP_LORA}
        echo "LoRA conversion completed"
    else
        echo "LoRA already converted, skipping"
    fi

    echo "--- Step 4: Generating datasets (128:128) ---"
    # Base dataset without LoRA
    python benchmarks/cpp/prepare_dataset.py \
        --output "${EG_DIR}/data/token-norm-dist.json" \
        --tokenizer ${TOKENIZER} \
        token-norm-dist \
        --num-requests ${NUM_REQUESTS} \
        --input-mean ${MAX_INPUT_LEN} --input-stdev 0 \
        --output-mean ${MAX_OUTPUT_LEN} --output-stdev 0

    # LoRA datasets
    for nloras in ${NUM_LORAS[@]}; do
        python benchmarks/cpp/prepare_dataset.py \
            --output "${EG_DIR}/data/token-norm-dist-lora-${nloras}.json" \
            --rand-task-id 0 $(( nloras - 1 )) \
            --tokenizer ${TOKENIZER} \
            token-norm-dist \
            --num-requests ${NUM_REQUESTS} \
            --input-mean ${MAX_INPUT_LEN} --input-stdev 0 \
            --output-mean ${MAX_OUTPUT_LEN} --output-stdev 0
    done

    echo "--- Step 5: Generating random LoRA weights ---"
    python benchmarks/cpp/utils/generate_rand_loras.py ${CPP_LORA} ${EG_DIR}/loras 16

    echo "--- Step 6: Running benchmarks ---"

    # Baseline benchmark (no LoRA)
    echo "Running baseline benchmark (no LoRA) for TP=${TP}"
    mkdir -p ${EG_DIR}/logs/baseline
    if [ ${TP} -eq 1 ]; then
        cpp/build/benchmarks/gptManagerBenchmark \
            --engine_dir ${LORA_ENGINE} \
            --type IFB \
            --dataset "${EG_DIR}/data/token-norm-dist.json" \
            --lora_host_cache_bytes 8589934592 \
            --lora_num_device_mod_layers $(( 32 * NUM_LAYERS * NUM_LORA_MODS * MAX_LORA_RANK )) \
            --kv_cache_free_gpu_mem_fraction 0.70 \
            --log_level info \
            --eos_id ${EOS_ID} \
            > ${EG_DIR}/logs/baseline/output.log 2>&1
    else
        mpirun -n ${TP} --allow-run-as-root \
            --output-filename ${EG_DIR}/logs/baseline \
            cpp/build/benchmarks/gptManagerBenchmark \
            --engine_dir ${LORA_ENGINE} \
            --type IFB \
            --dataset "${EG_DIR}/data/token-norm-dist.json" \
            --lora_host_cache_bytes 8589934592 \
            --lora_num_device_mod_layers $(( 32 * NUM_LAYERS * NUM_LORA_MODS * MAX_LORA_RANK )) \
            --kv_cache_free_gpu_mem_fraction 0.70 \
            --log_level info \
            --eos_id ${EOS_ID}
    fi

    # LoRA benchmarks
    for nloras in ${NUM_LORAS[@]}; do
        echo "Running LoRA benchmark with ${nloras} adapters for TP=${TP}"
        mkdir -p ${EG_DIR}/logs/lora-${nloras}

        if [ ${TP} -eq 1 ]; then
            # Single GPU
            cpp/build/benchmarks/gptManagerBenchmark \
                --engine_dir ${LORA_ENGINE} \
                --type IFB \
                --dataset "${EG_DIR}/data/token-norm-dist-lora-${nloras}.json" \
                --streaming \
                --lora_host_cache_bytes 8589934592 \
                --lora_num_device_mod_layers $(( 16 * NUM_LAYERS * NUM_LORA_MODS * MAX_LORA_RANK )) \
                --kv_cache_free_gpu_mem_fraction 0.70 \
                --log_level info \
                --eos_id ${EOS_ID} \
                --lora_dir ${EG_DIR}/loras \
                > ${EG_DIR}/logs/lora-${nloras}/output.log 2>&1
        else
            # Multi-GPU with MPI
            mpirun -n ${TP} --allow-run-as-root \
                --output-filename ${EG_DIR}/logs/lora-${nloras} \
                cpp/build/benchmarks/gptManagerBenchmark \
                --engine_dir ${LORA_ENGINE} \
                --type IFB \
                --dataset "${EG_DIR}/data/token-norm-dist-lora-${nloras}.json" \
                --streaming \
                --lora_host_cache_bytes 8589934592 \
                --lora_num_device_mod_layers $(( 16 * NUM_LAYERS * NUM_LORA_MODS * MAX_LORA_RANK )) \
                --kv_cache_free_gpu_mem_fraction 0.70 \
                --log_level info \
                --eos_id ${EOS_ID} \
                --lora_dir ${EG_DIR}/loras
        fi
    done

    echo "=== Completed Benchmark for TP=${TP} ==="
    echo "Results saved to: ${EG_DIR}/logs/"
    echo ""
}

# Function to parse and display results
display_results() {
    echo ""
    echo "=== BENCHMARK RESULTS SUMMARY ==="
    echo "=================================="

    for TP in ${TP_SIZES[@]}; do
        EG_DIR="${BASE_DIR}/results-tp${TP}"
        echo ""
        echo "TP=${TP} Results:"
        echo "----------------"

        # Check baseline
        if [ -f "${EG_DIR}/logs/baseline/output.log" ]; then
            echo "  Baseline (no LoRA):"
            grep -E "(throughput|latency|TTFT)" ${EG_DIR}/logs/baseline/output.log | head -3 || echo "    No metrics found"
        fi

        # Check LoRA results
        for nloras in ${NUM_LORAS[@]}; do
            if [ -f "${EG_DIR}/logs/lora-${nloras}/output.log" ]; then
                echo "  LoRA-${nloras}:"
                grep -E "(throughput|latency|TTFT)" ${EG_DIR}/logs/lora-${nloras}/output.log | head -3 || echo "    No metrics found"
            elif [ -d "${EG_DIR}/logs/lora-${nloras}" ]; then
                echo "  LoRA-${nloras}:"
                find ${EG_DIR}/logs/lora-${nloras} -name "*.log" -exec grep -E "(throughput|latency|TTFT)" {} \; | head -3 || echo "    No metrics found"
            fi
        done
    done

    echo ""
    echo "=== Full logs available at: ${BASE_DIR}/results-tp*/logs/ ==="
}

# Main execution
main() {
    echo "Checking prerequisites..."

    # Check if model and lora paths exist
    if [ ! -d "${MODEL_CHECKPOINT}" ]; then
        echo "ERROR: Model checkpoint not found at ${MODEL_CHECKPOINT}"
        exit 1
    fi

    if [ ! -d "${SOURCE_LORA}" ]; then
        echo "ERROR: Source LoRA not found at ${SOURCE_LORA}"
        exit 1
    fi

    # Check if build directory exists
    if [ ! -f "cpp/build/benchmarks/gptManagerBenchmark" ]; then
        echo "ERROR: gptManagerBenchmark not found. Please build TensorRT-LLM first."
        exit 1
    fi

    echo "Prerequisites check passed!"

    # Run benchmarks for each TP size
    for TP in ${TP_SIZES[@]}; do
        run_benchmark_for_tp ${TP}
    done

    # Display results summary
    display_results

    echo ""
    echo "=== LoRA Benchmarking Complete! ==="
    echo "All results saved to: ${BASE_DIR}/"
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [TP_SIZE]"
    echo ""
    echo "Run LoRA benchmarking for TensorRT-LLM"
    echo ""
    echo "Arguments:"
    echo "  TP_SIZE    Optional. Single TP size to test (1, 4, or 8)"
    echo "             If not specified, tests all: 1, 4, 8"
    echo ""
    echo "Examples:"
    echo "  $0         # Test all TP sizes: 1, 4, 8"
    echo "  $0 4       # Test only TP=4"
    exit 0
fi

# If single TP specified, override array
if [ ! -z "$1" ]; then
    if [[ "$1" =~ ^[148]$ ]]; then
        TP_SIZES=($1)
        echo "Running benchmark for TP=${1} only"
    else
        echo "ERROR: Invalid TP size. Must be 1, 4, or 8"
        exit 1
    fi
fi

# Run main function
main
