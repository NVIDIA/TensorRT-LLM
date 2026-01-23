#!/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Benchmark NVFP4 MoE with multiple backends
#
# Usage:
#   ./run_benchmark.sh                    # Run all backends
#   ./run_benchmark.sh --backend CUTLASS  # Run specific backend
#   ./run_benchmark.sh --profile          # Enable nsys profiling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
BACKENDS=("CUTLASS" "TRTLLM" "DENSEGEMM")
SEQ_LEN=128
HIDDEN_SIZE=7168
INTERMEDIATE_SIZE=256
NUM_EXPERTS=256
TOP_K=8
ITERATIONS=100
WARMUP=10
ENABLE_CUDAGRAPH=true
PROFILE=false
OUTPUT_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKENDS=("$2")
            shift 2
            ;;
        --seq-len)
            SEQ_LEN=$2
            shift 2
            ;;
        --hidden-size)
            HIDDEN_SIZE=$2
            shift 2
            ;;
        --intermediate-size)
            INTERMEDIATE_SIZE=$2
            shift 2
            ;;
        --num-experts)
            NUM_EXPERTS=$2
            shift 2
            ;;
        --top-k)
            TOP_K=$2
            shift 2
            ;;
        --iterations)
            ITERATIONS=$2
            shift 2
            ;;
        --warmup)
            WARMUP=$2
            shift 2
            ;;
        --no-cudagraph)
            ENABLE_CUDAGRAPH=false
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --backend BACKEND       Run specific backend (CUTLASS, TRTLLM, DENSEGEMM)"
            echo "  --seq-len N             Sequence length (default: 128)"
            echo "  --hidden-size N         Hidden size (default: 7168)"
            echo "  --intermediate-size N   Intermediate size (default: 256)"
            echo "  --num-experts N         Number of experts (default: 256)"
            echo "  --top-k N               Top-K experts (default: 8)"
            echo "  --iterations N          Benchmark iterations (default: 100)"
            echo "  --warmup N              Warmup iterations (default: 10)"
            echo "  --no-cudagraph          Disable CUDA Graph"
            echo "  --profile               Enable nsys profiling"
            echo "  --output-dir DIR        Output directory for results (default: results)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build common arguments
COMMON_ARGS="--seq_len $SEQ_LEN"
COMMON_ARGS+=" --hidden_size $HIDDEN_SIZE"
COMMON_ARGS+=" --intermediate_size $INTERMEDIATE_SIZE"
COMMON_ARGS+=" --num_experts $NUM_EXPERTS"
COMMON_ARGS+=" --top_k $TOP_K"
COMMON_ARGS+=" --iterations $ITERATIONS"
COMMON_ARGS+=" --warmup $WARMUP"

if [ "$ENABLE_CUDAGRAPH" = true ]; then
    COMMON_ARGS+=" --enable_cudagraph"
fi

echo "============================================================"
echo "NVFP4 MoE Benchmark"
echo "============================================================"
echo "Backends: ${BACKENDS[*]}"
echo "Seq len: $SEQ_LEN, Hidden: $HIDDEN_SIZE, Intermediate: $INTERMEDIATE_SIZE"
echo "Experts: $NUM_EXPERTS, Top-K: $TOP_K"
echo "Iterations: $ITERATIONS, Warmup: $WARMUP"
echo "CUDA Graph: $ENABLE_CUDAGRAPH"
echo "Profile: $PROFILE"
echo "Output dir: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run benchmark for each backend
for BACKEND in "${BACKENDS[@]}"; do
    echo "------------------------------------------------------------"
    echo "Running benchmark: $BACKEND"
    echo "------------------------------------------------------------"

    REPORT_NAME="${OUTPUT_DIR}/${BACKEND}_seq${SEQ_LEN}"

    if [ "$PROFILE" = true ]; then
        # Run with nsys profiling
        # Use --cuda-graph-trace=node to trace CUDA Graph node executions
        nsys profile -t cuda,nvtx -o "$REPORT_NAME" \
            --force-overwrite true --export=sqlite \
            --cuda-graph-trace=node \
            python bench_nvfp4_moe.py --moe_backend "$BACKEND" $COMMON_ARGS \
            2>&1 | tee "${REPORT_NAME}.log"

        # Parse the report
        if [ -f "${REPORT_NAME}.sqlite" ]; then
            echo ""
            echo "Parsing nsys report..."
            python parse_nsys_report.py "${REPORT_NAME}.sqlite" | tee -a "${REPORT_NAME}.log"
        fi
    else
        # Run without profiling
        python bench_nvfp4_moe.py --moe_backend "$BACKEND" $COMMON_ARGS \
            2>&1 | tee "${REPORT_NAME}.log"
    fi

    echo ""
done

echo "============================================================"
echo "Benchmark completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
