#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark script for MoE as Dense GEMM kernels (FC1 and FC2)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
M_CUSTOM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--m)
            M_CUSTOM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-m M_VALUES]"
            echo "  -m M_VALUES: Comma-separated list of m values (e.g., '1,4,16,64')"
            exit 1
            ;;
    esac
done

# Output log files
FC1_LOG="${SCRIPT_DIR}/bench_fc1.log"

# Clear previous logs
> "$FC1_LOG"

# Common parameters
EXPERT_COUNT=256
ITERATIONS=50
WARMUP=10

# FC1 parameters
FC1_N=131072
FC1_K=7168
FC1_C_DTYPE="Float4E2M1FN"

# MMA tiler configurations (mma_tiler_mn:cluster_shape_mn)
# cluster_m = mma_m / 128, cluster_n = 1
FC1_MMA_CONFIGS=("128,128:1,1" "128,256:1,1" "256,256:2,1")

# Generate m values: 1, 2, 4, 8, 16, 32, 48, 64, 80, ..., 512
if [ -n "$M_CUSTOM" ]; then
    # Parse comma-separated m values
    IFS=',' read -ra M_VALUES <<< "$M_CUSTOM"
else
    # Use default m values
    M_VALUES=(1 2 4 8 16)
    for m in $(seq 32 16 512); do
        M_VALUES+=($m)
    done
fi

echo "Benchmarking FC1 kernels..."
echo "M values: ${M_VALUES[*]}"
echo "FC1 MMA configs: ${FC1_MMA_CONFIGS[*]}"
echo ""

# Benchmark FC1
echo "===== FC1 Benchmark =====" | tee -a "$FC1_LOG"
echo "N=${FC1_N}, K=${FC1_K}, expert_count=${EXPERT_COUNT}, c_dtype=${FC1_C_DTYPE}" | tee -a "$FC1_LOG"
echo "" | tee -a "$FC1_LOG"

for config in "${FC1_MMA_CONFIGS[@]}"; do
    mma_config="${config%%:*}"
    cluster_config="${config##*:}"
    echo "--- FC1 with mma_tiler_mn=${mma_config}, cluster_shape_mn=${cluster_config} ---" | tee -a "$FC1_LOG"
    for m in "${M_VALUES[@]}"; do
        echo "Running FC1 with M=${m}, mma_tiler_mn=${mma_config}, cluster_shape_mn=${cluster_config}..." | tee -a "$FC1_LOG"
        python3 "${SCRIPT_DIR}/run_moe_as_dense_gemm_fc1.py" \
            --mnkl "${m},${FC1_N},${FC1_K},1" \
            --expert_count ${EXPERT_COUNT} \
            --c_dtype ${FC1_C_DTYPE} \
            --mma_tiler_mn ${mma_config} \
            --cluster_shape_mn ${cluster_config} \
            --iterations ${ITERATIONS} \
            --warmup_iterations ${WARMUP} \
            --skip_ref_check \
            --use_cupti 2>&1 | tee -a "$FC1_LOG"
        echo "" | tee -a "$FC1_LOG"
    done
done

echo ""
echo "Benchmark complete!"
echo "FC1 results: $FC1_LOG"
