#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is used to test all features and configuration options in the examples/ directory

set +e  # Do not exit immediately when encountering errors

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test configuration
# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Calculate the absolute path of the examples directory
EXAMPLES_DIR="$SCRIPT_DIR/../examples"
TEST_OUTPUT_DIR="$SCRIPT_DIR/test_outputs"
FAST_MODE=${FAST_MODE:-false}  # Environment variable to control fast mode

# Create output directory
mkdir -p "$TEST_OUTPUT_DIR"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run tests
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_output="$3"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    log_info "Running test: $test_name"
    echo "Command: $command"

    # Record start time
    start_time=$(date +%s)

    # Run command and capture output
    if eval "$command" > "$TEST_OUTPUT_DIR/$test_name.log" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        # Check if output file exists
        if [ -n "$expected_output" ] && [ -f "$expected_output" ]; then
            log_success "Test passed: $test_name (${duration}s)"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        elif [ -z "$expected_output" ]; then
            log_success "Test passed: $test_name (${duration}s)"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            log_error "Test failed: $test_name - Expected output not found"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        log_error "Test failed: $test_name - Command execution failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        # Show last few lines of error log
        echo "Last 10 lines of error log:"
        tail -n 10 "$TEST_OUTPUT_DIR/$test_name.log"
    fi

    echo "---"
}

# Check GPU count
check_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
        log_info "Detected $gpu_count GPU(s)"
        return $gpu_count
    else
        log_warning "nvidia-smi not found, assuming 1 GPU"
        return 1
    fi
}

# Get GPU count
check_gpu_count
GPU_COUNT=$?

log_info "Starting Examples comprehensive test suite"
log_info "Fast mode: $FAST_MODE"
log_info "Available GPUs: $GPU_COUNT"

# Basic parameter configuration
if [ "$FAST_MODE" = true ]; then
    # Wan2.1 T2V 1.3B
    WAN_2_1_T2V="--model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --num_frames 9 --num_inference_steps 2 --num_warmup_steps 1"
    # Wan2.1 I2V 14B 480P
    WAN_2_1_I2V="--model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --num_frames 9 --num_inference_steps 2 --num_warmup_steps 1"
    # Wan2.2 T2V A14B
    WAN_2_2_T2V="--model_path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --num_frames 65 --height 480 --width 832 --num_inference_steps 2 --num_warmup_steps 1"
    # Wan2.2 I2V A14B
    WAN_2_2_I2V="--model_path Wan-AI/Wan2.2-I2V-A14B-Diffusers  --num_frames 65 --disable_parallel_vae --num_inference_steps 2 --num_warmup_steps 1"
    # Flux
    FLUX="--model_path black-forest-labs/FLUX.1-dev --num_inference_steps 2 --num_warmup_steps 1"
else
    # Wan2.1 T2V 14B
    WAN_2_1_T2V="--model_path Wan-AI/Wan2.1-T2V-14B-Diffusers --num_frames 81 --height 720 --width 1280 --num_inference_steps 50"
    # Wan2.1 I2V 14B 720P
    WAN_2_1_I2V="--model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --num_inference_steps 50"
    # Wan2.2 T2V A14B
    WAN_2_2_T2V="--model_path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --num_frames 65 --height 480 --width 832"
    # Wan2.2 I2V A14B
    WAN_2_2_I2V="--model_path Wan-AI/Wan2.2-I2V-A14B-Diffusers  --num_frames 65 --disable_parallel_vae"
    # Flux
    FLUX="--model_path black-forest-labs/FLUX.1-dev --num_inference_steps 50"
fi

# # 2. Multi-GPU parallel tests
if [ $GPU_COUNT -gt 1 ]; then
    log_info "=== 2. Multi-GPU Parallelization Tests ==="

    ## Flux
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="flux_ulysses_parallel_4gpu"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc-per-node 4 flux.py $FLUX  --linear_type trtllm-fp8-blockwise --ulysses 4 --enable_async_cpu_offload --enable_teacache  --output_path $TEST_OUTPUT_DIR/$TEST_NAME.png" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.png"

        TEST_NAME="flux_ulysses_parallel_4gpu_int8_ulysses"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc-per-node 4 flux.py $FLUX  --linear_type trtllm-fp8-blockwise --ulysses 4 --int8_ulysses --enable_async_cpu_offload --enable_teacache  --output_path $TEST_OUTPUT_DIR/$TEST_NAME.png" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.png"

        TEST_NAME="flux_ulysses_parallel_4gpu_int8_fuseqkv_ulysses"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc-per-node 4 flux.py $FLUX  --linear_type trtllm-fp8-blockwise --ulysses 4 --int8_ulysses --fuse_qkv_in_ulysses --enable_async_cpu_offload --enable_teacache  --output_path $TEST_OUTPUT_DIR/$TEST_NAME.png" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.png"
    fi

    ## Wan2.2
    # Wan2.2 T2V A14B
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.2_t2v_a14b_ring_ulysses_parallel_4gpu"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_t2v.py $WAN_2_2_T2V --ulysses 2 --cfg 2 --linear_type trtllm-fp8-blockwise --enable_async_cpu_offload  --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # Wan2.2 I2V A14B
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.2_i2v_a14b_ring_ulysses_parallel_4gpu"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_2_I2V --ulysses 2 --cfg 2 --linear_type trtllm-fp8-blockwise --enable_async_cpu_offload  --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    ## Wan2.1
    # CFG Parallelism (2 GPUs)
    if [ $GPU_COUNT -ge 2 ]; then
        TEST_NAME="wan2.1_i2v_cfg_parallel_2gpu_test"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=2 wan_i2v.py $WAN_2_1_I2V --cfg 2 --attn_type default --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # Ulysses Parallelism (4 GPUs)
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_ulysses_parallel_4gpu"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --ulysses 4 --attn_type default --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ulysses Parallelism (4 GPUs)
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ulysses_parallel_4gpu"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ulysses 2 --attn_type default --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ulysses Parallelism (4 GPUs) + trtllm attention
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ulysses_parallel_4gpu_trtllm_attn"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ulysses 2 --attn_type trtllm-attn --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ulysses Parallelism (4 GPUs) + sage attention
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ulysses_parallel_4gpu_sage_attn"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ulysses 2 --attn_type sage-attn --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ulysses Parallelism (4 GPUs) + sage attention + FP8 Per-tensor Linear
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ulysses_parallel_4gpu_sage_attn_fp8_per_tensor"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ulysses 2 --attn_type sage-attn --linear_type trtllm-fp8-per-tensor --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ulysses Parallelism (4 GPUs) + sage attention + FP8 Blockwise Linear
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ulysses_parallel_4gpu_sage_attn_fp8_blockwise"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ulysses 2 --attn_type sage-attn --linear_type trtllm-fp8-blockwise --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ulysses Parallelism (4 GPUs) + sage attention + FP8 Blockwise Linear + TeaCache
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ulysses_parallel_4gpu_sage_attn_fp8_blockwise_teacache"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ulysses 2 --enable_teacache --attn_type sage-attn --linear_type trtllm-fp8-blockwise --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ulysses Parallelism (4 GPUs) + sage attention + FP8 Blockwise Linear + TeaCache + Torch Compile
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ulysses_parallel_4gpu_sage_attn_fp8_blockwise_teacache_torch_compile"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ulysses 2 --enable_teacache --attn_type sage-attn --linear_type trtllm-fp8-blockwise  --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # CFG + Ring Parallelism + sage attention(4 GPUs)
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ring_parallel_4gpu"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --cfg 2 --ring 2 --attn_type sage-attn --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

    # Ring Parallelism + ulysses(4 GPUs)
    if [ $GPU_COUNT -ge 4 ]; then
        TEST_NAME="wan2.1_i2v_cfg_ring_ulysses_parallel"
        run_test "$TEST_NAME" \
            "cd $EXAMPLES_DIR && torchrun --nproc_per_node=4 wan_i2v.py $WAN_2_1_I2V --ring 2 --ulysses 2 --attn_type sage-attn --output_path $TEST_OUTPUT_DIR/$TEST_NAME.mp4" \
            "$TEST_OUTPUT_DIR/$TEST_NAME.mp4"
    fi

else
    log_warning "Skipping multi-GPU tests (only 1 GPU available)"
fi

# Test result summary
log_info "=== Test Summary ==="
echo "======================================"
echo "Total tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo "Success rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo "======================================"

if [ $FAILED_TESTS -eq 0 ]; then
    log_success "All tests passed! 🎉"
    exit 0
else
    log_error "$FAILED_TESTS tests failed. Check logs in $TEST_OUTPUT_DIR/"

    # Show failed tests
    echo "Failed tests:"
    for log_file in "$TEST_OUTPUT_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            test_name=$(basename "$log_file" .log)
            if grep -q "ERROR\|FAILED\|Exception" "$log_file"; then
                echo "  - $test_name"
            fi
        fi
    done

    exit 1
fi
