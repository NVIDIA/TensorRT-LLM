#!/bin/bash

export PATH=$PATH:/home/dev_user/.local/bin/

cd /home/scratch.timothyg_gpu/TensorRT-LLM

# Configure which test to run
# Options: "all", "fp8" (only FP8 tests), "nvfp4" (only NVFP4 tests), or specific test names
TEST_TO_RUN="${1:-all}"  # First argument, defaults to "all"

OUTPUT_DIR="/home/scratch.timothyg_gpu/TensorRT-LLM/tests/integration/defs/accuracy/output"
BASE_TEST="tests/integration/defs/accuracy/test_disaggregated_serving.py::TestLlama3_1_8BInstruct::test_mixed_precision_configs"

# Define all test combinations (6 configs × 2 benchmarks = 12 tests)
ALL_TESTS=(
    "Ctx16_Gen16-MMLU"
    "Ctx16_Gen16-GSM8K"
    "Ctx16_Gen8-MMLU"
    "Ctx16_Gen8-GSM8K"
    "Ctx8_Gen8-MMLU"
    "Ctx8_Gen8-GSM8K"
    "Ctx16_GenNVFP4-MMLU"
    "Ctx16_GenNVFP4-GSM8K"
    "CtxNVFP4_Gen16-MMLU"
    "CtxNVFP4_Gen16-GSM8K"
    "CtxNVFP4_GenNVFP4-MMLU"
    "CtxNVFP4_GenNVFP4-GSM8K"
)

# Define subsets of tests
FP8_TESTS=(
    "Ctx16_Gen16-MMLU"
    "Ctx16_Gen16-GSM8K"
    "Ctx16_Gen8-MMLU"
    "Ctx16_Gen8-GSM8K"
    "Ctx8_Gen8-MMLU"
    "Ctx8_Gen8-GSM8K"
)

NVFP4_TESTS=(
    "Ctx16_GenNVFP4-MMLU"
    "Ctx16_GenNVFP4-GSM8K"
    "CtxNVFP4_Gen16-MMLU"
    "CtxNVFP4_Gen16-GSM8K"
    "CtxNVFP4_GenNVFP4-MMLU"
    "CtxNVFP4_GenNVFP4-GSM8K"
)

run_single_test() {
    local test_name=$1
    echo "=========================================="
    echo "Starting test: $test_name"
    echo "=========================================="
    
    local log_file="${OUTPUT_DIR}/log_${test_name}_$(date +%Y%m%d_%H%M%S).log"
    # Run pytest and allow failures without breaking the loop.
    # Capture the *first* element of PIPESTATUS which is pytest's real exit code.
    pytest "${BASE_TEST}[${test_name}]" -v -s 2>&1 | tee "$log_file" || true
    local exit_code=${PIPESTATUS[0]}
    
    echo ""
    echo "Test $test_name completed with exit code: $exit_code"
    echo "LOG LOG LOG LOG LOG Log saved to: $log_file"
    echo ""
    
    # Additional safety margin for TCP TIME_WAIT states to clear
    # (Test already waits 30s internally, this adds 20s more = 50s total)
    echo "Waiting 20 seconds for TCP ports to fully release before next test..."
    sleep 40
    echo ""
    
    return $exit_code
}

# Main execution
if [ "$TEST_TO_RUN" == "all" ]; then
    echo "Running all 12 tests sequentially..."
    echo "Total: 6 configurations × 2 benchmarks (MMLU, GSM8K)"
    echo ""
    for test in "${ALL_TESTS[@]}"; do
        run_single_test "$test"
    done
    echo "All tests completed!"
    
elif [ "$TEST_TO_RUN" == "fp8" ]; then
    echo "Running FP8 tests only (6 tests)..."
    echo ""
    for test in "${FP8_TESTS[@]}"; do
        run_single_test "$test"
    done
    echo "FP8 tests completed!"
    
elif [ "$TEST_TO_RUN" == "nvfp4" ]; then
    echo "Running NVFP4 tests only (6 tests)..."
    echo ""
    for test in "${NVFP4_TESTS[@]}"; do
        run_single_test "$test"
    done
    echo "NVFP4 tests completed!"
    
else
    # Check if the specified test is valid
    # if [[ " ${ALL_TESTS[@]} " =~ " ${TEST_TO_RUN} " ]]; then
    run_single_test "$TEST_TO_RUN"
    # else
    #     echo "Error: Invalid test name: $TEST_TO_RUN"
    #     echo "Valid options are:"
    #     echo "  - all          (run all 12 tests sequentially)"
    #     echo "  - fp8          (run only FP8 tests: 6 tests)"
    #     echo "  - nvfp4        (run only NVFP4 tests: 6 tests)"
    #     echo ""
    #     echo "Individual tests:"
    #     for test in "${ALL_TESTS[@]}"; do
    #         echo "  - $test"
    #     done
    #     exit 1
    # fi
fi