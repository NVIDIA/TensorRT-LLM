#!/bin/bash

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for LLM_MODELS_ROOT environment variable
if [ -z "$LLM_MODELS_ROOT" ]; then
    echo "❌ Error: LLM_MODELS_ROOT environment variable is not set!"
    echo "Please set the LLM_MODELS_ROOT environment variable before running this script."
    echo "For example, export LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models"
    exit 1
else
    echo "✅ LLM_MODELS_ROOT is set to: $LLM_MODELS_ROOT"
fi

echo "=========================================="
echo "Starting to run Python test files in Ray folder"
echo "=========================================="

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create timestamp-based log directory
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Initialize test results tracking
declare -A TEST_RESULTS
declare -A TEST_LOG_FILES

# Function: run Python file and record logs
run_python_file() {
    local file_path="$1"
    shift  # Remove first argument (file_path)
    local args="$@"  # Get remaining arguments
    local file_name=$(basename "$file_path" .py)

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$LOG_DIR/${file_name}_${timestamp}.log"

    # Create a unique test name with arguments
    local test_name="${file_name}"
    if [ $# -gt 0 ]; then
        test_name="${file_name}(${args// /_})"
    fi

    echo "Running: $file_path $args"
    echo "Log file: $log_file"
    echo "------------------------------------------"

    # Run Python file with arguments and record output
    python3 "$file_path" $args 2>&1 | tee "$log_file"

    # Check run result and store it
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ $file_name ran successfully"
        TEST_RESULTS["$test_name"]="PASS"
    else
        echo "❌ $file_name failed to run"
        TEST_RESULTS["$test_name"]="FAIL"
    fi
    TEST_LOG_FILES["$test_name"]="$log_file"
    echo ""
}

# Function: run disagg serving test
run_disagg_serving_test() {
    local disagg_dir="disagg_serving_local"
    local test_name="disagg_serving_test"

    # Store the original directory path for log files
    local original_dir=$(pwd)
    local log_file="$original_dir/$LOG_DIR/disagg_serving_test.log"
    local test_log_file="$original_dir/$LOG_DIR/disagg_serving_test_output.log"

    echo "Running disagg serving test in $disagg_dir"
    echo "Log file: $log_file"
    echo "------------------------------------------"

    # Check if disagg_serving_local directory exists
    if [ ! -d "$disagg_dir" ]; then
        echo "❌ $disagg_dir directory not found"
        TEST_RESULTS["$test_name"]="FAIL"
        TEST_LOG_FILES["$test_name"]="$log_file"
        return 1
    fi

    # Change to disagg_serving_local directory
    cd "$disagg_dir"

    # Start the serving in background
    echo "Starting disagg serving..."
    bash -e disagg_serving_local.sh --executor ray > "$log_file" 2>&1 &
    local serving_pid=$!

    # Wait for the serving to start (look for startup message)
    echo "Waiting for serving to start..."
    local max_wait=120
    local wait_count=0
    while [ $wait_count -lt $max_wait ]; do
        if grep -q "INFO:     Application startup complete." "$log_file" 2>/dev/null; then
            echo "✅ Serving started successfully"
            break
        fi
        sleep 2
        wait_count=$((wait_count + 2))
    done

    if [ $wait_count -ge $max_wait ]; then
        echo "❌ Timeout waiting for serving to start"
        kill $serving_pid 2>/dev/null
        cd "$original_dir"
        TEST_RESULTS["$test_name"]="FAIL"
        TEST_LOG_FILES["$test_name"]="$log_file"
        return 1
    fi

    # Wait a bit more for Uvicorn to be fully ready
    sleep 5

    # Run the test directly in the current script process
    echo ""
    echo "=========================================="
    echo "Serving is now running on http://localhost:8000"
    echo "Running disagg_serving_test.py in current process..."
    echo "=========================================="

    # Run the test directly and capture output
    echo "Running disagg_serving_test.py..."
    echo "Current directory: $(pwd)"
    echo "----------------------------------------"

    # Run the test and capture output
    python disagg_serving_test.py 2>&1 | tee "$test_log_file"
    local test_result=$?

    echo "----------------------------------------"
    echo "Test completed with exit code: $test_result"

    # Check test result and store it
    if [ $test_result -eq 0 ]; then
        echo "✅ disagg_serving_test ran successfully"
        TEST_RESULTS["$test_name"]="PASS"
    else
        echo "❌ disagg_serving_test failed with exit code $test_result"
        TEST_RESULTS["$test_name"]="FAIL"
    fi
    TEST_LOG_FILES["$test_name"]="$test_log_file"

    # Stop the serving
    echo "Stopping serving..."
    ray stop
    kill $serving_pid 2>/dev/null
    wait $serving_pid 2>/dev/null

    # Go back to original directory
    cd "$original_dir"

    echo "✅ disagg_serving_test completed"
    echo ""
}

# Function: get SM version using Python script
get_sm_version() {
    local sm_major=0
    local sm_minor=0
    local gpu_count=0

    # Check if the Python script exists
    if [ -f "current_node_gpu_type.py" ]; then
        # Run the Python script and capture output
        local python_output=$(python3 current_node_gpu_type.py 2>/dev/null)

        if [ $? -eq 0 ]; then
            # Parse the output which is in format "sm_major sm_minor gpu_count"
            # The Python script outputs space-separated values like "9 0 2"
            read sm_major sm_minor gpu_count <<< "$python_output"
        fi
    fi

    echo "$sm_major $sm_minor $gpu_count"
}

# Run Python files in ray folder
echo "Running Python files in ray folder:"
echo ""

read sm_major sm_minor gpu_count <<< $(get_sm_version)
echo "Detected GPU info:"
echo "  - SM version: $sm_major.$sm_minor"
echo "  - GPU count: $gpu_count"

# 1. simple_ray_single_node.py - Run with llama model and deepseek-v3-lite model
if [ -f "simple_ray_single_node.py" ]; then
    run_python_file "simple_ray_single_node.py" "--model_dir=TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Check SM version for DeepSeek-V3-Lite (only run on Hopper and Blackwell)
    if [ "$sm_major" -ge 9 ] && [ "$sm_minor" -ge 0 ]; then
        echo "✅ SM version $sm_major.$sm_minor is compatible with DeepSeek-V3-Lite (Hopper/Blackwell)"
        run_python_file "simple_ray_single_node.py" "--model_dir=$LLM_MODELS_ROOT/DeepSeek-V3-Lite/bf16"
    else
        echo "⚠️  Skipping DeepSeek-V3-Lite test - requires Hopper or Blackwell GPU (SM 9.0+), but detected: $sm_major.$sm_minor"
        echo "   DeepSeek-V3-Lite test was not run due to incompatible SM version"
        echo ""
        # Record this as a skipped test
        TEST_RESULTS["simple_ray_single_node(DeepSeek-V3-Lite)"]="SKIP"
        TEST_LOG_FILES["simple_ray_single_node(DeepSeek-V3-Lite)"]="N/A"
    fi
fi

# 2. rlhf_colocate.py - Check GPU count first
if [ -f "rlhf/rlhf_colocate.py" ]; then
    echo "Available GPUs: $gpu_count"

    if [ "$gpu_count" -ge 4 ]; then
        run_python_file "rlhf/rlhf_colocate.py"
    else
        echo "⚠️  Skipping rlhf_colocate.py - requires at least 4 GPUs, but only $gpu_count GPU(s) available"
        echo "   rlhf_colocate.py was not run due to insufficient GPU count (< 4)"
        echo ""
        # Record this as a skipped test
        TEST_RESULTS["rlhf_colocate"]="SKIP"
        TEST_LOG_FILES["rlhf_colocate"]="N/A"
    fi
fi

# 3. llm_inference_async_ray.py
if [ -f "llm_inference_async_ray.py" ]; then
    run_python_file "llm_inference_async_ray.py"
fi

# 4. test_update_weight_from_ipc.py
if [ -f "test_update_weight_from_ipc.py" ]; then
    run_python_file "test_update_weight_from_ipc.py"
fi

# Run MPI guarding tests
run_python_file "../llm_inference.py"

# Run disagg_serving_local test
echo "Running disagg_serving_local test:"
echo ""

run_disagg_serving_test

echo "=========================================="
echo "All tests completed!"
echo "Log files saved in: $LOG_DIR/"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Print test results summary
echo ""
echo "=========================================="
echo "TEST RESULTS SUMMARY"
echo "=========================================="


# Print each test result
for test_name in "${!TEST_RESULTS[@]}"; do
    result="${TEST_RESULTS[$test_name]}"
    log_file="${TEST_LOG_FILES[$test_name]}"

    if [ "$result" = "PASS" ]; then
        echo "✅ $test_name - PASS"
    elif [ "$result" = "SKIP" ]; then
        echo "⏭️  $test_name - SKIP"
    else
        echo "❌ $test_name - FAIL"
    fi
done

echo ""
echo "To view specific log file, run:"
echo "cat $SCRIPT_DIR/$LOG_DIR/[filename].log"
