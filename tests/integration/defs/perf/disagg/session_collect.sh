#!/bin/bash
# Session Collection Script
# Collects system information and TensorRT-LLM version

# Get parameters from environment or command line
INSTALL_MODE="${1:-none}"
REPO_DIR="${2:-.}"
WORK_DIR="${3:-.}"
OUTPUT_PATH="${4:-./output}"
WHEEL_PATH="${5:-}"

echo "=========================================="
echo "Session Collect Job Started"
echo "Time: $(date)"
echo "Install Mode: $INSTALL_MODE"
echo "=========================================="

# Step 1: Collect system information (no dependencies)
echo ""
echo "Step 1: Collecting system information..."
cd "$WORK_DIR"
python3 "$WORK_DIR/simple_collect.py" "$OUTPUT_PATH" 2>&1
echo "System information collection completed"

# Step 2: Handle different installation modes
echo ""
echo "Step 2: Installing TensorRT-LLM..."
if [ "$INSTALL_MODE" = "none" ]; then
    echo "Using built-in TensorRT-LLM, skipping installation"

elif [ "$INSTALL_MODE" = "wheel" ]; then
    echo "Installing TensorRT-LLM wheel..."
    echo "Wheel path pattern: $WHEEL_PATH"

    # Expand wildcard and install
    for wheel_file in $WHEEL_PATH; do
        if [ -f "$wheel_file" ]; then
            echo "Found wheel: $wheel_file"
            pip3 install "$wheel_file" 2>&1 || echo "Wheel install failed, continuing..."
            break
        fi
    done
    echo "Wheel installation completed"

elif [ "$INSTALL_MODE" = "source" ]; then
    echo "Installing TensorRT-LLM from source..."
    cd "$REPO_DIR"
    pip3 install -e . 2>&1 || echo "Source install failed, continuing..."
    echo "Source installation completed"

else
    echo "ERROR: Invalid install mode: $INSTALL_MODE"
    exit 1
fi

# Step 3: Collect TensorRT-LLM version information
echo ""
echo "Step 3: Collecting TensorRT-LLM version information..."
VERSION_FILE="$OUTPUT_PATH/trtllm_version.txt"
python3 -c "import tensorrt_llm; print(f'[TensorRT-LLM] TensorRT-LLM version: {tensorrt_llm.__version__}')" > "$VERSION_FILE" 2>&1 || echo "[TensorRT-LLM] TensorRT-LLM version: unknown" > "$VERSION_FILE"
echo "TensorRT-LLM version written to: $VERSION_FILE"

echo ""
echo "=========================================="
echo "Session Collect Job Completed"
echo "Time: $(date)"
echo "=========================================="

exit 0

