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

# Step 2: Collect TensorRT-LLM version information (only for none mode)
if [ "$INSTALL_MODE" = "none" ]; then
    echo ""
    echo "Step 2: Collecting TensorRT-LLM version information..."
    VERSION_FILE="$OUTPUT_PATH/trtllm_version.txt"
    python3 -c "import tensorrt_llm; print(f'[TensorRT-LLM] TensorRT-LLM version: {tensorrt_llm.__version__}')" > "$VERSION_FILE" 2>&1 || echo "[TensorRT-LLM] TensorRT-LLM version: unknown" > "$VERSION_FILE"
    echo "TensorRT-LLM version written to: $VERSION_FILE"
else
    echo ""
    echo "Step 2: Skipping TensorRT-LLM version collection (install_mode=$INSTALL_MODE)"
fi

echo ""
echo "=========================================="
echo "Session Collect Job Completed"
echo "Time: $(date)"
echo "=========================================="

exit 0
