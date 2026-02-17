#!/bin/bash
# HuggingFace Baseline Tests - Official Diffusers Implementation
#
# Usage:
#   export PROJECT_ROOT=/path/to/tekit
#   export MODEL_ROOT=/path/to/models
#   ./hf_examples.sh
#
# Or inline:
#   PROJECT_ROOT=/workspace/gitlab/tekit-b200 MODEL_ROOT=/llm-models ./hf_examples.sh

set -e  # Exit on error

# Environment variables with defaults
PROJECT_ROOT=${PROJECT_ROOT:-"/workspace/gitlab/tekit-b200"}
MODEL_ROOT=${MODEL_ROOT:-"/llm-models"}

# Log configuration
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-"INFO"}

echo "============================================"
echo "HuggingFace Diffusers Baseline Tests"
echo "============================================"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "MODEL_ROOT:   $MODEL_ROOT"
echo "LOG_LEVEL:    $TLLM_LOG_LEVEL"
echo ""
echo "Purpose: Establish baseline results using"
echo "         official diffusers implementations"
echo "============================================"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
MISSING_DEPS=""

if ! python -c "import diffusers" 2>/dev/null; then
    echo "❌ ERROR: diffusers not found"
    MISSING_DEPS="$MISSING_DEPS diffusers"
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "❌ ERROR: torch not found"
    MISSING_DEPS="$MISSING_DEPS torch"
fi

if [ -n "$MISSING_DEPS" ]; then
    echo ""
    echo "❌ Missing required dependencies:$MISSING_DEPS"
    echo "Install with: pip install$MISSING_DEPS"
    exit 1
fi

echo "✅ All required dependencies found"
echo ""

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU: $GPU_NAME"
else
    echo "⚠️  WARNING: nvidia-smi not found"
    echo "   Continuing with CPU (very slow!)"
    GPU_COUNT=0
fi
echo ""

# Create output directory (in current directory)
OUTPUT_DIR="./baseline_outputs"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR ($(pwd)/baseline_outputs)"
echo ""

#############################################
# WAN (Wan2.1) Baseline Test
#############################################

echo "============================================"
echo "1/1: WAN Baseline Test"
echo "============================================"
echo ""

WAN_MODEL="${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers/"
WAN_OUTPUT="${OUTPUT_DIR}/wan_baseline.gif"

if [ -d "$WAN_MODEL" ]; then
    echo "Testing WAN with official diffusers..."
    python ${PROJECT_ROOT}/examples/visual_gen/hf_wan.py \
        --model_path "$WAN_MODEL" \
        --output_path "$WAN_OUTPUT" \
        --prompt "A cute cat playing piano" \
        --height 480 \
        --width 832 \
        --num_frames 33 \
        --steps 50 \
        --guidance_scale 7.0 \
        --seed 42
    echo ""
    echo "✅ WAN baseline test completed"
    echo "   Output: $WAN_OUTPUT"
else
    echo "⚠️  SKIPPED: WAN model not found at $WAN_MODEL"
fi

echo ""

#############################################
# Summary
#############################################

echo "============================================"
echo "Baseline Tests Complete!"
echo "============================================"
echo ""
echo "Output files saved to: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "No outputs generated"
echo ""
echo "Next Steps:"
echo "  1. Verify outputs are correct (images/videos generated)"
echo "  2. Compare with custom implementation outputs"
echo "  3. Use these as reference/baseline for debugging"
echo ""
echo "Comparison command:"
echo "  diff -r $OUTPUT_DIR <custom_implementation_outputs>"
echo "============================================"
