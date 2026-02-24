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
echo "1/3: WAN Baseline Test"
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
# FLUX.1 Baseline Test
#############################################

echo "============================================"
echo "2/3: FLUX.1 Baseline Test"
echo "============================================"
echo ""

FLUX1_MODEL="${MODEL_ROOT}/FLUX.1-dev/"
FLUX1_OUTPUT="${OUTPUT_DIR}/flux1_baseline.png"

if [ -d "$FLUX1_MODEL" ]; then
    echo "Testing FLUX.1 with official diffusers..."
    python ${PROJECT_ROOT}/examples/visual_gen/hf_flux.py \
        --model_path "$FLUX1_MODEL" \
        --output_path "$FLUX1_OUTPUT" \
        --prompt "A cat holding a sign that says hello world" \
        --height 1024 \
        --width 1024 \
        --steps 50 \
        --guidance_scale 3.5 \
        --seed 42
    echo ""
    echo "✅ FLUX.1 baseline test completed"
    echo "   Output: $FLUX1_OUTPUT"
else
    echo "⚠️  SKIPPED: FLUX.1 model not found at $FLUX1_MODEL"
fi

echo ""

#############################################
# FLUX.2 Baseline Test
#############################################

echo "============================================"
echo "3/3: FLUX.2 Baseline Test"
echo "============================================"
echo ""

FLUX2_MODEL="${MODEL_ROOT}/FLUX.2-dev/"
FLUX2_OUTPUT="${OUTPUT_DIR}/flux2_baseline.png"

if [ -d "$FLUX2_MODEL" ]; then
    echo "Testing FLUX.2 with official diffusers..."
    python ${PROJECT_ROOT}/examples/visual_gen/hf_flux2.py \
        --model_path "$FLUX2_MODEL" \
        --output_path "$FLUX2_OUTPUT" \
        --prompt "A cat holding a sign that says hello world" \
        --height 1024 \
        --width 1024 \
        --steps 50 \
        --guidance_scale 3.5 \
        --seed 42
    echo ""
    echo "✅ FLUX.2 baseline test completed"
    echo "   Output: $FLUX2_OUTPUT"
else
    echo "⚠️  SKIPPED: FLUX.2 model not found at $FLUX2_MODEL"
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
