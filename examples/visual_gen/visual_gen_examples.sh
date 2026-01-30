#!/bin/bash
# Visual Generation Examples - Test different models and configurations
#
# This script runs a comprehensive suite of visual generation examples including:
# - WAN T2V: Baseline, TeaCache, CFG parallelism, Ulysses parallelism, and combinations
# - WAN I2V: Baseline, TeaCache, CFG parallelism, Ulysses parallelism, and combinations
#
# The script automatically detects GPU count and runs appropriate examples:
# - 1 GPU:  Single-GPU examples only
# - 2 GPUs: + CFG parallelism, Ulysses parallelism
# - 4 GPUs: + CFG + Ulysses combined
# - 8 GPUs: + Large-scale high-resolution examples
#
# Usage:
#   export PROJECT_ROOT=/path/to/tekit
#   export MODEL_ROOT=/path/to/models
#   ./visual_gen_examples.sh
#
# Or inline:
#   PROJECT_ROOT=/workspace/gitlab/tekit-b200 MODEL_ROOT=/llm-models ./visual_gen_examples.sh

set -e  # Exit on error

# Environment variables with defaults
PROJECT_ROOT=${PROJECT_ROOT:-"/workspace/gitlab/tekit-b200"}
MODEL_ROOT=${MODEL_ROOT:-"/llm-models"}

# Log configuration
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-"INFO"}

echo "============================================"
echo "Visual Generation Examples"
echo "============================================"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "MODEL_ROOT:   $MODEL_ROOT"
echo "LOG_LEVEL:    $TLLM_LOG_LEVEL"
echo "============================================"
echo ""


# Detect GPU count
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo "Note: Multi-GPU examples will be skipped"
        SKIP_MULTI_GPU=1
    elif [ "$GPU_COUNT" -ge 8 ]; then
        echo "Note: Will run all examples including 8-GPU configurations"
    elif [ "$GPU_COUNT" -ge 4 ]; then
        echo "Note: Will run examples up to 4-GPU configurations"
    else
        echo "Note: Will run 2-GPU examples only"
    fi
else
    echo "WARNING: nvidia-smi not found. Assuming single GPU."
    GPU_COUNT=1
    SKIP_MULTI_GPU=1
fi
echo ""

#############################################
# WAN (Wan2.1) Text-to-Video Examples
#############################################
# Demonstrates:
# - Single GPU: Baseline and TeaCache
# - 2 GPUs: CFG only, Ulysses only
# - 4 GPUs: CFG + Ulysses combined
# - 8 GPUs: Large-scale parallelism
#############################################

echo "=== WAN Example 1: Baseline (no optimization) ==="
python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_t2v.py \
    --height 480 \
    --width 832 \
    --num_frames 33 \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers/ \
    --prompt "A cute cat playing piano" \
    --output_path wan_cat_piano.png

echo ""
echo "=== WAN Example 2: With TeaCache ==="
python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_t2v.py \
    --height 480 \
    --width 832 \
    --num_frames 33 \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --output_path wan_cat_piano_teacache.png \
    --enable_teacache

if [ -z "$SKIP_MULTI_GPU" ]; then
    echo ""
    echo "=== WAN Example 3: CFG Only (2 GPUs) ==="
    python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_t2v.py \
        --height 480 \
        --width 832 \
        --num_frames 33 \
        --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers/ \
        --prompt "A cute cat playing piano" \
        --output_path wan_cfg_2gpu.mp4 \
        --attention_backend TRTLLM \
        --cfg_size 2 \
        --ulysses_size 1
else
    echo ""
    echo "=== WAN Example 3: Skipped (requires 2 GPUs) ==="
fi

if [ -z "$SKIP_MULTI_GPU" ]; then
    echo ""
    echo "=== WAN Example 4: Ulysses Only (2 GPUs) ==="
    python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_t2v.py \
        --height 480 \
        --width 832 \
        --num_frames 33 \
        --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers/ \
        --prompt "A cute cat playing piano" \
        --output_path wan_ulysses_2gpu.mp4 \
        --attention_backend TRTLLM \
        --cfg_size 1 \
        --ulysses_size 2
else
    echo ""
    echo "=== WAN Example 4: Skipped (requires 2 GPUs) ==="
fi

if [ "$GPU_COUNT" -ge 4 ]; then
    echo ""
    echo "=== WAN Example 5: CFG + Ulysses (4 GPUs) ==="
    python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_t2v.py \
        --height 480 \
        --width 832 \
        --num_frames 33 \
        --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers/ \
        --prompt "A cute cat playing piano" \
        --output_path wan_cfg_ulysses_4gpu.mp4 \
        --attention_backend TRTLLM \
        --cfg_size 2 \
        --ulysses_size 2
else
    echo ""
    echo "=== WAN Example 5: Skipped (requires 4 GPUs) ==="
fi

if [ "$GPU_COUNT" -ge 8 ]; then
    echo ""
    echo "=== WAN Example 6: Large-Scale (8 GPUs) ==="
    python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_t2v.py \
        --height 480 \
        --width 832 \
        --num_frames 33 \
        --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers/ \
        --prompt "A cute cat playing piano" \
        --output_path wan_cfg_ulysses_8gpu.mp4 \
        --attention_backend TRTLLM \
        --cfg_size 2 \
        --ulysses_size 4
else
    echo ""
    echo "=== WAN Example 6: Skipped (requires 8 GPUs) ==="
fi

#############################################
# WAN 2.2 (Two-Stage) Text-to-Video Examples
#############################################

echo ""
echo "=== WAN 2.2 T2V Example: Two-stage with optimizations (FP8 + TRT-LLM + TeaCache) ==="
python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_t2v.py \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --model_path ${MODEL_ROOT}/Wan2.2-T2V-A14B-Diffusers \
    --prompt "A cute cat playing piano" \
    --output_path wan22_t2v_cat_piano_optimized.gif \
    --linear_type trtllm-fp8-blockwise \
    --attention_backend TRTLLM \
    --enable_teacache \
    --teacache_thresh 0.2 \
    --guidance_scale 3.0 \
    --guidance_scale_2 2.5 \
    --boundary_ratio 0.85

#############################################
# WAN 2.1 Image-to-Video Examples
#############################################

echo ""
echo "=== WAN 2.1 I2V Example: Single-stage with optimizations (FP8 + TRT-LLM + TeaCache) ==="
python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_i2v.py \
    --height 480 \
    --width 832 \
    --num_frames 33 \
    --model_path ${MODEL_ROOT}/Wan2.1-I2V-14B-480P-Diffusers \
    --image_path ${PROJECT_ROOT}/examples/visual_gen/cat_piano.png \
    --prompt "It snows as the cat plays piano, lots of snow \
    appearing all over the screen, snowflakes, blizzard,
    gradually more snow" \
    --negative_prompt "blurry, low quality" \
    --output_path wan21_i2v_cat_piano_optimized.gif \
    --linear_type trtllm-fp8-per-tensor \
    --attention_backend TRTLLM \
    --enable_teacache \
    --teacache_thresh 0.2 \
    --guidance_scale 6.0

#############################################
# WAN 2.2 (Two-Stage) Image-to-Video Examples
#############################################

echo ""
echo "=== WAN 2.2 I2V Example: Two-stage with optimizations (FP8 + TRT-LLM + TeaCache) ==="
python ${PROJECT_ROOT}/examples/visual_gen/visual_gen_wan_i2v.py \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --model_path ${MODEL_ROOT}/Wan2.2-I2V-A14B-Diffusers \
    --image_path ${PROJECT_ROOT}/examples/visual_gen/cat_piano.png \
    --prompt "It snows as the cat plays piano, lots of snow \
    appearing all over the screen, snowflakes, blizzard,
    gradually more snow" \
    --negative_prompt "blurry, low quality" \
    --output_path wan22_i2v_cat_piano_optimized.gif \
    --linear_type trtllm-fp8-blockwise \
    --attention_backend TRTLLM \
    --enable_teacache \
    --teacache_thres 0.2 \
    --guidance_scale 6.0 \
    --guidance_scale_2 5.0 \
    --boundary_ratio 0.85

echo ""
echo "============================================"
echo "All examples completed successfully!"
echo "============================================"
