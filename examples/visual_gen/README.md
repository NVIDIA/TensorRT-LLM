# Visual Generation Examples

Quick reference for running visual generation models (WAN, FLUX2, LTX2).

## Prerequisites

```bash
# Install dependencies
pip install -r ${PROJECT_ROOT}/requirements-dev.txt
pip install git+https://github.com/huggingface/diffusers.git
pip install av
```

## Quick Start

```bash
# Set environment variables
export PROJECT_ROOT=/workspace/gitlab/tekit-b200
export MODEL_ROOT=/llm-models

# Run all examples (auto-detects GPUs)
cd examples/visual_gen
./visual_gen_examples.sh
```


## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | `/workspace/gitlab/tekit-b200` | Path to tekit repository |
| `MODEL_ROOT` | `/llm-models` | Path to model directory |
| `TLLM_LOG_LEVEL` | `INFO` | Logging level |

---

## WAN (Text-to-Video)

### Basic Usage

**Single GPU:**
```bash
python visual_gen_wan_t2v.py \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --output_path output.mp4
```

**With TeaCache:**
```bash
python visual_gen_wan_t2v.py \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --enable_teacache \
    --output_path output.mp4
```

### Multi-GPU Parallelism

WAN supports two parallelism modes that can be combined:
- **CFG Parallelism**: Split positive/negative prompts across GPUs
- **Ulysses Parallelism**: Split sequence across GPUs for longer sequences


**Ulysses Only (2 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --cfg_size 1 --ulysses_size 2 \
    --output_path output.mp4
```
GPU Layout: GPU 0-1 share sequence (6 heads each)

**CFG Only (2 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --cfg_size 2 --ulysses_size 1 \
    --output_path output.mp4
```
GPU Layout: GPU 0 (positive) | GPU 1 (negative)

**CFG + Ulysses (4 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --cfg_size 2 --ulysses_size 2 \
    --output_path output.mp4
```
GPU Layout: GPU 0-1 (positive, Ulysses) | GPU 2-3 (negative, Ulysses)

**Large-Scale (8 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --cfg_size 2 --ulysses_size 4 \
    --output_path output.mp4
```
GPU Layout: GPU 0-3 (positive) | GPU 4-7 (negative)

---

## Common Arguments

| Argument | WAN | FLUX2 | LTX2 | Default | Description |
|----------|-----|-------|------|---------|-------------|
| `--height` | ✓ | ✓ | ✓ | 720 | Output height |
| `--width` | ✓ | ✓ | ✓ | 1280 | Output width |
| `--num_frames` | ✓ | - | ✓ | 81 | Number of frames |
| `--steps` | ✓ | ✓ | ✓ | 50 | Denoising steps |
| `--guidance_scale` | ✓ | ✓ | ✓ | 5.0 | CFG guidance strength |
| `--seed` | ✓ | ✓ | ✓ | 42 | Random seed |
| `--enable_teacache` | ✓ | ✓ | - | False | Cache optimization |
| `--attention_backend` | ✓ | ✓ | - | VANILLA | VANILLA or TRTLLM |
| `--cfg_size` | ✓ | - | ✓ | 1 | CFG parallelism |
| `--ulysses_size` | ✓ | - | - | 1 | Sequence parallelism |
| `--linear_type` | ✓ | ✓ | ✓ | default | Quantization type |

## Troubleshooting

**Out of Memory:**
- Use quantization: `--linear_type trtllm-fp8-blockwise`
- Reduce resolution or frames
- Enable TeaCache: `--enable_teacache`
- Use Ulysses parallelism with more GPUs

**Slow Inference:**
- Enable TeaCache: `--enable_teacache`
- Use TRTLLM backend: `--attention_backend TRTLLM`
- Use multi-GPU: `--cfg_size 2` or `--ulysses_size 2`

**Import Errors:**
- Run from repository root
- Install necessary dependencies, e.g., `pip install -r requirements-dev.txt`

**Ulysses Errors:**
- `ulysses_size` must divide 12 (WAN heads)
- Total GPUs = `cfg_size × ulysses_size`
- Sequence length must be divisible by `ulysses_size`

## Output Formats

- **WAN**: `.mp4` (video), `.gif` (animated), `.png` (single frame)
- **FLUX2**: `.png`, `.jpg`
- **LTX2**: `.mp4` (with audio), `.gif` (video only), `.png` (single frame)

## Baseline Validation

Compare with official HuggingFace Diffusers implementation:

```bash
# Run HuggingFace baselines
./hf_examples.sh

# Or run individual models
python hf_wan.py --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers
```

Compare outputs with same seed for correctness verification.
