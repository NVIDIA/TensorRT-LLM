# Visual Generation Examples

Quick reference for running visual generation models (FLUX, WAN).

## Prerequisites

```bash
# Install dependencies (from repository root)
pip install -r requirements-dev.txt
pip install git+https://github.com/huggingface/diffusers.git
pip install av
```

## Quick Start

```bash
# Set MODEL_ROOT to your model directory (required for examples)
export MODEL_ROOT=/llm-models
# Optional: PROJECT_ROOT defaults to repo root when run from examples/visual_gen

# Run all examples (auto-detects GPUs)
cd examples/visual_gen
./visual_gen_examples.sh
```


## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | Auto-detected | Path to repository root (set when running from `examples/visual_gen`) |
| `MODEL_ROOT` | `/llm-models` | Path to model directory |
| `TLLM_LOG_LEVEL` | `INFO` | Logging level |

---

## FLUX (Text-to-Image)

Supports both FLUX.1-dev and FLUX.2-dev. The pipeline type is auto-detected from the model checkpoint (`model_index.json`).

### Basic Usage

**FLUX.1-dev:**
```bash
python visual_gen_flux.py \
    --model_path ${MODEL_ROOT}/FLUX.1-dev \
    --prompt "A cat sitting on a windowsill" \
    --guidance_scale 3.5 \
    --output_path output.png
```

**FLUX.2-dev:**
```bash
python visual_gen_flux.py \
    --model_path ${MODEL_ROOT}/FLUX.2-dev \
    --prompt "A cat sitting on a windowsill" \
    --guidance_scale 4.0 \
    --output_path output.png
```

**With FP8 Quantization:**
```bash
python visual_gen_flux.py \
    --model_path ${MODEL_ROOT}/FLUX.2-dev \
    --prompt "A cat sitting on a windowsill" \
    --linear_type trtllm-fp8-per-tensor \
    --output_path output.png
```

**With TeaCache:**
```bash
python visual_gen_flux.py \
    --model_path ${MODEL_ROOT}/FLUX.1-dev \
    --prompt "A cat sitting on a windowsill" \
    --enable_teacache \
    --output_path output.png
```

### Batch Mode

Generate multiple images from a prompts file (one prompt per line):

```bash
python visual_gen_flux.py \
    --model_path ${MODEL_ROOT}/FLUX.1-dev \
    --prompts_file prompts.txt \
    --output_dir results/bf16/ \
    --seed 42
```

```bash
# With FP8 quantization
python visual_gen_flux.py \
    --model_path ${MODEL_ROOT}/FLUX.2-dev \
    --prompts_file prompts.txt \
    --output_dir results/fp8/ \
    --linear_type trtllm-fp8-per-tensor
```

Images are saved as `00.png`, `01.png`, etc. with a `timing.json` summary.

### Multi-GPU Parallelism

FLUX supports CFG and Ulysses parallelism, same as WAN.

**CFG + Ulysses (4 GPUs):**
```bash
python visual_gen_flux.py \
    --model_path ${MODEL_ROOT}/FLUX.1-dev \
    --prompts_file prompts.txt \
    --output_dir results/ \
    --cfg_size 2 --ulysses_size 2
```

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

| Argument | FLUX | WAN | Default | Description |
|----------|------|-----|---------|-------------|
| `--height` | ✓ | ✓ | 1024 / 720 | Output height |
| `--width` | ✓ | ✓ | 1024 / 1280 | Output width |
| `--num_frames` | | ✓ | 81 | Number of frames |
| `--steps` | ✓ | ✓ | 50 | Denoising steps |
| `--guidance_scale` | ✓ | ✓ | 3.5 / 5.0 | CFG guidance strength |
| `--seed` | ✓ | ✓ | 42 | Random seed |
| `--enable_teacache` | ✓ | ✓ | False | Cache optimization |
| `--teacache_thresh` | ✓ | ✓ | 0.2 | TeaCache similarity threshold |
| `--attention_backend` | ✓ | ✓ | VANILLA | VANILLA or TRTLLM |
| `--cfg_size` | ✓ | ✓ | 1 | CFG parallelism |
| `--ulysses_size` | ✓ | ✓ | 1 | Sequence parallelism |
| `--linear_type` | ✓ | ✓ | default | Quantization type |
| `--prompts_file` | ✓ | | — | Batch mode prompts file |
| `--output_dir` | ✓ | | — | Batch mode output directory |
| `--disable_torch_compile` | ✓ | ✓ | False | Disable torch.compile |

## Troubleshooting

**Out of Memory:**
- Use quantization: `--linear_type trtllm-fp8-blockwise` (WAN) or `--linear_type trtllm-fp8-per-tensor` (FLUX)
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
- `ulysses_size` must divide the model's head count (12 for WAN)
- Total GPUs = `cfg_size × ulysses_size`
- Sequence length must be divisible by `ulysses_size`

## Output Formats

- **FLUX**: `.png` (image)
- **WAN**: `.mp4` (video), `.gif` (animated), `.png` (single frame)

## Baseline Validation

Compare with official HuggingFace Diffusers implementation:

```bash
# Run HuggingFace baselines
./hf_examples.sh

# Or run individual models
python hf_wan.py --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers
```

Compare outputs with same seed for correctness verification.
