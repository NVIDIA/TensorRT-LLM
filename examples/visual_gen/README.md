# Visual Generation Examples

Quick reference for running visual generation models.
Please refer to [the VisualGen doc](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html)
about the details of the feature.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| [`models/`](models/) | Per-model example scripts — slim API examples (~40 lines) that focus on model-specific request construction and output processing |
| [`configs/`](configs/) | YAML configs shared by offline examples (`--extra_visual_gen_options`) and `trtllm-serve` |
| [`serve/`](serve/) | `trtllm-serve` usage, benchmarking, and client examples |

## Quick Start

[`quickstart_example.py`](quickstart_example.py) — generate a video in ~30 lines (Wan T2V, 1 GPU).

## Per-Model Examples

Each script under `models/` demonstrates a single model with the VisualGen API.
Engine config (quantization, parallelism, TeaCache, etc.) is an optional YAML
file passed via `--extra_visual_gen_options` — the same flag that `trtllm-serve` uses.

```bash
# Default: 1 GPU, model defaults
python models/wan_t2v.py

# With a shared config for NVFP4 quantization
python models/wan_t2v.py --extra_visual_gen_options configs/wan2.2-t2v-fp4-1gpu.yaml
```

## Prerequisites

```bash
# Install dependencies (from repository root)
pip install -r requirements-dev.txt
```


## FLUX (Text-to-Image)

### Basic Usage

**FLUX.1:**

```bash
python visual_gen_flux.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --prompt "A cat sitting on a windowsill" \
    --height 1024 --width 1024 \
    --guidance_scale 3.5 \
    --output_path output.png
```

**With FP8 quantization:**

```bash
python visual_gen_flux.py \
    --model_path black-forest-labs/FLUX.2-dev \
    --prompt "A cat sitting on a windowsill" \
    --linear_type trtllm-fp8-per-tensor \
    --output_path output_fp8.png
```

**Batch mode (multiple prompts from file):**

```bash
python visual_gen_flux.py \
    --model_path black-forest-labs/FLUX.1-dev \
    --prompts_file prompts.txt \
    --output_dir results/ --seed 42
```


## WAN (Text-to-Video)

### Basic Usage

**Single GPU:**

```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --output_path output.mp4
```

**With SageAttention (FP8/INT8 per-block quantized attention):**
```bash
python visual_gen_wan_t2v.py \
    --model_path ${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --enable_sage_attention \
    --output_path output.mp4
```

**With TeaCache:**
```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --enable_teacache \
    --output_path output.mp4
```

### Multi-GPU Parallelism

WAN supports two parallelism modes that can be combined:
- **CFG Parallelism**: Split positive/negative prompts across GPUs
- **Sequence Parallelism**:
  - *Ulysses*: Split sequence along head dimension across GPUs; requires `ulysses_size` to divide the model's head count
  - *Attention2D*: 2D mesh sequence parallelism; no head-count constraint; requires `--attention_backend FA4`
  - Combining Ulysses and Attention2D is not yet supported


**Ulysses Only (2 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
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
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --cfg_size 2 --ulysses_size 1 \
    --output_path output.mp4
```
GPU Layout: GPU 0 (positive) | GPU 1 (negative)

**Attention2D Only (4 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend FA4 \
    --cfg_size 1 \
    --attn2d_row_size 2 --attn2d_col_size 2 \
    --output_path output.mp4
```
GPU Layout: Sequence equally split among GPU 0-3

**CFG + Ulysses (4 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --cfg_size 2 --ulysses_size 2 \
    --output_path output.mp4
```
GPU Layout: GPU 0-1 (positive, Ulysses) | GPU 2-3 (negative, Ulysses)

**CFG + Attention2D (8 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend FA4 \
    --cfg_size 2 \
    --attn2d_row_size 2 --attn2d_col_size 2 \
    --output_path output.mp4
```
GPU Layout: GPU 0-3 (positive, Attention2D) | GPU 4-7 (negative, Attention2D)

**Large-Scale (64 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 720 --width 1280 --num_frames 81 \
    --attention_backend FA4 \
    --attn2d_row_size 8 --attn2d_col_size 4 \
    --cfg_size 2 \
    --output_path output.mp4
```
GPU Layout: GPU 0-31 (positive, Attention2D 8×4) | GPU 32-63 (negative, Attention2D 8×4)


## WAN (Image-to-Video)

```bash
python visual_gen_wan_i2v.py \
    --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --image_path input_image.jpg \
    --prompt "She turns around and smiles" \
    --height 480 --width 832 --num_frames 81 \
    --output_path output_i2v.mp4
```


## LTX2 (Text/Image-to-Video with Audio)

LTX2 generates video **with audio** from text prompts or input images.
It uses a Gemma3 text encoder (provided separately via `--text_encoder_path`)
and supports BF16, FP8, and FP4 precision checkpoints.

Please refer to tensorrt_llm/_torch/visual_gen/models/ltx2/LTX_2_CHECKPOINT_FORMAT.md for model checkpoint info.

### Basic Usage

**Text-to-Video (single GPU):**
```bash
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --height 720 --width 1280 --num_frames 121 \
    --steps 40 --guidance_scale 4.0 --seed 42 \
    --output_path output_t2v.mp4
```

**Image-to-Video:**
```bash
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --image ${PROJECT_ROOT}/examples/visual_gen/cat_piano.png \
    --image_cond_strength 1.0 \
    --height 720 --width 1280 --num_frames 121 \
    --steps 40 --seed 42 \
    --output_path output_i2v.mp4
```

### Precision Variants

LTX2 ships checkpoints at three precision levels. Simply point `--model_path` at the
appropriate directory:

```bash
# FP8
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/fp8/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --height 720 --width 1280 --num_frames 121 \
    --output_path output_fp8.mp4

# FP4
python visual_gen_ltx2.py \
    --model_path ${MODEL_ROOT}/LTX-2-checkpoint/fp4/ \
    --text_encoder_path ${MODEL_ROOT}/gemma-3-12b-it \
    --prompt "A cute cat playing piano" \
    --height 512 --width 768 --num_frames 121 \
    --output_path output_fp4.mp4
```

---

## Common Arguments

| Argument | FLUX | WAN | LTX2 | Default | Description |
|----------|------|-----|------|---------|-------------|
| `--model_path` | ✓ | ✓ | — | Path to model checkpoint directory |
| `--text_encoder_path` | — | ✓ | — | Path to Gemma3 text encoder |
| `--prompt` | ✓ | ✓ | — | Text prompt for generation |
| `--negative_prompt` | — | ✓ | *(built-in)* | Negative prompt |
| `--height` | ✓ | ✓ | ✓ | 1024 / 720 | Output height |
| `--width` | ✓ | ✓ | ✓ | 1024 / 1280 | Output width |
| `--num_frames` | — | ✓ | ✓ | 81 / 121 | Number of frames |
| `--frame_rate` | — | ✓ | 24.0 | Output frame rate (fps) |
| `--steps` | ✓ | ✓ | ✓ | 50 / 40 | Denoising steps |
| `--guidance_scale` | ✓ | ✓ | ✓ | 3.5 / 5.0 / 4.0 | Guidance strength |
| `--seed` | ✓ | ✓ | ✓ | 42 | Random seed |
| `--image` | — | ✓ | None | Input image for image-to-video |
| `--image_cond_strength` | — | ✓ | 1.0 | Image conditioning strength |
| `--enable_teacache` | ✓ | ✓ | — | False | Cache optimization |
| `--teacache_thresh` | ✓ | ✓ | — | 0.2 | TeaCache similarity threshold |
| `--attention_backend` | ✓ | ✓ | — | VANILLA | `VANILLA`, `TRTLLM`, or `FA4` |
| `--enable_sage_attention` | ✓ | ✓ | — | False | SageAttention (requires `TRTLLM` attention backend) |
| `--cfg_size` | — | ✓ | — | 1 | CFG parallelism |
| `--ulysses_size` | ✓ | ✓ | — | 1 | Ulysses parallelism |
| `--attn2d_row_size` | ✓ | ✓ | ✓ | 1 | Attention2D mesh row size |
| `--attn2d_col_size` | ✓ | ✓ | ✓ | 1 | Attention2D mesh column size |
| `--linear_type` | ✓ | ✓ | — | default | Quantization type |
| `--enhance_prompt` | — | ✓ | False | Gemma3 prompt enhancement |
| `--stg_scale` | — | ✓ | 0.0 | Spatiotemporal guidance scale |
| `--modality_scale` | — | ✓ | 1.0 | Cross-modal guidance scale |
| `--rescale_scale` | — | ✓ | 0.0 | Variance-preserving rescale factor |

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
- `ulysses_size` must divide the model's head count (12 for WAN); if your GPU
  count does not divide the head count, use Attention2D instead
  (`--attention_backend FA4 --attn2d_row_size <row> --attn2d_col_size <col>`)
- Total GPUs = `cfg_size × ulysses_size`
- Sequence length must be divisible by `ulysses_size`

**Attention2D Errors:**
- Requires `--attention_backend FA4`
- Combining with `--ulysses_size` is not yet supported
- Total GPUs = `cfg_size × attn2d_row_size × attn2d_col_size`
- Sequence length must be divisible by `attn2d_row_size × attn2d_col_size`

## Output Formats

- **FLUX**: `.png` (image)
- **WAN**: `.mp4` if FFmpeg is installed, otherwise `.avi` (video)
- **LTX2**: `.mp4` (video with audio) if FFmpeg is installed, otherwise `.avi` (video)

## Serving

See [`serve/README.md`](serve/README.md) for `trtllm-serve` examples including image generation (FLUX), video generation (WAN T2V/I2V), and API endpoint reference.
