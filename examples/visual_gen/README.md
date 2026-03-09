# Visual Generation Examples

Quick reference for running visual generation models.
Please refer to [the VisualGen doc](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html)
about the details of the feature.

## Prerequisites

```bash
# Install dependencies (from repository root)
pip install -r requirements-dev.txt
pip install git+https://github.com/huggingface/diffusers.git
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
- **Ulysses Parallelism**: Split sequence across GPUs for longer sequences


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

**Large-Scale (8 GPUs):**
```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --attention_backend TRTLLM \
    --cfg_size 2 --ulysses_size 4 \
    --output_path output.mp4
```


## WAN (Image-to-Video)

```bash
python visual_gen_wan_i2v.py \
    --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --image_path input_image.jpg \
    --prompt "She turns around and smiles" \
    --height 480 --width 832 --num_frames 81 \
    --output_path output_i2v.mp4
```


## Common Arguments

| Argument | FLUX | WAN | Default | Description |
|----------|------|-----|---------|-------------|
| `--height` | ✓ | ✓ | 1024 / 720 | Output height |
| `--width` | ✓ | ✓ | 1024 / 1280 | Output width |
| `--num_frames` | — | ✓ | 81 | Number of frames |
| `--steps` | ✓ | ✓ | 50 | Denoising steps |
| `--guidance_scale` | ✓ | ✓ | 3.5 / 5.0 | Guidance strength |
| `--seed` | ✓ | ✓ | 42 | Random seed |
| `--enable_teacache` | ✓ | ✓ | False | Cache optimization |
| `--teacache_thresh` | ✓ | ✓ | 0.2 | TeaCache similarity threshold |
| `--attention_backend` | ✓ | ✓ | VANILLA | VANILLA or TRTLLM |
| `--cfg_size` | — | ✓ | 1 | CFG parallelism |
| `--ulysses_size` | ✓ | ✓ | 1 | Sequence parallelism |
| `--linear_type` | ✓ | ✓ | default | Quantization type |

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
- **WAN**: `.mp4` if FFmpeg is installed, otherwise `.avi` (video)

## Serving

See [`serve/README.md`](serve/README.md) for `trtllm-serve` examples including image generation (FLUX), video generation (WAN T2V/I2V), and API endpoint reference.
