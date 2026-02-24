# Visual Generation (Diffusion Models) [Beta]

- [Background and Motivation](#background-and-motivation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Usage with `trtllm-serve`](#usage-with-trtllm-serve)
- [Quantization](#quantization)
- [Developer Guide](#developer-guide)
  - [Architecture Overview](#architecture-overview)
  - [Implementing a New Diffusion Model](#implementing-a-new-diffusion-model)
- [Summary and Future Work](#summary-and-future-work)
  - [Current Status](#current-status)
  - [Future Work](#future-work)

## Background and Motivation

Visual generation models based on diffusion transformers (DiT) have become the standard for high-quality image and video synthesis. These models iteratively denoise latent representations through a learned transformer backbone, then decode the final latents with a VAE to produce pixels. As model sizes and output resolutions grow, efficient inference becomes critical — demanding multi-GPU parallelism, weight quantization, and runtime caching to achieve practical throughput and latency.

TensorRT-LLM **VisualGen** module provides a unified inference stack for diffusion models. Key capabilities include (subject to change as the feature matures):

- A shared pipeline abstraction for diffusion model families, covering the denoising loop, guidance strategies, and component loading.
- Pluggable attention backends.
- Quantization support (dynamic and static) using the [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) configuration format.
- Multi-GPU parallelism strategies.
- **TeaCache** — a runtime caching optimization for the transformer backbone.
- `trtllm-serve` integration with OpenAI-compatible API endpoints.

> **Note:** This is the initial release of TensorRT-LLM VisualGen. APIs, supported models, and optimization options are actively evolving and may change in future releases.

## Quick Start

### Prerequisites

```bash
pip install -r requirements-dev.txt
pip install git+https://github.com/huggingface/diffusers.git
pip install av
```

**Optional: Flash Attention V4 (Blackwell GPUs / sm100)**

Flash Attention V4 (FA4) provides higher speedup on Blackwell GPUs (sm100). If you want to enable it, we recommend using this version which has been validated by us:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git tensorrt_llm/_torch/visual_gen/3rdparty/flash-attention
cd tensorrt_llm/_torch/visual_gen/3rdparty/flash-attention && git checkout ea8f73506369d7cdd498396474107a978858138c && cd -
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}/3rdparty/flash-attention/
```

### Python API

The example scripts under `examples/visual_gen/` demonstrate direct Python usage. For Wan2.1 text-to-video generation:

```bash
cd examples/visual_gen

python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --height 480 --width 832 --num_frames 33 \
    --output_path output.mp4
```

Run `python visual_gen_wan_t2v.py --help` for the full list of arguments. Key options control resolution, denoising steps, quantization mode, attention backend, parallelism, and TeaCache settings.

### Usage with `trtllm-serve`

The `trtllm-serve` command automatically detects diffusion models (by the presence of `model_index.json`) and launches an OpenAI-compatible visual generation server.

**1. Create a YAML configuration file:**

```yaml
# wan_config.yml
linear:
  type: default
teacache:
  enable_teacache: true
  teacache_thresh: 0.2
parallel:
  dit_cfg_size: 1
  dit_ulysses_size: 1
```

**2. Launch the server:**

```bash
trtllm-serve Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --extra_visual_gen_options wan_config.yml
```

**3. Send requests** using curl or any OpenAI-compatible client:

Synchronous video generation:

```bash
curl -X POST "http://localhost:8000/v1/videos/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cool cat on a motorcycle in the night",
    "seconds": 4.0,
    "fps": 24,
    "size": "480x832"
  }' -o output.mp4
```

Asynchronous video generation:

```bash
# Submit the job
curl -X POST "http://localhost:8000/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cool cat on a motorcycle in the night",
    "seconds": 4.0,
    "fps": 24,
    "size": "480x832"
  }'
# Returns: {"id": "<video_id>", "status": "processing", ...}

# Poll for status
curl -X GET "http://localhost:8000/v1/videos/<video_id>"

# Download when complete
curl -X GET "http://localhost:8000/v1/videos/<video_id>/content" -o output.mp4
```

The server exposes OpenAI-compatible endpoints for image generation (`/v1/images/generations`), video generation (`/v1/videos`, `/v1/videos/generations`), video management, and standard health/model info endpoints.

The `--extra_visual_gen_options` YAML file configures quantization (`linear`), TeaCache (`teacache`), and parallelism (`parallel`). See [`examples/visual_gen/serve/configs/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/serve/configs) for reference configurations.

## Quantization

TensorRT-LLM VisualGen supports both **dynamic quantization** (on-the-fly at weight-loading time from BF16 checkpoints) and **static quantization** (loading pre-quantized checkpoints with embedded scales). Both modes use the same [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) `quantization_config` format.

**Quick start — dynamic quantization via `--linear_type`:**

```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --linear_type trtllm-fp8-per-tensor \
    --output_path output_fp8.mp4
```

Supported `--linear_type` values: `default` (BF16/FP16), `trtllm-fp8-per-tensor`, `trtllm-fp8-blockwise`, `svd-nvfp4`.

**ModelOpt `quantization_config` format:**

Both dynamic and static quantization use the [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) `quantization_config` format — the same format found in a model's `config.json` under the `quantization_config` field. This config can be passed as a dict to `DiffusionArgs.quant_config` when constructing the pipeline programmatically:

```python
from tensorrt_llm._torch.visual_gen.config import DiffusionArgs

args = DiffusionArgs(
    checkpoint_path="/path/to/model",
    quant_config={"quant_algo": "FP8", "dynamic": True},  # dynamic FP8
)
```

The `--linear_type` CLI flag is a convenience shorthand that maps to these configs internally (e.g., `trtllm-fp8-per-tensor` → `{"quant_algo": "FP8", "dynamic": True}`).

Key fields: `"dynamic"` controls load-time quantization (`true`) vs pre-quantized checkpoint (`false`); `"ignore"` excludes specific modules from quantization.

## Developer Guide

This section describes the TensorRT-LLM VisualGen module architecture and guides developers on how to add support for new diffusion model families.

### Architecture Overview

The VisualGen module lives under `tensorrt_llm._torch.visual_gen`. At a high level, the flow is:

1. **Config** — User-facing `DiffusionArgs` (CLI / YAML) is merged with checkpoint metadata into `DiffusionModelConfig`.
2. **Pipeline creation & loading** — `AutoPipeline` detects the model type from `model_index.json`, instantiates the matching `BasePipeline` subclass, and loads weights (with optional dynamic quantization) and standard components (VAE, text encoder, tokenizer, scheduler).
3. **Execution** — `DiffusionExecutor` coordinates multi-GPU inference via worker processes.

> **Note:** Internal module structure is subject to change. Refer to inline docstrings in `tensorrt_llm/_torch/visual_gen/` for the latest details.

### Implementing a New Diffusion Model

Adding a new model (e.g., a hypothetical "MyDiT") requires four steps. The framework handles weight loading, parallelism, quantization, and serving automatically once the pipeline is registered.

#### 1. Create the Transformer Module

Create the DiT backbone in `tensorrt_llm/_torch/visual_gen/models/mydit/transformer_mydit.py`. It should be an `nn.Module` that:

- Uses existing modules (e.g., `Attention` with configurable attention backend, `Linear` for builtin linear ops) wherever possible.
- Implements `load_weights(weights: Dict[str, torch.Tensor])` to map checkpoint weight names to module parameters.

#### 2. Create the Pipeline Class

Create a pipeline class extending `BasePipeline` in `tensorrt_llm/_torch/visual_gen/models/mydit/`. Override methods for transformer initialization, component loading, and inference. `BasePipeline` provides the denoising loop, CFG handling, and TeaCache integration — your pipeline only needs to implement model-specific logic. See `WanPipeline` for a reference implementation.

#### 3. Register the Pipeline

Use the `@register_pipeline("MyDiTPipeline")` decorator on your pipeline class to register it in the global `PIPELINE_REGISTRY`. Make sure to export it from `models/__init__.py`.

#### 4. Update AutoPipeline Detection

In `pipeline_registry.py`, add detection logic for your model's `_class_name` in `model_index.json`.

After these steps, the framework automatically handles:

- Weight loading with optional dynamic quantization via `PipelineLoader`
- Multi-GPU execution via `DiffusionExecutor`
- TeaCache integration (if you call `self._setup_teacache()` in `post_load_weights()`)
- Serving via `trtllm-serve` with the full endpoint set

## Summary and Future Work

### Current Status

**Supported models:** Wan2.1 and Wan2.2 families (text-to-video, image-to-video; 1.3B and 14B variants).

**Supported features:**

| Feature | Status |
|---------|--------|
| **Multi-GPU Parallelism** | CFG parallel, Ulysses sequence parallel (more strategies planned) |
| **TeaCache** | Caches transformer outputs when timestep embeddings change slowly |
| **Quantization** | Dynamic (on-the-fly from BF16) and static (pre-quantized checkpoints), both via ModelOpt `quantization_config` format |
| **Attention Backends** | Vanilla (torch SDPA), TRT-LLM optimized fused kernels, and Flash Attention V4 (FA4, Blackwell / sm100) |
| **`trtllm-serve`** | OpenAI-compatible endpoints for image/video generation (sync + async) |

### Future Work

- **Additional model support**: Extend to more diffusion model families.
- **More attention backends**: Support for additional attention backends.
- **Advanced parallelism**: Additional parallelism strategies for larger models and higher resolutions.
- **Serving enhancements**: Improved throughput and user experience for production serving workloads.
