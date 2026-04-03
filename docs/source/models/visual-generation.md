# Visual Generation (Prototype)

```{note}
This feature is in **prototype** stage. APIs, supported models, and optimization options are
actively evolving and may change in future releases.
```

## Background

Visual generation models based on diffusion transformers (DiT) have become the standard for high-quality image and video synthesis. These models iteratively denoise latent representations through a learned transformer backbone, then decode the final latents with a VAE to produce pixels.

TensorRT-LLM **VisualGen** provides a unified inference stack for diffusion models, with a pipeline architecture separate from the LLM inference path. Key capabilities include:

- A shared pipeline abstraction covering the denoising loop, guidance strategies, and component loading.
- Pluggable attention backends (PyTorch SDPA and TRT-LLM optimized kernels).
- Quantization support (dynamic and static) using the [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) configuration format.
- Multi-GPU parallelism (CFG parallel, Ulysses sequence parallel).
- **TeaCache** — a runtime caching optimization that skips transformer steps when timestep embeddings change slowly.
- `trtllm-serve` integration with OpenAI-compatible API endpoints for image and video generation.

## Supported Models

| HuggingFace Model ID | Tasks |
|---|---|
| `black-forest-labs/FLUX.1-dev` | Text-to-Image |
| `black-forest-labs/FLUX.2-dev` | Text-to-Image |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | Image-to-Video |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | Image-to-Video |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Image-to-Video |
| `Lightricks/LTX-Video` | Text-to-Video (with Audio), Image-to-Video (with Audio) |

Models are auto-detected from the checkpoint directory. Diffusers-format models are detected via `model_index.json`; LTX-2 monolithic safetensors checkpoints are detected via embedded metadata. The `AutoPipeline` registry selects the appropriate pipeline class automatically.

### Feature Matrix

| Model | FP8 blockwise | NVFP4 | TeaCache | CFG Parallelism | Ulysses Parallelism | Parallel VAE | CUDA Graph | torch.compile | trtllm-serve |
|---|---|---|---|---|---|---|---|---|---|
| **FLUX.1** | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes |
| **FLUX.2** | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes |
| **Wan 2.1** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Wan 2.2** | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes |
| **LTX-2** | Yes | Yes | No | Yes | Yes | No | No | Yes | Yes |

[^1]: FLUX models use embedded guidance and do not have a separate negative prompt path, so CFG parallelism is not applicable.

## Quick Start

Here is a simple example to generate a video with Wan 2.1:

```{literalinclude} ../../../examples/visual_gen/quickstart_example.py
    :language: python
    :linenos:
```

To learn more about VisualGen, see [`examples/visual_gen/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen) for more examples including text-to-image, image-to-video, and batch generation.

### Usage with `trtllm-serve`

The `trtllm-serve` command automatically detects diffusion models (by the presence of `model_index.json`) and launches an OpenAI-compatible visual generation server with image and video generation endpoints.

See [`examples/visual_gen/serve/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/serve) for server launch instructions, example configurations, and API usage.

### Serving Endpoints

When served via `trtllm-serve`, the following OpenAI-compatible endpoints are available:

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/images/generations` | POST | Synchronous image generation |
| `/v1/images/edits` | POST | Image editing |
| `/v1/videos` | POST | Asynchronous video generation |
| `/v1/videos/generations` | POST | Synchronous video generation |
| `/v1/videos/{id}` | GET | Video status / metadata |
| `/v1/videos/{id}/content` | GET | Download generated video |
| `/v1/videos/{id}` | DELETE | Delete generated video |
| `/v1/videos` | GET | List all videos |

## Optimizations

### Quantization

VisualGen supports both **dynamic quantization** (on-the-fly at weight-loading time from BF16 checkpoints) and **static quantization** (loading pre-quantized checkpoints with embedded scales). Both modes use the [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) `quantization_config` format.

Dynamic quantization via `--linear_type`:

```bash
python visual_gen_wan_t2v.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cute cat playing piano" \
    --linear_type trtllm-fp8-per-tensor \
    --output_path output_fp8.mp4
```

Supported `--linear_type` values: `default` (BF16/FP16), `trtllm-fp8-per-tensor`, `trtllm-fp8-blockwise`, `trtllm-nvfp4`.

Programmatic usage via `VisualGenArgs.quant_config`:

```python
from tensorrt_llm import VisualGenArgs

args = VisualGenArgs(
    checkpoint_path="/path/to/model",
    quant_config={"quant_algo": "FP8", "dynamic": True},
)
```

### TeaCache

TeaCache caches transformer outputs when timestep embeddings change slowly between denoising steps, skipping redundant computation. Enable with `teacache.enable_teacache: true` (YAML config). The `teacache_thresh` parameter controls the similarity threshold.

### Multi-GPU Parallelism

Two parallelism modes can be combined:

- **CFG Parallelism** (`--cfg_size 2`): Splits positive/negative guidance prompts across GPUs.
- **Ulysses Parallelism** (`--ulysses_size N`): Splits the sequence dimension across GPUs for longer sequences.

Total GPU count = `cfg_size * ulysses_size`.

## Developer Guide

### Architecture Overview

The VisualGen module lives under `tensorrt_llm._torch.visual_gen`. At a high level, the inference flow is:

1. **Config** — User-facing `VisualGenArgs` (CLI / YAML) is merged with checkpoint metadata into `DiffusionModelConfig`.
2. **Pipeline creation & loading** — `AutoPipeline` detects the model type from `model_index.json`, instantiates the matching `BasePipeline` subclass, and loads weights (with optional dynamic quantization) and standard components (VAE, text encoder, tokenizer, scheduler).
3. **Execution** — `DiffusionExecutor` coordinates multi-GPU inference via worker processes communicating over ZeroMQ IPC.

Key components:

| Component | Location | Role |
|---|---|---|
| `VisualGen` | `tensorrt_llm/visual_gen/__init__.py` | High-level API: manages workers, `generate()` / `generate_async()` |
| `DiffusionExecutor` | `visual_gen/executor.py` | Worker process: loads pipeline, processes requests via ZeroMQ |
| `BasePipeline` | `visual_gen/pipeline.py` | Base class: denoising loop, CFG handling, TeaCache, CUDA graph |
| `AutoPipeline` | `visual_gen/pipeline_registry.py` | Factory: auto-detects model type, selects pipeline class |
| `PipelineLoader` | `visual_gen/pipeline_loader.py` | Resolves checkpoint, loads config/weights, creates pipeline |
| `TeaCacheBackend` | `visual_gen/teacache.py` | Runtime caching for transformer outputs |
| `WeightLoader` | `visual_gen/checkpoints/` | Loads transformer weights from safetensors/bin |

VisualGen is a parallel inference subsystem within TensorRT-LLM. It shares low-level primitives (`Mapping`, `QuantConfig`, `Linear`, `RMSNorm`, `ZeroMqQueue`, `TrtllmAttention`) but has its own executor, scheduler (diffusers-based), request types, and pipeline architecture separate from the LLM autoregressive decode path.

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
