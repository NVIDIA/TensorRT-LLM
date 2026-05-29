# Visual Generation (Beta)

```{note}
This feature is in **beta** stage. APIs, supported models, and optimization options are
actively evolving and may change in future releases.
```

## Background

Visual generation models based on diffusion transformers (DiT) have become the standard for high-quality image and video synthesis. These models iteratively denoise latent representations through a learned transformer backbone, then decode the final latents with a VAE to produce pixels.

TensorRT-LLM **VisualGen** provides a unified inference stack for diffusion models, with a pipeline architecture separate from the LLM inference path. Key capabilities include:

- A shared pipeline abstraction covering the denoising loop, guidance strategies, and component loading.
- Pluggable attention backends: PyTorch SDPA (`VANILLA`), TRT-LLM kernels (`TRTLLM`), TRT-LLM CuTe DSL kernels (`CUTEDSL`, Blackwell-class GPUs), and Flash Attention 4 (`FA4`).
- Quantization support (dynamic and static) using the [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) configuration format.
- Quantized attention support: `QK16PV8` to quantize Bmm2 on `CUTEDSL`, `SAGE` to run SageAttention on `TRTLLM` (requires Blackwell SM100).
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
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Text-to-Video, Image-to-Video |
| `Lightricks/LTX-2` | Text-to-Video (with Audio), Image-to-Video (with Audio) |

Models are auto-detected from the checkpoint directory. Diffusers-format models are detected via `model_index.json`; LTX-2 monolithic safetensors checkpoints are detected via embedded metadata. The `AutoPipeline` registry selects the appropriate pipeline class automatically.

### Feature Matrix

| Model | FP8 blockwise | NVFP4 | TeaCache | CFG Parallelism | Ulysses Parallelism | Parallel VAE | CUDA Graph | torch.compile | trtllm-serve | Attention2D | Ring Attention |
|---|---|---|---|---|---|---|---|---|---|--|--|
| **FLUX.1** | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes | Yes | Yes |
| **FLUX.2** | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes | Yes | Yes |
| **Wan 2.1** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Wan 2.2** | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **LTX-2** | Yes | Yes | No | Yes | Yes | No | No | Yes | Yes | Yes | Yes |

[^1]: FLUX models use embedded guidance and do not have a separate negative prompt path, so CFG parallelism is not applicable.

## Quick Start

Here is a simple example to generate a video with Wan 2.1:

```bash
python examples/visual_gen/quickstart_example.py
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
    model="/path/to/model",
    quant_config={"quant_algo": "FP8", "dynamic": True},
)
```

### Quantized Attention

In addition to linear-layer quantization, VisualGen exposes two **attention-level** quantization presets that operate inside the attention kernel. They are configured through `AttentionConfig.quant_attention_config` (or the `--quant_attention_mode` flag in the example scripts) and are mutually exclusive with each other.

- **QK16PV8** (`CUTEDSL` backend): Keeps Q & K in BF16 and quantizes only V to FP8 (E4M3, per-tensor), thus Bmm1 will be carried out in BF16 with Bmm2 in FP8. Targets Blackwell-class GPUs (`sm_100a` / `sm_103a`) with `head_dim = 128`.
- **SAGE** (`TRTLLM` backend): Quantizes Q, K, and V with per-block scaling factors. Q/K are stored as INT8 or FP8 (e4m3) and V as FP8 (e4m3); block sizes are tunable per axis (typically `(q, k, v) = (1, 4, 1)` for Wan-1.3B and `(1, 16, 1)` for larger Wan / FLUX checkpoints). Supported recipes are validated at runtime.


Python API for SageAttention:

```python
from tensorrt_llm import VisualGenArgs

args = VisualGenArgs(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    attention_config={
        "backend": "TRTLLM",
        "quant_attention_config": {
            "qk_dtype": "int8",
            "q_block_size": 1,
            "k_block_size": 16,
            "v_block_size": 1,
        },
    },
)
```

Python API for QK16PV8:

```python
from tensorrt_llm import VisualGenArgs

args = VisualGenArgs(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    attention_config={
        "backend": "CUTEDSL",
        "quant_attention_config": {
            "qk_dtype": "bf16",
            "q_block_size": 0,
            "k_block_size": 0,
            "v_block_size": 0,
        },
    },
)
```

### TeaCache

TeaCache caches transformer outputs when timestep embeddings change slowly between denoising steps, skipping redundant computation. Enable via `VisualGenArgs.cache_config` (YAML or programmatic):

```yaml
cache_config:
  cache_backend: teacache
  teacache_thresh: 0.2
```

The `teacache_thresh` parameter controls the similarity threshold. Cache-DiT is also supported via `cache_backend: cache_dit` with its own set of knobs (see `CacheDiTConfig`).

### Multi-GPU Parallelism

5 parallelism modes can be combined:

- **CFG Parallelism** (`--cfg_size 2`): Splits positive/negative guidance prompts across GPUs.
- **Ulysses Parallelism** (`--ulysses_size N`): Splits the sequence dimension across GPUs for longer sequences.
- **Parallel VAE** (`--parallel_vae_size N`): Shards the final VAE decode along a spatial axis across GPUs, useful to reduce VAE latency and improve GPU utilization (Constraint: `parallel_vae_size ≤ world_size`). Currently only supported for WAN models.
- **Attention Parallel**: There are 2 methods supported to run attention parallel. Both of these methods require the attention backend to support LSE (`FA4` and `CUTEDSL`) - 
    - **Attention2D Parallelism** (`--attn2d_row_size N`, `--attn2d_col_size M`): Shards the sequence axis across a 2D `N x M` device mesh, all-gathering Q along rows and K/V along columns so each rank computes a sub-block of the attention matrix (total CP degree = `N * M`; not currently combinable with Ulysses).
    - **Ring Attention Parallelism** (`--ring_size N`): Shards the sequence axis across a 1D ring of `N` ranks and streams K/V blocks around the ring so each rank computes its attention output without materializing the full K/V (mutually exclusive with Attention2D).
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
| `TeaCacheAccelerator` / `CacheDiTAccelerator` | `visual_gen/cache/` | Runtime caching backends (TeaCache, Cache-DiT) wrapping the transformer forward |
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
- Cache acceleration (if you call `self._setup_cache_acceleration(self.transformer, coefficients=...)` in `post_load_weights()`; supports both TeaCache and Cache-DiT via `VisualGenArgs.cache_config`)
- Serving via `trtllm-serve` with the full endpoint set
