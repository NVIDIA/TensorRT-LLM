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
- Sparse attention support: see [VisualGen Sparse Attention](../visual-gen/features/sparse-attention.md).
- Multi-GPU parallelism (CFG parallel, Ulysses sequence parallel, Tensor parallelism).
- **Step caching** — two runtime caching backends (**TeaCache** and **Cache-DiT**) that skip transformer computation on steps where the step-to-step change is small.
- `trtllm-serve` integration with OpenAI-compatible API endpoints for image and video generation.

## Supported Models

| HuggingFace Model ID | Tasks |
|---|---|
| `black-forest-labs/FLUX.1-dev` | Text-to-Image |
| `black-forest-labs/FLUX.2-dev` | Text-to-Image |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Text-to-Video |
| `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` | Text-to-Video (VSA) |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | Image-to-Video |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | Image-to-Video |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Text-to-Video |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Image-to-Video |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Text-to-Video, Image-to-Video |
| `Lightricks/LTX-2` | Text-to-Video (with Audio), Image-to-Video (with Audio) |
| `Qwen/Qwen-Image` | Text-to-Image |
| `Qwen/Qwen-Image-2512` | Text-to-Image |
| `nvidia/Cosmos3-Nano` | Text-to-Image, Text-to-Video, Image-to-Video |
| `nvidia/Cosmos3-Super` | Text-to-Image, Text-to-Video, Image-to-Video |

Models are auto-detected from the checkpoint directory. Diffusers-format models are detected via `model_index.json`; LTX-2 monolithic safetensors checkpoints are detected via embedded metadata. The `AutoPipeline` registry selects the appropriate pipeline class automatically.

### Feature Matrix

| Model | FP8 blockwise | NVFP4 | TeaCache | Cache-DiT | CFG Parallelism | Ulysses Parallelism | Parallel VAE | CUDA Graph | torch.compile | trtllm-serve | Attention2D | Ring Attention | Tensor Parallelism | VSA |
|---|---|---|---|---|---|---|---|---|---|---|--|--|--|--|
| **FLUX.1** | Yes | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **FLUX.2** | Yes | Yes | Yes | Yes | No [^1] | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Wan 2.1** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Wan 2.1 VSA** [^2] | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes | Yes |
| **Wan 2.2** | Yes | Yes | Yes [^3] | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **LTX-2** | Yes | Yes | Yes [^4] | Yes | Yes | Yes | No | No | Yes | Yes | Yes | Yes | No | No |
| **Qwen-Image** [^5] | Yes | Yes | No | No | No | Yes | No | Yes | Yes | Yes | Yes | Yes | No | No |
| **Cosmos3** | Yes | Yes | No | No | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No |

[^1]: FLUX models use embedded guidance and do not have a separate negative prompt path, so CFG parallelism is not applicable.

[^2]: `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` — VSA-fine-tuned checkpoint with learned sparse-attention gates. Requires `CUTEDSL` on Blackwell sm_100+ (falls back to dense SDPA on older hardware). Ring and Attention2D not supported (no LSE output); Ulysses supported.

[^3]: Wan 2.2 has two stage transformers; TeaCache requires explicit `teacache.coefficients` (high-noise) and `teacache.coefficients_2` (low-noise). There is no built-in coefficient table for Wan 2.2.

[^4]: LTX-2 has no built-in TeaCache coefficient table in TRT-LLM; set `teacache.coefficients` explicitly when enabling TeaCache.

[^5]: Qwen-Image ships a native BF16 implementation with per-module numerical parity against `diffusers.QwenImagePipeline` (cosine similarity >= 0.999 on the full 20B transformer) and supports `trtllm-serve` / `/v1/images/generations`. VisualGen supports FP8 blockwise and NVFP4 dynamic quantization from BF16 checkpoints, as well as direct loading of statically quantized FP8 and NVFP4 ModelOpt checkpoints.

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

Configure via `VisualGenArgs.quant_config` (YAML or programmatic):

```yaml
quant_config:
  quant_algo: FP8        # or FP8_BLOCK_SCALES, NVFP4
  dynamic: true
```

```python
from tensorrt_llm import VisualGenArgs
args = VisualGenArgs(model="/path/to/model", quant_config={"quant_algo": "FP8", "dynamic": True})
```

Omit `quant_config` for BF16/FP16 baseline.

### Quantized Attention

In addition to linear-layer quantization, VisualGen exposes two **attention-level** quantization presets that operate inside the attention kernel. They are configured through `AttentionConfig.quant_attention_config` and are mutually exclusive with each other.

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

### CUDA Graphs

VisualGen CUDA graphs capture transformer forward calls during denoising and replay them for later steps with compatible inputs. See [VisualGen CUDA Graphs](../visual-gen/features/cuda-graph.md) for capture scope, graph keys, and sparse-attention phase behavior.

### Step Caching

Both caching backends are configured through `VisualGenArgs.cache_config`. The backend is selected by the `cache_backend` discriminator field.

#### TeaCache

TeaCache caches transformer outputs when timestep embeddings change slowly between denoising steps, skipping redundant computation. Enable via `VisualGenArgs.cache_config` (YAML or programmatic):

```yaml
cache_config:
  cache_backend: teacache
  teacache_thresh: 0.2
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `teacache_thresh` | float | `0.2` | Accumulated timestep-embedding distance threshold. A step is skipped when the accumulated polynomial-rescaled L1 change stays below this value; higher values cache more aggressively (more speedup, possible quality loss). The example configs use `0.6` for FLUX.1 and `0.2` for FLUX.2 and Wan 2.1. |
| `use_ret_steps` | bool | `false` | Enable retention-step caching variant. |
| `coefficients` | list[float] | per-model | Polynomial coefficients used by the TeaCache decision function. Set automatically at load time based on the checkpoint. |

#### Cache-DiT

Cache-DiT uses residual-difference gating (`DBCache`) to adaptively skip transformer blocks, with optional TaylorSeer polynomial prediction and step-computation mask (`SCM`).

Enable via `VisualGenArgs.cache_config`:

```yaml
cache_config:
  cache_backend: cache_dit
```

```python
from tensorrt_llm import VisualGenArgs
from tensorrt_llm.visual_gen import CacheDiTConfig

args = VisualGenArgs(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    cache_config=CacheDiTConfig(
        residual_diff_threshold=0.20,
        max_continuous_cached_steps=4,
    ),
)
```

**Commonly used parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Fn_compute_blocks` | int | `1` | Number of leading transformer blocks that are always fully computed at every denoising step (Fn in the Cache-DiT paper). |
| `Bn_compute_blocks` | int | `0` | Number of trailing transformer blocks used for prediction refinement (Bn). |
| `max_warmup_steps` | int | `4` | Initial denoising steps that always run a full forward pass; caching is disabled for this many steps at the start. |
| `max_cached_steps` | int | `-1` | Total cap on cached (skipped) steps across the run; `-1` means unlimited. |
| `max_continuous_cached_steps` | int | `3` | Maximum consecutive cached steps before a forced full-compute step is inserted. `-1` means unlimited. |
| `residual_diff_threshold` | float | `0.24` | L1-distance threshold for DBCache residual gating. Increase to cache more aggressively (higher speedup, potential quality loss); decrease for more conservative caching. |
| `enable_taylorseer` | bool | `false` | Enable TaylorSeer calibration. Uses Taylor series expansion to approximate hidden states at cached steps, improving output quality over plain residual reuse. |
| `taylorseer_order` | int | `1` | Polynomial order for TaylorSeer (1–4). Only used when `enable_taylorseer=true`. |
| `scm_steps_mask_policy` | str \| None | `None` | Named step-computation mask policy from the `cache_dit` library (`"slow"`, `"medium"`, `"fast"`, `"ultra"`). |
| `scm_steps_policy` | `"dynamic"` \| `"static"` | `"dynamic"` | Execution policy for the SCM mask; only active when `scm_steps_mask_policy` is set. |
| `force_refresh_step_hint` | int \| None | `None` | Step index at which a forced full-compute pass is injected (useful for scheduled quality checkpoints). |
| `force_refresh_step_policy` | `"once"` \| `"repeat"` | `"once"` | Whether `force_refresh_step_hint` fires only on the first call (`"once"`) or at that interval repeatedly (`"repeat"`). |

**Wan 2.2 dual-transformer note:** Wan 2.2 uses two expert transformers (high-noise and low-noise stacks). All `CacheDiTConfig` parameters apply to both stacks, except `max_warmup_steps` and `max_cached_steps`: the low-noise stack always uses fixed internal caps (`max_warmup_steps=2`, `max_cached_steps=20`) regardless of user config.

### Video Sparse Attention (VSA)

VSA reduces the compute cost of self-attention in video diffusion models by selectively attending to only the most relevant spatial-temporal blocks. It uses a two-branch design: a lightweight coarse mean-pool branch computes block-level attention scores to identify the top-K most relevant token blocks, then a fine branch runs a block-sparse CuTe kernel over only those blocks. The two outputs are blended with learned gates.

**Requirements:**
- VSA-fine-tuned checkpoint: [`FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers`](https://huggingface.co/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers). Standard Wan checkpoints do not have the learned VSA gates.
- Blackwell GPU (sm_100+) for the CuTe JIT kernel. Falls back to dense SDPA on older hardware with no accuracy loss.
- `CUTEDSL` attention backend.
- Not compatible with Ring attention or Attention2D (VSA does not produce per-split LSE). Ulysses is supported.

**`vsa_sparsity`** controls the fraction of K/V blocks skipped in the fine branch (0.0 = dense, 0.9 = 90% blocks skipped). Higher sparsity gives more speedup at the cost of some quality.

Python API:

```python
from tensorrt_llm import VisualGenArgs
from tensorrt_llm.visual_gen.args import AttentionConfig, VideoSparseAttentionConfig

args = VisualGenArgs(
    model="FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
    attention_config=AttentionConfig(
        backend="CUTEDSL",
        sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.9),
    ),
)
```

YAML (for use with `--visual_gen_args` or `trtllm-serve`):

```yaml
attention_config:
  backend: CUTEDSL
  sparse_attention_config:
    algorithm: vsa
    vsa_sparsity: 0.90
```


### Multi-GPU Parallelism

Configured under `VisualGenArgs.parallel_config`. Modes can be combined:

- **CFG Parallelism** (`cfg_size: 2`): Splits positive/negative guidance prompts across GPUs.
- **Ulysses Parallelism** (`ulysses_size: N`): Splits the sequence dimension across GPUs for longer sequences.
    - **Async Ulysses A2A pipeline** (`async_ulysses: true` in `parallel_config`): Overlaps per-rank V/Q/K projection compute with the cross-rank all-to-all on a dedicated side stream. Requires `ulysses_size > 1` and an NVLink-connected GPU domain (uses PyTorch `_SymmetricMemory` with CUDA IPC for peer pushes; not currently supported across nodes without MNNVL). Currently wired for WAN and LTX-2 self-attention.
- **Parallel VAE** (`parallel_vae_size: N`): Shards the final VAE decode along a spatial axis (constraint: `parallel_vae_size ≤ world_size`; WAN/Cosmos3 only).
- **Context Parallel (CP)** — Partitions the sequence into shards so that each rank computes partial attention. Requires an LSE-capable attention backend (`FA4` or `CUTEDSL`). CP can be composed with Ulysses, giving a total sequence-parallel (SP) degree = `cp_size · ulysses_size`. The CP degree depends on the implementation below:
    - **Attention2D** (`attn2d_size: [N, M]`): Shards the sequence axis across an `N × M` device mesh (CP degree = `N · M`; total SP degree = `N · M · ulysses_size`).
    - **Ring Attention** (`ring_size: N`): Shards the sequence axis across a 1D ring of `N` ranks, streaming K/V blocks (CP degree = `N`; total SP degree = `N · ulysses_size`; mutually exclusive with Attention2D).
- **Tensor Parallelism** (`tp_size: N`): Splits attention heads and transformer MLPs across GPUs for faster compute and reduced memory usage.
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
| `BasePipeline` | `visual_gen/pipeline.py` | Base class: denoising loop, CFG handling, step caching (TeaCache / Cache-DiT), CUDA graph |
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
