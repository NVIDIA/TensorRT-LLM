# Examples

This directory contains example of popular diffusion pipelines.

## Overview

We provides state-of-the-art diffusion model capabilities with optimized inference for various model architectures. Currently supported models include:

### Supported Models

| Series | Model ID | Parameters | Model Type |
|--------|----------|------------|------------|
| Wan2.1 T2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 1.3B | **Text-to-Video** |
| Wan2.1 T2V | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 14B | **Text-to-Video** |
| Wan2.2 T2V | `Wan-AI/Wan2.2-T2V-14B-Diffusers` | 14B | **Text-to-Video** |
| Wan2.1 I2V | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 14B | **Image-to-Video** |
| Wan2.1 I2V | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 14B | **Image-to-Video** |
| Wan2.2 I2V | `Wan-AI/Wan2.2-I2V-14B-Diffusers` | 14B | **Image-to-Video** |
| Flux.1 [dev] | `black-forest-labs/FLUX.1-dev` | 12B | **Text-to-Image** |
| FLUX.2 [dev] | `black-forest-labs/FLUX.2-dev` | 32B | **Text-to-Image** |
| HunyuanImage2.1 | `tencent/HunyuanImage-2.1` | 17B | **Text-to-Image** |
| Cosmos | `nvidia/Cosmos-Predict2-2B-Video2World` | 2B | **Video-to-World** |
| Cosmos | `nvidia/Cosmos-Predict2-14B-Video2World` | 14B | **Video-to-World** |
| Cosmos | `nvidia/Cosmos-Predict2-2B-Text2Image` | 2B | **Text-to-Image** |
| Cosmos | `nvidia/Cosmos-Predict2-14B-Text2Image` | 14B | **Text-to-Image** |
| Cosmos | `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` | 7B | **Text-to-World** |

## Quick Start

### Single GPU

#### Wan

**Wan2.1 T2V 1.3B**
```bash
python wan_t2v.py --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --height 480 --width 832 --num_frames 33 --enable_teacache
```

**Wan2.1 T2V 14B**
```bash
python wan_t2v.py --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers" --height 720 --width 1280 --num_frames 81 --enable_teacache
```

**Wan2.1 I2V 480P**
```bash
python wan_i2v.py --model_path "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" --height 480 --width 832 --num_frames 81 --enable_teacache
```

**Wan2.1 I2V 720P**
```bash
python wan_i2v.py --model_path "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" --height 720 --width 1280 --num_frames 81 --enable_teacache
```

**Wan2.2 T2V**
```bash
python wan_t2v.py --model_path "Wan-AI/Wan2.2-I2V-A14B-Diffusers" --height 720 --width 1280 --num_frames 81
```

**Wan2.2 I2V**
```bash
python wan_i2v.py --model_path "Wan-AI/Wan2.2-I2V-A14B-Diffusers" --height 720 --width 1280 --num_frames 81
```

#### Flux

**Flux.1 Dev (1024x1024)**
```bash
python flux.py --model_path "black-forest-labs/FLUX.1-dev" --height 1024 --width 1024 --enable_teacache --enable_cuda_graph
```

**Flux.2 Dev (1024x1024)**
```bash
python flux2.py --model_path "black-forest-labs/FLUX.2-dev" --height 1024 --width 1024 --enable_cuda_graph
```

#### HunyuanImage-2.1
```bash
# HunyuanImage2.1 is independent with diffusers, we have to access relative codes from Github
git clone git@github.com:Tencent-Hunyuan/HunyuanImage-2.1.git
cd HunyuanImage-2.1 && export PYTHONPATH=$PYTHONPATH:`pwd` && cd ..
# Single GPU
python hunyuan_image2.1.py  --attn_type flash-attn3 --linear_type trtllm-fp8-blockwise --enable_cuda_graph
# 4 GPUs
torchrun --nproc-per-node 4 hunyuan_image2.1.py  --attn_type flash-attn3 --linear_type trtllm-fp8-blockwise --cp 4 --enable_cuda_graph
```

#### Cosmos

**Cosmos Predict2 2B Video2World**
```bash
python cosmos_i2v.py --model_path "nvidia/Cosmos-Predict2-2B-Video2World" --height 704 --width 1280 --num_frames 93
```

**Cosmos Predict2 14B Video2World**
```bash
python cosmos_i2v.py --model_path "nvidia/Cosmos-Predict2-14B-Video2World" --height 704 --width 1280 --num_frames 93
```

**Cosmos Predict2 2B Text2Image**
```bash
python cosmos_t2i.py --model_path "nvidia/Cosmos-Predict2-2B-Text2Image" --height 768 --width 1360
```

**Cosmos 1.0 Diffusion 7B Text2World**
```bash
python cosmos_t2v.py --model_path "nvidia/Cosmos-1.0-Diffusion-7B-Text2World" --height 704 --width 1280 --num_frames 121 --fps 30
```

### Multi-GPU

#### Wan

**Wan2.1 T2V 1.3B (2 GPUs with CFG)**
```bash
torchrun --nproc_per_node=2 wan_t2v.py --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --cfg 2 --height 480 --width 832 --num_frames 33 --enable_teacache
```

**Wan2.1 T2V 14B (4 GPUs with CFG + Ulysses)**
```bash
torchrun --nproc_per_node=4 wan_t2v.py --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers" --cfg 2 --ulysses 2 --height 720 --width 1280 --num_frames 81 --enable_teacache
```

**Wan2.1 I2V 480P (2 GPUs with CFG)**
```bash
torchrun --nproc_per_node=2 wan_i2v.py --model_path "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" --cfg 2 --height 480 --width 832 --num_frames 81 --enable_teacache
```

**Wan2.1 I2V 720P (4 GPUs with CFG + Ulysses)**
```bash
torchrun --nproc_per_node=4 wan_i2v.py --model_path "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" --cfg 2 --ulysses 2 --height 720 --width 1280 --num_frames 81 --enable_teacache
```

**Wan2.2 T2V (8 GPUs with CFG + Ulysses)**
```bash
torchrun --nproc_per_node=8 wan_t2v.py --model_path "Wan-AI/Wan2.2-T2V-A14B-Diffusers" --cfg 2 --ulysses 4 --height 720 --width 1280 --num_frames 81 --enable_teacache
```

**Wan2.2 I2V (4 GPUs with CFG + Ulysses + FSDP)**
```bash
torchrun --nproc_per_node=4 wan_i2v.py --model_path "Wan-AI/Wan2.2-I2V-A14B-Diffusers" --cfg 2 --ulysses 2 --fsdp 2 --height 720 --width 1280 --num_frames 81 --enable_teacache
```

#### Flux

**Flux.1 Dev (2 GPUs with Ulysses)**
```bash
torchrun --nproc_per_node=2 flux.py --model_path "black-forest-labs/FLUX.1-dev" --ulysses 2 --height 1024 --width 1024 --enable_teacache
```

**Flux.1 Dev (4 GPUs with Ulysses)**
```bash
torchrun --nproc_per_node=4 flux.py --model_path "black-forest-labs/FLUX.1-dev" --ulysses 4 --height 1024 --width 1024 --enable_teacache
```

#### Cosmos

**Cosmos Video2World (4 GPUs with CFG + Ulysses)**
```bash
torchrun --nproc_per_node=4 cosmos_i2v.py --model_path "nvidia/Cosmos-Predict2-2B-Video2World" --cfg 2 --ulysses 2 --height 704 --width 1280 --num_frames 93
```

**Cosmos Video2World (8 GPUs with CFG + Ulysses)**
```bash
torchrun --nproc_per_node=8 cosmos_i2v.py --model_path "nvidia/Cosmos-Predict2-2B-Video2World" --cfg 2 --ulysses 4 --height 704 --width 1280 --num_frames 93
```

**Cosmos Text2Image (4 GPUs with Ulysses)**
```bash
torchrun --nproc_per_node=4 cosmos_t2i.py --model_path "nvidia/Cosmos-Predict2-2B-Text2Image" --ulysses 4 --height 768 --width 1360
```

**Cosmos Text2World (8 GPUs with CFG + Ulysses)**
```bash
torchrun --nproc_per_node=8 cosmos_t2v.py --model_path "nvidia/Cosmos-1.0-Diffusion-7B-Text2World" --cfg 2 --ulysses 4 --height 704 --width 1280 --num_frames 121 --fps 30
```

## Arguments Reference

### Basic Generation Arguments

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path or HuggingFace model ID |
| `--prompt` | Text prompt for generation |
| `--negative_prompt` | Negative prompt for generation |
| `--height` | Image/Video height in pixels (default: 512) |
| `--width` | Image/Video width in pixels (default: 512) |
| `--num_frames` | Number of frames to generate (video models only) |
| `--fps` | FPS for exported video (default: 16) |
| `--image` | Input image URL or local path (for image-to-video models) |
| `--guidance_scale` | Guidance scale (default: 5.0) |
| `--num_inference_steps` | Number of inference steps (default: 50) |
| `--num_warmup_steps` | Number of warmup steps (default: 1) |
| `--random_seed` | Random seed (default: 42) |
| `--output_path` | Output path for generated content |

### Performance Optimization Arguments

| Parameter | Description |
|-----------|-------------|
| `--attn_type` | Attention type: `default`, `sage-attn`, `sparse-videogen`, `trtllm-attn`, `flash-attn3`, `flash-attn3-fp8` (default: `default`) |
| `--linear_type` | Linear type: `default`, `trtllm-fp8-blockwise`, `trtllm-fp8-per-tensor`, `trtllm-nvfp4` (default: `default`) |
| `--disable_torch_compile` | Disable torch compile (enabled by default) |
| `--torch_compile_models` | Models to compile with torch compile (default: `transformer`) |
| `--torch_compile_mode` | Torch compile mode: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `--enable_teacache` | Enable TeaCache for acceleration |
| `--teacache_thresh` | Threshold for TeaCache (default: 0.2) |
| `--disable_qkv_fusion` | Disable QKV fusion (enabled by default) |

### Parallelization Arguments

| Parameter | Description |
|-----------|-------------|
| `--dp` | Data parallelism degree for the transformer model, enable this only when it has multiple prompts |
| `--cfg` | CFG parallelism degree for the transformer model, enable this only when the model uses Classifier Free Guidance |
| `--ulysses` | Ulysses attention parallelism degree for the transformer model |
| `--ring` | Ring parallelism degree, an implementation to this [paper](https://arxiv.org/abs/2310.01889) |
| `--cp` | Context parallelism degree, only one `all_gather` to gather all KV to local rank, suitable for small comm size |
| `--fsdp` | FSDP sharding degree for the transformer model |
| `--t5_fsdp` | FSDP sharding degree for T5 encoder |

### Memory Optimization Arguments

| Parameter | Description |
|-----------|-------------|
| `--enable_sequential_cpu_offload` | Enable sequential CPU offload |
| `--enable_model_cpu_offload` | Enable model CPU offload |
| `--enable_async_cpu_offload` | Enable block-wise async CPU offload strategy |
| `--visual_gen_block_cpu_offload_stride` | Stride for block CPU offload (default: 1, higher values use more GPU memory) |

### VAE Optimization Arguments
| `--disable_parallel_vae` | Disable parallel VAE processing |
| `--parallel_vae_split_dim` | Split dimension for parallel VAE: `height` or `width` (default: `width`) |

### Advanced Arguments

| Parameter | Description |
|-----------|-------------|
| `--num_timesteps_high_precision` | Fraction of timesteps to use high precision attention (default: 0.0) |
| `--num_layers_high_precision` | Fraction of layers to use high precision attention (default: 0.0) |
| `--high_precision_attn_type` | High precision attention type: `default` or `sage-attn` |
| `--sparsity` | Sparsity level for sparse attention (default: 0.25) |
| `--max_sequence_length` | Max sequence length for text encoder (Flux only, default: 512) |


## Choosing Better Arguments

We take **Wan2.1 Text-to-Video** as an example to show how to choose a better argument to befinit from further optimizations.

### Multi-GPU Configurations

**Multi-GPU Parallelization Options**

We currently offers several multi-GPU parallelization strategies:

* Data Parallelism (DP): 
   - Best for scenarios with multiple prompts
   - Each GPU processes a subset of the batch independently

* Classifier-Free Guidance Parallelism (CFG):
   - Distributes CFG computation across GPUs
   - Ideal for accelerating CFG-based generation, such as Wan2.1

* Ulysses Parallelism:
   - A kind of CP (context parallel) method
   - Optimizes attention computation by distributing attention heads

* Fully Sharded Data Parallelism (FSDP):
   - Distributes model parameters across GPUs
   - Helps prevent out-of-memory (OOM) errors
   - Recommended for large models or limited GPU memory

Choose the appropriate parallelization strategy based on your specific hardware setup and performance requirements.

The parallelism size of each must follow these constraints:

* DP/CFG/Ulysses Size Requirements:
    - The product of DP, CFG, and Ulysses sizes must equal the total number of GPUs (world size)
    - Example: For 4 GPUs, valid combinations include:
        - DP=1, CFG=2, Ulysses=2
        - DP=2, CFG=2, Ulysses=1
        - DP=4, CFG=1, Ulysses=1

* FSDP Size Requirements:
   - FSDP size must be a divisor of the world size
   - Example: For 4 GPUs, valid FSDP sizes are 1, 2, or 4
   - T5 FSDP size follows the same rule


**2 GPUs with CFG Parallelism:**
```bash
torchrun --nproc_per_node=2 wan_t2v.py --cfg 2
```

**2 GPUs with Ulysses Parallelism:**
```bash
torchrun --nproc_per_node=2 wan_t2v.py --ulysses 2
```

> we can further accelerate ulysses by quantization. We can do quant/dequant before/after the all2all comm.
```bash
torchrun --nproc_per_node=2 wan_t2v.py --ulysses 2 --int8_ulysses
```

> we can fuse qkv to bigger data size and thus improve data transfer efficiency
```bash
torchrun --nproc_per_node=2 wan_t2v.py --ulysses 2 --int8_ulysses --fuse_qkv_in_ulysses
```

**4 GPUs with CFG and Ulysses Attention:**
```bash
torchrun --nproc_per_node=4 wan_t2v.py --cfg 2 --ulysses 2
```

### Performance Optimization

**Attention**

In large video diffusion models, >70% of the DiT transformer latency is contributed by the attention layers. Select from a variety of attention optimizations

*1. SageAttention*

SageAttention leverages quantization to accelerate the attention operator, which brings good performance but also has impact on accuracy. We find it is good at performance-accuracy tradeoff, so we recommend to enable it.

```bash
python wan_t2v.py --attn_type "sage-attn"

```

*2. Sparse VideoGen*

[Sparse VideoGen (SVG)](https://github.com/svg-project/Sparse-VideoGen/tree/main) leverages inherent spatial and temporal sparsity in the 3D Full Attention operations. We added an example to show how to use it. You can tune related options `num_timesteps_high_precision`, `num_layers_high_precision` and `sparsity` to trade off efficiency and accuracy. Refer to [Higher Accuracy](#higher-accuracy) for more details. We take the 14B model as example since it can get good output quality with SVG. Note that you need to install SVG before use:
```bash
python wan_t2v.py --attn_type "sparse-videogen" --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers"  --num_frames 81 --height 720 --width 1280 --num_timesteps_high_precision 0.075 --num_layers_high_precision 0.025
```

As for I2V, `num_timesteps_high_precision` is slightly different according to the SVG official implementation:
```bash
python wan_i2v.py --attn_type "sparse-videogen"  --num_timesteps_high_precision 0.09 --num_layers_high_precision 0.025
```

We follow the official SVG implementation, which use high precision attention in some denoising steps and layers to maintain output quality. To further speedup these parts, we can set the high precision type to "sage-attn":
```bash
python wan_t2v.py --attn_type "sparse-videogen" --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers"  --num_frames 81 --height 720 --width 1280 --num_timesteps_high_precision 0.075 --num_layers_high_precision 0.025 --high_precision_attn_type "sage-attn"
```

We also support it with multi-GPUs parallelism:
```bash
torchrun --nproc_per_node=4  wan_t2v.py --attn_type "sparse-videogen" --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers"  --num_frames 81 --height 720 --width 1280 --num_timesteps_high_precision 0.075 --num_layers_high_precision 0.025 --high_precision_attn_type "sage-attn" --cfg 2 --ulysses 2
```

*3. TRT-LLM Attention*

TRT-LLM has high performance implementation for Attention. It is in high precision (bfloat16) and we observed obvious speedup compared to the `default` PyTorch attention. It is not as fast as `sage-attn` because we don't use its quantized version for now.
```bash
python wan_t2v.py --attn_type "trtllm-attn"
```

*4. Flash Attention 3*

**FlashAttention-3** is especially optimized for **Hopper GPUs** (e.g., H100). After installing it, we can run it by:
```bash
python wan_t2v.py --attn_type "flash-attn3"
```

Leverage **FlashAttention-3 FP8** for further speedup:
```bash
python wan_t2v.py --attn_type "flash-attn3-fp8"
```


*5. Sparse VideoGen2*

[Sparse VideoGen2 (SVG2)](https://github.com/svg-project/Sparse-VideoGen/tree/main) leverages a semantic-aware permutation to maximize the accuracy of critical block identification and minimize the computation waste of sparse computation. We added an example to show how to use it. You can tune related options `num_timesteps_high_precision` and `num_layers_high_precision` to trade off efficiency and accuracy. Refer to [Higher Accuracy](#higher-accuracy) for more details. We take the 14B model as example since it can get good output quality with SVG2. Note that you need to install SVG2 before use:
```bash
python wan_t2v.py --attn_type "sparse-videogen2" --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers"  --num_frames 81 --height 720 --width 1280 --num_timesteps_high_precision 0.075 --num_layers_high_precision 0.025
```

As for I2V, `num_timesteps_high_precision` is slightly different according to the SVG official implementation:
```bash
python wan_i2v.py --attn_type "sparse-videogen2"  --num_timesteps_high_precision 0.09 --num_layers_high_precision 0.025
```

We follow the official SVG implementation, which use high precision attention in some denoising steps and layers to maintain output quality. To further speedup these parts, we can set the high precision type to "sage-attn":
```bash
python wan_t2v.py --attn_type "sparse-videogen2" --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers"  --num_frames 81 --height 720 --width 1280 --num_timesteps_high_precision 0.075 --num_layers_high_precision 0.025 --high_precision_attn_type "sage-attn"
```

We also support it with multi-GPUs parallelism:
```bash
torchrun --nproc_per_node=4  wan_t2v.py --attn_type "sparse-videogen2" --model_path "Wan-AI/Wan2.1-T2V-14B-Diffusers"  --num_frames 81 --height 720 --width 1280 --num_timesteps_high_precision 0.075 --num_layers_high_precision 0.025 --high_precision_attn_type "sage-attn" --cfg 2 --ulysses 2
```

**Linear**

Leverage the accelerated 4 & 8-bit tensor cores on NVIDIA GPUs using the various linear / gemm layer optimizations 

*1. NVFP4 Linear*

* TRT-LLM NVFP4 Linear
```bash
python wan_t2v.py --linear_type "trtllm-nvfp4" --linear_recipe static
```

* FlashInfer NVFP4 Linear
```bash
python wan_t2v.py --linear_type "flashinfer-nvfp4-cutlass" --linear_recipe static
```

If you set `linear_recipe` to `static`, it means we will use a default global scale for NVFP4 gemm. Otherwise, it will use `dynamic` strategy, the global scale is computed on the fly and thus have higher accuracy.

*2. FP8 Linear*

Enable TRT-LLM FP8 Linear for improved performance.
* TRT-LLM FP8 Blockwise Linear
```bash
python wan_t2v.py --linear_type "trtllm-fp8-blockwise"
```
* TRT-LLM FP8 Per-tensor Linear
```bash
python wan_t2v.py --linear_type "trtllm-fp8-per-tensor"
```

**TeaCache Acceleration:**

Enable TeaCache for improved inference speed:
```bash
python wan_t2v.py --enable_teacache
```

**Torch Compile**

[torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels. We compile each block of the transformer, which make sure cache methods can still work.

By default, we will enable `torch.compile` to get more speedup. To disable it, using:
```bash
python wan_t2v.py --disable_torch_compile
```


### Memory Optimization

**Blockwise Async CPU offload**

We provide an async weight offload strategy which copies model weights while computing. In this way, time of weight copy can be overlapped with computing, which reduces the offload overhead. Besides, it works in a block-wise manner, which means the offload is happening between blocks in `transformer.blocks` and these blocks can share the same GPU buffer and thus save more GPU memory than model-wise strategy.

We can enable the blockwise async cpu offload by:
```
python wan_t2v.py --enable_async_cpu_offload
```

What's more, we can set the offload stride, which is the interval to offload a transformer block. By default, the stride is set to 1 and every block will be offloaded. Higher stride means more blocks are retaining in GPU memory. We can set a higher stride to make it more likely to overlap data copy and computing, but this costs more GPU memory.
```
python wan_t2v.py --enable_async_cpu_offload --visual_gen_block_cpu_offload_stride 2
```


**Diffusers' CPU offload**

We inherit from [diffusers' pipeline class](https://huggingface.co/docs/diffusers/api/pipelines/overview), so we can use diffusers' CPU offload capabilities. Currently, diffusers includes two CPU offload methods: `model_cpu_offload` and `sequential_cpu_offload`. The former consumes more GPU memory but has higher inference efficiency; the latter consumes less GPU memory but has relatively lower inference efficiency. **However, we recommend using `Blockwise Async CPU offload` since it consumes less memory while having high performance through the overlapping strategy.**

When GPU memory is sufficient, we recommend using the former:
```
python wan_t2v.py --enable_model_cpu_offload
```
We can also enable it with multi-GPUs:
```
torchrun --nproc_per_node=4 wan_t2v.py --cfg 2 --ulysses 2 --enable_model_cpu_offload
```

If you still encountered OOM, then you can try:
```
python wan_t2v.py --enable_sequential_cpu_offload
```


**FSDP (Fully Sharded Data Parallel):**

If you are using multi-GPUs, we can leverage FSDP to shard model weights across GPUs. Note that this may reduce the inference efficiency, so only use FSDP if you encounter out-of-memory (OOM) errors:
```bash
torchrun --nproc_per_node=4 wan_t2v.py --cfg 2 --ulysses 2 --fsdp 4 --t5_fsdp 4
```


### Higher Accuracy

Since we can use low-bit or sparse Attention, such as Sparse VideoGen, this may influence the quality of generated videos. We provide the following options to keep Attentions of specified denoising steps or transformer layers in high precision.

*1. num_timesteps_high_precision*

When `timestep < num_inference_steps * num_timesteps_high_precision`, use high precision attention operators. Since `timestep` decreases during inference, higher `num_timesteps_high_precision` will lead to more denoising steps using high precision attention and achieving higher output accuracy.

*2. num_layers_high_precision*

When `layer idx < num_layers * num_layers_high_precision`, use high precision attention operators. So higher `num_layers_high_precision` will lead to more layers in transformers using high precision attention.

*3. high_precision_attn_type*

We can set the type of `high_precision_attn_type`. By default, it is `BaseAttn`, which uses PyTorch's `scaled_dot_product_attention`. We can set it to `sage-attn` or any attention type if you want to speed up the high precision parts. 

For example, if you set `attn_type` to `sparse-videogen` and set some layers or denoising steps to use high precision attention for good output quality, but you think `sage-attn` is good enough for the high precision parts and can help further accelerate the inference, then you can set `high_precision_attn_type` to `sage-attn`.


## Export and Load Quantized Checkpoint

By default, the quantization is processed on the fly, which may be finished in the warmup stage. However, if you want to reduce the warmup time or you have limited memory capacity in your online environment, we provide methods to save the quantized checkpoint previously and directly initialize the model from the quantized checkpoint in deployment.

Export quantized checkpoint:
```bash
python wan_t2v.py --linear_type "trtllm-fp8-blockwise" --disable_qkv_fusion --export_visual_gen_dit --visual_gen_ckpt_path ./visual_gen_dit
```

Load the exported checkpoint and run infernece:
```bash
python wan_t2v.py --linear_type "trtllm-fp8-blockwise" --disable_qkv_fusion --load_visual_gen_dit --visual_gen_ckpt_path ./visual_gen_dit
```

**Note**, we have to `disable_qkv_fusion` if want to export and load checkpoint. This will be fixed in the future.


## Performance Tips

1. **Memory Management**: Use FSDP/cpu_offload if you encounter OOM errors
2. **Speed Optimization**: Enable SageAttn/TeaCache/TRT-LLM FP8 Linear, and FP4 Linear for faster inference
3. **Resource Matching**: Match parallelization degrees to your available GPU count.
