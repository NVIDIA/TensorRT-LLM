TRT-LLM visual_gen space contains prototype implementation for accelerating large-scale image & video content generation pipelines on NVIDIA Datacenter GPUs in single and multi-GPU configurations.

Given a diffusion model implemented in native PyTorch or HuggingFace Diffusers pipeline, you can accelerate compute heavy components of the pipeline by invoking low-precision kernels for linear and attention layers. 

## ✨ Key Features

<div align="center">

| 🎯 **Category** | 🚀 **Feature** | 📋 **Description** | 🎖️ **Status** |
|:---------------:|:---------------|:-------------------|:-------------:|
| **Attention Ops** | [Default](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | PyTorch's scaled dot product attention | ✅ |
| | [**SageAttention**](https://github.com/thu-ml/SageAttention) | 8-bit quantized attention for plug-and-play acceleration | ✅ |
| | [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM) Attention | High-performance attention optimized by TensorRT-LLM | ✅ |
| | [**Flash Attention 3**](https://github.com/Dao-AILab/flash-attention) | Memory-efficient attention optimized for Hopper GPUs | ✅ |
| | [**Flash Attention 3 - FP8**](https://github.com/Dao-AILab/flash-attention) | FP8 variant of Flash Attention 3 | ✅ |
| | [**Sparse VideoGen**](https://github.com/svg-project/Sparse-VideoGen) | Sparse attention for video generation acceleration | ✅ |
| **Linear Ops** | [Default](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) | Standard PyTorch linear layer | ✅ |
| | [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM) FP8 Blockwise | DeepSeek-like blockwise FP8 quantization | ✅ |
| | [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM) FP8 Per-Tensor | Per-tensor FP8 quantization | ✅ |
| | [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM) NVFP4 Blockwise | Double quantization + blockise NVFP4 | ✅ |
| | [**SVDQuant**](https://github.com/nunchaku-tech/nunchaku) NVFP4 |  SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models | 🚧 |
| **Hybrid Parallel** | **CFG Parallel** | Classifier-free guidance parallelism | ✅ |
| | **Ulysses Parallel** | Distributes attention heads to minimize communication | ✅ |
| | **Ring Parallel** | Ring topology for overlapping compute and communication | ✅ |
| | **Data Parallel (DP)** | Split batch across devices | ✅ |
| | **FSDP** | Fully Sharded Data Parallel | ✅ |
| **Diffusion Cache** | [**TeaCache**](https://github.com/ali-vilab/TeaCache) | Cache and reuse intermediate attention outputs | ✅ |
| **Memory Optimization** | **CPU Offloading** | Asynchronous block-wise CPU offloading | ✅ |
| **Others** | **Torch Compile** | JIT compilation for performance optimization | ✅ |
| | **QKV Fusion** | Fused query-key-value operations | ✅ |

</div>

> **Status Legend:**  
> ✅ **Available and ready to use**  
> 🚧 **In development or partial support**

## 📚 Supported Models

### Wan

| Model Name | Resolution | Parameters | Description |
|------------|------------|------------|-------------|
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | - | 1.3B | Text-to-video |
| Wan-AI/Wan2.1-T2V-14B-Diffusers | - | 14B | Text-to-video |
| Wan-AI/Wan2.1-I2V-14B-480P-Diffusers | 480P | 14B | Image-to-video |
| Wan-AI/Wan2.1-I2V-14B-720P-Diffusers | 720P | 14B | Image-to-video |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | 480P/720P | A14B | Text-to-video |
| Wan-AI/Wan2.2-I2V-A14B-Diffusers | 480P/720P | A14B | Image-to-video |

### Flux
| Model Name | Resolution | Parameters | Description |
|------------|------------|------------|-------------|
| black-forest-labs/FLUX.1-dev | - | 12B | Image generation |
| black-forest-labs/FLUX.2-dev | - | 32B | Image generation |

### Cosmos
| Model Name | Resolution | Parameters | Description |
|------------|------------|------------|-------------|
| nvidia/Cosmos-Predict2.5-2B-Base | - | 2B | Text-to-image |
| nvidia/Cosmos-Predict2-2B-Video2World | - | 2B | Video-to-world |
| nvidia/Cosmos-Predict2-14B-Video2World | - | 14B | Video-to-world |
| nvidia/Cosmos-Predict2-2B-Text2Image | - | 2B | Text-to-image |
| nvidia/Cosmos-Predict2-14B-Text2Image | - | 14B | Text-to-image |
| nvidia/Cosmos-1.0-Diffusion-7B-Text2World | - | 7B | Text-to-world |

## 📚 Supported GPUs

GPUs with arch `sm_89`, `sm_90`, `sm_100`, `sm_120` are supported. Other GPUs may work as well, but they have not been validated.

You can run this CML to know your GPU's sm arch:
```bash
nvidia-smi --query-gpu=name,compute_cap
```

## 🚀 Installation

Recommended docker: `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc4`

### Install
```bash
pip install -e .
```

### Install SageAttention
```bash
git clone https://github.com/thu-ml/SageAttention.git 3rdparty/SageAttention
cd 3rdparty/SageAttention/ && git checkout 96a29ccfaf3bcf5fada4731ac050456458f617f7 && pip install . --no-deps && cd -
```

### Install TransformerEngine (Optional)

To use TransformerEngine operators like Linear operators(`te-fp8-blockwise`, `te-fp8-per-tensor`), you need to install Transfomer Engine. 

We recommend you install TransformerEngine in the following way.
```bash
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine && git checkout release_v2.10
export NVTE_FRAMEWORK=pytorch         # Optionally set framework
pip3 install --no-build-isolation .   # Build and install
```

### Install Sparse VideoGen (Optional)

Follow the installation guide in the Sparse VideoGen [repo](https://github.com/svg-project/Sparse-VideoGen/tree/main). Then export the project to `PYTHONPATH`. For example, if the project is downloaded at `${Sparse_VideoGen_PATH}`:
```bash
export PYTHONPATH=$PYTHONPATH:${Sparse_VideoGen_PATH}
```

### Install Flash Attention v3/v4 (Optional)

Flash Attention V3 (FA3) provides higher speedup especially on **Hopper GPUs**. If you want to enable it, please follow the [instructions in FA3](https://github.com/Dao-AILab/flash-attention/tree/main?tab=readme-ov-file#flashattention-3-beta-release) to install it.

Recommend using this version which has been validated by us:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git 3rdparty/flash-attention
cd 3rdparty/flash-attention/ && git checkout 6c9eef9e2f93246bcb7d03e07c642a1c103e53d2 && cd -
```

Flash Attention V4 (FA4) provides higher speedup on **Blackwell GPUs (sm100)**. If you want to enable it, please add `./3rdparty/flash-attention/` to `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}/3rdparty/flash-attention/
```

## 🎯 Quick Start

### Core Code Snippets

```python
import torch
from visual_gen.pipelines.wan_pipeline import ditWanPipeline
from visual_gen.configs.parallel import VAEParallelConfig
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Define visual_gen configs to setup ops, diffusion cache methods, parallelism, etc.
dit_configs = {
    "teacache": {
        "enable_teacache": True,
        "use_ret_steps": True,
    },
    "attn": {
        "type": "default",
    },
    "linear": {
        "type": "default",
    },
    "parallel": {
        "dit_dp_size": 1,
        "dit_ulysses_size": 1,
        "dit_ring_size": 1,
        "dit_cfg_size": 1,
    },
}

VAEParallelConfig.set_config(
        disable_parallel_vae=True,
)
pipe = ditWanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, **dit_configs)

pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
export_to_video(frames, "wan-t2v.mp4", fps=16)

```

### Examples
We have some [examples](./examples/README.md) for popular video generation models.

## 🏗️ Architecture

### 🔧 Customized Operators

It's easy to customize an operator by the registry mechanism.

Let's take a customized attention as an example.

1. Register the customized attention in [./visual_gen/ops/attention.py](./visual_gen/ops/attention.py) or anywhere.
```python
from visual_gen.ops.attention import DefaultAttn
from visual_gen.configs.op_manager import AttentionOpManager

# Register a custom attention implementation
@AttentionOpManager.register_attn("custom")
class CustomAttn(BaseAttn):
    def __call__(self, query, key, value, attn_mask=None, **kwargs):
        # Your custom attention implementation
        pass
```

2. Specify the attention type and pass it to a ditPipeline
```python
dit_configs = {
    "attn": {
        "type": "custom",
    }
}

pipe = ditWanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, **dit_configs)

```

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
