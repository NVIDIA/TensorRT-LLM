# Stable Diffusion XL

This document showcases how to build and run the [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model on multiple GPUs using TensorRT-LLM. The community-contributed SDXL example in TRT-LLM is intended solely to showcase distributed inference for high-resolution use cases. For an optimized single-GPU setup in Stable Diffusion inference, please refer to the [TensorRT DemoDiffusion example](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion).

The design of distributed parallel inference comes from the CVPR 2024 paper [DistriFusion](https://github.com/mit-han-lab/distrifuser) from [MIT HAN Lab](https://hanlab.mit.edu/). To simplify the implementation, all communications in this example are handled synchronously.

## Usage

### 1. Build TensorRT Engine

```bash
# 1 gpu
python build_sdxl_unet.py --size 1024

# 2 gpus
mpirun -n 2 --allow-run-as-root python build_sdxl_unet.py --size 1024
```

### 2. Generate images using the engine


```bash
# 1 gpu
python run_sdxl.py --size 1024 --prompt "flowers, rabbit"

# 2 gpus
mpirun -n 2 --allow-run-as-root python run_sdxl.py --size 1024 --prompt "flowers, rabbit"
```

## Latency Benchmark
This benchmark is provided as reference points and should not be considered as the peak inference speed that can be delivered by TensorRT-LLM.

| Framework | Resolution | n_gpu | A100 latency (s) | A100 speedup | H100 latency (s) | H100 speedup |
|:---------:|:----------:|:-----:|:---------------:|:-------------:|:---------------:|:-------------:|
|   Torch   |  1024x1024  |   1   |      6.280      |       1       |      5.820      |       1       |
|  TRT-LLM  |  1024x1024  |   2   |      2.803      |     **2.24x** |      1.719      |     **3.39x** |
|  TRT-LLM  |  1024x1024  |   4   |      2.962      |     **2.12x** |      2.592      |     **2.25x** |
|   Torch   |  2048x2048  |   1   |     27.865      |       1       |     18.330      |       1       |
|  TRT-LLM  |  2048x2048  |   2   |     13.152      |     **2.12x** |      7.943      |     **2.31x** |
|  TRT-LLM  |  2048x2048  |   4   |      9.781      |     **2.85x** |      7.596      |     **2.41x** |

torch v2.5.0. TRT-LLM v0.15.0.dev2024102900, `--num-warmup-runs=5; --avg-runs=20`. All communications are synchronous.
