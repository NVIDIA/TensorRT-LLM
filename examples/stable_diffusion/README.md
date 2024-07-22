# Stable Diffusion XL

This document elaborates how to build the [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model to runnable engines on single or multiple GPUs and perform a image generation task using these engines.

The design of distributed parallel inference comes from the CVPR 2024 paper [Distrifusion](https://github.com/mit-han-lab/distrifuser). In order to reduce the difficulty of implementation, all communications in the example are synchronous.

## Usage

### 1. Build TensorRT Engine(s)

```bash
# 1 gpu
python build_sdxl_unet.py --size 1024

# 2 gpus
mpirun -n 2 python build_sdxl_unet.py --size 1024
```

### 2. Generate images using the engine(s)


```bash
# 1 gpu
python run_sdxl.py --size 1024 --prompt "flowers, rabbit"

# 2 gpus
mpirun -n 2 python run_sdxl.py --size 1024 --prompt "flowers, rabbit"
```
