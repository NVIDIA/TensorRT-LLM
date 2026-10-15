# Deployment Guide for FLUX.3 Klein on TensorRT LLM

## Introduction

This is a quick-start guide for running the FLUX.3 Klein image generation models with TensorRT LLM VisualGen. It covers the Base and distilled checkpoints in BF16, FP8, and NVFP4 variants, generation with the example script, and serving through `trtllm-serve`. For feature details, see the [Visual Generation](../models/visual-generation.md) doc.

## Prerequisites

* GPU: NVIDIA Hopper or Blackwell for the BF16 and FP8 variants. The NVFP4 variant requires Blackwell.
* OS: Linux
* Docker with NVIDIA Container Toolkit installed

## Models

* [FLUX.3 Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base): `black-forest-labs/flux-3-image-klein-base` (text-to-image and image-to-image; supports CFG parallelism)
* [FLUX.3 Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein): `black-forest-labs/flux-3-image-klein` (text-to-image and image-to-image; no CFG parallelism)

Weight variants are selected with `--variant` in the example script, or with `pipeline_config.variant` in a YAML config:

* BF16: omit the variant (root package)
* FP8: `fp8r`
* NVFP4: `nvfp4`

## Deployment Steps

### Run Docker Container

Build and run the docker container. See the [Docker guide](source:docker/README.md) for details.

```shell
cd TensorRT-LLM

make -C docker release_build IMAGE_TAG=flux3-local

make -C docker release_run IMAGE_NAME=tensorrt_llm IMAGE_TAG=flux3-local LOCAL_USER=1
```

Inside the container, the example scripts and configs are at `/app/tensorrt_llm/examples/visual_gen`.

### Generate Images with the Example Script

Run from `/app/tensorrt_llm/examples/visual_gen`:

```shell
python models/flux3.py --model black-forest-labs/flux-3-image-klein-base \
  --visual_gen_args configs/flux3-images-1gpu.yaml \
  --prompt "A red fox standing in fresh snow at sunrise" \
  --output_path flux3.png
```

To use the packaged FP8 or NVFP4 weights, add `--variant`:

```shell
python models/flux3.py --model black-forest-labs/flux-3-image-klein-base \
  --variant nvfp4 --visual_gen_args configs/flux3-images-1gpu.yaml
```

For image-to-image, pass one or more reference images with `--image`:

```shell
python models/flux3.py --model black-forest-labs/flux-3-image-klein-base \
  --visual_gen_args configs/flux3-images-1gpu.yaml \
  --image /path/to/image.png --prompt "Change the yarn color to blue"
```

The distilled checkpoint runs the same way with `--model black-forest-labs/flux-3-image-klein`.

### Launch the TensorRT LLM Server

Native FLUX.3 checkpoints need `--visual_gen_args` to select VisualGen:

```shell
trtllm-serve black-forest-labs/flux-3-image-klein-base \
  --host 0.0.0.0 --port 8000 \
  --visual_gen_args configs/flux3-images-1gpu.yaml
```

For the FP8 or NVFP4 variant, set `pipeline_config: {variant: fp8r}` or `variant: nvfp4` in the YAML file.

The server exposes OpenAI-compatible image endpoints, including `POST /v1/images/generations`. See [`examples/visual_gen/serve/`](source:examples/visual_gen/serve/README.md) for request examples.

## Multi-GPU Parallelism

The 1-GPU config is a starting point. For CFG, Ulysses, or tensor parallelism, see the [Multi-GPU Parallelism](../models/visual-generation.md) section of the Visual Generation doc.
