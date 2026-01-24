# Cosmos Predict2.5

## Overview

This example shows how to integrate our plug-and-play feature to quickly support Cosmos Predict2.5, which is not a diffusers dependent model.

## APIs
We supported it with a few APIs. 

### 1. `visual_gen.setup_configs`

It is used to specify the operator and parallel strategy we would like to use.

Regarding operator selection, we rely on some kernels from TRT-LLM, Sage Attention and TE. We are trying to integrate or develop more kernels. All operators have a `default` option, which means use the default PyTorch API. However, we provide more efficient kernels with good accuracy. 

For **attention operators on Hopper**, we support `sage-attn`, `te` and `te-fp8`, for **attention operators on Blackwell**, we recommend you to try both `te`, `te-fp8` and `flash-attn4`, to see which have best perf/accuracy in your case.

For **linear layers on Hopper**, we support `trtllm-fp8-blockwise`, `trtllm-fp8-per-tensor`, `te-fp8-blockwise` and `te-fp8-per-tensor`. For **linear layers on Blackwell**, we support `trtllm-nvfp4`, and `te-fp8-per-tensor`. More gemms on Blackwell are still in development.

For multi-gpu usage, we can use `ulysses`, `ring` and `cp`. `ulysses` is the recommended one.

### 2. convert to visual_gen's modules

For attention modules, we can set cosmos `attn_op` to a `ditCosmosAttention` instance. Then every thing will be automatically handled, including to operator selection and parallel.

for `Linear` or `RMSNorm` modules, we can directly call `apply_visual_gen_linear` and `apply_visual_gen_norm` to replace it with visual_gen's.

That's all we need to support Cosmos Predict2.5!

## Installation

### Install Cosmos Predict2.5

Just clone codes from the official repo of Cosmos Predict2.5 and set up environments as their [guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md#installation). Currently, our codes are based on this commit `07563da` of Cosmos Predict2.5.

Recommend to use [their docker](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md#docker-container) and activate the environments:

```bash
# Pre-Blackwell
uv sync --extra=cu128
source .venv/bin/activate
```

```bash
# Blackwell
apt update && apt install -y libgl1 libglib2.0-0
uv sync --extra=cu130
source .venv/bin/activate
```

### Install visual_gen

After that we can install what required by `visual_gen`.

For Hopper GPUs:
```bash
uv pip install -e . --no-build-isolation --prerelease=allow --no-deps
uv pip install flashinfer-python
cd 3rdparty/SageAttention && uv pip install . --no-deps  --no-build-isolation && cd -
# trtllm
apt update && apt install libopenmpi-dev -y
uv pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm==1.0.0rc4
export LD_LIBRARY_PATH="$(
  find / -name 'libpython3.10.so.1.0' -print0 \
    | xargs -0 dirname \
    | tr '\n' ':' \
    | sed 's/:$//'
):$LD_LIBRARY_PATH"
```

For Blackwell GPUs:
```bash
uv pip install -e . --no-build-isolation --prerelease=allow --no-deps
uv pip install flashinfer-python
```
Install FlashAttention4, which is included as a 3rd party module [here](../../cosmos-predict2.5-inference.py/3rdparty/flash-attention/flash_attn/cute).
```bash
export PYTHONPATH=$PYTHONPATH:${visual_gen_root_path}/3rdparty/flash-attention/
```

Some kernels from TransformerEngine can work for both Hopper and Blackwell GPUs. So we can install TE in both GPUs:
```bash
uv pip install pybind11
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine && git checkout release_v2.10
export NVTE_FRAMEWORK=pytorch         # Optionally set framework
uv pip install --no-build-isolation .   # Build and install
```

## Run

Copy `inference.py` and `multiview.py` to Cosmos Predict2.5's [exmaples folder](https://github.com/nvidia-cosmos/cosmos-predict2.5/tree/main/examples). They will overwrite the original files so that you can see what we modified by `git diff`.

After that, you can run the codes follow Cosmos's [guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference.md#example),
