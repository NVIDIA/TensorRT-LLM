# ChatGLM2-6B

This is a repo to build and inference ChatGLM2-6B with TRT-LLM.
This document explains how to build the [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) model using TensorRT-LLM and run on a single GPU.

## Overview

The TensorRT-LLM ChatGLM2-6B implementation can be found in [`example/chatglm2-6b/model.py`](./model.py). The TensorRT-LLM ChatGLM2-6B example code is located in [`examples/chatglm2-6b`](./). There are serveral main files in that folder:

* [`build.py`](./build.py) to load a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT-LLM Chatglm2-6B network, and build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the ChatGLM2-6B model,
* [`run.py`](./run.py) to run the inference on an input text,


## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Prepare environment and download weights from HuggingFace Transformers

```bash
apt-get update
apt-get install git-lfs
git clone https://huggingface.co/THUDM/chatglm2-6b pyTorchModel
```

### 2. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) after loaded the weight from HuggingFace pytorch Model.

The [`build.py`](./build.py) script requires a single GPU to build the TensorRT engine(s).

Examples of build invocations:

```bash

python3 build.py --model_dir=./pyTorchModel \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16
```

#### Fused MultiHead Attention (FMHA)

You can enable the int 8 weight-only quantization by adding `--use_weight_only`, this will siginficantly lower the latency and memory footprint.

You can enable the FMHA kernels for ChatGLM2-6B by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

### 3. Run

#### Single node, single GPU

To run a TensorRT-LLM ChatGLM2-6B model on a single GPU, you can use `python3`:

```bash
# Run the ChatGLM2-6B model on a single GPU.
python3 run.py
```




## Benchmark

(TODO)
