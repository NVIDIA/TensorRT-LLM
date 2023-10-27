# ChatGLM2-6B

This document explains how to build the [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) model using TensorRT-LLM and run on a single GPU.

## Overview

The TensorRT-LLM ChatGLM2-6B implementation can be found in [`tensorrt_llm/models/chatglm2_6b/model.py`](../../tensorrt_llm/models/chatglm6b/model.py).
The TensorRT-LLM ChatGLM2-6B example code is located in [`examples/chatglm2-6b`](./). There are 3 main files in that folder:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the ChatGLM-6B model.
* [`run.py`](./run.py) to run the inference on an input text.
* [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Prepare environment and download weights from HuggingFace Transformers

```bash
apt-get update
apt-get install git-lfs
git clone https://huggingface.co/THUDM/chatglm2-6b pyTorchModel
```

### 2. Build TensorRT engine(s)

+ This ChatGLM2-6B example in TensorRT-LLM builds TensorRT engine(s) using HF checkpoint directly (rather than using FT checkpoints such as GPT example).
+ If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using dummy weights.
+ The [`build.py`](./build.py) script requires a single GPU to build the TensorRT engine(s).
+ You can enable parallel builds to accelerate the engine building process if you have more than one GPU in your system (of the same model).
+ For parallel building, add the `--parallel_build` argument to the build command (this feature cannot take advantage of more than a single node).
+ The number of TensorRT engines depends on the number of GPUs that will be used to run inference.

#### Examples of build invocations:

```bash
# Build a single-GPU float16 engine using FT weights.
# --use_gpt_attention_plugin must be used to deal with inputs with different length in one batch
# --use_gemm_plugin, --use_layernorm_plugin, --enable_context_fmha, --enable_context_fmha_fp32_acc are used to improve accuracy or performance.
python3 build.py --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16
```

#### INT8 Weight Only

+ Enable the int 8 weight-only quantization by adding `--use_weight_only`, this will siginficantly lower the latency and memory footprint.

#### Fused MultiHead Attention (FMHA)

+ Use `--enable_context_fmha` or `--enable_context_fmha_fp32_acc` to enable FMHA kernels, which can provide better performance and low GPU memory occupation.

+ Switch `--use_gpt_attention_plugin float16` must be used when using FMHA.

+ `--enable_context_fmha` uses FP16 accumulator, which might cause low accuracy. In this case, `--enable_context_fmha_fp32_acc` should be used to protect accuracy at a cost of small performance drop.

#### In-flight batching and paged KV cache

+ The engine must be built accordingly if [in-flight batching in C++ runtime](../../docs/in_flight_batching.md) will be used.

+ Use `--use_inflight_batching` to enable In-flight Batching.

+ Switch `--use_gpt_attention_plugin=float16`, `--paged_kv_cache`, `--remove_input_padding` will be set when using In-flight Batching.

+ It is possible to use `--use_gpt_attention_plugin float32` In-flight Batching.

+ The size of the block in paged KV cache can be conteoled additionally by using `--tokens_per_block=N`.

### 3. Run

#### Single node, single GPU

Run TensorRT-LLM ChatGLM-6B model on a single GPU

```bash
# Run the ChatGLM2-6B model on a single GPU.
python3 run.py
```

Run comparison of performance and accuracy

```bash
# Run the summarization task.
python3 summarize.py
```

## Benchmark

+ [TODO] The TensorRT-LLM ChatGLM2-6B benchmark is located in [benchmarks/](../../benchmarks/README.md)
