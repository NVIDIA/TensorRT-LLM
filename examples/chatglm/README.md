# ChatGLM

This document explains how to build the [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b), [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) and [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b), [ChatGLM2-6B-32k](https://huggingface.co/THUDM/chatglm2-6b-32k), [ChatGLM3-6B-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) models using TensorRT-LLM and run on a single GPU, a single node with multiple GPUs or multiple nodes with multiple GPUs.

## Overview

The TensorRT-LLM ChatGLM implementation can be found in [`tensorrt_llm/models/chatglm/model.py`](../../tensorrt_llm/models/chatglm/model.py).
The TensorRT-LLM ChatGLM example code is located in [`examples/chatglm`](./). There are 3 main files in that folder:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the ChatGLM model.
* [`run.py`](./run.py) to run the inference on an input text.
* [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Support Matrix

* FP16
* Weight Only Quantization (int8 / int4)
* Paged KV cache
* Remove Input Padding
* Tensor Parallel
* Strongly Typed

## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Download repo and weights from HuggingFace Transformers

```bash
pip install -r requirements.txt
apt-get update
apt-get install git-lfs
rm -rf chatglm*

# clone one or more models we want to build
git clone https://huggingface.co/THUDM/chatglm-6b
git clone https://huggingface.co/THUDM/chatglm2-6b
git clone https://huggingface.co/THUDM/chatglm3-6b
git clone https://huggingface.co/THUDM/chatglm2-6b-32k
git clone https://huggingface.co/THUDM/chatglm3-6b-32k
```

### 2. Build TensorRT engine(s)

* This ChatGLM example in TensorRT-LLM builds TensorRT engine(s) using HF checkpoint directly (rather than using FT checkpoints such as GPT example).
* If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using dummy weights.
* The [`build.py`](./build.py) script requires a single GPU to build the TensorRT engine(s).
* You can enable parallel builds to accelerate the engine building process if you have more than one GPU in your system (of the same model).
* For parallel building, add the `--parallel_build` argument to the build command (this feature cannot take advantage of more than a single node).
* The number of TensorRT engines depends on the number of GPUs that will be used to run inference.
* argument [--model_version/-m] is required, which can be one of "1", "2", "3", "2-32k" or "3-32k" for ChatGLM-6B, ChatGLM2-6B, ChatGLM3-6B, ChatGLM2-6B-32K or ChatGLM3-6B-32K respectively.

#### Examples of build invocations

```bash
# Build a default engine of ChatGLM3-6B on single GPU with FP16, GPT Attention plugin, Gemm plugin, RMS Normolization plugin
python3 build.py -m 3

# Build a engine on single GPU with FMHA kernels (see introduction below), other configurations are the same as default example
python3 build.py -m 3 --enable_context_fmha  # or --enable_context_fmha_fp32_acc

# Build a engine on single GPU with int8/int4 Weight-Only quantization, other configurations are the same as default example
python3 build.py -m 3 --use_weight_only  # or --use_weight_only --weight_only_precision int4

# Build a engine on single GPU with int8_kv_cache and remove_input_padding, other configurations are the same as default example
python3 build.py -m 3 --paged_kv_cache --remove_input_padding

# Build a engine on two GPU, other configurations are the same as default example
python3 build.py -m 3 --world_size 2

# Build a engine of ChatGLM-6B on single GPU, other configurations are the same as default example
python3 build.py -m 1

# Build a engine of ChatGLM2-6B on single GPU, other configurations are the same as default example
python3 build.py -m 2

# Build a engine of ChatGLM2-6B-32k on single GPU, other configurations are the same as default example
python3 build.py -m 2-32k

# Build a engine of ChatGLM3-6B-32k on single GPU, other configurations are the same as default example
python3 build.py -m 3-32k
```

#### Enabled plugins

* Use `--use_gemm_plugin <DataType>` to configure GPT Attention plugin (default as float16)
* Use `--use_gemm_plugin <DataType>` to configure GEMM normolization plugin (default as float16)
* Use `--use_layernorm_plugin <DataType>` (for ChatGLM-6B) to configure RMS normolization plugin (default as float16)
* Use `--use_rmsnorm_plugin <DataType>` (for ChatGLM2-6B and ChatGLM3-6B) to configure RMS normolization plugin (default as float16)

#### Fused MultiHead Attention (FMHA)

* Use `--enable_context_fmha` or `--enable_context_fmha_fp32_acc` to enable FMHA kernels, which can provide better performance and low GPU memory occupation.

* Switch `--use_gpt_attention_plugin float16` must be used when using FMHA.

* `--enable_context_fmha` uses FP16 accumulator, which might cause low accuracy. In this case, `--enable_context_fmha_fp32_acc` should be used to protect accuracy at a cost of small performance drop.

#### Weight Only quantization

* Use `--use_weight_only` to enable INT8-Weight-Only quantization, this will siginficantly lower the latency and memory footprint.

* Furthermore, use `--weight_only_precision int8` or `--weight_only_precision int4` to configure the data type of the weights.

#### In-flight batching and paged KV cache [TODO]

* The engine must be built accordingly if [in-flight batching in C++ runtime](../../docs/in_flight_batching.md) will be used.

* Use `--use_inflight_batching` to enable In-flight Batching.

* Switch `--use_gpt_attention_plugin=float16`, `--paged_kv_cache`, `--remove_input_padding` will be set when using In-flight Batching.

* It is possible to use `--use_gpt_attention_plugin float32` In-flight Batching.

* The size of the block in paged KV cache can be conteoled additionally by using `--tokens_per_block=N`.

### 3. Run

#### Single node, single GPU

```bash
# Run the default engine of ChatGLM3-6B on single GPU, other model version is available if built.
python3 run.py -m 3
```

#### Single node, multi GPU

```bash
# Run the Tensor Parallel 2 engine of ChatGLM3-6B on two GPU, other model version is available if built.
mpirun -n 2 python run.py -m 3
```

* `--allow-run-as-root` might be needed if using `mpirun` as root.

#### Run comparison of performance and accuracy

```bash
# Run the summarization of ChatGLM3-6B task, other model version is available if built.
python3 summarize.py -m 3
```

## Benchmark

* The TensorRT-LLM ChatGLM benchmark is located in [benchmarks/](../../benchmarks/README.md)
