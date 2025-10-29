# Command R

This document explains how to build the [C4AI Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01), [C4AI Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus), [Aya-23-8B](https://huggingface.co/CohereForAI/aya-23-8B), [Aya-23-35B](https://huggingface.co/CohereForAI/aya-23-35B) models using TensorRT LLM and run on a single GPU or a single node with multiple GPUs.

- [Command R](#Command-R)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [1. Download repo and weights from HuggingFace Transformers](#1-download-repo-and-weights-from-huggingface-transformers)
    - [2. Convert weights from HF Transformers to TensorRT LLM format](#2-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
    - [3. Build TensorRT engine(s)](#3-build-tensorrt-engines)
    - [4. Run inference](#4-run-inference)
      - [Single node, single GPU](#single-node-single-gpu)
      - [Single node, multi GPU](#single-node-multi-gpu)
    - [5. Run summarization task](#5-run-summarization-task)
    - [Weight Only quantization](#weight-only-quantization)


## Overview

The TensorRT LLM Command-R implementation can be found in [`tensorrt_llm/models/commandr/model.py`](../../../../tensorrt_llm/models/commandr/model.py).
The TensorRT LLM Command-R example code is located in [`examples/models/core/commandr`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix

  * FP16
  * INT8 & INT4 Weight-Only
  * Tensor Parallel

## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Download repo and weights from HuggingFace Transformers

```bash
pip install -r requirements.txt
apt-get update
apt-get install git-lfs

# clone one or more models we want to build
git clone https://huggingface.co/CohereForAI/c4ai-command-r-v01         command_r_v01
git clone https://huggingface.co/CohereForAI/c4ai-command-r-plus        command_r_plus
git clone https://huggingface.co/CohereForAI/aya-23-8B                  aya_23_8B
git clone https://huggingface.co/CohereForAI/aya-23-35B                 aya_23_35B
```

### 2. Convert weights from HF Transformers to TensorRT LLM format

The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT LLM checkpoints. The number of checkpoint files (in .safetensors format) is same to the number of GPUs used to run inference.

```bash
# Command-R: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir command_r_v01 --output_dir trt_ckpt/command_r_v01/fp16/1-gpu

# Command-R+: 4-way tensor parallelism
python3 convert_checkpoint.py --model_dir command_r_plus --tp_size 4 --output_dir trt_ckpt/command_r_plus/fp16/4-gpu

# Aya-23-8B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir aya_23_8B --output_dir trt_ckpt/aya_23_8B/fp16/1-gpu

# Aya-23-35B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir aya_23_35B --output_dir trt_ckpt/aya_23_35B/fp16/1-gpu
```

### 3. Build TensorRT engine(s)

The `trtllm-build` command builds TensorRT LLM engines from TensorRT LLM checkpoints. The number of engine files is also same to the number of GPUs used to run inference.

Normally, the `trtllm-build` command only requires a single GPU, but you can enable parallel building by passing the number of GPUs to the `--workers` argument.

```bash
# Command-R: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/command_r_v01/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/command_r_v01/fp16/1-gpu

# Command-R+: 4-way tensor parallelism
trtllm-build --checkpoint_dir trt_ckpt/command_r_plus/fp16/4-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/command_r_plus/fp16/4-gpu

# Command-R: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/aya_23_8B/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/aya_23_8B/fp16/1-gpu

# Command-R: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/aya_23_35B/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/aya_23_35B/fp16/1-gpu
```

If the engines are built successfully, you will see output like (Command-R as the example):

```txt
......
[09/19/2024-03:34:30] [TRT] [I] Engine generation completed in 26.9495 seconds.
[09/19/2024-03:34:30] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 4 MiB, GPU 70725 MiB
[09/19/2024-03:34:55] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 176260 MiB
[09/19/2024-03:34:55] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:52
[09/19/2024-03:34:55] [TRT] [I] Serialized 26 bytes of code generator cache.
[09/19/2024-03:34:55] [TRT] [I] Serialized 315007 bytes of compilation cache.
[09/19/2024-03:34:55] [TRT] [I] Serialized 12 timing cache entries
[09/19/2024-03:34:55] [TRT-LLM] [I] Timing cache serialized to model.cache
[09/19/2024-03:34:55] [TRT-LLM] [I] Build phase peak memory: 176257.29 MB, children: 17.65 MB
[09/19/2024-03:34:55] [TRT-LLM] [I] Serializing engine to trt_engines/command_r_v01/fp16/1-gpu/rank0.engine...
[09/19/2024-03:35:20] [TRT-LLM] [I] Engine serialized. Total time: 00:00:25
[09/19/2024-03:35:23] [TRT-LLM] [I] Total time of building all engines: 00:01:47
```

### 4. Run inference

#### Single node, single GPU

```bash
# Run the default engine of Command-R on single GPU.
python3 ../../../run.py --max_output_len 50 \
        --tokenizer_dir command_r_v01 \
        --engine_dir trt_engines/command_r_v01/fp16/1-gpu

# Run the default engine of Command-R on single GPU, using streaming output.
python3 ../../../run.py --max_output_len 50 \
        --tokenizer_dir command_r_v01 \
        --engine_dir trt_engines/command_r_v01/fp16/1-gpu \
        --streaming

# Run the default engine of Aya-23-8B on single GPU.
python3 ../../../run.py --max_output_len 50 \
        --tokenizer_dir aya_23_8B \
        --engine_dir trt_engines/aya_23_8B/fp16/1-gpu

# Run the default engine of Aya-23-35B on single GPU.
python3 ../../../run.py --max_output_len 50 \
        --tokenizer_dir aya_23_35B \
        --engine_dir trt_engines/aya_23_35B/fp16/1-gpu
```

#### Single node, multi GPU

```bash
# Run the Tensor Parallel 4 engine of Command-R+ on 4 GPUs.
mpirun -n 4 \
    python ../../../run.py  --max_output_len 50 \
        --tokenizer_dir command_r_plus \
        --engine_dir trt_engines/command_r_plus/fp16/4-gpu
```

If the engines are run successfully, you will see output like (Command-R as the example):

```txt
......
Input [Text 0]: "<BOS_TOKEN>Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef in Paris and worked in the kitchens of the French royal family. He came to England in 1814 and worked in a number of London hotels and restaurants, including the Reform Club and the London Tavern. He also opened his own restaurant"
```

### 5. Run summarization task

```bash
# Run the summarization of Command-R task.
python3 ../../../summarize.py --test_trt_llm \
        --hf_model_dir command_r_v01 \
        --engine_dir trt_engines/command_r_v01/fp16/1-gpu
```

If the engines are run successfully, you will see output like (Command-R as the example):

```txt
......
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT LLM (total latency: 81.05689692497253 sec)
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT LLM (total output tokens: 2000)
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT LLM (tokens per second: 24.67402621952367)
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT LLM beam 0 result
[01/26/2024-02:51:56] [TRT-LLM] [I]   rouge1 : 24.06804397902119
[01/26/2024-02:51:56] [TRT-LLM] [I]   rouge2 : 6.456513335555016
[01/26/2024-02:51:56] [TRT-LLM] [I]   rougeL : 16.77644999660741
[01/26/2024-02:51:56] [TRT-LLM] [I]   rougeLsum : 20.57359472317834
```

### Weight Only quantization

Use `--use_weight_only` to enable INT8-Weight-Only quantization, this will significantly lower the latency and memory footprint. Furthermore, use `--weight_only_precision int8` or `--weight_only_precision int4` to configure the data type of the weights.

```bash
# Command-R: single gpu, int8 weight only quantization
python3 convert_checkpoint.py --model_dir command_r_v01 \
        --use_weight_only \
        --weight_only_precision int8 \
        --output_dir trt_ckpt/command_r_v01/int8_wo/1-gpu

# Command-R: single-gpu engine with int8 weight only quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/command_r_v01/int8_wo/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/command_r_v01/int8_wo/1-gpu

# Run inference.
python3 ../../../run.py --max_output_len 50 \
        --tokenizer_dir command_r_v01 \
        --engine_dir trt_engines/command_r_v01/int8_wo/1-gpu
```
