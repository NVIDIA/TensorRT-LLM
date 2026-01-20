# ChatGLM

This document explains how to build the [glm-4-9b](https://huggingface.co/THUDM/glm-4-9b) models using TensorRT LLM and run on a single GPU, a single node with multiple GPUs or multiple nodes with multiple GPUs.

- [glm-4-9b](#glm-4-9b)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Model comparison](#model-comparison)
  - [Tokenizer and special tokens comparison](#tokenizer-and-special-tokens-comparison)
  - [Usage](#usage)
    - [1. Download repo and weights from HuggingFace Transformers](#1-download-repo-and-weights-from-huggingface-transformers)
    - [2. Convert weights from HF Transformers to TensorRT LLM format](#2-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
    - [3. Build TensorRT engine(s)](#3-build-tensorrt-engines)
      - [Enable plugins](#enable-plugins)
      - [In-flight batching](#in-flight-batching)
    - [4. Run inference](#4-run-inference)
      - [Single node, single GPU](#single-node-single-gpu)
      - [Single node, multi GPU](#single-node-multi-gpu)
    - [5. Run summarization task](#5-run-summarization-task)
    - [Weight Only quantization](#weight-only-quantization)
    - [Smooth Quantization (SQ)](#smooth-quantization-sq)
    - [Activation-aware Weight Quantization (AWQ)](#activation-aware-weight-quantization-awq)
    - [FP8 Quantization](#fp8-quantization)
  - [Benchmark](#benchmark)


## Overview

The TensorRT LLM ChatGLM implementation can be found in [`tensorrt_llm/models/chatglm/model.py`](../../../../tensorrt_llm/models/chatglm/model.py).
The TensorRT LLM ChatGLM example code is located in [`examples/models/core/glm-4-9b`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix

|    Model Name    | FP16  | FMHA  |  WO   |  SQ   |  AWQ  |  FP8  |  TP   |  PP   |  ST   |  C++  | benchmark |  IFB  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-------: | :---: |
|     glm_4_9b     |   Y   |   Y   |   Y   |       |       |       |   Y   |       |       |   Y   |           |       |
|  glm_4_9b_chat   |   Y   |   Y   |   Y   |       |       |       |   Y   |       |       |   Y   |           |       |

* Model Name: the name of the model, the same as the name on HuggingFace
* FMHA: Fused MultiHead Attention (see introduction below)
* WO: Weight Only Quantization (int8 / int4)
* SQ: Smooth Quantization (int8)
* AWQ: Activation Aware Weight Quantization (int4)
* FP8: FP8 Quantization
* TP: Tensor Parallel
* PP: Pipeline Parallel
* ST: Strongly Typed
* C++: C++ Runtime
* benchmark: benchmark by python / C++ Runtime
* IFB: In-flight Batching (see introduction below)

## Model comparison

|       Name       |  nL   |  nAH  |  nKH  |  nHW  |  nH   |  nF   | nMSL  |   nV   | bP2D  | bBQKV | bBDense | Comments                                                           |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: | :---: | :-----: | :----------------------------------------------------------------- |
|     glm_4_9b     |  40   |  32   |   2   |  128  | 4096  | 13696 | 8192  | 151552 |   N   |   Y   |    N    |                                                                    |
|  glm_4_9b_chat   |  40   |  32   |   2   |  128  | 4096  | 13696 | 8192  | 151552 |   N   |   Y   |    N    |                                                                    |

* nL: number of layers
* nAH: number of attention heads
* nKH: number of kv heads (less than nAH if multi_query_attention is used)
* nHW: head width
* nH: hidden size
* nF: FFN hidden size
* nMSL: max sequence length (input + output)
* nV: vocabulary size
* bP2D: use position_encoding_2d (Y: Yes, N: No)
* bBQKV: use bias for QKV multiplication in self-attention
* bBDense: use bias for Dense multiplication in self-attention

## Tokenizer and special tokens comparison

|       Name       |    Tokenizer     |  bos   |  eos   |  pad  |  cls  | startofpiece | endofpiece |  mask  | smask | gmask  |
| :--------------: | :--------------: | :----: | :----: | :---: | :---: | :----------: | :--------: | :----: | :---: | :----: |
|     glm_4_9b     | ChatGLM4Tokenizer |       | 151329 | 151329 |      |    151333    |            |        |       |        |
|  glm_4_9b_chat   | ChatGLM4Tokenizer |       | 151329 | 151329 |      |    151333    |            |        |       |        |

## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Download repo and weights from HuggingFace Transformers

```bash
pip install -r requirements.txt
apt-get update
apt-get install git-lfs

# clone one or more models we want to build
git clone https://huggingface.co/THUDM/glm-10b          glm_10b
git clone https://huggingface.co/THUDM/glm-4-9b         glm_4_9b
```

### 2. Convert weights from HF Transformers to TensorRT LLM format

The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT LLM checkpoints. The number of checkpoint files (in .safetensors format) is same to the number of GPUs used to run inference.

```bash
# GLM-4-9B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir glm_4_9b --output_dir trt_ckpt/glm_4_9b/fp16/1-gpu
```

### 3. Build TensorRT engine(s)

The `trtllm-build` command builds TensorRT LLM engines from TensorRT LLM checkpoints. The number of engine files is also same to the number of GPUs used to run inference.

Normally, the `trtllm-build` command only requires a single GPU, but you can enable parallel building by passing the number of GPUs to the `--workers` argument.

```bash
# GLM-4-9B: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/glm_4_9b/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/glm_4_9b/fp16/1-gpu
```

If the engines are run successfully, you will see output like (glm-4-9b as the example):

```txt
......
[01/26/2024-02:40:36] [TRT] [I] Engine generation completed in 136.52 seconds.
[01/26/2024-02:40:36] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1016 MiB, GPU 11909 MiB
[01/26/2024-02:40:36] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +11909, now: CPU 0, GPU 11909 (MiB)
[01/26/2024-02:40:40] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 29706 MiB
[01/26/2024-02:40:40] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:02:20
[01/26/2024-02:40:42] [TRT-LLM] [I] Serializing engine to trt_engines/glm_4_9b/fp16/1-gpu/rank0.engine...
[01/26/2024-02:42:29] [TRT-LLM] [I] Engine serialized. Total time: 00:01:47
[01/26/2024-02:42:30] [TRT-LLM] [I] Total time of building all engines: 00:05:19
```

#### Enable plugins

* Use `--gpt_attention_plugin <DataType>` to configure GPT Attention plugin (default as float16)
* Use `--gemm_plugin <DataType>` to configure GEMM plugin (default as float16)
* Use `--context_fmha enable` to enable FMHA kernels, which can provide better performance and low GPU memory occupation.
  * `--gpt_attention_plugin float16` must be used when using FMHA.

#### In-flight batching

* The engine(s) must be built accordingly if [in-flight batching in C++ runtime](../../docs/in_flight_batching.md) will be used.
* Use `--gpt_attention_plugin float16`, `--paged_kv_cache enable`, `--remove_input_padding enable` to build engine(s) supporting In-flight Batching.
  * It is possible to use `--gpt_attention_plugin float32` In-flight Batching.
  * The size of the block in paged KV cache can be controlled additionally by using `--tokens_per_block=N`.

### 4. Run inference

#### Single node, single GPU

```bash
# Run the default engine of GLM-4-9B on single GPU, other model name is available if built.
python3 ../../../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/fp16/1-gpu
```

#### Single node, multi GPU

```bash
# Run the Tensor Parallel 2 engine of glm_4_9b on two GPU, other model name is available if built.
mpirun -n 2 \
    python ../../../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/fp16/2-gpu
```

* `--allow-run-as-root` might be needed if using `mpirun` as root.
* `trtllm-build` flag `--context_fmha enable` uses FP16 accumulator, which might cause low accuracy. In this case, add `--enable_context_fmha_fp32_acc` to the inference command should be used to protect accuracy at a cost of small performance drop.

If the engines are run successfully, you will see output like (ChatGLM3-6B as the example):

```txt
......
Input [Text 0]: "[gMASK]sop What's new between ChatGLM3-6B and ChatGLM2-6B?"
Output [Text 0 Beam 0]: "There is no new information provided in the official documentation, but I found some differences in the code. The ChatGLM3-6B has an additional parameter called 'config', which is not present in ChatGLM2-6B. Additionally"
```

### 5. Run summarization task

```bash
# Run the summarization of glm_4_9b task, other model name is available if built.
python3 ../../../summarize.py --test_trt_llm \
        --hf_model_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/fp16/1-gpu
```

### Weight Only quantization

Use `--use_weight_only` to enable INT8-Weight-Only quantization, this will significantly lower the latency and memory footprint. Furthermore, use `--weight_only_precision int8` or `--weight_only_precision int4` to configure the data type of the weights.

```bash
# glm_4_9b: single gpu, int8 weight only quantization
python3 convert_checkpoint.py --model_dir glm_4_9b \
        --use_weight_only \
        --weight_only_precision int8 \
        --output_dir trt_ckpt/glm_4_9b/int8_wo/1-gpu

# glm_4_9b: single-gpu engine with int8 weight only quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/glm_4_9b/int8_wo/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/glm_4_9b/int8_wo/1-gpu

# Run inference.
python3 ../../../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/int8_wo/1-gpu
```

### Smooth Quantization (SQ)

Use `--smoothquant` to enable smooth quantization.

```bash
# glm_4_9b: single gpu, int8 smooth quantization
python3 convert_checkpoint.py --model_dir glm_4_9b \
        --smoothquant 0.5 \
        --per_channel \
        --per_token \
        --output_dir trt_ckpt/glm_4_9b/sq/1-gpu

# glm_4_9b: single-gpu engine with int8 smooth quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/glm_4_9b/sq/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/glm_4_9b/sq/1-gpu

# Run inference.
python3 ../../../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/sq/1-gpu
```

### Activation-aware Weight Quantization (AWQ)

The [`quantize.py`](../../../quantization/quantize.py) script can be used to quantize the models and export TensorRT LLM checkpoints.

```bash
# glm_4_9b: single gpu, int4 awq quantization
python ../../../quantization/quantize.py --model_dir glm_4_9b \
        --dtype float16 \
        --qformat int4_awq \
        --output_dir trt_ckpt/glm_4_9b/int4_awq/1-gpu

# glm_4_9b: single-gpu engine with int4 awq quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/glm_4_9b/int4_awq/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/glm_4_9b/int4_awq/1-gpu

# Run inference.
python3 ../../../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/int4_awq/1-gpu
```

### FP8 Quantization

The [`quantize.py`](../../../quantization/quantize.py) script can be used to quantize the models and export TensorRT LLM checkpoints.

```bash
# glm_4_9b: single gpu, fp8 quantization
python ../../../quantization/quantize.py --model_dir glm_4_9b \
        --dtype float16 \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --output_dir trt_ckpt/glm_4_9b/fp8/1-gpu

# glm_4_9b: single-gpu engine with fp8 quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/glm_4_9b/fp8/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/glm_4_9b/fp8/1-gpu

# Run inference.
python3 ../../../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/fp8/1-gpu
```
