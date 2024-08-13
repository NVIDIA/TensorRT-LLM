# ChatGLM

This document explains how to build the [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b), [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b), [ChatGLM2-6B-32k](https://huggingface.co/THUDM/chatglm2-6b-32k), [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b), [ChatGLM3-6B-Base](https://huggingface.co/THUDM/chatglm3-6b-base), [ChatGLM3-6B-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) models using TensorRT-LLM and run on a single GPU, a single node with multiple GPUs or multiple nodes with multiple GPUs.

- [ChatGLM](#chatglm)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Model comparison](#model-comparison)
  - [Tokenizer and special tokens comparison](#tokenizer-and-special-tokens-comparison)
  - [Usage](#usage)
    - [1. Download repo and weights from HuggingFace Transformers](#1-download-repo-and-weights-from-huggingface-transformers)
    - [2. Convert weights from HF Transformers to TensorRT-LLM format](#2-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
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

The TensorRT-LLM ChatGLM implementation can be found in [`tensorrt_llm/models/chatglm/model.py`](../../tensorrt_llm/models/chatglm/model.py).
The TensorRT-LLM ChatGLM example code is located in [`examples/chatglm`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT-LLM format.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix

|    Model Name    | FP16  | FMHA  |  WO   |  SQ   |  AWQ  |  FP8  |  TP   |  PP   |  ST   |  C++  | benchmark |  IFB  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-------: | :---: |
|    chatglm_6b    |   Y   |       |   Y   |       |       |       |   Y   |       |   Y   |   Y   |     Y     |   Y   |
|   chatglm2_6b    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm2_6b_32k  |   Y   |   Y   |   Y   |       |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
|   chatglm3_6b    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm3_6b_base |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm3_6b_32k  |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
|     glm_10b      |   Y   |   Y   |   Y   |       |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
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
|    chatglm_6b    |  28   |  32   |  32   |  128  | 4096  | 16384 | 2048  | 130528 |   Y   |   Y   |    Y    |                                                                    |
|   chatglm2_6b    |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    | Multi_query_attention, RMSNorm rather than LayerNorm in chatglm_6b |
| chatglm2_6b_32k  |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    | RoPE base=160000 rather than 10000 in chatglm2_6b                  |
|   chatglm3_6b    |  28   |  32   |   2   |  128  | 4096  | 13696 | 8192  | 65024  |   N   |   Y   |    N    | Different in preprocess and postprocess than chatglm2_6b           |
| chatglm3_6b_base |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    |                                                                    |
| chatglm3_6b_32k  |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    | RoPE base=500000 rather than 10000 in chatglm3_6b                  |
|     glm_10b      |  48   |  64   |  32   |  64   | 4096  | 16384 | 1024  | 50304  |   Y   |   Y   |    Y    |                                                                    |
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
|    chatglm_6b    | ChatGLMTokenizer | 130004 | 130005 |   3   |       |    130004    |   130005   | 130000 |       | 130001 |
|   chatglm2_6b    | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            |        |       |        |
| chatglm2_6b_32k  | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            |        |       |        |
|   chatglm3_6b    | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
| chatglm2_6b_base | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
| chatglm2_6b_32k  | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
|     glm_10b      | GLMGPT2Tokenizer | 50257  | 50256  | 50256 | 50259 |    50257     |   50258    | 50260  | 50264 | 50263  |
|     glm_4_9b     | ChatGLM4Tokenizer |       | 151329 | 151329 |      |    151333    |            |        |       |        |
|  glm_4_9b_chat   | ChatGLM4Tokenizer |       | 151329 | 151329 |      |    151333    |            |        |       |        |

## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Download repo and weights from HuggingFace Transformers

```bash
pip install -r requirements.txt
apt-get update
apt-get install git-lfs
rm -rf chatglm*

# clone one or more models we want to build
git clone https://huggingface.co/THUDM/chatglm-6b       chatglm_6b
git clone https://huggingface.co/THUDM/chatglm2-6b      chatglm2_6b
git clone https://huggingface.co/THUDM/chatglm2-6b-32k  chatglm2_6b_32k
git clone https://huggingface.co/THUDM/chatglm3-6b      chatglm3_6b
git clone https://huggingface.co/THUDM/chatglm3-6b-base chatglm3_6b_base
git clone https://huggingface.co/THUDM/chatglm3-6b-32k  chatglm3_6b_32k
git clone https://huggingface.co/THUDM/glm-10b          glm_10b
git clone https://huggingface.co/THUDM/glm-4-9b         glm_4_9b

# replace tokenization file if using transformers-4.36.1 for model ChatGLM-6B (this might be needless in the future)
cp chatglm_6b/tokenization_chatglm.py chatglm_6b/tokenization_chatglm.py-backup
cp tokenization_chatglm.py chatglm_6b
```

### 2. Convert weights from HF Transformers to TensorRT-LLM format

The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT-LLM checkpoints. The number of checkpoint files (in .safetensors format) is same to the number of GPUs used to run inference.

```bash
# ChatGLM3-6B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir chatglm3_6b --output_dir trt_ckpt/chatglm3_6b/fp16/1-gpu

# ChatGLM3-6B: 2-way tensor parallelism
python3 convert_checkpoint.py --model_dir chatglm3_6b --tp_size 2 --output_dir trt_ckpt/chatglm3_6b/fp16/2-gpu

# Chatglm2-6B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir chatglm2_6b --output_dir trt_ckpt/chatglm2_6b/fp16/1-gpu

# Chatglm-6B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir chatglm_6b --output_dir trt_ckpt/chatglm_6b/fp16/1-gpu

# GLM-10B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir glm_10b --output_dir trt_ckpt/glm_10b/fp16/1-gpu

# GLM-4-9B: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir glm_4_9b --output_dir trt_ckpt/glm_4_9b/fp16/1-gpu
```

### 3. Build TensorRT engine(s)

The `trtllm-build` command builds TensorRT-LLM engines from TensorRT-LLM checkpoints. The number of engine files is also same to the number of GPUs used to run inference.

Normally, the `trtllm-build` command only requires a single GPU, but you can enable parallel building by passing the number of GPUs to the `--workers` argument.

Using ChatGLM2-6B-32K / ChatGLM3-6B-32K models, we need to guarantee `max_batch_size * max_beam_width * max_seq_len <= 78398 = 2^31 / (13696 * 2)` due to constrain of TensorRT. For example, we will fail to build engine while using default max_batch_size (8) and adding arguments `--max_beam_width=4 --max_input_len=20000 --max_seq_len=20100`.

```bash
# ChatGLM3-6B: single-gpu engine
trtllm-build --checkpoint_dir trt_ckpt/chatglm3_6b/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# ChatGLM3-6B: 2-way tensor parallelism
trtllm-build --checkpoint_dir trt_ckpt/chatglm3_6b/fp16/2-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm3_6b/fp16/2-gpu

# ChatGLM2-6B: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/chatglm2_6b/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm2_6b/fp16/1-gpu

# ChatGLM-6B: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/chatglm_6b/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm_6b/fp16/1-gpu

# GLM-10B: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/glm_10b/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/glm_10b/fp16/1-gpu

# GLM-4-9B: single-gpu engine with dtype float16, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/glm_4_9b/fp16/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/glm_4_9b/fp16/1-gpu
```

If the engines are run successfully, you will see output like (ChatGLM3-6B as the example):

```txt
......
[01/26/2024-02:40:36] [TRT] [I] Engine generation completed in 136.52 seconds.
[01/26/2024-02:40:36] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1016 MiB, GPU 11909 MiB
[01/26/2024-02:40:36] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +11909, now: CPU 0, GPU 11909 (MiB)
[01/26/2024-02:40:40] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 29706 MiB
[01/26/2024-02:40:40] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:02:20
[01/26/2024-02:40:42] [TRT-LLM] [I] Serializing engine to trt_engines/chatglm3_6b/fp16/1-gpu/rank0.engine...
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
# Run the default engine of ChatGLM3-6B on single GPU, other model name is available if built.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu

# Run the default engine of ChatGLM3-6B on single GPU, using streaming output, other model name is available if built.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu \
        --streaming

# Run the default engine of GLM3-10B on single GPU, other model name is available if built.
# Token "[MASK]" or "[sMASK]" or "[gMASK]" must be included in the prompt as the original model commanded.
python3 ../run.py --input_text "Peking University is [MASK] than Tsinghua University." \
        --max_output_len 50 \
        --tokenizer_dir glm_10b \
        --engine_dir trt_engines/glm_10b/fp16/1-gpu

# Run the default engine of GLM-4-9B on single GPU, other model name is available if built.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir glm_4_9b \
        --engine_dir trt_engines/glm_4_9b/fp16/1-gpu
```

#### Single node, multi GPU

```bash
# Run the Tensor Parallel 2 engine of ChatGLM3-6B on two GPU, other model name is available if built.
mpirun -n 2 \
    python ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/fp16/2-gpu
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
# Run the summarization of ChatGLM3-6B task, other model name is available if built.
python3 ../summarize.py --test_trt_llm \
        --hf_model_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu
```

If the engines are run successfully, you will see output like (ChatGLM3-6B as the example):

```txt
......
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT-LLM (total latency: 12.688004493713379 sec)
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT-LLM (total output tokens: 1390)
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT-LLM (tokens per second: 109.5522941128145)
[01/26/2024-02:51:56] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[01/26/2024-02:51:56] [TRT-LLM] [I]   rouge1 : 23.926583062537716
[01/26/2024-02:51:56] [TRT-LLM] [I]   rouge2 : 6.945058457209619
[01/26/2024-02:51:56] [TRT-LLM] [I]   rougeL : 17.89273173719794
[01/26/2024-02:51:56] [TRT-LLM] [I]   rougeLsum : 21.22686350784501
```

### Weight Only quantization

Use `--use_weight_only` to enable INT8-Weight-Only quantization, this will significantly lower the latency and memory footprint. Furthermore, use `--weight_only_precision int8` or `--weight_only_precision int4` to configure the data type of the weights.

```bash
# ChatGLM3-6B: single gpu, int8 weight only quantization
python3 convert_checkpoint.py --model_dir chatglm3_6b \
        --use_weight_only \
        --weight_only_precision int8 \
        --output_dir trt_ckpt/chatglm3_6b/int8_wo/1-gpu

# ChatGLM3-6B: single-gpu engine with int8 weight only quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/chatglm3_6b/int8_wo/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm3_6b/int8_wo/1-gpu

# Run inference.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/int8_wo/1-gpu
```

### Smooth Quantization (SQ)

Use `--smoothquant` to enable smooth quantization.

```bash
# ChatGLM3-6B: single gpu, int8 smooth quantization
python3 convert_checkpoint.py --model_dir chatglm3_6b \
        --smoothquant 0.5 \
        --per_channel \
        --per_token \
        --output_dir trt_ckpt/chatglm3_6b/sq/1-gpu

# ChatGLM3-6B: single-gpu engine with int8 smooth quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/chatglm3_6b/sq/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm3_6b/sq/1-gpu

# Run inference.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/sq/1-gpu
```

### Activation-aware Weight Quantization (AWQ)

The [`../quantization/quantize.py`](../quantization/quantize.py) script can be used to quantize the models and export TensorRT-LLM checkpoints.

```bash
# ChatGLM3-6B: single gpu, int4 awq quantization
python ../quantization/quantize.py --model_dir chatglm3_6b \
        --dtype float16 \
        --qformat int4_awq \
        --output_dir trt_ckpt/chatglm3_6b/int4_awq/1-gpu

# ChatGLM3-6B: single-gpu engine with int4 awq quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/chatglm3_6b/int4_awq/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm3_6b/int4_awq/1-gpu

# Run inference.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/int4_awq/1-gpu
```

### FP8 Quantization

The [`../quantization/quantize.py`](../quantization/quantize.py) script can be used to quantize the models and export TensorRT-LLM checkpoints.

```bash
# ChatGLM3-6B: single gpu, fp8 quantization
python ../quantization/quantize.py --model_dir chatglm3_6b \
        --dtype float16 \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --output_dir trt_ckpt/chatglm3_6b/fp8/1-gpu

# ChatGLM3-6B: single-gpu engine with fp8 quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir trt_ckpt/chatglm3_6b/fp8/1-gpu \
        --gemm_plugin float16 \
        --output_dir trt_engines/chatglm3_6b/fp8/1-gpu

# Run inference.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
        --max_output_len 50 \
        --tokenizer_dir chatglm3_6b \
        --engine_dir trt_engines/chatglm3_6b/fp8/1-gpu
```

### Long context evaluation

* SlimPajama-6B with PPL evaluation

SlimPajama-6B is a dataset which contains long context inputs. We use this dataset to run PPL evaluation.

```bash
git-lfs clone https://huggingface.co/datasets/DKYoon/SlimPajama-6B
```

```bash
git-lfs clone https://huggingface.co/THUDM/chatglm3-6b-128k


python examples/chatglm/convert_checkpoint.py --model_dir chatglm3-6b-128k \
                              --output_dir /tmp/chatglm3-6b-128k/trt_ckpts \
                              --dtype float16

python -m tensorrt_llm.commands.build --checkpoint_dir /tmp/chatglm3-6b-128k/trt_ckpts \
            --output_dir /tmp/chatglm3-6b-128k/trt_engines \
            --gemm_plugin float16 \
            --gather_all_token_logits \
            --max_batch_size 8 \
            --max_input_len 25600 \
            --max_num_tokens 25600

python examples/summarize.py --engine_dir /tmp/chatglm3-6b-128k/trt_engines \
                             --tokenizer_dir chatglm3-6b-128k \
                             --dataset_dir ./ \
                             --eval_task eval_context_ppl \
                             --test_trt_llm \
                             --hf_model_dir chatglm3-6b-128k  \
                             --max_input_len 25600 \
                             --batch_size 1 \
                             --max_ite 1600 \
                             --check_accuracy \
                             --tensorrt_llm_ppl_threshold 14 \
                             --use_py_session

[05/14/2024-08:01:49] [TRT-LLM] [I] TensorRT-LLM (total latency: 71.61979579925537 sec)
[05/14/2024-08:01:49] [TRT-LLM] [I] TensorRT-LLM (total output tokens: 1599)
[05/14/2024-08:01:49] [TRT-LLM] [I] TensorRT-LLM (tokens per second: 22.32622953131381)
[05/14/2024-08:01:49] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[05/14/2024-08:01:53] [TRT-LLM] [I]   Per-token perplexity: 13.595022946447134
```

* needle in haystack (passkey) evaluation

```bash
python3 examples/infinitebench/construct_synthetic_dataset.py --test_case build_passkey

python examples/chatglm/convert_checkpoint.py --model_dir chatglm3-6b-128k \
                              --output_dir /tmp/chatglm3-6b-128k/trt_ckpts \
                              --dtype float16

python -m tensorrt_llm.commands.build --checkpoint_dir /tmp/chatglm3-6b-128k/trt_ckpts \
            --output_dir /tmp/chatglm3-6b-128k/trt_engines \
            --gemm_plugin float16 \
            --gather_all_token_logits \
            --max_batch_size 1 \
            --max_input_len 12800 \
            --max_num_tokens 12800

python examples/eval_long_context.py  --task passkey \
                                      --engine_dir /tmp/chatglm3-6b-128k/trt_engines \
                                      --tokenizer_dir chatglm3-6b-128k \
                                      --stop_idx 20 \
                                      --max_input_length 12800 \
                                      --use_py_session

```

## Benchmark

* The TensorRT-LLM ChatGLM benchmark is located in [benchmarks/](../../benchmarks/README.md)
