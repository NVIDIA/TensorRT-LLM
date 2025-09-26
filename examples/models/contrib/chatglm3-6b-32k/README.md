# ChatGLM

This document explains how to build the [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b), [ChatGLM3-6B-Base](https://huggingface.co/THUDM/chatglm3-6b-base), [ChatGLM3-6B-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) models using TensorRT LLM and run on a single GPU, a single node with multiple GPUs or multiple nodes with multiple GPUs.

- [ChatGLM](#chatglm)
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

The TensorRT LLM ChatGLM implementation can be found in [`tensorrt_llm/models/chatglm/model.py`](../../tensorrt_llm/models/chatglm/model.py).
The TensorRT LLM ChatGLM example code is located in [`examples/models/contrib/chatglm3-6b-32k`](./). There is one main file:

* [`examples/models/core/glm-4-9b/convert_checkpoint.py`](../../../glm-4-9b/convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix

|    Model Name    | FP16  | FMHA  |  WO   |  SQ   |  AWQ  |  FP8  |  TP   |  PP   |  ST   |  C++  | benchmark |  IFB  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-------: | :---: |
|   chatglm3_6b    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm3_6b_base |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm3_6b_32k  |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |

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
|   chatglm3_6b    |  28   |  32   |   2   |  128  | 4096  | 13696 | 8192  | 65024  |   N   |   Y   |    N    | Different in preprocess and postprocess than chatglm2_6b           |
| chatglm3_6b_base |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    |                                                                    |
| chatglm3_6b_32k  |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    | RoPE base=500000 rather than 10000 in chatglm3_6b                  |

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
|   chatglm3_6b    | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
| chatglm3_6b_base | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
| chatglm3_6b_32k  | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |

## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Download repo and weights from HuggingFace Transformers

```bash
pip install -r requirements.txt
apt-get update
apt-get install git-lfs
rm -rf chatglm*

# clone one or more models we want to build
git clone https://huggingface.co/THUDM/chatglm3-6b      chatglm3_6b
git clone https://huggingface.co/THUDM/chatglm3-6b-base chatglm3_6b_base
git clone https://huggingface.co/THUDM/chatglm3-6b-32k  chatglm3_6b_32k
```

For more example codes, please refer to the [examples/models/core/glm-4-9b/README.md](../../../glm-4-9b/README.md).
