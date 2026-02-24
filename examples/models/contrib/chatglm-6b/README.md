# ChatGLM

This document explains how to build the [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) models using TensorRT LLM and run on a single GPU, a single node with multiple GPUs or multiple nodes with multiple GPUs.

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
The TensorRT LLM ChatGLM example code is located in [`examples/models/contrib/chatglm-6b`](./). There is one main file:

* [`examples/models/core/glm-4-9b/convert_checkpoint.py`](../../../glm-4-9b/convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix

|    Model Name    | FP16  | FMHA  |  WO   |  SQ   |  AWQ  |  FP8  |  TP   |  PP   |  ST   |  C++  | benchmark |  IFB  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-------: | :---: |
|    chatglm_6b    |   Y   |       |   Y   |       |       |       |   Y   |       |   Y   |   Y   |     Y     |   Y   |
|     glm_10b      |   Y   |   Y   |   Y   |       |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |

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
|     glm_10b      |  48   |  64   |  32   |  64   | 4096  | 16384 | 1024  | 50304  |   Y   |   Y   |    Y    |                                                                    |

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
|     glm_10b      | GLMGPT2Tokenizer | 50257  | 50256  | 50256 | 50259 |    50257     |   50258    | 50260  | 50264 | 50263  |

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
git clone https://huggingface.co/THUDM/glm-10b          glm_10b

# replace tokenization file if using transformers-4.36.1 for model ChatGLM-6B (this might be needless in the future)
cp chatglm_6b/tokenization_chatglm.py chatglm_6b/tokenization_chatglm.py-backup
cp tokenization_chatglm.py chatglm_6b
```

For more example codes, please refer to the [examples/models/core/glm-4-9b/README.md](../../../glm-4-9b/README.md).
