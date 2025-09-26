# MPT

This document explains how to build the [MPT](https://huggingface.co/mosaicml/mpt-7b) model using TensorRT LLM and run on a single GPU and a single node with multiple GPUs.

- [MPT](#mpt)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
    - [MPT 7B](#mpt-7b)
      - [1.1 Convert from HF Transformers in FP](#11-convert-from-hf-transformers-in-fp)
      - [1.2 Convert from HF Transformers with weight-only quantization](#12-convert-from-hf-transformers-with-weight-only-quantization)
      - [1.3 Convert from HF Transformers with SmoothQuant quantization](#13-convert-from-hf-transformers-with-smoothquant-quantization)
      - [1.4 Convert from HF Transformers with INT8 KV cache quantization](#14-convert-from-hf-transformers-with-int8-kv-cache-quantization)
      - [1.5 AWQ weight-only quantization with Modelopt](#15-awq-weight-only-quantization-with-modelopt)
      - [1.6 FP8 Post-Training Quantization with Modelopt](#16-fp8-post-training-quantization-with-modelopt)
      - [1.6 Weight-only quantization with Modelopt](#16-weight-only-quantization-with-modelopt)
      - [1.7 SmoothQuant and INT8 KV cache with Modelopt](#17-smoothquant-and-int8-kv-cache-with-modelopt)
    - [2.1 Build TensorRT engine(s)](#21-build-tensorrt-engines)
    - [MPT 30B](#mpt-30b)
      - [1. Convert weights from HF Transformers to TRTLLM format](#1-convert-weights-from-hf-transformers-to-trtllm-format)
      - [2. Build TensorRT engine(s)](#2-build-tensorrt-engines)
      - [3. Run TRT engine to check if the build was correct](#3-run-trt-engine-to-check-if-the-build-was-correct)

## Overview

The TensorRT LLM MPT implementation can be found in [`tensorrt_llm/models/mpt/model.py`](../../tensorrt_llm/models/mpt/model.py). The TensorRT LLM MPT example code is located in [`examples/models/contrib/mpt`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * FP8 (with FP8 KV Cache)
  * INT8 & INT4 Weight-Only
  * INT8 Smooth Quant
  * INT4 AWQ
  * Tensor Parallel
  * MHA, MQA & GQA
  * STRONGLY TYPED

### MPT 7B

Please install required packages first:

```bash
pip install -r requirements.txt
```

The [`convert_checkpoint.py`](./convert_checkpoint.py) script allows you to convert weights from HF Transformers format to TRTLLM checkpoints.

#### 1.1 Convert from HF Transformers in FP

```bash
# Generate FP16 checkpoints.
python convert_checkpoint.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/fp16/ --dtype float16

# Generate FP32 checkpoints with TP=4.
python convert_checkpoint.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/fp32_tp4/ --dtype float32 --tp_size 4
```

#### 1.2 Convert from HF Transformers with weight-only quantization

```bash
# Use int8 weight-only quantization.
python convert_checkpoint.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/int8_wo/ --use_weight_only

# Use int4 weight-only quantization.
python convert_checkpoint.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/int4_wo/ --use_weight_only --weight_only_precision int4
```

#### 1.3 Convert from HF Transformers with SmoothQuant quantization

```bash
# Use int8 smoothquant (weight and activation) quantization.
python convert_checkpoint.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/int8_sq/ --smoothquant 0.5
```

#### 1.4 Convert from HF Transformers with INT8 KV cache quantization

```bash
# Use int8 kv cache quantization.
python convert_checkpoint.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/fp16_int8kv/ --dtype float16 --calibrate_kv_cache
```
***INT8-KV-cache can be used with SQ and Weight-only at the same time***


***We now introduce Modelopt to do all quantization***
First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

#### 1.5 AWQ weight-only quantization with Modelopt

```bash
# INT4 AWQ quantization using Modelopt.
python ../../../quantization/quantize.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/int4_awq/ --qformat int4_awq
```

#### 1.6 FP8 Post-Training Quantization with Modelopt

```bash
# FP8 quantization using Modelopt.
python ../../../quantization/quantize.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/fp8/ --qformat fp8 --kv_cache_dtype fp8
```

#### 1.6 Weight-only quantization with Modelopt

```bash
# INT8 Weight-only quantization using Modelopt with TP=2.
python ../../../quantization/quantize.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/int8_wo/ --qformat int8_wo --tp_size 2

# INT4 Weight-only quantization using Modelopt.
python ../../../quantization/quantize.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/int4_wo/ --qformat int4_wo
```

#### 1.7 SmoothQuant and INT8 KV cache with Modelopt

```bash
# Use int4 awq quantization.
python ../../../quantization/quantize.py --model_dir mosaicml/mpt-7b --output_dir ./ckpts/mpt-7b/sq_int8kv/ --qformat int8_sq --kv_cache_dtype int8
```
***INT8-KV-cache can also be used with Weight-only at the same time***


### 2.1 Build TensorRT engine(s)

All of the checkpoint generated by `convert_checkpoint.py` or `quantize.py` (Modelopt) can share the same building commands.

```bash
# Build a single-GPU float16 engine using TRTLLM checkpoints.
trtllm-build --checkpoint_dir=./ckpts/mpt-7b/fp16 \
             --max_batch_size 32 \
             --max_input_len 1024 \
             --max_seq_len 1536 \
             --gemm_plugin float16 \
             --workers 1 \
             --output_dir ./trt_engines/mpt-7b/fp16
```

### MPT 30B

Same commands can be changed to convert MPT 30B to TRT LLM format. Below is an example to build MPT30B fp16 4-way tensor parallelized TRT engine

#### 1. Convert weights from HF Transformers to TRTLLM format

The [`convert_checkpoint.py`](./convert_checkpoint.py) script allows you to convert weights from HF Transformers format to TRTLLM format.

```bash
python convert_checkpoint.py --model_dir mosaicml/mpt-30b --output_dir ./ckpts/mpt-30b/fp16_tp4/ --tp_szie 4 --dtype float16
```

#### 2. Build TensorRT engine(s)

Examples of build invocations:

```bash
# Build 4-GPU MPT-30B float16 engines
trtllm-build --checkpoint_dir ./ckpts/mpt-30b/fp16_tp4 \
             --max_batch_size 32 \
             --max_input_len 1024 \
             --max_seq_len 1536 \
             --gemm_plugin float16 \
             --workers 4 \
             --output_dir ./trt_engines/mpt-30b/fp16_tp4
```

#### 3. Run TRT engine to check if the build was correct

```bash
# Run 4-GPU MPT-30B TRT engine on a sample input prompt
mpirun -n 4 --allow-run-as-root \
    python ../../../run.py --max_output_len 10 \
                     --engine_dir ./trt_engines/mpt-30b/fp16/4-gpu/ \
                     --tokenizer_dir mosaicml/mpt-30b
```
