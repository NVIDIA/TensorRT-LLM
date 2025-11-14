# BLOOM

This document shows how to build and run a BLOOM model in TensorRT LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Table of Contents

- [BLOOM](#bloom)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
      - [INT8 weight only + INT8 KV cache](#int8-weight-only--int8-kv-cache)
      - [SmoothQuant](#smoothquant)
      - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
    - [Run](#run)

## Overview

The TensorRT LLM BLOOM implementation can be found in [tensorrt_llm/models/bloom/model.py](../../tensorrt_llm/models/bloom/model.py). The TensorRT LLM BLOOM example code is located in [`examples/models/contrib/bloom`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * INT8 & INT4 Weight-Only
  * INT8 KV CACHE
  * Smooth Quant
  * Tensor Parallel
  * FP8 and FP8 KV cache

## Usage

The TensorRT LLM BLOOM example code locates at [examples/models/contrib/bloom](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Please install required packages first:

```bash
pip install -r requirements.txt
```

Need to prepare the HF BLOOM checkpoint by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/bloom.

e.g. To install BLOOM-560M

```bash
# Setup git-lfs
git lfs install
rm -rf ./bloom/560M
mkdir -p ./bloom/560M && git clone https://huggingface.co/bigscience/bloom-560m ./bloom/560M
```

TensorRT LLM BLOOM builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT LLM will build engine(s) with dummy weights.

Normally `trtllm-build` only requires single GPU, but if you've already got all the GPUs needed for inference, you could enable parallel building to make the engine building process faster by adding `--workers` argument. Please note that currently `workers` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# Try gemm_plugin to prevent accuracy issue. TODO check this holds for BLOOM

# Single GPU on BLOOM 560M
python convert_checkpoint.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --output_dir ./bloom/560M/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir ./bloom/560M/trt_ckpt/fp16/1-gpu/ \
                --gemm_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/fp16/1-gpu/

# Build the BLOOM 560M using a single GPU and apply INT8 weight-only quantization.
python convert_checkpoint.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --use_weight_only \
                --output_dir ./bloom/560M/trt_ckpt/int8_weight_only/1-gpu/
trtllm-build --checkpoint_dir ./bloom/560M/trt_ckpt/int8_weight_only/1-gpu/ \
                --gemm_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/int8_weight_only/1-gpu/

# Use 2-way tensor parallelism on BLOOM 560M
python convert_checkpoint.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --output_dir ./bloom/560M/trt_ckpt/fp16/2-gpu/ \
                --tp_size 2
trtllm-build --checkpoint_dir ./bloom/560M/trt_ckpt/fp16/2-gpu/ \
                --gemm_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/fp16/2-gpu/

# Use 8-way tensor parallelism on BLOOM 176B
# Currently, TensorRT does not support tensors with more than 2^31-1 elements,
# so we have to shard the embedding table to multi-GPUs.

# sharding embedding table in the vocab dimension (the lookup plugin is optional)
python convert_checkpoint.py --model_dir ./bloom/176B/ \
                --dtype float16 \
                --output_dir ./bloom/176B/trt_ckpt/fp16/8-gpu/ \
                --tp_size 8 \
                --use_parallel_embedding \
                --embedding_sharding_dim 0
trtllm-build --checkpoint_dir ./bloom/176B/trt_ckpt/fp16/8-gpu/ \
                --gemm_plugin float16 \
                --output_dir ./bloom/176B/trt_engines/fp16/8-gpu/ \
                --workers 2

# sharding embedding table in the hidden dimension
python convert_checkpoint.py --model_dir ./bloom/176B/ \
                --dtype float16 \
                --output_dir ./bloom/176B/trt_ckpt/fp16/8-gpu/ \
                --tp_size 8 \
                --use_parallel_embedding \
                --embedding_sharding_dim 1
trtllm-build --checkpoint_dir ./bloom/176B/trt_ckpt/fp16/8-gpu/ \
                --gemm_plugin float16 \
                --output_dir ./bloom/176B/trt_engines/fp16/8-gpu/ \
                --workers 2

# share embedding table between embedding() and lm_head() layers
# To reduce the generated engine size, we can turn off gemm plugin and shard the embedding table in the vocab dimension.
python convert_checkpoint.py --model_dir ./bloom/176B/ \
                --dtype float16 \
                --output_dir ./bloom/176B/trt_ckpt/fp16/8-gpu/ \
                --tp_size 8 \
                --use_parallel_embedding \
                --embedding_sharding_dim 0
trtllm-build --checkpoint_dir ./bloom/176B/trt_ckpt/fp16/8-gpu/ \
                --output_dir ./bloom/176B/trt_engines/fp16/8-gpu/ \
                --workers 2
```

#### INT8 weight only + INT8 KV cache


For INT8 KV cache, [`convert_checkpoint.py`](./convert_checkpoint.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache.

In addition, it could be combined with INT8 weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python convert_checkpoint.py --model_dir ./bloom/560m/ \
                --dtype float16 \
                --int8_kv_cache \
                --use_weight_only --output_dir ./bloom/560m/trt_ckpt/int8/1-gpu/
trtllm-build --checkpoint_dir ./bloom/560m/trt_ckpt/int8/1-gpu/ \
                --gemm_plugin float16 \
                --output_dir ./bloom/560m/trt_engines/int8/1-gpu/ \
```


#### SmoothQuant

Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python convert_checkpoint.py --model_dir bloom/560M/  --output_dir tllm_checkpoint_1gpu --smoothquant 0.5 --per_token --per_channel
```

[`convert_checkpoint.py`](./convert_checkpoint.py) add new options for the support of INT8 inference of SmoothQuant models.

`--smoothquant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.



```bash
# Build model for SmoothQuant with below command.

trtllm-build  --checkpoint_dir tllm_checkpoint_1gpu  --output_dir ./engine_outputs
```
Note that GPT attention plugin is required to be enabled for SmoothQuant for now.


Note we use `--bin_model_dir` instead of `--model_dir` since SmoothQuant model needs INT8 weights and various scales from the binary files.

#### FP8 Post-Training Quantization

```
# Quantize HF Bloom 3B into FP8 and export trtllm checkpoint
python ../../../quantization/quantize.py --model_dir /home/scratch.trt_llm_data/llm-models/bloom-3b \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir /tmp/bloom/3b/trt_ckpts/fp8/1-gpu/ \
                                   --calib_size 512 \
                                   --tp_size 1

trtllm-build --checkpoint_dir /tmp/bloom/3b/trt_ckpts/fp8/1-gpu/ \
             --output_dir /tmp/bloom/3b/trt_engines/fp8/1-gpu/ \
             --gemm_plugin float16 \
             --use_fp8_context_fmha enable \
             --workers 1
```

### Run

```bash
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./bloom/560M/ \
                       --data_type fp16 \
                       --engine_dir ./bloom/560M/trt_engines/fp16/1-gpu/

python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./bloom/560M/ \
                       --data_type fp16 \
                       --engine_dir ./bloom/560M/trt_engines/int8_weight_only/1-gpu/

mpirun -n 2 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm \
                           --hf_model_dir ./bloom/560M/ \
                           --data_type fp16 \
                           --engine_dir ./bloom/560M/trt_engines/fp16/2-gpu/

mpirun -n 8 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm \
                           --hf_model_dir ./bloom/176B/ \
                           --data_type fp16 \
                           --engine_dir ./bloom/176B/trt_engines/fp16/8-gpu/

python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir /home/scratch.trt_llm_data/llm-models/bloom-3b \
                       --data_type fp16 \
                       --engine_dir /tmp/bloom/3b/trt_engines/fp8/1-gpu/
```
