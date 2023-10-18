# BLOOM

This document shows how to build and run a BLOOM model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM BLOOM implementation can be found in [tensorrt_llm/models/bloom/model.py](../../tensorrt_llm/models/bloom/model.py). The TensorRT-LLM BLOOM example code is located in [`examples/bloom`](./). There are three main files in that folder::

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the BLOOM model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Support Matrix
  * FP16
  * INT8 & INT4 Weight-Only
  * INT8 KV CACHE
  * Smooth Quant
  * Tensor Parallel

## Usage

The TensorRT-LLM BLOOM example code locates at [examples/bloom](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF BLOOM checkpoint first by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/bloom.

e.g. To install BLOOM-560M

```bash
# Setup git-lfs
git lfs install
rm -rf ./bloom/560M
mkdir -p ./bloom/560M && git clone https://huggingface.co/bigscience/bloom-560m ./bloom/560M
```

TensorRT-LLM BLOOM builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed for inference, you could enable parallel building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# Try use_gemm_plugin to prevent accuracy issue. TODO check this holds for BLOOM

# Single GPU on BLOOM 560M
python build.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/fp16/1-gpu/

# Build the BLOOM 560M using a single GPU and apply INT8 weight-only quantization.
python build.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --use_weight_only \
                --output_dir ./bloom/560M/trt_engines/int8_weight_only/1-gpu/

# Use 2-way tensor parallelism on BLOOM 560M
python build.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/fp16/2-gpu/ \
                --world_size 2

# Use 8-way tensor parallelism on BLOOM 176B
# Currently, TensorRT does not support tensors with more than 2^31-1 elements,
# so we have to shard the embedding table to multi-GPUs.

# sharding embedding table in the vocab dimension (the lookup plugin is optional)
python build.py --model_dir ./bloom/176B/ \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./bloom/176B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --use_parallel_embedding \
                --embedding_sharding_dim 0 \
                --use_lookup_plugin  float16

# sharding embedding table in the hidden dimension
python build.py --model_dir ./bloom/176B/ \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./bloom/176B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --use_parallel_embedding \
                --embedding_sharding_dim 1

# share embedding table between embedding() and lm_head() layers
# To reduce the generated engine size, we has to use gemm and lookup plugin (--use_gemm_plugin --use_lookup_plugin) and must shard the embedding table in the vocab dimension.
python build.py --model_dir ./bloom/176B/ \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./bloom/176B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --use_parallel_embedding \
                --embedding_sharding_dim 0 \
                --use_lookup_plugin float16 \
                --use_embedding_sharing
```

#### INT8 weight only + INT8 KV cache
For INT8 KV cache, [`hf_bloom_convert.py`](./hf_bloom_convert.py) features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
python3 hf_bloom_convert.py -i bloom/560M -o ./c-model/bloom/int8_kv_cache/560M --calibrate-kv-cache -t float16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache.

In addition, it could be combined with INT8 weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python build.py --bin_model_dir=./c-model/bloom/int8_kv_cache/560M/1-gpu \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_layernorm_plugin \
                --int8_kv_cache \
                --use_weight_only
```

#### SmoothQuant

Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 hf_bloom_convert.py -i bloom/560M -o ./c-model/bloom-smooth/560M --smoothquant 0.5 --tensor-parallelism 1 --storage-type float16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_tensor_ mode.
python3 build.py --bin_model_dir=./c-model/bloom-smooth/560M/1-gpu \
                 --use_smooth_quant --use_gpt_attention_plugin float16

# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --bin_model_dir=./c-model/bloom-smooth/560M/1-gpu \
                 --use_smooth_quant --use_gpt_attention_plugin float16 \
                 --per_token \
                 --per_channel
```
Note that GPT attention plugin is required to be enabled for SmoothQuant for now.


Note we use `--bin_model_dir` instead of `--model_dir` since SmoothQuant model needs INT8 weights and various scales from the binary files.

### 4. Run

```bash
python summarize.py --test_trt_llm \
                    --hf_model_location ./bloom/560M/ \
                    --data_type fp16 \
                    --engine_dir ./bloom/560M/trt_engines/fp16/1-gpu/

python summarize.py --test_trt_llm \
                    --hf_model_location ./bloom/560M/ \
                    --data_type fp16 \
                    --engine_dir ./bloom/560M/trt_engines/int8_weight_only/1-gpu/

mpirun -n 2 --allow-run-as-root \
    python summarize.py --test_trt_llm \
                        --hf_model_location ./bloom/560M/ \
                        --data_type fp16 \
                        --engine_dir ./bloom/560M/trt_engines/fp16/2-gpu/

mpirun -n 8 --allow-run-as-root \
    python summarize.py --test_trt_llm \
                        --hf_model_location ./bloom/176B/ \
                        --data_type fp16 \
                        --engine_dir ./bloom/176B/trt_engines/fp16/8-gpu/
```
