# BLOOM

This document shows how to build and run a BLOOM model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM BLOOM implementation can be found in [tensorrt_llm/models/bloom/model.py](../../tensorrt_llm/models/bloom/model.py). The TensorRT-LLM BLOOM example code is located in [`examples/bloom`](./). There are three main files in that folder::

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the BLOOM model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

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

### 4. Run

```bash
python summarize.py --test_trt_llm \
                    --hf_model_location ./bloom/560M/ \
                    --data_type fp16 \
                    --engine_dir ./bloom/560M/trt_engines/fp16/1-gpu/

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
