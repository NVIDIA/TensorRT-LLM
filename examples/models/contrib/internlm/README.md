# InternLM

This document shows how to build and run InternLM 7B / 20B models in TensorRT LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

- [InternLM](#internlm)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
      - [INT8 weight only + INT8 KV cache](#int8-weight-only--int8-kv-cache)
      - [SmoothQuant](#smoothquant)
    - [Run](#run)
    - [Summarization using the InternLM model](#summarization-using-the-internlm-model)

## Overview

The TensorRT LLM InternLM implementation is based on the LLaMA model. The implementation can
be found in [tensorrt_llm/models/llama/model.py](../../tensorrt_llm/models/llama/model.py).
The TensorRT LLM InternLM example code lies in [`examples/models/contrib/internlm`](./):

* [`convert_checkpoint.py`](../../../llama/convert_checkpoint.py) converts the Huggingface Model of InternLM into TensorRT LLM checkpoint.
* [`convert_checkpoint.py`] to to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16 / BF16
  * INT8 & INT4 Weight-Only
  * Smooth Quant
  * INT8 KV Cache
  * Tensor Parallel & Pipeline Parallel

## Usage

The TensorRT LLM InternLM example code locates at [examples/models/contrib/internlm](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Please install required packages first:

```bash
pip install -r requirements.txt
```

TensorRT LLM InternLM builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT LLM will build engine(s) with dummy weights.

InternLM has released several checkpoints of different size or capabilities under https://huggingface.co/internlm. Users can pick any one repository and follow instructions to prepare the checkpoint.

Below examples use [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) and [internlm-chat-20b](https://huggingface.co/internlm/internlm-chat-20b) and assume these repositories are cloned or linked under this directory, for example `./internlm-chat-7b/`.

Normally `trtllm-build` only requires single GPU, but if you've already got all the GPUs needed for inference, you could enable parallel building to make the engine building process faster by adding `--workers` argument. Please note that currently `--workers` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# gpt_attention_plugin is necessary in InternLM.
# Try use_gemm_plugin to prevent accuracy issue.
cd examples/models/core/llama

# Convert the InternLM 7B model using a single GPU and FP16.
python convert_checkpoint.py --model_dir ./internlm-chat-7b/ \
                --dtype float16 \
                --output_dir ./internlm-chat-7b/trt_engines/fp16/1-gpu/
# Note: setting `--dtype bfloat16` to use bfloat16 precision.

# BUild the InternLM 7B model using a single GPU
trtllm-build --checkpoint_dir ./internlm-chat-7b/trt_engines/fp16/1-gpu/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16

# Convert the InternLM 7B model using a single GPU and apply INT8 weight-only quantization..
python convert_checkpoint.py --model_dir ./internlm-chat-7b/ \
                --dtype float16 \
                --output_dir ./internlm-chat-7b/trt_engines/int8/1-gpu/ \
                --use_weight_only \
                --weight_only_precision int8

trtllm-build --checkpoint_dir ./internlm-chat-7b/trt_engines/int8/1-gpu/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16

# Note: setting `--weight_only_precision int4` to use INT4 weight-only quantization

# Build InternLM 7B using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./internlm-chat-7b/ \
                --dtype float16 \
                --output_dir ./internlm-chat-7b/trt_engines/fp16/2-gpu/ \
                --tp_size 2

trtllm-build --checkpoint_dir ./internlm-chat-7b/trt_engines/fp16/2-gpu/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16

# Build InternLM 20B using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./internlm-chat-20b/ \
                --dtype bfloat16 \
                --output_dir ./internlm-chat-20b/trt_engines/bf16/2-gpu/ \
                --tp_size 2 --workers 2

trtllm-build --checkpoint_dir ./internlm-chat-7b/trt_engines/bf16/2-gpu/ \
             --output_dir ./engine_outputs \
             --gpt_attention_plugin bfloat16  \
             --gemm_plugin bfloat16
```

#### INT8 weight only + INT8 KV cache

For INT8 KV cache, [`convert_checkpoint.py`](./convert_checkpoint.py) features a
`--int8_kv_cache` option. Setting `--int8_kv_cache` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
cd examples/models/core/llama

# For 7B models
python convert_checkpoint.py --model_dir ./internlm-chat-7b  \
                             --output_dir ./internlm-chat-7b/smooth_internlm/int8_kv_cache/ \
                             --dtype float16  \
                             --use_weight_only \
                             --weight_only_precision int8 \
                             --int8_kv_cache

# Build 7B model with both INT8 weight-only and INT8 KV cache enabled
trtllm-build --checkpoint_dir ./internlm-chat-7b/smooth_internlm/int8_kv_cache/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16 \
```


```bash
cd examples/models/core/llama

# For 20B models
python convert_checkpoint.py --model_dir ./internlm-chat-20b  \
                             --output_dir ./internlm-chat-20b/smooth_internlm/int8_kv_cache/ \
                             --dtype float16  \
                             --use_weight_only \
                             --weight_only_precision int8 \
                             --int8_kv_cache

# Build 20B model with both INT8 weight-only and INT8 KV cache enabled
trtllm-build --checkpoint_dir ./internlm-chat-20b/smooth_internlm/int8_kv_cache/ \
  --output_dir ./engine_outputs \
  --gemm_plugin float16 \
```


Test with `../../../run.py` or `../../../summarize.py`:

```bash
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir ./internlm-chat-7b/trt_engines/int8_kv_cache_weight_only/1-gpu

python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-20b/ \
                 --engine_dir ./internlm-chat-20b/trt_engines/int8_kv_cache_weight_only/1-gpu

python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-7b/trt_engines/int8_kv_cache_weight_only/1-gpu

python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-20b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-20b/trt_engines/int8_kv_cache_weight_only/1-gpu
```

#### SmoothQuant

Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
cd examples/models/core/llama

# For 7B models
python convert_checkpoint.py --model_dir ./internlm-chat-7b  --output_dir ./internlm-chat-7b/smooth_internlm/sq0.5/ --dtype float16 --smoothquant 0.5
# Build the engine
trtllm-build --checkpoint_dir ./internlm-chat-7b/smooth_internlm/sq0.5/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16

# For 20B models
cd examples/models/core/llama

python convert_checkpoint.py --model_dir ./internlm-chat-20b  --output_dir ./internlm-chat-20b/smooth_internlm/sq0.5/ --dtype float16 --smoothquant 0.5
trtllm-build --checkpoint_dir ./internlm-chat-20b/smooth_internlm/sq0.5/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16
```

[`convert_checkpoint.py`](./convert_checkpoint.py) add new options for the support of INT8 inference of SmoothQuant models.

`--smoothquant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
cd examples/models/core/llama

# 7B model
python convert_checkpoint.py --model_dir ./internlm-chat-7b  --output_dir ./internlm-chat-7b/smooth_internlm/sq0.5/ --dtype float16 --smoothquant 0.5 --per_channel --per_token

# 20B model
python convert_checkpoint.py --model_dir ./internlm-chat-20b  --output_dir ./internlm-chat-20b/smooth_internlm/sq0.5/ --dtype float16 --smoothquant 0.5 --per_channel --per_token
```


Test with `../../../run.py` or `../../../summarize.py`:

```bash
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir ./internlm-chat-7b/smooth_internlm/sq0.5/

python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-20b/ \
                 --engine_dir ./internlm-chat-20b/smooth_internlm/sq0.5/

python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-7b/smooth_internlm/sq0.5/

python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-20b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-20b/smooth_internlm/sq0.5/
```

### Run

To run a TensorRT LLM InternLM model using the engines generated by `trtllm-build`

```bash
# InternLM 7B with fp16
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir=./internlm-chat-7b/trt_engines/fp16/1-gpu/

# InternLM 7B with bf16
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir=./internlm-chat-7b/trt_engines/bf16/1-gpu/

# InternLM 7B with int8 weight only quantization
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir=./internlm-chat-7b/trt_engines/weight_only/1-gpu/

# InternLM 7B with fp16 and tensor parallelism
mpirun -n 2 --allow-run-as-root \
    python ../../../run.py --max_output_len=120 \
                     --input_text 'Tell me about yourself.' \
                     --tokenizer_dir ./internlm-chat-7b/ \
                     --engine_dir=./internlm-chat-7b/trt_engines/fp16/2-gpu/

# InternLM 20B with fp16 and tensor parallelism and pipeline parallelism
mpirun -n 4 --allow-run-as-root \
    python ../../../run.py --max_output_len=120 \
                     --input_text 'Tell me about yourself.' \
                     --tokenizer_dir ./internlm-chat-7b/ \
                     --engine_dir=./internlm-chat-7b/trt_engines/bf16/4-gpu/
```

### Summarization using the InternLM model

```bash
# Run summarization using the InternLM 7B model in FP16.
python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b/ \
                       --data_type fp16 \
                       --engine_dir ./engine_outputs

# Run summarization using the InternLM 7B model quantized to INT8.
python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b/ \
                       --data_type fp16 \
                       --engine_dir ./engine_outputs

# Run summarization using the InternLM 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm --test_hf \
                           --hf_model_dir ./internlm-chat-7b/ \
                           --data_type fp16 \
                           --engine_dir ./internlm-chat-7b/trt_engines/fp16/2-gpu/

# Run summarization using the InternLM 20B model in BF16 using 4 GPUs.
mpirun -n 4 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm --test_hf \
                           --hf_model_dir ./internlm-chat-20b/ \
                           --data_type bf16 \
                           --engine_dir ./internlm-chat-20b/trt_engines/bf16/4-gpu/
```
