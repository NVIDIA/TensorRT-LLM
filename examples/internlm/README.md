# InternLM

This document shows how to build and run InternLM 7B / 20B models in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM InternLM implementation can be found in [tensorrt_llm/models/internlm/model.py](../../tensorrt_llm/models/internlm/model.py). The TensorRT-LLM InternLM example code is located in [`examples/internlm`](./). There is one main file:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the InternLM model.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
  * FP16 / BF16
  * INT8 & INT4 Weight-Only
  * Smooth Quant
  * INT8 KV Cache
  * Tensor Parallel & Pipeline Parallel

## Usage

The TensorRT-LLM InternLM example code locates at [examples/internlm](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

TensorRT-LLM InternLM builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

InternLM has released several checkpoints of different size or capabilities under https://huggingface.co/internlm. Users can pick any one repository and follow instructions to prepare the checkpoint.

Below examples use [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) and [internlm-chat-20b](https://huggingface.co/internlm/internlm-chat-20b) and assume these repositories are cloned or linked under this directory, for example `./internlm-chat-7b/`.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallel building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# use_gpt_attention_plugin is necessary in InternLM.
# Try use_gemm_plugin to prevent accuracy issue.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance

# Build the InternLM 7B model using a single GPU and FP16.
python build.py --model_dir ./internlm-chat-7b/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./internlm-chat-7b/trt_engines/fp16/1-gpu/

# Build the InternLM 7B model using a single GPU and BF16.
python build.py --model_dir ./internlm-chat-7b/ \
                --dtype bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --use_gemm_plugin bfloat16 \
                --output_dir ./internlm-chat-7b/trt_engines/bf16/1-gpu/

# Build the InternLM 7B model using a single GPU and apply INT8 weight-only quantization.
python build.py --model_dir ./internlm-chat-7b/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --output_dir ./internlm-chat-7b/trt_engines/weight_only/1-gpu/

# Note: setting `--weight_only_precision int4` to use INT4 weight-only quantization

# Build InternLM 7B using 2-way tensor parallelism.
python build.py --model_dir ./internlm-chat-7b/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./internlm-chat-7b/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2 \
                --parallel_build

# Build InternLM 20B using 2-way tensor parallelism and 2-way pipeline parallelism.
python build.py --model_dir ./internlm-chat-20b/ \
                --dtype bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --use_gemm_plugin bfloat16 \
                --output_dir ./internlm-chat-20b/trt_engines/bf16/4-gpu/ \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2 \
                --parallel_build
```

#### INT8 weight only + INT8 KV cache

For INT8 KV cache, [`hf_internlm_convert.py`](./hf_internlm_convert.py) features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
# For 7B models
python hf_internlm_convert.py -i ./internlm-chat-7b -o ./internlm-chat-7b/smooth_internlm/int8_kv_cache/ --calibrate-kv-cache -t fp16
# For 20B models
python hf_internlm_convert.py -i ./internlm-chat-20b -o ./internlm-chat-20b/smooth_internlm/int8_kv_cache/ --calibrate-kv-cache -t fp16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache.

In addition, it could be combined with INT8 weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build 7B model with both INT8 weight-only and INT8 KV cache enabled
python build.py --ft_model_dir=./internlm-chat-7b/smooth_internlm/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./internlm-chat-7b/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only

# Build 20B model with both INT8 weight-only and INT8 KV cache enabled
python build.py --ft_model_dir=./internlm-chat-20b/smooth_internlm/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./internlm-chat-20b/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only
```

Test with `../run.py` or `../summarize.py`:

```bash
python ../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir ./internlm-chat-7b/trt_engines/int8_kv_cache_weight_only/1-gpu

python ../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-20b/ \
                 --engine_dir ./internlm-chat-20b/trt_engines/int8_kv_cache_weight_only/1-gpu

python ../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-7b/trt_engines/int8_kv_cache_weight_only/1-gpu

python ../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-20b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-20b/trt_engines/int8_kv_cache_weight_only/1-gpu
```

#### SmoothQuant

Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
# For 7B models
python hf_internlm_convert.py -i ./internlm-chat-7b -o ./internlm-chat-7b/smooth_internlm/sq0.5/ -sq 0.5 --tensor-parallelism 1 --storage-type fp16
# For 20B models
python hf_internlm_convert.py -i ./internlm-chat-20b -o ./internlm-chat-20b/smooth_internlm/sq0.5/ -sq 0.5 --tensor-parallelism 1 --storage-type fp16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
# 7B model
python build.py --ft_model_dir=./internlm-chat-7b/smooth_internlm/sq0.5/1-gpu/ \
                --use_gpt_attention_plugin float16 \
                --remove_input_padding \
                --enable_context_fmha \
                --use_smooth_quant \
                --per_token \
                --per_channel \
                --output_dir ./internlm-chat-7b/trt_engines/smoothquant/1-gpu

# 20B model
python build.py --ft_model_dir=./internlm-chat-20b/smooth_internlm/sq0.5/1-gpu/ \
                --use_gpt_attention_plugin float16 \
                --remove_input_padding \
                --enable_context_fmha \
                --use_smooth_quant \
                --per_token \
                --per_channel \
                --output_dir ./internlm-chat-20b/trt_engines/smoothquant/1-gpu
```

Note we use `--ft_model_dir` instead of `--model_dir` and `--meta_ckpt_dir` since SmoothQuant model needs INT8 weights and various scales from the binary files.

Test with `../run.py` or `../summarize.py`:

```bash
python ../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir ./internlm-chat-7b/trt_engines/smoothquant/1-gpu

python ../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-20b/ \
                 --engine_dir ./internlm-chat-20b/trt_engines/smoothquant/1-gpu

python ../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-7b/trt_engines/smoothquant/1-gpu

python ../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-20b \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-20b/trt_engines/smoothquant/1-gpu
```

### Run

To run a TensorRT-LLM InternLM model using the engines generated by build.py

```bash
# InternLM 7B with fp16
python ../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir=./internlm-chat-7b/trt_engines/fp16/1-gpu/

# InternLM 7B with bf16
python ../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir=./internlm-chat-7b/trt_engines/bf16/1-gpu/

# InternLM 7B with int8 weight only quantization
python ../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm-chat-7b/ \
                 --engine_dir=./internlm-chat-7b/trt_engines/weight_only/1-gpu/

# InternLM 7B with fp16 and tensor parallelism
mpirun -n 2 --allow-run-as-root \
    python ../run.py --max_output_len=120 \
                     --input_text 'Tell me about yourself.' \
                     --tokenizer_dir ./internlm-chat-7b/ \
                     --engine_dir=./internlm-chat-7b/trt_engines/fp16/2-gpu/

# InternLM 20B with fp16 and tensor parallelism and pipeline parallelism
mpirun -n 4 --allow-run-as-root \
    python ../run.py --max_output_len=120 \
                     --input_text 'Tell me about yourself.' \
                     --tokenizer_dir ./internlm-chat-7b/ \
                     --engine_dir=./internlm-chat-7b/trt_engines/bf16/4-gpu/
```

### Summarization using the InternLM model

```bash
# Run summarization using the InternLM 7B model in FP16.
python ../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b/ \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-7b/trt_engines/fp16/1-gpu/

# Run summarization using the InternLM 7B model quantized to INT8.
python ../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm-chat-7b/ \
                       --data_type fp16 \
                       --engine_dir ./internlm-chat-7b/trt_engines/weight_only/1-gpu/

# Run summarization using the InternLM 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm --test_hf \
                           --hf_model_dir ./internlm-chat-7b/ \
                           --data_type fp16 \
                           --engine_dir ./internlm-chat-7b/trt_engines/fp16/2-gpu/

# Run summarization using the InternLM 20B model in BF16 using 4 GPUs.
mpirun -n 4 --allow-run-as-root \
    python ../summarize.py --test_trt_llm --test_hf \
                           --hf_model_dir ./internlm-chat-20b/ \
                           --data_type bf16 \
                           --engine_dir ./internlm-chat-20b/trt_engines/bf16/4-gpu/
```
