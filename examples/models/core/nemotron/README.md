# Nemotron

This document demonstrates how to build the Nemotron models using TensorRT LLM and run on a single GPU or multiple GPUs.

- [Nemotron](#nemotron)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Download weights from HuggingFace Transformers](#download-weights-from-huggingface-transformers)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
      - [FP8 Quantization](#fp8-quantization)
      - [INT4 AWQ Quantization](#int4-awq-quantization)
    - [Run Inference](#run-inference)

## Overview

The TensorRT LLM Nemotron implementation is based on the GPT model, which can be found in [`tensorrt_llm/models/gpt/model.py`](../../../../tensorrt_llm/models/gpt/model.py). The TensorRT LLM Nemotron example is located in [`examples/models/core/nemotron`](./).

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16/BF16
  * FP8
  * INT4 AWQ
  * Tensor Parallel
  * Pipeline Parallel
  * Inflight Batching
  * PAGED_KV_CACHE
  * STRONGLY TYPED
  * checkpoint type: Nemo, Huggingface (HF)

## Nemo checkpoint - Usage

### Download weights from HuggingFace Transformers


Install the dependencies and setup `git-lfs`.

```bash
# Install dependencies
pip install -r requirements.txt

# Setup git-lfs
git lfs install
```

Download one or more Nemotron models that you would like to build to TensorRT LLM engines. You can download from the [HuggingFace](https://huggingface.co) hub:

```bash
# Download nemotron-3-8b-base-4k
git clone https://huggingface.co/nvidia/nemotron-3-8b-base-4k

# Download nemotron-3-8b-chat-4k-sft
git clone https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-sft

# Download nemotron-3-8b-chat-4k-rlhf
git clone https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-rlhf
```

### Build TensorRT engine(s)
The [`examples/quantization/quantize.py`](../../../quantization/quantize.py) script can quantize the Nemotron models and export to TensorRT LLM checkpoints. You may optionally skip the quantization step by specifying `--qformat full_prec` and thus export float16 or bfloat16 TensorRT LLM checkpoints.

The `trtllm-build` command builds TensorRT LLM engines from TensorRT LLM checkpoints. The number of engine files is same to the number of GPUs used to run inference. Normally, `trtllm-build` uses one GPU by default, but if you have already more GPUs available at build time, you may enable parallel builds to make the engine building process faster by adding the `--workers` argument.

Here are some examples:

```bash
# single gpu, dtype bfloat16
python3 ../../../quantization/quantize.py \
        --nemo_ckpt_path nemotron-3-8b-base-4k/Nemotron-3-8B-Base-4k.nemo \
        --dtype bfloat16 \
        --batch_size 64 \
        --qformat full_prec \
        --output_dir nemotron-3-8b/trt_ckpt/bf16/1-gpu

trtllm-build --checkpoint_dir nemotron-3-8b/trt_ckpt/bf16/1-gpu \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --output_dir nemotron-3-8b/trt_engines/bf16/1-gpu
```

```bash
# 2-way tensor parallelism
python3 ../../../quantization/quantize.py \
        --nemo_ckpt_path nemotron-3-8b-base-4k/Nemotron-3-8B-Base-4k.nemo \
        --dtype bfloat16 \
        --batch_size 64 \
        --qformat full_prec \
        --tp_size 2 \
        --output_dir nemotron-3-8b/trt_ckpt/bf16/tp2

trtllm-build --checkpoint_dir nemotron-3-8b/trt_ckpt/bf16/tp2 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --workers 2 \
        --output_dir nemotron-3-8b/trt_engines/bf16/tp2
```

```bash
# 2-way tensor parallelism for both calibration and inference
mpirun -np 2 \
    python3 ../../../quantization/quantize.py \
        --nemo_ckpt_path nemotron-3-8b-base-4k/Nemotron-3-8B-Base-4k.nemo \
        --dtype bfloat16 \
        --batch_size 64 \
        --qformat full_prec \
        --calib_tp_size 2 \
        --tp_size 2 \
        --output_dir nemotron-3-8b/trt_ckpt/bf16/tp2

trtllm-build --checkpoint_dir nemotron-3-8b/trt_ckpt/bf16/tp2 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --workers 2 \
        --output_dir nemotron-3-8b/trt_engines/bf16/tp2
```

#### FP8 Quantization

Quantize the Nemotron models to FP8 by specifying `--qformat fp8` to `quantize.py`.

```bash
# single gpu, fp8 quantization
python3 ../../../quantization/quantize.py \
        --nemo_ckpt_path nemotron-3-8b-base-4k/Nemotron-3-8B-Base-4k.nemo \
        --dtype bfloat16 \
        --batch_size 64 \
        --qformat fp8 \
        --output_dir nemotron-3-8b/trt_ckpt/fp8/1-gpu

trtllm-build --checkpoint_dir nemotron-3-8b/trt_ckpt/fp8/1-gpu \
        --gpt_attention_plugin bfloat16 \
        --output_dir nemotron-3-8b/trt_engines/fp8/1-gpu
```

#### INT4 AWQ Quantization

Quantize the Nemotron models using INT4 AWQ by specifying `--qformat int4_awq` to `quantize.py`.

```bash
# single gpu, int4 awq quantization
python3 ../../../quantization/quantize.py \
        --nemo_ckpt_path nemotron-3-8b-base-4k/Nemotron-3-8B-Base-4k.nemo \
        --dtype bfloat16 \
        --batch_size 64 \
        --qformat int4_awq \
        --output_dir nemotron-3-8b/trt_ckpt/int4_awq/1-gpu

trtllm-build --checkpoint_dir nemotron-3-8b/trt_ckpt/int4_awq/1-gpu \
        --gpt_attention_plugin bfloat16 \
        --output_dir nemotron-3-8b/trt_engines/int4_awq/1-gpu
```

### Run Inference

The `summarize.py` script can run the built engines to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

```bash
# single gpu
python3 ../../../summarize.py --test_trt_llm \
        --no_add_special_tokens \
        --engine_dir nemotron-3-8b/trt_engines/bf16/1-gpu \
        --vocab_file nemotron-3-8b/trt_ckpt/bf16/1-gpu/tokenizer.model

# multiple gpus
mpirun -np 2 \
    python3 ../../../summarize.py --test_trt_llm \
        --no_add_special_tokens \
        --engine_dir nemotron-3-8b/trt_engines/bf16/tp2 \
        --vocab_file nemotron-3-8b/trt_ckpt/bf16/tp2/tokenizer.model
```

If the engines are run successfully, you will see output like:
```
......
[04/23/2024-09:55:54] [TRT-LLM] [I] TensorRT LLM (total latency: 14.926485538482666 sec)
[04/23/2024-09:55:54] [TRT-LLM] [I] TensorRT LLM (total output tokens: 2000)
[04/23/2024-09:55:54] [TRT-LLM] [I] TensorRT LLM (tokens per second: 133.99001357980129)
[04/23/2024-09:55:54] [TRT-LLM] [I] TensorRT LLM beam 0 result
[04/23/2024-09:55:54] [TRT-LLM] [I]   rouge1 : 19.48743720965424
[04/23/2024-09:55:54] [TRT-LLM] [I]   rouge2 : 6.272381295466071
[04/23/2024-09:55:54] [TRT-LLM] [I]   rougeL : 15.011005943152721
[04/23/2024-09:55:54] [TRT-LLM] [I]   rougeLsum : 17.76145734406502
```

## HF checkpoint - Usage
Support for Nemotron models was added with transformers 4.44.0 release.

```bash
# install transformers library
pip install transformers>=4.44.0
# Download hf minitron model
git clone https://huggingface.co/nvidia/Minitron-4B-Base

# Convert to TensorRT LLM checkpoint
python3 ../gpt/convert_checkpoint.py --model_dir Minitron-4B-Base \
        --dtype bfloat16 \
        --output_dir minitron/trt_ckpt/bf16/1-gpu

# Build TensorRT LLM engines
trtllm-build --checkpoint_dir minitron/trt_ckpt/bf16/1-gpu \
        --gemm_plugin auto \
        --output_dir minitron/trt_engines/bf16/1-gpu

# Run inference
python3 ../../../run.py --engine_dir minitron/trt_engines/bf16/1-gpu \
        --tokenizer_dir Minitron-4B-Base \
        --input_text "def print_hello_world():" \
        --max_output_len 20
```
