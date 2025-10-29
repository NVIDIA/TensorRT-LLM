# Mamba

This document shows how to build and run a [Mamba](https://github.com/state-spaces/mamba) model in TensorRT LLM on a single GPU.

- [Mamba](#mamba)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [1. Download weights from HuggingFace Transformers](#1-download-weights-from-huggingface-transformers)
    - [2. Convert weights from HF Transformers to TensorRT LLM format](#2-convert-dweights-from-hf-transformers-to-tensorrt-llm-format)
    - [3. Build TensorRT engine(s)](#3-build-tensorrt-engines)
    - [4. Run summarization task with the TensorRT engine(s)](#4-run-summarization-task-with-the-tensorrt-engines)

## Overview

The TensorRT LLM Mamba implementation can be found in [`tensorrt_llm/models/mamba/model.py`](../../../../tensorrt_llm/models/mamba/model.py). The TensorRT LLM Mamba example code is located in [`examples/models/core/mamba`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.


## Support Matrix

|    Model Name    | FP16  | BF16  | TP  |
| :--------------: | :---: | :---: | :-: |
|    Mamba1        |   Y   |   Y   |  N  |
|    Mamba2        |   Y   |   Y   |  Y  |

* Mamba2: TensorRT LLM can only support the pure Mamba model for now, will support the hybrid models later.

## Usage

The next two sections describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
format to the TensorRT LLM format.

### 1. Download weights from HuggingFace Transformers

Please install required packages first and setup `git-lfs`:

```bash
pip install -r requirements.txt
git lfs install
```

There are different HF checkpoints available. For Mamba1, TensorRT LLM can support those Transformers compatible models. Here're some examples to fetch the checkpoint.

```bash
# mamba-2.8b
git clone https://huggingface.co/state-spaces/mamba-2.8b-hf ./mamba_model/mamba-2.8b

# mamba-130m
git clone https://huggingface.co/state-spaces/mamba-130m-hf ./mamba_model/mamba-130m

# mamba2-2.7b
git clone https://huggingface.co/state-spaces/mamba2-2.7b ./mamba_model/mamba2-2.7b

# mamba2-130m
git clone https://huggingface.co/state-spaces/mamba2-130m ./mamba_model/mamba2-130m

# mamba-codestral-7B-v0.1
git clone https://huggingface.co/mistralai/mamba-codestral-7B-v0.1 ./mamba_model/mamba-codestral-7B-v0.1
```

Since mamba models use tokenizer from gpt-neox-20b model, use the following command to fetch the checkpoint of gpt-neox-20b.

```bash
# gpt-neox-20b
git clone https://huggingface.co/EleutherAI/gpt-neox-20b ./mamba_model/gpt-neox-20b
```

### 2. Convert weights from HF Transformers to TensorRT LLM format
The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT LLM checkpoints.

For the Mamba2 models, if they can support tensor parallelism, you can run them with 1, 2, 4 or 8 GPUs. Here we use
mamba-codestral-7B-v0.1 as an example.

```bash
# mamba-2.8b
python convert_checkpoint.py --model_dir ./mamba_model/mamba-2.8b/ \
                             --dtype bfloat16 \
                             --output_dir ./mamba_model/mamba-2.8b/trt_ckpt/bf16/1-gpu/

# mamba-130m
python convert_checkpoint.py --model_dir ./mamba_model/mamba-130m/ \
                             --dtype float16 \
                             --output_dir ./mamba_model/mamba-130m/trt_ckpt/fp16/1-gpu/

# mamba2-2.7b
python convert_checkpoint.py --model_dir ./mamba_model/mamba2-2.7b/ \
                             --dtype float16 \
                             --output_dir ./mamba_model/mamba2-2.7b/trt_ckpt/fp16/1-gpu/

# mamba2-130m
python convert_checkpoint.py --model_dir ./mamba_model/mamba2-130m/ \
                             --dtype float16 \
                             --output_dir ./mamba_model/mamba2-130m/trt_ckpt/fp16/1-gpu/

# mamba-codestral-7B-v0.1
python convert_checkpoint.py --model_dir ./mamba_model/mamba-codestral-7B-v0.1/ \
                             --dtype float16 \
                             --output_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_ckpt/fp16/1-gpu/

# mamba-codestral-7B-v0.1 with 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./mamba_model/mamba-codestral-7B-v0.1/ \
                             --dtype float16 \
                             --world_size 2 \
                             --output_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_ckpt/fp16/2-gpu/
```

### 3. Build TensorRT engine(s)
The `trtllm-build` command builds TensorRT LLM engines from TensorRT LLM checkpoints.

```bash
# mamba-2.8b
trtllm-build --checkpoint_dir ./mamba_model/mamba-2.8b/trt_ckpt/bf16/1-gpu/ \
             --paged_kv_cache disable \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./mamba_model/mamba-2.8b/trt_engines/bf16/1-gpu/

# mamba-130m
trtllm-build --checkpoint_dir ./mamba_model/mamba-130m/trt_ckpt/fp16/1-gpu/ \
             --paged_kv_cache disable \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./mamba_model/mamba-130m/trt_engines/fp16/1-gpu/

# mamba2-2.7b
trtllm-build --checkpoint_dir ./mamba_model/mamba2-2.7b/trt_ckpt/fp16/1-gpu/ \
             --paged_kv_cache disable \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./mamba_model/mamba2-2.7b/trt_engines/fp16/1-gpu/

# mamba2-130m
trtllm-build --checkpoint_dir ./mamba_model/mamba2-130m/trt_ckpt/fp16/1-gpu/ \
             --paged_kv_cache disable \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./mamba_model/mamba2-130m/trt_engines/fp16/1-gpu/

# mamba-codestral-7B-v0.1
trtllm-build --checkpoint_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_ckpt/fp16/1-gpu/ \
             --paged_kv_cache disable \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_engines/fp16/1-gpu/

# mamba-codestral-7B-v0.1 with 2-way tensor parallelism.
trtllm-build --checkpoint_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_ckpt/fp16/2-gpu/ \
             --paged_kv_cache disable \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_engines/fp16/2-gpu/
```

Note that when building Mamba models, you need to disable the `paged_kv_cache` as it is used for
transformer-based models. Mamba models use `paged_state` instead and it is enabled by default.
If `paged_state` is disabled, engine will be built with the contiguous stage cache.

### 4. Run summarization task with the TensorRT engine(s)

The following section describes how to run a TensorRT LLM Mamba model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.

```bash
# mamba-2.8b
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./mamba_model/mamba-2.8b/ \
                       --tokenizer_dir ./mamba_model/gpt-neox-20b/ \
                       --data_type bf16 \
                       --engine_dir ./mamba_model/mamba-2.8b/trt_engines/bf16/1-gpu/

# mamba-130m
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./mamba_model/mamba-130m/ \
                       --tokenizer_dir ./mamba_model/gpt-neox-20b/ \
                       --data_type fp16 \
                       --engine_dir ./mamba_model/mamba-130m/trt_engines/fp16/1-gpu/

# mamba2-2.7b
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./mamba_model/mamba2-2.7b/ \
                       --tokenizer_dir ./mamba_model/gpt-neox-20b/ \
                       --data_type fp16 \
                       --engine_dir ./mamba_model/mamba2-2.7b/trt_engines/fp16/1-gpu/

# mamba2-130m
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./mamba_model/mamba2-130m/ \
                       --tokenizer_dir ./mamba_model/gpt-neox-20b/ \
                       --data_type fp16 \
                       --engine_dir ./mamba_model/mamba2-130m/trt_engines/fp16/1-gpu/

# mamba-codestral-7B-v0.1
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./mamba_model/mamba-codestral-7B-v0.1/ \
                       --tokenizer_dir ./mamba_model/mamba-codestral-7B-v0.1/ \
                       --data_type fp16 \
                       --engine_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_engines/fp16/1-gpu/

# mamba-codestral-7B-v0.1 with 2-way tensor parallelism.
mpirun -n 2 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm \
                           --hf_model_dir ./mamba_model/mamba-codestral-7B-v0.1/ \
                           --tokenizer_dir ./mamba_model/mamba-codestral-7B-v0.1/ \
                           --data_type fp16 \
                           --engine_dir ./mamba_model/mamba-codestral-7B-v0.1/trt_engines/fp16/2-gpu/
```
