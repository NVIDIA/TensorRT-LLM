# OPT

This document explains how to build the [OPT](https://huggingface.co/docs/transformers/model_doc/opt) model using TensorRT LLM and run on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs.

- [OPT](#opt)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [1. Download weights from HuggingFace Transformers](#1-download-weights-from-huggingface-transformers)
    - [2. Convert weights from HF Transformers to TensorRT LLM format](#2-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
    - [3. Build TensorRT engine(s)](#3-build-tensorrt-engines)
    - [4. Summarization using the OPT model](#4-summarization-using-the-opt-model)
      - [Fused MultiHead Attention (FMHA)](#fused-multihead-attention-fmha)
  - [Tensor Parallelism for Embedding Lookup Table.](#tensor-parallelism-for-embedding-lookup-table)
    - [1. Enable this feature](#1-enable-this-feature)
    - [2. Choose the dimension for tensor parallelism](#2-choose-the-dimension-for-tensor-parallelism)

## Overview

The TensorRT LLM OPT implementation can be found in [`tensorrt_llm/models/opt/model.py`](../../tensorrt_llm/models/opt/model.py). The TensorRT LLM OPT example code is located in [`examples/models/contrib/opt`](./). There is one file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * INT8 & INT4 Weight-Only
  * Tensor Parallel

## Usage

The next two sections describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
format to the TensorRT LLM format.

### 1. Download weights from HuggingFace Transformers

You have to make sure `git-lfs` is properly installed to load the checkpoints.

```bash
pip install -r requirements.txt && sudo apt-get install git-lfs
```

There are four different checkpoints available. Use one of the following commands to fetch the checkpoint you are interested in.

```bash
# OPT-125M
git-lfs clone https://huggingface.co/facebook/opt-125m

# OPT-350M
git-lfs clone https://huggingface.co/facebook/opt-350m

# OPT-2.7B
git-lfs clone https://huggingface.co/facebook/opt-2.7b

# OPT-66B
git-lfs clone https://huggingface.co/facebook/opt-66b
```

### 2. Convert weights from HF Transformers to TensorRT LLM format

```bash
# OPT-125M
python3 convert_checkpoint.py --model_dir ./opt-125m \
                --dtype float16 \
                --output_dir ./opt/125M/trt_ckpt/fp16/1-gpu/

# OPT-350M
python3 convert_checkpoint.py --model_dir ./opt-350m \
                --dtype float16 \
                --output_dir ./opt/350M/trt_ckpt/fp16/1-gpu/

# OPT-2.7B
python3 convert_checkpoint.py --model_dir ./opt-2.7b \
                --dtype float16 \
                --output_dir ./opt/2.7B/trt_ckpt/fp16/1-gpu/

# OPT-66B
python3 convert_checkpoint.py --model_dir ./opt-66b \
                --dtype float16 \
                --tp_size 4 \
                --output_dir ./opt/66B/trt_ckpt/fp16/4-gpu/ \
                --workers 2
```

### 3. Build TensorRT engine(s)

```bash
# OPT-125M
trtllm-build --checkpoint_dir ./opt/125M/trt_ckpt/fp16/1-gpu/ \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --output_dir ./opt/125M/trt_engines/fp16/1-gpu/

# OPT-350M
trtllm-build --checkpoint_dir ./opt/350M/trt_ckpt/fp16/1-gpu/ \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --output_dir ./opt/350M/trt_engines/fp16/1-gpu/

# OPT-2.7B
trtllm-build --checkpoint_dir ./opt/2.7B/trt_ckpt/fp16/1-gpu/ \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --output_dir ./opt/2.7B/trt_engines/fp16/1-gpu/

# OPT-66B
trtllm-build --checkpoint_dir ./opt/66B/trt_ckpt/fp16/4-gpu/ \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --output_dir ./opt/66B/trt_engines/fp16/4-gpu/ \
                --workers 2
```

### 4. Summarization using the OPT model

The following section describes how to run a TensorRT LLM OPT model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF OPT model.

```bash
# OPT-125M
python3 ../../../summarize.py --engine_dir ./opt/125M/trt_engines/fp16/1-gpu/ \
                        --test_hf \
                        --batch_size 1 \
                        --test_trt_llm \
                        --hf_model_dir opt-125m \
                        --data_type fp16 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=14

# OPT-350M
python3 ../../../summarize.py --engine_dir ./opt/350M/trt_engines/fp16/1-gpu/ \
                        --test_hf \
                        --batch_size 1 \
                        --test_trt_llm \
                        --hf_model_dir opt-350m \
                        --data_type fp16 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=20

# OPT-2.7B
python3 ../../../summarize.py --engine_dir ./opt/2.7B/trt_engines/fp16/1-gpu/ \
                        --test_hf \
                        --batch_size 1 \
                        --test_trt_llm \
                        --hf_model_dir opt-2.7b \
                        --data_type fp16 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=20

# OPT-66B
mpirun -n 4 --allow-run-as-root \
    python3 ../../../summarize.py --engine_dir ./opt/66B/trt_engines/fp16/4-gpu/ \
                            --batch_size 1 \
                            --test_trt_llm \
                            --hf_model_dir opt-66b \
                            --data_type fp16 \
                            --check_accuracy \
                            --tensorrt_llm_rouge1_threshold=20
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for OPT by adding `--enable_context_fmha` to the invocation of `trtllm-build`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc` to the inference command (`run.py` or `summarize.py`). However, it is expected to see performance drop.

Note `--context_fmha` has to be used together with `--gpt_attention_plugin float16`.

## Tensor Parallelism for Embedding Lookup Table.
Since the embedding lookup table can be several gigabytes in size. We can distribute this weight across multiple GPUs in order to reduce the memory consumption per GPU.

### 1. Enable this feature
To enable this feature, add the flag `--use_parallel_embedding` to `trtllm-build`.

### 2. Choose the dimension for tensor parallelism

Assume the size of embedding lookup table is (vocab\_size \* hidden\_size), we can shard it along the vocab\_size (`--embedding_sharding_dim 0`) or hidden\_size (`--embedding_sharding_dim 1`) dimension.

2.1 To shard the embedding lookup table along the hidden\_size dimension, set the flag `--use_parallel_embedding --embedding_sharding_dim 1`. Here is an example:

```Bash
python3 convert_checkpoint.py --model_dir ./opt-125m \
                --dtype float16 \
                --output_dir ./opt/125M/trt_ckpt/fp16/2-gpu/ \
                --tp_size 2 \
                --use_parallel_embedding \
                --embedding_sharding_dim 1
```
2.2 To shard the embedding lookup table along the vocab\_size dimension, set the flag `--use_parallel_embedding --embedding_sharding_dim 0`.

Meanwhile, we provide a lookup plugin to support tensor parallelism on vocab\_size dimension.

- An example of sharing along vocab\_size dimension with lookup plugin:

```Bash
python3 convert_checkpoint.py --model_dir ./opt-125m \
                --dtype float16 \
                --output_dir ./opt/125M/trt_ckpt/fp16/2-gpu/ \
                --tp_size 2 \
                --use_parallel_embedding \
                --embedding_sharding_dim 0

trtllm-build --checkpoint_dir ./opt/125M/trt_ckpt/fp16/2-gpu/ \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --output_dir ./opt/125M/trt_engines/fp16/2-gpu/ \
                --workers 2

mpirun -n 2 --allow-run-as-root \
      python3 ../../../summarize.py --engine_dir ./opt/125M/trt_engines/fp16/2-gpu/ \
                        --batch_size 1 \
                        --test_trt_llm \
                        --hf_model_dir opt-125m \
                        --data_type fp16 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=14
```

- An example of sharing along vocab\_size dimension without lookup plugin:

```Bash
trtllm-build --checkpoint_dir ./opt/125M/trt_ckpt/fp16/2-gpu/ \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --output_dir ./opt/125M/trt_engines/fp16/2-gpu/ \
                --workers 2
```
