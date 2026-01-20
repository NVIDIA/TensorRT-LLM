# GPT-NeoX

This document explains how to build the [GPT-NeoX](https://huggingface.co/EleutherAI/gpt-neox-20b) model using TensorRT LLM and run on a single GPU and a single node with
multiple GPUs.

- [GPT-NeoX](#gpt-neox)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [1. Download weights from HuggingFace (HF) Transformers](#1-download-weights-from-huggingface-hf-transformers)
    - [2. Convert weights from HF Transformers to TensorRT LLM format](#2-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
    - [3. Build TensorRT engine(s)](#3-build-tensorrt-engines)
    - [4. Summarization using the GPT-NeoX model](#4-summarization-using-the-gpt-neox-model)
  - [Apply groupwise quantization GPTQ](#apply-groupwise-quantization-gptq)
    - [1. Download weights from HuggingFace (HF)](#1-download-weights-from-huggingface-hf)
    - [2. Generating quantized weights](#2-generating-quantized-weights)
    - [3. Convert weights from HF Transformers to TensorRT LLM format](#3-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
    - [4. Build TensorRT engine(s)](#4-build-tensorrt-engines)
    - [5. Summarization using the GPT-NeoX model](#5-summarization-using-the-gpt-neox-model)

## Overview

The TensorRT LLM GPT-NeoX implementation can be found in [`tensorrt_llm/models/gptneox/model.py`](../../tensorrt_llm/models/gptneox/model.py). The TensorRT LLM GPT-NeoX example code is located in [`examples/models/contrib/gptneox`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * INT8 Weight-Only
  * INT4 GPTQ
  * Tensor Parallel

## Usage

The TensorRT LLM GPT-NeoX example code locates at [examples/models/contrib/gptneox](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### 1. Download weights from HuggingFace (HF) Transformers

Please install required packages first:

```bash
pip install -r requirements.txt
```

```bash
# Weights & config
git clone https://huggingface.co/EleutherAI/gpt-neox-20b gptneox_model
```

### 2. Convert weights from HF Transformers to TensorRT LLM format

If you want to use Int8 weight only quantization, just need to add `--use_weight_only` flag.

```bash
# Single GPU
python3 convert_checkpoint.py --model_dir ./gptneox_model \
                              --dtype float16 \
                              --output_dir ./gptneox/20B/trt_ckpt/fp16/1-gpu/
# With 2-way Tensor Parallel
python3 convert_checkpoint.py --model_dir ./gptneox_model \
                              --dtype float16 \
                              --tp_size 2 \
                              --workers 2 \
                              --output_dir ./gptneox/20B/trt_ckpt/fp16/2-gpu/
# Single GPU with int8 weight only
python3 convert_checkpoint.py --model_dir ./gptneox_model \
                              --dtype float16 \
                              --use_weight_only \
                              --output_dir ./gptneox/20B/trt_ckpt/int8_wo/1-gpu/
# With 2-way Tensor Parallel with int8 weight only
python3 convert_checkpoint.py --model_dir ./gptneox_model \
                              --dtype float16 \
                              --use_weight_only \
                              --tp_size 2 \
                              --workers 2 \
                              --output_dir ./gptneox/20B/trt_ckpt/int8_wo/2-gpu/
```

### 3. Build TensorRT engine(s)
```bash
# Single GPU
trtllm-build --checkpoint_dir ./gptneox/20B/trt_ckpt/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./gptneox/20B/trt_engines/fp16/1-gpu/
# With 2-way Tensor Parallel
trtllm-build --checkpoint_dir ./gptneox/20B/trt_ckpt/fp16/2-gpu/ \
             --gemm_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --workers 2 \
             --output_dir ./gptneox/20B/trt_engines/fp16/2-gpu/
# Single GPU with int8 weight only
trtllm-build --checkpoint_dir ./gptneox/20B/trt_ckpt/int8_wo/1-gpu/ \
             --gemm_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./gptneox/20B/trt_engines/int8_wo/1-gpu/
# With 2-way Tensor Parallel with int8 weight only
trtllm-build --checkpoint_dir ./gptneox/20B/trt_ckpt/int8_wo/2-gpu/ \
             --gemm_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --workers 2 \
             --output_dir ./gptneox/20B/trt_engines/int8_wo/2-gpu/
```

### 4. Summarization using the GPT-NeoX model

The following section describes how to run a TensorRT LLM GPT-NeoX model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF GPT-NeoX model.

```bash
# Single GPU
python3 ../../../summarize.py --engine_dir ./gptneox/20B/trt_engines/fp16/1-gpu/ \
                        --test_trt_llm \
                        --hf_model_dir gptneox_model \
                        --data_type fp16
# With 2-way Tensor Parallel
mpirun -np 2 --oversubscribe --allow-run-as-root \
    python3 ../../../summarize.py --engine_dir ./gptneox/20B/trt_engines/fp16/2-gpu/ \
                            --test_trt_llm \
                            --hf_model_dir gptneox_model \
                            --data_type fp16
# Single GPU with int8 weight only
python3 ../../../summarize.py --engine_dir ./gptneox/20B/trt_engines/int8_wo/1-gpu/ \
                        --test_trt_llm \
                        --hf_model_dir gptneox_model \
                        --data_type fp16
# With 2-way Tensor Parallel with int8 weight only
mpirun -np 2 --oversubscribe --allow-run-as-root \
    python3 ../../../summarize.py --engine_dir ./gptneox/20B/trt_engines/int8_wo/2-gpu/ \
                            --test_trt_llm \
                            --hf_model_dir gptneox_model \
                            --data_type fp16
```

## Apply groupwise quantization GPTQ

### 1. Download weights from HuggingFace (HF)

```bash
# Weights & config
sh get_weights.sh
```

### 2. Generating quantized weights

In this example, the weights are quantized using [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa). Note that the parameter `--act-order` referring to whether to apply the activation order GPTQ heuristic is **not supported** by TRT-LLM.

```bash
sh gptq_convert.sh
```

### 3. Convert weights from HF Transformers to TensorRT LLM format

To apply groupwise quantization GPTQ, addition command-line flags need to be passed to `convert_checkpoint.py`:
Here `--quant_ckpt_path` flag specifies the output safetensors of `gptq_convert.sh` script.

```bash
# Single GPU
python3 convert_checkpoint.py --model_dir ./gptneox_model \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4_gptq \
                              --quant_ckpt_path ./gptneox_model/gptneox-20b-4bit-gs128.safetensors \
                              --output_dir ./gptneox/20B/trt_ckpt/int4_gptq/1-gpu/
# With 2-way Tensor Parallel
python3 convert_checkpoint.py --model_dir ./gptneox_model \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4_gptq \
                              --tp_size 2 \
                              --workers 2 \
                              --quant_ckpt_path ./gptneox_model/gptneox-20b-4bit-gs128.safetensors \
                              --output_dir ./gptneox/20B/trt_ckpt/int4_gptq/2-gpu/
```

### 4. Build TensorRT engine(s)

The command to build TensorRT engines to apply GPTQ does not change:

```bash
# Single GPU
trtllm-build --checkpoint_dir ./gptneox/20B/trt_ckpt/int4_gptq/1-gpu/ \
             --gemm_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --output_dir ./gptneox/20B/trt_engines/int4_gptq/1-gpu/
# With 2-way Tensor Parallel
trtllm-build --checkpoint_dir ./gptneox/20B/trt_ckpt/int4_gptq/2-gpu/ \
             --gemm_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_seq_len 1024 \
             --workers 2 \
             --output_dir ./gptneox/20B/trt_engines/int4_gptq/2-gpu/
```

### 5. Summarization using the GPT-NeoX model

The command to run summarization with GPTQ quantized model also does not change:

```bash
# Single GPU
python3 ../../../summarize.py --engine_dir ./gptneox/20B/trt_engines/int4_gptq/1-gpu/ \
                        --test_trt_llm \
                        --hf_model_dir gptneox_model \
                        --data_type fp16
# With 2-way Tensor Parallel
mpirun -np 2 --oversubscribe --allow-run-as-root \
    python3 ../../../summarize.py --engine_dir ./gptneox/20B/trt_engines/int4_gptq/2-gpu/ \
                            --test_trt_llm \
                            --hf_model_dir gptneox_model \
                            --data_type fp16
```
