# DBRX

This document shows how to build and run a DBRX model in TensorRT-LLM. DBRX is a leading large language model trained by Databricks. Read more details about the model: [DBRX Technical Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).

- [DBRX](#dbrx)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Download weights from HuggingFace Transformers](#download-weights-from-huggingface-transformers)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
      - [Weight Only Quantization](#weight-only-quantization)
      - [INT8 KV Cache](#int8-kv-cache)
    - [Run inference](#run-inference)

## Overview

The TensorRT LLM DBRX implementation can be found in [tensorrt_llm/models/dbrx/model.py](../../../../tensorrt_llm/models/dbrx/model.py).

## Support Matrix
  * BF16
  * FP16
  * INT8 Weight-Only
  * INT4 Weight-Only
  * INT8 KV CACHE
  * Tensor Parallel
  * Pipeline Parallel
  * Expert Parallel

## Usage

### Download weights from HuggingFace Transformers

Install the dependencies and setup `git-lfs`.

```bash
# Install dependencies
# DBRX uses tiktoken as the tokenizer; make sure it is installed
pip install -r requirements.txt

# Setup git-lfs
git lfs install
```

Download one or more DBRX models that you would like to build to TensorRT LLM engines. You can download from the [HuggingFace](https://huggingface.co) hub:

```bash
# Download dbrx-base
git clone https://huggingface.co/databricks/dbrx-base

# Download dbrx-instruct
git clone https://huggingface.co/databricks/dbrx-instruct
```

### Build TensorRT engine(s)

The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT LLM checkpoints. A DBRX model has 132B parameters, so you need at least 4 x 80GB GPUs to load the model in 16-bit precision for weight conversion.

The `trtllm-build` command builds TensorRT LLM engines from TensorRT LLM checkpoints. The number of engine files is same to the number of GPUs used to run inference. Normally, `trtllm-build` uses one GPU by default, but if you have already more GPUs available at build time, you may enable parallel builds to make the engine building process faster by adding the `--workers` argument.

Here are some examples:

```bash
# 8-way tensor parallelism, dtype bfloat16
python convert_checkpoint.py --model_dir dbrx-base \
        --dtype bfloat16 \
        --tp_size 8 \
        --workers 8 \
        --output_dir dbrx/trt_ckpt/bf16/tp8

trtllm-build --checkpoint_dir dbrx/trt_ckpt/bf16/tp8 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --moe_plugin bfloat16 \
        --workers 8 \
        --output_dir dbrx/trt_engines/bf16/tp8
```

```bash
# 8-way tensor parallelism, dtype float16
python convert_checkpoint.py --model_dir dbrx-base \
        --dtype float16 \
        --tp_size 8 \
        --workers 8 \
        --output_dir dbrx/trt_ckpt/fp16/tp8

trtllm-build --checkpoint_dir dbrx/trt_ckpt/fp16/tp8 \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --moe_plugin float16 \
        --workers 8 \
        --output_dir dbrx/trt_engines/fp16/tp8
```

```bash
# 4-way tensor parallelism and 2-way pipeline parallelism, dtype bfloat16
python convert_checkpoint.py --model_dir dbrx-base \
        --dtype bfloat16 \
        --tp_size 4 \
        --pp_size 2 \
        --workers 8 \
        --output_dir dbrx/trt_ckpt/bf16/tp4pp2

trtllm-build --checkpoint_dir dbrx/trt_ckpt/bf16/tp4pp2 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --moe_plugin bfloat16 \
        --workers 8 \
        --output_dir dbrx/trt_engines/bf16/tp4pp2
```


```bash
# Build DBRX with expert parallelism for DbrxExperts layer and tensor parallelism for rest
python convert_checkpoint.py --model_dir dbrx-base \
        --dtype bfloat16 \
        --tp_size 8 \
        --moe_tp_size 1 \
        --moe_ep_size 8 \
        --workers 8 \
        --output_dir dbrx/trt_ckpt/bf16/ep8

trtllm-build --checkpoint_dir dbrx/trt_ckpt/bf16/ep8 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --moe_plugin bfloat16 \
        --workers 8 \
        --output_dir dbrx/trt_engines/bf16/ep8
```

#### Weight Only Quantization

[`convert_checkpoint.py`](./convert_checkpoint.py) features a `--use_weight_only` option that can enable weight-only quantization. You can further set the weight-only precision by passing `int8` or `int4` to the `--weight_only_precision` flag.

```bash
# 4-way tensor parallelism, int8 weight-only
python convert_checkpoint.py --model_dir dbrx-base \
        --dtype float16 \
        --use_weight_only \
        --weight_only_precision int8 \
        --tp_size 4 \
        --workers 4 \
        --output_dir dbrx/trt_ckpt/int8-wo/tp4

trtllm-build --checkpoint_dir dbrx/trt_ckpt/int8-wo/tp4 \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --moe_plugin float16 \
        --workers 4 \
        --output_dir dbrx/trt_engines/int8-wo/tp4
```

```bash
# 4-way tensor parallelism, int4 weight-only
python convert_checkpoint.py --model_dir dbrx-base \
        --dtype float16 \
        --use_weight_only \
        --weight_only_precision int4 \
        --tp_size 4 \
        --workers 4 \
        --output_dir dbrx/trt_ckpt/int4-wo/tp4

trtllm-build --checkpoint_dir dbrx/trt_ckpt/int4-wo/tp4 \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --moe_plugin float16 \
        --workers 4 \
        --output_dir dbrx/trt_engines/int4-wo/tp4
```

#### INT8 KV Cache
INT8 KV cache can be enabled to reduce the memory footprint. It will bring performance gains at large batch sizes and long sequence lengths.

For INT8 KV cache, [`convert_checkpoint.py`](./convert_checkpoint.py) features a `--int8_kv_cache` option. Setting `--int8_kv_cache` will calibrate the model, and then export the scaling factors needed for INT8 KV cache inference.

```bash
# 4-way tensor parallelism, int8 kv-cache
python convert_checkpoint.py --model_dir dbrx-base \
        --dtype float16 \
        --int8_kv_cache \
        --tp_size 4 \
        --workers 4 \
        --output_dir dbrx/trt_ckpt/int8kv/tp4

trtllm-build --checkpoint_dir dbrx/trt_ckpt/int8kv/tp4 \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --moe_plugin float16 \
        --workers 4 \
        --output_dir dbrx/trt_engines/int8kv/tp4
```

### Run inference

You can test your engines with the [run.py](../../../run.py) script:

```bash
mpirun -n 8 \
    python3 ../run.py --engine_dir dbrx/trt_engines/bf16/tp8 \
        --tokenizer_dir dbrx-base \
        --max_output_len 10 \
        --input_text "What is AGI?"
```

If the engines are run successfully, you will see output like:
```
......
Input [Text 0]: "What is AGI?"
Output [Text 0 Beam 0]: " How do I find it?
AGI stands for"
```


You can also evaluate with the [summarize.py](../../../summarize.py) script:
```bash
mpirun -n 8 \
    python ../summarize.py --engine_dir dbrx/trt_engines/bf16/tp8 \
        --hf_model_dir dbrx-base \
        --test_trt_llm
```

If the engines are run successfully, you will see output like:
```
......
[04/02/2024-11:16:37] [TRT-LLM] [I] TensorRT LLM (total latency: 9.962657451629639 sec)
[04/02/2024-11:16:37] [TRT-LLM] [I] TensorRT LLM (total output tokens: 1189)
[04/02/2024-11:16:37] [TRT-LLM] [I] TensorRT LLM (tokens per second: 119.34566713477734)
[04/02/2024-11:16:37] [TRT-LLM] [I] TensorRT LLM beam 0 result
[04/02/2024-11:16:37] [TRT-LLM] [I]   rouge1 : 26.842471264679535
[04/02/2024-11:16:37] [TRT-LLM] [I]   rouge2 : 9.979512100961314
[04/02/2024-11:16:37] [TRT-LLM] [I]   rougeL : 19.50336050538688
[04/02/2024-11:16:37] [TRT-LLM] [I]   rougeLsum : 22.00400189383231
```
