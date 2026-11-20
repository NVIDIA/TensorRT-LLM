# Deployment Guide for Llama3.3 70B on TensorRT LLM - Blackwell & Hopper Hardware

(Trimmed excerpt for this exercise.)

## Introduction

This guide covers serving `nvidia/Llama-3.3-70B-Instruct-FP8` with the TensorRT LLM
PyTorch backend through `trtllm-serve`.

## Prepare the config

The checked-in starting config is `examples/configs/curated/llama-3.3-70b.yaml`. It
was prepared for the high-throughput 1024/1024 ISL/OSL test.

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/llama-3.3-70b.yaml
```

## Launch the TensorRT LLM Server

```shell
trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP8 --host 0.0.0.0 --port 8000 --config ${EXTRA_LLM_API_FILE}
```

## LLM API Options (YAML Configuration)

#### `tensor_parallel_size`
Sets the tensor-parallel size. This should typically match the number of GPUs you
intend to use for a single model instance.

#### `max_batch_size`
The maximum number of user requests that can be grouped into a single batch.

#### `max_num_tokens`
The maximum total number of tokens (across all requests) allowed inside a single
scheduled batch.

#### `max_seq_len`
The maximum possible sequence length for a single request, including both input and
generated output tokens.

#### `trust_remote_code`
Allows loading and executing custom Python code provided by the model repository.

#### `kv_cache_config`
`dtype`: data type for the KV cache. Default: auto (uses the checkpoint's data type).

#### `cuda_graph_config`
`enable_padding`: pad input batches to the nearest CUDA graph batch size.
`max_batch_size`: maximum batch size for which a CUDA graph is created.
Recommendation: set this to the same value as `max_batch_size`.
