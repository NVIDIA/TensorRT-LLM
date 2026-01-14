# RocketKV Sparse Attention

This document details enabling RocketKV sparse attention within TensorRT LLM.

RocketKV is a training-free, two-stage KV cache compression method designed to accelerate long-context LLM inference. It combines permanent KV token eviction (in context phase) with dynamic KV token selection (in generation phase) to significantly reduce memory bandwidth usage and increase throughput while maintaining high accuracy.

For more technical details, please refer to the paper: [RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](https://arxiv.org/pdf/2502.14051). Here is an official implement which provides a reference: [RocketKV Repo](https://github.com/NVlabs/RocketKV).

## Overview

In Transformer-based LLM inference, the KV cache grows linearly with sequence length, becoming a major bottleneck. RocketKV mitigates this issue through a two-stage process:

1.  **Context Phase (Stage 1):** It performs **permanent KV cache eviction**. Instead of storing the full history, it selects and keeps a `prompt_budget` of the most important tokens based on attention scores.
2.  **Generation Phase (Stage 2):** It utilizes a **dynamic Top-K token selection**. It maintains a lightweight, compressed auxiliary cache (KT Cache) to dynamically predict which tokens of the KV cache are relevant for the current token, and loading only those tokens to do the attention computation.

RocketKV is integrated into TensorRT LLM as a specialized attention backend, accessible via the LLM API. Specifically, the core sparse KV prediction kernels are implemented using **Triton** kernels, achieving highly optimized performance on modern NVIDIA GPUs.

## Support Matrix

  * GPU Compute Capability >= 10.0 (Blackwell or newer)
  * FP16 / BF16 / FP8
  * Paged KV Cache
  * Tensor Parallel
  * Cuda Graph

**Note:** 
1. RocketKV currently requires `enable_block_reuse=False` in the KV cache configuration, as the sparse eviction logic is incompatible with standard block reuse mechanisms. 
2. RocketKV doesn't support `enable_chunked_prefill=True` for now.
3. RocketKV doesn't support *disagg-serving* as well, since it needs the KV cache transmission from prefill engine to the decode engine. But currently RocketKV uses a python kt cache manager and it cannot support this transmission.

## Usage

To enable RocketKV, configure `RocketSparseAttentionConfig` and pass it to the `LLM` class constructor.

### Python API

Integrate RocketKV into your workflows using the `tensorrt_llm.llmapi` interface.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import RocketSparseAttentionConfig, KvCacheConfig

# 1. Define the RocketKV Sparse Attention Configuration
rocket_config = RocketSparseAttentionConfig(
    window_size=32,       # Size of the recent window to always keep
    kernel_size=63,       # Pooling kernel size for importance scoring
    prompt_budget=2048,   # Number of tokens to keep from the prompt (Stage 1)
    topk=64,              # Number of tokens to retrieve during generation (Stage 2)
    topr=128,             # Number of query channels to keep for scoring
    kt_cache_dtype='float8_e5m2' # Dtype for the auxiliary Key-Token cache
)

# 2. Initialize the LLM with the config and 'pytorch' backend
# Note: Block reuse must be disabled for RocketKV
kv_config = KvCacheConfig(enable_block_reuse=False)

llm = LLM(
    model="<path_to_model>",
    backend='pytorch',    # RocketKV currently requires the PyTorch backend
    sparse_attention_config=rocket_config,
    kv_cache_config=kv_config, 
)

# 3. Generate
prompts = ["To be or not to be, that is the question."]
sampling_params = SamplingParams(max_tokens=128)
outputs = llm.generate(prompts, sampling_params)
```

### Running the Example Script

We provide a reference script `examples/llm-api/llm_sparse_attention.py` to demonstrate RocketKV capabilities.

**Example Command:**

```bash
# Adjust --model_path to your local Llama checkpoint
python3 ../llm-api/llm_sparse_attention.py \
    --model_path <path_to_model> \
    --algo ROCKETKV \
    --attention_backend TRTLLM \
    --window_size 32 \
    --kernel_size 63 \
    --prompt_budget 2048 \
    --topk 64 \
    --topr 128 \
    --kt_cache_dtype float8_e5m2 \
    --max_seq_len 10240 \
    --max_num_tokens 10240 \
    --max_new_tokens 128
```


### Usage with `trtllm-bench` and `trtllm-serve`

Sparse attention options must be specified via `--config config.yaml` for both `trtllm-bench` and `trtllm-serve`. All sparse attetnion options can be specified in this YAML file and the argument names/valid values are the same as in their corresponding configuration described in the Configuration Arguments section. For example, a YAML configuration could look like this:

```
backend: pytorch
attn_backend: TRTLLM
sparse_attention_config:
  algorithm: rocket
  kt_cache_dtype: float8_e5m2
  window_size: 32
  prompt_budget: 2048
kv_cache_config:
  enable_block_reuse: false
enable_chunked_prefill: false
```

Run the command with the config file:
```bash
trtllm-eval/trtllm-bench/trtllm-serve --model <model_path> --config extra_config.yaml ...
```

For example, users can evaluate a model with trtllm-eval on LongBenchV2 task like this:

```bash
trtllm-eval --model <path_to_model> --config extra_config.yaml longbench_v2 --max_output_length 1024 ...
```

## Configuration Arguments

The `RocketSparseAttentionConfig` allows fine-grained control over compression ratios and performance trade-offs:

*   **`prompt_budget`** (int, default=2048): The number of tokens to retain from the input prompt (context). RocketKV compresses the prompt to this size by evicting less important tokens based on importance scores.
*   **`topk`** (int, default=64): The number of KT pages to select dynamically during the generation phase. Note that the selection is performed at the granularity of KT cache pages, but the actual attention kernel retrieves data based on the granularity of KV cache page size.
*   **`topr`** (int/float, default=128): The number of query feature dimensions to use when computing the relevance score between Query and KT Cache. This acts as a dimensionality reduction to speed up the selection process. However, it's recommended to set it equal to `head_dim` to skip `topr_filter` computations for better performance and accuracy.
*   **`window_size`** (int, default=32): The size of the sliding window in RocketKV. In the context phase, RocketKV uses the last `window_size` tokens of the Query and the Key prefix to compute importance scores for eviction. These recent tokens are always retained in the cache, and `prompt_budget-window_size` important tokens in the prefix are retained in the cache also.
*   **`kernel_size`** (int, default=63): The size of the 1D max-pooling kernel used in the context phase. It smooths attention scores to better identify locally important regions rather than just isolated high-score tokens.
*   **`kt_cache_dtype`** (str, default='float8_e5m2'): The data type for the auxiliary "Key-Token" (KT) cache used for relevance prediction.
    *   `float8_e5m2`: Recommended. Provides memory savings for the auxiliary structure and speedup for the prediction kernels.
    *   `bfloat16`: Standard precision.
*   **`page_size`** (int, default=4): The granularity of the sparse token selection (KT page). Currently, only **powers of 2** are supported due to Triton kernel limitations. Accuracy is generally maintained well for `page_size <= 4`.
