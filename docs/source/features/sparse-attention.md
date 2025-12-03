# Sparse Attention

- [Motivation](#motivation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Usage with `trtllm-bench` or `trtllm-serve`](#usage-with-trtllm-bench-or-trtllm-serve)
- [Developer Guide](#developer-guide)
  - [Architecture Overview](#architecture-overview)
  - [Framework Implementation](#framework-implementation)
  - [Implementing a New Algorithm](#implementing-a-new-algorithm)
    - [1. Implement the prediction module in Attention Backend](#1-implement-the-prediction-module-in-attention-backend)
    - [2. Manage Auxiliary Memory Pool](#2-manage-auxiliary-memory-pool)
    - [3. Configuration Class](#3-configuration-class)
    - [4. Registration and Dispatch](#4-registration-and-dispatch)
- [Future Work](#future-work)

## Motivation

As Large Language Models (LLMs) are applied to increasingly complex tasks such as long-document summarization, code generation, and autonomous agents, the demand for processing long contexts and extended generation has surged. In Transformer-based models, the attention mechanism's computational complexity and memory usage grow quadratically and linearly with sequence length. This creates significant bottlenecks in both the **Context (Prefill)** and **Generation (Decode)** phases:

*   **Context Phase**: Processing long prompts requires substantial memory bandwidth and computation, affecting time-to-first-token (TTFT). Since the context phase is typically compute-bound, reducing the computational load here is critical.
*   **Generation Phase**: The Key-Value (KV) cache grows with every generated token, consuming vast amounts of GPU memory and bandwidth. Since the generation phase is memory-bound, reducing the memory footprint directly alleviates memory pressure, improves token-to-token latency (TPOT), and allows for larger batch sizes.

Consequently, using sparse attention to reduce overhead in both context and generation phases has attracted significant research interest. Several state-of-the-art models and techniques, such as DeepSeek's NSA/DSA and RocketKV, are evolving to minimize these overheads. Sparse Attention addresses these challenges by selectively attending to the most important tokens rather than the entire history. By reducing the number of tokens involved in calculation and storage, Sparse Attention can:

*   **Reduce Memory Footprint**: Allow for larger batch sizes or longer context windows. Saving memory is particularly beneficial in the memory-bound generation phase.
*   **Lower Computation Overhead**: Decrease the FLOPs required for attention. This is especially important for the compute-intensive context phase.

To support these emerging techniques, TensorRT LLM has designed a general, extensible Sparse Attention framework (which is continuously being optimized) to compatibly integrate advanced sparse algorithms.

## Quick Start

This section provides a brief guide on enabling Sparse Attention in TensorRT LLM. For a detailed walkthrough of a specific algorithm, please refer to [RocketKV Sparse Attention](../../examples/sparse_attention/RocketKV.md).

### Python API

To use Sparse Attention, you need to configure a specific `SparseAttentionConfig` (e.g., `RocketSparseAttentionConfig`) and pass it to the `LLM` constructor.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import RocketSparseAttentionConfig, KvCacheConfig

# 1. Configure Sparse Attention
# Example: RocketKV configuration
rocket_config = RocketSparseAttentionConfig(
    prompt_budget=2048,
    kt_cache_dtype='float8_e5m2'
)

# 2. Configure KV Cache
# Note: Some sparse algorithms (like RocketKV) may require disabling block reuse
kv_config = KvCacheConfig(enable_block_reuse=False)

# 3. Initialize LLM
llm = LLM(
    model="<path_to_model>",
    backend='pytorch',    # Currently requires the PyTorch backend
    sparse_attention_config=rocket_config,
    kv_cache_config=kv_config,
)

# 4. Generate
prompts = ["To be or not to be..."]
outputs = llm.generate(prompts, SamplingParams(max_tokens=128))
```

### Usage with `trtllm-bench` or `trtllm-serve`

You can enable Sparse Attention in benchmarking and serving tools by providing a `sparse_attention_config` in an `extra_config.yaml` file.

**extra_config.yaml:**
```yaml
backend: pytorch
attn_backend: TRTLLM
sparse_attention_config: # RocketKV as an example
  algorithm: rocket
  kt_cache_dtype: float8_e5m2
  prompt_budget: 2048
kv_cache_config:
  enable_block_reuse: false
enable_chunked_prefill: false
```

Run the command with the config file:
```bash
trtllm-bench/trtllm-serve --model <model_path> --extra_llm_api_options extra_config.yaml ...
```

For example, users can evaluate a model with trtllm-eval on LongBenchV2 task like this:

```bash
trtllm-eval --model <path_to_model> --extra_llm_api_options extra_config.yaml longbench_v2 --max_output_length 1024 ...
```

## Developer Guide

This section describes the Sparse Attention framework architecture and guides developers on how to implement new sparse attention algorithms in TensorRT LLM. Unless otherwise specified, this framework primarily targets **MHA/MQA/GQA-based** attention mechanisms.

### Architecture Overview

TensorRT LLM abstracts sparse attention into a prediction-based workflow: *a prediction module first identifies the sparse indices (tokens/blocks to keep or attend to), which are then used by the subsequent attention operator*.

TensorRT LLM currently supports the following operations:

1.  **Context Phase (Sparse Storage)**:
    *   **Goal**: Reduce the size of the KV cache populated during the context phase.
    *   **Mechanism**: Identify important tokens from the prompt and permanently evict non-essential tokens before entering the generation phase.

2.  **Generation Phase (Sparse Computation)**:
    *   **Goal**: Accelerate attention computation during token generation.
    *   **Mechanism**: For each new token, dynamically select a subset of relevant blocks/tokens from the KV cache to attend to.

Currently, for standard attention, TensorRT LLM supports sparse storage in the context phase and sparse computation in the generation phase as above mentioned. It's allowed to have different sparse indices in different kv heads (but shared with same sparse pattern across q heads). It does **not** yet support sparse computation in the context phase or sparse storage (dynamic eviction) in the generation phase.

However, Multi-head Latent Attention (MLA), used by algorithms like DSA, is a special case. It currently supports sparse computation in both context and generation phases, but does not support sparse storage. Its sparse computation implementation is handled directly within the TRTLLM-GEN MLA kernel and does not use the general pass described below.

### Framework Implementation

To hide the complexity of sparse algorithms, the main prediction logic is encapsulated within the `tensorrt_llm._torch.attention_backend` module.

We have extended the existing `AttentionBackend` to include a prediction step that retrieves sparse indices before the attention operation. The logic flow in `TrtllmAttention` is conceptually:

```python
# Predict indices for sparse storage (context phase)
sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(
    q, k, metadata, **kwargs)

# Predict indices for sparse computation (generation phase)
sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(
    q, k, metadata, **kwargs)
```

The specific prediction logic is hidden in the subclasses, where developers implement `sparse_kv_predict` and `sparse_attn_predict`.

The key files located in `tensorrt_llm/_torch/attention_backend/sparse/` are:

*   `rocket.py`, `dsa.py`: Implementations of specific algorithms (e.g., RocketKV, DSA).
*   `kernel.py`: Custom Triton kernels for importance scoring or selection.
*   `utils.py`: Dispatch related logic.

### Implementing a New Algorithm

To add a new Sparse Attention algorithm, you need to implement a new backend inheriting from `TrtllmAttention` and define its configuration.

#### 1. Implement the prediction module in Attention Backend

Create a new class inheriting from `TrtllmAttention` (in `tensorrt_llm/_torch/attention_backend/trtllm.py`). You typically need to override two main prediction methods:

**`sparse_kv_predict(self, q, k, metadata, ...)`**
*   **Purpose**: Predict indices for **sparse storage** during the context phase.
*   **Behavior**: This function performs prediction to return the indices of tokens to be preserved in the KV cache.
*   **Output**: `sparse_kv_indices` (tokens to keep).
*   **KV Cache Update**: The system calls `updateSparseKvCacheAfterFmha` to gather the KV cache based on these indices. This effectively "compresses" the prompt's KV cache.
*   **Constraint**: Returned indices must be **sorted** to ensure safe in-place gathering in memory (implemented in `updateSparseKvCacheAfterFmha`). Note that this post-processing "gather" step introduces some overhead, but significantly improves flexibility, allowing compatibility with features in context like chunked prefill.

**`sparse_attn_predict(self, q, k, metadata, ...)`**
*   **Purpose**: Predict indices for **sparse computation** during the generation phase.
*   **Behavior**: For the current query token, predict which pages/blocks in the KV cache are relevant.
*   **Output**: `sparse_attn_indices` (relevant blocks/tokens).
*   **KV Cache Selection**: These indices are passed to the underlying C++ attention operator. A specific kernel, `gatherKvPageOffsetsKernel`, uses these indices to gather `kv_page_offsets` and update `kv_len`, but not the KV cache directly to reduce the overhead. This enables the attention kernel to perform "dense" attention on just the selected pages. The `gatherKvPageOffsetsKernel` supports **arbitrary granularity** (e.g., token-level or page-level) and **unordered indices**. The granularity is obtained from `sparse_attention_config.get_indices_block_size`. The kernel automatically maps these arbitrary, potentially unordered indices to the granularity of KV cache page size.
*   **Constraint**: The generation phase sparse computation is supported for NVIDIA Blackwell GPUs and newer (SM 100+) using TRTLLM-GEN kernels. However, it is flexible enough to extend to different architectures. Currently, only KV cache's **page-level** granularity is supported for sparse computation.

**Note**: The prediction process can be time-consuming, especially in low-latency scenarios where it might account for a significant portion of the attention time. It is highly recommended to optimize this step using custom Triton or CUDA kernels.

#### 2. Manage Auxiliary Memory Pool

Many sparse algorithms (like RocketKV or DSA) require auxiliary structures (e.g., a "KT cache" or "Kcache") to select relevant tokens. There are two primary ways to manage this memory in TensorRT LLM:

**Option A: Python-level Custom Manager**

You can implement a custom manager in Python.
*   **Use Case**: Algorithms like RocketKV use this approach to store the KT cache (e.g., `RocketKVCacheManager` in `rocket.py`).
*   **Implementation**: Create a Python level cache manager that handles the allocation and lifecycle of the auxiliary tensors.
*   **BlockManager Integration**: It is recommended to use the existing `BlockManager` to manage the auxiliary pools if possible. This allows the auxiliary pool to share block logic with the main KV cache, reducing implementation overhead.
*   **Key Methods to Override**:
    *   `get_cache_size_per_token` / `get_cache_bytes_per_token`: Update `kv_factor` correctly to include the size of the auxiliary structures so TensorRT LLM allocates sufficient GPU memory.
    *   `add_dummy_requests` / `prepare_resources`: Ensure the auxiliary pool allocates correct resources/tokens for new requests.
*   **Pros**: More flexible and easier to implement.
*   **Cons**: This approach operates at the Python level, making it difficult to share features of the KV cache managed at the C++ level (e.g., advanced transmission or kvcache reuse features tied to the C++ manager).

**Option B: C++ Integrated Manager**

For tighter integration, you can manage the auxiliary memory within the C++ `KVCacheManager`.
*   **Use Case**: Algorithms like DSA use this approach to store the Kcache.
*   **Pros**: Enables compatibility with advanced features such as KV cache reuse and disagg-serving. For example, DSA's low-rank Kcache can be reused or transmitted between context and generation engines.
*   **Cons**: Higher implementation complexity. The current C++ `KVCacheManager` is optimized for the standard KV cache pool. Adding custom pools often requires significant modifications or manual implementation of the pool management logic within the C++ level.

**Note**: If your algorithm involves sparse storage, standard KV cache block reuse is generally incompatible because eviction modifies the block content uniquely for each request. However, algorithms like DSA that use low-rank approximation without eviction can support block reuse.

#### 3. Configuration Class

Define a configuration class in `tensorrt_llm/llmapi/llm_args.py` inheriting from `BaseSparseAttentionConfig`. This class should hold user-tunable parameters.

```python
@dataclass
class MySparseAttentionConfig(BaseSparseAttentionConfig):
    topk: int = 64
    # ... other parameters
```

#### 4. Registration and Dispatch

*   Register your config and backend in `tensorrt_llm/_torch/attention_backend/sparse/utils.py` and `tensorrt_llm/_torch/pyexecutor/_util.py` to ensure the system routes the request to your new backend when the config is present.
*   Add initialization logic in `cpp/tensorrt_llm/thop/attentionOp.cpp` and `cpp/tensorrt_llm/kernels/sparseAttentionKernels.h` if new C++ level parameters are required.

## Future Work

*   **Sparse Computation in Context Phase**: We plan to introduce sparse computation support for the context phase, allowing the TensorRT LLM sparse attention framework to cover most scenarios.
*   **Dynamic Eviction in Generation Phase**: Dynamically evicting KV cache blocks during the generation phase poses significant challenges to KV cache flexibility. While difficult to implement in the current framework, block-level eviction appears to be a promising compromise and is under further exploration.
*   **Code Refactoring**: As more sparse attention algorithms are integrated, the framework will undergo refactoring to unify code and improve maintainability.
*   **Optimization and Feature Integration**: We are discussing further optimizations, such as enabling fine-grained token-level sparse computation. Additionally, we are exploring integration with other advanced features like KV cache reuse and disagg-serving.
