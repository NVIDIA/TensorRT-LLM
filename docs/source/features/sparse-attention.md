# Sparse Attention

- [Background and Motivation](#background-and-motivation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Usage with trtllm-bench or trtllm-serve](#usage-with-trtllm-bench-or-trtllm-serve)
- [Developer Guide](#developer-guide)
  - [Architecture Overview](#architecture-overview)
  - [Framework Implementation](#framework-implementation)
  - [Implementing a New Algorithm](#implementing-a-new-algorithm)
    - [1. Configuration Class](#1-configuration-class)
    - [2. Implement the prediction module in Attention Backend](#2-implement-the-prediction-module-in-attention-backend)
    - [3. Manage Auxiliary Memory Pool](#3-manage-auxiliary-memory-pool)
    - [4. Registration and Dispatch](#4-registration-and-dispatch)
- [Summary and Future Work](#summary-and-future-work)
    - [Current State](#current-state)
    - [Future Work](#future-work)

## Background and Motivation

As Large Language Models (LLMs) are applied to increasingly complex tasks such as long-document summarization, code generation, and autonomous agents, the demand for processing long contexts and extended generation has surged. In Transformer-based models, the attention mechanism's computational complexity and memory usage grow quadratically and linearly with sequence length, respectively. This creates significant bottlenecks in both the **Context (Prefill)** and **Generation (Decode)** phases:

*   **Context Phase**: Processing long prompts requires substantial memory bandwidth and computation, affecting time-to-first-token (TTFT). Since the context phase is typically compute-bound, reducing the computational load here is critical.
*   **Generation Phase**: The Key-Value (KV) cache grows with every generated token, consuming vast amounts of GPU memory and bandwidth. Since the generation phase is usually memory-bound, reducing the memory footprint directly alleviates memory pressure, improves token-to-token latency (TPOT), and allows for larger batch sizes.

Fortunately, key observations indicate that attention scores naturally exhibit sparsity, meaning not all K/V tokens are necessary for attention computation. To enhance the efficiency of long-sequence LLMs, numerous methods have been proposed to optimize performance by leveraging approximate sparse attention. Among those methods, sparsity can be applied to different dimensions of the attention: head dimension, hidden dimension, and sequence dimension. When applying sparsity to the sequence dimension, those methods selectively compute only the most important query-key pairs. This approach can be referred to as token sparsity. Token sparsity has been widely explored in lots of recent academic works, and it is also a kind of structured sparse method that is friendly for GPU. TensorRT LLM will focus on the sparse attention methods that leverages token sparsity.

Token sparsity can be applied to two distinct aspects of LLM inference:
*   **Sparse Computation**: If a query token does not require the entire history, just skip the computation for irrelevant tokens, thereby reducing attention computational costs.
*   **Sparse KV cache**: Evicts KV tokens from the cache that are not required for future generation steps. This reduces GPU memory usage and lowers computation overhead for subsequent steps.
Both methods can be enabled simultaneously to achieve better performance.

To support these emerging techniques, TensorRT LLM has designed a general, extensible Sparse Attention framework (which is continuously being optimized) to compatibly integrate advanced sparse algorithms. Currently we can support [RocketKV](https://arxiv.org/pdf/2502.14051) and [DSA](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf).

## Quick Start

This section provides a brief guide on enabling sparse attention in TensorRT LLM. For a detailed walkthrough of a specific algorithm, please refer to [RocketKV sparse attention](../../examples/sparse_attention/RocketKV.md).

### Python API

To use sparse attention, you need to configure a specific `SparseAttentionConfig` (for example, `RocketSparseAttentionConfig`) and pass it to the `LLM` constructor.

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

You can enable sparse attention in benchmarking and serving tools by providing a `sparse_attention_config` in an `extra_config.yaml` file.

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

This section describes the sparse attention framework architecture and guides developers on how to implement new sparse attention algorithms in TensorRT LLM. Unless otherwise specified, this framework primarily targets **MHA/MQA/GQA-based** attention mechanisms.

### Architecture Overview

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_doc/docs/source/blogs/media/tech_blog15_sparse_attention_framework.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: The sparse attention framework in TensorRT LLM.</em></sub></p>

Our goal is to design a generic, extensible, and flexible sparse attention framework. Figure 1 illustrates the overall design. The architecture is built by inheriting from the existing `AttentionBackend` to define algorithm-specific sparse attention backends. Within these backends, a `prediction` method is implemented to generate the corresponding sparse indices. These indices are then passed as arguments to the `AttentionOp` to perform the sparse attention computation. This approach balances system flexibility with extensibility, allowing new algorithms to be integrated by simply defining their prediction logic **without** modifying the core attention kernels.

TensorRT LLM abstracts sparse attention into a prediction-based workflow: *a prediction module first identifies the sparse indices (tokens/blocks to keep or attend to), which are then used by the subsequent attention operator*. Currently, for standard attention, TensorRT LLM supports **KV cache compression** in the context phase and **sparse computation** in the generation phase as mentioned above. Different KV heads are allowed to use different sparse indices, while Q heads that map to the same KV head share the same sparse pattern. It does **not** yet support sparse computation in the context phase or KV cache compression in the generation phase.

TensorRT LLM currently supports the following operations for standard attention:

1.  **Context Phase (KV cache compression)**:
    *   **Goal**: Reduce the size of the KV cache populated during the context phase.
    *   **Mechanism**: Identify important tokens from the prompt and permanently evict non-essential tokens before entering the generation phase.

2.  **Generation Phase (sparse computation)**:
    *   **Goal**: Accelerate attention computation during token generation.
    *   **Mechanism**: For each new token, dynamically select a subset of relevant blocks/tokens from the kv cache to attend to.

However, Multi-head Latent Attention (MLA), used by algorithms like DSA, is a special case. It currently supports sparse computation in both context and generation phases, but does not support KV cache compression. Its sparse computation implementation is handled directly within the TRTLLM-GEN MLA kernel and does not use the general pass described below.

### Framework Implementation

To hide the complexity of sparse algorithms, the main prediction logic is encapsulated within the `tensorrt_llm._torch.attention_backend` module.

We have extended the existing `AttentionBackend` to include a prediction step that retrieves sparse indices before the attention operation. The logic flow in `TrtllmAttention` is conceptually:

```python
# Predict indices for KV Cache compression (context phase)
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

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_doc/docs/source/blogs/media/tech_blog15_sparse_attention_op.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Sparse attention operator workflow in TensorRT LLM.</em></sub></p>

In `AttentionOp`, as illustrated in Figure 2, for the GQA/MHA path we have implemented two kernels, `updateSparseKvCacheAfterFmha` and `gatherKvPageOffsetsKernel`, applied in the context and generation phases respectively:

*   **`updateSparseKvCacheAfterFmha`**: Invoked in the post-processing stage after the context attention computation. It performs a rewrite of the KV cache based on the selected indices, effectively implementing KV cache compression.

*   **`gatherKvPageOffsetsKernel`**: Executed before the attention computation in the generation phase. It converts the input sparse indices (which can be of arbitrary granularity) into page-aligned indices. It then gathers `kv_page_offsets` and updates `kv_len` to produce new metadata, which is fed into the subsequent attention kernel for computation.

Currently, for GQA/MHA, sparse attention only supports sparse computation at page-size granularity in the generation phase. In addition, we provide a sparse MLA kernel that supports token-level sparse computation in both the context and generation phases.

Many sparse attention algorithms also require additional auxiliary memory. In the current system, there are two paths to fulfill this requirement:

*   Implement a simple, custom CacheManager at the Python level, inheriting from `KVCacheManager`.

*   Use `KVCacheManagerCpp` to simultaneously manage both the KV Cache and auxiliary memory.

Each option has its own advantages and disadvantages, which we summarize below.

### Implementing a New Algorithm

#### 1. Configuration Class

Define a configuration class in `tensorrt_llm/llmapi/llm_args.py` inheriting from `BaseSparseAttentionConfig`. This class should hold user-tunable parameters for your algorithm.

```python
@dataclass
class MySparseAttentionConfig(BaseSparseAttentionConfig):
    topk: int = 64
    # ... other parameters
```

#### 2. Implement the prediction module in Attention Backend

Create a new class inheriting from `TrtllmAttention` (in `tensorrt_llm/_torch/attention_backend/trtllm.py`). You typically need to override two main prediction methods:

**`sparse_kv_predict(self, q, k, metadata, ...)`**
*   **Purpose**: Predict indices for KV cache compression during the context phase.
*   **Behavior**: This function performs prediction to return the indices of tokens to be preserved in the KV cache.
*   **Output**: `sparse_kv_indices` (tokens to keep).
*   **KV Cache Update**: The system calls `updateSparseKvCacheAfterFmha` to gather the KV cache based on these indices. This effectively "compresses" the prompt's KV cache.
*   **Constraint**: Returned indices must be **sorted** to ensure safe in-place gathering in memory. Note that this post-processing "gather" step introduces some overhead, but significantly improves flexibility, allowing compatibility with features in context like chunked prefill.

**`sparse_attn_predict(self, q, k, metadata, ...)`**
*   **Purpose**: Predict indices for **sparse computation** during the generation phase.
*   **Behavior**: For the current query token, predict which pages/blocks in the KV cache are relevant.
*   **Output**: `sparse_attn_indices` (relevant blocks/tokens).
*   **KV Cache Selection**: These indices are passed to the underlying C++ attention operator. The `gatherKvPageOffsetsKernel` uses these indices to gather `kv_page_offsets` and update `kv_len`, enabling the attention kernel to perform "dense" attention on just the selected pages.
*   **Constraint**: The generation phase sparse computation is supported for NVIDIA Blackwell GPUs and newer (SM 100+) using TRTLLM-GEN kernels. However, it is flexible enough to extend to different architectures. Currently, only KV cache's **page-level** granularity is supported for sparse computation.

**Note**: The prediction process can be time-consuming, especially in low-latency scenarios where it might account for a significant portion of the attention time. It is highly recommended to optimize this step using custom Triton or CUDA kernels.

#### 3. Manage Auxiliary Memory Pool

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

**Note**: If your algorithm involves KV cache compression, standard KV cache block reuse is generally incompatible because eviction modifies the block content uniquely for each request. However, algorithms like DSA that use low-rank approximation without eviction can support block reuse.

#### 4. Registration and Dispatch

*   Register your config and backend in `tensorrt_llm/_torch/attention_backend/sparse/utils.py` and `tensorrt_llm/_torch/pyexecutor/_util.py` to ensure the system routes the request to your new backend when the config is present.
*   Add initialization logic in `cpp/tensorrt_llm/thop/attentionOp.cpp` and `cpp/tensorrt_llm/kernels/sparseAttentionKernels.h` if new C++ level parameters are required.

## Summary and Future Work

### Current State

Currently, the status of the Sparse Attention framework is as follows:

1.  **Supported Operations**: The `AttentionOp` currently supports **KV cache compression** in the context phase and **sparse computation** in the generation phase. Other combinations (for example, sparse computation in the context phase) are not yet supported for MHA/GQA. For MLA, sparse computation is supported in both the context and generation phases.
2.  **Algorithm Support**: RocketKV is supported in both the vanilla (PyTorch) backend and the TRTLLM backend, while DSA is supported in the TRTLLM backend. These implementations validate the generality and flexibility of the framework.
3.  **Auxiliary Memory**: Both Python-level and C++-level implementations are algorithm-specific. There is no unified abstraction for auxiliary memory management yet.

### Future Work

*   **Sparse Computation in Context Phase**: We plan to introduce sparse computation support for the context phase for MHA/GQA, allowing the TensorRT LLM sparse attention framework to cover most scenarios.
*   **Dynamic Eviction in Generation Phase**: Dynamically evicting KV cache blocks during the generation phase poses significant challenges to KV cache flexibility. While difficult to implement in the current framework, block-level eviction appears to be a promising compromise and is under further exploration.
*   **Unified Auxiliary Memory Management**: We are exploring a unified mechanism to manage auxiliary memory pools. This would allow users to define custom auxiliary spaces more flexibly while automatically inheriting advanced features from the KV cache, such as reuse and offloading.
*   **Code Refactoring**: As more sparse attention algorithms are integrated, the framework will undergo refactoring to unify code and improve maintainability.
*   **Optimization and Feature Integration**: We are discussing further optimizations, such as enabling fine-grained token-level sparse computation for MHA/GQA. Additionally, we are exploring integration with other advanced features like Disaggregated Serving.
