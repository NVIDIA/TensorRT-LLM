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
    - [Current Status](#current-status)
    - [Future Work](#future-work)

## Background and Motivation

As Large Language Models (LLMs) are applied to increasingly complex tasks such as long-document summarization, code generation, and autonomous agents, the demand for processing long contexts and extended generation has surged. In Transformer-based models, the attention mechanism's computational complexity and memory usage grow quadratically and linearly with sequence length, respectively. This creates significant bottlenecks in both the **Context (Prefill)** and **Generation (Decode)** phases:

*   **Context Phase**: Processing long prompts requires substantial memory bandwidth and computation, affecting time-to-first-token (TTFT). Since the context phase is typically compute-bound, reducing the computational load here is critical.
*   **Generation Phase**: The Key-Value (KV) cache grows with every generated token, consuming vast amounts of GPU memory and bandwidth. Since the generation phase is usually memory-bound, reducing the memory footprint directly alleviates memory pressure, improves token-to-token latency (TPOT), and allows for larger batch sizes.

Fortunately, key observations indicate that attention scores naturally exhibit sparsity, meaning not all K/V tokens are necessary for attention computation. To enhance the efficiency of long-sequence LLMs, numerous methods have been proposed to optimize performance by leveraging approximate sparse attention. Among those methods, sparsity can be applied to different dimensions of the attention: head dimension, hidden dimension, and sequence dimension. When applying sparsity to the sequence dimension, those methods selectively compute only the most important query-key pairs. This approach can be referred to as token sparsity. Token sparsity has been widely explored in lots of recent academic works, and it is also a kind of structured sparse method that is friendly for GPU. Currently, TensorRT LLM focuses on the sparse attention methods that leverages token sparsity.

Token sparsity can be applied to two distinct aspects of LLM inference:
*   **Sparse Computation**: If a query token does not require the entire history, just skip the computation for irrelevant tokens, thereby reducing attention computational costs.
*   **Sparse KV cache**: Evicts KV tokens from the cache that are not required for future generation steps. This reduces GPU memory usage and lowers computation overhead for subsequent steps.

Both methods can be enabled simultaneously to achieve better performance.

To support these emerging techniques, TensorRT LLM has designed a general, extensible and flexible **sparse attention framework** (which is continuously being optimized) to compatibly integrate advanced sparse algorithms. Currently we can support [RocketKV](https://arxiv.org/pdf/2502.14051) and [DSA](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf).

## Quick Start

This section provides a brief guide on enabling sparse attention in TensorRT LLM, using RocketKV as an example. For more details, please refer to [RocketKV sparse attention](../../examples/sparse_attention/RocketKV.md).

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
trtllm-bench/trtllm-serve --model <model_path> --config extra_config.yaml ...
```

For example, users can evaluate a model with trtllm-eval on LongBenchV2 task like this:

```bash
trtllm-eval --model <path_to_model> --config extra_config.yaml longbench_v2 --max_output_length 1024 ...
```

## Developer Guide

This section describes the sparse attention framework architecture and guides developers on how to implement new sparse attention algorithms in TensorRT LLM. Unless otherwise specified, this framework primarily targets **MQA/GQA/MLA-based** attention mechanisms.

### Architecture Overview

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/sparse_attention_framework.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: The sparse attention framework in TensorRT LLM.</em></sub></p>

Our goal is to design a general, extensible, and flexible sparse attention framework. In this framework, the attention operator provides the unified APIs to support both **sparse computation** and **sparse KV cache** that leverage token sparsity, while the users/developers can only focus on the algorithm of sparse attentions, i.e. how to accurately identify important query-key pairs.

For the generality, TensorRT LLM abstracts sparse attention into a prediction-based workflow: *a prediction module first identifies the sparse indices (tokens/blocks to keep or attend to), which are then used by the subsequent attention operator*. Currently, for standard attention (MQA/GQA), TensorRT LLM supports **sparse KV cache** in the context phase and **sparse computation** in the generation phase. Different KV heads are allowed to use different sparse indices, while Q heads that map to the same KV head share the same sparse pattern. It does **not** yet support sparse computation in the context phase or sparse KV cache in the generation phase.

For the scalability, figure 1 illustrates the overall design. The architecture is built by inheriting from the existing `AttentionBackend` to define algorithm-specific sparse attention backends. Within these backends, `prediction` methods are implemented to generate the corresponding sparse indices. These indices are then passed as arguments to the `AttentionOp` to perform the sparse attention computation. This approach balances system flexibility with extensibility, allowing new algorithms to be integrated by simply defining their prediction logic **without** modifying the core attention kernels.

TensorRT LLM currently supports the following features:

1.  **Context Phase**:
    *   **sparse computation**: MLA
    *   **sparse KV cache**: MQA/GQA

2.  **Generation Phase**:
    *   **sparse computation**: MLA/MQA/GQA
    *   **sparse KV cache**: no support yet

### Framework Implementation

To hide the complexity of sparse algorithms, the main prediction logic is encapsulated within the `tensorrt_llm._torch.attention_backend` module.

We have extended the existing `AttentionBackend` to include a prediction step that retrieves sparse indices before the attention operation. These indices are generated using two prediction methods:

```python
# Predict indices for sparse KV Cache
sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(
    q, k, metadata, **kwargs)

# Predict indices for sparse computation
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
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/sparse_attention_op.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Sparse attention operator workflow in TensorRT LLM.</em></sub></p>

In `AttentionOp`, currently, the MQA/GQA sparse attention only supports sparse computation at block granularity in the generation phase, where the block size equals to the page size of the KV cache. It means that we can skip the attention computation of those unimportant pages. In addition, we provide a sparse MLA kernel that supports token-level sparse computation in both the context and generation phases.

To support those features, as illustrated in figure 2, we have implemented two kernels for the MQA/GQA path, `updateSparseKvCacheAfterFmha` and `gatherKvPageOffsetsKernel`, applied in the context and generation phases respectively:

*   **`updateSparseKvCacheAfterFmha`**: Invoked in the post-processing stage after the context attention computation. It selects the important KV tokens and write those K/V vectors to the KV cache to reduce the KV cache size.

*   **`gatherKvPageOffsetsKernel`**: Executed before the attention computation in the generation phase. It converts the input sparse indices (which can be of arbitrary granularity) into page-aligned indices. This means that if a single token is selected, the entire page it is included in the attention computation. After this conversion, we will get a new `kv_page_offsets` and also an updated `kv_len` that is the number of those selected KV tokens. Then these new metadata are fed into the subsequent attention kernel for computation.

For sparse MLA, the kernel supports token sparsity directly, eliminating the need for `gatherKvPageOffsetsKernel`. However, please note that sparse KV cache support is not yet available.

Many sparse attention algorithms also require additional auxiliary memory. In the current system, there are two paths to support this feature:

*   Implement a simple, custom CacheManager at the Python level, inheriting from `KVCacheManager`.

*   Use `KVCacheManagerCpp` to simultaneously manage both the KV Cache and auxiliary memory.

Each option has its own advantages and disadvantages, please refer to the [Manage Auxiliary Memory Pool](#3-manage-auxiliary-memory-pool) for more details.

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

**`sparse_kv_predict(self, q, k, metadata, **kwargs)`**
*   **Behavior**: This function performs prediction to return the indices of tokens to be preserved in the KV cache.
*   **Output**: 
    - `sparse_kv_indices`: The token indices of the important tokens on sequence dimension, shape `(nHeads, nTokens)`, where `nHeads` is the number of KV heads and `nTokens` is the total number of selected tokens across all samples in the batch.
    - `sparse_kv_offsets`: The offset for the `sparse_kv_indices`, shape `(nBatch + 1)`, where `nBatch` is the number of the batch size. The index for head `h` and sample `n` can be obtained via `sparse_kv_indices[h, sparse_kv_offsets[n]]`.
*   **Constraint**: Returned indices must be **sorted** to ensure safe in-place gathering in memory. Note that this post-processing "gather" step introduces some overhead, but significantly improves flexibility, allowing compatibility with features in context like chunked prefill.

**`sparse_attn_predict(self, q, k, metadata, **kwargs)`**
*   **Behavior**: For the current query tokens, predict and return the sparse indices for sparse computation.
*   **Output**: 
    - `sparse_attn_indices`: The block indices of the block sparse attention on the KV sequence dimension, shape `(nHeads, nBlocks)`, where `nHeads` is the number of KV heads and `nBlocks` is the total number of selected blocks across all samples in the batch. For block sparse attention, the block size is defined by `sparse_attn_indices_block_size`, which supports arbitrary values.
    - `sparse_attn_offsets`: The offset for the `sparse_attn_indices`, shape `(nBatch + 1)`, where `nBatch` is the number of the batch size. The index for head `h` and sample `n` can be obtained via `sparse_attn_indices[h, sparse_attn_offsets[n]]`.
*   **Constraint**: The generation phase sparse computation is supported for NVIDIA Blackwell GPUs and newer (SM 100+) using TRTLLM-GEN kernels. However, it is flexible enough to extend to different architectures. Currently, only KV cache's **page-level** granularity is supported for sparse computation.

**Note**: The prediction process can be time-consuming, especially in low-latency scenarios where it might account for a significant portion of the attention time. It is highly recommended to optimize this step using custom kernels.

#### 3. Manage Auxiliary Memory Pool

Many sparse algorithms (like RocketKV or DSA) require auxiliary structures (e.g., a "KT cache" in RocketKV) to select relevant tokens. There are two primary ways to manage this memory in TensorRT LLM:

**Option A: Python-level Custom Manager**

You can implement a custom manager in Python.
*   **Use Case**: Algorithms like RocketKV use this approach to store the KT cache (e.g., `RocketKVCacheManager` in `rocket.py`).
*   **Implementation**: Create a Python level cache manager that handles the allocation and lifecycle of the auxiliary tensors. It is recommended to use the existing `BlockManager` to manage the auxiliary pools if possible. This allows the auxiliary pool to share block manager logics, reducing implementation overhead.
*   **Key Methods to Override**:
    *   `get_cache_size_per_token` / `get_cache_bytes_per_token`: Update `kv_factor` correctly to include the size of the auxiliary structures so TensorRT LLM allocates sufficient GPU memory.
    *   `add_dummy_requests` / `prepare_resources`: Ensure the auxiliary pool allocates correct resources/tokens for new requests.
*   **Pros**: The custom cache manager is more flexible and easier to implement because it can share the same blocks managed by the `KVCacheManager`.
*   **Cons**: This approach operates at the Python level, making it difficult to share features of the KV cache managed at the C++ level (e.g., advanced transmission or kvcache reuse features tied to the C++ manager).

**Option B: C++ Integrated Manager**

For tighter integration, you can manage the auxiliary memory within the C++ `KVCacheManager`.
*   **Use Case**: Algorithms like DSA use this approach to store the indexer Kcache.
*   **Pros**: Enables compatibility with advanced features such as KV cache reuse and disagg-serving. For example, DSA's low-rank indexer Kcache can be reused or transmitted between context and generation engines.
*   **Cons**: Higher implementation complexity. The current C++ `KVCacheManager` is optimized for the standard KV cache pool. Adding custom pools often requires significant modifications or manual implementation of the pool management logic within the C++ level.

**Note**: If your algorithm involves sparse KV cache, standard KV cache block reuse is generally incompatible because eviction modifies the block content uniquely for each request. However, algorithms like DSA that use low-rank approximation without eviction can support block reuse.

#### 4. Registration and Dispatch

*   Register your config and backend in `tensorrt_llm/_torch/attention_backend/sparse/utils.py` and `tensorrt_llm/_torch/pyexecutor/_util.py` to ensure the system routes the request to your new backend when the config is present.
*   Add initialization logic in `cpp/tensorrt_llm/thop/attentionOp.cpp` and `cpp/tensorrt_llm/kernels/sparseAttentionKernels.h` if new C++ level parameters are required.

## Summary and Future Work

### Current Status

Currently, the status of the sparse attention framework is as follows:

1.  **Supported Operations**: The `AttentionOp` currently supports **sparse KV cache** in the context phase and **sparse computation** in the generation phase. Other combinations (for example, sparse computation in the context phase) are not yet supported for MQA/GQA. For MLA, sparse computation is supported in both the context and generation phases.
2.  **Algorithm Support**: RocketKV is supported in both the vanilla (PyTorch) backend and the TRTLLM backend, while DSA is supported in the TRTLLM backend. These implementations validate the generality and scalability of the framework.

### Future Work

*   **Sparse Computation in Context Phase**: We plan to introduce sparse computation support for the context phase for MQA/GQA, allowing the TensorRT LLM sparse attention framework to cover more scenarios.
*   **Dynamic Eviction in Generation Phase**: Dynamically evicting KV cache blocks during the generation phase poses significant challenges to KV cache flexibility. While difficult to implement in the current framework, block-level eviction appears to be a promising compromise and is under further exploration.
*   **Unified Auxiliary Memory Management**: We are exploring a unified mechanism to manage auxiliary memory pools. This would allow users to define custom auxiliary spaces more flexibly while automatically inheriting advanced features from the KV cache, such as reuse and offloading.
*   **Code Refactoring**: As more sparse attention algorithms are integrated, the framework will undergo refactoring to unify code and improve maintainability.
*   **Optimizations**: We are discussing further optimizations, such as improving DSA performance.
