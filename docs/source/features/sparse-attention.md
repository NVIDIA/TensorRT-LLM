# Sparse Attention

- [Background and Motivation](#background-and-motivation)
- [Algorithm Overview](#algorithm-overview)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Configure via YAML file](#configure-via-yaml-file)
- [Sparse Attention Implementation](#sparse-attention-implementation)
  - [Framework-Level Sparse Attention](#framework-level-sparse-attention)
    - [Overview](#overview)
    - [Architecture](#architecture)
    - [Framework Implementation](#framework-implementation)
    - [Implementing a New Algorithm](#implementing-a-new-algorithm)
      - [1. Configuration Class](#1-configuration-class)
      - [2. Implement the prediction module in Attention Backend](#2-implement-the-prediction-module-in-attention-backend)
      - [3. Manage Auxiliary Memory Pool](#3-manage-auxiliary-memory-pool)
      - [4. Registration and Dispatch](#4-registration-and-dispatch)
      - [Future Work](#future-work)
  - [Kernel-Level Sparse Attention](#kernel-level-sparse-attention)
  - [Summary](#summary)

## Background and Motivation

As Large Language Models (LLMs) are applied to increasingly complex tasks such as long-document summarization, code generation, and autonomous agents, the demand for processing long contexts and extended generation has surged. In Transformer-based models, the attention mechanism's computational complexity and memory usage grow quadratically and linearly with sequence length, respectively. This creates significant bottlenecks in both the **Context (Prefill)** and **Generation (Decode)** phases:

*   **Context Phase**: Processing long prompts requires substantial memory bandwidth and computation, affecting time-to-first-token (TTFT). Since the context phase is typically compute-bound, reducing the computational load here is critical.
*   **Generation Phase**: The Key-Value (KV) cache grows with every generated token, consuming vast amounts of GPU memory and bandwidth. Since the generation phase is usually memory-bound, reducing the memory footprint directly alleviates memory pressure, improves token-to-token latency (TPOT), and allows for larger batch sizes.

Sparse attention methods aim to exploit structured sparsity in attention. Especially, exploiting the token sparsity in the sequence dimension to concentrate on the most important query-key pairs is very common. The goal of sparse attention is accelerating long-context inference, while balancing performance gains with acceptable approximation error and system complexity.

## Algorithm Overview
The design space of sparse attention is quite large, so we cannot assume there is a single implementation strategy that covers all variants. TensorRT LLM uses `sparse_attention_config` in the `LLM` API as a unified interface for **describing and enabling** different sparse attention algorithms, while allowing each technique to choose the most suitable implementation path. Each *algorithm* has its own configuration class inheriting from `BaseSparseAttentionConfig`.

TensorRT LLM currently exposes the following algorithms differentiated by `sparse_attention_config.algorithm`:

- **RocketKV** (`algorithm: rocket`, `RocketSparseAttentionConfig`, [ref](https://arxiv.org/pdf/2502.14051)): A two-stage algorithm, where the first stage performs permanent KV cache eviction and the second stage performs dynamic token selection.
- **DeepSeek Sparse Attention (DSA)** (`algorithm: dsa`, `DeepSeekSparseAttentionConfig`, [ref](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)): DeepSeek's model-native sparse attention solution, introduced in DeepSeek V3.2.
- **Skip Softmax Attention (BLASST)** (`algorithm: skip_softmax`, `SkipSoftmaxAttentionConfig`, [ref](https://arxiv.org/pdf/2512.12087)): A drop-in method that dynamically skips Softmax and BMM2 work for unimportant KV blocks, which could be fully implemented inside the attention kernels.

## Quick Start

This section shows how to enable sparse attention through the `LLM` API or YAML config. 

### Python API

To use sparse attention, configure a `BaseSparseAttentionConfig` subclass and pass it to the `LLM` constructor. Each algorithm has its own configuration class inheriting from `BaseSparseAttentionConfig`. To learn about the meaning of specific parameters, please refer to the docstring of the corresponding configuration class.

#### RocketKV

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import RocketSparseAttentionConfig, KvCacheConfig

# 1. Configure sparse attention (RocketKV)
sparse_attention_config = RocketSparseAttentionConfig(
    prompt_budget=2048,
    kt_cache_dtype='float8_e5m2'
)

# 2. Configure KV cache
# Note: some framework-based algorithms may require disabling block reuse.
kv_config = KvCacheConfig(enable_block_reuse=False)

# 3. Initialize LLM
llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=sparse_attention_config,
    kv_cache_config=kv_config,
)

# 4. Generate
prompts = ["To be or not to be..."]
outputs = llm.generate(prompts, SamplingParams(max_tokens=128))
```

#### DSA

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import DeepSeekSparseAttentionConfig

# Example: DSA configuration (exact values depend on model + use case)
sparse_attention_config = DeepSeekSparseAttentionConfig(
    index_topk=64,
)

llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=sparse_attention_config,
)
```

#### Skip Softmax Attention

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import SkipSoftmaxAttentionConfig

# One value for both phases:
sparse_attention_config = SkipSoftmaxAttentionConfig(threshold_scale_factor=1000.0)

# Or configure prefill/decode separately:
sparse_attention_config = SkipSoftmaxAttentionConfig(
    threshold_scale_factor={"prefill": 1000.0, "decode": 500.0}
)

llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=sparse_attention_config,
)
```

### Configure via YAML file
Besides Python API, you can also configure sparse attention via YAML file. This is typically more convenient in bash commands, such as `trtllm-serve` and `trtllm-eval`.

**Rocket KV**
```yaml
sparse_attention_config:
  algorithm: rocket
  kt_cache_dtype: float8_e5m2
  prompt_budget: 2048
kv_cache_config:
  enable_block_reuse: false
enable_chunked_prefill: false
```

**DSA**
```yaml
sparse_attention_config:
  algorithm: dsa
  index_topk: 64
```

**Skip Softmax Attention**
```yaml
attn_backend: TRTLLM
sparse_attention_config:
  algorithm: skip_softmax
  threshold_scale_factor:
    prefill: 1000.0
    decode: 500.0
```

Run the command with the config file:
```bash
trtllm-bench/trtllm-serve --model <model_path> --config extra_config.yaml ...
```

For example, users can evaluate a model with trtllm-eval on LongBenchV2 task like this:

```bash
trtllm-eval --model <path_to_model> --config extra_config.yaml longbench_v2 --max_output_length 1024 ...
```

## Sparse Attention Implementation

This section provides deeper technical details on how each algorithm of sparse attention is implemented in TensorRT LLM. If you just want to enable sparse attention, see [Quick Start](#quick-start) above.

Ideologically, the current available sparse attention algorithms can be categorized into two types:

- **Framework-level sparse attention**: uses TensorRT LLM's sparse-attention framework (prediction hooks + metadata) to drive sparse computation and/or KV-cache behavior. Examples: **RocketKV**, **DSA**.
- **Kernel-level sparse attention**: implemented directly inside the attention kernels, with no extra modification on the runtime logic. Example: **Skip Softmax Attention**.

### Framework-Level Sparse Attention

Framework-level sparse attention refers to methods that use TensorRT LLM's extensible sparse-attention framework—a set of prediction hooks and metadata interfaces that drive sparse computation and/or KV-cache behavior. Currently, **RocketKV** and **DSA** are the supported framework-level sparse attention algorithms in TensorRT LLM.

#### Overview

Attention scores often exhibit strong structure and sparsity: for many queries, only a small fraction of the historical tokens meaningfully contribute to the output. To exploit this, a wide range of approximate sparse-attention methods have been proposed. These methods can introduce sparsity along different dimensions (e.g., sequence, head, hidden). TensorRT LLM’s **framework-level** support for sparse attention primarily targets approaches that leverage **token/sequence sparsity** into a GPU-friendly, structured way.

#### Architecture

This section describes the framework architecture and guides developers on how to implement new framework-level sparse attention algorithms in TensorRT LLM.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/sparse_attention_framework.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: The framework support for sparse attention in TensorRT LLM.</em></sub></p>

Our goal is to design a general, extensible, and flexible sparse attention framework. In this framework, the attention operator provides the unified APIs to support both **sparse computation** and **sparse KV cache** that leverage token sparsity, while the users/developers can only focus on the algorithm of sparse attentions, i.e. how to accurately identify important query-key pairs.

For the generality, TensorRT LLM abstracts sparse attention into a prediction-based workflow: *a prediction module first identifies the sparse indices (tokens/blocks to keep or attend to), which are then used by the subsequent attention operator*. Currently, for standard attention (MQA/GQA), TensorRT LLM supports **sparse KV cache** in the context phase and **sparse computation** in the generation phase. Different KV heads are allowed to use different sparse indices, while Q heads that map to the same KV head share the same sparse pattern. It does **not** yet support sparse computation in the context phase or sparse KV cache in the generation phase.

For the scalability, Figure 1 illustrates the overall design. The architecture is built by inheriting from the existing `AttentionBackend` to define algorithm-specific sparse attention backends. Within these backends, `prediction` methods are implemented to generate the corresponding sparse indices. These indices are then passed as arguments to the `AttentionOp` to perform the sparse attention computation. This approach balances system flexibility with extensibility, allowing new algorithms to be integrated by simply defining their prediction logic **without** modifying the core attention kernels.

TensorRT LLM currently supports the following features in the framework:

1.  **Context Phase**:
    *   **sparse computation**: MLA
    *   **sparse KV cache**: MQA/GQA

2.  **Generation Phase**:
    *   **sparse computation**: MLA/MQA/GQA
    *   **sparse KV cache**: no support yet

#### Framework Implementation

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

To support those features, as illustrated in Figure 2, we have implemented two kernels for the MQA/GQA path, `updateSparseKvCacheAfterFmha` and `gatherKvPageOffsetsKernel`, applied in the context and generation phases respectively:

*   **`updateSparseKvCacheAfterFmha`**: Invoked in the post-processing stage after the context attention computation. It selects the important KV tokens and write those K/V vectors to the KV cache to reduce the KV cache size.

*   **`gatherKvPageOffsetsKernel`**: Executed before the attention computation in the generation phase. It converts the input sparse indices (which can be of arbitrary granularity) into page-aligned indices. This means that if a single token is selected, the entire page is included in the attention computation. After this conversion, we will get a new `kv_page_offsets` and also an updated `kv_len` that is the number of those selected KV tokens. Then these new metadata are fed into the subsequent attention kernel for computation.

For sparse MLA, the kernel supports token sparsity directly, eliminating the need for `gatherKvPageOffsetsKernel`. However, please note that sparse KV cache support is not yet available.

Many sparse attention algorithms also require additional auxiliary memory. In the current system, there are two paths to support this feature:

*   Implement a simple, custom CacheManager at the Python level, inheriting from `KVCacheManager`.

*   Use `KVCacheManagerCpp` to simultaneously manage both the KV Cache and auxiliary memory.

Each option has its own advantages and disadvantages, please refer to the [Manage Auxiliary Memory Pool](#3-manage-auxiliary-memory-pool) for more details.

#### Implementing a New Algorithm Inside the Sparse Attention Framework

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

#### Future Work

*   **Sparse Computation in Context Phase**: We plan to introduce sparse computation support for the context phase for MQA/GQA, allowing the framework to cover more scenarios.
*   **Dynamic Eviction in Generation Phase**: Dynamically evicting KV cache blocks during the generation phase poses significant challenges to KV cache flexibility. Block-level eviction appears to be a promising compromise and is under exploration.
*   **Unified Auxiliary Memory Management**: We are exploring a unified mechanism to manage auxiliary memory pools, allowing custom auxiliary spaces to automatically inherit advanced features from the KV cache (e.g., reuse, offloading).
*   **Code Refactoring**: As more sparse attention algorithms are integrated, the framework will undergo refactoring to unify code and improve maintainability.

### Kernel-Level Sparse Attention

Unlike framework-level methods, **kernel-level sparse attention** is implemented directly inside the attention kernels. There is no external prediction/gather workflow—the kernel itself decides what to skip based on runtime criteria.

**Skip Softmax Attention (BLASST)** is TensorRT LLM's kernel-level sparse attention method, supported on both **Hopper** and **Blackwell** GPUs for MHA/GQA/MLA, in both prefill and decode phases. It dynamically skips Softmax and BMM2 computation for KV blocks whose contribution falls below a threshold. Because the logic lives entirely inside the kernel, it requires no auxiliary data structures or framework hooks—just set `threshold_scale_factor` in the config. As a result, the runtime overhead is zero and the attention kernel performance improvement could be directly reflected in the end-to-end speedup.

For algorithm details and end-to-end results, please refer to the following resources:
- **Paper**: [BLASST: Dynamic Blocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/pdf/2512.12087)
- **NVIDIA developer blog**: [Accelerating Long-Context Inference with Skip Softmax Attention](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)
- **Tech blog**: [Accelerating Long-Context Inference with Skip Softmax Attention](../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md)

Skip Softmax Attention is supported only with the **trtllm** attention backend, implemented inside TensorRT-LLM's high-performance attention kernels:
- **Hopper prefill**: [fmha_v2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/fmha_v2)
- **Hopper decode**: [XQA](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/xqa)
- **Blackwell**: [trtllm-gen](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/trtllmGenKernels)


### Summary

The following table compares the three sparse attention algorithms available in TensorRT LLM:

| Aspect | RocketKV | DSA | Skip Softmax |
|--------|----------|-----|--------------|
| Prefill Acceleration | No | Yes | Yes |
| Decode Acceleration | Yes | Yes | Yes |
| KV Cache Reduction | Yes | No | No |
| Framework-Level Support Required | Yes | Yes | No |
| Model Native | No | Yes | No |
