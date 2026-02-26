# Sparse Attention in TensorRT LLM

## Introduction and Motivation

As Large Language Models (LLMs) are applied to increasingly complex tasks such as long-document summarization, code generation, and autonomous agents, the demand for processing long contexts and extended generation has surged. In Transformer-based models, the attention mechanism's computational complexity and memory usage grow quadratically and linearly with sequence length, respectively. This creates significant bottlenecks in both the **Context (Prefill)** and **Generation (Decode)** phases:

- **Context Phase**: Processing long prompts requires substantial memory bandwidth and computation, affecting time-to-first-token (TTFT). Since the context phase is typically compute-bound, reducing the computational load here is critical.
- **Generation Phase**: The Key-Value (KV) cache grows with every generated token, consuming vast amounts of GPU memory and bandwidth. Since the generation phase is memory-bound, reducing the memory footprint directly alleviates memory pressure, improves token-to-token latency (TPOT), and allows for larger batch sizes.

A wide range of sparse attention methods have been proposed to address these bottlenecks. They can be broadly classified along two dimensions: **where** sparsity is applied (context phase, generation phase, or both) and **how** sparsity is realized (sparse KV cache, sparse computation, or both). Some methods compress the KV cache by evicting less important tokens after the context phase, then perform sparse computation during generation on the reduced cache. Others compute attention sparsely in the context phase itself. A complementary class of techniques implements sparsity directly inside the attention kernel by dynamically skipping computation for low-contribution KV blocks.

To bring these ideas into production, TensorRT LLM introduces a **unified sparse attention framework** that provides common abstractions—prediction hooks, sparse indices, and metadata interfaces—for different sparse attention algorithms. Built on this framework, TensorRT LLM currently supports three algorithms, each targeting a different point in the design space:

- **[RocketKV](https://arxiv.org/pdf/2502.14051)**: A training-free, two-stage method that performs KV cache eviction in the context phase and dynamic Top-K token selection in the generation phase.
- **[DeepSeek Sparse Attention (DSA)](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)**: A model-native sparse attention mechanism introduced with DeepSeek V3.2, featuring a lightweight indexer for fine-grained token-level sparse MLA computation.
- **[Skip Softmax Attention (BLASST)](https://arxiv.org/pdf/2512.12087)**: A kernel-level method that dynamically skips Softmax and BMM2 for unimportant KV blocks, requiring no framework changes or auxiliary data structures.

In the following sections, we first provide an overview of the sparse attention capabilities in TensorRT LLM, then describe the framework design that makes it possible, walk through how each algorithm is implemented on top of it, and finally present evaluation results.

## Overview of Sparse Attention in TensorRT LLM

Before diving into the framework design, this section provides a high-level overview of what sparse attention capabilities TensorRT LLM offers today.

We begin with two demo videos that compare RocketKV and DSA against full attention under long-context workloads. Both demos use `max_batch_size=64` and `samples=128`. The results show clear improvements in both max-throughput and min-latency scenarios.

https://github.com/user-attachments/assets/26ad6ba4-8254-4eb7-bf28-e40e4434d2f0
<p align="center"><sub><em>Video 1: RocketKV v.s. Full Attention on 16k/2k workloads​.</em></sub></p>

https://github.com/user-attachments/assets/eeaa4eef-e822-4f60-9aa1-1653c95c2df8
<p align="center"><sub><em>Video 2: DSA v.s. Full Attention on 128k/1k workloads​.</em></sub></p>

TensorRT LLM currently supports three sparse attention algorithms. These algorithms span two complementary levels:

- **Framework-level**: An extensible sparse attention framework—prediction hooks, sparse indices, and metadata interfaces—that drives sparse computation and KV cache behavior. RocketKV is built entirely on this framework; DSA also relies on it for index prediction and integration with the serving system.
- **Kernel-level**: Sparsity logic implemented directly inside the attention kernels. Skip Softmax Attention is a pure kernel-level method. DSA additionally introduces a token-level **sparse MLA kernel** for sparse computation on Blackwell GPUs.

The following table summarizes each algorithm:

| Algorithm    | Attention Type  | Key Idea                           |
| ------------ | --------------- | ---------------------------------- |
| **RocketKV** | MQA/MHA/GQA     | KV cache eviction + dynamic Top-K  |
| **DSA**      | MLA             | Neural indexer + sparse MLA kernel |
| **BLASST**   | MQA/MHA/GQA/MLA | Dynamic block skipping in kernel   |

Along the dimensions of phase and operation type, TensorRT LLM currently supports the following combinations:

| Kernel Type         | Context Phase Support | Generation Phase Support |
| ------------------- | --------------------- | ------------------------ |
| **MQA / MHA / GQA** | Sparse KV cache       | Sparse computation       |
| **MLA**             | Sparse computation    | Sparse computation       |

In other words:

- **MQA/MHA/GQA** evicts tokens before generation and then performs dynamic sparse attention during generation.
- **MLA** supports sparse computation in both phases, but does not compress the KV cache.

**Note**: Today, sparse attention support in TensorRT LLM is primarily targeted at NVIDIA Blackwell and newer architectures.

This blog focuses on the **framework-level** design that is common across algorithms. For kernel-level optimizations (Skip Softmax Attention, sparse MLA kernel, etc.), please refer to other dedicated documents.

## Sparse Attention Framework Design

The sparse attention framework in TensorRT LLM is designed to provide a common runway for different algorithms while hiding most of the complexity from end users. Our goal is to make it straightforward for developers to integrate new sparse attention methods through a set of unified interfaces, without modifying the core attention kernels or the serving infrastructure.

### Design Philosophy

Despite their diversity, most sparse attention algorithms follow a common pattern:

- **Predict** which tokens or blocks are important.
- **Compress** KV tokens in cache.
- **Execute** attention computation on the selected subset.

The framework abstracts this pattern into two standardized index types:

- **sparse_kv_indices**: Which tokens to **keep** in the KV cache.
- **sparse_attn_indices**: Which tokens or pages to **attend to** during computation.

The prediction step is embedded within the `AttentionBackend`. Each sparse attention algorithm implements its own prediction function inside a custom backend, and different attention layers within a single model can use different backends—meaning different layers can employ different sparse attention strategies. This gives the framework significant flexibility and extensibility. Meanwhile, the `AttentionOp` interface is enriched to support sparse attention computation in a unified way: algorithms only need to provide indices that satisfy the expected format, and the same `AttentionOp` handles the rest. This means a new algorithm only needs to implement its prediction logic and produce the right indices—the rest of the pipeline handles KV cache layout, memory management, and kernel dispatch automatically.

### Architecture Overview

At a system level, the sparse attention framework is built around three key components:

- A **prediction module** that generates sparse indices (`sparse_kv_indices` and `sparse_attn_indices`) for sparse KV cache and sparse computation.
- An **attention operator** that consumes these indices and, via a small set of pre/post kernels, turns them into concrete KV cache layouts and attention workloads.
- An **auxiliary memory subsystem** that manages extra structures such as KT caches or low-rank K caches alongside the main KV cache.

From a user perspective, all of this is controlled by a high-level `sparse_attention_config`. When such a config is provided, the system automatically selects the appropriate sparse attention backend. Compared with full attention, the key addition is the prediction module that decides *which* tokens or blocks to keep or attend to; the attention computation then runs only on that selected subset.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/media/tech_blog17_sparse_attention_framework.png" width="700">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Sparse attention framework in TensorRT LLM.</em></sub></p>

Figure 1 summarizes how these components work together along the request path. In practice, most of the complexity is encapsulated inside the `AttentionBackend`: each sparse attention algorithm is implemented as a custom backend that defines its own prediction logic, while a shared `AttentionOp` and attention kernels perform the actual sparse computation in a unified way. This design keeps the user-facing API simple while allowing `AttentionMetadata`, `AttentionBackend`, `AttentionOp`, and backend kernels to handle the heavy lifting behind the scenes.

### Prediction Module

The prediction module is the heart of the framework. Its job is to turn model inputs and internal states into sparse indices that describe which tokens or blocks matter most. It produces two main types of indices.

The first is `sparse_kv_indices`, which indicate which tokens to keep in the KV cache after the context phase. Different KV heads can have different retention patterns, providing fine-grained control over what information is preserved. After context attention finishes, a kernel called `updateSparseKvCacheAfterFmha` uses these indices to rewrite the KV cache in-place, leaving only the selected tokens. Because the gather operates in-place, the indices must be **sorted** to avoid overwriting data that has not yet been read. This approach is fully compatible with existing context attention flows—including features like chunked prefill—but it does write the KV cache twice: once in full during the context phase, and once in compressed form afterward.

The second is `sparse_attn_indices`, which indicate which parts of the KV cache to actually attend to during generation. The granularity of these indices varies by attention type: for MQA/MHA/GQA this is typically at **page** granularity, while for MLA it can be **token**-level. Before running the attention kernel, a kernel called `gatherKvPageOffsetsKernel` takes potentially unordered and fine-grained indices and maps them to ordered, page-aligned offsets, also computing the effective KV length per head. Moving this logic into a separate kernel keeps the core attention kernel relatively simple and stable, and makes it easier to evolve the selection strategy independently. The downside is an extra kernel launch per generation step and the current restriction to page-level sparsity for MQA/MHA/GQA.

The framework's generality is illustrated by how different algorithms implement their prediction:

- **RocketKV** uses attention-score-based heuristics and produces both `sparse_kv_indices` (for context-phase KV eviction) and `sparse_attn_indices` (for generation-phase Top-K selection), operating at **page** granularity.
- **DSA** uses a trained neural-network indexer and produces `sparse_attn_indices` at **token** granularity, which are consumed directly by the sparse MLA kernel without page alignment.

While token-level sparsity would be ideal for all attention types, the current design deliberately starts with page-level sparsity for MQA/MHA/GQA to minimize changes to existing kernels. Future work will explore finer-grained options as the ecosystem matures.

Because the prediction step can be computationally heavy—especially in low-latency scenarios—it is typically implemented with custom Triton or CUDA kernels rather than generic PyTorch ops.

### Attention Operator Design

From an operator perspective, sparse attention in TensorRT LLM is realized inside a common attention operator. The prediction module produces indices, but it is `AttentionOp` that turns them into concrete KV cache layouts and attention workloads. Figure 2 shows how these pieces work together.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/media/tech_blog17_sparse_attention_op.png" width="700">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Sparse attention operator workflow in TensorRT LLM.</em></sub></p>

The `AttentionOp` supports two main categories of sparse behavior: **sparse computation** and **sparse KV cache**.

**Sparse computation.** Sparse attention computation can operate at two levels of granularity: coarse-grained (page-level) and fine-grained (token-level). Coarse-grained selection tends to be more forgiving in terms of accuracy, since keeping an entire page when any token in it is important avoids missing relevant context. Fine-grained token-level control, on the other hand, can yield greater speedups by eliminating more irrelevant data. Currently in TensorRT LLM, MQA/GQA/MHA supports only coarse-grained (page-level) sparse attention, while MLA supports only fine-grained (token-level) computation.

For fine-grained sparse MLA, the attention kernel is modified to directly support token-level sparse computation. The framework supplies `sparse_attn_indices` that specify, for each query token, exactly which KV tokens to attend to. Notably, the current sparse MLA implementation expects **global** KV cache pool addresses with token-level offsets, rather than logical KV positions within a request.

For coarse-grained sparse MQA/GQA/MHA, the framework employs `gatherKvPageOffsetsKernel` before each generation step. This kernel converts `sparse_attn_indices` into page-aligned KV cache offsets and updates the effective KV length per KV head. The subsequent attention kernel then runs a standard dense computation on this reduced set of pages, achieving sparse attention without deeply modifying the core attention logic.

**Sparse KV cache.** TensorRT LLM supports token-level sparse KV cache during the context phase. After context attention finishes, a kernel called `updateSparseKvCacheAfterFmha` rewrites the KV cache in-place according to `sparse_kv_indices`, keeping only the selected tokens. This effectively compresses the KV cache while remaining fully compatible with features like chunked prefill, since the context attention itself runs in the standard dense manner and compression is applied as a post-processing step.

It is worth noting that the **sparse MLA** path bypasses `updateSparseKvCacheAfterFmha` and `gatherKvPageOffsetsKernel` entirely. Because the sparse MLA kernel natively supports token-level sparsity, it only requires the framework to supply correct sparse indices—no intermediate page-alignment or KV cache rewriting is needed.

This separation of responsibilities—prediction module → `AttentionOp` pre/post kernels → core attention—provides a clean layering: algorithms can iterate on prediction and indexing strategies while relying on a stable, high-performance attention kernel underneath.

### Auxiliary Memory Management

Most practical sparse attention algorithms need **auxiliary memory** beyond the main KV cache. Examples include:

- A compressed **KT cache** for RocketKV, used to score token importance.
- A low-rank **K cache** for DSA, used to approximate attention over long histories.

TensorRT LLM currently provides two KV cache manager implementations: `KVCacheManagerV1`, which is primarily implemented in C++, and `KVCacheManagerV2`, which is primarily implemented in Python. We recommend that developers prioritize `KVCacheManagerV2` for new integrations, as it offers several advantages for managing complex memory pool configurations.

**KVCacheManagerV2** is designed around a flexible, hierarchical storage model. Its key strength is the ability to support **heterogeneous memory pools across layers**: different layers can have pools of different types and sizes, which is essential for sparse attention algorithms that attach auxiliary buffers (such as KT caches or indexer K caches) alongside standard KV data. Under the hood, KVCacheManagerV2 groups layers by their *lifecycle* (eviction strategy and buffer configuration) and automatically coalesces buffers of the same size within each group. Layers with identical lifecycle configurations share the same pool group, and slots within a pool group are allocated and freed in lockstep across all constituent pools. This automatic coalescing minimizes memory fragmentation even in complex multi-pool scenarios—for instance, when a model has both full-attention layers and sliding-window layers, or when auxiliary caches have different sizes from the main KV buffers. The trade-off is a modest amount of management overhead compared with a monolithic, fixed-layout allocator, but this cost is negligible in practice and well justified by the gains in flexibility and reduced fragmentation. KVCacheManagerV2 also provides a clean Python API for defining custom `AttentionLayerConfig` and `BufferConfig` per layer, making it straightforward to extend the memory layout for new sparse attention algorithms without touching the C++ runtime. We have new sparse attention which is in developing is using KVCacheManagerV2 to manage the required buffers.

For `KVCacheManagerV1`, which is the manager currently used by the existing sparse attention algorithms, TensorRT LLM supports two approaches for managing auxiliary memory.

The first approach is **Python-level cache managers**. These are lightweight managers implemented in Python, often inheriting from `KVCacheManager`. They are easy to prototype and iterate on, and can reuse `BlockManager` to track blocks and share some logic with the main KV cache. However, because they live above the C++ runtime, they cannot automatically benefit from advanced features like KV cache reuse or disaggregated serving. Memory sizing and resource preparation must be handled carefully at the Python level. RocketKV's `RocketKVCacheManager` for the KT cache is an example of this approach.

The second approach is **C++ integrated managers shared with `KVCacheManagerCpp`**. Here, auxiliary memory is integrated directly into the C++ KV cache manager, gaining access to the full set of KV cache features including reuse and transmission between engines. This path is well suited for production, long-lived deployments, but is significantly more complex to implement—there is currently no generic plugin-style interface for custom pools, so each algorithm needs its own integration. DSA's indexer K cache follows this approach.

As a rule of thumb, we recommend starting with Python-level managers when experimenting with new ideas, and moving to a C++-integrated design once the algorithm is stable and you need advanced features like KV cache reuse at scale.

## Algorithm Implementations

In this section, we walk through the three sparse attention algorithms currently implemented in TensorRT LLM, focusing on how each algorithm works and how it integrates with the framework. Each algorithm demonstrates how the framework's abstractions are used (or bypassed) in practice, validating the generality of the design. For a quick-start guide and configuration details, please refer to the [Sparse Attention documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md).

### RocketKV

#### Overview

RocketKV is a training-free, two-stage sparse attention method that reduces the KV cache bottleneck in long-context LLM inference. In the **context phase**, it performs permanent KV cache eviction, selecting and retaining only a `prompt_budget` of the most important tokens based on attention scores. In the **generation phase**, it maintains a lightweight, compressed auxiliary cache (KT Cache) to dynamically predict which tokens are most relevant for each new query, loading only those tokens for attention computation.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/media/tech_blog17_rocketkv_overview.png" width="700">
</figure>
</div>
<p align="center"><sub><em>Figure 3: RocketKV Overview</em></sub></p>

For more technical details, please refer to the paper: [RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](https://arxiv.org/pdf/2502.14051). An official implementation is available as a reference: [RocketKV Repo](https://github.com/NVlabs/RocketKV).

#### How It Works in TensorRT LLM

Within TensorRT LLM, RocketKV is integrated as a specialized sparse attention backend, accessible via the standard LLM API.

Directly adapting the original research implementation to a production inference engine required addressing several practical limitations. The original code only supported a batch size of one, stored the auxiliary KT Cache as a single monolithic tensor (severely limiting system throughput), and relied on standard PyTorch operations for prediction steps (introducing significant latency). To address these issues, we implemented the following optimizations.

**Prediction module.** We provide two backend implementations—`RocketVanillaAttention` and `RocketTrtllmAttention`—which inherit from `VanillaAttention` and `TrtllmAttention`, respectively. At runtime, the system dispatches to the appropriate backend based on the user-specified `attention_backend` and `sparse_config`. Within each backend, the core work centers on implementing `sparse_kv_predict` and `sparse_attn_predict`, which produce `sparse_kv_indices` and `sparse_attn_indices` as described earlier.

The native PyTorch prediction logic has very high overhead and many practical limitations. We replaced the critical operations with **custom Triton kernels**, enabling support for batch sizes greater than one while substantially reducing prediction latency. We invested significant effort in optimizing kernels for Top-K selection and batched matrix multiply (BMM) to ensure low end-to-end latency. While the current implementation is highly optimized compared with the Python baseline, there is still room for improvement—for example, via additional operator fusion and further kernel-level tuning.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/media/tech_blog17_rocketkv_prediction.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 4: RocketKV Prediction Overview</em></sub></p>

Figure 4 illustrates the prediction implementation within TensorRT LLM. To support the prediction module, we define corresponding metadata classes—`RocketVanillaAttentionMetadata` and `RocketTrtllmAttentionMetadata`—that pre-allocate all necessary buffers. This is one key advantage of encapsulating sparse attention within the `AttentionBackend` layer: by leveraging the metadata infrastructure, many CPU-side preparation steps can be computed in advance during the `prepare` phase, enabling better overlap between CPU and GPU work.

**Attention operator.** Once prediction produces `sparse_kv_indices` and `sparse_attn_indices`, they are passed to `AttentionOp`. Since RocketKV typically operates with GQA attention, it fits naturally into the framework's sparse computation path. In the context phase, `updateSparseKvCacheAfterFmha` post-processes the KV cache to retain only the budgeted tokens per KV head. In the generation phase, `gatherKvPageOffsetsKernel` selects the relevant pages based on the sparse indices, and the attention kernel then computes over this reduced set.

**Auxiliary memory management.** Managing the paged KT cache presented another challenge. `RocketKVCacheManager` inherits from `KVCacheManagerV1` and extends it with a dedicated `BlockManager` for the auxiliary KT cache at the Python level. The main KV cache and the KT cache share block IDs for each request, so that the lifecycle of KT cache blocks is automatically tied to the corresponding KV cache blocks. The `BlockManager` handles slot allocation and deallocation for the KT cache independently, while `RocketKVCacheManager` overrides methods such as `get_cache_bytes_per_token` and `prepare_resources` to ensure that memory sizing accounts for the extra KT cache footprint and that the correct KT cache pointers are passed to prediction kernels at each step. This design keeps the integration lightweight and easy to iterate on, though it inherits the limitations of Python-level management—namely, no automatic support for KV cache reuse or disaggregated serving. 

The concrete implementation can be found in `tensorrt_llm/_torch/attention_backend/sparse/rocket.py`.

### DeepSeek Sparse Attention (DSA)

#### Overview

DeepSeek Sparse Attention (DSA) is a model-native sparse attention mechanism introduced with [DeepSeek V3.2](https://api-docs.deepseek.com/news/news251201). Unlike RocketKV, which is a training-free technique applicable to standard attention architectures, DSA is an architectural modification that uses a learned indexer for fine-grained token-level sparse MLA computation.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/media/tech_blog15_dsa_architecture.png" alt="DSA Architecture" width="700">
</figure>
</div>
<p align="center"><sub><em>Figure 5: The architecture of DSA.</em></sub></p>

At a high level, DSA works as follows:

1. A **lightweight indexer** determines the importance of all key-value entries for each query token using low-rank projections and an MQA-style scoring mechanism.
2. A **Top-K selector** retains only the top-*k* entries (typically k=2048) based on the index scores.
3. **Sparse MLA** computes attention exclusively between the query token and these selected entries.

#### How It Works in TensorRT LLM

DSA integrates into TensorRT LLM through the same `sparse_attn_predict` interface used by RocketKV, validating the framework's generality beyond the use case it was originally designed for. The key differences are that DSA's prediction module is a **neural network** (the indexer) rather than a heuristic scoring function, and it produces **token-level** indices rather than page-level ones.

The indexer consists of two low-rank linear projections (for Q and K), a LayerNorm, RoPE, and a weight projection layer. Given query token $c_t^Q$ and hidden state $h_t$, it computes index scores via an MQA-style dot product:

$$I_{t} = \sum_{j=1}^{h}W_j^I \cdot \text{ReLU}(Q_{t, j}^I (K_t^I)^T)$$

A Top-K operation selects the most relevant indices, producing a `topk_indices` tensor of shape `[num_tokens, topk]` containing **request-local** token positions.

**Prediction module.** Currently, DSA only supports the `TrtllmAttention` backend, implemented as `DSATrtllmAttention`. Similar to RocketKV, the backend's primary responsibility is sparse index prediction, which consists of two stages: the **indexer module** and **index conversion**. The indexer module is determined by the model architecture itself—it computes the logical sparse indices that identify which KV tokens each query should attend to. The index conversion step is dictated by the sparse MLA kernel's requirements—it transforms these logical, request-local indices into physical KV cache addresses expressed as token-level offsets relative to the KV cache pool base.

Concretely, a Triton kernel `triton_convert_req_index_to_global_index` performs the address translation, converting logical positions within each request sequence into physical addresses in the global KV cache memory pool. Unlike RocketKV, which uses `gatherKvPageOffsetsKernel` for page alignment, DSA bypasses this step entirely—the sparse MLA kernel natively supports token-level sparsity.

As with RocketKV, a dedicated metadata class `DSATrtllmAttentionMetadata` is defined to pre-allocate and prepare the buffers needed by the prediction and conversion kernels, enabling efficient CPU–GPU overlap.

**Attention operator.** Because the sparse MLA attention kernel already supports token-level sparse computation natively, DSA requires minimal changes at the operator level. The primary task is to ensure that the inputs conform to the kernel's expectations—specifically, providing the KV cache pool base address and global token offsets.

**Auxiliary memory management.** DSA requires an auxiliary **indexer K cache** to store the low-rank K projections for reuse across decoding steps. `DSAKVCacheManager` inherits from `KVCacheManagerV1`, but unlike RocketKV's Python-level KT cache management, DSA's indexer K cache is integrated directly into the C++ `KVCacheManager`. This design enables compatibility with advanced features such as KV cache reuse, chunked prefill, and disaggregated serving—features that would be difficult to support with a Python-level manager. 

The concrete implementation can be found in `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`.

For a comprehensive description of DSA kernel optimizations, precision strategies, feature support (MTP, disaggregated serving, Wide-EP), and benchmark results, please refer to the dedicated blog post: [Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs](blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md).

### Skip Softmax Attention (BLASST)

Unlike RocketKV and DSA, **Skip Softmax Attention** is a purely **kernel-level** sparse attention method. It does not use the framework's prediction hooks or auxiliary memory—instead, it dynamically skips Softmax and BMM2 computation for low-contribution KV blocks entirely inside the attention kernel. This makes it a zero-overhead, drop-in technique that works with nearly all existing features (FP8 attention, KV cache reuse, chunked prefill) on both Hopper and Blackwell GPUs.

Since Skip Softmax Attention does not involve any framework-level components, we do not cover it in depth here. For algorithm details and implementation, please refer to:

- **Paper**: [BLASST: Dynamic Blocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/pdf/2512.12087)
- **Tech blog**: [Accelerating Long-Context Inference with Skip Softmax Attention](blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md)
- **Feature documentation**: [Sparse Attention — Kernel-Level Sparse Attention](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md#kernel-level-sparse-attention)

## Evaluation

This section consolidates accuracy and performance results for the sparse attention algorithms supported in TensorRT LLM.

### RocketKV

Unless otherwise specified, the experiments below use the following default settings: `budget=2048`, `window_size=32`, `kt_page_size=4`, `kt_cache_dtype=fp8`, and `topk=64`.

#### Accuracy

We evaluate accuracy on several models using the LongBenchV1 dataset:

| Model                    | RocketKV | Full Attention |
| ------------------------ | -------- | -------------- |
| Llama3.1-8B-Instruct     | 48.15    | 48.70          |
| Llama3.1-70B-Instruct    | 51.27    | 51.90          |
| Mistral-7B-Instruct v0.3 | 48.12    | 49.91          |
| Qwen3-8B                 | 36.28    | 37.31          |

Compared with the full attention baseline, RocketKV incurs an accuracy drop of around 1.5%, which we consider acceptable given the latency and memory savings it enables.

#### Performance

We benchmark RocketKV against the full attention baseline on three models: Llama3.1-8B-Instruct, Llama3.1-70B-Instruct, and Qwen3-8B. We use two representative long-context workloads: **8k prompt / 1k generation** (left) and **32k prompt / 4k generation** (right). All experiments are run on a B200 system; Llama3.1-8B-Instruct and Qwen3-8B use a single GPU, while Llama3.1-70B-Instruct runs with tensor parallelism (TP=4).

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/media/tech_blog17_rocketkv_performance.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 6: RocketKV vs. full attention — performance overview</em></sub></p>

Figure 6 shows the throughput–latency Pareto curves for RocketKV (orange) and full attention (red) across all three models and both workloads. Each point on a curve corresponds to a different batch size; curves further to the upper-right indicate better throughput at equivalent latency. As the workload grows from 8k/1k to 32k/4k, the gap between RocketKV and full attention widens across all three models, reflecting the increasing benefit of reducing the effective KV footprint at longer contexts. In heavily batched, throughput-oriented settings, RocketKV consistently delivers higher throughput; however, in low-latency configurations with small batch sizes, the additional prediction stage can dominate, leading to slightly higher per-request latency. We summarize the results using two metrics: **tokens/s per GPU (tps/gpu)** for max-throughput scenarios and **tokens/s per user (tps/user)** for min-latency, small-batch scenarios:

| Model                        | Workload | Max Throughput Speedup (tps/gpu) | Min Latency Speedup (tps/user) |
| ---------------------------- | -------- | -------------------------------- | ------------------------------ |
| Llama3.1-8B-Instruct         | 8k/1k    | 1.40×                            | 1.80×                          |
| Llama3.1-8B-Instruct         | 32k/4k   | 2.26×                            | 3.40×                          |
| Qwen3-8B                     | 8k/1k    | 1.53×                            | 1.75×                          |
| Qwen3-8B                     | 32k/4k   | 2.51×                            | 2.78×                          |
| Llama3.1-70B-Instruct (TP=4) | 8k/1k    | 1.05×                            | 1.53×                          |
| Llama3.1-70B-Instruct (TP=4) | 32k/4k   | 1.21×                            | 1.83×                          |

Overall, RocketKV tends to be most beneficial in **max-throughput** scenarios, and the benefit becomes larger on heavier long-context workloads. In **min-latency** scenarios, the speedup can be smaller because attention accounts for a smaller fraction of the end-to-end step time, while RocketKV adds an extra prediction stage.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sprase_attention_tech_blog/docs/source/blogs/media/tech_blog17_rocketkv_breakdown.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 7: Attention Breakdowns</em></sub></p>

To better understand where the speedup comes from, Figure 7 breaks down the per-step and prediction time under different request patterns:

- **Context phase**: RocketKV shows limited benefit here. Attention is still computed densely during prefill, and the post-processing step to compress the KV cache introduces extra overhead.
- **Generation phase**: The speedup mainly comes from reduced attention time, enabled by (1) the compressed KV cache produced after prefill and (2) dynamic Top-K selection during decode. As batch size increases, dense attention cost grows quickly, so reducing the effective KV footprint yields larger gains. By contrast, the prediction overhead does not shrink proportionally, so its relative impact becomes more visible in low-latency scenarios. In the TP=4 case, per-GPU workload is smaller, so the end-to-end speedup is more modest than one might expect from the reduction in attention work alone.

### DSA

For comprehensive DSA evaluation results—including kernel optimizations, precision strategies, and benchmarks with MTP, disaggregated serving, and Wide-EP—please refer to the dedicated blog post: [Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs](blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md).

### Skip Softmax Attention (BLASST)

For Skip Softmax Attention accuracy evaluation and performance benchmarks, please refer to: [Accelerating Long-Context Inference with Skip Softmax Attention](blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md).

## Summary and Future Work

### Current State

TensorRT LLM now provides a **unified sparse attention framework** that supports three algorithms across two complementary levels:

- **Framework-level**: The prediction-based workflow (`sparse_kv_predict` / `sparse_attn_predict`) drives sparse KV cache and sparse computation through a unified `AttentionOp`. RocketKV demonstrates this for MQA/MHA/GQA with page-level sparsity; DSA extends it to MLA with token-level sparsity. Both algorithms manage auxiliary memory (KT cache and indexer K cache, respectively) through different integration paths.
- **Kernel-level**: Skip Softmax Attention and DSA's sparse MLA kernel implement sparsity directly inside the attention kernels, requiring no framework-level coordination.

The framework's design ensures that new sparse attention algorithms can be integrated by implementing the prediction interface and producing standardized indices, without modifying the core attention kernels or the serving infrastructure.

### Future Work

- **Sparse Computation in Context Phase**: We plan to introduce sparse computation support for the context phase for MQA/MHA/GQA, allowing the framework to cover a broader range of scenarios.
- **Dynamic Eviction in Generation Phase**: Dynamically evicting KV cache blocks during the generation phase poses significant challenges to KV cache flexibility. Block-level eviction appears to be a promising compromise and is under further exploration.
- **Unified Auxiliary Memory Management**: We are exploring a unified mechanism to manage auxiliary memory pools. This would allow users to define custom auxiliary spaces more flexibly while automatically inheriting advanced features from the KV cache, such as reuse and offloading.
- **Fine-grained Sparsity and Feature Integration**: We are pursuing fine-grained token-level sparse computation for MQA/MHA/GQA, and deeper integration with advanced features like Disaggregated Serving and MTP.
- **Code Refactoring**: As more sparse attention algorithms are integrated, the framework will undergo refactoring to unify code and improve maintainability.
