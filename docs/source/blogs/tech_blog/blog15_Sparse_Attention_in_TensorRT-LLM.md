# Sparse Attention in TensorRT LLM




## Background and Motivation

As Large Language Models (LLMs) are applied to increasingly complex tasks such as long-document summarization, code generation, and autonomous agents, the demand for processing long contexts and extended generation has surged. In Transformer-based models, the attention mechanism's computational complexity and memory usage grow quadratically and linearly with sequence length, respectively. This creates significant bottlenecks in both the **Context (Prefill)** and **Generation (Decode)** phases:

*   **Context Phase**: Processing long prompts requires substantial memory bandwidth and computation, affecting time-to-first-token (TTFT). Since the context phase is typically compute-bound, reducing the computational load here is critical.
*   **Generation Phase**: The Key-Value (KV) cache grows with every generated token, consuming vast amounts of GPU memory and bandwidth. Since the generation phase is memory-bound, reducing the memory footprint directly alleviates memory pressure, improves token-to-token latency (TPOT), and allows for larger batch sizes.

Consequently, using sparse attention to reduce overhead in both context and generation phases has attracted significant research interest. Several state-of-the-art models and techniques are evolving to minimize these overheads. Based on our research, we categorize sparse attention methods as follows:

<div align="center">
<table>
    <thead>
        <tr>
            <th colspan="2" align="center">Context</th>
            <th colspan="2" align="center">Generation</th>
            <th rowspan="2" align="center">Training-Free</th>
            <th rowspan="2" align="center">Methods</th>
        </tr>
        <tr>
            <th align="center">Sparse Computation</th>
            <th align="center">KV Cache Compression</th>
            <th align="center">Sparse Computation</th>
            <th align="center">KV Cache Compression</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">StreamingLLM</td>
        </tr>
        <tr>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">DuoAttention</td>
        </tr>
        <tr>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">H2O</td>
        </tr>
        <tr>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">Minference</td>
        </tr>
        <tr>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">Quest</td>
        </tr>
        <tr>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">XAttention</td>
        </tr>
        <tr>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">NSA,DSA</td>
        </tr>
        <tr>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">No</td>
            <td align="center">MoBA</td>
        </tr>
        <tr>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">Yes</td>
            <td align="center">No</td>
            <td align="center">Yes</td>
            <td align="center">RocketKV</td>
        </tr>
    </tbody>
</table>
</div>

The table above summarizes several representative sparse attention algorithms. DuoAttention, NSA and MoBA perform sparse computation in the context phase, but they require structural changes to the model and are therefore architecture-specific methods. For the other methods, we observe that most follow a pattern of performing KV cache compression in the context phase and sparse computation in the generation phase. Approaches such as StreamingLLM and H2O also dynamically compress (or evict) the KV cache during generation in addition to sparse computation, typically following a fixed pattern. Based on these observations, TensorRT LLM first focuses on supporting KV cache compression in the context phase and sparse computation in the generation phase, with RocketKV as the primary reference implementation. With the release of the DeepSeek V3.2 model that adopts sparse attention, we have also added support for this model. In the future, we plan to further explore and support sparse computation in the context phase and KV cache compression in the generation phase.


## Sparse Attention in TensorRT LLM

So far, we have looked at why sparse attention matters and how different research methods position themselves. In this section, we explain how TensorRT LLM brings these ideas into a production-ready system and what kinds of sparse behavior it supports today.

### What the framework provides

The sparse attention framework in TensorRT LLM is designed to provide a common runway for different algorithms while hiding most of the complexity from end users. At a high level, it offers three core capabilities:

1.  **KV cache compression in the context phase**  
    During the context phase, TensorRT LLM can selectively keep only the most important tokens in the KV cache instead of storing the entire prompt. Internally, this is implemented by a kernel called `updateSparseKvCacheAfterFmha`, which is invoked after the context attention computation. It uses `sparse_kv_indices` to gather important tokens in-place into the KV cache. Different KV heads can use different sparse patterns, giving algorithms fine-grained control over what information is preserved.

2.  **Sparse attention computation in the generation phase**  
    During decoding, TensorRT LLM can speed up attention by attending only to relevant parts of the KV cache. A dedicated kernel, `gatherKvPageOffsetsKernel`, converts `sparse_attn_indices` into page-aligned KV cache offsets and updates the effective KV length before the attention kernel runs. This mechanism also supports different selections per KV head.

3.  **Token-level sparse computation for MLA**  
    For Multi-Head Latent Attention (MLA), TensorRT LLM supports sparse computation in both the context and generation phases. Instead of working at page granularity, MLA can apply sparsity directly at the token level, which is particularly powerful for algorithms like DSA.

**Note**: Today, sparse attention support in TensorRT LLM is primarily targeted at NVIDIA Blackwell and newer architectures.

### Architecture at a glance

Conceptually, we can break down sparse attention in TensorRT LLM along two axes:

*   **Context phase** vs. **Generation phase**
*   **Sparse computation** vs. **KV cache compression**

Along these dimensions, TensorRT LLM currently supports the following combinations:

| Kernel Type | Context Phase Support | Generation Phase Support |
| :--- | :--- | :--- |
| **MQA / MHA / GQA** | KV cache compression | Sparse computation |
| **MLA** | Sparse computation | Sparse computation |

In other words:

*   **MQA/MHA/GQA** evict tokens before generation (KV cache compression in context) and then perform sparse attention during generation.
*   **MLA** supports sparse computation in both phases, but does not compress the KV cache.

At a system level, the sparse attention framework is built around three key components:

*   A **prediction module** that generates sparse indices (`sparse_kv_indices` and `sparse_attn_indices`) for KV cache compression and sparse computation.
*   An **attention operator** that consumes these indices and, via a small set of pre/post kernels, turns them into concrete KV cache layouts and attention workloads.
*   An **auxiliary memory subsystem** that manages extra structures such as KT caches or low-rank Kcaches alongside the main KV cache.

From a user perspective, all of this is controlled by a high-level `sparse_attention_config`. When such a config is provided, the system automatically selects the appropriate sparse attention backend. Compared with full attention, the key addition is the prediction module that decides *which* tokens or blocks to keep or attend to; the attention computation then runs only on that selected subset.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_tech_blog/docs/source/blogs/media/tech_blog15_sparse_attention_framework.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Sparse attention framework in TensorRT LLM.</em></sub></p>

Figure 1 summarizes how these components work together along the request path. In practice, most of the complexity is encapsulated inside the `AttentionBackend` layer: each sparse attention algorithm is implemented as a custom backend that defines its own prediction logic, while a shared `AttentionOp` and attention kernels perform the actual sparse computation in a unified way. This design keeps the user-facing API simple while allowing `AttentionMetadata`, `AttentionBackend`, `AttentionOp`, and backend kernels to handle the heavy lifting behind the scenes.

### Prediction module

The prediction module is the heart of the framework. Its job is to turn model inputs and internal states into sparse indices that describe which tokens or blocks matter most. It produces two main types of indices:

1.  **`sparse_kv_indices`**  
    *   **Role**: Indicate which tokens to keep in the KV cache after the context phase. Different KV heads can have different patterns.  
    *   **How it is used**: After context attention finishes, `updateSparseKvCacheAfterFmha` uses these indices to rewrite the KV cache in-place, leaving only the selected tokens. Because the gather is in-place, the indices must be **sorted** to avoid overwriting data.  
    *   **Trade-offs**: This approach is fully compatible with existing context attention flows (including features like chunked prefill), but it writes the KV cache twice: once densely, then once in compressed form.

2.  **`sparse_attn_indices`**  
    *   **Role**: Indicate which parts of the KV cache to actually attend to during generation. For MQA/MHA/GQA this is typically at **page** granularity; for MLA it can be **token**-level.  
    *   **How it is used**: Before running the attention kernel, `gatherKvPageOffsetsKernel` takes potentially unordered and fine-grained indices and maps them to ordered, page-aligned offsets. It also computes the effective KV length per head.  
    *   **Trade-offs**: Moving this logic into a separate kernel keeps the core attention kernel relatively simple and stable, and makes it easier to evolve the selection strategy independently. The downside is an extra kernel launch per generation step and the current restriction to page-level sparsity for MQA/MHA/GQA.

While token-level sparsity would be ideal for all attention types, the current design deliberately starts with page-level sparsity for MQA/MHA/GQA to minimize changes to existing kernels. Future work will explore finer-grained options as the ecosystem matures.

Because the prediction step can be computationally heavy—especially in low latency scenarios—it is typically implemented with custom Triton or CUDA kernels rather than generic PyTorch ops.

### Attention operator design

From an operator perspective, sparse attention in TensorRT LLM is realized inside a common attention operator. The prediction module produces indices, but it is `AttentionOp` that turns them into concrete KV cache layouts and attention workloads. Figure 2 shows how these pieces work together.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_tech_blog/docs/source/blogs/media/tech_blog15_sparse_attention_op.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Sparse attention operator workflow in TensorRT LLM.</em></sub></p>

At a high level, two kernels play a central role:

*   **`updateSparseKvCacheAfterFmha` (context phase)**: After context attention finishes, this kernel rewrites the KV cache in-place according to `sparse_kv_indices`, keeping only the selected tokens. This effectively implements KV cache compression while remaining compatible with features like chunked prefill.
*   **`gatherKvPageOffsetsKernel` (generation phase)**: Before each generation step, this kernel converts `sparse_attn_indices` into page-aligned KV cache offsets and updates the effective KV length. The subsequent attention kernel then runs a dense computation on this reduced set of pages, achieving sparse computation without deeply modifying the core attention kernel.

This separation of responsibilities—prediction module → `AttentionOp` pre/post kernels → core attention—provides a clean layering: algorithms can iterate on prediction and indexing strategies while relying on a stable, high-performance attention kernel underneath.

### Auxiliary memory management

Most practical sparse attention algorithms also need **auxiliary memory** beyond the main KV cache. Examples include:

*   A compressed **KT cache** for RocketKV, used to score token importance.
*   A low-rank **Kcache** for DSA, used to approximate attention over long histories.

TensorRT LLM supports two main ways to manage this extra memory:

1.  **Python-level cache managers**  
    *   **What it is**: A lightweight manager implemented in Python, often inheriting from `KVCacheManager`.  
    *   **Pros**: Easy to prototype and iterate on; can reuse `BlockManager` to track blocks and share some logic with the main KV cache.  
    *   **Cons**: Lives above the C++ runtime, so it cannot automatically benefit from advanced features like KV cache reuse or disaggregated serving. Memory sizing and resource preparation must be handled carefully at the Python level.
    *   **Example**: `KT cache` in RocketKV's  `RocketKVCacheManager` 

2.  **C++ integrated managers shared with `KVCacheManagerCpp`**  
    *   **What it is**: Auxiliary memory is integrated directly into the C++ KV cache manager.  
    *   **Pros**: Gains access to the full set of KV cache features, including reuse and transmission between engines. Well suited for production, long-lived deployments.  
    *   **Cons**: Significantly more complex to implement. There is currently no generic plugin-style interface for custom pools, so each algorithm needs its own integration.

As a rule of thumb, we recommend starting with Python-level managers when experimenting with new ideas, and moving to a C++-integrated design once the algorithm is stable and you need advanced features like KV cache reuse at scale.

One practical detail is that we often need to update caches *before* the attention kernel runs, at which point Keys and Values typically must already have RoPE applied. Because TensorRT LLM currently fuses RoPE into the attention kernel, we instead set `rope_fusion=False` and apply RoPE externally using FlashInfer kernels. This preserves correctness but introduces some additional overhead.

## Examples: How sparse attention is used in TensorRT LLM

In this section, we highlight two sparse attention algorithms currently implemented in TensorRT LLM: RocketKV and DSA.

### RocketKV Implementation

#### Overview

In Transformer-based LLM inference, the KV cache grows linearly with sequence length, becoming a major bottleneck. RocketKV mitigates this issue through a two-stage process:

1.  **Context Phase (Stage 1):** It performs **permanent KV cache eviction**. Instead of storing the full history, it selects and keeps a `prompt_budget` of the most important tokens based on attention scores.
2.  **Generation Phase (Stage 2):** It utilizes a **dynamic Top-K token selection**. It maintains a lightweight, compressed auxiliary cache (KT Cache) to dynamically predict which tokens of the KV cache are relevant for the current token, and loading only those tokens to do the attention computation.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_tech_blog/docs/source/blogs/media/tech_blog15_rocketkv_overview.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 3: RocketKV Overview</em></sub></p>

For more technical details, please refer to the paper: [RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](https://arxiv.org/pdf/2502.14051). An official implementation is available as a reference: [RocketKV Repo](https://github.com/NVlabs/RocketKV).

Within TensorRT LLM, RocketKV is integrated as a specialized sparse attention backend, accessible via the standard LLM API. The core sparse KV prediction kernels are implemented using optimized Triton kernels to achieve high performance on modern NVIDIA GPUs.

#### Implementation details

The RocketKV algorithm fits naturally into our sparse attention framework; in fact, we used RocketKV as the prototype when designing the framework. However, directly adapting the original research implementation to a high-performance inference engine like TensorRT LLM required addressing several limitations:

*   **Batch Size Limitation**: The original implementation was limited to a batch size of 1, which is inefficient for production environments that rely on batching for throughput.
*   **Inefficient Auxiliary Memory**: The original RocketKV stored the auxiliary "KT Cache" in a single monolithic tensor. Directly applying this to TensorRT LLM would severely impact system throughput and limit scalability.
*   **Prediction Overhead**: Reliance on standard PyTorch operations for prediction steps introduced significant latency, negating some performance gains.

To address these issues, we implemented the following optimizations:

1.  **Custom Triton kernels**: We replaced PyTorch operations with optimized Triton kernels to support batch sizes greater than one and minimize prediction overhead.
2.  **Paged KT Cache**: We implemented a paged memory management system for the KT Cache via a simplified `RocketKVCacheManager` in Python, which greatly improves system throughput.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_tech_blog/docs/source/blogs/media/tech_blog15_rocketkv_prediction.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 4: RocketKV Prediction Overview</em></sub></p>

Figure 4 illustrates the specific prediction implementation of the RocketKV algorithm within our current architecture. For the prediction module, we primarily use Triton and CUDA kernels to replace the original PyTorch-based operators.

The concrete implementation can be found in `tensorrt_llm/_torch/attention_backend/sparse/rocket.py`. To support calculations within the prediction module, we define `RocketTrtllmAttentionMetadata` to pre-allocate the necessary buffers. In `RocketTrtllmAttention`, we implement `sparse_kv_predict` and `sparse_attn_predict` to produce the indices required for the context and generation phases, respectively.

We enforce a specific shape for the output indices to optimize performance:
*   `sparse_kv_indices`: `[num_kv_heads, total_tokens]`
*   `sparse_attn_indices`: `[num_kv_heads, total_tokens]`

We place `num_kv_heads` in the leading dimension to optimize memory access patterns in subsequent kernels. Since the indices are flattened, we also maintain corresponding `sparse offsets` to track the boundaries for each sequence. These indices drive KV cache compression in the context phase and page-based sparse computation in the generation phase.

Simultaneously, the prediction module updates the KT cache using Triton kernels (found in `tensorrt_llm/_torch/attention_backend/sparse/kernel.py`). We invested significant effort in optimizing critical kernels such as Top-K and batched matrix multiply (BMM) to ensure low latency. While the current implementation is highly optimized compared with the Python baseline, there is still room for improvement, for example via additional operator fusion and further kernel-level tuning.

Beyond prediction, managing the paged KT cache presented another challenge. To prioritize ease of use and development efficiency, we implemented a simplified `RocketKVCacheManager` in Python by inheriting from `KVCacheManager`. By sharing block IDs for each request and utilizing the existing `BlockManager`, we reduced development complexity. We also overrode methods such as `get_cache_bytes_per_token` and `prepare_resources` to ensure accurate memory allocation and proper resource assignment for the paged KT cache layout.

#### Results

Unless otherwise specified, the experiments below use the following default settings for RocketKV in TensorRT LLM: `budget=2048`, `window_size=32`, `kt_page_size=4`, `kt_cache_dtype=fp8`, and `topk=64`.

**Accuracy**

We evaluate accuracy on several models, including Llama3.1-8B-Instruct, Llama3.1-70B-Instruct, Mistral-7B-Instruct v0.3, and Qwen3-8B-Instruct, using the LongBenchV1 dataset. The results are summarized below:

<div align="center">

| Model | RocketKV | Full Attention |
| :--- | :--- | :--- |
| Llama3.1-8B-Instruct | 48.15 | 48.70 |
| Llama3.1-70B-Instruct | 51.27 | 51.90 |
| Mistral-7B-Instruct v0.3 | 48.12 | 49.91 |
| Qwen3-8B | 36.28 | 37.31 |

</div>

As shown in the table, compared with the full attention baseline in TensorRT LLM, RocketKV incurs an accuracy drop of around 1.5%, which we consider acceptable given the latency and memory savings it enables.

**Performance**

We benchmark RocketKV sparse attention against the TensorRT LLM full attention baseline on three models: Llama3.1-8B-Instruct, Llama3.1-70B-Instruct, and Qwen3-8B. We use two representative long-context workloads: **8k prompt / 1k generation** (left) and **32k prompt / 4k generation** (right). Results are shown in Figure 5 (red: full attention, orange: RocketKV). All experiments are run on a B200 system; Llama3.1-8B-Instruct and Qwen3-8B use a single GPU, while Llama3.1-70B-Instruct runs with tensor parallelism (TP=4).

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_tech_blog/docs/source/blogs/media/tech_blog15_rocketkv_performance.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 5: RocketKV vs. full attention — performance overview</em></sub></p>

From Figure 5, we observe a clear pattern. As the workload grows from 8k/1k to 32k/4k, the gap between RocketKV and full attention widens across all three models, reflecting the increasing benefit of reducing the effective KV footprint at longer contexts. In heavily batched, throughput-oriented settings, RocketKV consistently delivers higher throughput; however, in low-latency configurations with small batch sizes (for example, `batch_size < 16`), the additional prediction stage can dominate, leading to slightly higher per-request latency and lower end-to-end throughput than full attention.

We summarize the results using two metrics: **tokens/s per GPU (tps/gpu)**, which captures **max-throughput** scenarios, and **tokens/s per user (tps/user)**, which captures **min-latency**, small-batch scenarios. Concretely:

- Llama3.1-8B-Instruct: On the 8k/1k workload, RocketKV achieves up to 1.4× (tps/gpu) and 1.8× (tps/user) speedup over full attention; on 32k/4k, the speedups increase to 2.26× and 3.4×, respectively.
- Qwen3-8B-Instruct: On 8k/1k, RocketKV reaches up to 1.53× (tps/gpu) and 1.75× (tps/user); on 32k/4k, the gains further improve to 2.51× and 2.78×.
- Llama3.1-70B-Instruct: On 8k/1k, RocketKV provides 1.05× (tps/gpu) and 1.53× (tps/user) speedup; on 32k/4k, the corresponding speedups rise to 1.21× and 1.83×.

Overall, RocketKV tends to be most beneficial in **max-throughput** scenarios, and the benefit becomes larger on heavier long-context workloads. In **min-latency** scenarios, the speedup can be smaller because attention accounts for a smaller fraction of the end-to-end step time, while RocketKV adds an extra prediction stage.

<div align="center">
<figure>
  <img src="https://github.com/heyuhhh/TensorRT-LLM/blob/user/yuhangh/add_sparse_attention_tech_blog/docs/source/blogs/media/tech_blog15_rocketkv_breakdown.png" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 6: Attention Breakdowns</em></sub></p>

To better understand where the speedup comes from, Figure 6 demonstrates the per-step and prediction time of RocketKV and full attention under different request patterns. Three observations stand out:

- **Context phase**: RocketKV shows limited benefit here. Attention is still computed densely during prefill, and RocketKV additionally performs a post-processing step to rewrite the KV cache into a compressed layout, which introduces extra overhead.
- **Generation phase**: The speedup mainly comes from reduced attention time, enabled by (1) the compressed KV cache produced after prefill and (2) the dynamic Top-K selection during decode. As batch size increases, dense attention cost grows quickly, so reducing the effective KV footprint yields larger gains. By contrast, the prediction overhead does not shrink proportionally, so its relative impact becomes more visible in low-latency or low-throughput scenarios. In the TP=4 case, per-GPU workload is smaller, so the attention portion per GPU does not grow as aggressively as in the single-GPU setting, while prediction overhead remains of similar magnitude. As a result, the end-to-end speedup is more modest than one might expect from the reduction in attention work alone.


#### Summary

By integrating RocketKV into TensorRT LLM, we both validate the current sparse attention framework end-to-end and demonstrate meaningful real speedups on long-context workloads. There is still substantial room for improvement—most notably further kernel fusion and optimization in the prediction path, as well as deeper integration with other features such as KV cache reuse and MTP. Going forward, we will continue optimizing RocketKV and expanding support for additional sparse-attention patterns, with the goal of making TensorRT LLM more flexible and performant across a wider range of sparse attention algorithms.


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
