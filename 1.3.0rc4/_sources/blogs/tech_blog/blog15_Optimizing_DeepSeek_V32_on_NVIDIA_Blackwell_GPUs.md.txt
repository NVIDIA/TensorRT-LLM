# Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs
By NVIDIA TensorRT LLM team

## Table of Contents
- [Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs](#optimizing-deepseek-v32-on-nvidia-blackwell-gpus)
    - [Table of Contents](#table-of-contents)
    - [Introduction](#introduction)
    - [DeepSeek Sparse Attention (DSA)](#deepseek-sparse-attention-dsa)
    - [Precision Strategy](#precision-strategy)
    - [Parallel Strategy](#parallel-strategy)
    - [Key Features](#key-features)
        - [MTP](#mtp)
        - [Disaggregated Serving](#disaggregated-serving)
        - [Chunked Prefill and KV Cache Reuse](#chunked-prefill-and-kv-cache-reuse)
        - [Wide Expert Parallelism (Wide-EP)](#wide-expert-parallelism-wide-ep)
		- [Chat Template and Tool Parser](#chat-template-and-tool-parser)
    - [Key Optimizations](#key-optimizations)
        - [Kernel Optimizations](#kernel-optimizations)
            - [Sparse MLA Kernel](#sparse-mla-kernel)
            - [Indexer Top-K Kernel](#indexer-top-k-kernel)
            - [DeepGEMM MQA Kernel](#deepgemm-mqa-kernel)
            - [Kernel Fusion](#kernel-fusion)
        - [System Optimizations](#system-optimizations)
            - [Multi-steams](#multi-steams)
            - [A Fast Path for Short Sequences](#a-fast-path-for-short-sequences)
    - [How to Reproduce](#how-to-reproduce)
        - [Accuracy Evaluation](#accuracy-evaluation)
        - [Benchmark on B200](#benchmark-on-b200)
            - [Min-latency](#min-latency)
            - [Max-throughput](#max-throughput)
        - [Benchmark with Wide-EP on GB200](#benchmark-with-wide-ep-on-gb200)
    - [Future Works](#future-works)
    - [Acknowledgement](#acknowledgement)

## Introduction
The open-sourced [DeepSeek-V3.2](https://api-docs.deepseek.com/news/news251201) series models proposed a new architecture with a fine-grained sparse attention mechanism, called DeepSeek Sparse Attention (DSA). It can help the DeepSeek-V3.2 model achieve better efficiency, especially in long sequence scenarios. Although DSA uses a lightweight indexer for prediction, realizing actual speedup from attention sparsity is still challenging. This blog introduces how TensorRT LLM supports key LLM inference features for DeepSeek-v3.2 and optimizes its performance on NVIDIA Blackwell GPUs.

## DeepSeek Sparse Attention (DSA)
DSA serves as a core component of the DeepSeek-v3.2 model, and it is the only architectural modification compared to its predecessors (DeepSeek-V3/R1/V3.1). It is a fine-grained sparse attention mechanism that only selects the important key-value entries for attention computation.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog15_dsa_architecture.png" alt="tech_blog15_dsa_architecture" width="700" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 1. The architecture of DSA. The green part illustrates how DSA selects the Top-K key-value entries according to the indexer.</em></sub></p>

Figure 1 illustrates the overall architecture: a lightning indexer first determines the importance of all key-value entries for each query token. Subsequently, the Top-K Selector retains only the top-$k$ entries (typically $k=2048$) based on the index scores. Finally, attention is computed exclusively between the query token and these selected entries.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog15_indexer_topk.png" alt="tech_blog15_indexer_topk" width="900" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 2. The architecture of the DSA indexer and Top-K logics.</em></sub></p>

Figure 2 illustrates the DSA indexer and the Top-K selection mechanism. Firstly, two low-rank linear layers project $c_t^Q$ and the input $h_t$ into lower-dimensional tensors. Following operations of LayerNorm to the K tensor and RoPE to both Q and K, we obtain the tensors $Q_t^I$ and $K_t^I$. Simultaneously, a separate weight projection layer processes $h_t$ to generate the weights $W_t^I$. These tensors are then used to compute the index scores (labeled as MQA Logits in Figure 2):

$$I_{t} = \sum_{j=1}^{h}W_j^I \cdot \text{ReLU}(Q_{t, j}^I (K_t^I)^T)$$

Finally, a Top-K operation is applied to the index scores to identify the most relevant indices, which are subsequently used for the sparse MLA computation. To reduce computational overhead, the K tensor $K_t^I$ is stored in the indexer K cache, allowing for reuse in subsequent iterations.
 
Regarding implementation, DSA diverges from the MLA used in DeepSeek-V3/R1/V3.1 models, which alternates between MHA mode (prefill) and MQA mode (decoding) as discussed in [Tech Blog 3](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md). Instead, our current DSA implementation operates only in MQA mode for both prefill and decoding phases to maximize kernel efficiency. We are continuing to explore further optimizations, including potential support for MHA mode in future iterations.

The DSA implementation is built upon the TensorRT LLM sparse attention framework, which is designed to provide flexible and extensible support for various sparse attention methods. For more information, please refer to the [sparse attention documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md), and a technical blog providing further details will be released soon.

## Precision Strategy
Because the DSA is the only architectural modification of DeepSeek-V3.2 from the DeepSeek-R1 model, the mixed precision recipe for other modules is the same as what is used for the DeepSeek-R1. This is the NVFP4 precision strategy used in the DSA module:
- Indexer
    - Low-rank linear layers: BF16
    - Weight projection layer: FP32, for model accuracy
    - MQA:
        - Indexer K cache: Blockwise FP8
        - Math: Blockwise FP8
    - Top-K: FP32
- QKV projection layer: BF16
- Output projection layer: NVFP4
- Sparse MLA
    - KV cache: Per-tensor FP8
    - Math: Per-tensor FP8

The MoE layers use NVFP4, which is the same as the DeepSeek-R1. Please refer to [Tech Blog 1](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md) and [Tech Blog 3](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md) for the MoE precision strategy. In addition to the NVFP4 version of DeepSeek-V3.2, TensorRT-LLM also supports the original FP8 model, as well as both BF16 and per-tensor FP8 KV caches.

We evaluated the accuracy of this NVFP4 checkpoint on the same datasets:
|                    | GSM8k | MMLU  | GPQA-Diamond |
| :----------------- | :---- | :---- | :----------- |
| [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)   | 95.91 | 87.84 | 84.34        |
| nvidia/DeepSeek-V3.2-NVFP4<sup>*</sup> | 95.26 | 87.54 | 84.85        |

<sub><em>\* Currently, the NVFP4 checkpoint has not yet been published on Hugging Face. Please stay tuned, or refer to the [How to reproduce](#how-to-reproduce) section to learn how to quantize the model to NVFP4.  
** Note there are some run-to-run variance for these evaluations. Our experiments indicate that the NVFP4 recipe delivers accuracy on par with FP8 on these datasets.</em></sub>

## Parallel Strategy
To achieve optimal throughput, DeepSeek-V3.2 adopts the same parallel strategy as DeepSeek-R1. Please refer to [Tech Blog 3](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md) for a detailed explanation of the performance benefits:
| Components         | Parallel Patterns           |
| :----------------- | :-------------------------- |
| Attention Modules  | Data Parallelism 8 (DP8)    |
| MoE Sparse Experts | Expert Parallelism 8 (EP8)  |
| MoE Shared Experts | DP8                         |
| Router GEMM        | DP8                         |
 
To scale DeepSeek-V3.2 inference on high-performance systems such as the GB200 NVL72, the model also leverages the parallel strategy from DeepSeek-R1. Please refer to [Tech Blog 4](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md), [Tech Blog 8](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.md), and [Tech Blog 14](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md) for more details.
 
The difference lies in the DSA indexer. When utilizing Tensor Parallelism (TP) for attention modules, typically in latency-oriented scenarios, TP is not applied to the indexer layers. Instead, it is applied exclusively to the MLA components (i.e., the remaining layers of the attention module).

## Key Features
In TensorRT LLM, there are many advanced features that are crucial for maximizing LLM inference performance, such as CUDA Graph, Overlap Scheduler, Speculative Decoding, etc. Given the architectural innovations in DeepSeek-V3.2, ensuring its compatibility with these features is important.

As illustrated in [Tech Blog 3](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md), both CUDA Graph and the Overlap Scheduler offer significant throughput improvements. For CUDA Graph support, which is typically enabled during decoding-only iterations where all requests are in the decoding phase, we must ensure that kernels in the DSA module support graph capture and that input/output tensor shapes remain consistent for a given batch size. Regarding the Overlap Scheduler, it is critical to eliminate any CPU-GPU synchronization within the DSA forward, as this would disrupt the execution pipeline. Other key features are discussed in the following subsections.

### MTP
Multi-Token Prediction (MTP) is a speculative decoding method used in DeepSeek series models. It verifies and accepts multiple draft tokens in a single iteration, significantly improving inference performance in both low-latency and high-throughput scenarios. The DeepSeek-V3.2 also supports MTP. For latency-critical scenarios, as detailed in [Tech Blog 1](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md), MTP-3 is recommended to maximize GPU utilization and achieve optimal performance. For other scenarios, MTP-1 typically offers performance gains as well.

However, the decoding indexer MQA kernel supports sequence lengths of only 1 or 2, limiting native support to MTP-off or MTP-1. To enable MTP > 1, we offer two solutions. The long-term solution involves updating the MQA kernel to support larger sequence lengths, which will be introduced in the MQA kernel optimization section. The immediate workaround (in [PR-9045](https://github.com/NVIDIA/TensorRT-LLM/pull/9045)) uses the existing kernel by flattening the sequence length dimension into the batch dimension, treating the input as a tensor with a sequence length of 1. While this approach ignores the causal mask during the indexer MQA forward, causing discrepancies in the diagonal regions compared to ground truth, the subsequent Top-K kernel handles causal masking correctly. Therefore, the final Top-K indices remain unaffected, allowing this workaround to support MTP-N for any N.

### Disaggregated Serving
Disaggregated serving decouples the prefill and decoding phases, allowing them to run on separate GPU pools with optimized parallel strategies. This feature is crucial for deploying LLMs on high-performance systems like GB200 NVIDIA GPU HWs. However, it requires transferring KV cache blocks from the prefill to the decoding GPUs. DeepSeek-V3.2 introduces an additional 'indexer K cache,' which presents unique challenges for cache management and transmission in a disaggregated setup.

To address this, [PR-8699](https://github.com/NVIDIA/TensorRT-LLM/pull/8699) integrated indexer K cache support into the existing kvCacheManager, enabling it to inherit existing cache features. Subsequently, [PR-8735](https://github.com/NVIDIA/TensorRT-LLM/pull/8735) extended disaggregated serving capabilities to DeepSeek-V3.2, allowing TensorRT LLM to handle the transmission of the indexer K cache. Currently, the implementation specifically targets the indexer K cache, but we plan to generalize this support in future updates.

### Chunked Prefill and KV Cache Reuse
Two additional critical features are chunked prefill and KV cache reuse. Chunked prefill removes input length constraints for long prompts and enables prefill chunks to be batched alongside more decoding requests, boosting throughput. KV cache reuse allows requests sharing common prefixes (e.g., system prompts or multi-turn conversations) to share cached blocks, drastically reducing time-to-first-token (TTFT).

On the implementation side, kvCacheManager already supports the newly introduced indexer K cache, extending compatibility to both chunked prefill and KV cache reuse. Then [PR-9376](https://github.com/NVIDIA/TensorRT-LLM/pull/9376) enabled DSA to perform prefill computation with past tokens saved in the cache, thereby unlocking chunked prefill support. Building on this, [PR-9383](https://github.com/NVIDIA/TensorRT-LLM/pull/9383) implemented KV cache reuse for DeepSeek-V3.2 by reusing the chunked prefill changes.

### Wide Expert Parallelism (Wide-EP) 
The Wide-EP is an important feature for boosting inference throughput in large-scale Mixture-of-Experts (MoE) models. For the DeepSeek-V3.2 model, after supporting the disaggregated serving, [PR-9245](https://github.com/NVIDIA/TensorRT-LLM/pull/9245) simply registered the model with the Expert Parallelism Load Balancer (EPLB). This integration allows Wide-EP and EPLB to be enabled, significantly enhancing performance.

### Chat Template and Tool Parser
DeepSeek-V3.2 introduces a new chat template compared to prior versions. This update incorporates support for tool calling and the 'thinking with tools' capability. These enhancements, along with the necessary tool parser, were implemented in [PR-9814](https://github.com/NVIDIA/TensorRT-LLM/pull/9814) and [PR-10126](https://github.com/NVIDIA/TensorRT-LLM/pull/10126). To enable this new chat template when deploying with `trtllm-serve` or `trtllm-eval`, please specify the argument `--custom_tokenizer deepseek_v32`.

## Key Optimizations
DeepSeek-V3.2 can inherit the MoE optimizations from DeepSeek-R1. Consequently, this section focuses exclusively on the DSA part, covering both kernel and system-level optimizations.

### Kernel Optimizations

#### Sparse MLA Kernel
Sparse MLA serves as the core kernel of DSA, enabling attention computation with fine-grained token sparsity. To efficiently support this sparsity pattern, we leverage the new TMALDG.Gather4 instruction on Blackwell GPUs. This instruction loads four rows from a source 2D tensor and coalesces them into a single destination tensor, making it ideal for fine-grained sparse attention operations.
 
Similar to the dense MLA kernel, FP8 KV cache optimization is crucial for reducing KV cache size and improving E2E throughput. For DSA, we employ per-tensor FP8 quantization: both Query (Q) and Key-Value (KV) tensors are quantized, and FP8 arithmetic is utilized for the sparse MLA computation. To validate the model accuracy under this configuration, the table below presents the GPQA-Diamond accuracy comparison between BF16 and per-tensor FP8 KV cache for the DeepSeek-V3.2-Exp model. PR-8692 introduced this FP8 sparse MLA support, yielding up to a 47.03% improvement in throughput (TPS/GPU).

| KV Cache Type                | FP8 checkpoint | NVFP4 checkpoint |
| :--------------------------- | :------------- | :--------------- |
| BF16 Sparse MLA and KV cache | 80.30          | 79.29            |
| FP8 Sparse MLA and KV cache  | 78.28          | 80.30            |

Another important optimization is SwapsMmaAb, designed specifically for Tensor Parallelism (TP) scenarios. When TP is enabled for sparse MLA, input tensors are partitioned along the Q head dimension. Consequently, each rank processes a reduced number of Q heads ($128 / \text{TP}$), leading to Tensor Core underutilization. SwapsMmaAb addresses this bottleneck by swapping the A and B operands during matrix multiplication to improve hardware utilization.

#### Indexer Top-K Kernel
DSA contains a module called Top-K Selector. It is a fine-grained token selection mechanism that retrieves only the key-value entries corresponding to the Top-K index scores. The index scores are from Lightning Indexer. This part will select the top 2048 tokens for each query. 

##### Deterministic Top-K vs Non-deterministic Top-K
The Top‑K problem aims to find the largest (or smallest) K elements from a set of N candidates. Because some of the N candidates may have identical values, there can be more than K elements that are tied with the K‑th element. In such cases, deciding which of the tied elements are included in the final Top‑K set affects whether the output is deterministic. If the tied elements are selected randomly, the results will be non‑deterministic. Conversely, if we always prioritize elements with smaller indices, the results will be deterministic.

Obtaining deterministic results generally requires a more complex algorithm and incurs higher latency than a non‑deterministic version. In DeepSeek V3.2, we first need to determine whether such determinism is actually necessary. We compare the accuracy between the deterministic (DE) and non‑deterministic versions of Top‑K with the GPQA-Diamond dataset. The scores are pretty close:
| GPQA-Diamond | DE Top-K | Non-DE Top-K |
| :----------- | :------ | :---------- |
| FP8 model    | 79.8    | 79.9        |
| NVFP4 model  | 80.3    | 79.4        |

So we decided to use the non‑DE parallel Top‑K algorithm for DeepSeek V3.2.

##### Radix-select-based Top-K Parallel Algorithm
<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog15_radix_select_topk.png" alt="tech_blog15_radix_select_topk" width="1280" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 3. Radix-select-based Top-K.</em></sub></p>

In general, there are two kinds of parallel Top‑K algorithms: partition‑based methods and priority‑queue‑based methods. The runtime of existing priority‑queue approaches grows rapidly as K increases, and the K value is as large as 2048 for the indexer Top-K in deepseek v3.2, so we choose partition‑based methods instead. Specifically, we adopt radix‑select as our baseline.
For 32‑bit values with 8‑bit digits, a naïve radix Top‑K algorithm runs 4 iterations, with 4 kernel launches per iteration. In each iteration, it (1) Histogram: counts how many elements fall into each digit bucket based on the current bits; (2) Prefix Sum: builds a prefix sum over these bucket counts; (3) Find target digits: identifies which bucket contains the K‑th element; and (4) Filtering: keeps all elements in smaller buckets as definite Top‑K, discards elements in larger buckets, and passes elements in the target bucket to the next iteration as new candidates.

##### Optimizations for Indexer Top-K
**Skip iterations with parallel sorting.** In addition to the basic radix‑select method, we introduce further optimizations to speed up the Top‑K computation. In practice, with either 8‑bit radix select (four iterations) or 11‑bit radix select (three iterations), the number of candidates typically drops sharply after the first one or two iterations on real datasets.
Our key optimization is to bypass the remaining radix‑select iterations and switch to a parallel sort once the candidate set becomes sufficiently small (smaller than 2048 in the current implementation). When the number of candidates is relatively small, we use a low-overhead naive O(N²) comparison-based ranking algorithm. For each element, we compare it against all others to determine its final position, and if this position is smaller than K, we keep it as part of the Top‑K output.  Otherwise, we use the parallel sort from CUB to get the results. The basic implementation and this optimization were added in [PR-8882](https://github.com/NVIDIA/TensorRT-LLM/pull/8882).

**Specialization for different cases.** When running with real datasets, we found that the number of candidates reaching the final sorting stage was larger than expected, which resulted in higher runtime overhead. To address this issue, [PR-9255](https://github.com/NVIDIA/TensorRT-LLM/pull/9255) introduced an additional preliminary bin-distribution step to reduce the number of candidates more efficiently before the final sort. This preprocessing step halves the candidate set and uses the leading 11 bits of each value to compute its bin index.  

##### Performance Results

<sub><em>Table1: Compare the performance of torch.topk and our customized Top-K op on B200.</em></sub>
| File                          | torch.topk(us) | TopKPerRow(us) | Speedup |
|-------------------------------|----------------|----------------|---------|
| topk_inputs_layer0_rank0.npy  | 106.877        | 14.069         | 7.596   |
| topk_inputs_layer0_rank1.npy  | 109.501        | 14.217         | 7.702   |
| topk_inputs_layer0_rank2.npy  | 104.616        | 14.079         | 7.431   |
| topk_inputs_layer0_rank3.npy  | 105.049        | 14.016         | 7.495   |
| topk_inputs_layer0_rank4.npy  | 105.526        | 14.073         | 7.498   |
| topk_inputs_layer0_rank5.npy  | 105.034        | 13.986         | 7.510   |
| topk_inputs_layer0_rank6.npy  | 104.516        | 14.079         | 7.423   |
| topk_inputs_layer0_rank7.npy  | 105.099        | 14.189         | 7.407   |
| topk_inputs_layer10_rank0.npy | 109.614        | 15.281         | 7.173   |
| topk_inputs_layer10_rank1.npy | 104.838        | 15.284         | 6.859   |
| Average                       | 106.067        | 14.327         | 7.410   |

We use the data that is exported from real datasets across different layers. The input tensor size for each case is [64, 9295]. We select the top 2048 from the valid candidates for each query. As shown in Table 1, compared to the native torch.topk implementation, our implementation achieves an average speedup of 7.41x. This significantly optimizes the duration of the indexer module.

Overall, by replacing the DE-version Top-K from PyTorch with our customized non-DE Top-K kernel, which brings 25%~40% and 14%~24% e2e speedup for the low latency and throughput scenarios.

#### DeepGEMM MQA Kernel
The DeepGEMM MQA kernel computes logits for the Top-K selection process. To enhance efficiency on Blackwell GPUs, several optimizations were implemented targeting both performance and ease of use:

- Larger MMA Tile Size: We increased the MMA tile size for both the prefill and decoding MQA kernels, yielding up to a 10% performance improvement. This optimization was implemented in commit [2f9d878](https://github.com/deepseek-ai/DeepGEMM/commit/2f9d87877ed691a62796c25f2e9496a5e0b7123a) and [fc97232](https://github.com/deepseek-ai/DeepGEMM/commit/fc97232c6f23bf5b4be5bdef52af8ce5dc499460).
- Flexible Paged KV Cache Configurations: The decoding MQA kernel now supports a wider range of configurations. While the initial version was restricted to a block size of 64 tokens, commit [c5d4d74](https://github.com/deepseek-ai/DeepGEMM/commit/c5d4d7448665ae90a81d9d31d60d445010da50f0) enabled support for any block size $B$ satisfying the condition $64 \% B = 0$.
- MTP-3 Support: Previously, the kernel was limited to MTP-0 or MTP-1 (predicting at most one draft token). Since MTP-3 typically delivers superior performance in low-latency scenarios, optimizations were introduced (see commit [2be3f36](https://github.com/deepseek-ai/DeepGEMM/commit/2be3f367854702e3887ff5b28b274cb16b441af9)) to enable native MTP-3 support.

#### Kernel Fusion
Kernel fusion is a standard optimization technique for improving performance. For DeepSeek-V3.2, we implemented specific fusion strategies:

- Custom Kernels for Indexer K Cache Population: The indexer MQA utilizes blockwise FP8 for both Q and K inputs, requiring the indexer K cache to store data in a specific blockwise FP8 format. During the forward pass, the indexer K tensor must be quantized, and both the values and scaling factors are saved to the cache. To optimize this, [PR-8701](https://github.com/NVIDIA/TensorRT-LLM/pull/8701) fused the blockwise FP8 quantization logic into a single kernel. Since the original PyTorch operations were a bottleneck, this resulted in a significant 32.64%–64.20% improvement in inference throughput. Subsequently, [PR-8960](https://github.com/NVIDIA/TensorRT-LLM/pull/8960) fused indexer K tensor storage operations into a custom kernel, delivering an additional 3.5%–13.4% end-to-end (E2E) performance gain.
- Fusing Small Kernels via torch.compile(): Beyond the major kernels, DSA involves numerous small kernels with low latencies. To reduce kernel launch overhead, we leverage torch.compile() to fuse these smaller operations:
    - [PR-8988](https://github.com/NVIDIA/TensorRT-LLM/pull/8988) consolidated indexer weight scaling for blockwise FP8 quantization.
    - [PR-9052](https://github.com/NVIDIA/TensorRT-LLM/pull/9052) fused LayerNorm operations, yielding around 1.42% speedup for low-latency scenarios and 1.90% for throughput-oriented scenarios.

### System Optimizations

#### Multi-steams
Multi-stream execution is leveraged in the following optimizations:

- [PR-8988](https://github.com/NVIDIA/TensorRT-LLM/pull/8988) employs multi-stream to overlap indexer weight scaling with the indexer K cache update. Combined with torch.compile() optimization for the indexer weight scaling, this yields approximately 2.53% speedup in low-latency scenarios.
- When improving the blockwise FP8 quantization in [PR-8701](https://github.com/NVIDIA/TensorRT-LLM/pull/8701), multi-stream is also used to enable concurrent quantization of the indexer Q and K tensors.
- [PR-9243](https://github.com/NVIDIA/TensorRT-LLM/pull/9243) changed the indexer weight projection GEMM to FP32 to improve accuracy. However, this introduced a performance regression compared to the low-precision implementation. To mitigate this, multi-stream is utilized to overlap the FP32 weight projection GEMM with the indexer low-rank Q projection GEMM, LayerNorm, and Q/K RoPE operations.

#### A Fast Path for Short Sequences
DeepSeek-V3.2 employs K=2048 for the Top-K selector. For sequences with length $N \le 2048$, all past KV tokens are inherently selected, rendering the MQA and Top-K operations redundant. [PR-9524](https://github.com/NVIDIA/TensorRT-LLM/pull/9524) implements a "fast path" to bypass these unnecessary operations for short sequences.

For the implementation, we can simply generate dense indices during DSA preparation, and directly change to use these dense indices in the indexer forward for prefill requests. However, decoding requests present a challenge due to CUDA Graph integration since the CUDA graph is usually enabled for decoding-only iterations. To ensure compatibility, we capture separate CUDA graphs for short and long sequences. At the start of each iteration, the system checks the sequence lengths: if any request in the batch exceeds the threshold, the long-sequence graph is triggered; otherwise, the short-sequence graph is utilized. This optimization yields approximately 1.03x speedup for 1K/1K scenarios.

## How to Reproduce
This section provides the reproducing steps for NVIDIA Blackwell B200 GPUs, for both model accuracy evaluation and performance benchmark.
 
The DeepSeek-V3.2 FP4 model is used for evaluation and benchmarking. You can follow [the command of the Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/deepseek#experimental-deepseek-v32) to quantize the original DeepSeek-V3.2 model to FP4.

### Accuracy Evaluation
Evaluate the model accuracy using trtllm-eval.

1. Prepare an advanced configuration file:
```
cat >./config.yml <<EOF
cuda_graph_config:
	enable_padding: true
	batch_sizes: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128]
enable_attention_dp: true
kv_cache_config:
	free_gpu_memory_fraction: 0.8
	dtype: fp8
moe_config:
	backend: TRTLLM
speculative_config:
	decoding_type: MTP
	num_nextn_predict_layers: 1
EOF
```
2. Evaluate accuracy on the [MMLU](https://people.eecs.berkeley.edu/~hendrycks/data.tar) dataset:
```
model_path=<your model path>
trtllm-eval --model ${model_path} \
	--tp_size 8 \
	--ep_size 8 \
	--kv_cache_free_gpu_memory_fraction 0.8 \
	--config ./config.yml \
	--custom_tokenizer deepseek_v32 \
	mmlu
```
3. Evaluate accuracy on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset:
```
trtllm-eval --model ${model_path} \
	--tp_size 8 \
	--ep_size 8 \
	--kv_cache_free_gpu_memory_fraction 0.8 \
	--config ./config.yml \
	--custom_tokenizer deepseek_v32 \
	gsm8k
```
4. Evaluate accuracy on the [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) dataset:
```
trtllm-eval --model ${model_path} \
	--tp_size 8 \
	--ep_size 8 \
	--kv_cache_free_gpu_memory_fraction 0.8 \
	--config ./config.yml \
	--custom_tokenizer deepseek_v32 \
	gpqa_diamond \
	--apply_chat_template \
	--chat_template_kwargs '{"thinking": true}' \
	--max_output_length 120000
```

### Benchmark on B200

#### Min-latency
Our benchmark results are based on Batch = 1, ISL = 8K, OSL = 1K, num_requests = 10 from a synthetic dataset.
To do the benchmark, run the following command:
```
data_path=<your dataset file following the format>
model_path=<your model path>
 
cat <<EOF > ./config.yml
cuda_graph_config:
	enable_padding: true
	batch_sizes: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128]
kv_cache_config:
	free_gpu_memory_fraction: 0.8
	dtype: fp8
moe_config:
	backend: TRTLLM
speculative_config:
	decoding_type: MTP
	num_nextn_predict_layers: 3
EOF
 
trtllm-bench -m deepseek-ai/DeepSeek-V3.2-Exp \
	--model_path ${model_path} throughput \
	--tp 4 \
	--warmup 1 \
	--dataset ${data_path} \
	--backend pytorch \
	--max_batch_size 1 \
	--max_num_tokens 8384 \
	--kv_cache_free_gpu_mem_fraction 0.8 \
	--concurrency 1 \
	--config ./config.yml \
	--num_requests 10 \
	--streaming
```
The expected results:
```
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                 	  0.2678
Total Output Throughput (tokens/sec):         	  274.1786
Total Token Throughput (tokens/sec):          	  2467.6070
Total Latency (ms):                           	  37347.9238
Average request latency (ms):                 	  3734.7334
Per User Output Throughput [w/ ctx] (tps/user):   276.2231
Per GPU Output Throughput (tps/gpu):          	  68.5446
Average time-to-first-token [TTFT] (ms):      	  425.9885
Average time-per-output-token [TPOT] (ms):    	  3.2344
Per User Output Speed (tps/user):             	  312.0708
```
<sub><em>\* Note that `max_num_tokens` is set to a large value to cover the maximum sequence length. Please refer to the [Best Performance Practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md#wip-enable-more-features-by-default) for more details on `max_num_tokens` configuration.</em></sub>

#### Max-throughput
Our benchmark results are based on Batch = 256, ISL = 8K, OSL = 1K, num_requests = 768 from a synthetic dataset. 
To do the benchmark, run the following command:
```
data_path=<your dataset file following the format>
model_path=<your model path>
 
cat <<EOF > ./config.yml
cuda_graph_config:
	enable_padding: true
    batch_sizes: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128]
enable_attention_dp: true
kv_cache_config:
	free_gpu_memory_fraction: 0.8
	dtype: fp8
moe_config:
	backend: TRTLLM
speculative_config:
	decoding_type: MTP
	num_nextn_predict_layers: 1
EOF
 
trtllm-bench -m deepseek-ai/DeepSeek-V3.2-Exp \
	--model_path ${model_path} throughput \
	--tp 8 \
	--ep 8 \
	--warmup 1 \
	--dataset ${data_path} \
	--backend pytorch \
	--max_batch_size 256 \
	--max_num_tokens 8576 \
	--kv_cache_free_gpu_mem_fraction 0.8 \
	--concurrency 256 \
	--config ./config.yml \
	--num_requests 768 \
	--streaming
```
The expected results:
```
===========================================================
= PERFORMANCE OVERVIEW 
===========================================================
Request Throughput (req/sec):                     8.4162
Total Output Throughput (tokens/sec):             8618.2158
Total Token Throughput (tokens/sec):              77563.9425
Total Latency (ms):                               365009.1921
Average request latency (ms):                     120325.7013
Per User Output Throughput [w/ ctx] (tps/user):   9.8876
Per GPU Output Throughput (tps/gpu):              1077.2770
Average time-to-first-token [TTFT] (ms):          19537.7776
Average time-per-output-token [TPOT] (ms):        98.5219
Per User Output Speed (tps/user):                 11.2591
```

### Benchmark with Wide-EP on GB200
To validate the efficacy of Wide-EP on DeepSeek-V3.2, we evaluated performance using the NVFP4 model on a GB200 NVL72 system. We compared EP16 and EP32 configurations against EP4 and EP8 baselines, with benchmarks conducted at ISL=8K and OSL=1K using the [Rate Matching](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.md#measurement-methodology) methodology.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog15_ds32_wide_ep.png" alt="tech_blog15_ds32_wide_ep" width="700" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 4. DeepSeek-V3.2 throughput on ISL/OSL 8k/1k. Note that the numbers were collected on November 20th, and more optimizations are still on-going.</em></sub></p>

As illustrated in Figure 4, Wide-EP yields up to a 2.28x improvement in per-GPU output throughput. To reproduce these results, please refer to the [examples/wide_ep/slurm_scripts](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/slurm_scripts) directory. These scripts demonstrate how to launch disaggregated serving with large-scale EP and associated features on a SLURM cluster.

## Future Works

- Optimize performance for long-sequence scenarios (e.g., ISL=32K, OSL=4K).
- Optimize performance for large Expert Parallelism (EP) configurations.
- Evaluate dense MHA versus MQA modes for context sparse MLA to determine the optimal configuration for processing short sequences.
- Explore more aggressive quantization strategies for DSA.
- Optimize the implementation of the indexer Top-K kernel.
- Investigate KV cache offloading mechanisms for DSA.

## Acknowledgement
Achieving these remarkable performance gains since the release of DeepSeek-V3.2-Exp was truly a collaborative triumph. We extend our deepest gratitude to everyone who contributed to the functional implementation and performance optimization of the DeepSeek-V3.2 model.

This work serves as a testament to TensorRT LLM's flexibility and effectiveness in supporting architectural innovations and novel sparse attention mechanisms. We hope this work paves the way for further advancements in sparse attention support.
