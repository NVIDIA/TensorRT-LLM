# Helix Parallelism in TensorRT-LLM: Scaling Multi-Million-Token Decoding with KV Cache Sharding

By NVIDIA TensorRT LLM Team

## Table of Contents

- [Introduction](#introduction)
- [Decoding Bottlenecks at Scale](#decoding-bottlenecks-at-scale)
  - [KV Cache Streaming](#kv-cache-streaming)
  - [FFN Weight Loading](#ffn-weight-loading)
  - [Roofline Motivation](#roofline-motivation)
- [How Helix Works](#how-helix-works)
  - [High-Level Execution Flow](#high-level-execution-flow)
  - [Attention Phase: KV Parallelism + Tensor Parallelism](#attention-phase-kv-parallelism--tensor-parallelism)
  - [Exact Attention via Log-Sum-Exp Rescaling](#exact-attention-via-log-sum-exp-rescaling)
  - [Communication Volume Analysis](#communication-volume-analysis)
  - [FFN Phase: Rank Reprovisioning](#ffn-phase-rank-reprovisioning)
  - [Distributed KV Concatenation](#distributed-kv-concatenation)
- [Implementation in TensorRT-LLM](#implementation-in-tensorrt-llm)
  - [Configuration](#configuration)
  - [Model Integration](#model-integration)
  - [Request Handling for Helix](#request-handling-for-helix)
  - [KV Cache Management](#kv-cache-management)
  - [Custom Kernels](#custom-kernels)
  - [Composability](#composability)
- [When to Use Helix](#when-to-use-helix)
- [Performance Results: DeepSeek-R1 (MoE, MLA)](#performance-results-deepseek-r1-moe-mla)
- [Future Work](#future-work)
- [Acknowledgment](#acknowledgment)

## Introduction

Modern AI applications increasingly rely on models that combine large parameter counts with multi-million-token context windows. Whether it is AI agents following months of conversation, legal assistants reasoning through gigabytes of case law, or coding copilots navigating sprawling repositories, preserving long-range context is essential for relevance and coherence. At the same time, users expect fast, interactive responses.

Helix Parallelism addresses the two dominant bottlenecks in long-context decoding: KV cache streaming and FFN weight loading. It does so by using different parallelism strategies for attention and FFN within each transformer layer, allowing the same pool of GPUs to operate in the configuration best suited to each stage's bottleneck. In this blog, we focus on sequence parallelism for attention and tensor parallelism (with optional expert parallelism for MoE) for the FFN.

Compared to conventional parallelism approaches, Helix delivers up to **32x** concurrency improvement and **1.8x** interactivity improvement for 1M sequence length on DeepSeek-R1 with GB300 silicon NVL72, pushing forward the throughput-latency Pareto frontier and making real-time inference with ultra-long sequences practical.

This blog describes how Helix Parallelism is implemented in TensorRT-LLM, covering the algorithm, architecture, configuration, kernel-level details, and how to get started. For the full algorithmic description, see the original paper: [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/pdf/2507.07120).

## Decoding Bottlenecks at Scale

To support real-time decoding at scale, a system must overcome two major bottlenecks during the generation phase.

### KV Cache Streaming

When handling multi-million-token contexts, each GPU must read a massive history of past tokens (KV cache) from DRAM per sample at every decoding step. This constant streaming can saturate DRAM bandwidth, increase token-to-token latency (TTL), and quickly become a major bottleneck as context length grows.

Consider Tensor Parallelism (TP) as an example: increasing TP can distribute weight loading across more GPUs, but in attention schemes like Grouped Query Attention (GQA) or Multi-Latent Attention (MLA), multiple query heads share a limited number of KV heads K. When TP exceeds K, the system must **duplicate** the multi-million-token KV cache across GPUs so each shard can serve its assigned query heads. Beyond this point, increasing TP neither shrinks per-GPU KV cache size nor speeds up KV reads, imposing a hard ceiling on attention time and forcing smaller batch sizes under tight TTL constraints. In the case of MLA (used by DeepSeek models), K is effectively 1, so any TP > 1 triggers full KV duplication.

### FFN Weight Loading

During autoregressive decoding, generating every new token requires loading large FFN weights from DRAM. With small batch sizes typical of latency-sensitive workloads, this cost cannot be amortized, making FFN weight reads a dominant source of latency. More TP helps here by splitting weights across GPUs, but the KV cache duplication ceiling from above limits how far TP can scale - if TP is capped at K shards, only K GPUs can be used to shard the FFN, which typically accounts for the vast majority of model parameters.

These two bottlenecks are difficult to optimize simultaneously using traditional parallelism strategies, because the optimal sharding for attention (minimize KV duplication, keep TP ≤ K) conflicts with the optimal sharding for FFN (maximize weight distribution, increase TP). Helix Parallelism resolves this conflict.

### Roofline Motivation

The following roofline analysis illustrates why decoupling attention and FFN sharding is essential. We model a dense LLM with batch B=8, MLA attention, query heads Q=128, KV heads K=1, QK head size Hsz=576, V head size Hsz=512, and FFN dimension F=65536 running on GB200 NVLink72. The MLA-attention parameters mirror DeepSeek-R1: Q=128 query heads with a per-token latent KV of `kv_lora_rank=512` plus `qk_rope_head_dim=64` (giving Hsz=576) and V head size 512. DSR1's actual FFN dimension is 18432; we use F=65536 here as a representative large dense FFN so that FFN weight-read time is on the same order as long-context attention reads and the bottleneck crossover is visible in a single plot. Both weights and KV cache are stored and fetched in FP8. Communication overhead from TP and KVP is not included; these plots show only the change in GPU DRAM-read latency as TP width and KVP width vary.

<div align="center">
<figure>
  <img src="../media/tech_blog21_roofline_analysis.png" alt="Roofline analysis for KV cache and Linear weight reads" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 1. Roofline analysis for KV cache and Linear weight reads on GB200 NVLink72. (Left) DRAM read latency vs. TP width - benefits plateau beyond TP=K due to full KV duplication, highlighting the need for KV sharding in Helix. (Middle) DRAM read time vs. KV length S - self-attention cost scales linearly with S, eventually dominating latency. (Right) DRAM read time vs. KVP width - Helix applies KVP in attention to reduce per-GPU memory traffic and achieve linear scaling, enabling multi-million-token inference.</em></sub></p>

The key insights from this analysis:

- **(Left)** Increasing TP beyond K yields diminishing returns for attention because KV cache must be duplicated, while FFN weight reads continue to benefit linearly from spreading weights across more GPUs - whether via wider TP_F (dense) or a TP_F × EP grid (MoE). This divergence motivates using different parallelism widths for the two stages.
- **(Middle)** Self-attention DRAM read time scales linearly with KV sequence length S at fixed TP. At 1M+ tokens, attention reads dominate total layer latency.
- **(Right)** KV Parallelism (KVP) distributes KV cache reads across GPUs, achieving linear scaling of attention time with KVP width. The same GPUs are then re-provisioned for TP in FFNs to reduce weight-read latency.

## How Helix Works

### High-Level Execution Flow

Helix is a hybrid sharding strategy that uses different parallelism strategies for attention and FFN within a single transformer layer. Instead of using one fixed parallelism configuration for the entire layer, Helix switches between two configurations using the same pool of N GPUs:

- **Attention phase**: N = KVP × TP_A (KV Parallelism × Tensor Parallelism for Attention, where TP_A ≤ K)
- **FFN phase**: N = TP_F × EP (Tensor Parallelism for FFN × Expert Parallelism)

<div align="center">
<figure>
  <img src="../media/tech_blog21_helix_execution_flow.png" alt="Helix execution flow per transformer layer" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 2. End-to-end Helix workflow for a single transformer layer. (Top) During attention, each KVP GPU independently computes QKV projections and runs FlashAttention on its local KV shard, producing partial outputs and log-sum-exp scalars. A single All-to-All exchanges these fragments across KVP ranks; each GPU rescales and sums them into the exact softmax-normalized result. (Bottom) For FFNs, the same N GPUs are re-provisioned as either TP_F=N for dense models, or TP_F × EP for MoE models. Adapted from <a href="https://arxiv.org/pdf/2507.07120">Bhatia et al., 2025</a>.</em></sub></p>

By decoupling the parallelism strategy per stage, each operates in the configuration tuned to its own bottleneck, all while reusing the same GPUs with zero idle time as computation flows through the model.

### Attention Phase: KV Parallelism + Tensor Parallelism

Helix applies KV Parallelism (KVP) by sharding the KV cache along the sequence dimension across KVP GPUs, while applying Tensor Parallelism across attention heads (TP_A). TP_A is kept ≤ K (number of KV heads) to avoid duplication.

<div align="center">
<figure>
  <img src="../media/tech_blog21_attention_sharding.png" alt="Comparison of attention sharding strategies" width="900">
</figure>
</div>
<p align="center"><sub><em>Figure 3. Attention sharding strategies for GQA with Q=4 query heads and K=2 KV heads. (a) No TP: all heads co-located, no duplication. (b) TP=2: clean split since TP ≤ K. (c) TP=4: more shards than KV heads, forcing KV cache duplication. (d) Helix (TP=2, KVP=2): avoids duplication by forming a 2D layout - TP splits heads, KVP splits the sequence dimension. Adapted from <a href="https://arxiv.org/pdf/2507.07120">Bhatia et al., 2025</a>.</em></sub></p>

This results in N = KVP × TP_A GPUs collaborating on the attention computation **without duplicating KV cache** across GPUs.

To avoid an expensive pre-attention All-Gather of queries across KVP GPUs, Helix has each KVP GPU independently compute the full QKV projections. Every GPU takes the full input batch [B, H] and multiplies by its local Q, K, V weight matrices, producing the query, key, and value tensors needed for its assigned heads. This redundant projection enables fully local FlashAttention on each KV shard in isolation - no cross-GPU communication is needed until attention is complete.

### Exact Attention via Log-Sum-Exp Rescaling

After local FlashAttention, each KVP GPU holds a partial attention output computed over only its KV shard, along with the corresponding log-sum-exp (LSE) scalar per query token. A single All-to-All along the query-head dimension across KVP GPUs exchanges these partial outputs and LSE values.

Each GPU then rescales and sums the received fragments to reconstruct the exact softmax-normalized attention - not an approximation. The correction works because:

1. Each KVP rank i computes local attention output O_i = softmax(Q · K_i^T) · V_i and the corresponding log-sum-exp LSE_i. Internally, each rank already runs Flash-Decoding across its own SMs, splitting its KV shard into chunks and combining their per-chunk partials and LSEs into the rank-local (O_i, LSE_i).
2. After the All-to-All, each GPU has all partial outputs {O_i} and scalars {LSE_i}
3. The global result is reconstructed as: O = Σ_i (exp(LSE_i - LSE_global) · O_i), where LSE_global = log(Σ_i exp(LSE_i))

In effect, Helix is a hierarchical Flash-Decoding scheme: the [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) online-softmax correction is applied first within each GPU across its SMs, and then a second time across KVP GPUs over the All-to-All. The result is mathematically identical to single-GPU attention.

### Communication Volume Analysis

A critical property of Helix's All-to-All is that its communication volume is **independent of the KV sequence length S**. The data exchanged consists of:

- Partial attention outputs: B × (H / KVP) per rank (half-precision)
- Log-sum-exp scalars: B × (num_heads / KVP) × 2 per rank (float32)

This means that even as S scales into multi-millions of tokens, the All-to-All cost remains constant - it depends only on batch size B and hidden dimension H. This property makes Helix highly scalable for ultra-long contexts, unlike approaches where communication grows with sequence length.

### FFN Phase: Rank Reprovisioning

After the attention All-to-All and post-attention linear projection (run in TP=N), the same pool of N GPUs is reprovisioned without idle time for the FFN block. Two modes are supported depending on the model type:

- **Dense FFNs** (EP=1): All N GPUs collaborate in tensor-parallel fashion (TP_F = N), each holding a shard of size H×(F/N), to amortize weight-read costs across all available devices.
- **MoE FFNs** (EP > 1): The N GPUs are repartitioned into a TP_F × EP grid. Tokens are routed to appropriate experts, TP is applied within each expert group, and intra-expert All-Reduce + inter-expert All-Gather produce the final output.

In TensorRT-LLM, this reprovisioning is implemented through the `repurpose_helix_cp_to_tp()` method in the `Mapping` class. The model holds two mappings:
- `mapping_with_cp`: the original mapping with CP ranks (used by attention layers)
- A repurposed mapping where `tp_size = tp_size × cp_size` and `cp_size = 1` (used by FFN/MoE layers)

### Distributed KV Concatenation

During decoding, each newly generated token is broadcast to all KVP GPUs for query computation. However, Helix staggers the KV cache updates across KVP ranks in round-robin fashion to prevent DRAM hot spots: tokens 1-32 go to KVP rank 0, tokens 33-64 to KVP rank 1, and so on, cycling through all KVP ranks. This staged concatenation guarantees uniform KV growth and balanced memory usage regardless of batch size or sequence length.

TensorRT-LLM's `partition_context_for_helix()` function implements this partitioning: given the full input token sequence, it assigns blocks of tokens to CP ranks based on `tokens_per_block`, with position IDs preserved for correct rotary embeddings.

## Implementation in TensorRT-LLM

### Configuration

Helix is exposed through TensorRT-LLM's context parallelism framework. The relevant configuration parameters are:

| Parameter | Description | Required |
|-----------|-------------|----------|
| `context_parallel_size` | Number of GPUs for KV parallelism (≥2 for Helix) | Yes |
| `cp_config.cp_type` | Must be `"HELIX"` or `CpType.HELIX` | Yes |
| `cp_config.tokens_per_block` | Tokens per KV cache block for Helix partitioning | Yes |
| `cp_config.use_nccl_for_alltoall` | Use NCCL for the All-to-All (default: `True`) | No |
| `cp_config.fifo_version` | FIFO version for All-to-All communication (default: `2`) | No |
| `kv_cache_config.tokens_per_block` | Must match `cp_config.tokens_per_block` | Yes |

Example YAML configuration for a generation server in disaggregated mode:

```yaml
context_parallel_size: 2
cp_config:
  cp_type: "HELIX"
  tokens_per_block: 32
  use_nccl_for_alltoall: false
  fifo_version: 2
kv_cache_config:
  tokens_per_block: 32
```

A validation rule in [`TorchLlmArgs`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm_args.py) (where `CpConfig` is also defined) ensures that `cp_config.tokens_per_block` matches `kv_cache_config.tokens_per_block` when Helix is active.

### Model Integration

Helix in TensorRT-LLM supports all three major attention variants on Blackwell GPU architecture: Multi-Latent Attention (MLA), Grouped-Query Attention (GQA), and Multi-Head Attention (MHA).

The integration in [`DeepseekV3ForCausalLM`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/modeling_deepseekv3.py) demonstrates the rank reprovisioning pattern, driven by helpers in [`tensorrt_llm/mapping.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/mapping.py):

1. During model initialization, when `has_cp_helix()` is true, a deep copy of the original mapping (with CP) is saved as `mapping_with_cp`.
2. The model config's mapping is repurposed via `repurpose_helix_cp_to_tp()`, which converts CP ranks to additional TP ranks (setting `tp_size = tp_size × cp_size`, `cp_size = 1`).
3. The attention layers (in [`tensorrt_llm/_torch/modules/attention.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/attention.py)) receive the original `mapping_with_cp` to execute with KV parallelism, while all other layers (FFN, MoE, embeddings) use the repurposed TP-only mapping.

This design means that from the perspective of non-attention layers, Helix is transparent: they simply see a larger TP group. Only the attention module needs awareness of the CP configuration.

### Request Handling for Helix

Since Helix is a **decode-only feature** designed for **disaggregated serving**, special request handling is needed on the generation server. Both helpers below live in [`tensorrt_llm/_torch/pyexecutor/request_utils.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/request_utils.py):

1. **Context partitioning** (`partition_context_for_helix`): When a new request arrives at the generation server, its input token sequence is partitioned across CP ranks. Each rank receives the blocks of tokens it is responsible for, along with corresponding position IDs. Blocks are assigned in round-robin fashion across CP ranks based on `tokens_per_block`.

2. **Request merging** (`merge_helix_requests`): Each CP rank creates its own view of the request with only its local token IDs and position IDs, while tracking the total input length across all CP ranks (`total_input_len_cp`) for correct attention computation.

### KV Cache Management

Helix changes both how the KV cache moves from the prefill (context) server to the decode (generation) server, and how it grows on the generation server during the decode loop. Both pieces are handled inside TensorRT-LLM's existing KV cache infrastructure, gated on `gen_cp_size > 1`.

#### Cache transmission from prefill to decode

The prefill server runs without context parallelism (`ctx_cp_size = 1`) and produces the full KV cache for a request, while the generation server runs Helix with `gen_cp_size > 1`. As the cache crosses the wire, [`MLACacheFormatter`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp) and the split/concat kernels in [`cacheSplitConcat.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/executor/cache_transmission/cacheSplitConcat.cu) spread the KV blocks across generation-side CP ranks in a round-robin fashion, balancing the cache as evenly as possible (the lowest-indexed ranks receive one extra block when the total isn't divisible by `gen_cp_size`). The prefill server stays Helix-unaware: the Helix-friendly layout is materialized purely by the formatter during transfer.

#### Decode-side round-robin cache growth

Once decoding begins, every newly generated token still has to live on exactly one CP rank. Helix continues the same round-robin policy at `tokens_per_block` granularity, so KV cache growth stays balanced across the KVP group throughout the decode loop. This is the runtime counterpart of the [Distributed KV Concatenation](#distributed-kv-concatenation) policy described earlier.

The logic lives in [`tensorrt_llm/_torch/pyexecutor/resource_manager.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/resource_manager.py). For each generation request, the active CP rank for the current step is computed as:

```python
decode_block_id = (req.py_decoding_iter - 1) // tokens_per_block
active = (decode_block_id % cp_size) == cp_rank
```

- If a rank is **active**, it allocates a new KV slot, increments `seqlen_this_rank_cp`, and runs attention with the new token writing into its local KV cache.
- If a rank is **inactive** for this step (`req.py_helix_is_inactive_rank = True`), it does not allocate KV and its attention kernels attend over previously cached tokens only. In [`tensorrt_llm/_torch/attention_backend/trtllm.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/attention_backend/trtllm.py), `kv_lens[active_rank] += seq_lens_kv` ensures inactive ranks see no cache growth for the current step.

`total_input_len_cp` and `seqlen_this_rank_cp` are tracked per request so the global position id and per-rank attention length stay correct even though only one CP rank "owns" each new token.

### Custom Kernels

The Helix implementation includes dedicated CUDA kernels, exposed to Python via [`tensorrt_llm/_torch/distributed/ops.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/distributed/ops.py) (which provides `alltoall_helix()` and `HelixAllToAllNative`):

- **[`helixAllToAll`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/helixAllToAll.cu)**: Implements the All-to-All collective for exchanging partial attention outputs (half-precision, shape [..., cp_size, kv_lora_rank]) and SoftMax statistics (float32, shape [..., cp_size, 2]) across KVP ranks. Two backends are available:
  - *NCCL backend*: uses grouped NCCL send/recv operations
  - *MNNVL backend* (`HelixAllToAllNative`, **recommended**): uses multi-node NVLink workspace memory for lower-latency transfers, avoiding NCCL overhead. Selected by setting `cp_config.use_nccl_for_alltoall: false`.
- **[`helixPostProcess`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/helixKernels.cu)** (Torch op binding in [`helixPostProcessOp.cpp`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/helixPostProcessOp.cpp)): Performs the SoftMax correction after the All-to-All, combining partial attention outputs from all KVP ranks into the final correct result using log-sum-exp rescaling.
- **Attention kernel integration**: The FMHA (Fused Multi-Head Attention) and XQA kernels are extended with Helix-specific parameters to support local attention over KV cache shards and produce the per-shard LSE statistics needed for postprocessing.

### Composability

Helix is implemented as an additional dimension on top of TensorRT-LLM's existing parallelism and runtime stack, so it composes seamlessly with the rest of the framework rather than replacing any of it:

- **Tensor Parallelism (TP)**: Helix multiplies onto the attention TP_A factor (with the constraint TP_A ≤ K) and is then repurposed into a wider TP_F group for FFN/MoE layers via `repurpose_helix_cp_to_tp()`.
- **Pipeline Parallelism (PP)**: PP is orthogonal to KVP. Each pipeline stage independently applies the Helix attention/FFN split across its own GPU pool.
- **Expert Parallelism (EP)**: For MoE models, the FFN phase runs as `TP_F × EP` over the same N GPUs reused from the attention phase, so EP composes naturally with Helix without any extra reshaping.
- **Attention Data Parallelism (AttnDP)**: Helix coexists with attention data parallelism, so different attention DP groups can independently run KVP-sharded attention.
- **CUDA graphs**: The decode iteration (including local FlashAttention, `alltoall_helix`, and `helixPostProcess`) is captured into CUDA graphs, eliminating per-step launch overhead just as it would for non-Helix decode.
- **Overlap scheduler**: Helix is fully compatible with TRT-LLM's overlap scheduler, allowing the next iteration's host-side scheduling to be hidden behind the current iteration's GPU work.

## When to Use Helix

Helix parallelism provides performance benefits when all of the following conditions apply:

1. **Disaggregated serving**: Helix targets the generation (decode) servers in a disaggregated deployment, where prefill and decode run on separate GPU pools.
2. **Long input sequences**: Performance gains typically appear with input sequence lengths **>64K tokens** or more, where KV cache streaming dominates DRAM bandwidth.
3. **Low batch sizes / latency-sensitive workloads**: Helix is optimal for operating points toward the high tokens/second/user end of the latency-throughput Pareto curve.
4. **Blackwell GPU architecture**: The custom kernels and MNNVL support are designed for Blackwell systems (GB200 NVL72 and similar).

## Performance Results: DeepSeek-R1 (MoE, MLA)

The following results for DeepSeek-R1 are obtained on GB300 NVL72 using TensorRT-LLM with FP4 precision in disaggregated serving mode (Helix on the generation servers).

<div align="center">
<figure>
  <img src="../media/tech_blog21_dsr1_fp4_pareto.png" alt="DeepSeek-R1 FP4 throughput-latency Pareto on GB300 NVL72 with Helix, with parallelism configuration (concurrency, KVP, TP/DP, EP, PP) annotated at each Pareto point" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 4. Throughput-latency Pareto frontier of serving DeepSeek-R1 (FP4) on GB300 NVL72 with Helix on the generation servers. Helix pushes the frontier outward, enabling both higher throughput and lower latency.</em></sub></p>
<p align="center"><sub><em>Note: KVP × TP × DP = MoE_TP × MoE_EP, with DP=1 when attention is tensor-parallel and TP=1 when attention is data-parallel. The label's EP is MoE_EP; any remaining width is MoE_TP. PP is shown only when PP > 1, and the total GPU count per generation instance is N = KVP × TP × DP × PP.</em></sub></p>

For DeepSeek-R1 at 1M sequence length on GB300 NVL72, Helix delivers up to **32x concurrency improvement** and **1.8x interactivity improvement** compared to conventional parallelism approaches.

The configurations on the baseline Pareto frontier at the low-latency end are largely TP-driven. DeepSeek-R1 uses Multi-Latent Attention (MLA), where the per-token KV cache is a single shared latent vector, so K is effectively 1. As noted in the [KV Cache Streaming](#kv-cache-streaming) section, any TP > 1 with MLA therefore forces the full multi-million-token KV cache to be duplicated on every TP rank. Per-GPU KV-read time stays pinned at its single-GPU value no matter how wide TP grows, and every replica must reserve HBM for the entire KV cache, capping the effective batch each rank can carry. Past a certain TP, additional GPUs only add duplicated KV traffic, so the baseline curve hits the attention DRAM ceiling and per-GPU throughput collapses.

Helix breaks this ceiling on the same pool of N GPUs by pinning TP_A ≤ K (for MLA, TP_A = 1, so no KV duplication) and spending the remaining width on KV Parallelism, which shards the KV cache along the sequence dimension so per-GPU attention DRAM traffic scales as ~1/KVP. The same N GPUs are then re-provisioned as TP_F × EP = N for the FFN block, so attention and FFN bottlenecks shrink together. This is why Helix extends the Pareto frontier into the low-latency region where the baseline degrades, yielding up to 1.8x interactivity improvement at low concurrency.

## Future Work

- **HOP-B pipelining**: Implement fine-grained batch-wise pipelining in TensorRT-LLM to overlap All-to-All communication with attention computation, delivering up to 12% additional improvement for dense models.
- **Speculative decoding support**: Extend Helix to compose with TensorRT-LLM's speculative decoding (e.g. MTP, EAGLE-3, NGram, draft model) so that long-context decode workloads can additionally benefit from multi-token-per-step generation.

## Acknowledgment

Helix Parallelism in TensorRT-LLM is the result of a close collaboration between NVIDIA Research, DevTech, and the TensorRT-LLM team, spanning algorithm design, kernel engineering, runtime integration, and distributed systems. It is a concrete example of a state-of-the-art research idea, originally proposed in the [Helix paper](https://arxiv.org/pdf/2507.07120), being co-designed with hardware-aware kernels and brought all the way into a production-grade inference stack. We hope that this work helps the developer community deploy long-context LLM inference more efficiently on NVIDIA GPU systems.
