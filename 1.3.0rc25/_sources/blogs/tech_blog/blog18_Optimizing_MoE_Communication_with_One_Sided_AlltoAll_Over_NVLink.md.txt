# Optimizing MoE Communication with One-Sided AlltoAll Over NVLink

Large-scale Mixture-of-Experts (MoE) models have become the dominant architecture for open-source LLMs. For inference, a common parallel strategy is **Attention Data Parallel (DP) + MoE Expert Parallel (EP)**: each GPU handles a portion of the requests (thus tokens) for attention and a portion of the experts for MoE. Since a token's routed experts may reside on other GPUs, efficient communication is crucial for performance.

In this blog, we introduce **NVLinkOneSided AlltoAll**, the MoE communication kernels in TensorRT LLM designed for NVLink-connected systems (DGX B200, GB200 NVL72, etc.). We describe the key design choices, implementation details and performance benchmark results.

## Table of Contents
- [Optimizing MoE Communication with One-Sided AlltoAll Over NVLink](#optimizing-moe-communication-with-one-sided-alltoall-over-nvlink)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Design Overview](#design-overview)
    - [NVLink Symmetric Memory](#nvlink-symmetric-memory)
    - [One-Sided Communication](#one-sided-communication)
    - [Rank-Major Buffer Layout](#rank-major-buffer-layout)
    - [Interface Between Communication and MoE](#interface-between-communication-and-moe)
    - [Quantization-Agnostic Communication](#quantization-agnostic-communication)
  - [Implementation Details](#implementation-details)
    - [Dispatch Kernel](#dispatch-kernel)
    - [Combine Kernel](#combine-kernel)
    - [Synchronization](#synchronization)
    - [Saturating NVLink Bandwidth](#saturating-nvlink-bandwidth)
  - [Performance Benchmark](#performance-benchmark)
    - [Methodology](#methodology)
    - [Scaling With Batch Size and EP Size](#scaling-with-batch-size-and-ep-size)
    - [Post-Quant Dispatch](#post-quant-dispatch)
    - [Reproduction](#reproduction)
  - [Conclusion](#conclusion)

## Background

There are two communication patterns that can be used between attention DP and MoE EP:

**1. AllGather + ReduceScatter.** After attention, AllGather brings all tokens to every rank. Each rank computes MoE on the full token set using its local experts. After MoE, ReduceScatter sums the partial results back to each rank.

**2. AlltoAll (Dispatch + Combine).** AllGather is redundant: if a token is not routed to any expert on a given rank, it does not need to be communicated there. The waste is especially significant when the EP size is larger than `top_k`, since a token is routed to at most `top_k` experts. AlltoAll sends each token only to the ranks that own its routed experts, eliminating the redundancy. The AlltoAll after attention is called **dispatch** and the AlltoAll after MoE is called **combine**.

Concretely, the total message size of AllGather/ReduceScatter scales as `ep_size × num_tokens_per_rank × hidden_dim` (including data movement to self), while AlltoAll scales as `top_k × num_tokens_per_rank × hidden_dim`, since each token is dispatched to at most `top_k` target ranks. When `ep_size > top_k`, AlltoAll communicates strictly less data.


## Design Overview

NVLinkOneSided AlltoAll rests on three design ideas:

- **One-sided data movement** — each kernel only reads OR only writes peer memory, eliminating cooperative send/recv and the extra copies it implies.
- **Rank-major recv buffer** — slots are indexed by source rank with no `top_k` duplication, shrinking buffer size and saving redundant communication.
- **Flexible interface** — quantization-agnostic data path, with a small API that exposes the recv buffer directly to the MoE module.

The remainder of this section walks through the NVLink symmetric memory substrate that the kernels are built on, then each of the three ideas in turn.

### NVLink Symmetric Memory

Communication across GPUs is enabled by CUDA's Virtual Memory Management (VMM) APIs. Each GPU allocates a piece of physical memory and exports a shareable handle for its allocation. These handles are then exchanged across all participating GPUs, allowing each GPU to import the remote handles and map the remote memory into its own virtual address space. The result is **symmetric memory**: any GPU can read from or write to any other GPU's portion of it via NVLink using standard instructions in kernels, as if they were normal global memory pointers. This works within a single NVLink domain — 8 GPUs on a DGX B200, or all 72 GPUs on a GB200 NVL72 rack.

<p align="center"><img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_symmetric_memory.png" alt="Symmetric memory across GPUs" width="600"/></p>
<p align="center"><em>Symmetric memory registration across GPUs. (Image courtesy: <a href="https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/">NVIDIA NCCL 2.27 Blog</a>)</em></p>

### One-Sided Communication

NVLinkOneSided eliminates the send/recv pairing typical of collective communication.

**Two-sided communication** uses a send/recv model: the sender and receiver must cooperate. With direct NVLink put, the sender writes into the target rank's symmetric memory, but the receiver still has to perform an **additional data movement** before the data lands in its final recv buffer. Common reasons for this extra step include:

- **Layout reconciliation.** The data lands in symmetric memory in an intermediate layout; the receiver re-permutes or repacks it into the final layout the downstream kernel expects.
- **FIFO drain.** Symmetric memory may be too small to hold the entire recv buffer, so it is used as a fixed-capacity FIFO; the receiver continuously drains it into local memory.

**One-sided communication** drops the cooperation entirely. Symmetric memory **is** the recv buffer, and the dispatch op returns **tensor views** directly into symmetric memory rather than allocating new tensors, so the MoE module consumes the dispatched data in place. In this way, extra local data movement on the receiver side is eliminated.

<p align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_one_sided_vs_two_sided.png" alt="One-sided vs two-sided communication" width="700"/>
</p>

### Rank-Major Buffer Layout

NVLinkOneSided uses a **rank-major** recv buffer: a tensor of shape `[ep_size, max_tokens_per_rank, ...]` split by source rank, where each rank's slice holds the tokens that source sent to this rank. Each token is sent **at most once per (src → tgt) pair** — even if multiple of its `top_k` experts land on the same target rank — so within a slice, no token appears more than once. Some slots may be empty after dispatch, depending on routing.

The figure below contrasts this layout against the **expert-major** layout used by some other implementations (e.g. [DeepEP Low Latency](https://github.com/deepseek-ai/DeepEP)) on a small example: `ep_size = 2`, `top_k = 2`, 3 tokens per rank, 2 experts per rank. T2 is routed to one expert on each rank, so the rank-major recv buffer (left) holds it once, while the expert-major recv buffer (right) duplicates it across both expert sub-rows.

<p align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_rank_major_vs_expert_major.png" alt="Rank-major vs expert-major recv buffer layout" width="700"/>
</p>

Rank-major layout has two advantages:

- **Smaller buffer.** The pre-allocated recv buffer is `1 / num_experts_per_rank` of an expert-major buffer.
- **Save duplicated communication.** When `top_k > ep_size` it is impossible to avoid duplication in an expert-major layout.

The MoE module consumes the rank-major recv buffer directly, and it is up to the MoE module to decide whether explicit expert permutation is needed for GroupGEMM. For example, [trtllm-gen MoE](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py) can efficiently load tokens from the raw rank-major buffer without additional permutation.


### Interface Between Communication and MoE

Dispatch feeds the MoE module; combine collects its outputs. In TensorRT LLM, the MoE module takes 4 payloads as input. The dispatch kernel communicates all of them to target ranks at once; the combine kernel performs the reverse — gathering results back and reducing them with router weights:

```python
# Dispatch: scatter local tokens to all ranks
def dispatch(
    hidden_states: Tensor,          # [local_num_tokens, hidden_size]
    hidden_states_sf: Tensor,       # [local_num_tokens, sf_size]
    token_selected_experts: Tensor, # [local_num_tokens, top_k]
    token_final_scales: Tensor,     # [local_num_tokens, top_k]
    ...
) -> Tuple[Tensor, ...]:
    # Returns same 4 payloads, each reshaped to [ep_size * max_tokens_per_rank, ...]
```

Here `hidden_states` is the (possibly quantized) activation tensor with `hidden_size` elements per token (or packed elements for certain quantization schemes — e.g., FP4 packs two elements per byte). `hidden_states_sf` carries the optional blockwise quantization scales. `token_selected_experts` contains the `top_k` routed expert ids per token, and `token_final_scales` holds the corresponding router weights.

The dispatch outputs are **tensor views** into symmetric memory — no allocation, no copy. The recv buffer is pre-allocated with `ep_size * max_tokens_per_rank` slots to accommodate the maximum number of tokens from all ranks, as described in [Rank-Major Buffer Layout](#rank-major-buffer-layout). The MoE module then performs GroupGEMM on the received payloads. Two points are worth noting:
- The MoE module only computes the experts on the local rank. For example, if a token's `token_selected_experts` is `[0, 1, 4, 7]` and only experts `[0, 1, 2, 3]` reside locally, the MoE output for that token is the weighted sum of experts `[0, 1]` only.
- Some slots are empty after dispatch. The dispatch kernel sets `token_selected_experts` of empty slots to an invalid expert id (`-1`), so the MoE module knows to skip them. (For instance, [trtllm-gen MoE](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py) consumes the raw-token recv buffer directly without re-permutation.)

The MoE module writes its output for each token to the same slot in the recv buffer. To obtain the final result, each rank combines the partial results from peer ranks.

```python
# Combine: gather results back to local tokens
def combine(
    final_hidden_states: Tensor,    # [ep_size * max_tokens_per_rank, hidden_size]
    ...
) -> Tensor:
    # Returns [local_num_tokens, hidden_size]
```

### Quantization-Agnostic Communication

NVLinkOneSided is **quantization-agnostic**. The AlltoAll kernel transports opaque bytes — quantization happens *before* dispatch (post-quant dispatch), and any recipe (FP8 block scales, MXFP8, NVFP4, BF16, …) plugs into the same code path. This has two consequences:

- **Flexibility.** The kernel does not need to change when the quantization format changes; the model's recipe drives the dispatch payload size directly.
- **Bandwidth savings.** Communicating quantized data (e.g., NVFP4 reduces dispatch payload to ~28% of BF16) directly reduces the bytes transferred over NVLink, unlike implementations that must communicate in BF16 and quantize afterward.

Pre-dispatch quantization is naturally aligned with the model's quantization recipe — the activations need to be quantized anyway before the MoE GEMMs. The combine path, by contrast, communicates in BF16 by default to match the kernel's reduction behavior; pre-combine quantization would alter the model's numerical output. Additionally, an optional **low-precision combine** path is available: the sender quantizes to FP8 before pushing into combine and the receiver dequantizes after pull, recovering most of the bandwidth saving with a small numerical impact from the round-trip quantization.

## Implementation Details

### Dispatch Kernel

The dispatch kernel **pushes** tokens from local memory into the target rank's symmetric memory. For each payload tensor, the recv buffer on each rank has shape `[ep_size, max_tokens_per_rank, ...]`. The slice `[src_rank, :, :]` stores data received from `src_rank`, so different source ranks never race when writing to the same peer.

Each CTA handles one token. For each of the `top_k` routed experts, the kernel computes:
- `target_rank` — the rank owning that expert, derived from `token_selected_experts`;
- `target_index` — the slot index within the `src_rank` slice of the destination's recv buffer.

To obtain a unique `target_index`, the kernel maintains atomic counters `send_counters[ep_size]` that track how many tokens have been sent from the local rank to each target rank — an atomic increment yields the next available slot. The slot assignment is local (decided by the sender's atomic state) and does not require knowing how tokens on other ranks are routed.

If multiple of a token's `top_k` experts map to the same rank, only the first gets a valid entry; duplicates are assigned `(target_rank, target_index) = (-1, -1)` and the kernel skips transmission for those entries. For each payload, the kernel loads the token data from local memory once and stores it to up to `top_k` peers' symmetric memory — one per unique target rank.

### Combine Kernel

The combine kernel **pulls** MoE outputs from peers' symmetric memory in the reverse direction. Each CTA handles one (reduced) token. The combine kernel reuses the `(target_rank, target_index)` pairs recorded during dispatch to read from the exact same slots — no routing computation is repeated. For each payload element, it loads from up to `top_k` peers' symmetric memory into registers, performs the reduction, and stores the result to local memory once. The reduction uses a pairwise tree pattern rather than sequential accumulation, keeping the dependency chain shallow.

The MoE module writes its output directly into symmetric memory, so that the combine kernel can read peer ranks' MoE output without an intermediate copy. This **zero-copy** path avoids a staging copy before the combine kernel starts; it requires small modifications to the MoE ops to accept a provided output buffer instead of allocating their own in PyTorch.

The diagram below illustrates the complete dispatch → MoE → combine flow with `ep_size = 2`, `top_k = 2`, and `batch_size = 3`. Dispatch computes `target_ranks` and `target_indices` to push tokens into the correct slots in the recv buffer; combine reuses the same routing to pull results back. Dashed arrows indicate deduplicated entries in the dispatch phase.

<p align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_dispatch_moe_combine.png" alt="Dispatch, MoE, and combine data flow with index calculation and deduplication" width="900"/>
</p>

### Synchronization

One-sided communication requires two producer-consumer barriers, scoped to the operations they belong to:

- **Tail of dispatch.** Before the MoE module can consume the dispatched data, all source ranks must have finished writing to symmetric memory.
- **Head of combine.** Before pulling MoE outputs from peer symmetric memory, every rank's MoE module must have finished writing its output.

Both barriers share a common write+poll structure built on **flag-based synchronization**. Each rank holds an array of `ep_size` flag slots in its symmetric memory and a monotonically increasing epoch counter. To enter a barrier:

1. Each rank writes its current epoch into every peer's flag slot for itself.
2. Each rank polls its own local flag slots until all `ep_size` peers have written the current epoch.


The two barriers differ only in their **memory-fence semantics**:

- The **dispatch barrier** issues a *release* membar BEFORE writing the flags, ensuring that all preceding peer-bound stores (token data written into peer symmetric memory) are globally visible by the time the flag store lands.
- The **combine barrier** issues an *acquire* membar AFTER the polling loop completes, ensuring that subsequent loads (reading MoE output from peer symmetric memory) observe the data written before each peer's release membar.

Together the release/acquire pair gives the producer-consumer ordering required between dispatch writes and MoE reads (and, symmetrically, between MoE writes and combine reads).

### Saturating NVLink Bandwidth

The key to high throughput is keeping bytes in flight across NVLink at all times. We apply both data-level parallelism (DLP) and instruction-level parallelism (ILP):

- **Vectorized load/store.** The kernel uses the widest vector type (up to 128-bit) allowed by the payload data's alignment.
- **Dispatch:** Load from local memory once, store to remote memory `top_k` times. The `top_k` loop is unrolled via template instantiation.
- **Combine:** load from `top_k` peers' symmetric memory (unrolled, kept in registers), reduce pairwise in a tree fashion rather than sequentially, and store to local memory once.

## Performance Benchmark

We benchmark the dispatch+combine communication kernels on **GB200 NVL72** using the DeepSeek-V3 model profile: `hidden_size=7168`, `top_k=8`, 256 total experts.

### Methodology

For each (ep_size, batch_size) configuration we report the dispatch and combine kernel latency (µs) and achieved NVLink bandwidth (GB/s). The achieved bandwidth is computed as:

```text
bandwidth = batch_size × min(ep_size, top_k) × bytes_per_token / latency
```

where `bytes_per_token` covers the activation payload **plus its scaling factors when present**. We neglect the smaller payloads (expert IDs and router weights) since they are an order of magnitude smaller than the hidden states. The `min(ep_size, top_k)` factor reflects the deduplication described in the [Rank-Major Buffer Layout](#rank-major-buffer-layout) section. Following the convention used by other MoE communication libraries (e.g. DeepEP), the reported bandwidth is **logical** — it includes the local-rank fraction of the traffic that does not actually traverse NVLink.


### Scaling With Batch Size and EP Size

We sweep ep_size ∈ {8, 16, 32, 64} with BF16 dispatch and combine:

**ep_size=8:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 18.9 | 6.07 | 31.0 | 3.70 |
| 2 | 18.4 | 12.47 | 31.1 | 7.36 |
| 4 | 18.1 | 25.34 | 31.4 | 14.59 |
| 8 | 17.6 | 52.22 | 31.4 | 29.23 |
| 16 | 18.5 | 99.34 | 32.6 | 56.31 |
| 32 | 18.2 | 201.72 | 32.7 | 112.25 |
| 64 | 22.3 | 328.68 | 34.1 | 215.18 |
| 128 | 31.7 | 462.40 | 38.1 | 384.96 |
| 256 | 50.8 | 578.18 | 53.0 | 554.44 |
| 512 | 89.5 | 656.24 | 91.7 | 640.56 |
| 1024 | 166.3 | 706.34 | 175.6 | 668.62 |
| 2048 | 311.8 | 753.28 | 322.6 | 728.20 |

**ep_size=16:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 18.8 | 6.10 | 30.8 | 3.72 |
| 2 | 19.1 | 12.02 | 31.4 | 7.31 |
| 4 | 18.5 | 24.84 | 31.6 | 14.54 |
| 8 | 18.9 | 48.62 | 31.8 | 28.86 |
| 16 | 18.4 | 99.50 | 33.0 | 55.61 |
| 32 | 20.2 | 181.98 | 34.2 | 107.46 |
| 64 | 24.1 | 304.90 | 35.6 | 206.35 |
| 128 | 34.6 | 424.10 | 39.9 | 367.72 |
| 256 | 54.8 | 535.84 | 57.0 | 514.83 |
| 512 | 97.1 | 604.54 | 98.2 | 597.79 |
| 1024 | 181.2 | 648.00 | 188.2 | 624.15 |
| 2048 | 343.3 | 684.10 | 348.4 | 674.15 |

**ep_size=32:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 19.8 | 5.78 | 31.4 | 3.65 |
| 2 | 19.7 | 11.64 | 31.3 | 7.34 |
| 4 | 19.6 | 23.44 | 31.4 | 14.60 |
| 8 | 19.4 | 47.28 | 32.2 | 28.53 |
| 16 | 18.9 | 97.16 | 33.4 | 54.89 |
| 32 | 21.1 | 173.58 | 34.6 | 105.95 |
| 64 | 26.2 | 279.66 | 36.4 | 201.45 |
| 128 | 37.8 | 387.86 | 41.1 | 356.77 |
| 256 | 61.5 | 477.10 | 59.7 | 492.00 |
| 512 | 109.6 | 535.94 | 102.5 | 572.90 |
| 1024 | 197.5 | 594.74 | 197.5 | 594.69 |
| 2048 | 373.7 | 628.54 | 369.5 | 635.66 |

**ep_size=64:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 27.8 | 4.12 | 32.6 | 3.52 |
| 2 | 24.8 | 9.24 | 32.2 | 7.13 |
| 4 | 29.1 | 15.74 | 33.0 | 13.92 |
| 8 | 20.4 | 45.02 | 33.3 | 27.53 |
| 16 | 20.6 | 89.14 | 35.0 | 52.46 |
| 32 | 23.1 | 159.20 | 36.1 | 101.68 |
| 64 | 28.8 | 255.24 | 37.7 | 194.70 |
| 128 | 41.6 | 352.70 | 44.5 | 330.08 |
| 256 | 65.3 | 449.76 | 62.9 | 466.65 |
| 512 | 115.7 | 507.34 | 106.7 | 550.16 |
| 1024 | 209.4 | 560.76 | 199.2 | 589.48 |
| 2048 | 399.2 | 588.34 | 372.1 | 631.18 |


The uni-directional NVLink Bandwidth of GB200 NVL72 is **900GB/s**. We plot the achieved bandwidth below:



<p align="left">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_bandwidth.png" alt="NVLinkOneSided bandwidth on GB200 NVL72" width="600"/>
</p>

**Key observations:**
- Given sufficient batch size, both dispatch and combine reach **~80%** of the peak NVLink bandwidth at `ep_size=8`. Dispatch lands slightly higher than combine at large batches despite carrying the routing/slot-index work, because combine does a top_k-way reduction on top of the same payload.
- Bandwidth slightly degrades as EP grows, because of the increased synchronization overhead inside the larger communication domain.

### Post-Quant Dispatch

We test the perf benefit of post-quant dispatch, with FP8 and FP4 respectively:

<p align="left">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_post_quant_dispatch.png" alt="Post-quant dispatch on B200" width="700"/>
</p>

At `bsz = 2048`:

| Recipe | bytes/token | byte-ratio vs BF16 | Disp µs | Disp GB/s | Speedup vs BF16 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| BF16  | 14336 | 1.00× | 311.8 | 753.3 | 1.00× |
| MXFP8 |  7392 | 1.94× | 172.2 | 703.4 | **1.81×** |
| NVFP4 |  4032 | 3.56× | 101.7 | 649.3 | **3.06×** |

**Key observations:**
- Quantization mainly helps at large batch size, where the communication is bandwidth bound.
- Dispatch speedup tracks the byte-ratio asymptote — quantization translates almost directly into faster dispatch.


### Reproduction

The benchmark is available in the TensorRT LLM repository. Example command:

```bash
python tests/microbenchmarks/bench_moe_comm.py --backend NVLINK_ONE_SIDED --profile deepseek_v3 --perfect_router --kernel_breakdown --iter_stats --ep_size 8 -b 1 -e 2048 -f 2 --output_file nvlink_one_sided.json
```
`srun` is required for multi-node NVLink benchmarking. See `python tests/microbenchmarks/bench_moe_comm.py --help` for the full set of options.

## Conclusion

NVLinkOneSided AlltoAll is a symmetric memory-based MoE communication kernel that combines a simple one-sided design with minimal overhead and a modular, quantization-agnostic interface. It is the default communication strategy within a single NVLink domain in TensorRT LLM and delivers the best performance among known MoE communication implementations for inference. NVLinkOneSided is also available in [FlashInfer](https://docs.flashinfer.ai/api/comm.html#flashinfer.comm.MoeAlltoAll).
