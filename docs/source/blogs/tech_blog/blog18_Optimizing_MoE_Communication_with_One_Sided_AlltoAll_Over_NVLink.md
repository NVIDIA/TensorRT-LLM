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
    - [Token-Major Data Layout](#token-major-data-layout)
    - [Quantization-Agnostic Communication](#quantization-agnostic-communication)
  - [Implementation Details](#implementation-details)
    - [Interface Between Communication and MoE](#interface-between-communication-and-moe)
    - [Dispatch Put and Combine Get](#dispatch-put-and-combine-get)
    - [Synchronization](#synchronization)
    - [Saturating NVLink Bandwidth](#saturating-nvlink-bandwidth)
  - [Performance Benchmark](#performance-benchmark)
    - [Reproduction](#reproduction)
  - [Future Work And Conclusion](#future-work-and-conclusion)

## Background

There are two communication patterns that can be used between attention DP and MoE EP:

**1. AllGather + ReduceScatter.** After attention, AllGather brings all tokens to every rank. Each rank computes MoE on the full token set using its local experts. After MoE, ReduceScatter sums the partial results back to each rank.

**2. AlltoAll (Dispatch + Combine).** AllGather is redundant: if a token is not routed to any expert on a given rank, it does not need to be communicated there. The waste is especially significant when the EP size is larger than `top_k`, since a token is routed to at most `top_k` experts. AlltoAll sends each token only to the ranks that own its routed experts, eliminating the redundancy. The AlltoAll after attention is called **dispatch** and the AlltoAll after MoE is called **combine**.

Concretely, the total message size of AllGather/ReduceScatter scales as `ep_size × num_tokens_per_rank × hidden_dim` (including data movement to self), while AlltoAll scales as `top_k × num_tokens_per_rank × hidden_dim`, since each token is dispatched to at most `top_k` target ranks. When `ep_size > top_k`, AlltoAll communicates strictly less data.


## Design Overview

NVLinkOneSided AlltoAll is built around three key design ideas: a shared memory substrate for direct GPU-to-GPU access, a one-sided communication model that eliminates intermediate copies, and a token-major data layout that enables deduplication. We discuss each in turn.

### NVLink Symmetric Memory

Communication across GPUs is enabled by CUDA's Virtual Memory Management (VMM) APIs. Each GPU allocates a piece of physical memory and exports a shareable handle for its allocation. These handles are then exchanged across all participating GPUs, allowing each GPU to import the remote handles and map the remote memory into its own virtual address space. The result is a **symmetric memory workspace**: any GPU can read from or write to any other GPU's portion of this workspace via NVLink using standard instructions in kernels, as if they were normal global memory pointers. This works within a single NVLink domain — 8 GPUs on a DGX B200, or all 72 GPUs on a GB200 NVL72 rack.

<p align="center"><img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_symmetric_memory.png" alt="Symmetric memory across GPUs" width="600"/></p>
<p align="center"><em>Symmetric memory registration across GPUs. (Image courtesy: <a href="https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/">NVIDIA NCCL 2.27 Blog</a>)</em></p>

### One-Sided Communication

Typically, collective communication employs a **two-sided** model: each rank acts as both sender and receiver. The sender loads data from local memory and stores it into a FIFO buffer in the receiver's local memory. The receiver then loads the data from the FIFO and stores it into its local output buffer. This has two shortcomings:

- **Extra data movement.** The receiver performs additional data movement after the data has already arrived over NVLink and resides in the target GPU's memory.
- **Data dependency.** The receiver's work depends on the arrival of the sender's data. For example, the FIFO has a fixed capacity, so the sender (producer) or receiver (consumer) may block when the FIFO is full or empty.

**NVLinkOneSided** eliminates both issues by simplifying the communication model:

- The **dispatch** kernel only *puts*: it loads token data from local memory and writes it directly into the target rank's workspace.
- The **combine** kernel only *gets*: it reads data from source ranks' workspaces, reduces, and writes to local memory.

The workspace itself serves as the recv buffer — no intermediate FIFO, no extra copy on the receiver side. The dispatch OP returns **tensor views** directly into the workspace rather than allocating new tensors, so the MoE module could directly take the data on the workspace as input. Each direction of communication involves one less data movement step compared to two-sided.

<p align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_one_sided_vs_two_sided.png" alt="One-sided vs two-sided communication" width="700"/>
</p>

### Token-Major Data Layout


Some AlltoAll implementations (e.g., [DeepEP Low Latency](https://github.com/deepseek-ai/DeepEP)) use an **expert-major layout**: the recv buffer is shaped as `[num_experts_per_rank, max_num_tokens, ...]`, so that each expert gets its assigned tokens. If a token routes to multiple experts on the same rank, it is sent and stored multiple times — once per expert. Note that `max_num_tokens = ep_size * max_tokens_per_rank`, which is enough for accommodating tokens from all the ranks.

NVLinkOneSided uses a **token-major layout**: the recv buffer is shaped as `[max_num_tokens, ...]`, indexed by source rank. The transformation from token-major to expert-major layout is known as expert **permutation**. By directly employing token-major layout, the recv buffer size is shrunk by `num_experts_per_rank`. When a token routes to multiple experts on the same rank, the token data is **sent only once**. This deduplication reduces communication volume, which is especially beneficial when `top_k` is large, e.g., if `top_k > ep_size` then duplication is guaranteed to occur. The figure below shows the permutation from token-major to expert-major layout for 6 tokens routed to 4 experts with `top_k=2`. NVLinkOneSided requires only `1/num_experts_per_rank` of the pre-allocated recv buffer compared to expert-major implementations like DeepEP Low Latency.

<p align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_token_major_vs_expert_major.png" alt="Token-major vs expert-major layout" width="700"/>
</p>

The MoE module directly consumes the token-major recv buffer, and it is up to the MoE module to decide whether expert permutation is needed for GroupGEMM. For example, [trtllm-gen MoE](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py) can efficiently operate on token-major input without additional permutation.


### Quantization-Agnostic Communication

NVLinkOneSided is **quantization-agnostic**. It does not perform any quantization internally — instead, the quantization happens *before* dispatch (post-quant dispatch). This design has two advantages:

- **Flexibility.** Any quantization recipe works — FP8 block scales, NVFP4, or no quantization at all. The AlltoAll kernel does not need to change when the quantization format changes.
- **Bandwidth savings.** Communicating quantized data (e.g., NVFP4 reduces data to ~28% of BF16) directly reduces the bytes transferred over NVLink, unlike implementations that must communicate in BF16 and quantize afterward.

Note that pre-dispatch quantization is naturally aligned with the model's quantization recipe — the activations need to be quantized anyway before the MoE GEMMs. Pre-combine quantization, on the other hand, would be an additional step that alters the model's numerical output. For this reason, combine currently communicates in BF16. Optionally, FP8 static per-tensor quantization can be applied before combine — this is fast and introduces minimal accuracy impact.

## Implementation Details

### Interface Between Communication and MoE

In TensorRT LLM, the MoE module takes 4 payloads as input. The dispatch kernel communicates all of them to target ranks at once; the combine kernel performs the reverse — gathering results back and reducing them with router weights:

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

The recv buffer is pre-allocated with `ep_size * max_tokens_per_rank` slots to accommodate the maximum number of tokens from all ranks, as described in [Token-Major Data Layout](#token-major-data-layout). The MoE module then performs GroupGEMM on the received payloads. Two points are worth noting:
- The MoE module only computes the experts on the local rank. For example, if a token's `token_selected_experts` is `[0, 1, 4, 7]` and only experts `[0, 1, 2, 3]` reside locally, the MoE output for that token is the weighted sum of experts `[0, 1]` only.
- Some slots are empty after dispatch. The dispatch kernel sets `token_selected_experts` of empty slots to an invalid expert id (`-1`), so the MoE module knows to skip them.

The MoE module writes its output for each token to the same slot in the recv buffer. To obtain the final result, each rank combines the partial results from peer ranks.

```python
# Combine: gather results back to local tokens
def combine(
    final_hidden_states: Tensor,    # [ep_size * max_tokens_per_rank, hidden_size]
    ...
) -> Tensor:
    # Returns [local_num_tokens, hidden_size]
```

### Dispatch Put and Combine Get

The dispatch and combine kernels have complementary data movement patterns.

**Dispatch (put).** For each payload tensor, the recv buffer on each rank has shape `[ep_size, max_num_tokens, ...]`. The split `[src_rank, :, :]` stores data received from `src_rank`, so different source ranks never race when writing to the same peer. Each CTA handles one token. For each of the `top_k` routed experts, the kernel computes `target_ranks` (the rank owning that expert, derived from `token_selected_experts`) and `target_indices` (the slot index within the `src_rank` split). To obtain a unique `target_indices`, the kernel maintains atomic counters `send_counters[ep_size]` that track how many tokens have been sent from the local rank to each target rank — an atomic increment yields the next available slot. In this way, the unique slot assignment can be accomplished while the rank does not need to know how are the tokens on other ranks are routed.


If multiple of a token's `top_k` experts map to the same rank, only the first gets a valid entry; duplicates are assigned `target_ranks` and `target_indices` of `-1`. For each payload, the kernel loads the token data from local memory once and stores it to up to `top_k` remote workspaces — one per unique target rank.

**Combine (get).** Each CTA handles one token in the reverse direction. The combine kernel reuses the `target_ranks` and `target_indices` recorded during dispatch to read from the exact same workspace slots. For each payload element, it loads from up to `top_k` remote workspaces into registers, performs a weighted reduction, and stores the result to local memory once. The reduction uses a pairwise tree pattern rather than sequential accumulation, keeping the dependency chain shallow.
The MoE module can write its output directly into the symmetric memory workspace, so that the combine kernel can read peer ranks' MoE output without an intermediate copy. This **zero-copy** path avoids a staging copy before the combine kernel starts; it requires small modifications to the MoE ops to accept a provided output buffer instead of allocating their own in PyTorch.

The following diagram illustrates the complete dispatch → MoE → combine flow with `ep_size = 2`, `top_k = 2` and `batch_size = 3`. Dispatch computes `target_ranks` and `target_indices` to put tokens into the correct slots in the recv buffer; combine reuses the same routing to get results back. Dashed arrows indicate deduplicated entries in the dispatch phase.

<p align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_dispatch_moe_combine.png" alt="Dispatch, MoE, and combine data flow with index calculation and deduplication" width="900"/>
</p>

### Synchronization

One-sided communication requires explicit synchronization at two points:

**End of dispatch.** Before the MoE module can consume the dispatched data, all source ranks must have finished writing to the workspace.
**Beginning of combine.** Before getting data from peer workspaces, the MoE module on each rank must have finished writing its output.

We implement **flag-based synchronization**. Each dispatch-combine round increments a monotonically increasing epoch counter, so flags never need to be reset between rounds — a rank simply waits until the polled value equals the current epoch.

At the **end of dispatch**, the kernel issues a release fence to ensure all preceding stores (token data written to peer workspaces) are globally visible, then writes the epoch value to every peer's flag entry. It then spin-polls its own flag entries until all peers have signaled completion — at that point, every rank has finished writing and the MoE module can safely consume the dispatched data.

A similar barrier is used at the **start of combine**: each rank signals readiness and polls until every peer has done the same. An acquire fence follows the polling loop to ensure that subsequent loads (reading MoE output from peer workspaces) observe the data written before each peer's release fence.

### Saturating NVLink Bandwidth

The key to high throughput is keeping bytes in flight across NVLink at all times. We apply both data-level parallelism (DLP) and instruction-level parallelism (ILP):

- **Vectorized load/store.** The kernel uses the widest vector type (up to 128-bit) allowed by the payload data's alignment.
- **Dispatch:** Load from local memory once, store to remote memory `top_k` times. The `top_k` loop is unrolled via template instantiation.
- **Combine:** load from `top_k` remote workspaces (unrolled, kept in registers), reduce pairwise in a tree fashion rather than sequentially, and store to local memory once.

## Performance Benchmark

We benchmark the dispatch+combine communication kernels on **GB200 NVL72** using the DeepSeek-V3 model profile: `hidden_size=7168`, `top_k=8`, 256 total experts, FP8 blockwise quantization.

The tables below report the dispatch and combine kernel latency (µs) and achieved NVLink bandwidth (GB/s) for **NVLinkOneSided** across varying EP sizes and batch sizes. The achieved bandwidth is computed as:

```
bandwidth = batch_size × min(ep_size, top_k) × bytes_per_token / latency
```

where `bytes_per_token` takes the quantization into account. We neglect the smaller payloads (quantization scaling factors, expert IDs, and expert scales) as they are small compared to the hidden states. The `min(ep_size, top_k)` factor reflects deduplication.

**ep_size=8:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 18.9 | 3.03 | 31.0 | 3.70 |
| 2 | 18.4 | 6.24 | 31.1 | 7.36 |
| 4 | 18.1 | 12.67 | 31.4 | 14.59 |
| 8 | 17.6 | 26.11 | 31.4 | 29.23 |
| 16 | 18.5 | 49.67 | 32.6 | 56.31 |
| 32 | 18.2 | 100.86 | 32.7 | 112.25 |
| 64 | 22.3 | 164.34 | 34.1 | 215.18 |
| 128 | 31.7 | 231.20 | 38.1 | 384.96 |
| 256 | 50.8 | 289.09 | 53.0 | 554.44 |
| 512 | 89.5 | 328.12 | 91.7 | 640.56 |
| 1024 | 166.3 | 353.17 | 175.6 | 668.62 |
| 2048 | 311.8 | 376.64 | 322.6 | 728.20 |

**ep_size=16:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 18.8 | 3.05 | 30.8 | 3.72 |
| 2 | 19.1 | 6.01 | 31.4 | 7.31 |
| 4 | 18.5 | 12.42 | 31.6 | 14.54 |
| 8 | 18.9 | 24.31 | 31.8 | 28.86 |
| 16 | 18.4 | 49.75 | 33.0 | 55.61 |
| 32 | 20.2 | 90.99 | 34.2 | 107.46 |
| 64 | 24.1 | 152.45 | 35.6 | 206.35 |
| 128 | 34.6 | 212.05 | 39.9 | 367.72 |
| 256 | 54.8 | 267.92 | 57.0 | 514.83 |
| 512 | 97.1 | 302.27 | 98.2 | 597.79 |
| 1024 | 181.2 | 324.00 | 188.2 | 624.15 |
| 2048 | 343.3 | 342.05 | 348.4 | 674.15 |

**ep_size=32:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 19.8 | 2.89 | 31.4 | 3.65 |
| 2 | 19.7 | 5.82 | 31.3 | 7.34 |
| 4 | 19.6 | 11.72 | 31.4 | 14.60 |
| 8 | 19.4 | 23.64 | 32.2 | 28.53 |
| 16 | 18.9 | 48.58 | 33.4 | 54.89 |
| 32 | 21.1 | 86.79 | 34.6 | 105.95 |
| 64 | 26.2 | 139.83 | 36.4 | 201.45 |
| 128 | 37.8 | 193.93 | 41.1 | 356.77 |
| 256 | 61.5 | 238.55 | 59.7 | 492.00 |
| 512 | 109.6 | 267.97 | 102.5 | 572.90 |
| 1024 | 197.5 | 297.37 | 197.5 | 594.69 |
| 2048 | 373.7 | 314.27 | 369.5 | 635.66 |

**ep_size=64:**

| Batch Size | Dispatch (µs) | Dispatch (GB/s) | Combine (µs) | Combine (GB/s) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 27.8 | 2.06 | 32.6 | 3.52 |
| 2 | 24.8 | 4.62 | 32.2 | 7.13 |
| 4 | 29.1 | 7.87 | 33.0 | 13.92 |
| 8 | 20.4 | 22.51 | 33.3 | 27.53 |
| 16 | 20.6 | 44.57 | 35.0 | 52.46 |
| 32 | 23.1 | 79.60 | 36.1 | 101.68 |
| 64 | 28.8 | 127.62 | 37.7 | 194.70 |
| 128 | 41.6 | 176.35 | 44.5 | 330.08 |
| 256 | 65.3 | 224.88 | 62.9 | 466.65 |
| 512 | 115.7 | 253.67 | 106.7 | 550.16 |
| 1024 | 209.4 | 280.38 | 199.2 | 589.48 |
| 2048 | 399.2 | 294.17 | 372.1 | 631.18 |


The uni-directional NVLink Bandwidth of GB200 NVL72 is **900GB/s**. We plot the achieved bandwidth below:



<p align="left">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog18_bandwidth.png" alt="NVLinkOneSided bandwidth on GB200 NVL72" width="600"/>
</p>

**Key observations:**
- Given sufficient batch size, the combine kernel achieves **~80%** of the peak NVLink bandwidth.
- The achieved bandwidth of dispatch is lower than that of combine, since the dispatch kernel also performs token routing and slot index computation. The combine kernel, by contrast, is more communication-centric.
- At small batch sizes, latency dominates the kernel execution time.

### Reproduction

The benchmark is available in the TensorRT LLM repository. Example command:

```bash
python tests/microbenchmarks/bench_moe_comm.py --backend NVLINK_ONE_SIDED --profile deepseek_v3 --perfect_router --kernel_breakdown --iter_stats --ep_size 8 -b 1 -e 2048 -f 2 --output_file nvlink_one_sided.json
```
`srun` is required for multi-node NVLink benchmarking. See `python tests/microbenchmarks/bench_moe_comm.py --help` for the full set of options.

## Future Work And Conclusion

At small batch sizes, the performance bottleneck of NVLinkOneSided shifts from data movement bandwidth to **latency**. Besides the data movement itself, the flag writes and memory fences also contribute to this latency floor. To quantify the synchronization cost, we measure kernel latency with the barrier disabled (note: this breaks correctness due to potential data races):

**ep_size=8:**

| Batch Size | Dispatch w/ sync (µs) | Dispatch w/o sync (µs) | Combine w/ sync (µs) | Combine w/o sync (µs) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 18.9 | 11.3 | 31.0 | 28.1 |
| 2 | 18.4 | 11.1 | 31.1 | 29.2 |
| 4 | 18.1 | 11.5 | 31.4 | 28.6 |
| 8 | 17.6 | 11.3 | 31.4 | 29.4 |
| 16 | 18.5 | 11.6 | 32.6 | 32.2 |
| 32 | 18.2 | 12.6 | 32.7 | 32.3 |
| 64 | 22.3 | 16.8 | 34.1 | 33.2 |
| 128 | 31.7 | 25.8 | 38.1 | 36.2 |
| 256 | 50.8 | 43.8 | 53.0 | 50.3 |
| 512 | 89.5 | 79.3 | 91.7 | 74.8 |
| 1024 | 166.3 | 149.5 | 175.6 | 145.3 |
| 2048 | 311.8 | 291.5 | 322.6 | 278.5 |

In the future, we will continue to optimize the latency-bound scenarios and explore more aggressive MoE communication-computation overlap/fusion.


In conclusion, NVLinkOneSided AlltoAll is a symmetric memory-based MoE communication kernel that combines a simple one-sided design with minimal overhead and a modular, quantization-agnostic interface. It is the default communication strategy within a single NVLink domain in TensorRT LLM and delivers the best performance among known MoE communication implementations for inference.
