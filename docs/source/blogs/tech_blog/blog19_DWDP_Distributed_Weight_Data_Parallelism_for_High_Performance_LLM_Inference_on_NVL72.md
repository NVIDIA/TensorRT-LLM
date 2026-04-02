# DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72

By NVIDIA TensorRT LLM Team

In LLM inference, workload imbalances and communication bottlenecks often lead to excessive synchronization overhead, limiting GPU utilization. We present DWDP (Distributed Weight Data Parallelism), an inference parallelization strategy that preserves data-parallel execution while offloading MoE weights across peer GPUs. By removing collective inter-rank synchronization, DWDP allows each GPU to progress independently. Implemented in TensorRT-LLM and evaluated with DeepSeek-R1 on GB200 NVL72, DWDP improves end-to-end output TPS/GPU by 8.8% at comparable TPS/user in the 20-100 TPS/user serving range under 8K input sequence length and 1K output sequence length. The DWDP implementation has been merged into TensorRT-LLM ([PR #12136](https://github.com/NVIDIA/TensorRT-LLM/pull/12136)). A more detailed technical introduction is also available on arXiv ([link placeholder](https://arxiv.org/abs/XXXX.XXXXX)).

## Table of Contents

- [Motivation](#motivation)
- [DWDP Overview](#dwdp-overview)
  - [High-Level Design](#high-level-design)
  - [Roofline Analysis](#roofline-analysis)
- [DWDP Implementation](#dwdp-implementation)
  - [Key Components](#key-components)
  - [Runtime Flow](#runtime-flow)
  - [Current Code-Level Constraints](#current-code-level-constraints)
- [Key Optimizations](#key-optimizations)
  - [Eliminating Split-Weight Merge Overhead](#eliminating-split-weight-merge-overhead)
  - [Mitigating Asynchronous Communication Contention](#mitigating-asynchronous-communication-contention)
- [Evaluation](#evaluation)
  - [Experimental Setup](#experimental-setup)
  - [Context-Only Evaluation](#context-only-evaluation)
  - [End-to-End Evaluation](#end-to-end-evaluation)
- [Summary](#summary)
- [Future Work](#future-work)
- [Acknowledgment](#acknowledgment)

## Motivation

Most existing inference parallelism strategies introduce layer-wise inter-rank synchronization. That synchronization becomes increasingly problematic in real-world LLM serving, where per-rank workloads are rarely balanced. At the request level, different ranks often see different sequence lengths and KV-cache hit rates. At the weight level, activated computation can also vary across ranks, especially for MoE models. Together, these effects create substantial per-rank latency variation during inference. Once the execution model synchronizes at layer boundaries, end-to-end throughput becomes bounded by the slowest rank.

This effect can be quantified using a DEP configuration for DeepSeek-R1 on GB200 with `ISL/OSL = 8K/1K` and input ratio `0.8`. In that setup, synchronization overhead reaches approximately `10%` when the coefficient of variation of per-rank sequence lengths is `20%`, which is well within the range observed in production workloads. In other words, synchronization overhead is not a corner case. Under realistic imbalance, it can materially reduce end-to-end inference throughput.

This leads to the key design question behind DWDP: can we remove collective synchronization and let each rank progress independently?

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/wanqian-nv/TensorRT-LLM/user/serli/dwdp_tech_blog/docs/source/blogs/media/tech_blog19_sync_overhead_in_dep.png" alt="Synchronization overhead caused by workload imbalance in DEP" width="600">
</figure>
</div>
<p align="center"><sub><em>Figure 1. Synchronization overhead caused by workload imbalance in DEP for DeepSeek-R1 on GB200 with <code>ISL/OSL = 8K/1K</code> and input ratio <code>0.8</code>.</em></sub></p>

## DWDP Overview

### High-Level Design

Figure 2 shows the core idea of DWDP on an MoE model such as DeepSeek-R1. DWDP preserves data-parallel execution across ranks while offloading MoE weights across peer GPUs. This design specifically targets MoE weights because they dominate the model memory footprint, whereas attention weights account for a much smaller share.

Within a DWDP group, attention weights are fully replicated on each rank, while the experts in every MoE layer are partitioned across ranks. As a result, each rank permanently stores only its local experts, and the remaining experts reside on peer GPUs. Before executing an MoE layer, the rank fetches the missing remote experts it needs for that layer.

At runtime, DWDP overlaps the asynchronous prefetch of remote experts for layer `l+1` with the MoE block of layer `l` and the attention block of layer `l+1`. Together, these two blocks create the compute window that hides remote weight prefetch. Before the MoE block of layer `l+1` begins, the rank waits only for its own prefetched experts to arrive. After the layer finishes, those prefetched remote experts are released. To sustain this pipeline across layers, DWDP uses double buffering with prefetching.

To eliminate collective inter-rank synchronization during inference, DWDP avoids NCCL-based collective remote-weight gathering such as all-gather. Instead, each rank pulls remote experts from peer GPUs through copy-engine-based `cudaMemcpyAsync`, which does not consume SM resources. These transfers are issued as serial peer-to-peer pulls, so they do not reintroduce synchronization across the group. Once a rank has the experts it needs for the next MoE block, it can continue independently.

DWDP also provides greater flexibility in expert placement. Because each rank only needs to fetch the weights for one layer before executing its MoE block, DWDP does not require the number of experts to be exactly divisible by the DWDP group size, and it does not require a perfectly disjoint expert partition across ranks. Instead, ranks can be configured with the same number of local experts while allowing redundant expert placement when necessary, for example to support group sizes that do not evenly divide the number of experts. This weaker placement constraint enables resource provisioning at single-rank granularity. When memory permits, the same redundancy can also reduce remote prefetch overhead by increasing the number of local experts on each rank.

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/wanqian-nv/TensorRT-LLM/user/serli/dwdp_tech_blog/docs/source/blogs/media/tech_blog19_dwdp_overview.png" alt="Overview of DWDP with DWDP group size 4" width="650">
</figure>
</div>
<p align="center"><sub><em>Figure 2. Overview of DWDP with DWDP group size 4.</em></sub></p>

### Roofline Analysis

We use a simple layer-wise roofline-style model to identify when DWDP can outperform DEP and what fundamentally limits its gain. This analysis focuses on the context phase of DeepSeek-R1 on GB200 and compares DWDP4 against DEP4, where both methods use a four-rank execution group.

We focus on two derived metrics in Table 1: `T_compute / T_prefetch`, which indicates whether DWDP can hide remote weight prefetch, and `T_DEP / T_DWDP`, which captures DWDP's expected advantage over DEP.

Table 1 shows that DWDP begins to outperform DEP at around 16K input tokens at batch size 1. As input sequence length increases, `T_compute / T_prefetch` grows from below `1` to above `1`, indicating that longer contexts provide a sufficiently large compute window to amortize and eventually hide remote prefetch overhead. This reveals a key limitation of DWDP: it needs enough computation per layer to cover remote weight prefetch. The 16K crossover is specific to the batch-size-1 setting. Increasing the batch size enlarges the compute window and can make DWDP beneficial even for shorter contexts.

| Input sequence length | `T_compute / T_prefetch` | `T_DEP / T_DWDP` |
| --- | ---: | ---: |
| 1024 | 0.19 | 0.10 |
| 8192 | 0.62 | 0.73 |
| 16384 | 1.52 | 1.27 |
| 32768 | 4.77 | 1.17 |

*Table 1. Roofline-style analysis data for DeepSeek-R1 context on GB200. The crossover around 16K tokens is where DWDP begins to outperform DEP at batch size 1.*

DWDP's advantage over DEP comes from eliminating synchronized all-to-all communication from the critical path. This advantage, however, is not monotonic in input sequence length. Once the sequence becomes very long, computation dominates both methods, so synchronized all-to-all overhead accounts for a smaller fraction of DEP's latency.
Accordingly, the marginal speedup of DWDP decreases as ISL grows further.

Importantly, this is a conservative analysis: it assumes perfectly balanced workloads and therefore does not capture the additional benefit DWDP can deliver under real-world imbalance, where avoiding synchronization overhead matters even more.


## DWDP Implementation

In this section, we focus on the main DWDP runtime components and the runtime flow during inference.


### Key Components

#### `DwdpConfig`

The configuration surface lives in `tensorrt_llm/llmapi/llm_args.py`. DWDP is off by default. In the current productized flow, this config is used on the context server of disaggregated serving.

The four fields are:

- `dwdp_size`: the number of GPUs in each DWDP group
- `num_groups`: the number of DWDP groups; total context workers = `num_groups * dwdp_size`
- `num_experts_per_worker`: the number of experts each worker keeps locally
- `num_prefetch_experts`: the number of experts each worker fetches from each peer rank

Together, these fields define the DWDP group structure and how experts are split between local residency and remote prefetch before inference starts.


#### `DwdpLayerHandleCollector`

Each DWDP-enabled MoE layer registers a `DwdpLayerHandleCollector`. During model initialization, it serves as the per-layer metadata carrier that later enables runtime prefetch.

- record the CUDA IPC handles for that layer's local MoE weights and related tensors
- record tensor shapes, dtypes, and allocation offsets
- hold peer pointers to that layer's remote MoE weights and related tensors on peer GPUs


#### `DwdpPrefetchBuffer`

`DwdpPrefetchBuffer` is the runtime buffer that stores prefetched remote experts. Its role is to keep the next layer's remote experts ready without overwriting the data still needed by the current layer.

- two prefetch buffers in ping-pong form
- a dedicated prefetch stream
- prefetch-completion events that tell compute when prefetched data is ready
- compute-completion events that tell the next prefetch when a buffer can be safely reused

#### `DwdpManager`

`DwdpManager`, implemented in `tensorrt_llm/_torch/pyexecutor/dwdp.py`, is the control center of the DWDP runtime. It owns the DWDP lifecycle and orchestrates when prefetch happens, while `DwdpPrefetchBuffer` provides the storage, stream, and events used by that pipeline. `DwdpManager` is responsible for:

- forming the DWDP group from the global MPI world
- creating and tracking one `DwdpLayerHandleCollector` for each DWDP-enabled MoE layer
- all-gathering metadata across the DWDP group, where the metadata records the local MoE parameter information on each GPU
- allocating and initializing the prefetch buffer
- triggering layer-by-layer prefetch at the right time


### Runtime Flow

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/wanqian-nv/TensorRT-LLM/user/serli/dwdp_tech_blog/docs/source/blogs/media/tech_blog19_dwdp_runtime_flow.png" alt="DWDP runtime flow" width="700">
</figure>
</div>

In the current DWDP code path, the runtime flow can be summarized by the diagram above.

1. Configuration and group formation

   If `dwdp_config` is present, `py_executor_creator.py` creates a `DwdpManager`. At that moment, DWDP forms the worker's DWDP group, determines its local DWDP rank within that group, and determines its local expert range.

2. Per-layer metadata registration during model initialization

   As each DWDP-enabled MoE layer is initialized, it calls `DwdpManager.add_layer(...)` and gets back a `DwdpLayerHandleCollector`. At this stage, DWDP is not moving any weights yet. It is only creating the per-layer objects that will later record the local MoE parameter metadata needed for remote prefetch.

3. Local metadata registration and handle exchange

   After `load_weights()` completes for a DWDP-enabled MoE layer, `DwdpLayerHandleCollector.register_weights(...)` records the local metadata for that layer. This metadata mainly includes the CUDA IPC handles for the local MoE weights, along with the tensor information needed for peer access. After all relevant layers have registered their local metadata, `DwdpManager.exchange_all_handles()` all-gathers that metadata within the DWDP group so each rank knows which peer-GPU tensors it can pull during prefetch.

4. Prefetch buffer initialization

   After handle exchange, `DwdpManager.initialize_prefetch_buffer()` allocates the `DwdpPrefetchBuffer` and initializes the events used to coordinate prefetch and compute. At this point, the runtime has everything it needs to start asynchronous layer-by-layer prefetch.

5. Warmup at the start of each forward step

   At the start of each forward step, `PyExecutor` calls `prefetch_first_layers()`. This primes the first DWDP prefetches so that the first DWDP-enabled MoE layers in that step do not enter with an empty pipeline. In architectures such as DeepSeek-R1, the dense and attention work between consecutive MoE blocks is the compute window that DWDP tries to use to hide remote prefetch.

6. Layer-by-layer prefetch during inference

   During inference, when a DWDP-enabled MoE layer is about to run, DWDP first waits for that layer's prefetched remote experts to be ready. After the layer finishes, `DwdpManager.record_compute_and_prefetch_next(...)` records compute completion for that layer and immediately triggers prefetch for the next layer that will reuse the same ping-pong slot. This is the steady-state loop that keeps prefetch and compute overlapped.

### Current Code-Level Constraints

The current implementation supports only a narrow set of code paths:

- DWDP only supports the `CuteDSL` MoE backend with `NVFP4`.
- DWDP only supports `TP = 1` inside each DWDP group.
- DWDP only supports the MPI worker launch flow used by `trtllm-serve disaggregated_mpi_worker`.
- DWDP does not support overlap scheduler.
- DWDP does not support EPLB on the same MoE path.

## Key Optimizations

### Eliminating Split-Weight Merge Overhead

DWDP naturally produces split weights for each MoE layer: local experts stay in the model weights, while remote experts arrive in prefetch buffers. Existing groupedGEMM kernels usually assume that all required weights already live in one contiguous buffer. A straightforward implementation would therefore merge local and remote experts through a device-to-device (D2D) copy before every MoE call.

That extra merge is expensive because it inserts another bandwidth-heavy step directly on the critical path. In a baseline context-only profiling case with DeepSeek-R1 on GB200x4 under `ISL = 8K`, `ratio = 0.8`, and `max_num_tokens = 32768`, the baseline DWDP pays an additional `34 us` of D2D copy for this pre-launch merge, which accounts for about `3%` of iteration latency.

To remove that overhead, we extend the cuteDSL groupedGEMM kernels to support TensorList-based inputs so the groupedGEMM kernel can consume multiple weight buffers directly. Instead of first materializing a merged expert-weight buffer, the kernel performs the required indexing and address calculation internally while remaining compatible with the existing layout and sharding scheme. Although this design introduces a small amount of additional instruction overhead, including extra address computations and descriptor loads, profiling and end-to-end evaluation show no meaningful performance regression. In practice, the dominant bottlenecks remain the main compute workload and memory traffic, indicating that the proposed approach effectively removes pre-merge D2D overhead without negatively affecting overall performance.

The relevant code changes are in:

- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py`
- `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`
- `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py`
- `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py`

### Mitigating Asynchronous Communication Contention

Another practical challenge in DWDP is that asynchronous remote-weight pulls can create many-to-one contention at the source-side copy engine. As Figure 3 shows, in each MoE layer multiple ranks may simultaneously pull missing remote experts from the same peer rank. When the layer-wise compute window is only comparable to the remote-weight prefetch time, this source-side serialization stretches the communication window and exposes visible compute bubbles before the next compute region can begin.

One mitigation is to split each remote-weight transfer into fixed-size slices and schedule those slices in a round-robin order across active destination ranks. This reduces random communication delay by time-multiplexing the source-side copy engine more evenly across competing pulls.

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/wanqian-nv/TensorRT-LLM/user/serli/dwdp_tech_blog/docs/source/blogs/media/tech_blog19_async_comm_contention.png" alt="Nsight Systems trace showing many-to-one source-side communication contention in DWDP" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 3. Nsight Systems trace showing many-to-one source-side communication contention in DWDP under a short compute-window setting.</em></sub></p>

Our experiments show that the additional gain is most visible when the compute window is short. For example, under the `ISL = 8K` context-only workload with `ISL ratio = 0.5` and `max_num_tokens (MNT) = 16384`, contention mitigation delivers an `8%` TPS/GPU gain over the DWDP version without this optimization.

It is important to emphasize that this optimization is part of the broader DWDP design exploration, but it is not yet included in the current productized DWDP code path.



## Evaluation

### Experimental Setup

The experiments in this section use the following setup.
Unless otherwise stated, the results in this section do not include the additional performance gain from the contention-mitigation optimization described above.

- Hardware: GB200 NVL72
- Commit: the measurements in this section are based on TensorRT-LLM commit `3a89495`
- Model: [DeepSeek-R1-0528-NVFP4-v2](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2)
- Serving mode: disaggregated serving, with DWDP applied on the context server

We split the discussion into context-only and end-to-end results.

### Context-Only Evaluation

The context-only study isolates the context phase, uses the Artificial Analysis dataset, and compares DWDP against a DEP baseline.

#### Results

We first examine a context-only iteration-latency breakdown of DEP4 and DWDP4 for DeepSeek-R1 under `ISL = 8K`, `ratio = 0.8`, and `max_num_tokens = 32768` on a GB200 context server. The last column reports per-category deltas normalized to the DEP4 iteration latency.


| Category | DEP4 (`us`) | DWDP4 (`us`) | `Delta / T_DEP4` |
| --- | ---: | ---: | ---: |
| Attention | 269.67 | 320.56 | -3.86% |
| GroupedGEMM | 342.40 | 337.42 | 0.38% |
| DenseGEMM | 177.50 | 189.28 | -0.89% |
| Others | 241.69 | 284.32 | -3.23% |
| Communication | 126.74 | 0.00 | 9.60% |
| P2P Copy | 0.00 | 429.00 | -- |
| Synchronization Cost | 161.85 | 0.00 | 12.26% |
| **Iteration Latency** | **1319.85** | **1131.58** | **14.26%** |

*Table 3. Context-only iteration-latency breakdown of DEP4 and DWDP4 for DeepSeek-R1 under `ISL = 8K`, `ratio = 0.8`, and `max_num_tokens = 32768` on a GB200 context server.*

The breakdown highlights both the promise and the remaining inefficiencies of DWDP. Relative to DEP, DWDP removes synchronization cost entirely and takes communication off the critical path. Together, these two effects correspond to a `21.86%` gross reduction in iteration latency.

At the same time, compute categories such as Attention and Others become slower. This slowdown reduces the realized gain to a net `14.26%` improvement. Our follow-up analysis shows that it comes from communication-computation interference, and the dominant cause is power-induced frequency throttling.

### End-to-End Evaluation

The end-to-end study uses the SemiAnalysis dataset with ISL=`8K`, OSL=`1K`, and input ratio `0.8`. The generation server configuration is kept fixed, and DWDP is applied only to the context server. The comparison is made against Pareto points from the DEP baseline.

#### Results

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/wanqian-nv/TensorRT-LLM/user/serli/dwdp_tech_blog/docs/source/blogs/media/tech_blog19_e2e_pareto_frontier.png" alt="End-to-end Pareto frontier comparison between baseline and DWDP" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 4. End-to-end Pareto frontier comparison between baseline and DWDP.</em></sub></p>

Figure 4 shows that DWDP pushes the end-to-end Pareto points toward better serving efficiency: at similar TPS/user, it achieves higher output TPS/GPU than the baseline across most of the target range.

Table 4 summarizes the average speedup in each TPS/user range. The gain is most pronounced at lower TPS/user.
Comparing Pareto points with similar TPS/user, we find that DWDP typically uses fewer context GPUs than the baseline. This suggests that the gain primarily comes from reduced context GPU demand.

The serving-efficiency benefit becomes smaller at high TPS/user. In this region, the system is more heavily generation-bottlenecked, and the context stage cannot accumulate enough tokens to amortize DWDP's prefetch overhead.

| TPS/user range | Avg. DWDP TPS/user speedup | Avg. DWDP TPS/GPU speedup |
| --- | ---: | ---: |
| 20-30 | 1.15 | 1.10 |
| 40-50 | 1.16 | 1.08 |
| 60-70 | 1.00 | 1.10 |
| 80-90 | 1.00 | 1.06 |
| 170-180 | 1.00 | 0.97 |

*Table 4. End-to-end performance summary of DWDP across target TPS/user ranges.*

We also evaluate median TTFT, including queueing time, the results are summarized in Table 5.
Compared with the baseline, DWDP increases TTFT across the evaluated TPS/user ranges. At low TPS/user, TTFT can increase substantially for pairs with more aggressive reductions in context GPU count. These regressions come from lowering the aggregate service rate of the context stage and worsening rate matching between the context and generation stages. We expect this issue to be mitigated by better request matching in future work, especially because DWDP enables finer-grained context configurations.

| TPS/user range | TPS/GPU speedup | Baseline TTFT (ms) | DWDP TTFT (ms) |
| --- | ---: | ---: | ---: |
| 20-30 | 1.10 | 2538 | 8314 |
| 40-50 | 1.08 | 1919 | 7012 |
| 60-70 | 1.12 | 965 | 1640 |
| 80-90 | 1.06 | 1669 | 2280 |
| 170-180 | 0.97 | 494 | 660 |

*Table 5. Median TTFT comparison across target TPS/user ranges.*


### Reproducing Steps

The above experiments use:

- GB200 NVL72
- DeepSeek-R1 NVFP4 checkpoint
- Artificial Analysis dataset with `ISL = 8K`, `OSL = 1`, and `ISL ratio = 0.8`


To reproduce the same style of end-to-end comparison in TensorRT-LLM today:

1. Keep the generation-server configuration fixed.
2. Apply DWDP only to the context server.
3. Launch the context and generation workers through the DWDP disaggregated serving path.
4. Compare Pareto points against a DEP baseline by varying the number of context GPUs.
5. Report TPS/user, output TPS/GPU, and TTFT together rather than reading any one metric in isolation.

## Summary

- DWDP's first advantage is that it removes the synchronization penalty caused by imbalanced workloads, which makes it a better fit for real LLM serving.
- DWDP's second advantage is flexibility: it gives the system finer-grained freedom when provisioning context GPUs in disaggregated serving.
- DWDP is not a universal win. It needs a sufficiently large compute window to hide remote expert prefetch, which is why it is best matched to the context side.
- DWDP depends on strong hardware support. High-bandwidth peer GPU connectivity such as GB200 NVL72 is what makes DWDP practical.
- DWDP introduces new engineering challenges, especially around split-weight handling and asynchronous remote-weight pull.
- Today, DWDP supports only a narrow set of code paths and deployment assumptions. Expanding that support remains future work.


## Future Work


### Integrate Contention Mitigation into the Productized Path

The contention-mitigation optimization discussed earlier is not yet included in the current productized DWDP path. Integrating it into the production runtime is a natural next step, especially for workloads with short compute windows where many-to-one copy-engine contention is more likely to surface.

### Decouple Launch-Time Coordination from MPI

DWDP relies on the `trtllm-serve disaggregated_mpi_worker` launch path and separate launch scripts because handle exchange and group formation currently depend on MPI communication across context workers. We will replace this MPI-based launch-time coordination with a TCP-based method.

### Move Beyond CUDA IPC for Broader Topologies

Remote expert access is built on CUDA IPC handles, which are not suitable for cross-node deployment. We will replace them with a fabric-capable remote-memory mechanism so that DWDP can support broader topologies.

### Reduce Reliance on Kernel-Specialized Split-Weight Handling

Longer term, it may be worth exploring memory-management approaches that present a more unified weight view to the kernel, for example through virtual-memory-based assembly. This could reduce the reliance on kernel-specialized handling of split weights.


## Acknowledgment

We would like to thank everyone who contributed to this work.
