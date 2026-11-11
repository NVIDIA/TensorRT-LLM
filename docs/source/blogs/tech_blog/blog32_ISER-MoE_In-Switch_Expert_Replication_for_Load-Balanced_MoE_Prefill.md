# ISER-MoE: In-Switch Expert Replication for Load-Balanced MoE Prefill

*Per-Layer Dynamic Expert Load Balancing with NVLink Multicast in TensorRT LLM*

By NVIDIA TensorRT LLM Team

An expert-parallel mixture-of-experts (MoE) layer runs at the pace of its most loaded GPU, because fused MoE kernels such as TensorRT LLM's MegaMoE synchronize all GPUs. In DeepSeek-V4-Pro prefill routing recorded from a mixed trace of agentic coding and benchmark prompts, with expert parallelism (EP) over eight GPUs (EP8) and 8,192 tokens/rank (tokens per GPU per forward pass), the most loaded GPU averaged **1.44×** the mean load, and 2.24× at EP24. Replicating the few hot experts on lightly loaded GPUs would balance the layer, but each replica is a whole expert that must arrive within the layer: **37.16 MB** for DeepSeek-V4-Pro in NVFP4, NVIDIA's 4-bit floating-point (FP4) format.

**ISER-MoE (In-Switch Expert Replication for MoE) multicasts each hot expert through the NVLink switch into a fixed, aligned group of GPUs, so each byte leaves its owner GPU once, however many replicas the group holds.** At every MoE layer, HALO-Q (Hot-Aligned, Locality-Optimal expert replication with Quota repair), a GPU scheduler, picks hot experts and groups from that layer's routing, and a replication kernel multicasts each hot expert from its owner or through SAMI (Scalable Asynchronous Multi-port In-switch broadcast), which spreads the injection over the members' NVLink ports. Both run beside the shared expert, which every token uses, so the MoE kernel waits only for the plan.

In simulated replay of that routing at 8,192 tokens/rank, HALO-Q removed **98–99% of the excess load** (the most loaded GPU's load above the mean) at EP4–EP24, with four helper slots (spare expert buffers) per rank. With attention data parallelism (DP) and EP8 (DEP8), ISER-MoE raised DeepSeek-V4-Pro's context-only (prefill-only) request throughput on GSM8K (Grade School Math 8K) by **6.9% on GB200** and **12.6% on pre-release Rubin GPUs**, reaching over **99% of forced balanced-routing throughput** (Figure 10). In disaggregated serving of agentic coding traces on a GB300 NVL72 rack, it improved P90 (90th-percentile) request latency by **3.1–4.3%** and output throughput per GPU by **1.4–1.9%**.

**Numerical equivalence.** Replicas are byte copies of an expert's weights and block scales, and tokens keep their routing weights, so ISER-MoE changes where a token-expert pair is calculated, not what is computed; in the [MegaMoE kernel measurements](#423-megamoe-kernel-time-with-replicas) below, OFF and ON outputs matched bitwise.

TensorRT LLM's Expert Parallelism Load Balancer (EPLB), from Scaling Expert Parallelism [Part 1](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html#ep-load-balancer) and [Part 2](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog08_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html#expert-parallelism-load-balancer-eplb), plans from load history, once (offline) or a few layers per iteration (online). ISER-MoE adds per-layer dynamic EPLB, which re-plans every MoE layer on every forward pass; pull request (PR) [#19646](https://github.com/NVIDIA/TensorRT-LLM/pull/19646) merged it as a prototype feature of the NVFP4 MegaMoE backend ([Using ISER-MoE in TensorRT LLM](#5-using-iser-moe-in-tensorrt-llm)). The method applies in principle to prefill, generation and training; this work implements and evaluates prefill, with attention DP inside one NVLink domain.

**Table of Contents**

- **[1. Rank Imbalance in Expert-Parallel MoE Prefill](#1-rank-imbalance-in-expert-parallel-moe-prefill)**
    - [1.1 Expert-Parallel MoE](#11-expert-parallel-moe)
    - [1.2 Rank Imbalance Across Workloads and Layers](#12-rank-imbalance-across-workloads-and-layers)
    - [1.3 Long-Tail Distribution and Hot-Expert Replication](#13-long-tail-distribution-and-hot-expert-replication)
- **[2. Challenges in Per-Layer Expert Replication](#2-challenges-in-per-layer-expert-replication)**
    - [2.1 More Weight-Copy Overhead Than Routed-Expert Computation](#21-more-weight-copy-overhead-than-routed-expert-computation)
    - [2.2 Source Broadcast Hot Spot](#22-source-broadcast-hot-spot)
    - [2.3 Latency and Bandwidth Overheads in Baseline Broadcasts](#23-latency-and-bandwidth-overheads-in-baseline-broadcasts)
- **[3. ISER-MoE Design](#3-iser-moe-design)**
    - [3.1 HALO-Q Scheduler](#31-halo-q-scheduler)
    - [3.2 Multi-Level Symmetric Multicast](#32-multi-level-symmetric-multicast)
    - [3.3 TMA Weight Replication and SAMI Multi-Port Broadcast](#33-tma-weight-replication-and-sami-multi-port-broadcast)
    - [3.4 Runtime Overlap](#34-runtime-overlap)
- **[4. Performance Evaluation](#4-performance-evaluation)**
    - [4.1 Methodology](#41-methodology)
    - [4.2 Component Evaluation](#42-component-evaluation)
        - [4.2.1 HALO-Q Trace Replay](#421-halo-q-trace-replay)
        - [4.2.2 SAMI Broadcast on GB200](#422-sami-broadcast-on-gb200)
        - [4.2.3 MegaMoE Kernel Time with Replicas](#423-megamoe-kernel-time-with-replicas)
    - [4.3 End-to-End Serving Evaluation](#43-end-to-end-serving-evaluation)
        - [4.3.1 End-to-End Prefill Serving](#431-end-to-end-prefill-serving)
        - [4.3.2 End-to-End Disaggregated Serving on a GB300 NVL72 Rack](#432-end-to-end-disaggregated-serving-on-a-gb300-nvl72-rack)
- **[5. Using ISER-MoE in TensorRT LLM](#5-using-iser-moe-in-tensorrt-llm)**
    - [5.1 Configuration](#51-configuration)
    - [5.2 Requirements and Constraints](#52-requirements-and-constraints)
    - [5.3 Performance Measurement](#53-performance-measurement)
    - [5.4 Applicability and Evaluated Configurations](#54-applicability-and-evaluated-configurations)
- **[6. Discussion](#6-discussion)**
    - [6.1 Lower-Bound Analysis of Hierarchical and Free Slot Usage](#61-lower-bound-analysis-of-hierarchical-and-free-slot-usage)
    - [6.2 Limitations](#62-limitations)
    - [6.3 Related Work](#63-related-work)
- **[7. Conclusion and Further Reading](#7-conclusion-and-further-reading)**
    - [7.1 Conclusion](#71-conclusion)
    - [7.2 Further Reading](#72-further-reading)

## 1. Rank Imbalance in Expert-Parallel MoE Prefill

Agentic workloads send long, multi-turn prompts, so prefill holds most of the GPUs in disaggregated serving: TensorRT LLM's AgentPerf configurations of DeepSeek-V4-Pro on GB300 used 24 context GPUs against 8 or 16 for generation ([DeepSeek-V4 on NVIDIA Blackwell](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.html#agentperf-results-and-external-validation)).

### 1.1 Expert-Parallel MoE

An MoE layer routes each token to $k$ of its $E$ experts. EP spreads the experts over $P$ ranks (one GPU each; EP8 means $P=8$), each holding $H=E/P$ *home* experts; every invocation *dispatches* tokens to their experts' ranks and *combines* the outputs back ([One-Sided AlltoAll over NVLink](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html)). Under DEP, each rank contributes its own tokens (tokens/rank, counted before top-$k$ expansion), and DeepSeek-V4 also runs a *shared expert* on every token. MegaMoE, the `MEGAMOE_CUTEDSL` backend, is TensorRT LLM's fused MoE kernel in the CuTe domain-specific language (DSL), after [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)'s Mega MoE design: one persistent kernel runs dispatch, both expert general matrix multiplications (GEMMs) and combine, and synchronizes all ranks, **so every rank finishes with the most loaded one.**

Per layer, the *load* $L_r$ of rank $r$ counts its *routes* (token-expert assignments), and the rank imbalance ratio is

$$
R=\frac{\max_r L_r}{\bar{L}},\qquad 1\le R\le P,
$$

where $\bar{L}$ is the mean load; $R-1$ is the *excess*, and ranks above (below) the mean are *hot* (*cold*). On GB200 (EP8, 8,192 tokens/rank, trace-derived routing), MegaMoE without replicas took $513+1{,}051\thinspace R$ µs per layer, a fit across the 61 MoE layers (Pearson correlation 0.938).

### 1.2 Rank Imbalance Across Workloads and Layers

DeepSeek-V4-Pro's prefill routing ([DeepSeek-V4 report](https://arxiv.org/abs/2606.19348): 61 MoE layers, $E=384$, $k=6$) was recorded before load balancing on random tokens and five public workloads: [GSM8K](https://arxiv.org/abs/2110.14168), Massive Multitask Language Understanding (MMLU), IFBench, SciCode and [agentic coding traces from SemiAnalysis](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126). At EP8, the 61-layer mean of whole-trace $R$ was 1.40–1.45 for four workloads, 1.69 for GSM8K and 2.00 for random tokens (Figure 1); it ranged over 1.13–2.51 across the agentic traces' layers, and any two workloads share at most one of their five most imbalanced layers, much as [Part 1](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html#observation-over-gsm8k-dataset) saw DeepSeek-R1's hot layer move between datasets. The means also grow with EP size (Figure 1(b)).

![Heat strips of per-layer R at EP8 for six workloads, with 61-layer means from 1.40 to 2.00, and lines in which every workload's mean R rises from EP4 to EP8.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_rank_imbalance.svg)

*Figure 1. Rank imbalance ratio R (the most loaded rank's routed tokens over the mean) of DeepSeek-V4-Pro prefill before load balancing (whole-trace loads, contiguous experts per rank): (a) per MoE layer at EP8, with 61-layer means; (b) 61-layer means at EP4, EP6 and EP8.*

**Per-invocation load imbalance.** In a mixed trace interleaving the five workloads by request (4,340 requests, 18.2M prefill tokens, 91% agentic), $R$ per iteration (forward pass) at 8,192 tokens/rank averaged 1.20, 1.44, 1.84 and 2.24 at EP4, EP8, EP16 and EP24, against 1.41 whole-trace at EP8. Most of the skew in a fixed workload mix persists across invocations: a static plan from load history, as in DeepSeek's [EPLB](https://github.com/deepseek-ai/EPLB), can remove most of it, but not its drift between batches.

### 1.3 Long-Tail Distribution and Hot-Expert Replication

Expert load is long-tailed (Figure 2): over the mixed trace, the four busiest of 384 experts received 13.02% of a layer's assignments on average (1.04% if uniform), and 13.6% per iteration (EP8, 16,384 tokens/rank). At EP8, the four busiest home experts of a layer's hottest rank carried at least that rank's load above the mean in 59 of the 61 layers.

![Six panels of the cumulative share of a layer's assignments over experts sorted by load, all bowed above the uniform diagonal; in the mixed trace, the top 4 of 384 experts take 13.02%.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_long_tail.svg)

*Figure 2. Expert load is long-tailed (DeepSeek-V4-Pro, whole datasets): cumulative share of a layer's assignments over its 384 experts sorted by load. Thin lines: MoE layers; thick: 61-layer average, labeled with the Top-1 to Top-16 shares; dashed: uniform load. Mixed: the mixed trace.*

A *replica* copies a hot expert into a *helper slot*, a per-rank buffer for one more expert (each rank has $S$ of them), and takes routes from the expert's *owner*, its home rank. **Which experts are hot changes with workload, layer and batch, so balance needs replicas planned per invocation, at every layer, each a whole expert copied within that invocation.**

## 2. Challenges in Per-Layer Expert Replication

Replicas balance a layer only if whole experts arrive within the layer that uses them, and three obstacles stand in the way: copies compete with token traffic for link time, one owner must feed several replicas, and each of the three broadcast options examined below pays a latency or bandwidth overhead.

### 2.1 More Weight-Copy Overhead Than Routed-Expert Computation

A peak-rate model, not a measurement, counts the copies that computation can hide. Per MoE layer and rank, $\tau$ tokens make $\tau k$ token-expert pairs, each costing $6H_eI$ floating-point operations (FLOPs) in its routed expert (expert input width $H_e$ and intermediate width $I$) and, under uniform routing, moving $H_eb$ bytes off the rank with probability $(P-1)/P$ in both dispatch and combine ($b=0.5625$ B per NVFP4 element). Each of the $n$ expert copies a rank sends or receives moves $W=3H_eIb$ bytes. Per link direction,

$$
\frac{\text{math}}{\text{comm}}=\frac{6\thinspace\tau k H_e I}{2\thinspace\frac{P-1}{P}\thinspace\tau k H_e b+nW}.
$$

Communication hides at peak rates only if this ratio exceeds the *device ratio* of dense FP4 FLOP/s to per-direction link bandwidth, 16,667 FLOP/B on GB300 (15 PFLOP/s, 900 GB/s; [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)). For DeepSeek-V4-Pro and [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3), both assumed NVFP4, the routed-only ratio without copies is 18,725 FLOP/B at EP8; each copy lowers it, and 12 of 16 scenarios (0, 1, 2 or 4 copies; 8,192 or 16,384 tokens/rank; two models) stay above GB300's ratio (Figure 3(a), orange). At 8,192 tokens/rank, DeepSeek-V4-Pro's routed computation (432.9 µs) exceeds its dispatch plus combine (385.4 µs) by only 1.15 copies of 41.3 µs; four copies would outlast it by 117.6 µs and delay the combine (Figure 3(b)).

The shared expert needs neither the plan nor token communication, so it can run before dispatch while the links are idle. GB300 computes FP4 at three times its 8-bit floating-point (FP8) rate, so weighting the shared expert's FP8 FLOPs by 3 (FP8 shared experts assumed in both models) raises the math/comm ratio to 19,661–32,768 FLOP/B, and all 16 scenarios allow overlap (Figure 3(a), green). DeepSeek-V4-Pro's shared expert lasts 216.5 µs at 8,192 tokens/rank, room for about five more copies at line rate. **Planning and weight replication should start immediately after routing to overlap with shared-expert computation.**

![Bars of the math/comm ratio with 0 to 4 copies per rank: four routed-only bars fall below the 16,667 FLOP/B line, but no bar with the shared expert does; timelines show four copies delaying the combine but hiding beside the shared expert.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_copy_vs_compute.svg)

*Figure 3. Peak-rate model per MoE layer and rank on GB300 (EP8, uniform routing, NVFP4). (a) Math/comm ratio with n copies per rank, routed experts only (orange) or with the FP8 shared expert weighted ×3 (green); hatched: below the device ratio (dashed). (b) DeepSeek-V4-Pro, 8,192 tokens/rank, four copies: they delay the combine beside the routed experts but hide beside the shared expert.*

With MegaMoE's default 16-bit brain floating-point (BF16) combine instead of the modeled FP4 one, the combine alone (685 µs) outlasts DeepSeek-V4-Pro's routed computation (433 µs) at 8,192 tokens/rank, so copies hide only beside the shared expert.

### 2.2 Source Broadcast Hot Spot

Hot experts often need several replicas each, and a hot rank's busiest home experts share one owner. If an owner replicates $K$ experts, expert $e$ to $d_e$ ranks, point-to-point copies send $\sum_e d_e\le K(P-1)$ payloads out of it; relays, as in [UltraEP](https://arxiv.org/abs/2606.04101), which identified this fan-out bottleneck, relieve the owner but add a store-and-forward hop. In-switch multicast, part of the NVLink Scalable Hierarchical Aggregation and Reduction Protocol (NVLink SHARP, NVLS), replicates stores inside the NVLink switches: a store to a [CUDA multicast object](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html) reaches every member GPU's memory, so each byte leaves the owner once (Figure 4(a, b)).

In simulated replay of the mixed trace (8,192 tokens/rank, four helper slots per rank), a greedy point-to-point planner modeled on UltraEP and [MegaFlux](https://arxiv.org/abs/2610.00671), with no limit on an owner's send-outs, balanced slightly better than HALO-Q ($R$ = 1.001–1.009 against 1.006–1.020 at EP8–EP24), but its busiest owner sent 7/9/10 expert weights per invocation at EP8/16/24 at the 99th percentile, 289–413 µs at line rate and longer than the 216.5 µs shared-expert window, and up to 10/12/14, against four or fewer with multicast (Figure 4(c)).

![Eight GPUs: point to point, GPU 0 sends hot expert A five times; with multicast, it sends once and the switch replicates. Bars of worst-case owner send-outs: 10, 12 and 14 point to point against 4 with multicast.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_owner_hotspot.svg)

*Figure 4. Point-to-point replication makes the owner a hot spot. (a, b) Idealized EP8 example: point to point, GPU 0 sends hot expert A five times; with in-switch multicast, each owner sends once. (c) The greedy planner in simulated replay (DeepSeek-V4-Pro, mixed trace, 8,192 tokens/rank, four helper slots per rank), point to point or multicast: the busiest rank's worst-case expert send-outs per layer invocation.*

Multicast membership is fixed before use, and only members map the object, so a writer is a member and receives its own replica. Stores land at the same offset on all members, and each object takes one of a limited number of switch multicast IDs (MLIDs; 1,024 per GB200 NVLink domain, per the [partition guide](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf)). Yet the GPUs that need a replica change with every layer and batch, which is why UltraEP set in-switch multicast aside.

### 2.3 Latency and Bandwidth Overheads in Baseline Broadcasts

Multicast lets each byte leave the owner once, but each of the following three ways to broadcast one expert to $g$ GPUs pays a latency or bandwidth overhead (Figure 5); A and B are modeled for one 37.16 MB expert at 900 GB/s per link direction and 5 µs per step, and C is measured:

- **A, software ring.** The NVIDIA Collective Communications Library ([NCCL](https://github.com/NVIDIA/nccl)) 2.31.2 broadcasts only along a pipelined ring ([Demystifying NCCL](https://arxiv.org/abs/2507.04786)), whose pipeline fill grows with the group: with 2-MiB slices, its algorithm bandwidth (payload/time) falls to 250 GB/s at $g=16$.
- **B, software scatter + all-gather.** The classic large-message broadcast of [MPICH's collectives](https://doi.org/10.1177/1094342005051521) spreads injection over all $g$ ports but moves the payload twice in series, capping its bandwidth near half the link bandwidth.
- **C, owner-direct (single-port) multicast issued by a copy engine (CE).** Everything enters the switch through one ingress port, replicated at a fixed rate: on four GB200 trays (160 MiB), the total output stayed at 2.0–2.1 TB/s, so each receiver got 535, 266 and 127 GB/s at $g$ = 4, 8 and 16.

![Diagrams of a ring, a scatter + all-gather and a copy-engine write into one switch port, and a chart in which the ring falls to 250 GB/s at 16 GPUs and the single port from 535 to 127 GB/s.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_broadcast_limits.svg)

*Figure 5. Three ways to broadcast one expert, each limited by latency or bandwidth. (a) One expert in four slices, from GPU 0 to g = 4 GPUs: A, software ring; B, software scatter + all-gather; C, owner-direct (single-port) multicast issued by a CE. (b) Algorithm bandwidth (BW) against g: A and B modeled, C measured on four GB200 trays.*

**One injection per byte is necessary but not sufficient; injection must also be spread over many ports.**

## 3. ISER-MoE Design

**ISER-MoE uses prebuilt multicast groups with per-invocation expert placement.** Right after routing, [HALO-Q](#31-halo-q-scheduler) decides which hot experts to replicate where, bounding each owner's exports, and maps every route to an expert slot. A replication kernel then multicasts each replica with the Tensor Memory Accelerator (TMA) into one of a few [prebuilt aligned groups](#32-multi-level-symmetric-multicast), from its owner or through [SAMI](#33-tma-weight-replication-and-sami-multi-port-broadcast). Both [run beside the shared expert](#34-runtime-overlap): MegaMoE waits only for the plan, and only its *helper tiles*, its work on replicas, wait for the weights (Figure 6).

![Schematic: the MAIN and COPY streams of one rank with the READY gate; HALO-Q balancing four ranks to 1,000 routes each and repairing quotas; SAMI members multicasting slices; the original reference levels of 16, 8 and 4 ranks.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_design_overview.svg)

*Figure 6. ISER-MoE in one MoE layer (schematic). (a) One rank under DEP: the highest-priority COPY stream runs HALO-Q and the weight replication (one fused kernel in the merged implementation) beside the shared expert; helper tiles wait at the READY gate until every rank has reached READY counter γ, which advances once per invocation, and a reuse fence orders the next layer's HALO-Q after MegaMoE. (b) HALO-Q on ranks 0–3: balance to the water level T, then quota repair. (c) SAMI with g = 4 and owner GPU 0. (d) Original reference levels of P, P/2 and P/4 ranks at EP16 with S = 4 helper slots per rank (h0–h3); PR #21373 adds the aligned two-rank level.*

### 3.1 HALO-Q Scheduler

HALO-Q runs on the GPU once per MoE layer invocation, with no host step, and pursues two objectives in order:

- **Objective 1, balance.** MegaMoE finishes a layer with its most loaded rank, so HALO-Q brings every rank's routed load down to the *water level* $T=\lceil\sum_r L_r/P\rceil$, the mean rank load.
- **Objective 2, locality.** A route served off its source rank costs a token sent in dispatch and returned in combine, so among balanced routings HALO-Q calculates as many tokens as possible locally.

It plans in five stages:

1. **Count exchange.** Ranks exchange their $E$ route counts through a symmetric NVLink buffer and run the same deterministic planner, as in UltraEP, so all obtain the same plan.
2. **Placement (Objective 1).** HALO-Q repeatedly takes the busiest rank above $T$ that can still export, and its heaviest home expert not yet exported. The expert goes to an aligned group, with or without the owner, whose members all have a free helper slot: at the finest level that can absorb the whole expert at $T$ (coarser only if none there has room), the group with the most room below $T$. Up to the owner's excess over $T$ of the expert's routes move there, coldest member first. With $S$ helper slots per rank, each owner exports at most $S$ experts, among its $\min(S,H)$ heaviest, and each rank receives at most $S$ replicas.
3. **Slot coloring.** Each placed expert gets one helper index free on all members of its group.
4. **Quota repair (Objective 2).** Placement fixes each replicated expert's total quota, not its split among hosts, so two ranks may send each other tokens of different experts; in two round-robin passes over rank pairs, partners trade equal amounts of spare quota on experts both host, canceling such flows at unchanged loads.
5. **Route rewrite, local first.** Each rank fills its own quota first, and surplus routes fill other hosts' spare quotas in rank order, as in UltraEP's local-first cumulative quotas and [MoonEP](https://github.com/MoonshotAI/MoonEP)'s prefix sums.

### 3.2 Multi-Level Symmetric Multicast

ISER-MoE predefines aligned rank groups and lets HALO-Q select the destination group for each exported expert. For power-of-two EP, the placement hierarchy can extend through groups of $P,P/2,\ldots,2$ consecutive ranks: [PR #21373](https://github.com/NVIDIA/TensorRT-LLM/pull/21373) adds the aligned two-rank level. Figure 6(d) illustrates the original 16/8/4-rank levels at EP16. The [slot-usage comparison](#61-lower-bound-analysis-of-hierarchical-and-free-slot-usage) is presented in the Discussion.

The extension separates logical placement groups from their copy paths. At power-of-two EP4–EP16, the multicast ladder includes pairs. At EP32 and non-power-of-two EP, pair records support multicast of the large weight planes and unicast of the small planes into receiver helper rows. Pair and transport-team multicast objects are reused across helper sets; coarse levels retain per-set mappings.

In the original arena, the default [helper-slot mode](#51-configuration), `per_layer_resident`, gives each MoE layer its own helper set, so the coarse-level multicast-object count grows with the number of MoE layers; `shared_banks` uses two sets for all layers. The pair extension's reuse scheme changes the object accounting, so the number of logical groups is not itself the CUDA multicast-object count. This post does not measure the MLID budget at large EP sizes or with multiple model instances.

Each rank holds its $H$ home experts in slots $[0,H)$ and $S$ helper slots after them, allocated in peer-accessible CUDA fabric-handle memory; a *broadcast* installs an expert's weights and block scales (its *bundle*) in one helper slot on every member of an aligned group.

**Symmetric helper slots.** A multicast store lands at the same offset on every member, so a broadcast needs one helper index free on all members. For power-of-two EP, aligned groups are nested or disjoint: coarse-to-fine coloring needs at most as many indices as the maximum number of broadcasts containing any rank. [Section 6.1](#61-lower-bound-analysis-of-hierarchical-and-free-slot-usage) compares the achievable load-balance bounds under Hierarchical and Free usage.

### 3.3 TMA Weight Replication and SAMI Multi-Port Broadcast

The replication kernel copies with TMA from streaming multiprocessors (SMs), avoiding a CE's host round trip: warps stage 8-KiB slices in shared memory and store them to the group's multicast address (`multimem.cp.async.bulk`), and the switch writes each slice into every member's helper slot. In *owner-direct* replication, the owner, a group member, multicasts the whole bundle through its own port. With SAMI, the $g$ members share the injection in two pipelined steps, with no barrier or reassembly: member $i$ pulls each slice $j\equiv i \pmod g$ from the owner (s1) and stores it through its own port to the multicast address (s2), and the switch delivers it to every member (s3).

SAMI is thus a scatter + all-gather broadcast whose all-gather is in-switch multicast, as in NCCL's NVLS AllGather: the owner still sends the expert's $W$ bytes, but each member injects only $W/g$, so the switch output fed by one port falls from $gW$ to $W$ and the issue work is shared, at the price of $W/g$ extra inbound traffic per non-owner member. Since a multicast writer must be a member, SAMI also serves owners outside the group.

**Per-slot replication policy.** In the original runtime evaluated here, owner-direct replication serves a slot whose owner is in the group $G$ if every broadcast that meets $G$ targets exactly $G$ with its owner inside, or if this is the owner's widest broadcast into a group containing it; SAMI serves the rest. Whether this rule pays off with many owners is untested, and [one measurement](#422-sami-broadcast-on-gb200) shows it picking the slower path.

### 3.4 Runtime Overlap

Planning and weight replication start immediately after routing to overlap with shared-expert computation, as motivated by the [peak-rate analysis](#21-more-weight-copy-overhead-than-routed-expert-computation):

- **Two streams.** The COPY stream, the model's highest-priority stream, runs HALO-Q and then the copy, in the merged implementation as one fused kernel with one cooperative thread array (CTA) on each of `auxiliary_sms` SMs (four by default). Meanwhile, the MAIN stream quantizes the routed input and computes the shared expert, which DeepSeek-V4 hands to `MoE.load_balancer_overlap(fn)` to run right after the COPY work is submitted.
- **MAIN waits only for the plan,** on the device and without holding an SM, then launches MegaMoE; dispatch never waits for the copy, and there is no steady-state host wait.
- **Only helper tiles wait for weights.** MegaMoE claims tiles in slot order, home slots first, so dispatch overlaps the home tiles, and combine overlaps the later home and helper tiles. After its writes drain, each rank's replication kernel multicasts the invocation's READY counter $\gamma$ to every rank, even with nothing to copy, and each MegaMoE CTA waits at the READY gate before its first helper tile until every rank has reached $\gamma$.

To first order, the copy is exposed only if it outlasts the shared expert and the home tiles; a copy issued inside MegaMoE, as described for MegaFlux, would miss the shared-expert window. **Overlap depends on SM scheduling:** the fused kernel's CTAs must get SMs while MAIN runs input quantization and the shared expert. MAIN starts MegaMoE only after every one of these CTAs has finished its share of HALO-Q, so the copy already holds its `auxiliary_sms` SMs when persistent MegaMoE CTAs start waiting at the READY gate. Input quantization and, on the SM100 family, DeepSeek-V4's fused shared-expert kernel leave these SMs free, in response to [measured interference](#422-sami-broadcast-on-gb200); this post does not evaluate the effect of this SM budget.

## 4. Performance Evaluation

### 4.1 Methodology

The five experiments in this section test ISER-MoE from the plan alone up to the whole serving system, all with DeepSeek-V4-Pro's MoE layer and NVFP4 routed experts; OFF has no helper slots, and ON has $S$ per rank. A GB200 compute tray holds four GPUs, so EP8 spans two trays. Unless stated otherwise, kernel times are the slowest rank's median over 20 iterations, reported as the time reduction (OFF − ON)/OFF or the speedup OFF/ON, and component results come from pre-merge builds, each labeled where it appears.

### 4.2 Component Evaluation

#### 4.2.1 HALO-Q Trace Replay

A simulator replays the mixed trace at 8,192 tokens/rank through the reference HALO-Q planner's stages, planning each (iteration, layer) independently and counting assignments, not time.

**With four helper slots per rank, HALO-Q removed 98.4–99.0% of the excess load**: the 61-layer mean $R$ fell from 1.20/1.44/1.84/2.24 to 1.002/1.006/1.010/1.020 at EP4/8/16/24, and every layer stayed within 13% of balance (Figure 7(a)); three slots removed 94.9–95.9%. An earlier *all-rank policy*, replicating four experts on every rank, left a mean $R$ of 1.038/1.193/1.348 at EP8/16/24.

![Per-layer R at EP4 to EP24, flat near 1 with HALO-Q (means 1.002 to 1.020), and bars of the largest per-rank token send-out and receive-in, down 2–8% and 19–55%.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_haloq_eval.svg)

*Figure 7. HALO-Q in simulated trace replay (DeepSeek-V4-Pro, mixed trace, 8,192 tokens/rank). (a) Balance: per-layer R without balancing and with HALO-Q at four (solid) or three (dashed) helper slots per rank, sorted by R without balancing; labels: 61-layer means, three slots in parentheses. (b) Locality (four helper slots): the most routed tokens any rank sends out or receives in during dispatch, mean per iteration and layer, and the change from no balancing.*

For locality, the most routed tokens any rank sent out fell by 2–8%, and the most any rank received in by 19–55% (Figure 7(b)); quota repair alone cut both by a further 0.5–2.3%.

A static per-layer plan with the same four extra slots per rank, built by a re-implementation of DeepSeek's EPLB rule from past or whole-trace loads under an optimistic even token split, left $R$ = 1.06–1.07 at EP8, against 1.006. By the [GB200 fit](#11-expert-parallel-moe), extrapolated below its fitted range and without helper-tile overhead, it would still capture an estimated 85–88% of the kernel-time gain of per-invocation planning, so the two approaches are complementary.

HALO-Q's chosen group excluded the owner in 53% of broadcasts at EP8 and 76% at EP16–EP24, and the evaluated copy path uses SAMI for these groups; restricted to groups containing the owner, $R$ rose to 1.038/1.201/1.296 at EP8/16/24.

#### 4.2.2 SAMI Broadcast on GB200

These reference measurements, taken with a CE prototype and pre-merge builds of the replication kernel rather than the merged implementation, broadcast bundles from one owner to $g$ GPUs (Figure 8).

![Four panels: SAMI 1.13×, 2.03× and 3.76× faster than owner-direct multicast at 4, 8 and 16 GPUs; SAMI at 481 GB/s from four copy SMs; SAMI above tuned ncclBroadcast; span increases of 2.0/0.6/0.3% for SAMI against 36.3/14.7/2.9%.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_sami_eval.svg)

*Figure 8. Broadcasts from one owner on GB200 (reference measurements). (a) CE prototype (g = 4: two trays, 144 MiB; g = 8, 16: four trays, 160 MiB). (b) Pre-merge replication kernel, standalone, slowest rank: eight 37.16 MB bundles (283.5 MiB) to 8 GPUs, owner included. (c) One broadcast at a time, until the slowest rank holds it: fastest SAMI variant against fastest tuned ncclBroadcast (nccl-tests, NCCL 2.31.2). (d) Time increase from the shared expert's start to the end of MegaMoE while the build of (b) copies four experts (4 × 36 MiB) during the shared expert (EP8). Labels in (a, d): SAMI's speedup.*

**Multi-port injection.** On a CE prototype that issues SAMI's steps as host CE copies, in the measured placements (Figure 8(a)), a single injecting port lost per-receiver bandwidth as $g$ grew, while SAMI stayed near the owner's egress limit, 1.13×, 2.03× and 3.76× faster at $g$ = 4, 8 and 16. At $g=4$, HALO-Q's group size at EP8 and EP16 in replay, one port's 535 GB/s per receiver on four trays at 160 MiB (Figure 5(b)) is already close to that limit.

**SM-issued copies** add a per-SM issue limit. In a pre-merge build of the replication kernel, which, like the merged implementation, uses four SMs by default (Figure 8(b)), owner-direct multicast was issue-bound, rising from 46 GB/s on one SM to 258 GB/s on eight, while SAMI, with every member issuing a share, reached 481 GB/s from four SMs on. The default [per-slot rule](#33-tma-weight-replication-and-sami-multi-port-broadcast) nonetheless picks owner-direct here.

**Against tuned NCCL** (Figure 8(c)), each broadcast completes on every rank before the next starts, and NCCL 2.31.2 was tuned at every point over protocol, buffer size, channel count, buffer registration and CUDA-graph launch; the LL128 protocol with CUDA-graph launch won all 12 points. **SAMI's fastest variant, mostly another TMA build with the owner outside the multicast group, was 1.07/1.21/1.34× faster for one expert (35.4 MiB) at 8/12/16 GPUs** and 1.22–1.29× faster for eight. With the owner inside the group, as in Figure 8(b), SAMI only matched tuned ncclBroadcast (1.00×) for eight experts on 8 GPUs, as the owner also receives every replica, which slows the pulls it serves.

**Overlap with a shared expert.** In Figure 8(d), the pre-merge build copied four experts (EP8, balanced synthetic routing, dummy weights) during a 7168→2048→7168 FP8 shared expert, narrower than the model's. SAMI's copy took 310–330 µs and lengthened the span by 2.0/0.6/0.3% at 8K/16K/32K tokens/rank, against 0.98–1.20 ms and 36.3/14.7/2.9% for owner-direct multicast, mostly where its copy outlasted the shared expert (256/480/936 µs); SAMI slowed the shared expert by 10/4.8/2.0% and MegaMoE by at most 0.5%.

**The overlap degrades when the copy competes for SMs.** With an earlier revision of the replication kernel on eight SMs, none set aside, SAMI slowed the shared expert at 8K tokens/rank by 49%: the copy ran beside the shared expert's GEMMs for 99.9% of its time. The merged implementation's `auxiliary_sms` budget is the design response; this post does not evaluate its effect.

#### 4.2.3 MegaMoE Kernel Time with Replicas

Each measurement times one MegaMoE call for a DeepSeek-V4-Pro layer on GB200, OFF and ON with four helper slots per rank, on a pre-merge revision with an FP4 combine instead of the default BF16. ON's replicas, routes (from the all-rank policy) and READY flags were installed before timing, so **the times exclude HALO-Q, the copy and any READY wait**; OFF ran before and after ON and was averaged, and outputs matched bitwise.

![MegaMoE kernel time OFF and ON: 1.24×, 1.52×, 1.21× and 1.22× on power-law cells, ON near balanced routing; over 61 layers, mean 1,999 µs OFF against 1,624 µs ON (1.23×).](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_megamoe_kernel.svg)

*Figure 9. MegaMoE kernel time on GB200 (DeepSeek-V4-Pro layer, NVFP4, FP4 combine, four helper slots per rank; slowest rank, median of 20 iterations; replicas and routes from an earlier all-rank policy, installed before timing). (a) Power-law routing (exponent 0.8) at EP4 and EP8, 8K or 16K tokens/rank. (b) 61 layers, routes sampled from the trace (EP8, 8,192 tokens/rank); L20: layer 20. Dashes: balanced routing (OFF).*

With **power-law routing** (exponent 0.8) at EP4 and EP8 and 8,192 or 16,384 tokens/rank, replicas cut the skewed cells' $R$ from 1.27–1.69 to 1.00–1.03 and kernel time by 17.2–34.1% (1.21–1.52×), **to within 0.3–3.5% of balanced routing** (Figure 9(a)); on balanced routing, time changed by −0.7% to +0.4%.

For **trace routing**, EP8 routes at 8,192 tokens/rank were sampled per layer from the mixed trace's expert popularity, calibrated to the layer's EP8 imbalance within 3% (one seed). Every layer improved, from 1.3% (layer 45, $R=1.12$) to 42.8% (layer 20, $R=2.36$), and the mean fell from 1,999.3 to 1,624.3 µs (1.23×), 2.1% above balanced routing (Figure 9(b)); ON time was uncorrelated with $R$ (Pearson 0.01). Both routings are stationary, so this measures the kernel's gain from replicas, not the value of per-invocation planning.

**Costs outside the kernel** come from earlier builds. Run sequentially, HALO-Q took 19.6–29.2 µs and the replication kernel (eight SMs, READY included) 128.7–340.6 µs, so the copy must overlap other work. Beside a 300 µs single-CTA delay kernel that stood in for the shared expert but moved no data, with ranks arriving 10 µs apart, ON took 15.6–29.2% less time than OFF on the skewed cells and 0.2–1.5% more on the balanced ones, planning, copy and READY waits included.

### 4.3 End-to-End Serving Evaluation

#### 4.3.1 End-to-End Prefill Serving

Figure 10 shows the whole system serving prefill only: DeepSeek-V4-Pro (NVFP4) in TensorRT LLM with attention DP. For GSM8K, `trtllm-bench` measured OFF and ON request throughput at DEP8 with one output token, using the A/B/C runs reported in [PR #19646](https://github.com/NVIDIA/TensorRT-LLM/pull/19646). ON used three helper slots per rank and four auxiliary SMs. FB uses TensorRT LLM's [perfect router](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-analysis.md#moe-expert-load-balance-analysis-perfect-router) to force synthetic, perfectly balanced routing.

On GB200, ON raised request throughput by **6.9%**, from 2,285 to 2,444 requests/s, against FB's 7.8%, reaching **99.2% of FB's throughput** (Figure 10). On pre-release Rubin GPUs, ON gained **12.6%** against FB's 13.4%, reaching **99.3% of FB**; a second run on other nodes reached 99.1%. These serving measurements do not isolate multicast against a point-to-point copy baseline.

![Context-only GSM8K results at DEP8: OFF, ON and forced balanced routing on GB200 and pre-release Rubin GPUs; ON raises request throughput by 6.9% and 12.6% and reaches 99.2% and 99.3% of forced balanced-routing throughput.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_e2e_context_only.svg)

*Figure 10. End-to-end (E2E) context-only (prefill-only) serving of DeepSeek-V4-Pro (NVFP4) on GSM8K at DEP8 on GB200 and pre-release Rubin GPUs: request throughput normalized to OFF (load balancer off), with ON (ISER-MoE) and FB (force expert balance: OFF with routing forced to be perfectly balanced; its outputs are not meaningful).*

#### 4.3.2 End-to-End Disaggregated Serving on a GB300 NVL72 Rack

DeepSeek-V4-Pro (NVFP4) was also served with prefill and decode disaggregated ([Disaggregated Serving in TensorRT LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog05_Disaggregated_Serving_in_TensorRT-LLM.html)) on a GB300 NVL72 rack, in configurations C1–C4. Decoding used speculative decoding with three draft tokens and acceptance fixed at 2.01; clients replayed the agentic coding traces with [AIPerf](https://github.com/ai-dynamo/aiperf) for 1,200 s after warm-up. ISER-MoE ran on the prefill instances with three helper slots per rank, and key-value (KV) cache reuse was 94.6–95.2%, equal in both arms of every configuration.

In C3 and C4, ON kept replicas resident, as the merged implementation's default `per_layer_resident` mode does, and C4 also skipped replications that would move fewer than 256 routes, a threshold the merged implementation does not have. In C1–C3, the arms split the prefill instances and shared the decode instance; C4 ran each arm in its own job. Since serving varies between runs, each arm reports the best of repeated runs of one fixed configuration.

| Config | Prefill | Decode | Total GPUs | Concurrency | P90 latency gain | Output throughput/GPU gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C1 | 5 × DEP8 | DEP16 | 56 | 2,626 | 4.0% | 1.6% |
| C2 | 3 × DEP8 | DEP16 | 40 | 1,152 | 4.1% | 1.4% |
| C3 | 2 × DEP8 | DEP32 | 48 | 736 | 3.1% | 1.6% |
| C4 | 1 × DEP8 | DEP32 | 40 | 388 | 4.3% | 1.9% |

Latency gain is `OFF/ON − 1`; output throughput per GPU gain is `ON/OFF − 1`. Both are reported as percentages.

![Output throughput per GPU against P90 request latency for C1 to C4, each ON point above and left of its OFF point: P90 3.1–4.3% better and throughput 1.4–1.9% higher.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_e2e_disagg.svg)

*Figure 11. Disaggregated serving of DeepSeek-V4-Pro (NVFP4) on a GB300 NVL72 rack with agentic coding traces: output throughput per GPU against P90 request E2E latency, OFF and ON, C1–C4 in order of concurrency. Labels: OFF/ON − 1 for latency and ON/OFF − 1 for throughput (tput); each arm's best run.*

**All four configurations improved both request latency and output throughput per GPU (Figure 11).**

## 5. Using ISER-MoE in TensorRT LLM

Per-layer dynamic EPLB is a prototype feature: all four of its configuration fields carry prototype status. It copies helper weights GPU to GPU through the NVLink switch, so online EPLB's host shared-memory and GDRCopy setup does not apply.

### 5.1 Configuration

The [per-layer dynamic EPLB guide](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer#per-layer-dynamic-ep-load-balancer) gives this example for a model with 384 routed experts on 8 GPUs:

```bash
cat > ./config_per_layer_dynamic.yaml <<EOF
enable_attention_dp: true
cuda_graph_config: null
moe_config:
    backend: MEGAMOE_CUTEDSL
    load_balancer:
      mode: per_layer_dynamic
      num_slots: 416    # 384 experts + 4 helper slots on each of 8 EP ranks
      auxiliary_sms: 4  # optional; 4 is the default
      helper_slot_mode: per_layer_resident  # optional; the default
EOF
```

Pass the file with `--config` to `trtllm-bench` (with `--tp 8 --ep 8`) or `trtllm-serve` (with `--tp_size 8 --ep_size 8`), since `num_slots` assumes eight EP ranks; the `load_balancer` section can also be a path to a YAML file with the same fields, and omitting it turns load balancing off ([`num_slots` field](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/wide_ep/README.md#num_slots)).

- **`num_slots`** $=E+P\cdot S$: each rank keeps its $H=E/P$ home experts and gets $S$ helper slots. At EP8, 416 gives DeepSeek-V4-Pro $S=4$, and 408 gives $S=3$.
- **`helper_slot_mode`**: the default, `per_layer_resident`, gives every MoE layer its own helper slots (about 6.4 GiB per GPU for DeepSeek-V4-Pro with $S=3$, per the guide) and skips a broadcast when every member of its group already holds the expert in the assigned helper slot. `shared_banks` alternates all layers between two banks and copies every helper expert in every forward pass; for DeepSeek-V4-Pro at $S=4$, it cuts helper storage per GPU from 9.07 GB for 61 per-layer sets to about 297 MB (both computed).
- **`auxiliary_sms`** (default 4, range 1–8, below the GPU's SM count) is the number of SMs, one CTA each, of the kernel that runs HALO-Q and then the copy. Input quantization and, on the SM100 family, DeepSeek-V4's fused shared-expert kernel leave these SMs free, so raising the value takes SMs from that work.

### 5.2 Requirements and Constraints

| Item | Requirement |
| :--- | :--- |
| Backend | `MEGAMOE_CUTEDSL` only; `AUTO` resolves to it, and other explicit backends are rejected. |
| Model | An architecture in `moe_model_arch_list` with NVFP4 MoE weights, at most 384 routed experts, and an expert count divisible by the EP size; results here cover DeepSeek-V4-Pro only. |
| GPUs | Evaluated on GB200, GB300 and pre-release Rubin GPUs; the [per-layer dynamic EPLB guide](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer#per-layer-dynamic-ep-load-balancer) lists supported GPUs and build requirements. All EP ranks must be in one multicast-capable NVLink domain, with CUDA 13.1 or later for the TMA multicast copy. |
| Parallelism | Attention DP with pure EP (no MoE tensor or pipeline parallelism), an even EP size from 2 to 32, and no MoE smart router; unsupported setups raise an error. |
| CUDA graphs | Not supported: a non-null `cuda_graph_config` is replaced with `null`, with a warning, and any `prefill_cuda_graph_backend` other than `DISABLED` is rejected. |
| Launch queues | `CUDA_SCALE_LAUNCH_QUEUES=4x` before CUDA initializes: the LLM API behind `trtllm-bench` and `trtllm-serve` sets it; a process that initializes CUDA before creating its `LLM` instance, or that does not use the LLM API, such as the layer-wise benchmarks, must export it. |
| Other | Distributed Weight Data Parallelism ([DWDP](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.html)) is not supported. |

**Failure handling:** HALO-Q traps once a peer exceeds its spin bound ([MoE developer guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md#scheduler--eplb-constraints)), MegaMoE traps after 120 s at the READY gate, and there is no fallback to home experts.

### 5.3 Performance Measurement

**First measure the imbalance** with the expert statistics (`EXPERT_STATISTIC_PATH`, `EXPERT_STATISTIC_ITER_RANGE`) and `report_load_statistics.py` from the [offline EPLB guide](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer#step-1-run-inference-and-collect-statistics): its `imbalance-ratio` column, (max − mean)/mean of the ranks' routed tokens averaged over iterations, is the mean of $R-1$. In this mode, the statistics record expert IDs before HALO-Q, so they show the imbalance HALO-Q starts from.

**Then compare OFF and ON in serving:** OFF has no `load_balancer` section, and both use otherwise identical configurations with `cuda_graph_config: null`. OFF with `ENABLE_PERFECT_ROUTER=expert_ids` ([perfect router](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-analysis.md#moe-expert-load-balance-analysis-perfect-router)) adds a balanced-routing reference (not a strict ceiling); its model outputs are not meaningful.

**On a layer slice**, the [layer-wise benchmarks](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/layer_wise_benchmarks/README.md) run three DeepSeek-V4 arms: load balancer off, per-layer dynamic, and perfect router. The per-layer dynamic arm, run from `examples/layer_wise_benchmarks/` on one node with eight GPUs through the README's OpenMPI launcher:

```bash
NP=8 ./mpi_launch.sh -x MEGAMOE_COMBINE_FORMAT=16e2m1xbf16 -x CUDA_SCALE_LAUNCH_QUEUES=4x ./run.sh config_ctx.yaml \
  --model $DSV4 --layer-indices 7,8 --tokens-per-block 128 --batch-size 64 \
  --seq-len-q 128 --max-seq-len 512 --moe-backend MEGAMOE_CUTEDSL \
  --balance-method NotModified --moe-load-balancer eplb_per_layer_dynamic.yaml
```

Here `eplb_per_layer_dynamic.yaml` is a file you create in that directory with the two lines `mode: per_layer_dynamic` and `num_slots: 408`. `$DSV4` is a local checkpoint directory, and `MEGAMOE_COMBINE_FORMAT=16e2m1xbf16` selects the FP4 combine; `CUDA_SCALE_LAUNCH_QUEUES=4x` is exported because the layer-wise runner does not go through the LLM API, which sets it. Without `--replay-file-path`, repeated inputs leave the planned experts in the helper slots, so copies are skipped: a best case. On GB200 or GB300 NVL72 compute trays, which hold four GPUs each, run the same `./run.sh` arguments over two trays with the README's Slurm launcher (`NODES=2 NP=8 ./slurm_launch.sh`), setting the environment variables in front of it instead of with `-x`.

### 5.4 Applicability and Evaluated Configurations

- **Routing imbalance.** In the [kernel measurements](#423-megamoe-kernel-time-with-replicas) (GB200, planning and copy excluded), the time saved at EP8 grew with $R$, from 1.3% (layer 45, $R=1.12$) to 42.8% (layer 20, $R=2.36$), while balanced routing changed by −0.7% to +0.4%.
- **Shared-expert overlap.** Without one, only home tiles hide the copy: at most about one DeepSeek-V4-Pro-sized copy (GB300 peak-rate model, EP8, 8,192 tokens/rank), and none with the BF16 combine.
- **Prefill workloads.** The [parallelism guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md#best-practices) recommends this mode for context-heavy serving whose hot experts change from batch to batch, and online EPLB for generation-heavy deployments that rely on CUDA graphs; generation is untested here.
- **Evaluated configurations.** On hardware, EP4–EP8 for kernels and DEP8 for serving; in replay, EP4–EP24. No controlled hardware point used fewer than 8,192 tokens/rank.

## 6. Discussion

### 6.1 Lower-Bound Analysis of Hierarchical and Free Slot Usage

**Hierarchical and Free slot usage have nearly the same algorithmic load-balancing limit in the tested Zipf workloads.** Across EP8–EP512 and one to four helper slots per rank, the largest difference in the **64-layer mean optimal imbalance ratio is below 0.0041 in R**, or **0.41 percentage points of mean rank load**, at EP8 with one slot (Figure 13). This measures the best achievable balance under each placement rule, independently of the HALO-Q heuristic.

**Optimization problem.** For $m\in\{\mathrm{Free},\mathrm{Hierarchical}\}$, given expert-token counts $w_e$ and their owner ranks, choose binary replica placements $x$ and token assignments $a_{er}\in\mathbb Z_{\ge0}$ to minimize the largest final rank load $T$. With $\bar L=\sum_e w_e/P$, the objective and load constraints are

$$
R_m^*=\min_{T,a,x}\frac{T}{\bar L},\qquad
\sum_r a_{er}=w_e,\qquad
\sum_e a_{er}\le T.
$$

Tokens may execute only on an expert's owner or a rank holding its replica. Each rank may export at most $S$ distinct experts and hold at most $S$ helper replicas; no rank both exports home-expert load and processes other owners' tokens. The placement constraints distinguish the two models:

- **Free:** each exported expert may use any subset of receiver ranks.
- **Hierarchical:** each exported expert uses one aligned group of size $2,4,\ldots,P$. Every group member consumes one helper slot, including an owner inside the group or a member assigned zero tokens.

Every Hierarchical plan is also feasible under Free placement after removing unused replicas, so $R_{\mathrm{free}}^*\le R_{\mathrm{hier}}^*$. Figure 12 illustrates the placement rules; the small average gap above is specific to the tested workloads.

![Free selects receiver slots independently; Hierarchical places the same expert on both members of each selected aligned pair.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_slot_usage.svg)

*Figure 12. Example helper-slot placement on receiver ranks 0–3, with one slot per rank. Free selects receivers independently; Hierarchical places the same expert on both members of each selected aligned pair. Letters identify experts; owners are omitted.*

**Solving the bound.** Mixed-integer linear programming gives lower bounds on $R_m^*$, and validated integral plans give feasible upper bounds. Equal bounds certify the optimum; unresolved cases retain an interval. Figure 13 uses 1,536 experts, top-16 routing, 131,072 tokens per step, random contiguous ownership, and Zipf popularity calibrated to a coefficient of variation of two.

![Free and Hierarchical mean optimal-load bounds across EP8–EP512, with one through four helper slots per rank.](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog32_slot_bounds.svg)

*Figure 13. Mean optimal imbalance bounds over 64 layers per EP for S = 1, 2, 3 and 4. Lines and markers show feasible upper bounds; shading spans lower to upper bounds, and † marks unresolved cases. Each panel has its own y-axis scale.*

These are load-only optimization bounds: they exclude copy time and GEMM efficiency, and do not imply that a runtime planner attains the optimum.

### 6.2 Limitations

Beyond the [tested range](#54-applicability-and-evaluated-configurations):

- **Missing measurements.** On GB200, a layer with the real shared expert was not measured, and no hardware result compares ISER-MoE with a static plan; the 85–88% estimate comes from a model fit.
- **Alignment cost.** Every group member spends a helper slot and receives the whole expert: in [replay](#421-halo-q-trace-replay), HALO-Q made 2.5–3.3× as many remote copies at EP8–EP24 as the greedy point-to-point planner capped at four copies per owner, which balanced worse ($R$ = 1.016–1.100).
- **Replica lifetime.** HALO-Q plans each replica for one invocation, although in replay 74–78% of the all-rank policy's expert selections recur in the next iteration; the merged implementation's default `per_layer_resident` mode skips a broadcast only when every member of its group still holds the expert in the helper slot HALO-Q assigns, and slot coloring does not try to keep an expert in its previous slot.
- **Static plus dynamic, and arrival skew.** Per-layer dynamic EPLB cannot currently start from an offline EPLB map, and placement cannot fix ranks that reach the layer at different times.

### 6.3 Related Work

- **Static plans from load history** complement per-invocation replication: [DeepSeek-V3](https://arxiv.org/abs/2412.19437)'s redundant prefill experts, DeepSeek's [EPLB](https://github.com/deepseek-ai/EPLB), TensorRT LLM's online EPLB, [vLLM v0.23.0](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)'s EPLB for DeepSeek-V4 Mega-MoE and the Linear-Programming-Based Load Balancer ([LPLB](https://github.com/deepseek-ai/LPLB)).
- **Per-invocation replication** copies replica weights point to point in [UltraEP](https://arxiv.org/abs/2606.04101), [MoonEP](https://github.com/MoonshotAI/MoonEP) (from the [Kimi K3 report](https://arxiv.org/abs/2607.24653)), the concurrent [MegaFlux](https://arxiv.org/abs/2610.00671), Elastic Cloning for Hot Experts (ECHO) in [Megatron Core](https://arxiv.org/abs/2603.07685) and [DeepEP](https://github.com/deepseek-ai/DeepEP) V2.5; ISER-MoE multicasts them into fixed groups, as [FasterMoE](https://doi.org/10.1145/3503221.3508418) broadcasts to one fixed group.
- **Hiding copies and multi-port injection.** [FEPLB](https://arxiv.org/abs/2604.19654), Topology-Aware Optimal Transport ([TAOT](https://arxiv.org/abs/2608.03676)) and [PROBE](https://arxiv.org/abs/2602.00509) hide replica copies behind computation, and NCCL's NVLS AllGather and TensorRT LLM's [MultiShot](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/) all-reduce already inject from every member's port, as SAMI does for weights.

Reported gains are not directly comparable: MegaFlux's 1.45× kernel-level mean speedup on eight B200 GPUs includes planning and transfers, which the 1.21–1.52× [MegaMoE speedup](#423-megamoe-kernel-time-with-replicas) above excludes, and UltraEP reports 1.29× the prefill throughput of SGLang with EPLB at EP40–EP64.

## 7. Conclusion and Further Reading

### 7.1 Conclusion

Per-invocation replicas of hot experts can remove most of the rank imbalance in expert-parallel MoE layers, but only if their weights arrive within the layer. **ISER-MoE multicasts hot experts through aligned NVLink multicast groups created at initialization, so each byte leaves its owner once.** HALO-Q plans balance, then locality, on the GPU; SAMI spreads the injection over the group's ports; and planning and replication overlap the shared expert, so only helper tiles wait for their weights.

The generation phase, training, attention-side imbalance, replication across NVLink domains and a hardware comparison with static plans remain future work. To try ISER-MoE, [enable](#5-using-iser-moe-in-tensorrt-llm) `moe_config.load_balancer.mode: per_layer_dynamic` and compare it against OFF on your own traffic.

### 7.2 Further Reading

Using per-layer dynamic EPLB:

- [Per-layer dynamic EPLB guide](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer#per-layer-dynamic-ep-load-balancer): how the mode works, an example configuration and its constraints.
- [Wide-EP `mode` field](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/wide_ep/README.md#mode) and [layer-wise benchmarks](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/layer_wise_benchmarks/README.md): choosing the mode, and comparing it on a DeepSeek-V4 layer slice.

Implementation:

- [PR #19646: Add DynamicEPLB kernels and runtime integration](https://github.com/NVIDIA/TensorRT-LLM/pull/19646), and the [MoE developer guide's scheduler and EPLB constraints](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md#scheduler--eplb-constraints).
- [PR #21373: HALO-Q planner and aligned-pair replication extension](https://github.com/NVIDIA/TensorRT-LLM/pull/21373): two-rank placement groups, updated planning, and pair copy paths.
- [Per-layer dynamic EPLB runtime](https://github.com/NVIDIA/TensorRT-LLM/tree/4c2238ee524c65dd25259274debf1b3bfe8181e9/tensorrt_llm/_torch/moe/fused_moe/per_layer_dynamic_eplb) and [HALO-Q and replication kernels](https://github.com/NVIDIA/TensorRT-LLM/tree/4c2238ee524c65dd25259274debf1b3bfe8181e9/cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb), including SAMI.

Earlier TensorRT LLM posts: Scaling Expert Parallelism [Part 1](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html), [Part 2](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog08_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html) and [Part 3](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html), [One-Sided AlltoAll over NVLink](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html), and [DeepSeek-V4 on NVIDIA Blackwell](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.html).
