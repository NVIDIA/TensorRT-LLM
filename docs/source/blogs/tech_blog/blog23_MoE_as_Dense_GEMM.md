# MoE as Dense GEMM: Optimizing Low-Latency MoE Inference on NVIDIA Blackwell

by NVIDIA TensorRT LLM team

## Abstract

For min-latency inference, large-scale MoE models (DeepSeek-V3/R1, Mixtral, Qwen3-MoE, etc.) typically adopt a pure TP (Attention TP + MoE TP) parallelization strategy, with one AllReduce on each side of the MoE module. When the batch is small (`num_tokens` in the tens or single digits), several intra-MoE costs dominate end-to-end layer latency: TopK and Router GEMM exposed on the critical path, low hardware utilization of grouped GEMMs at small N/K, and the difficulty of overlapping the shared expert with the routed experts. Combined, these can account for more than 50% of the per-layer latency.

This post introduces **DENSEGEMM**, a new MoE backend in TensorRT-LLM for NVFP4 MoE on NVIDIA Blackwell (SM100 / SM103). The core idea is: **in the memory-bound low-latency regime, restructure the per-expert grouped GEMM into a single dense GEMM over all experts, trading the redundant FLOPs available in a memory-bound regime for the elimination of structural overheads such as dispatch, permute, finalize, and short-N/K grouped GEMMs.** We use a roofline analysis to derive when this trade-off holds, describe the CuTe DSL kernel implementation, and report measured performance on a DeepSeek-V3 configuration.

<p align="center">
<img src="../media/tech_blog23_Picture1.png" alt="TP MoE critical path" width="700"/>
</p>

## Table of Contents

- [Background: MoE Inference Latency Regimes](#background-moe-inference-latency-regimes)
- [DENSEGEMM Design](#densegemm-design)
- [Advantages over the Grouped-GEMM Flow](#advantages-over-the-grouped-gemm-flow)
- [Roofline Analysis](#roofline-analysis)
- [Kernel Implementation (CuTe DSL on Blackwell)](#kernel-implementation-cute-dsl-on-blackwell)
- [Performance Evaluation](#performance-evaluation)
- [Future Work](#future-work)
- [References](#references)

## Background: MoE Inference Latency Regimes

Partitioning MoE inference workloads by `num_tokens` (the number of tokens entering the MoE module per forward) clarifies which optimization strategy applies in which regime. Using DeepSeek-V3 (`num_experts = 256`, `top_k = 8`) as a reference:

| Regime | num_tokens | Physical characteristics |
|---|---|---|
| **Min-Latency** | ≤ 32 | Some experts receive 0 tokens; the rest receive very few. All experts are deeply memory-bound. |
| **Low-Latency** | 32 – 320 | Most experts receive at least one token, but per-expert token counts remain small. All experts remain memory-bound. |
| **Throughput at Latency** | 320 – 10k | Some experts cross into compute-bound while others stay memory-bound; `num_tokens × top_k / num_experts ≈ 1`. |
| **Max Throughput** | > 10k | All experts are compute-bound; grouped GEMM approaches peak hardware utilization. |

DENSEGEMM primarily targets the **Low-Latency** regime and partially covers the boundaries of Min-Latency and Throughput-at-Latency. This regime corresponds to most latency-sensitive online inference workloads, including Multi-Token Prediction (MTP), single-request reasoning chains, and real-time chat.

## DENSEGEMM Design

DENSEGEMM restructures the FC1 / FC2 of routed experts and shared expert into a single dense GEMM. The design comprises four points.

### (1) Concatenate all experts along N / K in FC1 / FC2

- **FC1**: All experts' W1 / W3 weights are concatenated along N into a single weight matrix of shape `[hidden, (num_experts + 1) × 2 × intermediate]`. All tokens go through one `x @ Wᵀ`. W1 / W3 are interleaved in groups of 64 within N so SwiGLU can be fused in the epilogue. Each expert has its own output scale `alpha`, fused as a *block-wise scaling along N* in the epilogue.
- **FC2**: All experts' W2 are concatenated along K into a single weight matrix of shape `[(num_experts + 1) × intermediate, hidden]`. The FC1 output flows directly along K. Each expert occupies a 256-element block along K with its own `alpha`, fused as a *block-wise scaling along M & K* in the mainloop.

### (2) Shared expert merged into the same dense GEMM

In the DeepSeek family, the shared expert is a fixed FFN that every token must pass through, with the same shape as a routed expert. DENSEGEMM treats it as the `num_experts`-th "additional expert" (i.e. 256 + 1 = 257 columns) appended to the end of the dense weight matrix. The alpha generation rule is adjusted accordingly:

- **Shared expert column** (index = `num_experts`): alpha is constantly 1.0; every token activates it.
- **Routed expert columns** (index = 0 … `num_experts - 1`): alpha is `topk_score` if the token selects the expert, otherwise 0.

The shared expert and routed experts now share the same dense GEMM mainloop and need no separate scheduling — they are simply different slices along the N / K dimensions of the same weight matrix.

### (3) Skip-expert via alpha masking, no dispatch needed

DENSEGEMM does not rely on token routing for skip-expert. The `gen_fc2_alpha_fused` kernel scatters `(token_selected_experts, token_final_scales)` into an alpha buffer of shape `[num_tokens, num_experts + 1]`; positions corresponding to non-selected routed experts remain 0. The dense GEMM multiplies in alpha at the epilogue / mainloop, achieving multiply-by-zero masking.

In the **TP scenario**, this step does not directly save dispatch / combine (which do not exist in the TP path) but it is the necessary substrate for densification — it converts "which experts are selected" from *out-of-kernel data movement* into *in-kernel alpha multiplication*, enabling the dense concatenation in (1). In the **EP scenario**, the same mechanism additionally eliminates dispatch, local-reduction, and combine, providing extra benefit for EP deployments.

### (4) Router GEMM, TopK, and gen_alpha overlap with FC1

Router GEMM (projects `hidden` to `num_experts` scores), TopK, and gen_alpha execute serially within a single module, but the entire router-side module has no data dependency on FC1 — FC1 only consumes `hidden_states`, while alpha is consumed only at the epilogue. DENSEGEMM schedules the router-side module on an aux stream in parallel with FC1 on the main stream. As long as FC1's runtime exceeds the router-side module's runtime (the common case), the latter is fully hidden.

### Overall Data Flow (TP Scenario)

The figure below illustrates the data flow in the TP scenario. The two AllReduces flanking the MoE module are dictated by TP itself; DENSEGEMM does not modify communication. Inside the MoE module, the critical path collapses to *Quantize → FC1 → FC2*, while the router-side module runs on an aux stream in parallel with FC1.

<p align="center">
<img src="../media/tech_blog23_Picture2.png" alt="DENSEGEMM TP data flow" width="800"/>
</p>

The shared expert no longer exists as a standalone FFN; it is merged into the same dense GEMM as a slice along N / K.

## Advantages over the Grouped-GEMM Flow

Comparing the design above against the conventional grouped-GEMM TP path yields four key advantages:

1. **TopK and Router GEMM removed from the critical path.** In the conventional flow these two kernels sit immediately before FC1 and are fully exposed on the critical path. DENSEGEMM moves them, together with gen_alpha, to an aux stream that runs in parallel with FC1. As long as FC1 outweighs the router-side module, Router GEMM and TopK are fully hidden.

2. **Dispatch / permute removed.** A grouped-GEMM kernel requires tokens belonging to the same expert to be physically contiguous, hence a token-by-expert permutation must precede it. DENSEGEMM runs a single dense GEMM over all tokens × all experts; no token-to-expert reordering is required, and the dispatch / permute kernel is fully removed.

3. **Finalize removed.** The conventional flow requires a finalize kernel after FC2 to scatter the per-expert-permuted output back to its original token order and combine the partial results from `top_k` experts via weighted sum. DENSEGEMM produces token-major final output directly — no finalize, no unpermute, no weighted sum — because that work is already done inside the FC2 mainloop's alpha multiply and K-reduce.

4. **FC1 has a larger N, FC2 has a larger K, leading to higher GEMM efficiency.** The conventional grouped-GEMM splits tokens across experts; FC1's N is only `2 · intermediate` and FC2's K is only `intermediate`, often unable to fill MMA tiles in min-latency regimes. DENSEGEMM concatenates all experts at once, growing FC1's N to `(E + 1) · 2 · intermediate` and FC2's K to `(E + 1) · intermediate`. The dense GEMM saturates the tcgen05 / TMA pipeline, achieving roughly 15% higher SOL% than a grouped GEMM of the same shape.

The table below contrasts the two flows at the level of kernel sequence on the critical path. DENSEGEMM collapses the intra-MoE critical path to **Quantize → FC1 → FC2**:

| Stage | Grouped-GEMM TP Flow | DENSEGEMM TP Flow |
|---|---|---|
| Before entering MoE | AllReduce (from attention) | AllReduce (from attention) |
| Pre-MoE | Router GEMM → TopK → quantize | quantize (Router GEMM / TopK / gen_alpha overlapped with FC1 on aux stream) |
| Token-to-Expert | **Dispatch / Permute** | — |
| MoE compute | grouped FC1 → activation → grouped FC2 | dense FC1 (+ SwiGLU + alpha_post) → dense FC2 (+ alpha_scale) |
| Post-MoE | **Finalize / Unpermute / weighted sum** | — |
| Shared expert | Standalone FFN (separate stream, may be exposed) | Merged into the N / K concatenation of the dense FC1 / FC2 |
| Leaving MoE | AllReduce | AllReduce |

The combination of these four advantages is the root cause of DENSEGEMM's consistent lead over the grouped-GEMM TP path in the `num_tokens ∈ [48, 256]` range, as shown in the performance section below.

## Roofline Analysis

DENSEGEMM appears to introduce redundant compute: only `top_k = 8` experts are actually selected, yet the weights of all `E = 256` experts are loaded and multiplied, so the share of "wasted" arithmetic is `1 - top_k / E ≈ 96.9%`. We use a roofline analysis to derive the regime in which this trade-off holds, and the corresponding `num_tokens` range.

### Step 1: Notation

Below are the per-rank symbols (i.e. dimensions after TP / EP partitioning):

| Symbol | Meaning |
|---|---|
| `M` | num_tokens (number of tokens entering the MoE module per forward) |
| `E` | per-rank number of routed experts |
| `k` | top_k (experts activated per token, typically 4 – 8) |
| `H` | hidden size (per-rank, after TP) |
| `I` | intermediate size (per-rank) |
| `b` | bytes per weight element (NVFP4 ≈ 0.5) |
| `BW` | HBM bandwidth (B200 ≈ 8 TB/s) |
| `P` | NVFP4 dense peak FLOPs (B200 ≈ 10 PFLOPS) |

Total weight bytes per expert (W1 + W2 + W3 in SwiGLU form):

```
B_exp = (2·H·I + H·I) · b = 3·H·I·b
```

### Step 2: The Low-Latency Regime is Memory-Bound

The arithmetic-intensity ridge point at NVFP4 dense FLOPs and HBM bandwidth on Blackwell B200 is:

```
AI_crit = P / BW = 10e15 / 8e12 = 1250  FLOPs / Byte
```

The arithmetic intensity of dense FC1 over `M` tokens and `(E + 1)` experts (we ignore activation bytes since weight bytes dominate):

```
FLOPs_FC1 = 2 · M · (E·2I) · H
Bytes_FC1 = (E+1) · (2·H·I) · b
AI_FC1   ≈ 2·M·E·2I·H / (E·2I·H·b) = 2M / b
```

Plugging in NVFP4's `b = 0.5` gives `AI_FC1 ≈ 4M`. **Whether FC1 is memory-bound depends solely on M**, independent of `hidden`, `intermediate`, or `num_experts`.

Memory-bound condition:

```
AI_FC1 < AI_crit  ⇔  4M < 1250  ⇔  M < ~312
```

The same analysis gives `AI_FC2 ≈ 4M`, with the same ridge point of ~312. Hence: **for M ≲ 312, both FC1 and FC2 are entirely memory-bound**, and the redundant compute introduced by densification incurs no real latency cost. We thus obtain the roofline upper bound:

> **Roofline upper bound: M ≲ 312 (B200 NVFP4 dense, 10 PFLOPS / 8 TB/s).** Beyond this, the 96.9% redundant compute introduced by densification begins to translate into actual latency cost.

The empirical FC1 memory→compute crossover is at M ≈ 336 (see the roofline table in the performance section), in close agreement with the theoretical ridge point of 312. The empirical value is slightly higher because dense GEMM still sustains around 80% SOL% near the crossover, effectively pushing the equivalent-FLOPs ceiling slightly back.

## Kernel Implementation (CuTe DSL on Blackwell)

DENSEGEMM consists of two CuTe DSL kernels, both registered as `torch.library.custom_op`. The Python-side entry is `fused_moe_densegemm.py:run_moe_nvfp4`, with kernel bodies under `cute_dsl_kernels/blackwell/moe_as_dense_gemm/{fc1,fc2}.py`. Both share `Sm100BlockScaledPersistentDenseGemmKernel` — a dense GEMM template using persistent CTA, TMA load, and tcgen05 MMA — on top of which expert-aware alpha fusion is implemented separately.

| | FC1 | FC2 |
|---|---|---|
| Custom op | `cute_dsl_nvfp4_dense_gemm_swiglu_blackwell` | `cute_dsl_nvfp4_dense_gemm_fc2_blackwell` |
| Math form | `C = SwiGLU(α · X @ Wᵀ) · α_post` | `C = (α_scale · A) @ B` |
| Expert concat dim | N (per-expert tile = `2 · intermediate`) | K (per-expert tile = `intermediate`, 256-aligned) |
| Alpha fusion site | **Epilogue**, *block-wise along N* | **Mainloop**, *block-wise along M & K* |
| AutoTune space | MMA `[(128,128),(128,256),(256,256)]`, cluster `[(1,1),(1,2),(1,4),(2,1)]` | MMA `[(128,64),(128,128),(128,256)]`, cluster `[(1,1),(1,2),(1,4)]` |

Notes:

- **FC1 epilogue fusion**: each N tile knows which expert it belongs to and multiplies in `alpha_post[:, expert_id]` after SwiGLU. Since W1 / W3 are interleaved within 64-groups along N, SwiGLU can fetch `(gate, up)` and compute the activation in the epilogue without bank conflicts.
- **FC2 mainloop fusion**: every 256 elements along K belong to one expert; before each MMA K-block enters the tensor cores, `alpha_scale[:, expert_id]` is multiplied in. Non-activated experts have alpha = 0, equivalent to multiply-by-zero masking. This requires `intermediate_size` to be a multiple of 256.
- **Weight layout**: W2 is transposed from expert-major `(E, H, I)` to hidden-major `(H, E·I)`; the weight scale undergoes a `_transform_w2_weight_scale_for_min_latency` rearrangement so that the 256-aligned K scales can be loaded into SMEM by TMA in one shot and fed directly to the mainloop alpha multiply.
- **Optional fusion `TRTLLM_MOE_FUSED_FC2_ALPHA`**: pre-multiply the per-token-per-expert alpha of FC2 (after normalization) into FC1's `alpha_post`, reducing FC2 to a regular NVFP4 GEMM with scalar alpha (callable via `nvfp4_gemm` with cutlass / cublaslt / cutedsl backends). This further reduces latency on certain shapes and is on by default.

## Performance Evaluation

### Setup

- **GPU**: single NVIDIA B200 (SM100)
- **Workload profile**: DeepSeek-V3 configuration with `hidden = 7168`, `num_experts = 256`, `top_k = 8`, activation = SwiGLU, precision = NVFP4. `intermediate` may differ across comparison entries to reflect the per-rank shard size after TP partitioning.
- **Object under measurement**: per-rank MoE-module compute latency under TP deployment, i.e. router-side module (Router GEMM + TopK + gen_alpha) + FC1 + FC2 + necessary quantize. The two flanking AllReduces are excluded since they are identical across grouped and DENSEGEMM paths.
- **Comparison points**: DENSEGEMM (this work, dense path) vs TRTLLM-Gen (per-expert grouped GEMM, TP) vs CUTLASS (reference). None contain dispatch / local-reduce / combine, consistent with the pure-TP scenario.

### End-to-End MoE Module Latency

The table below reports latency for representative `num_tokens` values. The `Speedup` column is `t_TRTLLM-Gen / t_DENSEGEMM`; values greater than 1 indicate DENSEGEMM is faster.

| num_tokens | DENSEGEMM (µs) | TRTLLM-Gen (µs) | CUTLASS (µs) | DENSEGEMM vs TRTLLM-Gen |
|---:|---:|---:|---:|:---:|
| 1   | 153.62 | 30.32  | 109.44 | 0.20× |
| 8   | 144.44 | 56.08  | 156.41 | 0.39× |
| 16  | 145.34 | 80.28  | 202.66 | 0.55× |
| 32  | 137.98 | 113.35 | 259.59 | 0.82× |
| **48**  | **133.41** | 149.10 | 302.68 | **1.12×** |
| **64**  | **134.18** | 155.28 | 324.80 | **1.16×** |
| **128** | **136.16** | 176.79 | 344.26 | **1.30×** |
| **256** | **162.88** | 189.93 | 347.44 | **1.17×** |
| 272 | 221.21 | 191.64 | 346.75 | 0.87× |
| 336 | 225.74 | 193.54 | 354.81 | 0.86× |
| 512 | 263.12 | 202.36 | 361.31 | 0.77× |

Observations:

- **`num_tokens ∈ [48, 256]` is DENSEGEMM's sweet spot**, with a stable 12% – 30% lead over TRTLLM-Gen and a 2× – 3× lead over CUTLASS across the whole range.
- **DENSEGEMM is not advantageous at `num_tokens ≤ 32`.** The current implementation always loads weights of all 256 experts; at small batch the weight-load time is roughly fixed at ~145 µs, while TRTLLM-Gen only loads the 8 selected experts. This gap motivates the Min-Latency optimization in Future Work.
- **TRTLLM-Gen overtakes DENSEGEMM at `num_tokens ≥ 272`** for two reasons: (a) the FC1 best config switches to `128×256 @ (1,2)` and latency steps up (96 µs → 144 µs); (b) the workload approaches the compute-bound crossover, and the densification's redundant compute starts paying for itself.

### Internal Kernel Breakdown (num_tokens = 128)

| Component | Latency (µs) |
|---|---:|
| `deepseek_v3_topk_kernel` | 4.17 |
| FC1 (`densegemm_fc1`)     | 82.97 |
| FC2 (`densegemm_fc2`)     | 48.06 |
| `quantize_with_block_size` | 3.66 |
| Misc (gen_alpha, etc.)    | 1.47 |
| **Total**                 | **136.16** |

FC1 + FC2 sums to 131 µs — 96% of the total. TopK is overlapped on the aux stream, with only 4.17 µs visible on the critical path.

### FC1 Roofline

The table below reports, for each `num_tokens`, the (MMA tile, cluster) chosen by the AutoTuner and its comparison to the roofline. `xRoofline` is the measured latency divided by the roofline lower bound; `Eff_BW%` is the achieved fraction of peak HBM bandwidth.

| num_tokens | Roofline (µs) | Bound | Best Config | Latency (µs) | xRoofline | Eff_BW (TB/s) | Eff_BW% |
|---:|---:|---|---|---:|---:|---:|---:|
| 1   | 66.1 | Memory  | 128×256 @ (1,4) | 97.90  | 1.48× | 5.40 | 67.5% |
| 32  | 66.2 | Memory  | 128×128 @ (1,4) | 87.47  | 1.32× | 6.06 | 75.7% |
| **48**  | 66.3 | Memory  | **128×128 @ (1,1)** | **82.70** | **1.25×** | **6.41** | **80.2%** |
| 128 | 66.7 | Memory  | 128×128 @ (1,1) | 82.97  | 1.24× | 6.43 | 80.4% |
| 256 | 67.4 | Memory  | 256×256 @ (2,1) | 98.23  | 1.46× | 5.49 | 68.6% |
| **336** | 70.2 | **Compute** | 128×256 @ (1,2) | 145.38 | 2.07× | 3.73 | 46.6% |
| 512 | 106.9 | Compute | 256×256 @ (2,1) | 163.04 | 1.53× | 3.37 | 42.1% |

Two key crossovers:

- **`num_tokens = 48`**: enters the memory-bound sweet spot. FC1 reaches 80% of HBM peak bandwidth and is only 1.25× off the roofline lower bound, consistent with the end-to-end advantage over TRTLLM-Gen in this regime.
- **`num_tokens = 336`**: crosses from memory-bound to compute-bound. xRoofline jumps above 2×, Eff_BW% drops to 46% and continues to decline. Beyond this point, the redundant compute introduced by densification ceases to be free.

## Future Work

DENSEGEMM currently covers the Low-Latency regime. Three directions are planned:

1. **Min-Latency: skip zero-load experts.** For `num_tokens ≤ 32`, dynamically pick the subset of experts that are actually selected and only load / compute those weights. This addresses the ~5× gap at `num_tokens ≤ 32` versus TRTLLM-Gen and is the key to extending DENSEGEMM into the Min-Latency regime.
2. **Full TP/EP coverage in Low-Latency.** The current implementation assumes a specific TP / EP split for routed experts. Future work will adapt DENSEGEMM to arbitrary TPxEPy combinations, supporting deployments from 8×B200 to GB200 NVL72.
3. **Throughput at Latency: chunked MoE.** In the 320 – 10k tokens range, partition the grouped GEMM by experts into multiple chunks: chunks that still fall within the Low-Latency regime (≤ 320 tokens) run with dense GEMM; compute-bound chunks remain on the grouped path. Inter-chunk dispatch / collect is overlapped with MoE-AlltoAll, preserving latency without sacrificing throughput.

## References

- Source code: [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_densegemm.py`](../../../tensorrt_llm/_torch/modules/fused_moe/fused_moe_densegemm.py)
- Backend selection: [`tensorrt_llm/_torch/modules/fused_moe/create_moe.py`](../../../tensorrt_llm/_torch/modules/fused_moe/create_moe.py)
- FC1 kernel: [`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/moe_as_dense_gemm/fc1.py`](../../../tensorrt_llm/_torch/cute_dsl_kernels/blackwell/moe_as_dense_gemm/fc1.py)
- FC2 kernel: [`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/moe_as_dense_gemm/fc2.py`](../../../tensorrt_llm/_torch/cute_dsl_kernels/blackwell/moe_as_dense_gemm/fc2.py)
- Custom op registration: [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`](../../../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py)
- Companion blogs: [blog18 — Optimizing MoE Communication with One-Sided AlltoAll over NVLink](./blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md); [blog04 — Scaling Expert Parallelism in TensorRT-LLM](./blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
