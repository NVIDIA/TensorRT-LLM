---
orphan: true
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GVR V2: Benchmark Methodology and Figure Reproduction

This companion to [GVR V2: Faster Exact Top-K with Self-Sampling and Multi-Thresholding](../../tech_blog/blog29_GVR_V2_Self_Sampling_Exact_TopK_for_Sparse_Attention.md) contains the observations and definitions needed to reproduce the article's figures. The article focuses on the algorithm and its performance; this document records the measurement boundaries.

## Regenerate the Figures

With NumPy and Matplotlib installed, run from the repository root:

```bash
python docs/source/blogs/media/gvr_v2/plot_results.py
```

The script regenerates `summary.json` and eight SVGs: `speedup.svg`, `evolution.svg`, `algorithm.svg`, `sglang_map.svg`, `deepselect_map.svg`, `latency.svg`, `roofline.svg`, and `integration.svg`. It requires no GPU. The algorithm diagrams are schematic; every performance panel uses the bundled timing observations.

## Published Data

| File | Contents |
| :--- | :--- |
| [flash_timings.csv.gz](flash_timings.csv.gz) | 2,079 DeepSeek-V4 Flash cases |
| [pro_timings.csv.gz](pro_timings.csv.gz) | 2,970 DeepSeek-V4 Pro cases |
| [v32_timings.csv.gz](v32_timings.csv.gz) | 4,697 DeepSeek-V3.2 cases |
| [temporal_comparison.csv.gz](temporal_comparison.csv.gz) | Two temporal GVR observations for each of the same 9,746 cases |
| [provenance.json](provenance.json) | Published-file checksums, implementation labels, pairing, and roofline constants |
| [summary.json](summary.json) | Recomputed statistics |
| [plot_results.py](plot_results.py) | Figure generation |

The CSVs begin with a copyright comment. `cell`, `model`, `isl_bucket`, and `layer` identify a workload. `batch`, `n`, and `k` specify its dimensions. Columns ending in `_us` contain case-level mean kernel durations in microseconds. Join the temporal supplement by `(cell, batch)`. Blank entries indicate missing or unsupported measurements, never zero latency.

The files contain kernel timings and workload dimensions. They do not contain input scores, prompt contents, or individual timing repetitions. They reproduce the published statistics and charts; repeating the GPU experiment requires suitable score inputs and a benchmark harness.

## Measurement and Comparison Scope

Measurements use NVIDIA B200, FP32 indexer scores, and batch sizes from 1 to 1,024. Each case has 10 warmup calls, five warm-L2 repetitions, and 10 cold-L2 repetitions. The article reports mean GPU kernel duration from the cold repetitions. A 512 MB cache eviction runs outside the timed region. Compilation, input preparation, allocation during setup, and Python launch overhead are excluded; required device kernels remain timed.

The primary reference is the hint-free GVR V2 `run_varlen` implementation. DeepSelect FP32 and HPC-ops were measured in the same process as that reference. Radix CUDA, SGLang, FlashInfer, and the temporal GVR implementations use matched observations from separate runs. These comparisons describe the bundled implementations and workloads; they are not a current-release, equal-interface benchmark of entire serving frameworks.

| Implementation | Relevant comparison contract |
| :--- | :--- |
| GVR V2 | FP32 scores, valid row lengths, unordered INT32 indices |
| SGLang Top-K v2 | Main comparison includes plan + transform; transformation-only results are separate |
| FlashInfer 0.6.14 | Native `top_k` returns FP32 values and INT64 indices and scans a padded row |
| TensorRT-LLM radix CUDA | Production dispatcher, including short-row insertion and long-row split-work paths |
| DeepSelect v1.0.0 | Unsorted INT32 indices only; FP32 K=2048 emphasizes correctness coverage |
| HPC-ops FP32 | K=512 or 2048, with recommended workspace; K=1024 is unsupported |

The public baseline revisions for DeepSelect and HPC-ops are [8e70df71d2](https://github.com/deepseek-ai/DeepSelect/tree/8e70df71d2) and [2a2e265624](https://github.com/Tencent/hpc-ops/tree/2a2e265624). Complete build revisions for the historical SGLang and radix observations are unavailable in the timing export.

SGLang planning can be amortized across layers in a serving integration. FlashInfer's additional outputs and padded-row scan remain part of its timed native API. DeepSelect and HPC-ops receive preallocated output or workspace. The BF16 comparison uses its own paired `gvr_bf16_run_us` reference and preconverted BF16 input for DeepSelect; conversion time is excluded. Each dtype is checked against its own `torch.topk` result, so BF16 speed does not establish preservation of FP32 Top-K membership.

## Temporal GVR and Algorithm Evolution

The temporal R0 and tiered implementations correspond to public [PR #16457](https://github.com/NVIDIA/TensorRT-LLM/pull/16457) and [PR #16877](https://github.com/NVIDIA/TensorRT-LLM/pull/16877). They already include improvements beyond original scalar-search V1, including a threshold ladder and execution specialization. The original scalar-search implementation has no measurement on this grid. The temporal comparisons span full implementations, including scheduling and integration; they are not an isolated self-sampling ablation.

Figure 2 computes each implementation's speedup directly from its paired radix times. The geometric means are 2.586546× for temporal R0, 3.471858× for tiered temporal GVR, and 4.929143× for V2. Direct temporal/V2 time ratios are 1.905685× and 1.419742×. `summary.json` records these under `evolution_vs_radix` and `temporal_vs_v2`. Historical R0 observations for V3.2 lack explicit N/K metadata; workload identity supplies the join for those records.

## Aggregation and Coverage

Speedup is the geometric mean of per-case `baseline_us / gvr_v2_us` ratios. Every case has equal weight. A win is a ratio above one; minima and percentiles also use individual ratios. No slower case is discarded.

Figure 1 intersects all supported implementations within each model: 2,079 Flash, 2,970 Pro, and 4,466 V3.2 cases. V2 is fixed at 1.00; shorter bars mean less time. Pro omits unsupported HPC-ops. These values are recorded under `comparison_common_cases`. The article's overall and per-model tables use each baseline's full paired coverage.

SGLang and FlashInfer lack V3.2 layers 0–2 and cover 9,515 cases in total. HPC-ops covers 6,776 Flash/V3.2 cases. GVR, radix, and DeepSelect cover all 9,746 cases. Missing coverage is never filled with estimated timings.

The latency and roofline curves use arithmetic-mean durations over matching layers at each row-length/batch point: 21 layers for Flash, 30 for Pro, and 58 for V3.2. All layers at each plotted point have the same valid width. The SGLang and DeepSelect FP32 heatmaps (Figures 4 and 5) instead geometrically average per-layer speedups at each shape, using each baseline's full paired coverage. DeepSelect therefore includes all 61 V3.2 layers; SGLang includes 58. Both maps share a 0.8–8.0 scale with parity at 1.0, and cell labels round to one decimal place. A shape average can hide individual regressions.

Correctness compares selected value multisets with `torch.topk`, allowing tied indices to differ. The capture-grid checks do not establish NaN ordering parity or a universal tie order. The linked implementation PRs additionally cover padding, variable lengths, exceptional values, and graph replay.

## Additional Numerical Views

The article uses Figure 1 for the model-level comparison. The following table retains each baseline's full paired coverage; Figure 1 instead uses the common intersection within each model.

| Baseline | V4 Flash, $K=512$ | V4 Pro, $K=1024$ | V3.2, $K=2048$ |
| :--- | ---: | ---: | ---: |
| SGLang v2, plan + transform | 1.78× | 1.76× | 1.55× |
| FlashInfer 0.6.14 | 2.12× | 2.14× | 1.89× |
| TensorRT-LLM radix CUDA | 4.74× | 4.73× | 5.15× |
| DeepSelect FP32 | 1.98× | 2.07× | 2.79× |
| HPC-ops FP32 | 2.30× | Unsupported | 1.30× |

*Each model column uses the workloads supported by that baseline.*

For a concrete large-batch slice, the following times are at $B=1024$ and $N\approx131{,}072$, averaged over the same layers as Figure 6:

| Model | GVR V2 | SGLang | FlashInfer | Radix CUDA | DeepSelect FP32 | HPC-ops FP32 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 Flash | **113.1 µs** | 196.8 µs | 275.4 µs | 461.3 µs | 190.5 µs | 199.2 µs |
| V4 Pro | **132.8 µs** | 199.1 µs | 298.3 µs | 477.7 µs | 209.1 µs | — |
| V3.2 | **124.0 µs** | 211.0 µs | 388.8 µs | 496.6 µs | 419.6 µs | 159.7 µs |


## Roofline Definitions

The logical work is `W = B*N` abstract comparisons and the minimum traffic is `Q = 4*B*(N+K)` bytes. Operational intensity is `I = W/Q`; measured useful throughput is `P = B*N/(time_us*1e6)` Tcompare/s. Every kernel uses this same normalization. Extra outputs, padding, staging, repeated scans, and synchronization remain in measured time but do not enlarge the ideal work or minimum-traffic terms. The plotted throughput is not a hardware instruction rate or measured DRAM bandwidth.

The full model is `min(R, BW*I)`. The theoretical parameters are `R=37.224960 Tcompare/s` and `BW=8 TB/s`; calibrated parameters are `R=37.047490 Tcompare/s` and `BW=6.912116 TB/s`. Their knees are 4.65312 and approximately 5.35979 compare/byte, above Top-K's ideal `[0.125, 0.25)` intensity range. The semantic comparison convention counts two binary comparisons per FMNMX3 result; it is not FP32 FLOPS.

Figure 7B uses linear axes at B=1024. In this article, a Pareto curve denotes each operator's plotted intensity–throughput curve at that fixed batch. Figure 6 also shows B=1, and both heatmaps cover all 11 batches.

Reachable rate is `100 * P / min(R, BW*I)` percent, using the calibrated roof. Compute each point from the arithmetic-mean duration across the same matched layers used in Figure 7B. The average reachable rate is the unweighted arithmetic mean of these point-level percentages, and the peak is their maximum. Flash and Pro each contribute nine intensity points; V3.2 contributes seven. The average is not weighted by row length or serving frequency. Unsupported HPC-ops Pro results remain absent. `summary.json` records the point counts and average/peak percentages under `roofline_reachable_rate`.

The read-dominated roof is optimistic; mixed read/write behavior and additional kernel work can lower achievable throughput. Reachable rate describes useful selection work relative to this model, not measured DRAM bandwidth utilization.

## Serving Results

Decode TPOT results come from public [PR #18410](https://github.com/NVIDIA/TensorRT-LLM/pull/18410); prefill results come from public [PR #18702](https://github.com/NVIDIA/TensorRT-LLM/pull/18702). These are separate serving experiments, not transformations of the operator speedups. The 2.9–4.5% throughput increase isolates adding V2 prefill to a deployment already using V2 decode. It is not the throughput gain of replacing radix in both phases.
