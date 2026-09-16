<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GVR V2: Faster Exact Top-K with Self-Sampling and Multi-Thresholding

By NVIDIA TensorRT-LLM Team

## Introduction

Selecting 1,024 INT32 indices from 131,072 FP32 scores writes just **4 KiB of output**, yet one complete read of the scores moves **512 KiB**. Every additional full-row pass pays that input cost again. For a sparse-attention indexer, finding the Top-K boundary can therefore cost far more than emitting the winners.

GVR V2 makes each full-row pass more useful. **Self-sampling estimates where the current row's Top-K boundary lies; multi-thresholding derives many exact population counts from one classification pass.** Together, they concentrate exact refinement on the small group of scores still competing for the final slots.

On B200, this design delivers **4.93× geometric-mean speedup over TensorRT-LLM radix CUDA and 1.66× over SGLang v2**, with **2.01× over FlashInfer, 2.37× over DeepSelect FP32, and 1.55× over HPC-ops FP32**. The workloads span DeepSeek-V3.2, DeepSeek-V4 Flash, and DeepSeek-V4 Pro indexers. The SGLang result includes planning; the transform-only comparison is **1.42×**.

![Three horizontal bar-chart panels compare GVR V2, temporal GVR R0 and tiered versions, SGLang, FlashInfer, radix CUDA, DeepSelect FP32, and HPC-ops FP32 on common cases per model. GVR V2 is 1.00; shorter bars mean less kernel time.](../media/gvr_v2/speedup.svg)

*Figure 1. Kernel time relative to GVR V2, geometrically averaged over the same workloads within each model; shorter is faster. The temporal bars show the R0 and tiered implementations. SGLang includes planning; HPC-ops does not support Pro.*

[The original GVR blog](blog21_Temporal_Correlation_Meets_Sparse_Attention.md) described a temporal shortcut: the previous decode step's selected indices predict the next step's winners. V2 makes the row itself the source of the guess. This removes a dependent gather and the need to carry Top-K state across decode steps, while preserving the **Guess–Verify–Refine** exactness contract. It also makes the design useful for prefill, where a previous decode selection does not exist.

## Table of Contents

- [From GVR V1 to V2: Two Costs to Remove](#from-gvr-v1-to-v2-two-costs-to-remove)
- [Self-Sampling: Calibrate the Search to This Row](#self-sampling-calibrate-the-search-to-this-row)
- [Multi-Thresholding: Make Each Full-Row Pass Count](#multi-thresholding-make-each-full-row-pass-count)
- [Why the Two Ideas Work Together](#why-the-two-ideas-work-together)
- [Mapping Selection to Blackwell](#mapping-selection-to-blackwell)
- [Performance Against Five Baselines](#performance-against-five-baselines)
- [The Roofline Model: Fewer Passes, More Useful Work](#the-roofline-model-fewer-passes-more-useful-work)
- [Decode and Prefill in TensorRT-LLM](#decode-and-prefill-in-tensorrt-llm)
- [Further Reading](#further-reading)
- [Conclusion](#conclusion)

## From GVR V1 to V2: Two Costs to Remove

Original GVR V1 gathers the current scores at the previous step's Top-K indices, estimates a bracket from their minimum, maximum, and mean, then uses a secant-style search on the monotone count function

$$
C(T)=\sum_{i=0}^{N-1}\mathbf{1}[x_i\ge T].
$$

It seeks an admission threshold with enough survivors to contain Top-K, but few enough to fit its candidate capacity. A good temporal prediction makes that search short. Two costs remain: loading the old indices before the score addresses are known, and scanning the row again when another threshold must be tested. The first is a memory dependency; the second multiplies traffic as $N$ grows.

V2 addresses both. Its streaming paths read regularly spaced, coalesced samples of the current row to choose a useful bracket. They then classify the full row into bins whose cumulative counts evaluate many thresholds together. The algorithm spends a small amount of work learning where to look, then obtains much more information from each expensive full-row pass.

V1 also used a histogram during local refinement after candidate collection. The distinction is where that information becomes available: V2 makes multi-threshold population information central to verification, so it can pass an already identified crossing to refinement.

| Algorithm question | Original GVR V1 | Streaming GVR V2 |
| :--- | :--- | :--- |
| Where does the guess come from? | Current scores gathered through previous-step indices | A coalesced sample of the current row |
| What guides admission? | Hint statistics and scalar secant-style count queries | Sample-derived primary threshold, lower safety floor, and upper anchor |
| What does verification learn? | A count for the trial threshold | Exact bin populations and counts at many boundaries |
| Where is the remaining uncertainty? | The admitted candidate set | The crossing bin containing rank $K$ |
| What state crosses decode steps? | Per-layer prior indices | No Top-K prior |
| How does a bad guess affect the result? | More verification/refinement work | Lower admission or exact recovery; membership remains exact |

![GVR V1 and V2 data-flow comparison beside measured progression from temporal R0 to tiered temporal GVR and self-sampling GVR V2.](../media/gvr_v2/evolution.svg)

*Figure 2. Original V1 uses temporal hints and scalar threshold search. The later temporal R0 and tiered implementations introduce broader verification and execution specialization; V2 combines these advances with current-row self-sampling. Bars show speedup over radix CUDA.*

Temporal R0 and tiered GVR achieve **2.59× and 3.47×** speedup over radix CUDA; V2 reaches **4.93×**. V2 is **1.91× faster than temporal R0** and **1.42× faster than tiered temporal GVR**, combining current-row sampling, multi-threshold verification, and specialized execution paths.

## Self-Sampling: Calibrate the Search to This Row

The first job is to place the search near the upper tail of this row's score distribution. V2 takes a small, deterministic sample distributed across the row and uses its ranks to estimate thresholds for the full population. The sample supplies a starting region; the later full-row verification decides whether that region contains enough candidates.

In the `main` family, a sample unit is two adjacent `float4` vectors: eight scores loaded together. The clustered streaming family uses four vectors, or sixteen scores. Sample units are regularly spaced across the valid interval. Their addresses follow from the row layout, so sampling does not wait for previous-step indices. Sampling can still miss an unusual tail; its role is to reduce work, not to decide output membership.

Let $S$ be the sample size and $A\ge K$ the desired full-row candidate population. A **256-bin sample histogram** approximates the score distribution. Descending sample ranks

$$
r_A\approx\frac{AS}{N},\qquad r_K\approx\frac{KS}{N},\qquad r_{2A}\approx\frac{2AS}{N}
$$

provide three anchors:

- **Primary threshold $T$:** aim for roughly $A$ survivors, leaving room around the desired $K$ winners.
- **Upper anchor $T_K$:** estimate the neighborhood of rank $K$; together with $T$, it defines the upper classification bound $H$ (`HIC` in the implementation).
- **Lower floor $T_{\mathrm{floor}}$:** admit a larger population if the first estimate is too aggressive (`TSH` in the implementation).

The concrete sample budget, safety margin in $A$, rank rounding, and capacity clamps depend on the launch family. The algorithm does not assume independent random samples or a known analytical distribution. Its useful property is that the bracket is calibrated from the same row whose boundary will be verified.

![Self-sampling chooses a bracket, multi-threshold verification obtains exact bin counts, and refinement emits 980 certain winners plus 44 of 73 boundary candidates for K=1024.](../media/gvr_v2/algorithm.svg)

*Figure 3. Two histograms serve different purposes: the sample histogram estimates a useful region; the verification histogram contains exact populations from the complete valid row or its complete admitted candidates. The pictured bin heights and 980/73/44 example are illustrative. Register-resident shortcuts are described below.*

## Multi-Thresholding: Make Each Full-Row Pass Count

A scalar verification pass answers one question: how many scores exceed $T$? Multi-thresholding obtains a family of answers from the same classification work.

Conceptually, divide the bracket $[T,H]$ into $M$ ordered bins, with boundaries $t_0,\ldots,t_M$, and let $h_j$ be the exact population of bin $j$. A descending cumulative scan yields

$$
C(t_j)=\sum_{\ell=j}^{M-1}h_\ell.
$$

The streaming verification histogram has 256 bins. Each survivor is assigned to a bin once; a small on-chip cumulative scan then exposes counts at all its boundaries. **One classification contributes to an entire family of threshold counts.** The expensive score reads are shared, and the remaining scan touches only the bin counters. Verification can locate where the population crosses $K$ without issuing another full-row query for each trial threshold.

Values above $H$ saturate into the top bin and remain candidates. Full-row accounting establishes whether enough survivors exist. The implementation may build the histogram during streaming, merge shard histograms through distributed shared memory, or reconstruct it from a complete bounded staging slab. Those execution choices preserve the same logical result: exact populations must cover the admitted set before its crossing is trusted.

### Turn a Threshold Search into a Boundary Problem

Scanning bins from high scores to low identifies the crossing bin $j^{\ast}$ with

$$
a=\sum_{\ell\gt j^{\ast}}h_\ell\lt K,\qquad a+h_{j^{\ast}}\ge K.
$$

Every score in a higher bin is a certain winner. Every score in a lower bin is unnecessary. Only $K-a$ winners must be chosen from the crossing bin's $m=h_{j^{\ast}}$ candidates:

$$
\mathrm{TopK}(x)=\lbrace \text{all positions in higher bins}\rbrace
\quad\cup\quad \mathrm{Top}_{K-a}(\text{crossing bin}).
$$

For $K=1024$, suppose 980 scores lie above the crossing bin and 73 lie inside it. Emit the 980 directly and select the best **44 of those 73**. The remaining ranking problem has shrunk from the full row to a narrow boundary population. If the entire crossing bin is needed, it can be emitted directly. Small crossings use direct ranking; larger ones use exact order-preserving FP32 key refinement.

The histogram discretizes the search region, not the selected scores. Exact comparisons within the crossing bin resolve its coarse boundaries, including ties. Thus fewer passes do not require approximate Top-K membership.

### Two Kinds of Thresholds, Two Different Jobs

The **sample-derived admission ladder** ($T$, a lower floor, and a conservative sentinel) repairs a poor initial guess. The **verification bin boundaries** answer many exact population queries within an admitted region. They work at different levels and should not be confused.

For non-split streaming, too few survivors can trigger another full-row scan at a valid lower floor and then at the sentinel. Split-row streaming can stage down to the lower floor within its scan and use exact recovery if necessary. Candidate overflow cannot be treated as a successful partial selection: the kernel re-scans or falls back to exact key-space selection over a complete candidate set or the whole row.

## Why the Two Ideas Work Together

The two ideas address successive sources of work. Self-sampling gives the histogram a useful region to resolve. Multi-threshold verification locates the exact crossing within that region. Refinement ranks only the candidates whose membership is still undecided. The expensive global operation gets progressively more information before the algorithm commits to more work.

A good sample without broad verification could still pay for repeated full-row threshold tests. A histogram without a useful bracket could put too many scores into the crossing bin and leave expensive refinement. **V2 uses the sample to focus the bins, and the exact bin counts to make the sample safe.** Its performance objective is to avoid repeated expensive passes over $N$, while keeping most remaining work within a much smaller candidate region.

This is an optimization of work, not a fixed one-pass guarantee. Difficult distributions, staging overflow, and unusable brackets may require extra scans. Exactness includes those paths: [PR #18625](https://github.com/NVIDIA/TensorRT-LLM/pull/18625) fixes register-family infinite-width brackets by using exact whole-row key selection, preserving `+inf` among the winners.

The output contract is an exact selected **value multiset** with valid unique indices. Equal-valued candidates are interchangeable; index order need not match `torch.topk`. Short rows return all valid local indices followed by `-1` padding. NaN ordering remains implementation-specific. Guess quality controls the amount of work; complete counts and exact refinement control membership.

## Mapping Selection to Blackwell

A single scheduling policy cannot serve both one short row and thousands of long rows efficiently. GVR V2 uses four kernel families, implemented in CuTe DSL:

| Family | Where the scores or candidates live | Why it helps |
| :--- | :--- | :--- |
| `reg` | A row resides in one thread block's registers | Avoids repeated global loads when the row fits |
| `reg_clus` | Register slices across cooperating blocks | Exposes more parallelism for medium rows at small batch sizes |
| `clus` | Streaming shards; histograms and candidates shared within a hardware cluster | Merges through distributed shared memory |
| `main` | Streaming scan with bounded candidate staging | Covers the remaining shapes, including long rows and large batches |

A thread block is also called a cooperative thread array, or CTA. Blackwell thread-block clusters let cooperating CTAs exchange data through distributed shared memory. This reduces the need to materialize intermediate results in global memory for eligible shapes.

Register families bypass sparse sampling but retain exact histogram crossing and refinement. In the normal case, the first $K$ current-row values establish the initial bracket; near the short-row regime, a whole-row bracket is used. Full-row classification and exact crossing refinement still determine the output. The `main` family can assign multiple CTAs to a row when a small batch would otherwise leave much of the GPU idle.

This explains the two sources of performance improvement: a better starting threshold reduces selection work, and a suitable kernel family reduces the cost of executing that work. Neither eliminates the obligation to examine all valid scores.

## Performance Against Five Baselines

### The Gains Extend Beyond an Average

Figure 1 puts the GVR evolution and library baselines on the same scale. Across the three models, tiered temporal GVR takes **1.34–1.50×** V2's time, SGLang takes **1.55–1.78×**, and radix CUDA takes **4.73–5.15×**. The comparison also reveals model-dependent behavior: HPC-ops is closer on V3.2 than on Flash, while DeepSelect's FP32 V3.2 route leaves a larger gap.

GVR V2 is faster in every tested radix and FlashInfer comparison and in **99.91%** of SGLang comparisons. Figure 4 shows how the advantage changes with row length and batch size.

![Three heatmaps of GVR V2 speedup over SGLang for every captured row-length bucket and all eleven batch sizes, averaged geometrically across layers.](../media/gvr_v2/sglang_map.svg)

*Figure 4. SGLang plan + transform time divided by GVR V2 time, geometrically averaged across layers at each shape. A value of 1.0 means equal performance. Row lengths are rounded in the axis labels.*

Long rows at intermediate batch sizes show particularly strong gains: **V4 Pro reaches 4.67× at $N=262{,}127$, $B=64$**. This is the strongest shape-average result in the grid; most regions are around 1.3–2.0×.

Two details determine how these gains carry into an application: the actual row-length and batch distribution, and the work each API returns. The next sections make both explicit.

### Benchmark Setup

The benchmarks use FP32 indexer scores from the three models below on NVIDIA B200, with batch sizes from 1 to 1,024. Results report cold-L2 GPU kernel time, excluding compilation, input preparation, and Python overhead. Speedups are geometric means over matched workloads.

| Model | $K$ | Indexer compression | Valid row lengths $N$ |
| :--- | ---: | ---: | :--- |
| DeepSeek-V4 Flash | 512 | 4 | 1,027–262,127 |
| DeepSeek-V4 Pro | 1,024 | 4 | 1,027–262,127 |
| DeepSeek-V3.2 | 2,048 | 1 | 4,111–163,775 |

**$N$ is the indexer row width, not the original prompt length.** A roughly 512K-token V4 context yields a roughly 128K-wide indexer row because of 4× compression.

### Overall and Per-Model Results

| Baseline | Geomean speedup | Minimum speedup | GVR V2 faster |
| :--- | ---: | ---: | ---: |
| SGLang v2, plan + transform | **1.66×** | 0.95× | 99.91% |
| FlashInfer 0.6.14 `top_k` | **2.01×** | 1.20× | 100.00% |
| TensorRT-LLM radix CUDA dispatch | **4.93×** | 1.34× | 100.00% |
| DeepSelect v1.0.0 FP32 | **2.37×** | 0.84× | 99.60% |
| HPC-ops FP32 | **1.55×** | 0.71× | 98.38% |

The minimum column exposes individual regressions that a geometric mean can hide. GVR V2 wins every recorded radix and FlashInfer pair, while SGLang, DeepSelect, and HPC-ops retain individual winning cases.

| Baseline | V4 Flash, $K=512$ | V4 Pro, $K=1024$ | V3.2, $K=2048$ |
| :--- | ---: | ---: | ---: |
| SGLang v2, plan + transform | 1.78× | 1.76× | 1.55× |
| FlashInfer 0.6.14 | 2.12× | 2.14× | 1.89× |
| TensorRT-LLM radix CUDA | 4.74× | 4.73× | 5.15× |
| DeepSelect FP32 | 1.98× | 2.07× | 2.79× |
| HPC-ops FP32 | 2.30× | Unsupported | 1.30× |

*Each model column uses the workloads supported by that baseline.*

### What Explains the Differences

**SGLang.** The **1.66×** comparison includes both planning and transformation. Serving integrations can amortize planning across layers; against transformation alone, V2 achieves **1.42×** geometric-mean speedup and wins **98.78%** of comparisons.

**FlashInfer.** Its `top_k` API returns FP32 values and INT64 indices, while GVR V2 returns INT32 indices only. The **2.01×** result includes that additional output work and FlashInfer's scan of the padded row.

**TensorRT-LLM radix CUDA.** The baseline uses the production dispatcher, including short-row insertion and long-row split-work paths. V2's **4.93×** advantage is consistent with reducing full-row selection passes and matching execution to the workload.

**DeepSelect.** The FP32 comparison requests unsorted INT32 indices. Its $K=2048$ path emphasizes correctness coverage, which helps explain the larger **2.79×** gap on V3.2. BF16 is another important operating point, discussed below.

**HPC-ops.** FP32 support covers $K \in \lbrace 512,2048\rbrace$. V2's advantage is **2.30×** on Flash and **1.30×** on V3.2, with an overall **1.55×** speedup. HPC-ops retains individual wins on V3.2; Pro is unsupported.

### Latency Across Row Length and Batch Size

![Cold kernel latency for all six implementations versus valid row length, with separate panels for three models and batch sizes 1 and 1024.](../media/gvr_v2/latency.svg)

*Figure 5. Mean cold kernel time across matching layers. Each row is a model; the columns contrast batch sizes 1 and 1,024. Solid and dashed lines distinguish benchmark runs. Both axes are logarithmic; 1K means 1,024.*

At $B=1$, keeping a short row in registers and exposing parallelism within a longer row matter more than saturating HBM. At $B=1024$, streaming throughput becomes more visible. The different shapes of these curves are why a single average cannot identify every useful operating region.

For a concrete large-batch slice, the following times are at $B=1024$ and $N\approx131{,}072$, averaged over the same layers as Figure 5:

| Model | GVR V2 | SGLang | FlashInfer | Radix CUDA | DeepSelect FP32 | HPC-ops FP32 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 Flash | **113.1 µs** | 196.8 µs | 275.4 µs | 461.3 µs | 190.5 µs | 199.2 µs |
| V4 Pro | **132.8 µs** | 199.1 µs | 298.3 µs | 477.7 µs | 209.1 µs | — |
| V3.2 | **124.0 µs** | 211.0 µs | 388.8 µs | 496.6 µs | 419.6 µs | 159.7 µs |

The contrast between single-row latency and large-batch throughput reflects the importance of selecting the right execution family.

### BF16 Changes the Comparison

With BF16 input, DeepSelect becomes more competitive. GVR V2, still consuming FP32 scores, achieves **1.40×** geometric-mean speedup and wins **86.03%** of comparisons. DeepSelect wins in long-row, large-batch regions, particularly V4 rows around 131K–262K with batches of 256–1024; the minimum GVR speedup is **0.55×**.

BF16 halves the score-read bytes, but rounding can change Top-K membership. Each kernel is checked against its own input dtype. Conversion time is excluded, so this comparison applies when BF16 scores are already available.

## The Roofline Model: Fewer Passes, More Useful Work

The bar chart shows how much time GVR V2 saves. The roofline asks how much of that time is fundamentally needed to move the input and output. It connects the algorithm's goal—fewer full-row passes—to a hardware limit.

### Locate Top-K on the Hardware Roof

For FP32 input and INT32 index-only output, define the ideal work and minimum traffic as

$$
W=BN,\qquad Q_{\min}=4B(N+K)\ \text{bytes},\qquad
I=\frac{W}{Q_{\min}}=\frac{N}{4(N+K)}.
$$

Here one unit of work is one **abstract comparison per input score**. This is a common normalization for every implementation, not its measured instruction count. Since $1\le K\le N$, ideal Top-K intensity stays within **$0.125\le I\lt 0.25$ compare/byte**.

The theoretical and calibrated B200 roofs, in Tcompare/s, are

$$
P_{\mathrm{theory}}(I)=\min(37.225,8I),\qquad
P_{\mathrm{calibrated}}(I)=\min(37.047,6.912I).
$$

The calibrated limits are **6.912 TB/s** sustained read bandwidth and **37.047 Tcompare/s** semantic comparison throughput. They meet at **5.36 compare/byte**, more than 21 times the maximum ideal Top-K intensity. The entire workload band sits on the bandwidth slope in Figure 6A.

![A clean two-level roofline: the full B200 hardware model highlights Top-K's narrow bandwidth-limited band; three linear-scale zoom panels compare GVR V2 and five baselines at batch 1024, with GVR V2 highlighted in green.](../media/gvr_v2/roofline.svg)

*Figure 6.* A: the full theoretical and calibrated roofs. B: the Top-K band at $B=1024$, plotting useful throughput $P=BN/t$ against intensity $I=N/[4(N+K)]$ on linear axes. Green highlights V2; the dotted line is the calibrated bandwidth roof. All kernels share the same minimum-traffic model $Q_{\min}$, while extra reads and output work remain in measured time.

### Compare Useful Work at the Same Intensity

At fixed $N$ and $K$, every implementation has the same horizontal position. A faster kernel moves **upward**, toward the bandwidth roof. Figure 6B fixes the batch at 1,024 so the curves expose throughput with substantial parallel work available; Figure 5 retains the contrasting single-row view, and Figure 4 covers all 11 batch sizes. The linear vertical scale makes the remaining distance to the roof directly visible.

For the Flash slice in Figure 5, $B=1024$, $N=131{,}075$, and $K=512$ give an optimistic minimum time of **78.0 µs**. GVR V2 takes **113.1 µs**, reaching about **69%** of the calibrated roof under this shared-work normalization. SGLang takes 196.8 µs, or about **40%** of that roof; radix CUDA takes 461.3 µs, or about **17%**. All three solve the same logical selection problem. Their different vertical positions reflect how much elapsed time they spend beyond its minimum traffic requirement.

### Interpret the Remaining Gap

The full ideal bound is $T_{\mathrm{roof}}=\max(Q_{\min}/\mathrm{BW},W/R_{\mathrm{compare}})$. Within the Top-K band, traffic dominates. Sampling, histogram updates, candidate staging, exact refinement, and synchronization account for work beyond the ideal minimum. The read-dominated roof is optimistic, especially when $K/N$ is large; the plotted throughput describes useful selection work rather than measured DRAM traffic.

## Decode and Prefill in TensorRT-LLM

### Stable Launches, Dynamic Row Lengths

CUDA Graph replay needs a stable launch configuration even as requests grow. The host chooses a kernel family and launch envelope before capture. At execution time, each row reads its actual length from device-resident metadata, including its multi-token prediction offset and compression ratio.

Two details matter for correctness and integration. [PR #18683](https://github.com/NVIDIA/TensorRT-LLM/pull/18683) keeps the physical row-width bound separate from the routing bound and ensures warmup populates launchers for exact row counts. [PR #18446](https://github.com/NVIDIA/TensorRT-LLM/pull/18446) makes dispatch configuration-based and limits previous-step prior allocation and updates to the temporal engine. The device-side prior-seeding change in [PR #18646](https://github.com/NVIDIA/TensorRT-LLM/pull/18646) benefits that retained temporal path; V2 does not need prior seeding.

### The Same Approach Extends to Prefill

[PR #18702](https://github.com/NVIDIA/TensorRT-LLM/pull/18702) applies the streaming engine to each prefill row's valid interval `[start, end)`. Output indices are relative to `start`, and short windows retain identity indices with `-1` padding. There is no previous token's selection to initialize.

In B200 profiles, V2 makes prefill Top-K **1.84–2.61×** faster than radix CUDA. Adding V2 prefill to a deployment already using V2 decode improves serving throughput by **2.9–4.5%** on long-input workloads. This is the incremental benefit of the prefill change.

For decode, [PR #18410](https://github.com/NVIDIA/TensorRT-LLM/pull/18410) reports **6–19% lower time per output token** on 8×B200 with TP8/EP8 and batch/concurrency 1. The serving benefit depends on Top-K's share of total execution time; the kernel comparisons above do not measure competing serving stacks.

### Enable GVR V2

On a TensorRT-LLM revision containing the linked integration PRs, save the following as `gvr_v2.yaml`:

```yaml
sparse_attention_config:
  algorithm: deepseek_v4
  enable_heuristic_topk: true
  use_self_sampling_topk: true
```

Use `algorithm: dsa` for DeepSeek-V3.2. The checkpoint supplies the model's Top-K width. For example:

```bash
trtllm-serve deepseek-ai/DeepSeek-V4-Flash \
  --config gvr_v2.yaml --tp_size 8 --ep_size 8
```

`enable_heuristic_topk` defaults to `false`. Once it is enabled, `use_self_sampling_topk` defaults to `true`; the second field is explicit here for clarity. Supported prefill layers follow the same V2 selection. Setting `use_self_sampling_topk: false` selects temporal GVR for decode, whose prefill path remains radix.

V2 requires CUTLASS DSL, datacenter Blackwell SM100/SM103, $K\in\lbrace 512,1024,2048\rbrace$, and supported indexer compression ratios 1 or 4. The fast path expects FP32 scores with unit inner stride, a row stride divisible by four floats, and a 16-byte-aligned base. Unsupported layouts use exact native fallback selection. The retired `TRTLLM_GVR_SELF_SAMPLING` environment variable is no longer the enablement mechanism, and `use_cute_dsl_topk` is not required to select V2.

## Further Reading

[Benchmark methodology and figure reproduction](../media/gvr_v2/README.md) are available separately.

Implementation milestones:

- [PR #17821: original self-sampling decode integration](https://github.com/NVIDIA/TensorRT-LLM/pull/17821).
- [PR #18410: current-row brackets and hint-free production API](https://github.com/NVIDIA/TensorRT-LLM/pull/18410).
- [PR #18625: exact handling of positive infinity in register families](https://github.com/NVIDIA/TensorRT-LLM/pull/18625).
- [PR #18646: device-side prior seeding for the temporal path](https://github.com/NVIDIA/TensorRT-LLM/pull/18646).
- [PR #18683: physical envelopes and exact-row warmup](https://github.com/NVIDIA/TensorRT-LLM/pull/18683).
- [PR #18446: configuration-based dispatch and prior-state removal for V2](https://github.com/NVIDIA/TensorRT-LLM/pull/18446).
- [PR #18702: self-sampling prefill](https://github.com/NVIDIA/TensorRT-LLM/pull/18702).

For the surrounding model pipeline, see [Sparse Attention in TensorRT-LLM](blog17_Sparse_Attention_in_TensorRT-LLM.md) and [DeepSeek-V4 on NVIDIA Blackwell](blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md).

## Conclusion

GVR V2 reduces the cost of finding the boundary before optimizing the work left at that boundary. **Self-sampling** places the search near the current row's tail. **Multi-thresholding** turns a classification pass into many exact population counts. The crossing bin then isolates the candidates that still need ranking. This removes the previous-step Top-K state and dependent gather of temporal GVR while preserving exact membership through verification and recovery.

The B200 results connect that algorithmic change to practical kernels: 4.93× over radix CUDA, 1.66× over SGLang including planning, and broad gains over FlashInfer, DeepSelect FP32, and HPC-ops FP32 on their paired case sets. The roofline makes the objective concrete: move useful selection throughput closer to the bandwidth roof by spending less time revisiting the row. Register, streaming, and cluster specializations make this design work across launch sizes, while the shared exactness contract carries it from decode into prefill.
