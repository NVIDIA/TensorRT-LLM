---
name: perf-host-analysis
description: >
  Analyze host/CPU overhead in TensorRT-LLM inference from nsys traces.
  Detect whether host overhead is the bottleneck using GPU idle ratio, host prep
  exposed ratio, and per-phase evidence. For regressions, isolate forward steps
  via allreduce/NVTX patterns, compare host operation breakdowns across versions,
  and identify scheduling or request-management overhead. Supports optional
  inter-kernel gap, eager-vs-graph, pattern mapping, and multi-rank straggler
  drill-down. Use standalone or within perf-analysis. Triggers: host overhead,
  inter-step gap, scheduling overhead, forward step isolation, nsys iteration
  analysis, NVTX breakdown, request management overhead, GPU idle, host bottleneck,
  host prep exposed, inter-kernel gap, bubble analysis, graph coverage, eager kernel,
  rank imbalance, straggler detection.
license: Apache-2.0
tags:
  - analysis
  - profiling
  - host-overhead
  - nsys
  - inference
dependencies:
  - workload-instrumentation
  - trace-interpretation
metadata:
  author: NVIDIA Corporation
---

# Host Performance Analysis

Analyze host/CPU overhead in TensorRT-LLM inference workloads from nsys traces. This skill operates in two phases:

| Phase | Question | Input | Output |
|-------|----------|-------|--------|
| **Detection** | Is host overhead the bottleneck? | Single nsys trace | YES/NO verdict with metric evidence |
| **Root Cause** | What specifically regressed? | One or two nsys traces | NVTX per-step breakdown, regression sources, optional kernel-level drill-down |

## When to Use

- Before starting host optimization work -- confirms the bottleneck is real (Detection)
- As a sub-step of `perf-analysis` for bottleneck classification (Detection)
- When GPU utilization is suspiciously low and you need to know why (Detection)
- When throughput regressed but GPU kernel execution times are unchanged (Root Cause)
- When the gap between forward step iterations has increased (Root Cause)
- To compare inter-iteration overhead between two versions of the inference engine (Root Cause)
- When you need sub-operation granularity on inter-kernel gaps or graph coverage (Root Cause, kernel-level drill-down)
- When piecewise CUDA graph coverage is unexpectedly low (Root Cause, kernel-level drill-down)
- When multi-rank inference shows unexplained performance asymmetry (Root Cause, kernel-level drill-down)

Do NOT use when:
- The regression is in individual kernel performance (use `perf-nsight-compute-analysis`)
- You need to profile a workload from scratch (use `workload-instrumentation` first)
- The issue is NCCL communication (use distributed analysis)

## Prerequisites

- An nsys trace file (`.sqlite` or `.nsys-rep`) from a TRT-LLM benchmark run
- For Root Cause comparison: two traces (baseline and target)
- Python 3 with sqlite3 support

---

## Key Concepts

### Host Overhead in LLM Inference

In an LLM inference loop, each iteration consists of:
```
[inter-step gap] -> [_forward_step] -> [inter-step gap] -> [_forward_step] -> ...
```

The **forward step** includes GPU kernel execution (GEMM, attention, normalization, allreduce) plus host-side preparation. The **inter-step gap** includes host-side work between forward steps (scheduling, request fetching, broadcasting, sampling, response handling).

See [references/trtllm-nvtx-ranges.md](references/trtllm-nvtx-ranges.md) for the full per-operation breakdown and timing ranges.

### Hidden vs Exposed Host Overhead

Host overhead only hurts performance when it is **exposed** -- the GPU is idle waiting for work. When host prep overlaps with GPU execution, it is **hidden** and free. See [references/metrics.md](references/metrics.md) (M3 section) for diagrams and the exposed/hidden computation.

### Forward Step Isolation

In TP configurations, forward steps are isolated via allreduce kernel grouping (deterministic count per transformer layer). For TP=1, NVTX `_forward_step` ranges are used directly. See [references/iteration-isolation-techniques.md](references/iteration-isolation-techniques.md) for the full algorithm.

### Phase Classification (Context vs Generation)

Iterations are classified by NVTX marker text into **context** (eager, no CUDA graphs) and **generation** (CUDA graph replay). Per-phase analysis is critical because aggregate metrics can mask phase-specific bottlenecks. See [references/phase-classification.md](references/phase-classification.md).

---

## Phase 1: Detection (YES/NO Verdict)

Determine whether host overhead is the primary bottleneck.

### Detection Metrics

Six metrics in four categories. See [references/metrics.md](references/metrics.md) for full definitions, formulas, and SQL queries.

| # | Metric | Threshold | What it answers |
|---|--------|-----------|-----------------|
| M1 | GPU idle ratio | > 0.30 | Is the GPU starved for work? |
| M2 | Launch overhead ratio | > 0.10 | Is kernel launch itself expensive? |
| M3a | Host prep exposed ratio | > 0.50 | How well is host prep pipelined? |
| M3b | Host prep perf impact | > 0.05 | How much throughput does exposed prep cost? |
| M3c | Host prep idle attribution | > 0.50 | Is host prep the main cause of GPU idle? |
| M4 | GPU utilization | < 0.60 | Is GPU utilization too low? |
| M5 | NCCL ratio (caveat) | > 0.20 | Is communication a confounding factor? |

**Host prep confirmation rule**: Host prep is a confirmed bottleneck only when **both** M3b AND M3c cross their thresholds.

Thresholds are configurable with per-phase variants. See [references/thresholds.md](references/thresholds.md).

### Detection Workflow

#### Step 1: Input Validation

```bash
# Accept .sqlite or .nsys-rep
ls -la <trace_file>

# If .nsys-rep, export to SQLite first
nsys export -t sqlite -o <output.sqlite> <input.nsys-rep>
```

#### Step 2: Run Detection Script

```bash
python scripts/detect_host_overhead.py \
  --trace /path/to/trace.sqlite \
  --output /path/to/verdict.json
```

The script computes M1, M2, M4, M5 from SQL, optionally M3 via range intersection, applies the verdict logic, and outputs structured JSON. See [references/output-format.md](references/output-format.md) for the output schema.

For manual metric extraction via SQL, see [references/nsys-schema.md](references/nsys-schema.md).

#### Step 3: Interpret Verdict

**Overall Verdict:**
```
if aggregate_verdict == YES or context_verdict == YES or generation_verdict == YES:
    overall_verdict = YES
```

Per-phase analysis can **elevate** the verdict but never **demote** it.

Format using the template in [references/output-format.md](references/output-format.md).

**Next Steps:**
- **If YES** -> Proceed to Phase 2 (Root Cause) below, then use `perf-host-optimization` skill
- **If NO** -> Use `perf-nsight-compute-analysis` for kernel SOL% or `trace-interpretation` for full classification

---

## Phase 2: Root Cause Analysis

Identify which specific host operations regressed and by how much. Works with a single trace (breakdown) or two traces (comparison).

### Principles

1. **Isolate forward steps, not the full trace.** nsys traces contain warmup, JIT, model loading, and teardown.
2. **Use structural kernel patterns for iteration detection.** Allreduce grouping is more robust than kernel density.
3. **Compare steady-state iterations.** Filter to identical workload (same batch size, same ctx/gen mix).
4. **Per-step metrics, not totals.** Always compare per-step averages.

### Root Cause Workflow

#### Step 1: Collect nsys Traces

Profile both versions (if comparing) with identical settings:

```bash
nsys profile -o /path/to/trace \
  -t cuda,nvtx,osrt \
  --force-overwrite=true \
  --cuda-memory-usage=true \
  -w true \
  <benchmark_command> --num_requests 500
```

#### Step 2: Export to SQLite

```bash
nsys export --type=sqlite --force-overwrite=true -o trace.sqlite trace.nsys-rep
```

#### Step 3: Run Host Overhead Analysis

```bash
# Two-trace comparison
python scripts/analyze_host_overhead.py \
  --baseline /path/to/baseline/trace.sqlite \
  --target /path/to/target/trace.sqlite \
  --baseline-label "v1.1" \
  --target-label "main" \
  --output /path/to/output/analysis.txt

# Single-trace breakdown
python scripts/analyze_host_overhead.py \
  --baseline /path/to/trace.sqlite \
  --baseline-label "current"
```

#### Step 4: Interpret Results

The script produces:
1. **Allreduce-based iteration detection** -- confirms forward step boundaries
2. **Per-step wall time comparison** -- quantifies the regression
3. **NVTX per-step breakdown** -- identifies which host operations regressed
4. **GPU kernel comparison** -- confirms GPU execution is unchanged
5. **CUDA API comparison** -- detects kernel launch overhead changes

### Reading the Output

**Per-Step Wall Time:**
```
Avg wall time per step: 3,317 us (baseline) vs 3,978 us (target)  +19.9%
```
This is the primary regression metric.

**NVTX Breakdown:**
```
Operation           | baseline (us/step) | target (us/step) | Delta    | Status
_fetch_new_requests |               36   |             270  | +234     | REGRESSION
broadcast_requests  |                -   |             250  | +250     | NEW
_update_requests    |              413   |             723  | +310     | REGRESSION
```
Focus on operations with large absolute deltas.

**GPU Kernel Comparison:**
```
Kernels per step (launched): 6.2 (baseline) vs 21.9 (target)  +253%
```
More individual launches = more host-side launch overhead.

### Step 5: Kernel-Level Drill-Down (Optional)

When the NVTX breakdown identifies a regressing operation but does not reveal *why* (the overhead is inside the GPU dispatch, not between NVTX ranges), drill below NVTX operations into individual GPU kernel launches.

See [references/kernel-level-analysis.md](references/kernel-level-analysis.md) for full technique details, SQL queries, and examples.

**When to drill down:**
- An operation has high wall time but the overhead is inside GPU dispatch, not between NVTX ranges
- You need to understand how much of the forward pass is graph-captured vs eager
- Per-layer overhead is significant and you need to map kernels to functional groups
- Multi-rank inference shows unexplained performance asymmetry

#### Kernel-Level Techniques

| Technique | Question | Key Output |
|-----------|----------|------------|
| **Inter-Kernel Gap Analysis** | Where is the GPU idle between kernels? | Gap bucket distribution, top-N largest gaps with source mapping |
| **Eager vs Graph Classification** | What fraction of kernels are graph-captured? | Graph coverage ratio, list of eager kernels with source attribution |
| **Repeating-Pattern Mapping** | Which functional group within a layer has the most overhead? | Per-group gap totals, priority ranking |
| **Straggler Detection** | Is one rank consistently slower? | Straggler rank ID, root cause (extra host work, queue depth feedback loop) |

#### Workflow

1. **Start with Inter-Kernel Gap Analysis** — bucket the gap distribution to understand the dominant overhead type (graph dispatch, Python interpreter, host-device sync)
2. **If piecewise graph is in use**, run Eager vs Graph Classification to measure graph coverage and identify unnecessary eager kernels
3. **For per-layer overhead**, use Repeating-Pattern Mapping to isolate the highest-overhead functional group within a single layer
4. **For multi-rank setups**, run Straggler Detection if per-step wall time varies across ranks

#### Kernel-Level Findings to Optimization Patterns

| Finding | Optimization Pattern |
|---------|---------------------|
| Large gaps from Python tensor view chains | CUSTOM_OP — replace with C++ custom op |
| Graph-capturable kernels running eagerly | GRAPH_EXPAND — fix partition poisoning |
| Monolithic custom op blocking graph capture | GRAPH_SPLIT — split into capturable + eager parts |
| Host-device sync (`.item()`) in per-layer code | SYNC (Pattern 1: pre-compute on CPU) + HOIST (Variant B: pass from step level) |
| Per-layer buffer allocation | ALLOC — pre-allocate at init |
| Straggler rank with extra host work | Apply targeted optimization to coordinator-only code paths |

---

## Common Patterns and Root Causes

### Pattern 1: Request Management Refactor
**Symptom**: `_fetch_new_requests` regressed 5-10x, new `broadcast_requests` operation.
**Cause**: Request fetching refactored for multi-rank broadcasting in TP.
**Mitigation**: Optimize broadcast path; batch request state updates.

### Pattern 2: Increased Kernel Launch Count
**Symptom**: 3-5x more `cudaLaunchKernel` calls per step, similar GPU time.
**Cause**: Operations that were fused or graph-captured are now individual launches.
**Mitigation**: Re-fuse kernels; extend CUDA graph capture scope.

### Pattern 3: New Bookkeeping Operations
**Symptom**: New NVTX ranges like `_write_finish_reasons`, `handle_additional_outputs`.
**Cause**: New features added to the inference loop without overhead budgeting.
**Mitigation**: Defer non-critical bookkeeping to async paths; batch updates.

### Pattern 4: Flashinfer JIT Warmup Masquerading as Inference
**Symptom**: Massive elementwise/reduce kernel counts in "steady state" analysis.
**Cause**: Analysis window includes flashinfer JIT compilation phase.
**Fix**: Use allreduce-based iteration isolation, not kernel density or time windows.

### Pattern 5: Context-Only Bottleneck (Masked by Aggregate)
**Symptom**: Aggregate metrics below threshold, but context iterations have 50% GPU idle.
**Cause**: Generation iterations dilute the context-phase bottleneck.
**Fix**: Per-phase analysis in Detection phase catches this.

---

## Pitfalls

### 1. shortName is an Integer ID
In `CUPTI_ACTIVITY_KIND_KERNEL`, `shortName` is an integer referencing `StringIds.id`. Always join. See [references/nsys-schema.md](references/nsys-schema.md).

### 2. NVTX textId vs text
Most NVTX events have `textId` (integer) but NULL `text`. Join with StringIds. See [references/nsys-schema.md](references/nsys-schema.md).

### 3. Duplicate NVTX Ranges from TP Ranks
In TP configurations, each rank reports NVTX ranges independently. De-duplicate by grouping entries within 100us of each other.

### 4. Negative Inter-Step Gaps
When TP ranks report overlapping NVTX ranges, `gap = next_start - prev_end` can be negative. Use the maximum end time when de-duplicating.

### 5. Benchmark Window Selection
The allreduce-based window captures context+generation phases; steady-state NVTX filtering captures generation-only. Both are valid; use the appropriate one for your comparison goal.

---

## Handoff to Optimization

When analysis is complete and the verdict is **YES**, hand off to the `perf-host-optimization` skill with:

1. **Detection verdict and evidence**: Which metrics crossed thresholds (M1-M5), whether host prep was confirmed (M3b+M3c), and per-phase breakdown.

2. **NVTX-based triage** (from Root Cause): Top regressing operations by absolute delta (us/step). Map NVTX range names to source functions -- see [references/trtllm-nvtx-ranges.md](references/trtllm-nvtx-ranges.md).

3. **Handoff data block**: Include structured data from [references/output-format.md](references/output-format.md) (see "Handoff to Optimization" section).

4. **Kernel-level findings** (from drill-down, if performed): Inter-kernel gap distribution, graph coverage ratio, per-group overhead map, and straggler rank identification. Map findings to optimization patterns using the table in the Root Cause kernel-level drill-down section above.

---

## Reference

| File | Contents |
|------|----------|
| [references/metrics.md](references/metrics.md) | Full metric definitions, formulas, SQL queries, M3 sub-metric analysis |
| [references/thresholds.md](references/thresholds.md) | Aggregate and per-phase threshold tables |
| [references/phase-classification.md](references/phase-classification.md) | NVTX marker parsing, iteration classification, per-phase aggregation |
| [references/output-format.md](references/output-format.md) | Report template and integration JSON schema |
| [references/examples.md](references/examples.md) | Worked scenarios (aggregate, phase-specific, and case study) |
| [references/iteration-isolation-techniques.md](references/iteration-isolation-techniques.md) | Allreduce, NVTX, and kernel-density iteration isolation techniques |
| [references/trtllm-nvtx-ranges.md](references/trtllm-nvtx-ranges.md) | TRT-LLM NVTX range reference with per-operation timings |
| [references/kernel-level-analysis.md](references/kernel-level-analysis.md) | Kernel-level drill-down techniques: gap analysis, graph classification, pattern mapping, straggler detection |
| [references/nsys-schema.md](references/nsys-schema.md) | nsys SQLite schema reference and useful queries |
| [scripts/analyze_host_overhead.py](scripts/analyze_host_overhead.py) | Python script for Phase 2 root cause analysis |
| [scripts/detect_host_overhead.py](scripts/detect_host_overhead.py) | Python script for Phase 1 detection verdict |
