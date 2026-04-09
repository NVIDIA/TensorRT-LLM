---
name: perf-host-analysis
description: >
  Analyze host/CPU overhead in TensorRT-LLM inference from nsys traces.
  Phase 1 (Detection): determine whether host overhead is the bottleneck via
  a binary YES/NO verdict with metric evidence (GPU idle ratio, host prep
  exposed ratio, per-phase breakdown). Phase 2 (Root Cause): isolate forward
  step iterations via allreduce kernel patterns, compare NVTX-instrumented host
  operations across versions, and pinpoint scheduling/request-management
  regressions. Usable standalone or as a sub-step of perf-analysis.
  Triggers: host overhead, inter-step gap, scheduling overhead, forward
  step isolation, nsys iteration analysis, NVTX breakdown, request management
  overhead, inference loop overhead, between-step gap, GPU idle, host bottleneck
  detection, host prep exposed.
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
| **Root Cause** | What specifically regressed? | One or two nsys traces | NVTX per-step breakdown, regression sources |

## When to Use

- Before starting host optimization work — confirms the bottleneck is real (Detection)
- As a sub-step of `perf-analysis` for bottleneck classification (Detection)
- When GPU utilization is suspiciously low and you need to know why (Detection)
- When throughput regressed but GPU kernel execution times are unchanged (Root Cause)
- When the gap between forward step iterations has increased (Root Cause)
- To compare inter-iteration overhead between two versions of the inference engine (Root Cause)

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

The **forward step** includes GPU kernel execution (GEMM, attention, normalization, allreduce) plus host-side preparation (_prepare_inputs, resource allocation).

The **inter-step gap** includes all host-side work between forward steps:
- Request scheduling (_schedule)
- Request fetching (_fetch_new_requests)
- Request broadcasting (broadcast_requests — TP configurations)
- Sampling (_sample_async)
- Request processing (_process_requests)
- Response handling (_handle_responses)
- Request state updates (_update_requests)

### Hidden vs Exposed Host Overhead

Host overhead only hurts performance when it is **exposed** — the GPU is idle, waiting for the host to submit work. When host prep runs while the GPU is still busy with previous kernels, it is **hidden** and costs nothing in wall-clock time.

**Scenario A — Host prep hidden (GPU-bound, healthy)**
```
time ------>
Host: |prep N|launch|...wait...|post|prep N+1|launch|...wait...|post|
GPU:         |========= kernels N =========|======= kernels N+1 ======|
              GPU always busy; host prep is hidden behind GPU execution
              exposed host overhead = 0
```

**Scenario B — Host prep exposed (host-bound, bottleneck)**
```
time ------>
Host: |prep N|launch|post|======= long prep N+1 =======|launch|post|
GPU:         |kernels N|     ^^^^ GPU IDLE ^^^^         |kernels N+1|
                              exposed host overhead
                              (directly adds to wall time)
```

**Scenario C — Partially hidden (common in practice)**
```
time ------>
Host: |prep N|launch|post|======== prep N+1 ========|launch|post|
GPU:         |======== kernels N ========|   ^IDLE^  |kernels N+1|
              hidden portion ----------->   exposed
              (overlaps GPU, free)          (GPU idle, costs wall time)
```

### Forward Step Isolation via Allreduce Pattern

In tensor-parallel configurations, each forward step executes a fixed number of allreduce operations (one per transformer layer communication point).

The algorithm:
1. Find all allreduce_fusion kernels in the trace
2. Group consecutive allreduce kernels separated by < 1ms into "iterations"
3. Identify the most common iteration size (= kernels per forward step)
4. Detect phase boundaries where consecutive iterations are separated by > 100ms
5. Select the last phase with the common iteration size as the benchmark phase

See [references/iteration-isolation-techniques.md](references/iteration-isolation-techniques.md) for full details including NVTX-based and kernel-density approaches.

### Phase Classification (Context vs Generation)

TRT-LLM iterations are classified by NVTX marker text into **context** (eager, no CUDA graphs) and **generation** (CUDA graph replay). Aggregate metrics can mask phase-specific bottlenecks, so per-phase analysis is critical.

| Phase | Condition | Characteristics |
|-------|-----------|-----------------|
| Context | `N > 0` (any ctx reqs) | Eager execution, heavier host prep |
| Generation-only | `N == 0, M > 0` | CUDA graph replay, minimal host prep |

NVTX marker format: `[Executor] _forward_step {iter}: {N} ctx reqs, {M} gen reqs`

See [references/phase-classification.md](references/phase-classification.md) for extraction code and per-phase aggregation.

---

## Phase 1: Detection (YES/NO Verdict)

Determine whether host overhead is the primary bottleneck for a TRT-LLM workload.

### Detection Metrics

Six metrics, grouped into four categories. See [references/metrics.md](references/metrics.md) for full definitions and SQL queries.

| # | Metric | Formula | Threshold | What it answers |
|---|--------|---------|-----------|-----------------|
| M1 | GPU idle ratio | `gpu_idle_us / total_us` | > 0.30 | Is the GPU starved for work? |
| M2 | Launch overhead ratio | `cudaLaunchKernel_us / total_us` | > 0.10 | Is kernel launch itself expensive? |
| M3a | Host prep exposed ratio | `exposed_us / host_prep_total_us` | > 0.50 | How well is host prep pipelined? |
| M3b | Host prep perf impact | `exposed_us / total_us` | > 0.05 | How much throughput does exposed prep cost? |
| M3c | Host prep idle attribution | `exposed_us / gpu_idle_us` | > 0.50 | Is host prep the main cause of GPU idle? |
| M4 | GPU utilization | `gpu_active_us / total_us` | < 0.60 | Is GPU utilization too low? |
| M5 | NCCL ratio (caveat) | `nccl_us / gpu_active_us` | > 0.20 | Is communication a confounding factor? |

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

#### Step 2: Extract Metrics via SQL

All Detection metrics (M1, M2, M4, M5) are computed directly from the nsys SQLite trace using SQL queries. No external tools are required.

```sql
-- M1: GPU idle ratio + M4: GPU utilization
-- Step A: Get analysis window
SELECT MIN(start) AS window_start, MAX(end) AS window_end,
       (MAX(end) - MIN(start)) / 1000.0 AS total_time_us
FROM CUPTI_ACTIVITY_KIND_KERNEL;

-- Step B: Compute GPU active time (merge overlapping kernel ranges)
-- Export kernel start/end pairs and merge in Python, or use the
-- approximate sum (accurate when kernel overlap is minimal):
SELECT SUM(end - start) / 1000.0 AS approx_gpu_active_us
FROM CUPTI_ACTIVITY_KIND_KERNEL;
-- gpu_idle_us ≈ total_time_us - gpu_active_us
-- gpu_idle_ratio = gpu_idle_us / total_time_us  (M1, threshold >0.30)
-- gpu_utilization = 1 - gpu_idle_ratio           (M4, threshold <0.60)

-- M2: Launch overhead ratio
SELECT SUM(r.end - r.start) / 1000.0 AS cudaLaunchKernel_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value = 'cudaLaunchKernel';
-- launch_overhead_ratio = cudaLaunchKernel_us / total_time_us  (threshold >0.10)

-- M5: NCCL ratio (caveat metric)
SELECT SUM(k.end - k.start) / 1000.0 AS nccl_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE s.value LIKE '%nccl%';
-- nccl_ratio = nccl_us / gpu_active_us  (threshold >0.20)
```

For precise GPU active time (merging overlapping ranges), export kernel `(start, end)` pairs and merge in Python — see [references/metrics.md](references/metrics.md) for the full approach.

#### Step 3: Extract Per-Iteration Metrics

Parse `_forward_step` NVTX ranges and classify iterations into context/generation phases. See [references/phase-classification.md](references/phase-classification.md).

#### Step 4: Compute M3 (Optional — Advanced)

M3 (Host Prep Exposed Ratio) requires intersecting NVTX host-prep ranges with GPU idle gaps — a range-intersection computation that is not suitable for inline SQL. This metric is **optional** for the Detection verdict; M1+M2+M4+M5 are usually sufficient.

If M3 is needed, compute it in Python:
1. Export merged GPU idle gaps (from Step 2 range merging)
2. Export NVTX ranges matching the host-prep marker (see [references/metrics.md](references/metrics.md) for the configurable range name)
3. Intersect each NVTX range with the idle gaps to compute `exposed_us`
4. Apply M3a/M3b/M3c formulas from [references/metrics.md](references/metrics.md)

#### Step 5: Compute All Metrics and Apply Decision Logic

**Aggregate Verdict:**
```
# Core metrics (always available from SQL):
core_crossed = count of [M1, M2, M4] that cross their threshold

# Optional M3 metrics (if computed):
if M3 available:
    crossed_count = core_crossed + count of [M3a, M3b, M3c] that cross
    applicable_count = 6
else:
    crossed_count = core_crossed
    applicable_count = 3

if crossed_count >= 2:
    aggregate_verdict = YES

if M3 available AND M3b > threshold AND M3c > threshold:
    host_prep_confirmed = true

if nccl_ratio > NCCL_RATIO_CAVEAT_THRESHOLD:
    add caveat
```

**Per-Phase Verdicts:** Apply phase-specific thresholds to context and generation iterations separately.

**Overall Verdict:**
```
if aggregate_verdict == YES or context_verdict == YES or generation_verdict == YES:
    overall_verdict = YES
else:
    overall_verdict = NO
```

Per-phase analysis can **elevate** the verdict but never **demote** it.

#### Step 6: Generate Report

Format using the template in [references/output-format.md](references/output-format.md).

**Next Steps:**
- **If YES** -> Proceed to Phase 2 (Root Cause) below, then use `perf-host-optimization` skill
- **If NO** -> Use `perf-nsight-compute-analysis` for kernel SOL% or `trace-interpretation` for full classification

---

## Phase 2: Root Cause Analysis

Identify which specific host operations regressed and by how much. Works with a single trace (breakdown) or two traces (comparison).

### Principles

1. **Isolate forward steps, not the full trace.** nsys traces contain warmup, JIT compilation, model loading, and teardown. Only forward step iterations represent actual inference performance.

2. **Use structural kernel patterns for iteration detection.** Allreduce kernel grouping is more robust than kernel density or time-window heuristics.

3. **Compare steady-state iterations.** Filter to iterations with identical workload (same batch size, same ctx/gen mix) for clean comparison.

4. **Per-step metrics, not totals.** When benchmark windows differ in duration or step count, always compare per-step averages.

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
1. **Allreduce-based iteration detection** — confirms forward step boundaries
2. **Per-step wall time comparison** — quantifies the regression
3. **NVTX per-step breakdown** — identifies which host operations regressed
4. **GPU kernel comparison** — confirms GPU execution is unchanged
5. **CUDA API comparison** — detects kernel launch overhead changes

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
_sample_async       |            1,163   |             720  | -443     | IMPROVED
_process_requests   |            1,056   |             390  | -666     | IMPROVED
```
Focus on operations with large absolute deltas. Check whether improvements offset regressions.

**GPU Kernel Comparison:**
```
Kernels per step (launched): 6.2 (baseline) vs 21.9 (target)  +253%
```
More individual launches = more host-side launch overhead.

---

## Common Patterns and Root Causes

### Pattern 1: Request Management Refactor
**Symptom**: `_fetch_new_requests` regressed 5-10x, new `broadcast_requests` operation.
**Cause**: Request fetching refactored to support multi-rank broadcasting in TP.
**Impact**: +500-1000 us/step in TP configurations.
**Mitigation**: Optimize broadcast path; batch request state updates.

### Pattern 2: Increased Kernel Launch Count
**Symptom**: 3-5x more `cudaLaunchKernel` calls per step, similar GPU time.
**Cause**: Operations that were fused or graph-captured are now individual launches.
**Impact**: +50-100 us/step from launch overhead alone.
**Mitigation**: Re-fuse kernels; extend CUDA graph capture scope.

### Pattern 3: New Bookkeeping Operations
**Symptom**: New NVTX ranges like `_write_finish_reasons`, `handle_additional_outputs`.
**Cause**: New features added to the inference loop without overhead budgeting.
**Impact**: +100-200 us/step each.
**Mitigation**: Defer non-critical bookkeeping to async paths; batch updates.

### Pattern 4: Flashinfer JIT Warmup Masquerading as Inference
**Symptom**: Massive elementwise/reduce kernel counts in "steady state" analysis.
**Cause**: Analysis window includes flashinfer JIT compilation phase.
**Impact**: False positive — not a real regression.
**Detection**: Forward steps identified by allreduce pattern do NOT contain these kernels.
**Fix**: Use allreduce-based iteration isolation, not kernel density or time windows.

### Pattern 5: Context-Only Bottleneck (Masked by Aggregate)
**Symptom**: Aggregate metrics below threshold (e.g., GPU idle 25%), but context iterations have 50% GPU idle.
**Cause**: Generation iterations (healthy, CUDA graphs) dilute the context-phase bottleneck.
**Detection**: Per-phase analysis in Detection phase catches this.
**Fix**: Optimize context-phase host preparation (`_prepare_tp_inputs` eager path).

---

## Pitfalls

### 1. shortName is an Integer ID
In `CUPTI_ACTIVITY_KIND_KERNEL`, `shortName` is an integer referencing `StringIds.id`. Always join:
```sql
SELECT s.value, COUNT(*), SUM(k.end - k.start)/1000.0
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE k.start >= ? AND k.start < ?
GROUP BY s.value ORDER BY 3 DESC
```

### 2. NVTX textId vs text
Most NVTX events have `textId` (integer) but NULL `text`. Join with StringIds:
```sql
SELECT s.value, n.start, n.end
FROM NVTX_EVENTS n
JOIN StringIds s ON n.textId = s.id
WHERE s.value LIKE '%_forward_step%'
```

### 3. Duplicate NVTX Ranges from TP Ranks
In TP configurations, each rank reports NVTX ranges independently. De-duplicate by grouping entries within 100us of each other.

### 4. Negative Inter-Step Gaps
When TP ranks report overlapping NVTX ranges, `gap = next_start - prev_end` can be negative. Use the maximum end time when de-duplicating.

### 5. Benchmark Window Selection
The allreduce-based window captures context+generation phases; steady-state NVTX filtering captures generation-only. Both are valid; use the appropriate one for your comparison goal.

---

## nsys SQLite Schema Reference

### Key Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | GPU kernel executions | `start`, `end`, `shortName` (-> StringIds) |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | CUDA API calls | `start`, `end`, `nameId` (-> StringIds) |
| `NVTX_EVENTS` | NVTX ranged events | `start`, `end`, `textId` (-> StringIds), `text` |
| `StringIds` | String lookup table | `id`, `value` |

### Useful Queries

**Find all NVTX range names:**
```sql
SELECT DISTINCT s.value, COUNT(*)
FROM NVTX_EVENTS n
JOIN StringIds s ON n.textId = s.id
WHERE n.end > 0
GROUP BY s.value
ORDER BY COUNT(*) DESC
```

**Allreduce kernels timeline:**
```sql
SELECT (k.start - (SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL))/1e9 AS t_sec,
       (k.end - k.start)/1000.0 AS dur_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE s.value LIKE '%allreduce%'
ORDER BY k.start
```

**CUDA API breakdown during a time window:**
```sql
SELECT s.value, COUNT(*), SUM(r.end - r.start)/1000.0 AS total_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE r.start >= ? AND r.start < ?
GROUP BY s.value ORDER BY total_us DESC
```

---

## Case Study: Llama 3.2 1B TP=2 Regression (v1.1 -> main, Feb 2025)

### Symptom
19.2% throughput regression: 445.63 -> 360.02 req/sec.

### False Starts
1. **Whole-trace analysis** attributed regression to massive elementwise/reduce kernels -> flashinfer JIT, not inference.
2. **CUDA graph analysis** suggested graph usage changed -> both versions use similar patterns.
3. **Kernel density windowing** selected wrong time window -> captured JIT phase.

### Correct Analysis (Detection + Root Cause)

**Detection** confirmed host overhead is the bottleneck (GPU idle ratio high, GPU kernels unchanged).

**Root Cause** via allreduce + NVTX analysis:
1. Allreduce_fusion grouping (66 per iteration) isolated forward steps
2. GPU kernel profiles were **identical** between versions
3. Per-step wall time: 3,317 us -> 3,978 us (+20%)
4. Inter-step gap P50: 2,543 us -> 4,468 us (+76%)

### Root Cause Breakdown

| Source | Delta (us/step) | Notes |
|--------|----------------|-------|
| _update_requests | +310 | Nearly doubled |
| _fetch_new_requests | +234 | 36 -> 270 us (+643%) |
| broadcast_requests | +250 | NEW operation for TP sync |
| prepare_resources | +156 | +81% |
| Kernel launches | +57 | 3.5x more individual launches |
| _process_requests | -666 | Improved |
| _sample_async | -443 | Improved |

### Lesson
Regression was entirely in the **request management layer** between forward steps. GPU computation was unchanged. Structural iteration isolation and steady-state NVTX comparison were essential for the correct root cause.

---

## Handoff to Optimization

When analysis is complete and the verdict is **YES**, hand off to the `perf-host-optimization` skill with the following context:

1. **Detection verdict and evidence**: Which metrics crossed thresholds (M1-M5), whether host prep was confirmed as the bottleneck (M3b+M3c), and per-phase breakdown.

2. **NVTX-based triage** (from Root Cause output): The top regressing NVTX operations by absolute delta (us/step) guide which function to profile first with line_profiler. Map NVTX range names to source functions:
   - `_prepare_tp_inputs` → `PyTorchModelEngine._prepare_tp_inputs`
   - `_fetch_new_requests` / `broadcast_requests` → request management in `PyExecutor`
   - `_update_requests` / `_process_requests` → request lifecycle in `PyExecutor`
   - `_sample_async` → sampler pipeline

3. **Handoff data block**: Include the structured data from [references/output-format.md](references/output-format.md) (see "Handoff to Optimization" section) so the optimization skill can prioritize without re-running analysis.

---

## Reference

| File | Contents |
|------|----------|
| [references/metrics.md](references/metrics.md) | Full metric definitions, formulas, SQL queries, M3 sub-metric analysis |
| [references/thresholds.md](references/thresholds.md) | Aggregate and per-phase threshold tables |
| [references/phase-classification.md](references/phase-classification.md) | NVTX marker parsing, iteration classification, per-phase aggregation |
| [references/output-format.md](references/output-format.md) | Report template and integration JSON schema |
| [references/examples.md](references/examples.md) | Six worked scenarios (aggregate and phase-specific) |
| [references/iteration-isolation-techniques.md](references/iteration-isolation-techniques.md) | Allreduce, NVTX, and kernel-density iteration isolation techniques |
| [references/trtllm-nvtx-ranges.md](references/trtllm-nvtx-ranges.md) | TRT-LLM NVTX range reference with per-operation timings |
| [scripts/analyze_host_overhead.py](scripts/analyze_host_overhead.py) | Python script for automated root cause analysis |
