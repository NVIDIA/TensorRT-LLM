---
name: perf-host-optimization
description: Profiles and optimizes TensorRT-LLM host/CPU overhead using line_profiler (with nsys support planned). Runs iterative profile-analyze-optimize-validate rounds. Use when GPU utilization is low or optimizing PyExecutor throughput.
license: Apache-2.0
tags:
  - optimization
  - profiling
  - host-overhead
  - line-profiler
  - inference
metadata:
  author: NVIDIA Corporation
---

# Host Performance Optimization Skill

Automates detection and optimization of host-side (CPU) overhead in TensorRT-LLM's PyTorch backend.

## When to Use

- GPU utilization is low during inference (CPU bottleneck suspected)
- User asks to reduce host overhead or CPU latency
- Optimizing PyExecutor throughput (requests/sec)
- Need line-by-line profiling of specific Python functions

### Detecting a CPU Bottleneck

line_profiler measures *where* CPU time is spent but not *whether* CPU is the bottleneck.
When user asks for confirmation, or there is no clear ending conditions for optimizations, use **nsys** (system-level trace) to confirm CPU is the limiting factor:

| Indicator (from nsys) | Threshold | Meaning |
|----------------------|-----------|---------|
| GPU idle gaps between kernels | >30% of step time | CPU can't feed GPU fast enough |
| `cudaLaunchKernel` API time | >10% of total time | Kernel launch overhead |
| NVTX `_prepare_tp_inputs` range | >50% of step time | Input preparation dominates |
| GPU SM utilization | <60% with no comm/sync bottleneck | CPU-bound inference |

If nsys report is not available, use a rough heuristic: if doubling the batch size does not
proportionally increase GPU utilization or throughput, CPU overhead is likely the bottleneck.

### Using Analysis Skill Results

If the `perf-host-analysis` skill has already been run, use its output to skip the confirmation step and prioritize targets:

1. **Detection verdict**: If YES with host_prep_confirmed, start with `_prepare_tp_inputs`.
2. **NVTX triage** (from Root Cause): The `top_regressing_ops` in the handoff data block maps NVTX range names to source functions. Profile the function with the largest absolute delta first.
3. **Cross-function triage**: When the top NVTX regression is NOT in `_prepare_tp_inputs` (e.g., `_fetch_new_requests`, `broadcast_requests`, `_update_requests`), target that function's source file directly instead of defaulting to `_prepare_tp_inputs`. See [references/trtllm-nvtx-ranges.md](../perf-host-analysis/references/trtllm-nvtx-ranges.md) for the NVTX-to-source mapping.

---

## Profiling Setup

### line_profiler (Primary Method)

**Environment Variables:**
- `TLLM_LINE_PROFILER_ENABLED=True` — Enable the profiler
- `TLLM_LINE_PROFILER_PATH` — Output file path
- `TLLM_LINE_PROFILER_FUNCTIONS` — Additional functions to profile (comma-separated)

**Function specification format:**
```bash
# Class methods: module.path.ClassName.method_name
TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.model_engine.PyTorchModelEngine._prepare_tp_inputs"

# Standalone functions: module.path::function_name
TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.sampler::_group_requests_by_strategy_key"

# Multiple functions (comma-separated)
TLLM_LINE_PROFILER_FUNCTIONS="module.Class.method1,module.Class.method2"
```

### CPU Affinity (Environment Factor)

CPU core affinity can significantly affect host overhead measurements,
especially on multi-socket systems (e.g., B300). Pinning processes to cores
near the GPU's NUMA node reduces cross-socket memory access latency.

- Check current affinity: `taskset -p <pid>` or `numactl --show`
- Pin to local NUMA node: `numactl --cpunodebind=<node> --membind=<node>`
- **Impact**: Up to 2x difference in host overhead on B300 systems

When comparing profiling results across runs, ensure CPU affinity is consistent.
Do not externally modify the affinity, unless user requires to do this to examine the affects of this part.
Document the affinity setting in each round's report if it varies.

### Workspace & Suffix Management

Each profiling run should have a unique suffix to track progress across rounds:
```bash
EXTRA_SUFFIX=round0_baseline bash profile.sh
EXTRA_SUFFIX=round1_eliminate_redundant_iter bash profile.sh
```

---

## Autonomous Optimization Loop

Run N rounds (default 3) of the following cycle:

```
FOR round = 1 to MAX_ROUNDS:

  1. PROFILE (with Drill-Down)
  2. ANALYZE (Multi-Option)
  3. OPTIMIZE (Apply Change)
  4. TEST (Unit Test Validation)
  5. VALIDATE (Re-Profile)
  6. REPORT

END FOR → FINAL SUMMARY
```

### Phase 1: PROFILE (with Drill-Down)

- Run workload with profiler enabled
- Parse output: identify functions with highest Total time and lines with highest % Time
- **CRITICAL: Drill down into sub-functions that are not yet profiled** (see below)

#### Drill-Down Profiling

The default profiler covers top-level executor functions but **not all sub-functions**. When a profiled function shows most time in a single sub-call, you must drill down.

**When:** A single line consumes >80% of a function's time calling an unprofiled sub-function:
```
Line #      Hits         Time    Per Hit   % Time  Line Contents
==============================================================
  2848      4100  59200000000.0  14439024.4   98.7      output = self.model_engine.forward(...)
```

**How:**
1. Identify the sub-function's full qualified path (e.g., `tensorrt_llm._torch.pyexecutor.model_engine.PyTorchModelEngine._prepare_tp_inputs`)
2. Add it to `TLLM_LINE_PROFILER_FUNCTIONS`
3. Re-profile to get line-level data inside it
4. Now analyze the **inner** hotspots

For common drill-down targets, see [references/hot-path-files.md](references/hot-path-files.md).

### Phase 2: ANALYZE (Multi-Option)

For the chosen hotspot:

1. **Identify** the top hotspots by **absolute time** (not just %) within the target function
2. **Classify** each hotspot by type. Summary table:

| Type | Indicators | Severity |
|------|------------|----------|
| **SYNC** | `.item()`, `.cpu()`, `synchronize()` | Critical |
| **ALLOC** | `torch.zeros/empty/tensor()` in loops, `.clone()` | High |
| **PYLOOP** | `for x in collection:` with many iterations | High |
| **REDUNDANT_ITER** | Multiple passes over the same collection | High |
| **DEAD_WORK** | Object construction whose results are always discarded | High |
| **CONTAINER** | Dict/set lookups in hot loops | Medium |
| **FUNCALL** | Repeated method/property calls | Medium |
| **GIL** | Lock/queue contention | Medium |
| **GC** | Periodic latency spikes, non-deterministic pauses | Low |
| **COMPUTE** | Actual computation (may not be optimizable) | Low |

For detailed classification with code examples, see [references/hotspot-classification.md](references/hotspot-classification.md).

3. **Propose 2-4 optimization options** in a table:

| Option | Description | Estimated Savings | Risk | Complexity |
|--------|-------------|-------------------|------|------------|
| A | ... | ... | Low/Med/High | ... |
| B | ... | ... | ... | ... |

4. **Select the best option** and explain reasoning (prefer high-savings + low-risk)

For optimization patterns by type, see [references/optimization-patterns.md](references/optimization-patterns.md).

### Phase 3: OPTIMIZE (Apply Change)

- Apply the selected code change with Edit tool
- **One optimization per round** — keep changes minimal and targeted
- Record the exact change (file, line range, before/after) for potential rollback

### Phase 4: TEST (Unit Test Validation)

**Mandatory** after each optimization. Find and run related UTs to verify correctness.

**Finding related tests:**
```bash
# Search by modified file name
grep -rl "model_engine\|PyTorchModelEngine" tests/unittest/_torch/executor/

# Search by modified function name
grep -rl "_prepare_tp_inputs\|prepare_inputs" tests/
```

**Running tests:**
```bash
# Run specific test file with stop-on-first-failure
pytest tests/unittest/_torch/executor/test_pytorch_model_engine.py -v -x --timeout=120

# Run specific test method
pytest tests/unittest/_torch/executor/test_pytorch_model_engine.py::PyTorchModelEngineTestCase::test_position_id_preparation -v -x
```

For the full UT-to-file mapping, see [references/hot-path-files.md](references/hot-path-files.md).

**If tests fail:**
1. Read the failure message
2. Rollback immediately (`git checkout -- <file>`)
3. Analyze why the optimization broke correctness
4. Try the next-best option from Phase 2

### Phase 5: VALIDATE (Re-Profile)

- Re-run profiler with identical workload, using suffix `round<N>_<description>`
- Compare three things:
  1. Did the **target hotspot** time decrease?
  2. Did the **overall function** Total time decrease?
  3. Did **benchmark metrics** (TPOT, throughput) improve?

**If regression detected** (function time increased or metrics worsened):
- The "optimization" may have triggered a CPython pitfall — see [references/optimization-patterns.md](references/optimization-patterns.md) (CPython Pitfalls section)
- Rollback and try the next-best option from Phase 2

### Phase 6: REPORT

Log for this round:
- Round number
- Hotspot location (file:line) and classification
- Optimization applied (with before/after code summary)
- Time delta: function Total time before → after
- Benchmark delta: TPOT, throughput before → after

---

## Reading Profile Output

```
Timer unit: 1e-06 s
Total time: 1.234 s
File: /path/to/file.py
Function: my_function at line 100

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   100                                           def my_function(self):
   101       500    890000.0   1780.0     72.1       result = tensor.item()
   102       500    234567.0    469.1     19.0       return result
```

**How to read effectively:**
1. Start with **Total time** for each function — this is the overall budget
2. Sort lines mentally by **absolute Time**, not just % Time (3% of a 60s function = 1.8s)
3. Check **Hits count** to understand iteration patterns:
   - Hits = 2 × expected count → `for x in range(1):` loop overhead (2 hits = enter + exit check)
   - Hits ≫ expected → the line is inside a nested loop
4. Look for **repeated patterns**: if 10 lines each take 3% in a loop body, the loop itself costs 30%

---

## Stopping Criteria

Stop the optimization loop when:
1. **Iteration limit reached**: Completed N rounds (default 3)
2. **No actionable hotspots**: Top hotspots are pure GPU compute (COMPUTE type)
3. **Diminishing returns**: < 5% improvement in last 2 rounds
4. **Risk threshold**: Further optimizations require architectural changes (e.g., Cython, struct-of-arrays)
5. **Test failures**: Cannot find an optimization that passes UTs

**Primary success metric**: Benchmark throughput (requests/sec or tokens/sec) as measured by the profiling script. line_profiler time reductions are leading indicators, but throughput is the ground truth — a function-level speedup that doesn't improve throughput is not a real win.

---

## Final Summary Output

The final report should include:
- **Rounds executed**: Number of profile-optimize cycles completed
- **Cumulative improvement**: Total host time reduction (percentage and absolute)
- **Benchmark metrics**: Before/after comparison table (TPOT, throughput, ITL, E2EL)
- **Optimizations applied**: List of changes with file:line locations and classification
- **Failed attempts**: Any optimizations tried and reverted (with why)
- **Remaining hotspots**: Top bottlenecks that couldn't be optimized (with classification)
- **Recommendations**: Suggested follow-up for architectural changes if needed

For a concrete multi-round example, see [references/examples.md](references/examples.md).

---

## Reference Files

| File | Contents |
|------|----------|
| [references/optimization-patterns.md](references/optimization-patterns.md) | Pattern catalog by hotspot type + CPython pitfalls |
| [references/hotspot-classification.md](references/hotspot-classification.md) | Extended per-type indicators and code examples |
| [references/hot-path-files.md](references/hot-path-files.md) | Key file tables, drill-down targets, UT mapping |
| [references/examples.md](references/examples.md) | Usage examples and multi-round walkthrough |
| [trtllm-nvtx-ranges.md](../perf-host-analysis/references/trtllm-nvtx-ranges.md) | TRT-LLM NVTX range reference (from analysis skill) — maps range names to source functions |
