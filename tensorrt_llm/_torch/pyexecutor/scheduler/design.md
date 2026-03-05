# SimpleUnifiedScheduler Refactor: Design Document

## 1. Background

TensorRT-LLM has two scheduler implementations:

- **SimpleScheduler** (C++ bindings): The default scheduler. Uses C++ `BindCapacityScheduler`
  + `BindMicroBatchScheduler` via nanobind.
- **SimpleUnifiedScheduler** (pure Python): A Python mirror of SimpleScheduler, introduced
  for extensibility and experimentation. On main branch it follows the same two-pass
  structure as SimpleScheduler but implemented in Python.

The original two-pass Python implementation was slower due to Python interpreter overhead
and excessive Python→C++ boundary crossings. This refactor optimizes
`SimpleUnifiedScheduler` with a fused single-pass design, keeping scheduling intent and
major outputs aligned, with explicit intentional semantic differences documented in
Section 4.

## 2. Optimizations

### 2.1 Fused Single-Pass Scheduling

**Old**: Two sequential passes — capacity first, then microbatch (token budget + chunking).

```
PyCapacityScheduler.schedule_request(active_requests)
    → fitting_requests
PyMicroBatchScheduler.schedule(fitting_requests, inflight_ids)
    → context_requests, generation_requests
```

**New**: `TokenBudgetTracker` is passed into the capacity policy loop. Each request is
checked for both KV-block capacity AND token budget in one iteration. Chunking and sorting
are still performed in `tracker.finalize()`, but the separate microbatch iteration over
`fitting_requests` is eliminated.

**Impact**: Eliminates one full iteration over the fitting list.

### 2.2 Batched Block Decrements

**Old**: `decrement_reserved_blocks(req)` called per-request in first-pass loop →
O(N × W) C++ calls (N requests, W window sizes).

**New**: Deferred to `batch_decrement_list(scheduled_requests)` after the loop →
O(W) batch C++ calls using `get_remaining_blocks_to_completion_batch()`.

**Correctness**: `available_blocks` is not read during the first pass. `sync_to_dict()`
is called before the second pass starts.

### 2.3 Preview/Commit Block Reservation

**Old**: Second pass calls `enough_available_blocks(req)` then
`decrement_reserved_blocks(req)` → 2 × W C++ calls per request.

**New**: `preview_reserve(req)` checks AND caches needed blocks (1 × W C++ calls).
`commit_preview()` applies the cached decrement in pure Python.

### 2.4 Cached C++ Property Calls

| Property | Old (per request) | New |
|----------|------------------|-----|
| `req.is_disagg_generation_init_state` | Called 2× (guard + elif) | Cached as `is_disagg` once |
| `req.state_value` | Called each pass | Cached as `sv` once |
| `req.is_generation_in_progress_state` | 1 C++ call | `sv == _gen_in_progress` (Python int compare) |

### 2.5 Split Second-Pass Loops

**Old**: Combined loop over `[disagg_requests, context_requests]` with per-request
`is_disagg_generation_init_state` checks and routing.

**New**: Two typed loops — disagg loop skips `beneficial_to_skip` (never applies to
disagg) and routes directly to `fitting_disagg`; context loop skips disagg checks.

### 2.6 Single-Window Fast Path

`NoEvictScheduledBlocksManager` and `MaxUtilizationScheduledBlocksManager` detect
the common single-window case and use scalar arithmetic instead of dict iteration.

## 3. mypyc Compilation

### 3.1 Overview

`unified_scheduler.py` can be compiled with [mypyc](https://mypyc.readthedocs.io/) to
produce a native C extension (`.so`), eliminating Python interpreter overhead (attribute
lookups, frame creation, type dispatch) from the scheduling hot path.

mypyc compilation is optional and controlled by the `--mypyc` flag in `build_wheel.py`.
When not compiled, the module runs as normal Python.

### 3.2 What Gets Compiled

Only `unified_scheduler.py` is compiled — it contains all hot-path classes:
- `TokenBudgetTracker`
- `GuaranteedNoEvictPolicy`, `MaxUtilizationPolicy`
- `NoEvictScheduledBlocksManager`, `MaxUtilizationScheduledBlocksManager`
- `PyCapacityScheduler`
- `SimpleUnifiedScheduler`

Other scheduler files (`scheduler.py`, `adp_router.py`, `waiting_queue.py`) are thin
wrappers or C++ bindings that don't benefit from compilation.

### 3.3 Type Annotation Fixes for mypyc

mypyc enforces type annotations at runtime (unlike CPython which ignores them). Several
annotations were widened for compatibility:

| Change | Reason |
|--------|--------|
| `inflight_request_ids: set[int]` → `object = None` | Callers pass C++ `ReqIdsSet` (nanobind type), not Python `set` |
| `uniq_task_ids: set[int]` → `Optional[set[int]]` | Assigned `None` when PEFT is disabled |

### 3.4 Build Integration

```bash
# Standalone build (from pyexecutor/ directory):
python scheduler/setup_mypyc.py build_ext --inplace

# Via build_wheel.py:
python scripts/build_wheel.py --mypyc
```

`build_wheel.py` calls `build_pyexecutor_scheduler()` which invokes `setup_mypyc.py`.
When `--mypyc` is not set, stale `.so` artifacts are cleaned up to prevent accidental
use.

### 3.5 Profiling mypyc-Compiled Code

mypyc-compiled functions lack `__code__`, so `line_profiler` cannot hook them. The host
profiler automatically falls back to function-level timing wrappers for these targets.
Use `TLLM_LINE_PROFILER_PRESET=scheduler_hotpath` to profile the scheduler hot path.

## 4. Behavior Changes vs Main Branch

### 4.1 Intentional Semantic Changes

| # | Change | Details |
|---|--------|---------|
| 1 | **MaxUtilization pause avoidance** | When token budget is exhausted, returns `None` (stop) instead of `False` (try pause/backtrack). Avoids unnecessary pause/backtrack work when token budget is the bottleneck. |
| 2 | **`num_fitting_requests` semantics** | Now counts requests passing both capacity AND token budget, not just capacity. More accurately reflects the actual scheduled batch size. |
| 3 | **`speculation_permanently_disabled`** | New monotonic `False→True` flag. Set by executor when spec-decode acceptance rate drops below threshold. |

### 4.2 Internal Refactoring (no external semantic change)

| Change | Details |
|--------|---------|
| **Disagg request return path** | Capacity policy returns 3-tuple `(scheduled, fitting_disagg, paused)` instead of 2-tuple. `fitting_disagg` was already a separate output in `SchedulerOutput` — this is an internal plumbing change, not a new external behavior. |
| **Scheduling orchestration** | Validation, ADP routing, and drafter setup consolidated into `schedule_step()` instead of being scattered across `py_executor.py`. |

### 4.3 Preserved Behavior

| Area | Why Equivalent |
|------|----------------|
| State range check | Same conditions: disagg bypasses range, others check `_until <= sv < _after` |
| Block reservation | Same check-then-decrement logic, batched/cached |
| PEFT checks | Identical |
| `beneficial_to_skip` | Disagg always skipped it (old code had `not req.is_disagg` guard) |
| Context chunking | Same `EQUAL_PROGRESS` / `FIRST_COME_FIRST_SERVED` policies |
| Request sorting | Same LoRA-based sort in `finalize()` |

### 4.4 KV Allocation Semantics

Fused one-pass does NOT change final KV allocation semantics. `prepare_resources()`
runs on the final scheduled batch only — requests filtered by token budget never
allocate real KV blocks in either the old or new path. The savings are purely in
scheduler-side bookkeeping overhead.

## 5. Performance Results

**Experiment setting**: Llama 8B, B200 single GPU, 411 scheduling iterations.
Measured with the host profiler.

| Configuration | Total | Per-Iteration | vs Main |
|--------------|-------|---------------|---------|
| main branch | 7.16s | 17.4ms | baseline |
| Refactored (Python) | 4.33s | 10.5ms | **-39.6%** |
| Refactored (mypyc)* | 1.19s | 2.89ms | **-83.4%** |

\* mypyc measurement covers `schedule_step` (includes fetch/validate/drafter overhead beyond `schedule_request`). Approximate comparison only.

### Speedup Attribution (rough hypothesis, not precisely measured)

| Source | Estimated Contribution | Mechanism |
|--------|----------------------|-----------|
| Eliminate separate microbatch pass | Major | One fewer O(N) iteration; chunking/sorting still runs in `finalize()` |
| Reduce C++ boundary crossings | Moderate | Caching, batching, preview/commit |
| Python micro-optimizations | Minor | Local variable caching, int counters, `__slots__` |

## 6. Files Changed

| File | Change |
|------|--------|
| `scheduler/unified_scheduler.py` | Refactored TokenBudgetTracker, capacity policies, NoEvictScheduledBlocksManager, SimpleUnifiedScheduler |
| `scheduler/scheduler.py` | Removed old Python scheduling classes (moved to unified_scheduler.py) |
| `pyexecutor/py_executor.py` | Added `_prepare_and_schedule_batch_unified()` path |
| `pyexecutor/request_utils.py` | Extracted validation, ADP routing, drafter utilities |
| `pyexecutor/_util.py` | Instantiation gate: enabled when `TLLM_USE_PYTHON_SCHEDULER` is set to `1` |
| `scheduler/setup_mypyc.py` | mypyc build script for compiling `unified_scheduler.py` to native C extension |
| `scheduler/mypy_mypyc.ini` | mypy configuration for mypyc compilation (error suppressions for external types) |
| `scripts/build_wheel.py` | Added `build_pyexecutor_scheduler()` for mypyc integration via `--mypyc` flag |
| `tools/profiler/host_profile_tools/host_profiler.py` | Preset system (`scheduler_hotpath`) + timer fallback for mypyc-compiled functions |

## 7. Validation

### Recommended correctness checks

Compare scheduling outputs between `SimpleScheduler` (default) and
`SimpleUnifiedScheduler` on the same workload. At minimum, compare per-iteration:

- `len(context_requests)`, `len(generation_requests)`
- `len(paused_requests)`, `len(fitting_disagg_gen_init_requests)`
- `num_fitting_requests`
- Request ordering (verify LoRA sort and chunk partitioning match)
- Key state transitions (requests entering/leaving scheduled batch)

### Enable the refactored scheduler
```bash
export TLLM_USE_PYTHON_SCHEDULER=1
```

### Profile with scheduler preset
```bash
TLLM_USE_PYTHON_SCHEDULER=1 \
TLLM_LINE_PROFILER_PATH=./profile.txt \
TLLM_LINE_PROFILER_PRESET=scheduler_hotpath \
trtllm-bench --model <model> throughput --dataset <dataset>
```
