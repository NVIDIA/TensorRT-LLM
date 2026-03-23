# UnifiedScheduler Refactor: Design Document

## 1. Background

TensorRT-LLM has two scheduler implementations:

- **SimpleScheduler** (C++ bindings): The default scheduler. Uses C++ `BindCapacityScheduler`,
  `BindMicroBatchScheduler` via nanobind.
- **UnifiedScheduler** (pure Python): A Python mirror of SimpleScheduler, introduced
  for extensibility and experimentation. On main branch it follows the same two-pass
  structure as SimpleScheduler but implemented in Python.

The original two-pass Python implementation was slower due to Python interpreter overhead
and excessive Python→C++ boundary crossings. This refactor optimizes
`UnifiedScheduler` with a fused single-pass design, keeping scheduling intent and
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
- `UnifiedScheduler`

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

mypyc-compiled functions lack `__code__`, so `line_profiler` cannot hook them.

## 4. Behavior Changes vs Main Branch

### 4.1 Intentional Semantic Changes

#### 4.1.1 Fused first-pass break produces a lighter resource state

When token budget is exhausted in the first pass, the fused path breaks the
loop immediately. Requests after the break point — generation, context,
and disagg alike — are never evaluated. This affects both
`MaxUtilizationPolicy` (token failure returns `None` → break) and
`GuaranteedNoEvictPolicy` (generation token failure → break classification
loop).

Because the failing generation request is never admitted, it does not consume
KV blocks, request slots, or PEFT pages. The second pass (in
GuaranteedNoEvict) and downstream scheduling therefore see a **lighter
resource state** than the old two-pass path, where capacity admitted all
generation unconditionally and microbatch dropped the excess afterward.

This produces two kinds of differences vs the old path:

**a) `paused_requests` may have fewer entries (MaxUtilization only).**
The old path could pause requests to make room for later requests that
microbatch would then drop anyway — wasted work. The fused path avoids this.

**b) Second-pass requests may see more available resources
(GuaranteedNoEvict).** Because the failing generation request is never
admitted, it does not consume KV blocks, PEFT pages, or token budget. The
second pass — which processes both context and disagg-init requests — sees
a lighter state than the old path. This can admit context or disagg that the
old path would have blocked. (Disagg/context after the break point is still
never reached — only those classified before the break benefit.)

The two request types have different practical thresholds:
- **Extra context** requires speculative decoding or beam search, where each
  generation request consumes multiple tokens (e.g., `beam_width +
  draft_tokens`), creating enough token headroom for context. With standard
  beam=1 and no speculation, generation requests are 1 token each, leaving
  near-zero headroom.
- **Extra disagg-init** can happen with any configuration, because disagg
  bypasses token accounting — it only needs KV blocks and PEFT pages. The
  lighter KV/PEFT state from the unadmitted generation is sufficient.

These differences all result in **equal or better token budget utilization**
than the old path. The old path's behavior was an artifact of the two-pass
ordering (capacity admits everything, microbatch iterates generation-first),
not a deliberate scheduling priority.

**Example — MaxUtilization pause avoidance (token_budget=100):**

```
Old two-pass pipeline:

  Capacity (MaxUtil): iterates ALL requests, admits/pauses based on KV blocks only
    → Request A: KV ok → admit
    → Request B: KV ok → admit
    → Request C: KV fail → pause older request, retry → admit
    → Request D: KV ok → admit
    Result: fitting_requests = [A, B, C, D], paused = [old_req]

  Microbatch: iterates fitting_requests with token budget
    → A: 30 tokens → ok (30/100)
    → B: 80 tokens → 30+80=110 > 100 → break
    Result: scheduled = [A], B/C/D dropped silently

Fused single-pass pipeline:

  Capacity + Token (MaxUtil): iterates with fused check
    → Request A: KV ok, 30 tokens ok → admit
    → Request B: KV ok, 30+80=110 > 100 → token fail → None → BREAK
    → Request C: NEVER REACHED
    → Request D: NEVER REACHED
    Result: fitting = [A], paused = []
```

Paused requests differ ([] vs [old_req]). Scheduled output is the same.

**Example — GuaranteedNoEvict second pass benefits from lighter state:**

Setup: speculative decoding with 7 draft tokens → each generation request
consumes `beam_width(1) + draft_tokens(7) = 8 tokens`. Token budget = 100.
active_requests = [Gen_1..Gen_12, Ctx_X(4 tokens, chunked), Disagg_Y, Gen_13, Disagg_Z].

```
Old two-pass pipeline:

  Capacity first pass: no token budget — classifies ALL requests
    → Gen_1..Gen_12: generation → scheduled, blocks decremented (12 requests)
    → Ctx_X:   context → pending_requests
    → Disagg_Y: disagg → pending_dis_gen_init
    → Gen_13:  generation → scheduled, blocks decremented
    → Disagg_Z: disagg → pending_dis_gen_init

  batch_decrement_list([Gen_1..Gen_13]) → all 13 consume KV blocks

  Capacity second pass: evaluates pending against remaining blocks
    → Disagg_Y: blocks ok (after 13 gen consumed) → fitting_disagg
    → Disagg_Z: blocks ok → fitting_disagg
    → Ctx_X: blocks ok → added to scheduled
    Result: fittingRequests = [Gen_1..Gen_13, Ctx_X]
            fitting_disagg = [Disagg_Y, Disagg_Z]

  Microbatch: iterates fittingRequests (generation-first order)
    → Gen_1..Gen_12: 12×8 = 96 tokens → ok (96/100)
    → Gen_13: 96+8 = 104 > 100 → break
    → Ctx_X: NEVER REACHED
    Result: scheduled = [Gen_1..Gen_12]
            fitting_disagg = [Disagg_Y, Disagg_Z] (unchanged)

Fused single-pass pipeline:

  First pass: token budget checked inline
    → Gen_1..Gen_12: 96 tokens → admitted, blocks decremented
    → Ctx_X:   context → pending_requests (classified, not token-checked)
    → Disagg_Y: disagg → pending_dis_gen_init (classified)
    → Gen_13:  96+8 = 104 > 100 → break
    → Disagg_Z: NEVER REACHED (after break point)

  batch_decrement_list([Gen_1..Gen_12]) → only 12 gen consume KV blocks
                                          (Gen_13's blocks NOT consumed)

  Second pass: processes pending against remaining budget + lighter blocks
    → Disagg_Y: blocks ok (lighter state) → fitting_disagg
    → Ctx_X: 96+4 = 100 ≤ 100, blocks ok → admitted
    Result: scheduled = [Gen_1..Gen_12, Ctx_X]
            fitting_disagg = [Disagg_Y]
```

Differences vs the old path:
- Ctx_X admitted (100/100 tokens) vs dropped (96/100). Gen_13 not scheduled
  in either path.
- Disagg_Y evaluated against lighter block state (Gen_13's blocks not
  consumed). May fit where the old path would have blocked it.
- Disagg_Z deferred (after break point). Retried next iteration.

All differences are benign — better utilization for context/disagg that fit,
deferred requests retry next iteration. Extra context requires speculative
decoding or beam search (needs token headroom from multi-token generation).
Extra disagg can happen with any configuration — disagg bypasses token
accounting and only needs the lighter KV/PEFT state.

#### 4.1.2 `num_fitting_requests` semantics

Now counts requests admitted by the fused capacity + token-budget path
(`TokenBudgetTracker._num_fitting`), which is **more accurate** than the old
value. In `SimpleScheduler`, `num_fitting_requests` was
`len(fitting_requests)` from the capacity pass only — it included requests
that capacity admitted but microbatch would later drop for exceeding the
token budget. The new count reflects requests that passed both KV-block
capacity AND token-budget checks.

Note: this count is still computed before late pruning, so it can overcount
in two edge cases:

1. **Chunking**: `_num_fitting` is incremented when `try_add_context()`
   accepts a request, but `finalize()` may later drop requests with
   `context_chunk_size == 0` without decrementing.
2. **Post-scheduler filters**: `py_executor._schedule()` passes
   `num_fitting_requests` through unchanged after ADP balance or batch
   waiting may have shrunk the context batch.

### 4.2 Bug Fixes vs Main

#### 4.2.1 MaxUtilization PEFT page accumulation

Fixes a pre-existing bug in main's Python `MaxUtilizationPolicy` where
`num_scheduled_peft_pages` was passed by value to `_try_scheduling_request()`
and never accumulated across requests. Every request saw
`num_scheduled_peft_pages = 0`, so cumulative PEFT page limits were not
enforced. The same bug exists on main's `scheduler.py`.

Now returns the updated total from `_try_scheduling_request()`, matching the
C++ reference (`capacityScheduler.cpp` `trySchedulingRequestMaxUtilization`)
which passes by reference. `GuaranteedNoEvictPolicy` was already correct
(accumulates `claimed_peft_pages` locally).

This can change `context_requests` and `generation_requests` vs main on
workloads that use LoRA with MaxUtilization scheduling, because the old path
would over-admit requests that exceed the cumulative PEFT page budget.

### 4.3 Internal Refactoring (no external semantic change)

| Change | Details |
|--------|---------|
| **Disagg request return path** | Capacity policy returns 3-tuple `(scheduled, fitting_disagg, paused)` instead of 2-tuple. `fitting_disagg` was already a separate output in `SchedulerOutput` — this is an internal plumbing change, not a new external behavior. |
| **Drop-in interface** | `UnifiedScheduler.schedule_request()` returns the same `SchedulerOutput` as `SimpleScheduler`. `py_executor.py` uses a single code path for both schedulers. |

### 4.4 Preserved Behavior

| Area | Why Equivalent |
|------|----------------|
| State range check | Same conditions: disagg bypasses range, others check `_until <= sv < _after` |
| Block reservation | Same check-then-decrement logic, batched/cached |
| PEFT checks (`GuaranteedNoEvictPolicy`) | Identical to main (accumulates `claimed_peft_pages` locally) |
| `beneficial_to_skip` | Disagg always skipped it (old code had `not req.is_disagg` guard) |
| Context chunking | Same `EQUAL_PROGRESS` / `FIRST_COME_FIRST_SERVED` policies |
| Request sorting | Same LoRA-based sort in `finalize()` |

Note: `MaxUtilizationPolicy` PEFT behavior changed vs main — see Section 4.2.1.

### 4.5 KV Allocation Semantics

`prepare_resources()` runs on the final scheduled batch only — requests
filtered by token budget never allocate real KV blocks in either path.

However, the fused path's lighter resource state (Section 4.1.1) means:
- The main scheduled batch may contain additional context requests that the
  old path would have dropped (GuaranteedNoEvict). KV allocation for these
  extra requests is correct — they passed KV block checks in the second pass.
- `fitting_disagg_gen_init_requests` may differ. Those requests are fed into
  `_prepare_disagg_gen_init()` which prepares KV resources outside the main
  `prepare_resources()` batch.

## 5. Performance Results

**Experiment setting**: Llama 8B, B200 single GPU, 411 scheduling iterations.
Measured with the host profiler.

| Configuration | Total | Per-Iteration | vs Main |
|--------------|-------|---------------|---------|
| main branch | 7.16s | 17.4ms | baseline |
| Refactored (Python) | 4.33s | 10.5ms | **-39.6%** |
| Refactored (mypyc)* | 1.19s | 2.89ms | **-83.4%** |

\* mypyc measurement covers `schedule_request` only (capacity + token budget scheduling). Fetch, validate, and drafter setup run in py_executor (same path for both schedulers).

### Speedup Attribution (rough hypothesis, not precisely measured)

| Source | Estimated Contribution | Mechanism |
|--------|----------------------|-----------|
| Eliminate separate microbatch pass | Major | One fewer O(N) iteration; chunking/sorting still runs in `finalize()` |
| Reduce C++ boundary crossings | Moderate | Caching, batching, preview/commit |
| Python micro-optimizations | Minor | Local variable caching, int counters, `__slots__` |

## 6. Files Changed

| File | Change |
|------|--------|
| `scheduler/unified_scheduler.py` | Refactored TokenBudgetTracker, capacity policies, NoEvictScheduledBlocksManager, UnifiedScheduler |
| `scheduler/scheduler.py` | Removed old Python scheduling classes (moved to unified_scheduler.py) |
| `pyexecutor/py_executor.py` | No scheduler-specific code paths — same `_prepare_and_schedule_batch()` for both |
| `pyexecutor/_util.py` | Instantiation gate: `UnifiedScheduler` when `SchedulerConfig(use_python_scheduler=True)` |
| `scheduler/setup_mypyc.py` | mypyc build script for compiling `unified_scheduler.py` to native C extension |
| `scheduler/mypy_mypyc.ini` | mypy configuration for mypyc compilation (error suppressions for external types) |
| `scripts/build_wheel.py` | Added `build_pyexecutor_scheduler()` for mypyc integration via `--mypyc` flag |
| `tools/profiler/host_profile_tools/host_profiler.py` | Added `TLLM_LINE_PROFILER_NO_DEFAULTS` env var to disable default profiler targets |

## 7. Validation

### Recommended correctness checks

Compare scheduling outputs between `SimpleScheduler` (default) and
`UnifiedScheduler` on the same workload.

**All policies — must match:**
- Request ordering (verify LoRA sort and chunk partitioning match)
- Key state transitions (requests entering/leaving scheduled batch)

**All policies — expected to differ:**
- `num_fitting_requests`: now counts capacity + token-budget admissions, not
  just capacity (Section 4.1.2)

**GuaranteedNoEvict — must match:**
- `len(generation_requests)`
- `len(paused_requests)` (this policy does not pause)

**GuaranteedNoEvict — expected to differ when token budget is the bottleneck:**
- `len(context_requests)`: may have more entries with speculative decoding or
  beam search (needs token headroom from the lighter state) (Section 4.1.1b)
- `len(fitting_disagg_gen_init_requests)`: may have more entries with any
  configuration (disagg bypasses token accounting, only needs lighter KV/PEFT
  state) (Section 4.1.1b). Requests after the break point are deferred (fewer).

**MaxUtilization — must match:**
- `len(context_requests)`, `len(generation_requests)` (single loop breaks,
  no second pass to admit extra work)

**MaxUtilization — expected to differ when token budget is the bottleneck:**
- `len(paused_requests)`: fewer — the fused path avoids wasted pause/backtrack
  (Section 4.1.1a)
- `len(fitting_disagg_gen_init_requests)`: may have fewer entries — disagg
  after the break point is deferred (Section 4.1.1b)

**LoRA + MaxUtilization — expected to differ:**
- `len(context_requests)`, `len(generation_requests)`: may differ due to
  PEFT page accumulation bug fix (Section 4.2.1) — the old path over-admitted
  requests that exceed the cumulative PEFT page budget

### Enable the refactored scheduler
```python
from tensorrt_llm.llmapi import LLM, SchedulerConfig

llm = LLM(model, scheduler_config=SchedulerConfig(use_python_scheduler=True))
```

### Profile with trtllm-serve (no default profiler targets)
```yaml
# config.yaml
scheduler_config:
  use_python_scheduler: true
```

```bash
TLLM_LINE_PROFILER_PATH=./profile.txt \
TLLM_LINE_PROFILER_NO_DEFAULTS=1 \
TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.scheduler.unified_scheduler.UnifiedScheduler.schedule_request" \
trtllm-serve <model> --config config.yaml
```
