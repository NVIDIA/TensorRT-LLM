# Implementation Plan: SJF Waiting Queue

## Task Breakdown

### Task Group 1: Core Types & Config [UT]

**Task 1.1**: Add `SJF` to `WaitingQueuePolicy` enum in `llm_args.py`
- File: `tensorrt_llm/llmapi/llm_args.py:1856`
- Add `SJF = "sjf"` to the enum

**Task 1.2**: Add `SjfConfig` Pydantic model in `llm_args.py`
- File: `tensorrt_llm/llmapi/llm_args.py` (near `SchedulerConfig`)
- Fields: `length_median` (int, default=32768, gt=0), `time_median` (float, default=5.0, gt=0), `length_weight` (float, default=0.5, ge=0), `time_weight` (float, default=0.5, ge=0)
- Use `StrictBaseModel` per CODING_GUIDELINES.md

**Task 1.3**: Add `sjf_config` field to `SchedulerConfig`
- File: `tensorrt_llm/llmapi/llm_args.py:1892`
- `sjf_config: Optional[SjfConfig] = Field(default=None, ...)`

**Sync check:** Re-read designs/2026-03-21-sjf-waiting-queue.md. Does implementation still match?

### Task Group 2: SJFWaitingQueue Implementation [UT]

**Task 2.1**: Implement `SJFWaitingQueue` class in `waiting_queue.py`
- File: `tensorrt_llm/_torch/pyexecutor/scheduler/waiting_queue.py`
- Implement all WaitingQueue ABC methods
- Key internal state: `_requests`, `_prepended`, `_sorted`, `_arrival_times`, `_config`
- Scoring: `_compute_score()` with lazy sort via `_ensure_sorted()`
- Methods:
  - `add_request`: append to `_requests`, record fallback arrival time, set `_sorted=False`
  - `add_requests`: call `add_request` for each
  - `pop_request`: return from `_prepended` first, then sorted `_requests`
  - `peek_request`: same priority as pop but without removal
  - `prepend_request`: insert at front of `_prepended`
  - `prepend_requests`: extend front of `_prepended` (preserve input order)
  - `remove_by_ids`: filter both `_prepended` and `_requests`, clean `_arrival_times`
  - `__bool__`: `bool(self._prepended) or bool(self._requests)`
  - `__len__`: `len(self._prepended) + len(self._requests)`
  - `__iter__`: chain `_prepended` + sorted `_requests`

**Task 2.2**: Update `create_waiting_queue()` factory function
- Add `sjf_config` parameter
- Add `SJF` branch that creates `SJFWaitingQueue(sjf_config)`

**Task 2.3**: Update `__init__.py` exports
- File: `tensorrt_llm/_torch/pyexecutor/scheduler/__init__.py`
- Add `SJFWaitingQueue` to imports and `__all__`

**Sync check:** Re-read designs/2026-03-21-sjf-waiting-queue.md. Does implementation still match?

### Task Group 3: Plumbing [UT]

**Task 3.1**: Update `_util.py` to pass `sjf_config`
- File: `tensorrt_llm/_torch/pyexecutor/_util.py:1327`
- Extract `sjf_config` from `scheduler_config.sjf_config`
- Pass to `create_waiting_queue(policy, sjf_config=sjf_config)`

**Task 3.2**: Resolve `sjf_config` in `_util.py` and pass pre-built queue to `PyExecutor`
- File: `tensorrt_llm/_torch/pyexecutor/_util.py:1327`
- Instead of modifying PyExecutor's signature, resolve sjf_config in `_util.py` and pass it to `create_waiting_queue()` alongside the policy
- PyExecutor already receives `waiting_queue_policy` and calls `create_waiting_queue()` at line 499-500
- **Updated approach**: pass `sjf_config` alongside `waiting_queue_policy` to PyExecutor, who passes both to `create_waiting_queue()`

**Task 3.3**: Check `api_stability` test impact
- File: `tests/unittest/api_stability/`
- Adding `SjfConfig` and `sjf_config` to `SchedulerConfig` may require snapshot update
- If api_stability tests exist for SchedulerConfig, update the snapshot

**Sync check:** Re-read designs/2026-03-21-sjf-waiting-queue.md. Does implementation still match?

### Task Group 4: Unit Tests [UT]

**Task 4.1**: Add SJF unit tests to `test_waiting_queue.py`
- File: `tests/unittest/_torch/executor/test_waiting_queue.py`
- Test cases:
  - Short requests pop before long requests
  - Equal length: FCFS-like behavior (earlier arrival pops first)
  - Wait time aging: long request eventually gets priority
  - add/pop/peek/remove/prepend interface correctness
  - Empty queue raises IndexError
  - `__bool__`/`__len__`/`__iter__` work with dual-list
  - `prepend_requests` order preserved
  - peek→pop consistency (same item returned)
  - Factory function with SJF policy
  - Custom `SjfConfig` parameters respected

**Sync check:** Re-read designs/2026-03-21-sjf-waiting-queue.md. Does implementation still match?

## Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Data structure | heap vs list+sort | list+sort | Queue <1000, sort ~0.05ms vs prefill ~10-100ms; list simpler for remove/prepend |
| Score compute timing | per-comparison vs batch | batch (lazy sort) | Avoid time.time() in every comparison; dirty flag ensures peek/pop consistency |
| arrival_time storage | modify RequestQueueItem vs side dict | side dict | Don't modify shared data class, cleaner separation |
| prepend handling | merge into sorted list vs separate list | separate list | Prepended items were already highest priority, re-sorting risks unnecessary churn |
| Config exposure | constructor params vs config object | SjfConfig Pydantic model | Consistent with TRT-LLM patterns, YAML-serializable |

## Verification Plan

| Category | Value |
|----------|-------|
| Benchmarking tool | aiperf (via trtllm-serve) |
| Hardware | GB200 on OCI cluster |
| Model | DeepSeek-R1 (or available model at OCI) |
| Model configuration | FP8, TP as available |
| Workload | MLPerf DeepSeek-R1 eval dataset (4388 requests, natural length distribution) |
| Target metric | P99 TTFT, output throughput (tok/s) |
| Baseline | FCFS (same config, only `waiting_queue_policy: fcfs`) |
| Success criteria | Improvement in P99 TTFT or throughput in variable-length workload, no regression in uniform workload |
| Regression scope | FCFS behavior unchanged, all existing waiting_queue tests pass |
| UT scope | SJF ordering, aging, interface correctness, config validation |
| IT scope | aiperf A/B test on OCI with FCFS vs SJF |
| Concurrency levels | 8, 32, 64, 128 |
