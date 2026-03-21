# Review: SJF Implementation (Task Groups 1-4)

## Summary

The implementation faithfully follows the design doc and implementation plan. All six files specified in the plan have been modified/created. The `SJFWaitingQueue` correctly implements all `WaitingQueue` ABC methods, the scoring formula matches the design, the dirty-flag lazy sort works correctly, the `prepend_requests` reversal semantics are correct, and the plumbing from `SchedulerConfig` through `_util.py` and `PyExecutor` to `create_waiting_queue` is complete. Tests cover all critical paths listed in the plan.

Overall quality is high. The findings below are mostly hygiene and robustness items.

## Findings

### Must-Fix

**MF-1: Missing NVIDIA copyright headers on both new files**

Per `AGENTS.md`: "NVIDIA copyright header on ALL new files". Both `waiting_queue.py` and `test_waiting_queue.py` are new files and are missing the SPDX copyright header.

Files:
- `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/scheduler/waiting_queue.py` (line 1, starts directly with `import`)
- `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tests/unittest/_torch/executor/test_waiting_queue.py` (line 1, starts with docstring)

Add the standard header:
```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**MF-2: Copyright year not updated on modified `__init__.py`**

File: `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/scheduler/__init__.py` (line 1)

Current: `Copyright (c) 2022-2025`
Should be: `Copyright (c) 2022-2026`

### Should-Fix

**SF-1: Missing type annotation for `sjf_config` parameter in `PyExecutor.__init__`**

File: `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/py_executor.py` (line 289)

Current:
```python
sjf_config=None,
```

Should be:
```python
sjf_config: Optional[SjfConfig] = None,
```

This requires adding `SjfConfig` to the import from `llm_args`. Without the type annotation, static analysis tools and IDE autocompletion cannot help callers. Every other parameter in the constructor has a type annotation.

**SF-2: `remove_by_ids` does not reset `_sorted` flag**

File: `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/scheduler/waiting_queue.py` (line 215-224)

While removing items from a sorted list preserves relative order (so `_sorted=True` is technically still valid), the time-based scores become stale over time. If `remove_by_ids` is called in a scheduling round where no new `add_request` occurs (e.g., pure cancellation), subsequent `peek_request`/`pop_request` calls will use the stale sort order. In practice this is unlikely to cause issues since new requests are typically added every iteration, but for correctness the flag should be reset:

```python
def remove_by_ids(self, request_ids: set[int]) -> None:
    self._prepended = [req for req in self._prepended if req.id not in request_ids]
    self._requests = [req for req in self._requests if req.id not in request_ids]
    for rid in request_ids:
        self._arrival_times.pop(rid, None)
    self._sorted = False  # Force re-sort to account for elapsed time
```

**SF-3: `SjfConfig` and `WaitingQueuePolicy` should have `status="prototype"` in their Field definitions**

File: `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (lines 1940-1948)

The `waiting_queue_policy` and `sjf_config` fields on `SchedulerConfig` do not specify `status="prototype"`. Since this is a new, unvalidated feature without production benchmarking data yet, these fields should be marked prototype to signal to users that the API may change:

```python
waiting_queue_policy: WaitingQueuePolicy = Field(
    default=WaitingQueuePolicy.FCFS,
    status="prototype",
    description="The waiting queue scheduling policy")

sjf_config: Optional[SjfConfig] = Field(
    default=None,
    status="prototype",
    description="Configuration for SJF waiting queue scheduling. ...")
```

### Nice-to-Have

**NTH-1: `pop(0)` on list is O(n) -- consider `collections.deque` for `_prepended`**

File: `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/scheduler/waiting_queue.py` (lines 179, 185)

`self._prepended.pop(0)` and `self._requests.pop(0)` are O(n) operations on Python lists. For `_requests` this is unavoidable since it needs sorting. For `_prepended`, a `deque` would give O(1) popleft. However, given typical queue sizes (<1000), the practical impact is negligible. Worth noting for future optimization if queue sizes grow.

**NTH-2: Test for `None` request object in `_compute_score`**

The implementation correctly handles `item.request is None` in `_compute_score` (line 147-148) and `_get_arrival_time` (line 140-141), but no test exercises this path. Consider adding a test where a `RequestQueueItem` has `request=None` to verify the fallback behavior produces `prompt_len=0` and uses the side-dict arrival time.

**NTH-3: No validation that `sjf_config` is only set when `waiting_queue_policy=SJF`**

File: `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (lines 1925-1961)

A user could set `waiting_queue_policy: fcfs` with `sjf_config: {length_median: 1024}` and the `sjf_config` would be silently ignored. A Pydantic `model_validator` could warn or error in this case:

```python
@model_validator(mode='after')
def _validate_sjf_config(self) -> 'SchedulerConfig':
    if self.sjf_config is not None and self.waiting_queue_policy != WaitingQueuePolicy.SJF:
        from tensorrt_llm.logger import logger
        logger.warning("sjf_config is set but waiting_queue_policy is not 'sjf'; sjf_config will be ignored.")
    return self
```

**NTH-4: Consider a test for `prepend_requests` with the actual caller pattern**

The test `test_prepend_requests_order` (line 287-300) tests with `[1, 2]` directly. Consider also testing with the actual caller pattern `queue.prepend_requests(reversed([item1, item2, item3]))` to directly validate the contract with `request_utils.py:155`.

## Plan Alignment

All tasks from the implementation plan are addressed:

| Task | Status | Notes |
|------|--------|-------|
| 1.1: Add SJF enum | Done | Correct |
| 1.2: Add SjfConfig | Done | Correct, uses StrictBaseModel |
| 1.3: Add sjf_config to SchedulerConfig | Done | Correct |
| 2.1: SJFWaitingQueue class | Done | All ABC methods implemented |
| 2.2: Update factory | Done | Correct |
| 2.3: Update __init__.py exports | Done | Correct |
| 3.1: Update _util.py | Done | Correct |
| 3.2: Pass sjf_config to PyExecutor | Done | Missing type annotation (SF-1) |
| 3.3: Check api_stability | N/A | SchedulerConfig fields not tracked individually |
| 4.1: Unit tests | Done | All specified test cases covered |

One minor deviation from the plan: Task 2.1 specified `prepend_requests` should "preserve input order", but the implementation uses extendleft-style reversal to match `FCFSWaitingQueue` semantics. This is the **correct** behavior -- the design doc explicitly states this at section 4.4, and the caller (`request_utils.py:155`) passes `reversed(pending_requests)`, so the double reversal produces the original order at the front. The plan wording was slightly ambiguous but the implementation is correct.

## Verdict

**Approve with required fixes.** The two must-fix items (copyright headers) are mechanical. The should-fix items (type annotation, sorted flag reset, prototype status) improve robustness and should be addressed before merge. The nice-to-have items are optional improvements. The core implementation is sound, well-structured, and ready for the Phase 4 OCI verification.
