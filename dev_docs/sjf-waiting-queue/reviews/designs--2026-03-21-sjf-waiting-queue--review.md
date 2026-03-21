# Review: SJF Waiting Queue Design

## Summary

This is a well-structured design for adding SJF (Shortest Job First) waiting queue scheduling to TRT-LLM. The document clearly articulates the problem (head-of-line blocking with FCFS), proposes a clean scoring formula with aging, and correctly identifies the integration points in the existing codebase. The comparison with vLLM's approach is valuable and the design avoids several known issues in vLLM's implementation (dynamic `time.time()` in `__lt__`, hash mismatch on remove).

The design is largely ready for implementation, but there are several issues that need attention before proceeding -- most critically around the `peek_request`/`pop_request` contract with `get_from_waiting_queue()`, and a nonexistent field reference.

## Findings

### Must-Fix

**MF-1: `peek_request` and `pop_request` consistency is unspecified**

The design shows `pop_request` pulling from `_prepended` first, then from the sorted list. However, it does not specify `peek_request` behavior at all. This is critical because `get_from_waiting_queue()` in `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/request_utils.py` (lines 132-137) calls `peek_request()` first to check `child_req_ids` count, then calls `pop_request()` expecting to get the same item:

```python
while req_count < max_req_count and waiting_queue:
    req_item = waiting_queue.peek_request()
    num_children = len(req_item.child_req_ids) if req_item.child_req_ids else 0
    if (req_count + 1 + num_children) > max_req_count:
        break
    req_item = waiting_queue.pop_request()
```

If `peek_request()` and `pop_request()` do not return the same item, the `num_children` check becomes meaningless and could cause incorrect scheduling. The design must specify:
- When does sorting happen? On `peek_request`? On `pop_request`? On both?
- How do we guarantee `peek_request()` followed by `pop_request()` returns the same item without re-sorting in between?

Recommendation: Sort once per scheduling cycle lazily on the first `peek_request` call (set a dirty flag on `add_request`/`add_requests`), and have `pop_request` use the already-sorted order without re-sorting. Document this contract explicitly.

**MF-2: `item._added_time` does not exist**

The design references `item._added_time` as a fallback for `py_arrival_time`:

```python
arrival_time = getattr(item.request, 'py_arrival_time', None) or item._added_time
```

However, `RequestQueueItem` (defined in `/home/scratch.laliao_gpu/repos/trtllm-scheduler/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/executor_request_queue.py`, line 20) is a simple dataclass with fields `id`, `request`, `child_req_ids`, `is_canceled_request`, and `query`. There is no `_added_time` field. The design must either:
- (a) Add `_added_time` to `RequestQueueItem` (invasive, affects all queue types), or
- (b) Record the fallback time inside `SJFWaitingQueue.add_request()` using a side dict `self._arrival_times: dict[int, float]`, or
- (c) Confirm that `py_arrival_time` is always set and remove the fallback (risky).

Option (b) is recommended as it is self-contained within SJFWaitingQueue.

**MF-3: `input_token_ids` access path needs verification**

The design uses `item.request.input_token_ids` to get prompt length. However, `item.request` is an `ExecutorRequest` (the C++ pybind object). The attribute name on the pybind `ExecutorRequest` should be verified. In the codebase, usages like `req_item.request.input_token_ids` do appear (e.g., `request_utils.py:319`), so this is likely correct, but the design should note that this accesses the pybind property -- if the request has been through tokenization/chunking, the length may not reflect the original prompt length. For SJF purposes this is probably fine (we want to sort by actual prefill cost), but it should be stated explicitly.

### Should-Fix

**SF-1: `_prepended` list creates a subtle ordering bug with `extendleft` semantics**

The existing `FCFSWaitingQueue.prepend_requests` uses `deque.extendleft(requests)`, which reverses the order. The caller in `request_utils.py:155` already compensates by passing `reversed(pending_requests)`:

```python
waiting_queue.prepend_requests(reversed(pending_requests))
```

The design's `_prepended` list with `pop(0)` needs to match this contract. Since the caller passes `reversed(pending_requests)`, and `FCFSWaitingQueue.extendleft` reverses again (net effect: original order preserved at front), the SJF implementation must either:
- Accept the already-reversed iterable and store it as-is in `_prepended` (correct if using list `extend` + `pop(0)`), or
- Use `extendleft`-like semantics (reverse again).

The design should explicitly document which approach is taken and confirm it matches the caller's expectation. Getting this wrong silently corrupts priority order for prepended requests.

**SF-2: `__iter__` semantics not specified**

The `WaitingQueue` ABC requires `__iter__`. For SJF, should iteration return items in score order or insertion order? This matters for any code that iterates the queue for debugging/monitoring. The design should specify this.

**SF-3: `__bool__` and `__len__` must account for both `_requests` and `_prepended`**

The design shows two internal lists (`_requests` and `_prepended`) but does not specify `__bool__` or `__len__` implementations. Both must combine counts from both lists. This is straightforward but should be stated to avoid implementation bugs:

```python
def __len__(self):
    return len(self._requests) + len(self._prepended)

def __bool__(self):
    return bool(self._requests) or bool(self._prepended)
```

**SF-4: Factory function signature change not fully specified**

The design says `create_waiting_queue()` needs to accept `sjf_config`, but the current signature is:

```python
def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    priority_fn: Optional[Callable[[RequestQueueItem], float]] = None,
) -> WaitingQueue:
```

The design should specify: does `sjf_config` replace the `priority_fn` parameter? Or is it added alongside? The `priority_fn` was "reserved for future use" -- the design should clarify whether SJF subsumes this or coexists. Recommendation: add `sjf_config: Optional[SjfConfig] = None` as a new parameter and keep `priority_fn` for now to avoid breaking the signature.

**SF-5: No validation that `waiting_queue_policy=sjf` requires `sjf_config` or vice versa**

What happens if a user sets `waiting_queue_policy: sjf` but omits `sjf_config`? The design shows `SjfConfig()` as default in the constructor, which is fine. But what about the reverse -- `sjf_config` is set but `waiting_queue_policy` is `fcfs`? Should there be a Pydantic `model_validator` on `SchedulerConfig` that warns or errors? This is a usability concern worth addressing.

### Nice-to-Have

**NH-1: Consider adding a `score` field or logging for observability**

For debugging and tuning, it would be helpful to log or expose the computed scores. Consider adding an optional debug log (at TRACE or DEBUG level) that emits the top-N scored requests each scheduling cycle. This helps users tune `length_median`/`time_median` for their workloads.

**NH-2: The quantitative analysis could include worst-case sort frequency**

The analysis correctly estimates sort cost for 500 elements (~0.05ms). However, it should also note the frequency: `get_from_waiting_queue()` is called once per executor loop iteration, which is roughly once per forward pass. Under high concurrency with fast decode steps (~5-10ms), sorting could happen 100-200 times/sec. At 0.05ms each, that is 5-10ms/sec of CPU time -- still negligible but worth stating for completeness.

**NH-3: Default `length_median=32768` may be too high for most workloads**

A median of 32K tokens means that for typical workloads (ISL 1K-4K), the `length_score` will cluster near 1.0 for all requests, reducing SJF's discriminating power. Consider either:
- Recommending users tune this to their actual workload median, or
- Adding an auto-tuning mechanism that estimates the running median from observed requests (future enhancement).

**NH-4: The design could mention the `_to_pybind` implication**

`SchedulerConfig._to_pybind()` currently only passes `capacity_scheduler_policy`, `context_chunking_policy`, and `dynamic_batch_config` to the C++ `_SchedulerConfig`. The `sjf_config` field is Python-only and does not need pybind conversion, but this should be stated explicitly to avoid confusion during implementation -- someone might try to add it to `_to_pybind()`.

**NH-5: Test plan could be more specific**

Section 7 mentions adding tests but does not enumerate test cases. Consider listing:
- Basic SJF ordering (shorter prompt popped first)
- Aging (long-waiting request eventually overtakes short prompt)
- `prepend_requests` ordering preservation
- `remove_by_ids` correctness
- `peek_request`/`pop_request` consistency
- Edge cases: empty queue pop, single element, all same length
- `SjfConfig` validation (zero median rejection)

## Verdict

**PASS_WITH_CONDITIONS**

The design is sound in its core approach and well-motivated. The three must-fix items (MF-1: peek/pop consistency, MF-2: nonexistent `_added_time` field, MF-3: attribute verification) must be addressed before moving to implementation planning. The should-fix items (particularly SF-1 on prepend ordering and SF-3 on len/bool) should also be resolved to prevent subtle bugs during implementation.

Once the must-fix items are addressed, this design is ready for implementation.
