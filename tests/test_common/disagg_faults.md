<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Scripted asynchronous boundary faults

`test_common.disagg_faults` is a test-only helper for exercising a specific
asynchronous delivery or operation boundary. It creates no background tasks and
does not change production code, transports, or global clocks.

Each `Fault` targets a boundary name, a request/worker/storage key, and a positive
call occurrence. Occurrences are counted separately for every boundary/key
pair. Unmatched calls run normally. Two faults cannot target the same call.

The initial actions are:

- `drop`: skip the operation and return `None` (appropriate for notification
  delivery, not for silently discarding a value returned by a read).
- `delay`: stop before the operation until a named gate is released.
- `raise`: raise `InjectedFault` before starting the operation.

Delay gates control ordering; they do not advance elapsed time or trigger a
production timeout. Use the component's existing clock injection point when a
test needs simulated deadlines. Real `asyncio.wait_for` deadlines in examples
are safety bounds that fail a stuck test, not the mechanism driving its order.

For example, hold one delivery while allowing another worker to progress:

```python
import asyncio

from test_common.disagg_faults import Fault, FaultScenario

scenario = FaultScenario([
    Fault("watch.deliver", "worker-a", "delay", gate="resume-a"),
])
task = asyncio.create_task(scenario.run(
    "watch.deliver", "worker-a", lambda: queue.add_events([event_a]),
))
try:
    await asyncio.wait_for(scenario.wait_until_blocked("resume-a"), 1)
    await scenario.run("watch.deliver", "worker-b", lambda: queue.add_events([event_b]))
    assert await asyncio.wait_for(queue.drain(), 1) == [event_b]
    scenario.release("resume-a")
    await asyncio.wait_for(task, 1)
    assert await asyncio.wait_for(queue.drain(), 1) == [event_a]
finally:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
scenario.assert_consumed()
scenario.assert_idle()
```

Pass a coroutine **factory**, not a created coroutine. This prevents dropped
operations from starting or leaving unawaited coroutines behind. The test owns
all tasks it creates and must cancel/join them in `finally`, including when an
assertion fails. A scenario is used within one event loop, not across threads.

`assert_consumed()` detects a fault that was never reached, such as a misspelled
target. `assert_idle()` detects scenario calls still in progress; it does not
inspect component-owned resources or prove that every application task ended.
Assert those component invariants separately. `trace` records boundary, key,
occurrence and transition in execution order, without wall-clock timestamps or
retained payloads. If a delayed call is cancelled, the fault counts as reached
and its operation never starts; its gate does not apply again on retry.

The watch-queue examples in
`tests/unittest/disaggregated/test_cluster_storage_faults.py` exercise dropped
delivery/retry, delayed delivery with unaffected-worker progress, delivery
cancellation, injected failure/retry and cancellation of a pending drain. These
test queue behavior and the helper's use at a real component boundary. They do
not establish full DisaggClusterManager reconciliation, worker lifecycle,
KV-transfer timeout handling, NIXL/UCX cancellation, or GPU correctness.

The existing `test_heartbeat_survives_stalled_refresh_within_ttl` regression in
`tests/unittest/disaggregated/test_disagg_cluster_manager_worker.py` uses a
scripted delay on the first storage refresh. The worker's real per-attempt
timeout cancels that call, and the next attempt succeeds. The test retains its
registration-deadline and retry assertions and checks the cancellation/retry
trace and scenario cleanup. Its TTL checks still use the production clock;
the gate only replaces the artificial long-running storage call.

In a TensorRT-LLM development environment matching the checkout, run:

```bash
pytest -m cpu_only \
  tests/unittest/disaggregated/test_fault_scenario.py \
  tests/unittest/disaggregated/test_cluster_storage_faults.py \
  tests/unittest/disaggregated/test_disagg_cluster_manager_worker.py::test_heartbeat_survives_stalled_refresh_within_ttl
```

These tests use the normal production imports and require the corresponding
Python/native dependencies even though they do not perform GPU inference.
