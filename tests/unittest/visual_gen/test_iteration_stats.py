# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the visual-gen iteration-stats producer.

Covers the visual-gen ``/metrics`` snapshot shape and the lifecycle-event
transitions exercised by ``DiffusionRemoteClient``: enqueue → request started
→ request completed.  See TRTLLM-12751.
"""

from datetime import datetime

import pytest

from tensorrt_llm.visual_gen.visual_gen import _IterationStatsTracker

# Required snapshot fields per the ticket / docstring.
_REQUIRED_FIELDS = {
    "iter",
    "timestamp",
    "numQueuedRequests",
    "numActiveRequests",
    "currentStepsProcessed",
    "currentRequestId",
    "currentRequestStepIdx",
}


def _assert_shape(snapshot: dict) -> None:
    assert set(snapshot.keys()) == _REQUIRED_FIELDS, (
        f"snapshot has unexpected keys: {sorted(snapshot.keys())}"
    )
    assert isinstance(snapshot["iter"], int)
    # timestamp must be ISO 8601 parseable.
    datetime.fromisoformat(snapshot["timestamp"])
    assert isinstance(snapshot["numQueuedRequests"], int)
    assert isinstance(snapshot["numActiveRequests"], int)
    assert isinstance(snapshot["currentStepsProcessed"], int)
    assert snapshot["currentRequestId"] is None or isinstance(snapshot["currentRequestId"], int)
    assert snapshot["currentRequestStepIdx"] is None or isinstance(
        snapshot["currentRequestStepIdx"], int
    )


def test_initial_state_is_idle():
    tracker = _IterationStatsTracker()
    snap = tracker.current_snapshot(num_queued_requests=0)
    _assert_shape(snap)
    assert snap["numQueuedRequests"] == 0
    assert snap["numActiveRequests"] == 0
    assert snap["currentRequestId"] is None
    assert snap["currentRequestStepIdx"] is None
    assert snap["currentStepsProcessed"] == 0


def test_enqueue_records_snapshot_with_queue_depth():
    tracker = _IterationStatsTracker()
    tracker.record_enqueue(num_queued_requests=3)
    stats = tracker.drain()
    assert len(stats) == 1
    snap = stats[0]
    _assert_shape(snap)
    assert snap["numQueuedRequests"] == 3
    assert snap["numActiveRequests"] == 0
    assert snap["currentRequestId"] is None


def test_request_started_then_completed_transitions():
    tracker = _IterationStatsTracker()

    # Caller enqueued 1 then dispatcher pulled it: queue is 0, active is 1.
    tracker.record_enqueue(num_queued_requests=1)
    tracker.record_request_started(request_id=42, num_queued_requests=0)
    tracker.record_request_completed(request_id=42, num_queued_requests=0)
    stats = tracker.drain()

    assert len(stats) == 3
    enqueue_snap, started_snap, done_snap = stats
    for s in stats:
        _assert_shape(s)

    # Monotonic iter counter.
    assert [s["iter"] for s in stats] == [1, 2, 3]

    # Enqueue: nothing active yet.
    assert enqueue_snap["numQueuedRequests"] == 1
    assert enqueue_snap["numActiveRequests"] == 0
    assert enqueue_snap["currentRequestId"] is None

    # Dispatched: one in-flight, queue drained to 0.
    assert started_snap["numQueuedRequests"] == 0
    assert started_snap["numActiveRequests"] == 1
    assert started_snap["currentRequestId"] == 42

    # Completed: zero active, current_request_id cleared.
    assert done_snap["numActiveRequests"] == 0
    assert done_snap["currentRequestId"] is None


def test_drain_clears_buffer():
    tracker = _IterationStatsTracker()
    tracker.record_enqueue(0)
    tracker.record_enqueue(0)
    first = tracker.drain()
    second = tracker.drain()
    assert len(first) == 2
    assert second == []


def test_step_event_updates_step_idx_and_steps_processed():
    tracker = _IterationStatsTracker()
    tracker.record_request_started(request_id=7, num_queued_requests=0)
    tracker.record_step(request_id=7, step_idx=0, num_queued_requests=0)
    tracker.record_step(request_id=7, step_idx=1, num_queued_requests=0)
    stats = tracker.drain()
    assert len(stats) == 3
    _, step0, step1 = stats
    assert step0["currentRequestStepIdx"] == 0
    assert step0["currentStepsProcessed"] == 1
    assert step1["currentRequestStepIdx"] == 1
    assert step1["currentStepsProcessed"] == 2


def test_step_event_for_other_request_is_ignored():
    """Stale step events for a non-current request must not corrupt state."""
    tracker = _IterationStatsTracker()
    tracker.record_request_started(request_id=1, num_queued_requests=0)
    # Step from a different (e.g. previously-completed) request id.
    tracker.record_step(request_id=999, step_idx=5, num_queued_requests=0)
    stats = tracker.drain()
    assert stats[-1]["currentRequestId"] == 1
    assert stats[-1]["currentRequestStepIdx"] is None
    assert stats[-1]["currentStepsProcessed"] == 0


def test_buffer_max_len_caps_growth():
    tracker = _IterationStatsTracker(maxlen=3)
    for _ in range(5):
        tracker.record_enqueue(num_queued_requests=0)
    stats = tracker.drain()
    assert len(stats) == 3
    # iter counter still monotonically advanced for all 5 records, but only
    # the last 3 survive.
    assert [s["iter"] for s in stats] == [3, 4, 5]


def test_completed_request_decrements_active_count_only_once():
    tracker = _IterationStatsTracker()
    tracker.record_request_started(1, num_queued_requests=0)
    tracker.record_request_completed(1, num_queued_requests=0)
    # A duplicate completion must not push active below zero.
    tracker.record_request_completed(1, num_queued_requests=0)
    stats = tracker.drain()
    assert all(s["numActiveRequests"] >= 0 for s in stats)
    assert stats[-1]["numActiveRequests"] == 0


def test_duplicate_completion_does_not_disturb_other_inflight_requests():
    """Duplicate completion must not disturb other in-flight requests.

    A second completion for an already-completed id must NOT decrement the
    active count for an unrelated in-flight request.
    """
    tracker = _IterationStatsTracker()
    tracker.record_request_started(1, num_queued_requests=0)
    tracker.record_request_started(2, num_queued_requests=0)
    tracker.record_request_completed(1, num_queued_requests=0)
    # After completing #1, only #2 is in flight.
    snap_after_first_complete = tracker.drain()[-1]
    assert snap_after_first_complete["numActiveRequests"] == 1
    assert snap_after_first_complete["currentRequestId"] == 2
    # Duplicate completion for #1 must be a no-op for the count; #2 still in flight.
    tracker.record_request_completed(1, num_queued_requests=0)
    snap_after_duplicate = tracker.drain()[-1]
    assert snap_after_duplicate["numActiveRequests"] == 1
    assert snap_after_duplicate["currentRequestId"] == 2


def test_out_of_order_completion_falls_back_to_remaining_active_id():
    """Out-of-order completion falls back to a remaining active id.

    When the *current* request completes while others remain in flight,
    ``currentRequestId`` must fall back to a still-active id rather than
    parking at ``None``.
    """
    tracker = _IterationStatsTracker()
    tracker.record_request_started(10, num_queued_requests=0)
    tracker.record_request_started(20, num_queued_requests=0)
    tracker.record_request_started(30, num_queued_requests=0)
    # Complete the current (most-recent) one out of order.
    tracker.record_request_completed(30, num_queued_requests=0)
    snap = tracker.drain()[-1]
    assert snap["numActiveRequests"] == 2
    # Fallback to the most-recently-dispatched still-active id.
    assert snap["currentRequestId"] == 20
    # Now complete the middle one out of order; oldest (10) should remain.
    tracker.record_request_completed(20, num_queued_requests=0)
    snap = tracker.drain()[-1]
    assert snap["numActiveRequests"] == 1
    assert snap["currentRequestId"] == 10
    # Final completion drains the active set.
    tracker.record_request_completed(10, num_queued_requests=0)
    snap = tracker.drain()[-1]
    assert snap["numActiveRequests"] == 0
    assert snap["currentRequestId"] is None


def test_completion_for_unknown_request_is_a_noop():
    """Completion events for ids that never started must not change state."""
    tracker = _IterationStatsTracker()
    tracker.record_request_started(7, num_queued_requests=0)
    # Stray completion event for an id that was never started.
    tracker.record_request_completed(999, num_queued_requests=0)
    snap = tracker.drain()[-1]
    assert snap["numActiveRequests"] == 1
    assert snap["currentRequestId"] == 7


def test_current_snapshot_does_not_buffer():
    tracker = _IterationStatsTracker()
    snap = tracker.current_snapshot(num_queued_requests=2)
    _assert_shape(snap)
    assert snap["numQueuedRequests"] == 2
    # current_snapshot should not have appended to the drain buffer.
    assert tracker.drain() == []


@pytest.mark.asyncio
async def test_visual_gen_get_stats_async_yields_snapshots():
    """``VisualGen.get_stats_async`` should yield buffered tracker snapshots."""
    from tensorrt_llm.visual_gen.visual_gen import VisualGen

    # Build a VisualGen instance bypassing __init__ to avoid spawning workers.
    vg = VisualGen.__new__(VisualGen)
    fake_executor = type(
        "FakeExecutor",
        (),
        {
            "get_iteration_stats": lambda self: [{"iter": 1, "numActiveRequests": 0}],
            # ``VisualGen.__del__`` calls ``self.executor.shutdown()`` during
            # garbage collection; provide a no-op so GC doesn't raise an
            # unraisable AttributeError when the test object is collected.
            "shutdown": lambda self: None,
        },
    )()
    vg.executor = fake_executor

    out = []
    async for stat in vg.get_stats_async():
        out.append(stat)
    assert out == [{"iter": 1, "numActiveRequests": 0}]
