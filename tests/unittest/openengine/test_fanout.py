# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from tensorrt_llm.serve.kv_event_fanout import KvEventFanout, KvEventSubscriberOverflow
from tensorrt_llm.serve.stats_fanout import StatsFanout


@pytest.mark.asyncio
async def test_kv_fanout_closes_subscriber_instead_of_hiding_queue_overflow() -> None:
    fanout = KvEventFanout(object(), buffer_size=1)
    subscription = fanout.subscribe({0})
    try:
        first_item = asyncio.create_task(subscription.__anext__())
        await asyncio.sleep(0)

        first = {"data": {"type": "stored"}, "event": 1, "attention_dp_rank": 0}
        second = {"data": {"type": "stored"}, "event": 2, "attention_dp_rank": 0}
        third = {"data": {"type": "stored"}, "event": 3, "attention_dp_rank": 0}
        fanout._publish(first)
        assert await first_item == (1, first)
        fanout._publish(second)
        fanout._publish(third)
        with pytest.raises(KvEventSubscriberOverflow, match="overflowed"):
            await subscription.__anext__()

        assert fanout.subscriber_overflow_count == 1
        assert not fanout._subscribers
        assert fanout.drain_http_buffer() == [third]
        assert fanout._sequence_numbers == {0: 3}
    finally:
        await fanout.stop()


def test_kv_fanout_filters_non_global_attention_without_sequence_gaps() -> None:
    fanout = KvEventFanout(object())
    fanout._publish({"window_size": 128, "data": {"type": "created"}})
    fanout._publish({"window_size": 4096, "data": {"type": "created"}})
    fanout._publish({"window_size": 128, "data": {"type": "stored"}})
    global_event = {"window_size": 4096, "data": {"type": "stored"}}
    fanout._publish(global_event)

    assert fanout._sequence_numbers == {0: 1}
    assert fanout.drain_http_buffer() == [global_event]


def test_kv_fanout_detects_raw_event_id_gap_before_attention_filtering() -> None:
    fanout = KvEventFanout(object())
    queue = asyncio.Queue()
    fanout._subscribers[queue] = frozenset({0})
    fanout._publish({"event_id": 0, "window_size": 128, "data": {"type": "created"}})
    fanout._publish({"event_id": 1, "window_size": 4096, "data": {"type": "created"}})
    fanout._publish({"event_id": 2, "window_size": 128, "data": {"type": "stored"}})
    global_event = {"event_id": 3, "window_size": 4096, "data": {"type": "stored"}}
    fanout._publish(global_event)
    fanout._publish({"event_id": 5, "window_size": 128, "data": {"type": "stored"}})
    next_global = {"event_id": 6, "window_size": 4096, "data": {"type": "removed"}}
    fanout._publish(next_global)

    reset = {"attention_dp_rank": 0, "data": {"type": "all_cleared"}}
    assert fanout._last_engine_event_ids == {0: 6}
    assert fanout._sequence_numbers == {0: 3}
    assert [queue.get_nowait() for _ in range(3)] == [
        (1, global_event),
        (2, reset),
        (3, next_global),
    ]
    assert fanout.drain_http_buffer() == [global_event, next_global]


def test_synthetic_gap_reset_does_not_change_http_raw_event_contract() -> None:
    fanout = KvEventFanout(object())
    queue = asyncio.Queue()
    fanout._subscribers[queue] = frozenset()
    fanout._publish({"event_id": 0, "data": {"type": "created"}})
    stored = {"event_id": 2, "data": {"type": "stored"}}
    fanout._publish(stored)

    assert queue.get_nowait()[1]["data"]["type"] == "all_cleared"
    assert queue.get_nowait() == (2, stored)
    assert fanout.drain_http_buffer() == [stored]


@pytest.mark.asyncio
async def test_stats_fanout_is_single_consumer_for_http_metrics_and_load() -> None:
    class _Llm:
        calls = 0

        async def get_stats_async(self, timeout: float):
            assert timeout == 0.5
            self.calls += 1
            yield {"attentionDpRank": 0, "numActiveRequests": 1}
            yield {"attentionDpRank": 1, "numActiveRequests": 2}

    llm = _Llm()
    consumed = []
    fanout = StatsFanout(llm, buffer_size=4)
    fanout.start(consumed.append)
    fanout.wake()
    for _ in range(20):
        if len(consumed) == 2:
            break
        await asyncio.sleep(0)
    await fanout.stop()

    assert llm.calls == 1
    assert consumed == fanout.drain_http_buffer()
    assert set(fanout.latest_by_rank()) == {0, 1}
