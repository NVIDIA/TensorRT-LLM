# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted delivery faults at the production watch-queue boundary."""

import asyncio

import pytest
from test_common.disagg_faults import Fault, FaultScenario, InjectedFault

from tensorrt_llm.serve.cluster_storage import (
    StorageItem,
    WatchEvent,
    WatchEventQueue,
    WatchEventType,
)

pytestmark = pytest.mark.cpu_only


def _event(worker):
    return WatchEvent(StorageItem(key=f"workers/{worker}", value=worker), WatchEventType.SET)


def test_dropped_notification_retry_delivers_once():
    async def exercise():
        queue = WatchEventQueue(["workers/"])
        scenario = FaultScenario([Fault("watch.deliver", "r1", "drop")])
        event = _event("r1")

        async def deliver():
            await queue.add_events([event])

        await scenario.run("watch.deliver", "r1", deliver)
        assert queue.events.empty()
        await scenario.run("watch.deliver", "r1", deliver)
        assert await asyncio.wait_for(queue.drain(), 1) == [event]
        assert queue.events.empty()
        scenario.assert_consumed()
        scenario.assert_idle()

    asyncio.run(exercise())


def test_delayed_notification_allows_other_worker_to_progress():
    async def exercise():
        queue = WatchEventQueue(["workers/"])
        scenario = FaultScenario([Fault("watch.deliver", "r1", "delay", gate="r1-ready")])
        delayed, healthy = _event("r1"), _event("r2")
        task = asyncio.create_task(
            scenario.run("watch.deliver", "r1", lambda: queue.add_events([delayed]))
        )
        try:
            await asyncio.wait_for(scenario.wait_until_blocked("r1-ready"), 1)
            await scenario.run("watch.deliver", "r2", lambda: queue.add_events([healthy]))
            assert await asyncio.wait_for(queue.drain(), 1) == [healthy]
            assert not task.done()
            scenario.release("r1-ready")
            await asyncio.wait_for(task, 1)
            assert await asyncio.wait_for(queue.drain(), 1) == [delayed]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        scenario.assert_consumed()
        scenario.assert_idle()
        assert queue.events.empty()

    asyncio.run(exercise())


def test_cancelled_delivery_does_not_publish_a_late_event():
    async def exercise():
        queue = WatchEventQueue(["workers/"])
        scenario = FaultScenario([Fault("watch.deliver", "r1", "delay", gate="r1-ready")])
        task = asyncio.create_task(
            scenario.run("watch.deliver", "r1", lambda: queue.add_events([_event("r1")]))
        )
        try:
            await asyncio.wait_for(scenario.wait_until_blocked("r1-ready"), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            scenario.release("r1-ready")
            healthy = _event("r2")
            await scenario.run("watch.deliver", "r2", lambda: queue.add_events([healthy]))
            assert await asyncio.wait_for(queue.drain(), 1) == [healthy]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        scenario.assert_consumed()
        scenario.assert_idle()
        assert queue.events.empty()

    asyncio.run(exercise())


def test_failed_delivery_can_be_retried():
    async def exercise():
        queue = WatchEventQueue(["workers/"])
        scenario = FaultScenario([Fault("watch.deliver", "r1", "raise")])
        event = _event("r1")

        async def deliver():
            await queue.add_events([event])

        with pytest.raises(InjectedFault):
            await scenario.run("watch.deliver", "r1", deliver)
        assert queue.events.empty()
        await scenario.run("watch.deliver", "r1", deliver)
        assert await asyncio.wait_for(queue.drain(), 1) == [event]
        assert queue.events.empty()
        scenario.assert_consumed()
        scenario.assert_idle()

    asyncio.run(exercise())


def test_cancelled_drain_does_not_consume_a_future_notification():
    async def exercise():
        queue = WatchEventQueue(["workers/"])
        scenario = FaultScenario([Fault("watch.drain", "reader", "delay", gate="reader-ready")])
        started = asyncio.Event()

        async def drain():
            started.set()
            return await queue.drain()

        task = asyncio.create_task(scenario.run("watch.drain", "reader", drain))
        try:
            await asyncio.wait_for(scenario.wait_until_blocked("reader-ready"), 1)
            scenario.release("reader-ready")
            await asyncio.wait_for(started.wait(), 1)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            event = _event("r2")
            await queue.add_events([event])
            assert await asyncio.wait_for(queue.drain(), 1) == [event]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        scenario.assert_consumed()
        scenario.assert_idle()

    asyncio.run(exercise())
