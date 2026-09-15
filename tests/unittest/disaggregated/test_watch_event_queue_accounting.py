# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from tensorrt_llm.serve.cluster_storage import (
    StorageItem,
    WatchEvent,
    WatchEventQueue,
    WatchEventType,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
async def test_drain_acknowledges_every_dequeued_event():
    queue = WatchEventQueue(["worker/"])
    events = [
        WatchEvent(StorageItem(key="worker/1", value="one"), WatchEventType.SET),
        WatchEvent(StorageItem(key="worker/2", value="two"), WatchEventType.SET),
        WatchEvent(StorageItem(key="worker/3", value="three"), WatchEventType.DELETE),
    ]
    await queue.add_events(events)

    assert await queue.drain() == events
    await asyncio.wait_for(queue.events.join(), timeout=0.1)
