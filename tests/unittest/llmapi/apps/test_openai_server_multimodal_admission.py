# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU memory admission for multimodal requests in the OpenAI server."""

import asyncio

import numpy as np
import pytest

from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.inputs.utils import MultimodalDataTooLargeError, MultimodalDataTracker
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


def _make_tracker(max_bytes: int) -> MultimodalDataTracker:
    return MultimodalDataTracker(
        model_type="test_model",
        multimodal_server_config=MultimodalServerConfig(max_cpu_bytes_per_request=max_bytes),
    )


async def _loaded(value):
    return value


@pytest.mark.asyncio
async def test_shared_storage_is_counted_once():
    storage = np.zeros((16,), dtype=np.uint8)
    tracker = _make_tracker(max_bytes=16)
    tracker._data["image"].extend([_loaded(storage[:8]), _loaded(storage[8:])])

    data, _ = await tracker.retrieve_all_async()

    assert len(data["image"]) == 2


@pytest.mark.asyncio
async def test_limit_cancels_remaining_items():
    cancelled = asyncio.Event()

    async def _pending():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    tracker = _make_tracker(max_bytes=16)
    tracker._data["image"].extend([_loaded(np.zeros((17,), dtype=np.uint8)), _pending()])

    with pytest.raises(MultimodalDataTooLargeError, match="17 CPU bytes"):
        await tracker.retrieve_all_async()

    assert cancelled.is_set()


def test_multimodal_server_cpu_limits_are_consistent():
    config = MultimodalServerConfig(max_cpu_bytes=1024)
    assert config.max_cpu_bytes_per_request == 1024

    with pytest.raises(ValueError, match="cannot exceed"):
        MultimodalServerConfig(
            max_cpu_bytes=1024,
            max_cpu_bytes_per_request=2048,
        )


@pytest.mark.asyncio
async def test_total_limit_serializes_multimodal_preprocessing():
    server = object.__new__(OpenAIServer)
    server.multimodal_server_config = MultimodalServerConfig(max_cpu_bytes=1)
    server._mm_cpu_request_slots = asyncio.BoundedSemaphore(1)

    assert await server._acquire_mm_cpu_slot()
    next_request = asyncio.create_task(server._acquire_mm_cpu_slot())
    await asyncio.sleep(0)
    assert not next_request.done()

    server._mm_cpu_request_slots.release()
    assert await next_request
    server._mm_cpu_request_slots.release()
