# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU memory admission for multimodal requests in the OpenAI server."""

import asyncio
import base64
import json
import os

import numpy as np
import pytest
import torch
from fastapi import FastAPI

from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalServerConfig
from tensorrt_llm.inputs.utils import (
    MultimodalDataTooLargeError,
    MultimodalDataTracker,
    _cpu_storage_bytes,
    _release_shared_cpu_tensors,
)
from tensorrt_llm.serve.openai_server import OpenAIServer, _MultimodalRequestBodyLimitMiddleware

pytestmark = pytest.mark.cpu_only


def _make_tracker(max_bytes: int, initial_cpu_bytes: int = 0) -> MultimodalDataTracker:
    return MultimodalDataTracker(
        model_type="test_model",
        multimodal_server_config=MultimodalServerConfig(max_cpu_bytes_per_request=max_bytes),
        initial_cpu_bytes=initial_cpu_bytes,
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
async def test_raw_body_bytes_reduce_decoded_capacity():
    tracker = _make_tracker(max_bytes=16, initial_cpu_bytes=8)
    tracker._data["image"].append(_loaded(np.zeros((9,), dtype=np.uint8)))

    with pytest.raises(MultimodalDataTooLargeError, match="17 CPU bytes"):
        await tracker.retrieve_all_async()


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


def test_shared_cpu_tensor_handles_are_counted_once_and_released():
    storage = torch.ones(12)
    params = MultimodalParams(multimodal_data={"image": [storage[:6], storage[6:]]})
    params.to_handle("multimodal_data")
    del storage
    handles = params.multimodal_data["image"]
    shm_path = "/dev/shm" + base64.b64decode(handles[0]["storage_handle"]).decode()

    assert _cpu_storage_bytes(handles) == 48
    # Dropping unconsumed handles would leave the segment until process exit.
    assert os.path.exists(shm_path)
    _release_shared_cpu_tensors(params.multimodal_data)
    assert not os.path.exists(shm_path)


async def _run_body_limit(
    chunks: list[bytes],
    *,
    max_bytes: int,
    path: str = "/v1/chat/completions",
    content_length: int | None = None,
):
    received_body = bytearray()
    observed_state = {}

    async def app(scope, receive, send):
        while True:
            message = await receive()
            received_body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        observed_state.update(scope.get("state", {}))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index + 1 < len(chunks),
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
    }
    middleware = _MultimodalRequestBodyLimitMiddleware(app, max_bytes)
    await middleware(scope, receive, send)
    return sent[0]["status"], bytes(received_body), observed_state


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
async def test_content_length_rejects_before_reading_body(path):
    status, body, _ = await _run_body_limit([b"not read"], max_bytes=8, path=path, content_length=9)

    assert status == 413
    assert body == b""


@pytest.mark.asyncio
async def test_chunked_body_is_rejected_at_limit():
    status, body, _ = await _run_body_limit([b"12345", b"678901"], max_bytes=10)

    assert status == 413
    assert body == b"12345"


@pytest.mark.asyncio
async def test_chunked_body_stays_413_through_fastapi():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def route(payload: dict):
        return payload

    messages = [
        {"type": "http.request", "body": b'{"value":', "more_body": True},
        {"type": "http.request", "body": b'"too long"}', "more_body": False},
    ]

    async def receive():
        return messages.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
        "state": {},
    }
    middleware = _MultimodalRequestBodyLimitMiddleware(app, max_bytes=10)
    await middleware(scope, receive, send)

    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["type"] == "RequestTooLargeError"


@pytest.mark.asyncio
async def test_accepted_body_bytes_are_recorded_for_decoded_limit():
    status, body, state = await _run_body_limit([b"12345", b"67890"], max_bytes=10)

    assert status == 204
    assert body == b"1234567890"
    assert state["multimodal_request_body_bytes"] == 10


@pytest.mark.asyncio
async def test_non_multimodal_route_is_unchanged():
    status, body, _ = await _run_body_limit([b"123456789"], max_bytes=8, path="/v1/completions")

    assert status == 204
    assert body == b"123456789"


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
