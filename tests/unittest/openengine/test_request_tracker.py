# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from tensorrt_llm.serve.request_tracker import RequestTracker, track_http_response


class _Result:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


@pytest.mark.asyncio
async def test_tracker_abort_drain_and_external_admission() -> None:
    tracker = RequestTracker(object())
    result = _Result()
    tracker.admit("request", result)
    tracker.begin_external()
    assert tracker.active_count == 2

    await tracker.start_drain()
    with pytest.raises(RuntimeError, match="draining"):
        tracker.begin_external()
    assert await tracker.abort("request")
    assert result.aborted
    assert not await tracker.wait_empty(0)
    await tracker.finish_external()
    assert await tracker.wait_empty(0)


@pytest.mark.asyncio
async def test_streaming_http_admission_lives_until_body_closes() -> None:
    tracker = RequestTracker(object())
    tracker.begin_external()

    async def body():
        yield b"first"
        yield b"second"

    response = type("Response", (), {})()
    response.body_iterator = body()
    await track_http_response(response, tracker)

    assert tracker.active_count == 1
    chunks = [chunk async for chunk in response.body_iterator]
    assert chunks == [b"first", b"second"]
    assert tracker.active_count == 0


@pytest.mark.asyncio
async def test_abort_all_reaches_untracked_http_engine_results() -> None:
    result = _Result()
    llm = type("Llm", (), {})()
    llm._executor = type("Executor", (), {"_results": {1: result}})()
    tracker = RequestTracker(llm)
    tracker.begin_external()

    async def finish_http() -> None:
        while not result.aborted:
            await asyncio.sleep(0)
        await tracker.finish_external()

    finish_task = asyncio.create_task(finish_http())
    assert await tracker.abort_all() == 1
    assert await tracker.wait_empty()
    await finish_task
    assert result.aborted
