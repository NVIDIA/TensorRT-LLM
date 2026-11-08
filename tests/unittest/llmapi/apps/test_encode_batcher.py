# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the encode dynamic batcher (tensorrt_llm/serve/encode_batcher.py).

These tests exercise the pure-async coalescing logic with an injected, synchronous
`encode_fn`. They do not require a GPU or any TensorRT-LLM runtime.
"""

import pytest

from tensorrt_llm.serve.encode_batcher import EncodeBatcher

pytestmark = pytest.mark.asyncio


async def test_single_submission_returns_its_result():
    """A single submission is encoded and its own result is returned."""

    def encode_fn(batch):
        # batch is a list of token-id lists; return one result per input item.
        return [sum(token_ids) for token_ids in batch]

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=8,
        max_queue_delay=0.01,
        max_queue_size=100,
    )
    await batcher.start()
    try:
        result = await batcher.submit([1, 2, 3])
        assert result == 6
    finally:
        await batcher.shutdown()


async def test_concurrent_submissions_coalesce_into_one_call():
    """Concurrent submissions are coalesced into a single encode_fn call.

    Submissions arriving within the hold window join one batch, and each caller
    still gets its own result back.
    """
    import asyncio

    calls = []

    def encode_fn(batch):
        calls.append([list(t) for t in batch])
        return [sum(token_ids) for token_ids in batch]

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=8,
        max_queue_delay=0.05,
        max_queue_size=100,
    )
    await batcher.start()
    try:
        results = await asyncio.gather(
            batcher.submit([1]),
            batcher.submit([2, 2]),
            batcher.submit([3, 3, 3]),
        )
        # Each caller gets its own (correctly-ordered) result.
        assert results == [1, 4, 9]
        # All three were coalesced into exactly one encode_fn call.
        assert len(calls) == 1
        assert calls[0] == [[1], [2, 2], [3, 3, 3]]
    finally:
        await batcher.shutdown()


async def test_flush_when_token_budget_would_be_exceeded():
    """A batch never exceeds max_num_tokens.

    An item that would overflow the current batch is held back to seed the next
    batch.
    """
    import asyncio

    calls = []

    def encode_fn(batch):
        calls.append([list(t) for t in batch])
        return [len(token_ids) for token_ids in batch]

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=100,  # not the limiting factor here
        max_queue_delay=0.05,
        max_queue_size=100,
        max_num_tokens=5,
    )
    await batcher.start()
    try:
        # 3 + 3 = 6 > 5 tokens, so the two requests cannot share a batch.
        results = await asyncio.gather(
            batcher.submit([1, 1, 1]),
            batcher.submit([2, 2, 2]),
        )
        assert results == [3, 3]
        assert len(calls) == 2
        # No formed batch ever exceeds the token budget.
        for batch in calls:
            assert sum(len(token_ids) for token_ids in batch) <= 5
    finally:
        await batcher.shutdown()


async def test_submit_raises_queue_full_when_queue_is_full():
    """A full queue makes submit raise QueueFullError (mapped to HTTP 429).

    It must not block or silently drop the request.
    """
    import asyncio

    from tensorrt_llm.serve.encode_batcher import QueueFullError

    def encode_fn(batch):
        return [0] * len(batch)

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=8,
        max_queue_delay=1.0,
        max_queue_size=2,
    )
    # Intentionally do NOT start the worker, so nothing drains the queue.
    pending = [
        asyncio.ensure_future(batcher.submit([1])),
        asyncio.ensure_future(batcher.submit([2])),
    ]
    await asyncio.sleep(0.01)  # let the two submissions enqueue
    try:
        with pytest.raises(QueueFullError):
            await batcher.submit([3])
    finally:
        for task in pending:
            task.cancel()


async def test_encode_failure_propagates_and_worker_survives():
    """encode_fn failures are isolated to the failing batch.

    The failing batch's callers see the exception, and the worker keeps running so
    subsequent requests still succeed.
    """
    import asyncio

    state = {"fail_next": True}

    def encode_fn(batch):
        if state["fail_next"]:
            state["fail_next"] = False
            raise RuntimeError("boom")
        return [len(token_ids) for token_ids in batch]

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=1,  # one request per batch, so failures are isolated
        max_queue_delay=0.01,
        max_queue_size=100,
    )
    await batcher.start()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.wait_for(batcher.submit([1, 2]), timeout=1.0)
        # The worker is still alive: a later request succeeds normally.
        result = await asyncio.wait_for(batcher.submit([1, 2, 3]), timeout=1.0)
        assert result == 3
    finally:
        await batcher.shutdown()


async def test_cancelled_request_does_not_kill_worker():
    """A request cancelled while in flight must not take down the worker.

    If a caller's future is cancelled (e.g. the client disconnects) after its
    batch has been formed, the success path would call set_result() on an
    already-done future, raise InvalidStateError, and kill the sole worker —
    hanging every subsequent request. The worker must instead skip the cancelled
    future and keep serving.
    """
    import asyncio
    import threading

    encode_started = threading.Event()
    release = threading.Event()

    def encode_fn(batch):
        encode_started.set()
        # Block until the test has had a chance to cancel the in-flight request.
        release.wait(timeout=5.0)
        return [sum(token_ids) for token_ids in batch]

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=1,  # one request per batch, so the cancelled one dispatches alone
        max_queue_delay=0.01,
        max_queue_size=100,
    )
    await batcher.start()
    try:
        # Submit a request and wait until its batch is being encoded.
        task = asyncio.ensure_future(batcher.submit([1, 2, 3]))
        await asyncio.get_running_loop().run_in_executor(None, encode_started.wait, 5.0)
        # Cancel it mid-encode so its future is done (cancelled) before dispatch
        # tries to resolve it, then let encode_fn finish.
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The worker must still be alive and serving.
        assert batcher.is_alive()
        result = await asyncio.wait_for(batcher.submit([4, 5, 6]), timeout=1.0)
        assert result == 15
    finally:
        await batcher.shutdown()


async def test_event_loop_stays_responsive_during_blocking_encode():
    """A blocking encode_fn must not freeze the event loop.

    encode_fn is synchronous/blocking (like a GPU forward); the worker runs it off
    the event loop so other coroutines keep making progress meanwhile.
    """
    import asyncio
    import time

    window = {}

    def encode_fn(batch):
        window["start"] = time.monotonic()
        time.sleep(0.2)  # simulate a blocking forward pass
        window["end"] = time.monotonic()
        return [0] * len(batch)

    tick_times = []

    async def ticker():
        for _ in range(20):
            await asyncio.sleep(0.01)
            tick_times.append(time.monotonic())

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=1,
        max_queue_delay=0.001,
        max_queue_size=100,
    )
    await batcher.start()
    try:
        ticker_task = asyncio.ensure_future(ticker())
        await batcher.submit([1])  # triggers the 0.2s blocking encode
        await ticker_task
        # Ticks must have landed *during* the blocking encode window — proving the
        # loop was not frozen. Inline (loop-blocking) execution yields zero such ticks.
        ticks_during = [t for t in tick_times if window["start"] < t < window["end"]]
        assert len(ticks_during) >= 3
    finally:
        await batcher.shutdown()


async def test_submit_rejects_input_longer_than_max_seq_len():
    """An input longer than max_seq_len is rejected at submission.

    submit raises InputTooLongError (mapped to HTTP 400) before the input ever
    enters the queue.
    """
    from tensorrt_llm.serve.encode_batcher import InputTooLongError

    def encode_fn(batch):
        return [0] * len(batch)

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=8,
        max_queue_delay=0.01,
        max_queue_size=100,
        max_seq_len=4,
    )
    await batcher.start()
    try:
        with pytest.raises(InputTooLongError):
            await batcher.submit([1, 2, 3, 4, 5])  # 5 tokens > max_seq_len=4
        # A within-limit request still succeeds.
        assert await batcher.submit([1, 2, 3, 4]) == 0
    finally:
        await batcher.shutdown()


async def test_submit_rejects_single_input_over_token_budget():
    """A lone input larger than max_num_tokens is rejected at submission.

    A single request that exceeds the per-batch token budget can never form a
    valid batch, so submit raises InputTooLongError (HTTP 400) up front rather
    than letting encode_fn reject the batch later. validate_input() exposes the
    same check for callers that want to pre-validate before enqueueing.
    """
    from tensorrt_llm.serve.encode_batcher import InputTooLongError

    def encode_fn(batch):
        return [0] * len(batch)

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=8,
        max_queue_delay=0.01,
        max_queue_size=100,
        max_num_tokens=4,  # smaller than the oversize input below
        max_seq_len=100,  # not the limiting factor here
    )
    await batcher.start()
    try:
        # 5 tokens > max_num_tokens=4, even alone.
        with pytest.raises(InputTooLongError):
            batcher.validate_input([1, 2, 3, 4, 5])
        with pytest.raises(InputTooLongError):
            await batcher.submit([1, 2, 3, 4, 5])
        # A within-budget request still succeeds.
        assert await batcher.submit([1, 2, 3, 4]) == 0
    finally:
        await batcher.shutdown()


async def test_full_batch_flushes_immediately_without_waiting_window():
    """A batch that reaches max_batch_size is dispatched immediately.

    It does not wait out the (much longer) hold window.
    """
    import asyncio
    import time

    calls = []

    def encode_fn(batch):
        calls.append(len(batch))
        return [0] * len(batch)

    batcher = EncodeBatcher(
        encode_fn,
        max_batch_size=2,
        max_queue_delay=1.0,  # long window; must NOT be waited out when full
        max_queue_size=100,
    )
    await batcher.start()
    try:
        t0 = time.monotonic()
        await asyncio.gather(batcher.submit([1]), batcher.submit([2]))
        elapsed = time.monotonic() - t0
        assert calls == [2]  # coalesced into a single full batch
        assert elapsed < 0.5  # fired well before the 1.0s window elapsed
    finally:
        await batcher.shutdown()
