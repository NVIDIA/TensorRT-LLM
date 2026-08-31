# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline tests for the Anthropic Message Batches store.

No engine and no HTTP: the store is driven with a stub runner so the lifecycle,
counts and result rendering can be pinned without a GPU.
"""

import asyncio
import json

import pytest

from tensorrt_llm.serve.anthropic_batches import (
    AnthropicBatchStore,
    BatchStoreFullError,
    results_to_jsonl,
)
from tensorrt_llm.serve.anthropic_protocol import (
    AnthropicBatchRequestItem,
    AnthropicCreateBatchRequest,
)

# These tests are CPU-only (no GPU, engine or sockets) and run in the
# CPU-Generic CI stage, which selects with `-m cpu_only`.
pytestmark = pytest.mark.cpu_only

MODEL = "test-model"


def _item(custom_id: str, text: str = "hi") -> AnthropicBatchRequestItem:
    return AnthropicBatchRequestItem(
        custom_id=custom_id,
        params={
            "model": MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": text}],
        },
    )


def _ok_runner(delay: float = 0.0):
    async def runner(request):
        if delay:
            await asyncio.sleep(delay)
        return "succeeded", {"type": "message", "role": "assistant", "content": []}

    return runner


async def _drain(store, batch_id, timeout=5.0):
    """Wait for a batch to reach 'ended'."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if store.status_of(batch_id) == "ended":
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"batch {batch_id} did not end within {timeout}s")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_batch_runs_to_completion_and_counts_match():
    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        batch = store.create([_item("a"), _item("b"), _item("c")])
        # A fresh batch is in_progress with everything still counted as
        # processing - clients poll on exactly this.
        assert batch.processing_status == "in_progress"
        assert batch.request_counts.processing == 3
        assert batch.results_url is None
        assert batch.ended_at is None

        await _drain(store, batch.id)
        done = store.get(batch.id)
        assert done.processing_status == "ended"
        assert done.request_counts.succeeded == 3
        assert done.request_counts.processing == 0
        assert done.ended_at is not None
        assert done.results_url == f"/v1/messages/batches/{batch.id}/results"

    asyncio.run(scenario())


def test_counts_always_sum_to_the_batch_size():
    """The documented invariant, and the one clients divide by."""

    async def scenario():
        calls = {"n": 0}

        async def flaky(request):
            calls["n"] += 1
            if calls["n"] % 2:
                return "succeeded", {"type": "message"}
            raise RuntimeError("boom")

        store = AnthropicBatchStore(runner=flaky)
        batch = store.create([_item(f"id-{i}") for i in range(5)])
        await _drain(store, batch.id)

        counts = store.get(batch.id).request_counts
        total = (
            counts.succeeded + counts.errored + counts.canceled + counts.expired + counts.processing
        )
        assert total == 5
        assert counts.errored == 2

    asyncio.run(scenario())


def test_a_failing_request_does_not_kill_the_batch():
    """One bad request must not take its 99 siblings down with it."""

    async def scenario():
        async def explode_on_b(request):
            if request.messages[0].content == "b":
                raise RuntimeError("boom")
            return "succeeded", {"type": "message"}

        store = AnthropicBatchStore(runner=explode_on_b)
        batch = store.create([_item("x", "a"), _item("y", "b"), _item("z", "c")])
        await _drain(store, batch.id)

        counts = store.get(batch.id).request_counts
        assert counts.succeeded == 2 and counts.errored == 1

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Contract fidelity
# ---------------------------------------------------------------------------


def test_processing_status_uses_the_documented_enum():
    """in_progress/canceling/ended, not processing/completed/failed.

    A client deserializing MessageBatch rejects anything else, so this is
    pinned rather than left to a future refactor.
    """

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        batch = store.create([_item("a")])
        assert batch.processing_status in {"in_progress", "canceling", "ended"}
        await _drain(store, batch.id)
        assert store.get(batch.id).processing_status == "ended"

    asyncio.run(scenario())


def test_results_are_jsonl_keyed_by_custom_id():
    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        batch = store.create([_item("first"), _item("second")])
        await _drain(store, batch.id)

        rendered = results_to_jsonl(store.results(batch.id))
        lines = [json.loads(line) for line in rendered.splitlines()]
        assert {line["custom_id"] for line in lines} == {"first", "second"}
        assert all(line["result"]["type"] == "succeeded" for line in lines)
        # Every line must be independently parseable - that is what makes it
        # streamable rather than a JSON array.
        assert rendered.endswith("\n")

    asyncio.run(scenario())


def test_errored_result_carries_an_error_envelope():
    async def scenario():
        async def always_fail(request):
            raise RuntimeError("boom")

        store = AnthropicBatchStore(runner=always_fail)
        batch = store.create([_item("a")])
        await _drain(store, batch.id)

        line = store.results(batch.id)[0]
        assert line["result"]["type"] == "errored"
        # MessageBatchErroredResult.error is an ErrorResponse wrapping an
        # ErrorObject - two levels, not one. A bare ErrorObject here fails SDK
        # deserialization, and the nesting is easy to flatten by accident.
        envelope = line["result"]["error"]
        assert envelope["type"] == "error", envelope
        assert "request_id" in envelope
        assert envelope["error"]["type"] == "api_error", envelope
        assert envelope["error"]["message"]

    asyncio.run(scenario())


def test_duplicate_custom_id_is_rejected_at_admission():
    """Results are matched by custom_id, so duplicates are unanswerable."""

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        with pytest.raises(ValueError, match="unique"):
            store.create([_item("same"), _item("same")])

    asyncio.run(scenario())


@pytest.mark.parametrize("custom_id", ["", "a" * 65, "has space", "has/slash"])
def test_invalid_custom_id_is_rejected_by_the_schema(custom_id):
    with pytest.raises(Exception):
        AnthropicBatchRequestItem(
            custom_id=custom_id,
            params={
                "model": MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )


def test_empty_batch_is_rejected():
    with pytest.raises(Exception):
        AnthropicCreateBatchRequest(requests=[])


# ---------------------------------------------------------------------------
# Cancellation, results availability, deletion
# ---------------------------------------------------------------------------


def test_cancel_marks_canceling_then_ends_with_canceled_counts():
    async def scenario():
        # Slow runner so cancellation lands while work remains.
        store = AnthropicBatchStore(runner=_ok_runner(delay=0.05))
        batch = store.create([_item(f"id-{i}") for i in range(6)])
        await asyncio.sleep(0.02)

        cancelled = store.cancel(batch.id)
        assert cancelled.processing_status == "canceling"
        assert cancelled.cancel_initiated_at is not None

        await _drain(store, batch.id)
        counts = store.get(batch.id).request_counts
        assert counts.canceled >= 1
        assert (counts.succeeded + counts.canceled) == 6

    asyncio.run(scenario())


def test_results_unavailable_while_processing():
    """Distinct from 'no such batch' - the client should keep polling."""

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner(delay=0.05))
        batch = store.create([_item("a"), _item("b")])
        assert store.results(batch.id) is None
        assert store.has(batch.id) is True
        await _drain(store, batch.id)
        assert store.results(batch.id) is not None

    asyncio.run(scenario())


def test_delete_refuses_while_processing_and_succeeds_after():
    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner(delay=0.05))
        batch = store.create([_item("a"), _item("b")])
        with pytest.raises(ValueError, match="cannot be deleted"):
            store.delete(batch.id)
        await _drain(store, batch.id)
        assert store.delete(batch.id) is True
        assert store.get(batch.id) is None

    asyncio.run(scenario())


def test_list_reports_has_more_and_pages_by_cursor():
    """A truncated page must be distinguishable from the end of the list.

    A hardcoded has_more=false leaves a client that asked for N and received N
    unable to tell a full page from the last one, so it stops early having
    silently missed batches.
    """

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        created = []
        for i in range(5):
            b = store.create([_item(f"b{i}")])
            await _drain(store, b.id)
            created.append(b.id)

        page1, more1 = store.list(limit=2)
        assert len(page1) == 2 and more1 is True

        page2, more2 = store.list(limit=2, after_id=page1[-1].id)
        assert len(page2) == 2 and more2 is True
        # Pages must not overlap, or a client double-counts.
        assert not ({b.id for b in page1} & {b.id for b in page2})

        page3, more3 = store.list(limit=2, after_id=page2[-1].id)
        assert len(page3) == 1 and more3 is False

        walked = [b.id for b in page1 + page2 + page3]
        assert sorted(walked) == sorted(created), "paging lost or repeated a batch"

    asyncio.run(scenario())


def test_before_id_returns_the_page_immediately_older_than_the_cursor():
    """Newest-first means "before" is the tail of the prefix, not its head.

    Taking the head returns the newest batches instead, and a client walking
    backwards then cursors on its own first page and dead-ends immediately.
    """

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        ids = []
        for i in range(10):
            b = store.create([_item(f"b{i}")])
            await _drain(store, b.id)
            ids.append(b.id)
        newest_first = [b.id for b in store.list(limit=10)[0]]

        cursor = newest_first[6]
        page, more = store.list(limit=3, before_id=cursor)
        assert [b.id for b in page] == newest_first[3:6], (
            f"expected the three immediately older than {cursor}, got {[b.id for b in page]}"
        )
        assert more is True

    asyncio.run(scenario())


def test_unknown_cursor_raises_rather_than_looking_finished():
    """An evicted cursor must not silently send the client back to page one."""

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        b = store.create([_item("a")])
        await _drain(store, b.id)

        # An empty page is byte-identical to a finished walk, so the client
        # would stop early having skipped everything past the cursor. This is
        # the normal loop: page, process, DELETE, then cursor on a deleted id.
        with pytest.raises(LookupError):
            store.list(limit=10, after_id="msgbatch_evicted")
        with pytest.raises(LookupError):
            store.list(limit=10, before_id="msgbatch_evicted")

    asyncio.run(scenario())


def test_list_ordering_is_total_so_cursors_are_stable():
    """Batches created in the same clock tick still need a deterministic order.

    created_at alone is not a total order at this resolution; without the id
    tiebreak a cursor can skip or repeat entries between calls.
    """

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner())
        for i in range(6):
            b = store.create([_item(f"x{i}")])
            await _drain(store, b.id)

        first = [b.id for b in store.list(limit=6)[0]]
        second = [b.id for b in store.list(limit=6)[0]]
        assert first == second, "listing order is not stable between calls"

    asyncio.run(scenario())


def test_unknown_batch_reads_as_missing():
    store = AnthropicBatchStore(runner=_ok_runner())
    assert store.get("msgbatch_nope") is None
    assert store.results("msgbatch_nope") is None
    assert store.cancel("msgbatch_nope") is None
    assert store.delete("msgbatch_nope") is False


def test_retention_evicts_only_ended_batches():
    """A live batch must never be evicted - its client is still polling."""

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner(), max_retained=2)
        first = store.create([_item("a")])
        await _drain(store, first.id)
        second = store.create([_item("b")])
        await _drain(store, second.id)
        third = store.create([_item("c")])
        await _drain(store, third.id)

        # The oldest ended batch went; the newest two survive.
        assert store.get(third.id) is not None
        assert store.get(first.id) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("concurrency", "size", "delay"),
    [
        pytest.param(2, 8, 0.02, id="bounded_and_actually_reached"),
        pytest.param(8, 40, 0.01, id="a_large_batch_is_not_serialized"),
    ],
)
def test_concurrency_is_bounded_and_actually_reached(concurrency, size, delay):
    """Both bounds matter, and the lower one is the load-bearing assertion.

    `peak <= limit` on its own is satisfied by doing no concurrency at all, so
    it passes happily against a store that holds the semaphore inside a
    sequential loop and never has more than one request in flight. Asserting
    the limit is *reached* is what proves requests are dispatched together;
    asserting it is not exceeded is what protects interactive traffic.

    The larger case exists because a batch must not degrade to one-at-a-time as
    it grows: it has five times more requests than slots, so every slot has to
    be refilled repeatedly, and every request still has to be accounted for at
    the end.
    """

    async def scenario():
        live = {"now": 0, "peak": 0}

        async def tracking(request):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(delay)
            live["now"] -= 1
            return "succeeded", {"type": "message"}

        store = AnthropicBatchStore(runner=tracking, concurrency=concurrency)
        batch = store.create([_item(f"id-{i}") for i in range(size)])
        await _drain(store, batch.id)
        assert live["peak"] <= concurrency, f"exceeded the limit: {live['peak']}"
        assert live["peak"] == concurrency, (
            f"never reached the limit (peak={live['peak']}): requests are being "
            "run one at a time, so the batch is serial"
        )
        assert store.get(batch.id).request_counts.succeeded == size

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


def test_expiry_pads_every_unfinished_request_exactly_once():
    """Padding must match on custom_id, not list position.

    Items are dispatched concurrently and each result is appended as it lands,
    so `results` is ordered by completion, not by submission. Padding by
    slicing `items` would expire the wrong requests -- duplicating some
    custom_ids and dropping others. Here 'b' finishes first, which makes the
    two orderings disagree.
    """

    async def finish_b_first(request):
        text = request.messages[0].content
        await asyncio.sleep(0.0 if text == "b" else 5.0)
        return "succeeded", {"type": "message", "role": "assistant", "content": []}

    async def scenario():
        store = AnthropicBatchStore(runner=finish_b_first, concurrency=4)
        items = [_item(cid, text=cid) for cid in ("a", "b", "c")]
        batch = store.create(AnthropicCreateBatchRequest(requests=items).requests)

        # Let 'b' land while 'a' and 'c' are still in flight.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if store._batches[batch.id].results:
                break

        # Force the batch past its TTL and re-read it.
        record = store._batches[batch.id]
        record.batch.expires_at = "2000-01-01T00:00:00.000Z"
        store.get(batch.id)

        assert store.status_of(batch.id) == "ended"
        return store.results(batch.id), batch.id

    results, _ = asyncio.run(scenario())

    ids = [obj["custom_id"] for obj in results]
    assert sorted(ids) == ["a", "b", "c"], f"expected each id once, got {ids}"

    by_id = {obj["custom_id"]: obj["result"]["type"] for obj in results}
    assert by_id["b"] == "succeeded"
    assert by_id["a"] == "expired"
    assert by_id["c"] == "expired"


def test_expiry_reports_expired_not_canceled():
    """Padding state is per-reason: expiry must not masquerade as cancellation."""

    async def never_finishes(request):
        await asyncio.sleep(30.0)
        return "succeeded", {}

    async def scenario():
        store = AnthropicBatchStore(runner=never_finishes)
        batch = store.create([_item("only")])
        await asyncio.sleep(0.02)
        store._batches[batch.id].batch.expires_at = "2000-01-01T00:00:00.000Z"
        store.get(batch.id)
        return store.results(batch.id), store.get(batch.id)

    results, batch = asyncio.run(scenario())
    assert [obj["result"]["type"] for obj in results] == ["expired"]
    assert batch.request_counts.expired == 1
    assert batch.request_counts.canceled == 0


# ---------------------------------------------------------------------------
# Admission control
# ---------------------------------------------------------------------------


def test_store_full_of_running_batches_refuses_instead_of_growing():
    """The retention limit has to bind even when nothing can be evicted.

    _evict_if_needed cannot drop a running batch -- that would strand the
    client polling it -- so once every retained slot holds live work there is
    nothing to reclaim. Inserting anyway makes the limit meaningless: batches,
    their queued requests and their results accumulate until the process dies,
    taking every other in-flight batch with it. Refusing is recoverable.
    """

    async def never_finishes(request):
        await asyncio.sleep(30.0)
        return "succeeded", {}

    async def scenario():
        store = AnthropicBatchStore(runner=never_finishes, max_retained=2)
        store.create([_item("a")])
        store.create([_item("b")])
        await asyncio.sleep(0.02)

        with pytest.raises(BatchStoreFullError):
            store.create([_item("c")])

        # The refusal must not have partially admitted the third batch.
        return len(store._batches)

    assert asyncio.run(scenario()) == 2


def test_capacity_frees_up_once_a_batch_ends():
    """Refusal is a back-off signal, not a permanent wall."""

    async def scenario():
        store = AnthropicBatchStore(runner=_ok_runner(), max_retained=2)
        first = store.create([_item("a")])
        second = store.create([_item("b")])
        await _drain(store, first.id)
        await _drain(store, second.id)

        # Both ended, so the oldest is evictable and a new batch fits.
        third = store.create([_item("c")])
        return store.get(third.id) is not None, len(store._batches)

    admitted, held = asyncio.run(scenario())
    assert admitted
    assert held == 2
