# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over the in-memory driver.

Covers the Store document's behaviour cases (a late write after a fault, results and access end in
either order, cancel and shutdown races) and the backend's own rules: the executor thread never
waits on the driver, each operation's access end is counted once, a fault waits for the batches
already started, and shutdown never calls the driver.
Every synchronisation is an event, a gate or a bounded wait on a condition.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from cpu_fixtures import (
    SLOT_BYTES,
    WAIT_S,
    CpuRegion,
    CpuView,
    Endpoint,
    FakeHold,
    KeyGatedStore,
    Rows,
    commit_object,
    object_key,
    store_key,
    wait_until,
)
from memory_store import DEFAULT_CAPABILITIES, Commit, MemoryDriver, MemoryStore

from tensorrt_llm._torch.disaggregation.base.backend import (
    Capabilities,
    InvalidSubmission,
    SubmissionRejected,
)
from tensorrt_llm._torch.disaggregation.blob import backend as backend_module
from tensorrt_llm._torch.disaggregation.blob.backend import (
    SHUTDOWN_FAULT,
    BlobConfig,
    RegionBinding,
)
from tensorrt_llm._torch.disaggregation.blob.store import DriverRejected, RowIO, StoreError

pytestmark = pytest.mark.cpu_only

# A test that leaves an operation whose access never ends keeps that backend's progress thread
# polling, as it would in production; the thread-leak check would report it.
keeps_backend_threads = pytest.mark.threadleak(enabled=False)

PROMPT_S = 0.5
"""A call that must not wait on the driver returns well within this."""


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


def published(store: MemoryStore, rows: dict[str, bytes]) -> None:
    """Commits ``rows`` straight into ``store`` under the keys the backend derives."""
    for name, data in rows.items():
        assert commit_object(store, name, data) is Commit.STORED


def masks(outcome) -> list[list[bool]]:
    return [mask.tolist() for mask in outcome.delivered_masks]


def store_calls(endpoint: Endpoint) -> list[str]:
    return [call.method for call in endpoint.driver.calls if call.method in ("get", "put")]


# Results.


@pytest.mark.parametrize("case", ["fetch_not_ready", "publish_declined"])
def test_not_ready_and_declined_rows_are_undelivered_without_a_fault(case: str) -> None:
    # A fetch finds the first object still being written (NOT_READY); a publish finds no room
    # after the first row (DECLINED).
    store = MemoryStore(capacity=1 if case == "publish_declined" else None)
    endpoint = Endpoint(MemoryDriver(store), ["A", "B", "C"], rows_per_batch=1)
    if case == "fetch_not_ready":
        published(store, {"B": b"bbbbbbbb", "C": b"cccccccc"})
        store.begin_write(store_key("A"))
        attempt = endpoint.backend.fetch(endpoint.view("A", "B", "C"), None)
        expected = [False, True, True]
    else:
        for name in "ABC":
            endpoint.write(name, name.encode() * SLOT_BYTES)
        attempt = endpoint.backend.publish(endpoint.view("A", "B", "C"), None)
        expected = [True, False, False]
    outcome = endpoint.wait_outcome(attempt)
    assert outcome.fault is None and masks(outcome) == [expected]
    assert len(store_calls(endpoint)) == 3  # an untaken row stops nothing
    endpoint.wait_quiesced(attempt)
    endpoint.close()


def test_masks_follow_view_group_order(store: MemoryStore) -> None:
    published(store, {"A": b"aaaaaaaa", "C": b"cccccccc", "E": b"eeeeeeee"})
    reader = Endpoint(MemoryDriver(store), ["A", "B", "C", "D", "E"], rows_per_batch=2)
    view = CpuView([reader.rows(["C", "D"], 2), reader.rows(["A"], 0), reader.rows(["E", "B"], 1)])
    attempt = reader.backend.fetch(view, None)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[True, False], [True], [True, False]]
    outcome.check_mask_shapes(view)
    reader.wait_quiesced(attempt)
    reader.close()


def test_empty_view_concludes_and_quiesces_at_once(store: MemoryStore) -> None:
    reader = Endpoint(MemoryDriver(store), ["A"])
    view = CpuView([reader.rows([], 0), reader.rows([], 1)])
    attempt = reader.backend.fetch(view, None)
    outcome = attempt.poll()
    assert outcome is not None and outcome.fault is None
    assert [mask.shape for mask in outcome.delivered_masks] == [(0,), (0,)]
    assert attempt.quiesced()
    assert reader.wake.count >= 1
    assert store_calls(reader) == []
    reader.close()


@pytest.mark.parametrize("limit", ["rows", "bytes", "driver"])
def test_batches_respect_rows_bytes_and_driver_limits(store: MemoryStore, limit: str) -> None:
    names = [f"R{i}" for i in range(10)]
    if limit == "rows":
        writer, expected = Endpoint(MemoryDriver(store), names, rows_per_batch=3), [3, 3, 3, 1]
    elif limit == "bytes":
        writer = Endpoint(MemoryDriver(store), names, bytes_per_batch=2 * SLOT_BYTES + 1)
        expected = [2] * 5
    else:
        driver = MemoryDriver(store, capabilities=replace(DEFAULT_CAPABILITIES, max_rows_per_op=4))
        writer, expected = Endpoint(driver, names), [4, 4, 2]
    attempt = writer.backend.publish(writer.view(*names), None)
    assert masks(writer.wait_outcome(attempt)) == [[True] * 10]
    assert [len(call.rows) for call in writer.driver.calls_to("put")] == expected
    writer.wait_quiesced(attempt)
    writer.close()


def test_capabilities_are_the_driver_directions(store: MemoryStore) -> None:
    driver = MemoryDriver(store, capabilities=replace(DEFAULT_CAPABILITIES, get=False))
    writer = Endpoint(driver, ["A"])
    assert writer.backend.capabilities == Capabilities(fetch=False, publish=True, cancel=True)
    with pytest.raises(InvalidSubmission, match="cannot get"):
        writer.backend.fetch(writer.view("A"), None)
    assert store_calls(writer) == []
    attempt = writer.backend.publish(writer.view("A"), None)
    assert masks(writer.wait_outcome(attempt)) == [[True]]
    writer.wait_quiesced(attempt)
    writer.close()


def test_blob_config_refuses_bad_values() -> None:
    for prefix in ("", "a/b", "a b"):
        with pytest.raises(ValueError, match="prefix"):
            BlobConfig(key_prefix=prefix)
    with pytest.raises(ValueError, match="at least 1"):
        BlobConfig(rows_per_batch=0)
    with pytest.raises(TypeError, match="rows_per_bach"):
        BlobConfig(rows_per_bach=1)


# Results, access end and late writes.


def test_late_write_after_fault_changes_nothing(store: MemoryStore) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D, workers=2, rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_a, op_b = reader.op("A"), reader.op("B")
    driver.complete(op_a)
    driver.end_access(op_a)
    driver.fail(op_b, "timeout")
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[True, False]]
    assert "timeout" in outcome.fault
    assert not attempt.quiesced()

    driver.write_late(op_b, 0, b"bbbbbbbb")
    assert reader.read("B") == b"bbbbbbbb"
    assert attempt.poll() is outcome
    driver.end_access(op_b)
    reader.wait_quiesced(attempt)
    assert attempt.poll() is outcome
    reader.close()


def test_access_end_before_result_without_touching_the_view(store: MemoryStore) -> None:
    published(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"], fill=0x2D)
    view = reader.view("A")
    attempt = reader.backend.fetch(view, None)
    view.seal()
    op = reader.op("A")
    driver.transfer(op)
    driver.end_access(op)
    reader.wait_quiesced(attempt)
    assert attempt.poll() is None
    driver.publish_results(op)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[True]]
    assert reader.read("A") == b"aaaaaaaa"
    reader.close()


@pytest.mark.parametrize("first_row", ["ready", "faulty"])
def test_a_later_batch_fault_waits_for_earlier_batches_and_the_first_recorded_is_reported(
    first_row: str,
) -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("A")])
    a = b"aaaaaaaa" if first_row == "ready" else b"aaa"  # the wrong size: a second fault
    published(store, {"A": a, "B": b"short", "C": b"cccccccc"})
    reader = Endpoint(MemoryDriver(store), ["A", "B", "C"], workers=2, rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    try:
        store.wait_for_waiting(1)
        # B entered the driver after A, and its fault is recorded once only A's access is open.
        wait_until(
            lambda: (
                len(reader.driver.calls_to("get")) == 2
                and reader.backend.stats()["batches_access_open"] == 1
            ),
            "B's fault to be recorded",
        )
        assert attempt.poll() is None
        assert attempt.quiesced() is False
    finally:
        store.gate.set()
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[first_row == "ready", False, False]]  # A's row is kept for reuse
    assert "row 1: object is 5 bytes" in outcome.fault and "3 bytes" not in outcome.fault
    reader.wait_quiesced(attempt)
    assert [call.rows[0].key for call in reader.driver.calls_to("get")] == [
        store_key("A"),
        store_key("B"),
    ]
    reader.close()


@pytest.mark.parametrize("first_half", ["results_first", "access_end_first"])
def test_results_and_access_end_are_each_recorded_once_per_batch(
    store: MemoryStore, first_half: str
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_x, op_y = reader.op("A"), reader.op("B")
    # X lands one half only, so every progress round observes it again.
    if first_half == "results_first":
        driver.complete(op_x)
    else:
        driver.transfer(op_x)
        driver.end_access(op_x)

    def progress_polls_of_x() -> int:
        polls = reader.driver.calls_to("poll", op_x)
        return sum(call.thread.endswith("-progress") for call in polls)

    reader.driver.wait_for(lambda: progress_polls_of_x() >= 5, "five progress rounds")
    assert attempt.poll() is None  # Y has no result yet
    assert not attempt.quiesced()  # Y's access is open
    driver.complete(op_y)
    driver.end_access(op_y)
    if first_half == "results_first":
        driver.end_access(op_x)
    else:
        driver.publish_results(op_x)
    assert masks(reader.wait_outcome(attempt)) == [[True, True]]
    reader.wait_quiesced(attempt)
    reader.close()


@keeps_backend_threads
@pytest.mark.parametrize("ending", ["access_ended", "unprovable"])
def test_a_settled_operation_is_no_longer_observed(store: MemoryStore, ending: str) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D)
    first = reader.backend.fetch(reader.view("A"), None)
    op_a = reader.op("A")
    if ending == "access_ended":
        driver.complete(op_a)
        driver.end_access(op_a)
        reader.wait_quiesced(first)
    else:
        driver.fail(op_a, "the store gave up")
        driver.mark_unprovable(op_a)  # nothing the driver could wait on was issued
        assert "the store gave up" in reader.wait_outcome(first).fault
        wait_until(lambda: reader.backend.stats()["batches_unprovable"] == 1, "the count")
    polls_of_a = len(reader.driver.calls_to("poll", op_a))

    second = reader.backend.fetch(reader.view("B"), None)  # the backend serves on
    op_b = reader.op("B")

    def progress_polls_of_b() -> int:
        polls = reader.driver.calls_to("poll", op_b)
        return sum(call.thread.endswith("-progress") for call in polls)

    reader.driver.wait_for(lambda: progress_polls_of_b() >= 3, "three progress rounds")
    assert len(reader.driver.calls_to("poll", op_a)) == polls_of_a
    driver.complete(op_b)
    driver.end_access(op_b)
    reader.wait_quiesced(second)
    if ending == "access_ended":
        reader.close()
        return
    assert not first.quiesced()  # its slots stay held
    assert reader.backend.stats()["batches_access_open"] == 1
    assert reader.backend.shutdown(0) == frozenset({first.label, "region:0"})


def test_a_stuck_driver_call_is_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module, "STUCK_CALL_WARNING_S", 0.05)
    warnings: list[str] = []  # every line emitted, whatever the log level
    monkeypatch.setattr(
        backend_module.logger, "log", lambda severity, *msg: warnings.append(msg[0])
    )
    store = KeyGatedStore(gets=True)
    published(store, {"A": b"aaaaaaaa"})
    reader = Endpoint(MemoryDriver(store), ["A"])
    attempt = reader.backend.fetch(reader.view("A"), None)

    def stuck() -> list[str]:
        return [line for line in warnings if "has run for over" in line]

    try:
        store.wait_for_waiting(1)
        wait_until(stuck, "the stuck-call warning")
        reported_at_ms = reader.backend.stats()["oldest_driver_call_ms"]
        assert reported_at_ms >= 50
        # Several more progress rounds (at most 100 ms apart) see the same call.
        wait_until(
            lambda: reader.backend.stats()["oldest_driver_call_ms"] >= reported_at_ms + 300,
            "more progress rounds",
        )
        (line,) = stuck()
        assert attempt.label in line
    finally:
        store.gate.set()
    reader.wait_quiesced(attempt)
    assert reader.backend.stats()["oldest_driver_call_ms"] == 0
    reader.close()


@pytest.mark.parametrize(
    "method", ["drain", "unregister"], ids=["drain_answers_false", "unregister_raises"]
)
def test_the_progress_thread_retries_drain_and_revocation_and_the_hold_stays(
    store: MemoryStore, method: str
) -> None:
    endpoint = Endpoint(MemoryDriver(store), ["A"])
    if method == "drain":
        endpoint.driver.answer("drain", False, times=1_000_000)
    else:
        endpoint.driver.fail("unregister", StoreError("busy"), times=1_000_000)
    assert endpoint.backend.shutdown(0) == frozenset({"region:0"})
    endpoint.driver.wait_for(lambda: len(endpoint.driver.calls_to(method)) >= 3, "retries")
    if method == "drain":
        assert endpoint.driver.calls_to("unregister") == []  # never revoked before a True drain
    with pytest.raises(RuntimeError, match="region:0"):
        endpoint.binding.release_hold()
    assert endpoint.binding.live_labels() == frozenset({"region:0"})
    if method == "drain":
        endpoint.driver.answer("drain", False, times=0)
    else:
        endpoint.driver.fail("unregister", StoreError("busy"), times=0)
    # Nothing else calls in: the progress thread retries on its own.
    wait_until(lambda: not endpoint.binding.live_labels(), "the revocation to succeed")
    endpoint.close()


# Cancel.


@pytest.mark.parametrize("first", ["cancel_first", "complete_first"])
def test_cancel_racing_completion_gives_one_outcome_and_never_reaches_the_driver(
    store: MemoryStore, first: str
) -> None:
    published(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    attempt = reader.backend.fetch(reader.view("A"), None)
    op = reader.op("A")
    if first == "cancel_first":
        attempt.cancel()
        assert attempt.poll() is None
        assert reader.wake.count == 0  # nothing landed, so nothing to wake for
        driver.complete(op)
        driver.end_access(op)
    else:
        driver.complete(op)
        driver.end_access(op)
        reader.wait_outcome(attempt)
        reader.wait_quiesced(attempt)
        wakes = reader.wake.count
        attempt.cancel()
        assert reader.wake.count == wakes
    outcome = reader.wait_outcome(attempt)
    reader.wait_quiesced(attempt)
    assert attempt.poll() is outcome
    assert masks(outcome) == [[True]] and outcome.fault is None
    assert 1 <= reader.wake.count <= 2  # the outcome and the quiescing, together or apart
    assert reader.driver.calls_to("cancel") == []  # advisory: never forwarded
    reader.close()


# Driver failures.


def unbuildable_rows(*_) -> list[RowIO]:
    raise ValueError("x")


@keeps_backend_threads
@pytest.mark.parametrize(
    "raised", ["driver_rejected", "while_building_rows", "outside_the_protocol"]
)
def test_a_raise_faults_the_batch_and_only_a_refusal_ends_its_access(
    store: MemoryStore, raised: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    if raised == "driver_rejected":
        writer.driver.fail("put", DriverRejected("x"))
    elif raised == "while_building_rows":  # before the driver call
        monkeypatch.setattr(backend_module.BlobBackend, "_row_ios", unbuildable_rows)
    else:  # the rows may be accessed for good
        writer.driver.fail("put", RuntimeError("x"))
    attempt = writer.backend.publish(writer.view("A"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False]] and "x" in outcome.fault
    if raised != "outside_the_protocol":
        assert "refused: x" in outcome.fault
        writer.wait_quiesced(attempt)
        assert writer.backend.stats()["batches_unprovable"] == 0
        writer.close()
        return
    assert not attempt.quiesced()
    assert writer.backend.stats()["batches_unprovable"] == 1
    assert attempt.label in writer.backend.shutdown(0)
    with pytest.raises(RuntimeError, match="region:0"):
        writer.binding.release_hold()
    assert not writer.hold.released


class ShortPollOnceDriver(MemoryDriver):
    """Answers one result too few the first time it polls an operation of more than one row."""

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(store)
        self._answered_short: set[int] = set()

    def poll(self, op):
        results = super().poll(op)
        if len(results) > 1 and id(op) not in self._answered_short:
            self._answered_short.add(id(op))
            return results[:-1]
        return results


@keeps_backend_threads
def test_a_poll_with_the_wrong_row_count_faults_the_batch_and_keeps_it_held(
    store: MemoryStore,
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    reader = Endpoint(ShortPollOnceDriver(store), ["A", "B", "C"])
    reader.write("C", b"cccccccc")
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[False, False]]
    assert "gave 1 results for 2 rows" in outcome.fault
    # Later answers have the right length and say access ended; after the violation they prove
    # nothing, however often the progress thread asks.
    op = reader.op("A")

    def progress_polls() -> int:
        return sum(call.thread.endswith("-progress") for call in reader.driver.calls_to("poll", op))

    reader.driver.wait_for(lambda: progress_polls() >= 3, "three progress rounds")
    assert not attempt.quiesced()
    # The worker survived the bad answer and serves the next submission.
    later = reader.backend.publish(reader.view("C"), None)
    assert masks(reader.wait_outcome(later)) == [[True]]
    reader.wait_quiesced(later)
    assert attempt.label in reader.backend.shutdown(0)


@pytest.mark.parametrize("failure", ["access_ended_raises", "poll_answers_no_sequence"])
def test_an_observation_that_fails_is_left_to_the_progress_thread(
    store: MemoryStore, failure: str
) -> None:
    writer = Endpoint(MemoryDriver(store), ["A", "B"])
    workers = [thread for thread in writer.threads if "-worker-" in thread.name]
    writer.write("A", b"aaaaaaaa")
    writer.write("B", b"bbbbbbbb")
    if failure == "access_ended_raises":
        method = "access_ended"
        writer.driver.fail(method, RuntimeError("flaky"))
    else:
        method = "poll"
        writer.driver.answer(method, None)  # the worker's look at the first operation
    first = writer.backend.publish(writer.view("A"), None)
    assert masks(writer.wait_outcome(first)) == [[True]]
    writer.wait_quiesced(first)
    checks = writer.driver.calls_to(method)
    assert "-worker-" in checks[0].thread
    assert any(call.thread.endswith("-progress") for call in checks[1:])
    second = writer.backend.publish(writer.view("B"), None)  # the worker is back
    assert masks(writer.wait_outcome(second)) == [[True]]
    writer.wait_quiesced(second)
    assert all(worker.is_alive() for worker in workers)
    writer.close()


def test_a_raising_wake_does_not_stop_the_backend(store: MemoryStore) -> None:
    def broken_wake() -> None:
        raise RuntimeError("holder gone")

    writer = Endpoint(MemoryDriver(store), ["A", "B"], on_wake=broken_wake)
    writer.write("A", b"aaaaaaaa")
    writer.write("B", b"bbbbbbbb")
    first = writer.backend.publish(writer.view("A"), None)
    wait_until(lambda: first.poll() is not None and first.quiesced(), "A published")
    second = writer.backend.publish(writer.view("B"), None)
    wait_until(lambda: second.poll() is not None and second.quiesced(), "B published")
    assert masks(first.poll()) == masks(second.poll()) == [[True]]
    writer.close()


# Submission.


def _view_with(endpoint: Endpoint, **changes) -> CpuView:
    rows = endpoint.rows(["A"])
    fields = {
        "layer_group_index": 0,
        "object_keys": rows.object_keys,
        "addresses": rows.addresses,
        "region_index": 0,
    }
    fields.update(changes)
    return CpuView([Rows(**fields)])


def _slots(endpoint: Endpoint, *indexes: int) -> np.ndarray:
    return np.array([endpoint.region.slot(index) for index in indexes], dtype=np.int64)


INVALID_SUBMISSIONS: dict[str, Callable[[Endpoint], tuple[CpuView, object]]] = {
    "hint": lambda e: (e.view("A"), "peer-0"),
    "no_keys": lambda e: (_view_with(e, object_keys=None, addresses=None, region_index=None), None),
    "key_dtype": lambda e: (_view_with(e, object_keys=np.zeros((1, 4), dtype=np.int32)), None),
    "key_width": lambda e: (_view_with(e, object_keys=np.zeros((1, 0), dtype=np.uint8)), None),
    "address_dtype": lambda e: (_view_with(e, addresses=_slots(e, 0).astype(np.uint64)), None),
    "length": lambda e: (_view_with(e, rows=2), None),
    "past_the_region": lambda e: (_view_with(e, addresses=_slots(e, 2)), None),
    "before_the_region": lambda e: (_view_with(e, addresses=_slots(e, -1)), None),
    "misaligned": lambda e: (_view_with(e, addresses=_slots(e, 0) + 1), None),
    "region_index": lambda e: (_view_with(e, region_index=1), None),
    "negative_region_index": lambda e: (_view_with(e, region_index=-1), None),
    "same_slot_in_a_group": lambda e: (
        CpuView([Rows(0, np.stack([object_key("A"), object_key("B")]), _slots(e, 0, 0), 0)]),
        None,
    ),
    "same_slot_across_groups": lambda e: (CpuView([e.rows(["A"], 0), e.rows(["A"], 1)]), None),
}


@pytest.mark.parametrize("case", INVALID_SUBMISSIONS)
def test_invalid_views_are_rejected_before_anything_escapes(store: MemoryStore, case: str) -> None:
    reader = Endpoint(MemoryDriver(store), ["A", "B"])
    view, hint = INVALID_SUBMISSIONS[case](reader)
    with pytest.raises(InvalidSubmission):
        reader.backend.fetch(view, hint)
    assert store_calls(reader) == []
    assert reader.backend.stats()["attempts_active"] == 0
    reader.close()


def test_a_publish_may_read_one_slot_twice(store: MemoryStore) -> None:
    writer = Endpoint(MemoryDriver(store), ["A", "B"])
    writer.write("A", b"aaaaaaaa")
    keys = np.stack([object_key("A"), object_key("B")])
    view = CpuView([Rows(0, keys, np.array([writer.region.slot(0)] * 2, dtype=np.int64), 0)])
    attempt = writer.backend.publish(view, None)
    assert masks(writer.wait_outcome(attempt)) == [[True, True]]
    assert store.committed_payload(store_key("B")) == b"aaaaaaaa"
    writer.wait_quiesced(attempt)
    writer.close()


def test_publish_is_not_starved_by_fetches() -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("F0")])
    fetched = [f"F{i}" for i in range(11)]
    published(store, {name: bytes(SLOT_BYTES) for name in fetched})
    endpoint = Endpoint(MemoryDriver(store), [*fetched, "P1", "P2"])
    endpoint.write("P1", b"pppppppp")
    endpoint.write("P2", b"qqqqqqqq")
    blocker = endpoint.backend.fetch(endpoint.view("F0"), None)
    try:
        store.wait_for_waiting(1)  # the only worker is inside the store
        attempts = [endpoint.backend.fetch(endpoint.view(name), None) for name in fetched[1:]]
        attempts += [endpoint.backend.publish(endpoint.view(name), None) for name in ("P1", "P2")]
    finally:
        store.gate.set()
    for attempt in [blocker, *attempts]:
        assert masks(endpoint.wait_outcome(attempt)) == [[True]]
        endpoint.wait_quiesced(attempt)
    four_fetches_then_a_publish = ["get"] * 4 + ["put"]
    assert store_calls(endpoint)[:10] == four_fetches_then_a_publish * 2
    endpoint.close()


def test_submissions_return_while_every_worker_is_blocked_in_the_driver() -> None:
    store = KeyGatedStore(gets=True)
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    endpoint = Endpoint(MemoryDriver(store), ["A", "B", "C", "D"], workers=2, rows_per_batch=1)
    endpoint.write("C", b"cccccccc")
    blocked = endpoint.backend.fetch(endpoint.view("A", "B"), None)
    try:
        store.wait_for_waiting(2)
        calls_before = len(endpoint.driver.calls)
        for submit in (
            lambda: endpoint.backend.fetch(endpoint.view("D"), None),
            lambda: endpoint.backend.publish(endpoint.view("C"), None),
            blocked.cancel,
        ):
            started = time.monotonic()
            submit()
            assert time.monotonic() - started < PROMPT_S
        assert len(endpoint.driver.calls) == calls_before
        here = threading.current_thread().name
        assert all(
            call.thread != here for call in endpoint.driver.calls if call.method != "register"
        )
        # The cancel cleared the queue, but both batches are still inside the store.
        assert blocked.quiesced() is False
        assert blocked.label in endpoint.backend.shutdown(0)
    finally:
        store.gate.set()
    endpoint.wait_quiesced(blocked)
    endpoint.close()


# Shutdown.


def test_shutdown_forces_fault_outcomes_and_waits_out_its_budget(store: MemoryStore) -> None:
    published(store, {"B": b"short"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], workers=2, rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_a, op_b = reader.op("A"), reader.op("B")
    driver.complete(op_b)  # B's fault (the wrong size) lands; A has no result, neither access ends
    # Every poll from here on sees B's result, and B's next poll starts only after the last one
    # is recorded: two more polls mean the fault is recorded.
    polls_of_b = len(reader.driver.calls_to("poll", op_b))
    reader.driver.wait_for(
        lambda: len(reader.driver.calls_to("poll", op_b)) >= polls_of_b + 2, "B's fault"
    )
    started = time.monotonic()
    labels = reader.backend.shutdown(0.2)
    assert time.monotonic() - started >= 0.19  # both batches may still access their slots
    outcome = attempt.poll()
    assert outcome is not None and SHUTDOWN_FAULT in outcome.fault
    assert "first fault: group 0 row 1: object is 5 bytes" in outcome.fault
    assert masks(outcome) == [[False, False]]
    assert labels == frozenset({attempt.label, "region:0"})
    with pytest.raises(RuntimeError, match="region:0"):
        reader.binding.release_hold()
    assert not reader.hold.released
    reader.backend.shutdown(0)
    assert attempt.poll() is outcome
    driver.fail(op_a, "late")
    driver.end_access(op_a)
    driver.end_access(op_b)
    reader.wait_quiesced(attempt)
    assert attempt.poll() is outcome
    reader.close()


@pytest.mark.parametrize("stop", ["cancel", "shutdown"])
def test_cancel_and_shutdown_drop_the_batches_that_have_not_started(stop: str) -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("A")])
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    reader = Endpoint(MemoryDriver(store), ["A", "B", "C"], rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    try:
        store.wait_for_waiting(1)  # the only worker is inside the store with A
        if stop == "cancel":
            attempt.cancel()
            assert attempt.poll() is None  # A is still in the store
        else:
            assert attempt.label in reader.backend.shutdown(0)
    finally:
        store.gate.set()
    reader.wait_quiesced(attempt)
    outcome = attempt.poll()
    if stop == "cancel":  # A's row is delivered, without a fault
        assert outcome.fault is None and masks(outcome) == [[True, False, False]]
    else:
        assert SHUTDOWN_FAULT in outcome.fault and masks(outcome) == [[False, False, False]]
    assert store_calls(reader) == ["get"]
    assert reader.read("B") == reader.read("C") == bytes(SLOT_BYTES)
    reader.close()


@pytest.mark.parametrize("before", ["idle", "after_round_trip", "after_driver_violation"])
def test_clean_shutdown_returns_well_before_its_budget_and_rejects_later_submissions(
    store: MemoryStore, before: str
) -> None:
    endpoint = Endpoint(MemoryDriver(store), ["A"])
    endpoint.write("A", b"aaaaaaaa")
    if before == "after_driver_violation":
        endpoint.driver.fail("put", DriverRejected("refused"))
    if before != "idle":
        attempt = endpoint.backend.publish(endpoint.view("A"), None)
        endpoint.wait_quiesced(attempt)
    calls = len(store_calls(endpoint))
    started = time.monotonic()
    assert endpoint.backend.shutdown(5.0) == frozenset()
    assert time.monotonic() - started < PROMPT_S
    for submit in (endpoint.backend.fetch, endpoint.backend.publish):
        with pytest.raises(SubmissionRejected):
            submit(endpoint.view("A"), None)
    assert len(store_calls(endpoint)) == calls
    endpoint.close()


def test_shutdown_never_calls_the_driver_and_unregisters_in_the_background(
    store: MemoryStore,
) -> None:
    endpoint = Endpoint(MemoryDriver(store), ["A"])
    revoke = endpoint.driver.hold("unregister")
    try:
        started = time.monotonic()
        assert endpoint.backend.shutdown(0) == frozenset({"region:0"})
        assert time.monotonic() - started < PROMPT_S
        endpoint.driver.wait_for(lambda: endpoint.driver.calls_to("unregister"), "a revocation")
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="region:0"):
            endpoint.binding.release_hold()  # reads the registrations without waiting on it
        assert time.monotonic() - started < PROMPT_S
    finally:
        revoke.set()
    endpoint.close()
    background = [c for c in endpoint.driver.calls if c.method in ("drain", "unregister")]
    assert background and all(call.thread.endswith("-progress") for call in background)


class WakeProbe:
    """On each wake, reads the backend from a helper thread as a holder would. A wake made under
    the backend's lock would leave the helper blocked past its bound."""

    def __init__(self) -> None:
        self.attempt = None
        self.backend = None
        self.seen: list[dict] = []

    def __call__(self) -> None:
        attempt, backend, seen = self.attempt, self.backend, {}
        if attempt is None:
            return

        def read() -> None:
            seen["outcome"] = attempt.poll()
            seen["quiesced"] = attempt.quiesced()
            seen["stats"] = backend.stats()
            attempt.cancel()
            seen["finished"] = True

        helper = threading.Thread(target=read, daemon=True)
        helper.start()
        helper.join(WAIT_S)
        self.seen.append(seen)


@pytest.mark.parametrize("recorder", ["worker", "progress"])
def test_wake_runs_after_state_is_published_and_outside_the_lock_on_record(recorder: str) -> None:
    store = KeyGatedStore(gets=True)
    published(store, {"A": b"aaaaaaaa"})
    probe = WakeProbe()
    # A worker records an operation the driver settled before returning; the progress thread
    # records one the driver settles later.
    driver = MemoryDriver(store, controlled=recorder == "progress")
    reader = Endpoint(driver, ["A"], on_wake=probe)
    probe.backend = reader.backend
    probe.attempt = attempt = reader.backend.fetch(reader.view("A"), None)
    store.gate.set()
    if recorder == "progress":
        op = reader.op("A")
        driver.complete(op)
        driver.end_access(op)
    reader.wait_quiesced(attempt)
    reader.wait_outcome(attempt)
    assert probe.seen and all(seen.get("finished") for seen in probe.seen)
    assert any(seen["outcome"] is not None for seen in probe.seen)
    assert any(seen["quiesced"] for seen in probe.seen)
    reader.close()


def test_wake_runs_after_state_is_published_and_outside_the_lock_on_forced_shutdown() -> None:
    store = KeyGatedStore(gets=True)
    published(store, {"A": b"aaaaaaaa"})
    probe = WakeProbe()
    reader = Endpoint(MemoryDriver(store), ["A"], on_wake=probe)
    probe.backend = reader.backend
    probe.attempt = reader.backend.fetch(reader.view("A"), None)
    labels = []
    shutter = threading.Thread(target=lambda: labels.append(reader.backend.shutdown(WAIT_S)))
    try:
        store.wait_for_waiting(1)
        shutter.start()
        reader.wake.wait_for(lambda: reader.wake.count >= 1, "a wake for the forced outcome")
        assert shutter.is_alive()  # it cannot return while the batch is in the store
        (seen,) = probe.seen
        assert seen.get("finished") and SHUTDOWN_FAULT in seen["outcome"].fault
    finally:
        store.gate.set()
        shutter.join(WAIT_S)
    assert labels == [frozenset()]
    reader.close()


# Registration.


def test_region_binding_skips_regions_without_slots(store: MemoryStore) -> None:
    driver = MemoryDriver(store)
    binding = RegionBinding(
        (CpuRegion(SLOT_BYTES, 0), CpuRegion(SLOT_BYTES, 1)), driver, FakeHold()
    )
    binding.register()
    assert binding.live_labels() == frozenset({"region:1"})
    assert binding.unregister() and driver.live_registrations() == []
