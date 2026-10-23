# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over the in-memory driver.

Covers the Store document's behaviour cases (a late write after a fault, results and access end in
either order, cancel and shutdown races) and the backend's own rules: the submitting thread never
waits on the driver, only regions the driver can access are registered and rows elsewhere are
refused, each operation's access end is counted once, a fault waits for the batches already
started, and shutdown never calls the driver.
Every synchronisation is an event, a gate or a bounded wait on a condition.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from cpu_fixtures import (
    PROMPT_S,
    SLOT_BYTES,
    WAIT_S,
    CpuRegion,
    CpuView,
    Endpoint,
    FakeHold,
    KeyGatedStore,
    RollbackFailsDriver,
    Rows,
    TogetherDriverSpy,
    commit_object,
    in_a_child_process,
    masks,
    object_key,
    store_key,
    wait_until,
)
from together_driver import TogetherDriver

from tensorrt_llm._torch.disaggregation.base.backend import (
    Capabilities,
    InvalidSubmission,
    MemoryKind,
    SubmissionRejected,
)
from tensorrt_llm._torch.disaggregation.blob import backend as backend_module
from tensorrt_llm._torch.disaggregation.blob.backend import (
    FETCH_BATCHES_PER_PUBLISH_BATCH,
    SHUTDOWN_FAULT,
    BlobBackend,
    BlobConfig,
    ConstructionRejected,
    RegionBinding,
)
from tensorrt_llm._torch.disaggregation.blob.store import DriverRejected, RowIO, StoreError, faulted
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    DEFAULT_CAPABILITIES,
    Commit,
    MemoryDriver,
    MemoryStore,
)

pytestmark = pytest.mark.cpu_only

# A test that leaves an operation whose access never ends keeps that backend's progress thread
# polling, as it would in production; the thread-leak check would report it.
keeps_backend_threads = pytest.mark.threadleak(enabled=False)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


def published(store: MemoryStore, rows: dict[str, bytes]) -> None:
    """Commits ``rows`` straight into ``store`` under the keys the backend derives."""
    for name, data in rows.items():
        assert commit_object(store, name, data) is Commit.STORED


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
    assert len(store_calls(endpoint)) == 3  # a skipped row stops nothing
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


BATCH_LIMITS = {
    # name: (rows_per_batch, bytes_per_batch, the driver's max_rows_per_op, rows in each call)
    "rows": (3, None, None, [3, 3, 3, 1]),
    "bytes": (None, 2 * SLOT_BYTES + 1, None, [2] * 5),
    "driver": (None, None, 4, [4, 4, 2]),
    "the_smallest_wins": (5, 3 * SLOT_BYTES, 4, [3, 3, 3, 1]),
    "no_limit": (None, None, None, [10]),  # the whole group in one call
}


@pytest.mark.parametrize("case", BATCH_LIMITS)
def test_batches_respect_rows_bytes_and_driver_limits(store: MemoryStore, case: str) -> None:
    rows_per_batch, bytes_per_batch, max_rows_per_op, expected = BATCH_LIMITS[case]
    names = [f"R{i}" for i in range(10)]
    capabilities = replace(DEFAULT_CAPABILITIES, max_rows_per_op=max_rows_per_op)
    writer = Endpoint(
        MemoryDriver(store, capabilities=capabilities),
        names,
        rows_per_batch=rows_per_batch,
        bytes_per_batch=bytes_per_batch,
    )
    attempt = writer.backend.publish(writer.view(*names), None)
    assert masks(writer.wait_outcome(attempt)) == [[True] * 10]
    assert [len(call.rows) for call in writer.driver.calls_to("put")] == expected
    limited = case != "no_limit"
    assert writer.backend.batch_size(0) == (
        (expected[0], expected[0] * SLOT_BYTES) if limited else None
    )
    writer.wait_quiesced(attempt)
    writer.close()


REPEATED_KEYS = {
    # name: (rows_per_batch, the keys the view names at slots 0, 1, 2, ..., rows in each call)
    "no_limit": (None, "ABA", [2, 1]),
    "limit_of_two": (2, "AABC", [1, 2, 1]),
    "every_row_alike": (None, "AAA", [1, 1, 1]),
}


@pytest.mark.parametrize("direction", ["publish", "fetch"])
@pytest.mark.parametrize("case", REPEATED_KEYS)
def test_rows_naming_one_key_at_two_addresses_never_share_a_call(
    store: MemoryStore, case: str, direction: str
) -> None:
    """A view may name one key at two addresses; a driver refuses a key twice in one call."""
    rows_per_batch, keys, expected = REPEATED_KEYS[case]
    slots = [f"S{i}" for i in range(len(keys))]  # one slot per row, named apart from the keys
    endpoint = Endpoint(MemoryDriver(store), slots, rows_per_batch=rows_per_batch)
    payload = {key: key.encode() * SLOT_BYTES for key in set(keys)}
    view = CpuView(
        [
            Rows(
                0,
                np.stack([object_key(key) for key in keys]),
                endpoint.region.slots(*range(len(keys))),
                0,
            )
        ]
    )
    if direction == "publish":
        for slot, key in zip(slots, keys):
            endpoint.write(slot, payload[key])
        attempt = endpoint.backend.publish(view, None)
    else:
        published(store, payload)
        attempt = endpoint.backend.fetch(view, None)
    outcome = endpoint.wait_outcome(attempt)
    assert outcome.fault is None and masks(outcome) == [[True] * len(keys)]
    method = "put" if direction == "publish" else "get"
    assert [len(call.rows) for call in endpoint.driver.calls_to(method)] == expected
    if direction == "publish":
        assert {key: store.committed_payload(store_key(key)) for key in set(keys)} == payload
    else:
        assert [endpoint.read(slot) for slot in slots] == [payload[key] for key in keys]
    endpoint.wait_quiesced(attempt)
    endpoint.close()


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


def test_blob_config_defaults_assume_nothing_about_the_driver() -> None:
    assert BlobConfig() == BlobConfig(
        key_prefix="trtllm-kv", workers=1, rows_per_batch=None, bytes_per_batch=None
    )


def test_blob_config_refuses_bad_values() -> None:
    for prefix in ("", "a/b", "a b"):
        with pytest.raises(ValueError, match="prefix"):
            BlobConfig(key_prefix=prefix)
    with pytest.raises(ValueError, match="at least 1"):
        BlobConfig(rows_per_batch=0)
    for workers in (None, 0, True):
        with pytest.raises(ValueError, match="workers must be an int of at least 1"):
            BlobConfig(workers=workers)
    with pytest.raises(ValueError, match="max_queued_batches must be an int of at least 1"):
        BlobConfig(max_queued_batches=0)
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
        # B entered the driver after A; its fault is recorded once A's is the only access open.
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


def three_rows(store: MemoryStore, fetched: bool) -> Endpoint:
    """An endpoint whose rows A, B and C each go in a call of their own, B and C waiting behind A,
    with their bytes in the store when ``fetched`` and in the slots otherwise."""
    data = {name: name.encode() * SLOT_BYTES for name in "ABC"}
    if fetched:
        published(store, data)
    endpoint = Endpoint(MemoryDriver(store), list(data), rows_per_batch=1)
    if not fetched:
        for name, row in data.items():
            endpoint.write(name, row)
    return endpoint


def test_a_row_fault_stops_a_fetch(store: MemoryStore) -> None:
    reader = three_rows(store, fetched=True)
    reader.driver.answer("poll", (faulted("row lost"),))  # the worker's look at A's call
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    outcome = reader.wait_outcome(attempt)
    assert outcome.fault == "group 0 row 0: row lost"
    assert masks(outcome) == [[False, False, False]]  # no use for rows past the first gap
    assert store_calls(reader) == ["get"]
    reader.wait_quiesced(attempt)
    reader.close()


def test_a_row_fault_leaves_the_rest_of_a_publish_to_go_on(store: MemoryStore) -> None:
    writer = three_rows(store, fetched=False)
    writer.driver.answer("poll", (faulted("row lost"),))  # the worker's look at A's call
    attempt = writer.backend.publish(writer.view("A", "B", "C"), None)
    outcome = writer.wait_outcome(attempt)
    assert outcome.fault == "group 0 row 0: row lost"
    assert masks(outcome) == [[False, True, True]]  # a publish's rows stand alone
    assert store_calls(writer) == ["put"] * 3
    writer.wait_quiesced(attempt)
    writer.close()


def test_a_refused_put_stops_its_publish_but_not_the_backend(store: MemoryStore) -> None:
    writer = three_rows(store, fetched=False)
    writer.driver.fail("put", DriverRejected("busy"))
    attempt = writer.backend.publish(writer.view("A", "B", "C"), None)
    outcome = writer.wait_outcome(attempt)
    assert outcome.fault == "group 0 row 0: refused: busy"
    assert masks(outcome) == [[False, False, False]]  # the call failed as a whole
    assert store_calls(writer) == ["put"]
    writer.wait_quiesced(attempt)
    # A refusal breaks no protocol, so the backend takes the next submission.
    again = writer.backend.publish(writer.view("A", "B", "C"), None)
    assert masks(writer.wait_outcome(again)) == [[True, True, True]]
    writer.wait_quiesced(again)
    writer.close()


@pytest.mark.parametrize("first_half", ["results_first", "access_end_first"])
def test_results_and_access_end_are_each_recorded_once_per_batch(
    store: MemoryStore, first_half: str
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_a, op_b = reader.op("A"), reader.op("B")
    # A lands one half only, so every progress round sees that half again.
    if first_half == "results_first":
        driver.complete(op_a)
    else:
        driver.transfer(op_a)
        driver.end_access(op_a)
    reader.driver.wait_for(lambda: reader.driver.progress_polls(op_a) >= 5, "five progress rounds")
    # Recording A's half on every round would already have concluded the attempt (results_first)
    # or quiesced it (access_end_first).
    assert attempt.poll() is None  # B has no result yet
    assert not attempt.quiesced()  # B's access is open
    driver.complete(op_b)
    driver.end_access(op_b)
    if first_half == "results_first":
        driver.end_access(op_a)
    else:
        driver.publish_results(op_a)
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
    reader.driver.wait_for(lambda: reader.driver.progress_polls(op_b) >= 3, "three progress rounds")
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
    no_cancel = replace(DEFAULT_CAPABILITIES, cancel=False)
    driver = MemoryDriver(store, controlled=True, capabilities=no_cancel)
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
    assert reader.driver.calls_to("cancel") == []  # no cancel declared: nothing to forward
    reader.close()


# Driver failures.


def unbuildable_rows(*_) -> list[RowIO]:
    raise ValueError("x")


@pytest.mark.parametrize("raised", ["driver_rejected", "while_building_rows"])
def test_a_refusal_faults_the_batch_and_ends_its_access(
    store: MemoryStore, raised: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    if raised == "driver_rejected":
        writer.driver.fail("put", DriverRejected("x"))
    else:  # before the driver call
        monkeypatch.setattr(backend_module.BlobBackend, "_row_ios", unbuildable_rows)
    attempt = writer.backend.publish(writer.view("A"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False]] and outcome.fault == "group 0 row 0: refused: x"
    writer.wait_quiesced(attempt)  # nothing escaped
    assert writer.backend.stats()["batches_unprovable"] == 0
    writer.close()


@keeps_backend_threads
def test_any_other_raise_from_the_driver_holds_the_batch_and_refuses_later_work(
    store: MemoryStore,
) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    writer.driver.fail("put", RuntimeError("x"))
    attempt = writer.backend.publish(writer.view("A"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False]]
    assert outcome.fault == "group 0 row 0: the store driver raised RuntimeError('x')"
    assert not attempt.quiesced()  # the rows may be accessed for good
    assert writer.backend.stats()["batches_unprovable"] == 1
    with pytest.raises(SubmissionRejected, match="broke its protocol: the store driver raised"):
        writer.backend.publish(writer.view("A"), None)
    assert attempt.label in writer.backend.shutdown(0)
    with pytest.raises(RuntimeError, match="region:0"):
        writer.binding.release_hold()
    assert not writer.hold.released


@keeps_backend_threads
def test_a_protocol_violation_leaves_an_attempt_with_nothing_queued_alone() -> None:
    store = KeyGatedStore(gets=True)
    published(store, {"X": b"xxxxxxxx"})
    endpoint = Endpoint(MemoryDriver(store), ["X", "Y"], workers=2)
    endpoint.write("Y", b"yyyyyyyy")
    in_driver = endpoint.backend.fetch(endpoint.view("X"), None)
    try:
        store.wait_for_waiting(1)  # its only batch is inside the store
        endpoint.driver.fail("put", RuntimeError("lost"))
        broken = endpoint.backend.publish(endpoint.view("Y"), None)
        assert "the store driver raised" in endpoint.wait_outcome(broken).fault
    finally:
        store.gate.set()
    outcome = endpoint.wait_outcome(in_driver)
    assert outcome.fault is None and masks(outcome) == [[True]]
    endpoint.wait_quiesced(in_driver)
    assert endpoint.backend.shutdown(0) == frozenset({broken.label, "region:0"})


@keeps_backend_threads
def test_a_protocol_violation_drops_the_batches_other_attempts_have_queued(
    store: MemoryStore,
) -> None:
    writer = Endpoint(MemoryDriver(store), ["A", "B", "C"], rows_per_batch=1)
    for name in "ABC":
        writer.write(name, name.encode() * SLOT_BYTES)
    gate = writer.driver.hold("put")
    writer.driver.fail("put", RuntimeError("lost"))
    broken = writer.backend.publish(writer.view("A"), None)
    writer.driver.wait_for(lambda: writer.driver.calls_to("put"), "A's put")
    queued = writer.backend.publish(writer.view("B", "C"), None)  # behind the only worker
    gate.set()
    outcome = writer.wait_outcome(queued)
    assert outcome.fault == "the store driver broke its protocol: the store driver raised " + repr(
        RuntimeError("lost")
    )
    assert masks(outcome) == [[False, False]]
    writer.wait_quiesced(queued)  # none of its batches started
    assert store_calls(writer) == ["put"]
    assert writer.backend.shutdown(0) == frozenset({broken.label, "region:0"})


class ShortPollDriver(MemoryDriver):
    """Once ``short`` is set, answers the next poll of a multi-row operation one result short."""

    short = False

    def poll(self, op):
        results = super().poll(op)
        if self.short and len(results) > 1:
            self.short = False
            return results[:-1]
        return results


@keeps_backend_threads
def test_a_poll_with_the_wrong_row_count_faults_the_batch_and_keeps_it_held(
    store: MemoryStore,
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = ShortPollDriver(store)
    driver.short = True  # the worker's look at the operation
    reader = Endpoint(driver, ["A", "B", "C"])
    reader.write("C", b"cccccccc")
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[False, False]]
    assert "gave 1 results for 2 rows" in outcome.fault
    # Later answers would have the right length and say access ended; after the violation the
    # batch is unprovable instead: counted, never polled again, its slots held.
    assert reader.backend.stats()["batches_unprovable"] == 1
    time.sleep(PROMPT_S)  # a watched batch would be polled several times by now
    assert reader.driver.progress_polls(reader.op("A")) == 0
    assert not attempt.quiesced()
    # The worker survived the bad answer, and the backend refuses all later work, naming why.
    assert all(thread.is_alive() for thread in reader.threads)
    with pytest.raises(SubmissionRejected, match="broke its protocol: .* gave 1 results for 2"):
        reader.backend.publish(reader.view("C"), None)
    assert store_calls(reader) == ["get"]
    assert attempt.label in reader.backend.shutdown(0)


def test_a_wrong_row_count_after_the_access_end_was_recorded_leaves_the_batch_provable(
    store: MemoryStore,
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = ShortPollDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op = reader.op("A")
    driver.transfer(op)
    driver.end_access(op)
    wait_until(lambda: reader.backend.stats()["batches_access_open"] == 0, "the access end")
    driver.short = True  # the next poll, now that the access end is recorded
    driver.publish_results(op)
    outcome = reader.wait_outcome(attempt)
    assert "gave 1 results for 2 rows" in outcome.fault
    assert attempt.quiesced()
    assert reader.backend.stats()["batches_unprovable"] == 0
    with pytest.raises(SubmissionRejected, match="broke its protocol"):  # a violation all the same
        reader.backend.fetch(reader.view("A"), None)
    reader.close()


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


INVALID_SUBMISSIONS: dict[str, Callable[[Endpoint], tuple[CpuView, object]]] = {
    "hint": lambda e: (e.view("A"), "peer-0"),
    "no_keys": lambda e: (_view_with(e, object_keys=None, addresses=None, region_index=None), None),
    "key_dtype": lambda e: (_view_with(e, object_keys=np.zeros((1, 4), dtype=np.int32)), None),
    "key_width": lambda e: (_view_with(e, object_keys=np.zeros((1, 0), dtype=np.uint8)), None),
    "address_dtype": lambda e: (_view_with(e, addresses=e.region.slots(0).astype(np.uint64)), None),
    "len_disagrees_with_arrays": lambda e: (_view_with(e, rows=2), None),
    "past_the_region": lambda e: (_view_with(e, addresses=e.region.slots(2)), None),
    "before_the_region": lambda e: (_view_with(e, addresses=e.region.slots(-1)), None),
    "misaligned": lambda e: (_view_with(e, addresses=e.region.slots(0) + 1), None),
    "unknown_region_index": lambda e: (_view_with(e, region_index=1), None),
    "negative_region_index": lambda e: (_view_with(e, region_index=-1), None),
    "same_slot_in_a_group": lambda e: (
        CpuView([Rows(0, np.stack([object_key("A"), object_key("B")]), e.region.slots(0, 0), 0)]),
        None,
    ),
    "same_slot_across_groups": lambda e: (CpuView([e.rows(["A"], 0), e.rows(["A"], 1)]), None),
}


@pytest.mark.parametrize("case", INVALID_SUBMISSIONS)
def test_invalid_submissions_are_rejected_before_the_driver(store: MemoryStore, case: str) -> None:
    reader = Endpoint(MemoryDriver(store), ["A", "B"])
    view, hint = INVALID_SUBMISSIONS[case](reader)
    with pytest.raises(InvalidSubmission):
        reader.backend.fetch(view, hint)
    assert store_calls(reader) == []
    assert reader.backend.stats()["attempts_active"] == 0
    reader.close()


def test_a_fetch_hint_reaches_every_get_of_a_driver_that_reads_hints(store: MemoryStore) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    driver = MemoryDriver(store, capabilities=replace(DEFAULT_CAPABILITIES, hints=True))
    reader = Endpoint(driver, ["A", "B", "C"], rows_per_batch=2)
    source = ("peer", 0)
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), source)
    assert masks(reader.wait_outcome(attempt)) == [[True, True, True]]
    assert len(reader.driver.calls_to("get")) == 2
    assert driver.hints_seen == [source, source] and driver.hints_seen[0] is source
    with pytest.raises(InvalidSubmission, match="None destination"):  # a put takes no hint
        reader.backend.publish(reader.view("A"), source)
    reader.wait_quiesced(attempt)
    unhinted = reader.backend.fetch(reader.view("A"), None)
    assert masks(reader.wait_outcome(unhinted)) == [[True]]
    assert driver.hints_seen == [source, source]  # a None hint is never passed on
    reader.wait_quiesced(unhinted)
    reader.close()


def test_a_publish_may_read_one_slot_twice(store: MemoryStore) -> None:
    writer = Endpoint(MemoryDriver(store), ["A", "B"])
    writer.write("A", b"aaaaaaaa")
    keys = np.stack([object_key("A"), object_key("B")])
    view = CpuView([Rows(0, keys, writer.region.slots(0, 0), 0)])
    attempt = writer.backend.publish(view, None)
    assert masks(writer.wait_outcome(attempt)) == [[True, True]]
    assert store.committed_payload(store_key("B")) == b"aaaaaaaa"
    writer.wait_quiesced(attempt)
    writer.close()


def test_publish_is_not_starved_by_fetches() -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("F0")])
    # More fetches than the two publishes interleave with, so some are still queued after both.
    fetched = [f"F{i}" for i in range(3 * FETCH_BATCHES_PER_PUBLISH_BATCH)]
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
    # The blocker was taken before any publish waited, so it does not count toward one's turn.
    fetches_then_a_publish = ["get"] * FETCH_BATCHES_PER_PUBLISH_BATCH + ["put"]
    expected = ["get", *fetches_then_a_publish * 2]
    assert store_calls(endpoint)[: len(expected)] == expected
    endpoint.close()


def test_only_fetches_taken_while_a_publish_waits_count_toward_its_turn() -> None:
    fetched = [f"F{i}" for i in range(FETCH_BATCHES_PER_PUBLISH_BATCH + 2)]
    held = fetched[FETCH_BATCHES_PER_PUBLISH_BATCH]
    store = KeyGatedStore(gets=True, keys=[store_key(held)])
    published(store, {name: bytes(SLOT_BYTES) for name in fetched})
    endpoint = Endpoint(MemoryDriver(store), [*fetched, "P"])
    endpoint.write("P", b"pppppppp")
    # The only worker runs the first fetches with no publish waiting, then holds one in the store.
    attempts = [endpoint.backend.fetch(endpoint.view(name), None) for name in fetched[:-1]]
    try:
        store.wait_for_waiting(1)
        attempts.append(endpoint.backend.publish(endpoint.view("P"), None))
        attempts.append(endpoint.backend.fetch(endpoint.view(fetched[-1]), None))
    finally:
        store.gate.set()
    for attempt in attempts:
        assert masks(endpoint.wait_outcome(attempt)) == [[True]]
        endpoint.wait_quiesced(attempt)
    assert store_calls(endpoint) == ["get"] * len(fetched) + ["put"]  # the late fetch goes first
    endpoint.close()


def test_publishes_leave_one_worker_to_fetches() -> None:
    store = KeyGatedStore(commits=True, keys=[store_key("P1"), store_key("P2")])
    published(store, {"F": b"ffffffff"})
    endpoint = Endpoint(MemoryDriver(store), ["P1", "P2", "F"], workers=2, rows_per_batch=1)
    endpoint.write("P1", b"pppppppp")
    endpoint.write("P2", b"qqqqqqqq")
    publish = endpoint.backend.publish(endpoint.view("P1", "P2"), None)
    try:
        store.wait_for_waiting(1)
        time.sleep(PROMPT_S)  # the second publish batch would have taken the other worker by now
        assert endpoint.driver.in_flight == 1
        fetch = endpoint.backend.fetch(endpoint.view("F"), None)
        assert masks(endpoint.wait_outcome(fetch)) == [[True]]
        assert store_calls(endpoint) == ["put", "get"]
    finally:
        store.gate.set()
    assert masks(endpoint.wait_outcome(publish)) == [[True, True]]
    for attempt in (publish, fetch):
        endpoint.wait_quiesced(attempt)
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


@keeps_backend_threads
def test_shutdown_returns_once_every_open_attempt_is_pinned_by_an_unprovable_call(
    store: MemoryStore,
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B", "C"], rows_per_batch=1)
    pinned = reader.backend.fetch(reader.view("A", "B"), None)
    other = reader.backend.fetch(reader.view("C"), None)
    op_a, op_b, op_c = reader.op("A"), reader.op("B"), reader.op("C")
    driver.mark_unprovable(op_a)  # its results still pending: the batch stays watched
    wait_until(lambda: reader.backend.stats()["batches_unprovable"] == 1, "the count")
    # C's attempt can still quiesce, so a shutdown waits for it; B's watched batch cannot unpin
    # its attempt, so it does not.
    labels: list[frozenset[str]] = []
    shutter = threading.Thread(target=lambda: labels.append(reader.backend.shutdown(WAIT_S)))
    shutter.start()
    time.sleep(PROMPT_S)
    assert shutter.is_alive()
    driver.complete(op_c)
    driver.end_access(op_c)
    shutter.join(PROMPT_S)
    assert labels == [frozenset({pinned.label, "region:0"})]
    assert other.quiesced() and not op_b.access_ended
    started = time.monotonic()
    assert reader.backend.shutdown(WAIT_S) == frozenset({pinned.label, "region:0"})
    assert time.monotonic() - started < PROMPT_S
    # An unprovable batch stays held whatever the driver says of it later, so the answer stands.
    polls = reader.driver.progress_polls(op_a)
    for op in (op_a, op_b):
        driver.complete(op)
        driver.end_access(op)
    reader.driver.wait_for(lambda: reader.driver.progress_polls(op_a) > polls, "A's results")
    reader.wake.wait_for(lambda: reader.backend.stats()["batches_access_open"] == 1, "B's end")
    assert not pinned.quiesced()
    assert reader.backend.shutdown(0) == frozenset({pinned.label, "region:0"})


@keeps_backend_threads
def test_shutdown_returns_once_a_call_in_the_driver_turns_unprovable(store: MemoryStore) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    writer.write("A", b"aaaaaaaa")
    gate = writer.driver.hold("put")
    writer.driver.fail("put", RuntimeError("lost"))  # once the gate opens
    attempt = writer.backend.publish(writer.view("A"), None)
    writer.driver.wait_for(lambda: writer.driver.calls_to("put"), "the put")
    labels: list[frozenset[str]] = []
    shutter = threading.Thread(target=lambda: labels.append(writer.backend.shutdown(WAIT_S)))
    shutter.start()
    time.sleep(PROMPT_S)
    assert shutter.is_alive()  # the put may still end its access
    gate.set()
    shutter.join(PROMPT_S)
    assert labels == [frozenset({attempt.label, "region:0"})]


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


@pytest.mark.parametrize("before", ["idle", "after_round_trip", "after_refused_put"])
def test_clean_shutdown_returns_well_before_its_budget_and_rejects_later_submissions(
    store: MemoryStore, before: str
) -> None:
    endpoint = Endpoint(MemoryDriver(store), ["A"])
    endpoint.write("A", b"aaaaaaaa")
    if before == "after_refused_put":
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


def test_shutdown_waits_out_an_unbounded_budget_without_raising() -> None:
    store = KeyGatedStore(gets=True)
    published(store, {"A": b"aaaaaaaa"})
    reader = Endpoint(MemoryDriver(store), ["A"])
    reader.backend.fetch(reader.view("A"), None)
    labels: list[frozenset[str]] = []
    # A daemon thread, so a shutdown that never returns fails the wait below instead of the run.
    shutter = threading.Thread(
        target=lambda: labels.append(reader.backend.shutdown(math.inf)), daemon=True
    )
    try:
        store.wait_for_waiting(1)
        shutter.start()
        time.sleep(PROMPT_S)
        assert shutter.is_alive()  # waiting, since the batch is still in the store
    finally:
        store.gate.set()
    shutter.join(WAIT_S)
    assert labels == [frozenset()]
    reader.close()


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
    """On each wake, reads the backend from a helper thread as a lease holder would. A wake made
    under the backend's lock would leave the helper blocked past its bound."""

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


HOST_ONLY = replace(DEFAULT_CAPABILITIES, device_memory=False)


@pytest.mark.parametrize("device_memory", [False, True], ids=["host_only", "device_memory"])
def test_region_binding_registers_the_regions_with_slots_the_driver_can_access(
    store: MemoryStore, device_memory: bool
) -> None:
    driver = MemoryDriver(store, capabilities=replace(HOST_ONLY, device_memory=device_memory))
    regions = (
        CpuRegion(SLOT_BYTES, 0),
        CpuRegion(SLOT_BYTES, 1),
        CpuRegion(SLOT_BYTES, 1, memory_kind=MemoryKind.DEVICE),
    )
    binding = RegionBinding(regions, driver, FakeHold())
    binding.register()
    kinds = [registration.layout.memory_kind for registration in driver.live_registrations()]
    if device_memory:
        assert binding.live_labels() == frozenset({"region:1", "region:2"})
        assert kinds == [MemoryKind.HOST, MemoryKind.DEVICE]
    else:
        assert binding.live_labels() == frozenset({"region:1"}) and kinds == [MemoryKind.HOST]
    assert binding.unregister() and driver.live_registrations() == []


def test_a_binding_with_no_region_to_register_is_rejected_and_lets_the_hold_go(
    store: MemoryStore,
) -> None:
    driver, hold = MemoryDriver(store, capabilities=HOST_ONLY), FakeHold()
    regions = (CpuRegion(SLOT_BYTES, 0), CpuRegion(SLOT_BYTES, 1, memory_kind=MemoryKind.DEVICE))
    with pytest.raises(ConstructionRejected, match="no region"):
        RegionBinding(regions, driver, hold).register()
    assert hold.released and driver.live_registrations() == []


@pytest.mark.parametrize(
    "failure", ["register_rejected", "register_overlaps", "register_escaped", "rollback_failed"]
)
def test_a_failed_registration_releases_the_hold_only_when_nothing_escaped(failure: str) -> None:
    regions = (CpuRegion(SLOT_BYTES, 2), CpuRegion(SLOT_BYTES, 2))
    driver = (RollbackFailsDriver if failure == "rollback_failed" else MemoryDriver)(MemoryStore())
    if failure == "register_overlaps":  # the driver refuses the second region: DriverRejected
        regions = (regions[0], regions[0])
    elif failure != "rollback_failed":
        driver.fail_next_register("no memlock", escaped=failure == "register_escaped")
    hold = FakeHold()
    binding = RegionBinding(regions, driver, hold)
    with pytest.raises(RuntimeError) as raised:
        binding.register()
    if failure in ("register_rejected", "register_overlaps"):  # rolled back
        assert isinstance(raised.value, ConstructionRejected)
        assert hold.released and driver.live_registrations() == []
        assert binding.live_labels() == frozenset()
        return
    assert not isinstance(raised.value, ConstructionRejected)
    assert not hold.released and driver.live_registrations()
    with pytest.raises(RuntimeError, match="region:0"):
        binding.release_hold()  # a registration may be live: the hold stays
    assert not hold.released


def test_rows_in_a_region_the_backend_did_not_register_are_rejected(store: MemoryStore) -> None:
    device = CpuRegion(SLOT_BYTES, 1, memory_kind=MemoryKind.DEVICE)
    reader = Endpoint(MemoryDriver(store, capabilities=HOST_ONLY), ["A"], others=[device])
    assert reader.binding.live_labels() == frozenset({"region:0"})
    view = CpuView([Rows(0, object_key("A")[None, :], device.slots(0), 1)])
    with pytest.raises(InvalidSubmission, match="region 1 is not registered"):
        reader.backend.fetch(view, None)
    with pytest.raises(ValueError, match="region 1 is not registered"):
        reader.backend.batch_size(1)
    assert store_calls(reader) == []
    reader.close()


# Regions registered together.


def registration_calls(endpoint: Endpoint) -> list[str]:
    calls = [call.method for call in endpoint.driver.calls]
    return [method for method in calls if method.startswith(("register", "unregister"))]


def test_a_together_driver_gets_every_region_in_one_batch_and_gives_them_back_in_one(
    store: MemoryStore,
) -> None:
    driver = TogetherDriver(store)
    others = [CpuRegion(SLOT_BYTES, 0), CpuRegion(SLOT_BYTES, 2)]
    endpoint = Endpoint(driver, ["A", "B"], others=others)
    assert registration_calls(endpoint) == ["register_regions"]
    assert endpoint.binding.live_labels() == frozenset({"region:0", "region:2"})
    (batch,) = driver.live_batches()
    addresses = [registration.span.address for registration in batch.registrations]
    assert addresses == [endpoint.region.address, others[1].address]  # in region order
    published(store, {"A": b"aaaaaaaa"})
    outcome = endpoint.wait_outcome(endpoint.backend.fetch(endpoint.view("A", "B"), None))
    assert masks(outcome) == [[True, False]] and endpoint.read("A") == b"aaaaaaaa"
    endpoint.close()
    assert registration_calls(endpoint) == ["register_regions", "unregister_regions"]
    assert driver.live_batches() == [] and driver.live_registrations() == []


@pytest.mark.parametrize(
    "failure", ["rejected", "nothing_registered", "revocable", "revocation_fails", "no_handle"]
)
def test_a_failed_batch_registration_releases_the_hold_only_once_nothing_is_registered(
    failure: str,
) -> None:
    regions = (CpuRegion(SLOT_BYTES, 2), CpuRegion(SLOT_BYTES, 2))
    driver = TogetherDriver(MemoryStore())
    spy = TogetherDriverSpy(driver)
    if failure == "rejected":
        spy.fail("register_regions", DriverRejected("no memlock"))
    elif failure == "no_handle":
        spy.fail("register_regions", RuntimeError("the client crashed"))
    else:
        driver.fail_next_register_regions("no memlock", escaped=failure != "nothing_registered")
        if failure == "revocation_fails":
            driver.fail_next_unregister_regions("device busy")
    hold = FakeHold()
    binding = RegionBinding(regions, spy, hold)
    with pytest.raises(RuntimeError) as raised:
        binding.register()
    revocations = len(spy.calls_to("unregister_regions"))
    if failure in ("rejected", "nothing_registered", "revocable"):
        assert isinstance(raised.value, ConstructionRejected)
        assert hold.released and driver.live_registrations() == []
        assert binding.live_labels() == frozenset()
        assert revocations == (1 if failure == "revocable" else 0)
        return
    # Something may be registered: the regions keep their owner and the hold stays.
    assert not isinstance(raised.value, ConstructionRejected)
    assert not hold.released
    assert binding.live_labels() == frozenset({"region:0", "region:1"})
    with pytest.raises(RuntimeError, match="region:0"):
        binding.release_hold()
    if failure == "revocation_fails":  # a handle: a later revocation succeeds and frees the hold
        assert revocations == 1 and driver.live_registrations()
        assert binding.unregister() and driver.live_registrations() == []
        binding.release_hold()
        assert hold.released
    else:  # nothing to revoke by: the hold is kept for good
        assert revocations == 0 and not binding.unregister()
        assert not hold.released


def test_a_busy_region_keeps_the_whole_batch_registered(store: MemoryStore) -> None:
    driver = TogetherDriver(store, controlled=True)
    endpoint = Endpoint(driver, ["A"], others=[CpuRegion(SLOT_BYTES, 1)])
    endpoint.write("A", b"aaaaaaaa")
    attempt = endpoint.backend.publish(endpoint.view("A"), None)
    op = endpoint.op("A")
    assert not endpoint.binding.unregister()  # refused by the driver: the put may still read A
    assert endpoint.binding.live_labels() == frozenset({"region:0", "region:1"})
    assert driver.live_batches()
    driver.transfer(op)
    driver.publish_results(op)
    driver.end_access(op)
    endpoint.wait_quiesced(attempt)
    endpoint.close()
    assert driver.live_batches() == []


# Hints.


@pytest.mark.parametrize("case", ["driver_reads_no_hints", "publish"])
def test_a_hint_is_refused_before_the_driver_unless_a_fetch_can_use_it(
    store: MemoryStore, case: str
) -> None:
    reads = replace(DEFAULT_CAPABILITIES, hints=case == "publish")
    endpoint = Endpoint(MemoryDriver(store, capabilities=reads), ["A"])
    submit = endpoint.backend.publish if case == "publish" else endpoint.backend.fetch
    with pytest.raises(InvalidSubmission, match="hint|destination"):
        submit(endpoint.view("A"), "peer-0")
    assert store_calls(endpoint) == [] and endpoint.backend.stats()["attempts_active"] == 0
    endpoint.close()


def test_a_fetch_hint_reaches_every_batch_and_a_fetch_without_one_stays_hint_free(
    store: MemoryStore,
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    driver = MemoryDriver(store, capabilities=replace(DEFAULT_CAPABILITIES, hints=True))
    reader = Endpoint(driver, ["A", "B", "C"], rows_per_batch=1)
    hint = ("peer-0", 3)
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), hint)
    assert masks(reader.wait_outcome(attempt)) == [[True, True, True]]
    reader.wait_quiesced(attempt)
    assert len(driver.hints_seen) == 3 and all(seen is hint for seen in driver.hints_seen)
    plain = reader.backend.fetch(reader.view("A"), None)
    assert masks(reader.wait_outcome(plain)) == [[True]]
    reader.wait_quiesced(plain)
    assert len(driver.hints_seen) == 3  # the hint-free fetch carried none
    reader.close()


# Cancel forwarding.


def test_cancel_is_forwarded_once_to_each_started_batch_without_a_result(
    store: MemoryStore,
) -> None:
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B", "C"], rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    ops = [reader.op(name) for name in "ABC"]  # three handles back, no result yet
    driver.complete(ops[0])
    driver.end_access(ops[0])
    wait_until(lambda: reader.backend.stats()["batches_access_open"] == 2, "A to be recorded")
    attempt.cancel()
    # B's cancel goes from this thread; C's may go from its worker, whose call was still out.
    reader.driver.wait_for(lambda: len(reader.driver.calls_to("cancel")) == 2, "two cancels")
    forwarded = [call.op for call in reader.driver.calls_to("cancel")]
    assert all(any(op is seen for seen in forwarded) for op in ops[1:])  # not A: it is in
    outcome = reader.wait_outcome(attempt)  # the driver's cancel concludes B and C CANCELLED
    reader.wait_quiesced(attempt)
    assert masks(outcome) == [[True, False, False]] and outcome.fault is None
    attempt.cancel()  # after the outcome: nothing more to forward
    assert len(reader.driver.calls_to("cancel")) == 2
    reader.close()


def test_a_cancel_that_arrives_while_the_call_is_out_is_forwarded_when_the_handle_returns(
    store: MemoryStore,
) -> None:
    published(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    gate = reader.driver.hold("get")
    attempt = reader.backend.fetch(reader.view("A"), None)
    reader.driver.wait_for(lambda: bool(reader.driver.calls_to("get")), "the get call")
    attempt.cancel()  # the worker is inside the call: no handle to cancel yet
    assert reader.driver.calls_to("cancel") == [] and attempt.poll() is None
    gate.set()
    op = reader.op("A")
    reader.driver.wait_for(lambda: len(reader.driver.calls_to("cancel", op)) == 1, "the cancel")
    outcome = reader.wait_outcome(attempt)
    reader.wait_quiesced(attempt)
    assert masks(outcome) == [[False]] and outcome.fault is None
    assert len(reader.driver.calls_to("cancel")) == 1
    reader.close()


@pytest.mark.parametrize("driver_kind", ["declares_no_cancel", "ignores_the_cancel"])
def test_cancel_forwarding_follows_the_capability_and_is_no_access_end_evidence(
    store: MemoryStore, driver_kind: str
) -> None:
    published(store, {"A": b"aaaaaaaa"})
    ignores = driver_kind == "ignores_the_cancel"
    driver = MemoryDriver(
        store, controlled=True, capabilities=replace(DEFAULT_CAPABILITIES, cancel=ignores)
    )
    driver.set_cancel_effective(False)
    reader = Endpoint(driver, ["A"])
    attempt = reader.backend.fetch(reader.view("A"), None)
    op = reader.op("A")
    attempt.cancel()
    if ignores:  # forwarded from this thread, or from the worker if its call was still out
        reader.driver.wait_for(lambda: bool(reader.driver.calls_to("cancel", op)), "the cancel")
    assert attempt.poll() is None and not attempt.quiesced()  # the row is still the store's
    driver.complete(op)
    driver.end_access(op)
    outcome = reader.wait_outcome(attempt)
    reader.wait_quiesced(attempt)
    assert masks(outcome) == [[True]] and outcome.fault is None
    assert len(reader.driver.calls_to("cancel", op)) == (1 if ignores else 0)
    reader.close()


def test_a_cancel_during_the_handoff_between_the_call_and_its_observation_is_forwarded(
    store: MemoryStore,
) -> None:
    """The handle is back and the batch is not watched yet: the worker is about to observe it."""
    published(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    gate = reader.driver.hold("poll")  # pauses the worker right after the handle came back
    attempt = reader.backend.fetch(reader.view("A"), None)
    reader.driver.wait_for(lambda: bool(reader.driver.calls_to("poll")), "the worker's poll")
    op = reader.op("A")
    attempt.cancel()
    assert len(reader.driver.calls_to("cancel", op)) == 1  # forwarded from this thread, at once
    gate.set()
    outcome = reader.wait_outcome(attempt)
    reader.wait_quiesced(attempt)
    assert masks(outcome) == [[False]] and outcome.fault is None
    assert len(reader.driver.calls_to("cancel")) == 1
    reader.close()


# The queue bound.


def test_submissions_are_refused_while_the_queue_is_full_and_cancelled_work_leaves_it() -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("A")])
    published(store, {name: name.encode() * SLOT_BYTES for name in "ABCDEF"})
    endpoint = Endpoint(
        MemoryDriver(store), list("ABCDEF"), workers=1, rows_per_batch=1, max_queued_batches=2
    )
    first = endpoint.backend.fetch(endpoint.view("A"), None)
    try:
        store.wait_for_waiting(1)  # the only worker is inside the store with A
        queued = endpoint.backend.fetch(endpoint.view("B", "C"), None)  # fills the queue
        assert endpoint.backend.stats()["batches_queued"] == 2
        with pytest.raises(SubmissionRejected, match="max_queued_batches"):
            endpoint.backend.fetch(endpoint.view("D"), None)
        assert endpoint.backend.stats()["attempts_active"] == 2
        assert store_calls(endpoint) == ["get"]
        queued.cancel()  # its batches leave the count at once; the worker is still blocked
        assert endpoint.backend.stats()["batches_queued"] == 0
        assert endpoint.backend.stats()["batches_in_queues"] == 0  # and the queues themselves
        later = endpoint.backend.fetch(endpoint.view("D", "E"), None)
        with pytest.raises(SubmissionRejected):
            endpoint.backend.fetch(endpoint.view("F"), None)
    finally:
        store.gate.set()
    for attempt in (first, queued, later):
        endpoint.wait_outcome(attempt)
        endpoint.wait_quiesced(attempt)
    assert masks(queued.poll()) == [[False, False]] and queued.poll().fault is None
    assert masks(later.poll()) == [[True, True]] and endpoint.read("E") == b"EEEEEEEE"
    endpoint.close()


def test_repeated_submit_and_cancel_leaves_nothing_in_the_queues() -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("A")])
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    endpoint = Endpoint(
        MemoryDriver(store), ["A", "B", "C"], workers=1, rows_per_batch=1, max_queued_batches=2
    )
    first = endpoint.backend.fetch(endpoint.view("A"), None)
    try:
        store.wait_for_waiting(1)  # the only worker is inside the store with A for the duration
        for _ in range(50):
            endpoint.backend.fetch(endpoint.view("B", "C"), None).cancel()
        assert endpoint.backend.stats()["batches_in_queues"] == 0
        last = endpoint.backend.fetch(endpoint.view("B", "C"), None)
        assert endpoint.backend.stats()["batches_in_queues"] == 2
    finally:
        store.gate.set()
    for attempt in (first, last):
        endpoint.wait_outcome(attempt)
        endpoint.wait_quiesced(attempt)
    assert masks(last.poll()) == [[True, True]]
    endpoint.close()


def test_the_first_fault_drops_the_attempts_queued_batches() -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("A")])
    published(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    store.corrupt(store_key("A"), 3)
    reader = Endpoint(MemoryDriver(store), ["A", "B", "C"], rows_per_batch=1)
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    try:
        store.wait_for_waiting(1)  # the worker is inside the store with A; B and C wait
        assert reader.backend.stats()["batches_in_queues"] == 2
    finally:
        store.gate.set()
    outcome = reader.wait_outcome(attempt)
    reader.wait_quiesced(attempt)
    assert "checksum" in outcome.fault and masks(outcome) == [[False, False, False]]
    assert reader.backend.stats()["batches_in_queues"] == 0
    assert store_calls(reader) == ["get"]  # B and C never reached the driver
    reader.close()


# A protocol break seen by the progress thread.


@in_a_child_process
def test_a_protocol_break_seen_by_the_progress_thread_concludes_the_queue_at_once(
    store: MemoryStore,
) -> None:
    """No worker is free, the only one being inside the driver; the queued work still concludes."""
    published(store, {"A": b"aaaaaaaa", "A2": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = ShortPollDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "A2", "B", "C", "D"], rows_per_batch=2)
    watched = reader.backend.fetch(reader.view("A", "A2"), None)  # one batch, handle back
    reader.op("A")
    gate = reader.driver.hold("get")
    blocked = reader.backend.fetch(reader.view("B"), None)  # the only worker, held in the call
    reader.driver.wait_for(lambda: len(reader.driver.calls_to("get")) == 2, "the held get")
    queued = reader.backend.fetch(reader.view("C", "D"), None)
    assert reader.backend.stats()["batches_in_queues"] == 1
    driver.short = True  # the progress thread's next poll of A comes back one result short
    outcome = reader.wait_outcome(queued)
    reader.wait_quiesced(queued)  # concluded while the worker is still held
    assert not gate.is_set() and reader.backend.stats()["batches_in_queues"] == 0
    assert masks(outcome) == [[False, False]] and "broke its protocol" in outcome.fault
    assert "gave 1 results for 2 rows" in reader.wait_outcome(watched).fault
    gate.set()
    op_b = reader.op("B")  # dispatched before the break: it runs to its own end
    driver.complete(op_b)
    driver.end_access(op_b)
    assert masks(reader.wait_outcome(blocked)) == [[True]]
    reader.wait_quiesced(blocked)
    labels = reader.backend.shutdown(0)
    assert watched.label in labels and queued.label not in labels and blocked.label not in labels


# Construction.


@pytest.mark.parametrize("failing", ["second_worker", "progress_thread"])
def test_a_thread_that_cannot_start_leaves_no_backend_thread_and_the_binding_to_its_owner(
    store: MemoryStore, failing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second worker, or the progress thread, cannot be started: the worker already running
    leaves, no backend thread survives, and the binding still holds the registrations and the
    hold for the builder to revoke and release."""
    region, hold = CpuRegion(SLOT_BYTES, 1), FakeHold()
    binding = RegionBinding((region,), MemoryDriver(store), hold)
    binding.register()
    failing_call = 2 if failing == "second_worker" else 3  # workers first, the progress thread last
    original_start = threading.Thread.start
    starts = {"count": 0}

    def start(thread: threading.Thread) -> None:
        starts["count"] += 1
        if starts["count"] == failing_call:
            raise RuntimeError("can't start new thread")
        original_start(thread)

    before = set(threading.enumerate())
    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", start)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            BlobBackend(binding, lambda: None, BlobConfig(workers=2))
    assert set(threading.enumerate()) == before  # the started worker left and was joined
    assert binding.live_labels() == frozenset({"region:0"}) and not hold.released
    assert binding.unregister()  # the builder revokes and releases, as after any failed build
    binding.release_hold()
    assert hold.released
