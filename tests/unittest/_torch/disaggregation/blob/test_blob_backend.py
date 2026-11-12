# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over the in-memory driver.

Covers the store's behaviour cases (a late write after a fault, results and access end in
either order, cancel and shutdown races) and the backend's own rules: one driver call per layer
group, a submission returns while its operations run, only regions the driver can access are
registered and rows elsewhere are refused, each operation's access end is counted once, a fault
waits for every call, every call reaches the driver as it is submitted and a stopped attempt's
calls are cancelled there once each, a call is queried only when its driver calls its on_change,
a driver that breaks its protocol gets no more work, and shutdown never calls the driver. Every
wait is bounded; a sleep of PROMPT_S only stands for "nothing happened meanwhile".
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
    SHUTDOWN_FAULT,
    BlobBackend,
    BlobConfig,
    ConstructionRejected,
    RegionBinding,
)
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverRejected,
    RowBatch,
    RowIO,
    Span,
    StoreError,
)
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    DEFAULT_CAPABILITIES,
    Commit,
    MemoryDriver,
    MemoryOp,
    MemoryStore,
)

# An exception that escapes a thread fails the test.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# A test that leaves a call whose access never ends keeps that backend's progress thread alive, as
# it would in production; the thread-leak check would report it.
keeps_backend_threads = pytest.mark.threadleak(enabled=False)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


def commit_objects(store: MemoryStore, rows: dict[str, bytes]) -> None:
    """Commits ``rows`` straight into ``store`` under the keys the backend derives."""
    for name, data in rows.items():
        assert commit_object(store, name, data) is Commit.STORED


def store_calls(endpoint: Endpoint) -> list[str]:
    return [call.method for call in endpoint.driver.calls if call.method in ("get", "put")]


ONE_ROW_PER_CALL = replace(DEFAULT_CAPABILITIES, max_rows_per_op=1)


def one_row_per_call(store: MemoryStore, **options) -> MemoryDriver:
    """A driver that takes one row per operation, so each row of a group is a call of its own."""
    return MemoryDriver(store, capabilities=ONE_ROW_PER_CALL, **options)


# Results.


@pytest.mark.parametrize("case", ["fetch_not_ready", "publish_declined"])
def test_not_ready_and_declined_rows_are_undelivered_without_a_fault(case: str) -> None:
    # A fetch finds the first object still being written (NOT_READY); a publish finds no room
    # after the first row (DECLINED).
    store = MemoryStore(capacity=1 if case == "publish_declined" else None)
    endpoint = Endpoint(one_row_per_call(store), ["A", "B", "C"])
    if case == "fetch_not_ready":
        commit_objects(store, {"B": b"bbbbbbbb", "C": b"cccccccc"})
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
    commit_objects(store, {"A": b"aaaaaaaa", "C": b"cccccccc", "E": b"eeeeeeee"})
    reader = Endpoint(MemoryDriver(store), ["A", "B", "C", "D", "E"])
    view = CpuView([reader.rows(["C", "D"], 2), reader.rows(["A"], 0), reader.rows(["E", "B"], 1)])
    attempt = reader.backend.fetch(view, None)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[True, False], [True], [True, False]]
    outcome.check_mask_shapes(view)
    assert [len(call.rows) for call in reader.driver.calls_to("get")] == [2, 1, 2]
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


@pytest.mark.parametrize("max_rows_per_op", [None, 4], ids=["no_limit", "driver_limit"])
def test_each_group_goes_in_one_call_split_only_by_the_driver_limit(
    store: MemoryStore, max_rows_per_op: int | None
) -> None:
    names = [f"R{i}" for i in range(12)]
    capabilities = replace(DEFAULT_CAPABILITIES, max_rows_per_op=max_rows_per_op)
    writer = Endpoint(MemoryDriver(store, capabilities=capabilities), names)
    view = CpuView([writer.rows(names[:10], 0), writer.rows(names[10:], 1)])
    attempt = writer.backend.publish(view, None)
    assert masks(writer.wait_outcome(attempt)) == [[True] * 10, [True] * 2]
    expected = [10, 2] if max_rows_per_op is None else [4, 4, 2, 2]
    assert [len(call.rows) for call in writer.driver.calls_to("put")] == expected
    limit = None if max_rows_per_op is None else (4, 4 * SLOT_BYTES)
    assert writer.backend.driver_call_limit(0) == limit
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


def test_blob_config_holds_only_the_key_prefix() -> None:
    assert BlobConfig() == BlobConfig(key_prefix="trtllm-kv")
    for prefix in ("", "a/b", "a b", None):
        with pytest.raises(ValueError, match="prefix"):
            BlobConfig(key_prefix=prefix)
    # Tuning such as workers belongs to the driver's own section.
    with pytest.raises(TypeError, match="workers"):
        BlobConfig(workers=2)


# Results, access end and late writes.


def test_late_write_after_fault_changes_nothing(store: MemoryStore) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = one_row_per_call(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D)
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
    commit_objects(store, {"A": b"aaaaaaaa"})
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
def test_a_later_call_fault_waits_for_earlier_calls_and_the_first_recorded_is_reported(
    first_row: str,
) -> None:
    store = KeyGatedStore(gets=True, keys=[store_key("A")])
    a = b"aaaaaaaa" if first_row == "ready" else b"aaa"  # the wrong size: a second fault
    commit_objects(store, {"A": a, "B": b"short"})
    driver = one_row_per_call(store, background=True)
    # B's fault stops the fetch and forwards a cancel; ignoring it lets A still reach the gate.
    driver.set_cancel_effective(False)
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    try:
        store.wait_for_calls_at_gate(1)
        # B's fault is recorded while A is still in the store.
        wait_until(lambda: reader.backend.stats()["calls_awaiting_results"] == 1, "B's fault")
        assert attempt.poll() is None
        assert attempt.quiesced() is False
    finally:
        store.gate.set()
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[first_row == "ready", False]]  # A's row is kept for reuse
    assert "row 1: object is 5 bytes" in outcome.fault and "3 bytes" not in outcome.fault
    reader.wait_quiesced(attempt)
    reader.close()


def three_rows(store: MemoryStore, objects_in_store: bool, **options) -> Endpoint:
    """An endpoint whose rows A, B and C each go in a call of their own, handed over in that order,
    with their bytes in the store when ``objects_in_store`` and in the slots otherwise. ``options``
    go to the ``MemoryDriver``."""
    data = {name: name.encode() * SLOT_BYTES for name in "ABC"}
    if objects_in_store:
        commit_objects(store, data)
    endpoint = Endpoint(one_row_per_call(store, **options), list(data))
    if not objects_in_store:
        for name, row in data.items():
            endpoint.write(name, row)
    return endpoint


def test_a_row_fault_stops_a_fetch(store: MemoryStore) -> None:
    reader = three_rows(store, objects_in_store=True, controlled=True)
    driver = reader.driver.inner
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    op_a, op_b, op_c = (reader.op(name) for name in "ABC")
    driver.fail(op_a, "row lost")
    # A prefix reuse has no use for the rows past the first gap, so B's and C's calls, which the
    # driver still holds, are cancelled.
    reader.driver.wait_for(lambda: len(reader.driver.calls_to("cancel")) == 2, "two cancels")
    cancelled = [c.op for c in reader.driver.calls_to("cancel")]
    assert cancelled[0] is op_b and cancelled[1] is op_c
    outcome = reader.wait_outcome(attempt)
    assert outcome.fault == "group 0 row 0: row lost"
    assert masks(outcome) == [[False, False, False]]
    assert store_calls(reader) == ["get"] * 3  # every call reached the driver as submitted
    driver.end_access(op_a)  # B and C ended theirs as they were cancelled
    reader.wait_quiesced(attempt)
    reader.close()


def test_a_row_fault_leaves_the_rest_of_a_publish_to_go_on(store: MemoryStore) -> None:
    writer = three_rows(store, objects_in_store=False, controlled=True)
    driver = writer.driver.inner
    attempt = writer.backend.publish(writer.view("A", "B", "C"), None)
    op_a, op_b, op_c = (writer.op(name) for name in "ABC")
    driver.fail(op_a, "row lost")
    time.sleep(PROMPT_S)  # a cancel would have been forwarded by now
    assert writer.driver.calls_to("cancel") == []
    for op in (op_b, op_c):
        driver.complete(op)
    outcome = writer.wait_outcome(attempt)
    assert outcome.fault == "group 0 row 0: row lost"
    assert masks(outcome) == [[False, True, True]]  # a publish's rows stand alone
    for op in (op_a, op_b, op_c):
        driver.end_access(op)
    writer.wait_quiesced(attempt)
    writer.close()


class RefusesItsSecondPut(MemoryDriver):
    puts = 0

    def put(self, rows, *, on_change=None):
        self.puts += 1
        if self.puts == 2:
            raise DriverRejected("full")
        return super().put(rows, on_change=on_change)


def test_a_call_that_concludes_without_a_fault_stops_none_of_the_others(store: MemoryStore) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = one_row_per_call(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_a, op_b = reader.op("A"), reader.op("B")
    driver.complete(op_a)
    driver.end_access(op_a)
    wait_until(lambda: reader.backend.stats()["calls_awaiting_results"] == 1, "A's results")
    time.sleep(PROMPT_S)  # a cancel would have been forwarded by now
    assert reader.driver.calls_to("cancel") == []
    driver.complete(op_b)
    driver.end_access(op_b)
    assert masks(reader.wait_outcome(attempt)) == [[True, True]]
    reader.wait_quiesced(attempt)
    reader.close()


def test_a_refused_put_cancels_the_calls_of_its_publish_already_in_the_driver(
    store: MemoryStore,
) -> None:
    driver = RefusesItsSecondPut(store, controlled=True, capabilities=ONE_ROW_PER_CALL)
    writer = Endpoint(driver, ["A", "B", "C"])
    for name in "ABC":
        writer.write(name, name.encode() * SLOT_BYTES)
    attempt = writer.backend.publish(writer.view("A", "B", "C"), None)
    op_a = writer.op("A")
    writer.driver.wait_for(lambda: writer.driver.calls_to("cancel", op_a), "A's cancel")
    outcome = writer.wait_outcome(attempt)  # A cancelled, B refused, C never handed over
    assert masks(outcome) == [[False, False, False]]
    assert outcome.fault == "group 0 row 1: refused: full"
    writer.wait_quiesced(attempt)
    writer.close()


def test_a_refused_put_stops_its_publish_but_not_the_backend(store: MemoryStore) -> None:
    writer = three_rows(store, objects_in_store=False)
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


def test_an_outcome_waits_for_every_row_of_a_call_that_concludes_in_parts(
    store: MemoryStore,
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op = reader.op("A")  # one call holds both rows
    driver.complete(op, [0])  # tells of A; B is still pending
    assert attempt.poll() is None
    driver.complete(op, [1])
    driver.end_access(op)
    assert masks(reader.wait_outcome(attempt)) == [[True, True]]
    reader.wait_quiesced(attempt)
    reader.close()


@pytest.mark.parametrize("first_half", ["results_first", "access_end_first"])
def test_results_and_access_end_are_each_recorded_once_per_call(
    store: MemoryStore, first_half: str
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = one_row_per_call(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_a, op_b = reader.op("A"), reader.op("B")
    # A lands one half only, and the driver tells of it again and again.
    if first_half == "results_first":
        driver.complete(op_a)
    else:
        driver.transfer(op_a)
        driver.end_access(op_a)
    for _ in range(5):
        reader.driver.call_on_change(op_a)
    # Recording A's half on every on_change would already have concluded the attempt (results_first)
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
def test_news_of_a_settled_operation_changes_nothing(store: MemoryStore, ending: str) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D)
    first = reader.backend.fetch(reader.view("A"), None)
    op_a = reader.op("A")
    if ending == "access_ended":
        driver.complete(op_a)
        driver.end_access(op_a)
        reader.wait_outcome(first)  # quiescing alone may come before the results are recorded
        reader.wait_quiesced(first)
    else:
        driver.fail(op_a, "the store gave up")
        driver.mark_unprovable(op_a)  # nothing the driver could wait on was issued
        assert "the store gave up" in reader.wait_outcome(first).fault
        wait_until(lambda: reader.backend.stats()["calls_unprovable"] == 1, "the count")
    outcome, wakes = first.poll(), reader.wake.count
    reader.driver.call_on_change(op_a)  # a driver may call on_change more than once
    assert first.poll() is outcome and reader.wake.count == wakes

    second = reader.backend.fetch(reader.view("B"), None)  # the backend serves on
    op_b = reader.op("B")
    driver.complete(op_b)
    driver.end_access(op_b)
    reader.wait_outcome(second)
    reader.wait_quiesced(second)
    if ending == "access_ended":
        reader.close()
        return
    assert not first.quiesced()  # its slots stay held
    assert reader.backend.stats()["calls_awaiting_access_end"] == 1
    assert reader.backend.shutdown(0) == frozenset({first.label, "region:0"})


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
def test_cancel_racing_completion_gives_one_outcome(store: MemoryStore, first: str) -> None:
    commit_objects(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    attempt = reader.backend.fetch(reader.view("A"), None)
    op = reader.op("A")
    if first == "cancel_first":
        attempt.cancel()
        assert attempt.poll() is None
        assert reader.wake.count == 0  # nothing landed, so nothing to wake for
        # Forwarded: the driver cancels the row it had not started, which touched nothing.
        reader.driver.wait_for(lambda: reader.driver.calls_to("cancel", op), "the cancel")
        expected = [[False]]
    else:
        driver.complete(op)
        driver.end_access(op)
        reader.wait_outcome(attempt)
        reader.wait_quiesced(attempt)
        wakes = reader.wake.count
        attempt.cancel()
        assert reader.wake.count == wakes
        expected = [[True]]
    outcome = reader.wait_outcome(attempt)
    reader.wait_quiesced(attempt)
    assert attempt.poll() is outcome
    assert masks(outcome) == expected and outcome.fault is None
    assert 1 <= reader.wake.count <= 2  # the outcome and the quiescing, together or apart
    # Only a call the driver still holds is cancelled, once.
    assert len(reader.driver.calls_to("cancel")) == (first == "cancel_first")
    reader.close()


def test_a_cancel_reaches_the_driver_promptly_and_once(store: MemoryStore) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    driver.set_cancel_effective(False)  # the calls stay open
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A"), None)
    op = reader.op("A")
    time.sleep(0.05)  # lets the progress thread settle into its wait
    attempt.cancel()
    started = time.monotonic()
    reader.driver.wait_for(lambda: reader.driver.calls_to("cancel", op), "the cancel")
    assert time.monotonic() - started < PROMPT_S  # woken by the cancel's poke
    attempt.cancel()  # again: nothing more to forward
    # Another attempt's cancel runs another progress round while A's call is still open.
    other = reader.backend.fetch(reader.view("B"), None)
    other_op = reader.op("B")
    other.cancel()
    reader.driver.wait_for(lambda: reader.driver.calls_to("cancel", other_op), "B's cancel")
    assert len(reader.driver.calls_to("cancel", op)) == 1
    for each in (op, other_op):
        driver.complete(each)
        driver.end_access(each)
    reader.wait_quiesced(attempt)
    reader.wait_quiesced(other)
    reader.close()


class FaultsItsFirstGetInside(MemoryDriver):
    """Faults its first get inside ``get``, telling of it there, and gives the progress thread time
    for a round before ``get`` returns: the stop lands before any handle is known."""

    gets = 0

    def get(self, rows, hint=None, *, on_change=None):
        op = super().get(rows, hint, on_change=on_change)
        self.gets += 1
        if self.gets == 1:
            self.fail(op, "row lost")
            self.end_access(op)
            time.sleep(0.05)
        return op


def test_a_stop_during_the_hand_over_reaches_the_calls_handed_over_after_it(
    store: MemoryStore,
) -> None:
    commit_objects(store, {name: name.encode() * SLOT_BYTES for name in "ABC"})
    driver = FaultsItsFirstGetInside(store, controlled=True, capabilities=ONE_ROW_PER_CALL)
    reader = Endpoint(driver, ["A", "B", "C"])
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    outcome = reader.wait_outcome(attempt)  # B's and C's calls conclude only once cancelled
    assert masks(outcome) == [[False, False, False]]
    assert outcome.fault == "group 0 row 0: row lost"
    reader.wait_quiesced(attempt)
    reader.close()


def test_a_cancel_asked_while_the_progress_thread_is_cancelling_is_still_forwarded(
    store: MemoryStore,
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"])
    first = reader.backend.fetch(reader.view("A"), None)
    second = reader.backend.fetch(reader.view("B"), None)
    op_b = reader.op("B")
    gate = reader.driver.gate("cancel")
    first.cancel()
    reader.driver.wait_for(lambda: reader.driver.calls_to("cancel"), "A's cancel")
    second.cancel()  # the progress thread is inside the driver's cancel of A's call
    gate.set()
    reader.driver.wait_for(lambda: reader.driver.calls_to("cancel", op_b), "B's cancel")
    for attempt in (first, second):
        reader.wait_outcome(attempt)
        reader.wait_quiesced(attempt)
    reader.close()


def test_a_driver_without_cancel_is_never_asked_to(store: MemoryStore) -> None:
    commit_objects(store, {"B": b"bbbbbbbb"})  # A is missing: the fetch stops at its fault
    capabilities = replace(DEFAULT_CAPABILITIES, cancel=False, max_rows_per_op=1)
    driver = MemoryDriver(store, controlled=True, capabilities=capabilities)
    reader = Endpoint(driver, ["A", "B"])
    assert not reader.backend.capabilities.cancel  # every call is in the driver at once
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_a, op_b = reader.op("A"), reader.op("B")
    driver.fail(op_a, "lost")
    driver.end_access(op_a)
    time.sleep(PROMPT_S)  # a cancel would have been forwarded by now
    driver.complete(op_b)  # it ran on: nobody could cancel it
    driver.end_access(op_b)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[False, True]] and "lost" in outcome.fault
    reader.wait_quiesced(attempt)
    assert reader.driver.calls_to("cancel") == []
    reader.close()


# The driver tells of changes through on_change.


def test_a_call_is_queried_only_when_its_driver_calls_on_change(store: MemoryStore) -> None:
    commit_objects(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    attempt = reader.backend.fetch(reader.view("A"), None)
    op = reader.op("A")
    time.sleep(PROMPT_S)  # a polling backend would have queried several times by now
    assert reader.driver.calls_to("poll", op) == []
    driver.complete(op)  # tells
    assert masks(reader.wait_outcome(attempt)) == [[True]]
    driver.end_access(op)  # tells
    reader.wait_quiesced(attempt)
    # Queried on the thread that called on_change: here the test's own.
    queries = reader.driver.calls_to("poll", op)
    assert len(queries) == 2 and all(c.thread == threading.current_thread().name for c in queries)
    reader.close()


def test_a_teardown_retries_the_drain_with_backoff(store: MemoryStore) -> None:
    commit_objects(store, {"B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    endpoint = Endpoint(driver, ["A", "B"])
    # An operation outside the backend, already started: no drain can end it.
    row = RowIO(store_key("B"), (Span(endpoint.region.slot(1), SLOT_BYTES),))
    outside = driver.get(RowBatch.of([row]))
    driver.transfer(outside)
    assert endpoint.backend.shutdown(0) == frozenset({"region:0"})
    time.sleep(PROMPT_S)  # polled with backoff: about ten drains
    assert len(endpoint.driver.calls_to("drain")) < 30
    driver.publish_results(outside)
    driver.end_access(outside)
    wait_until(lambda: not endpoint.binding.live_labels(), "the revocation")
    endpoint.close()


def test_a_teardown_follows_the_last_attempt_to_quiesce(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Retries far apart: only the quiescing itself can bring the teardown on in time.
    monkeypatch.setattr(backend_module, "TEARDOWN_RETRY_MIN_S", 10 * WAIT_S)
    monkeypatch.setattr(backend_module, "TEARDOWN_RETRY_MAX_S", 10 * WAIT_S)
    commit_objects(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    attempt = reader.backend.fetch(reader.view("A"), None)
    op = reader.op("A")
    driver.transfer(op)  # running: no drain can end it
    assert reader.backend.shutdown(0) == frozenset({attempt.label, "region:0"})
    reader.driver.wait_for(lambda: reader.driver.calls_to("drain"), "the first teardown round")
    driver.publish_results(op)
    driver.end_access(op)
    wait_until(lambda: not reader.binding.live_labels(), "the revocation")
    reader.close()


def test_a_change_told_before_put_or_get_returns_is_not_lost(store: MemoryStore) -> None:
    commit_objects(store, {"A": b"aaaaaaaa"})
    # An automatic driver runs each operation, and tells of it, inside the put or get that takes
    # it: before the backend has the operation's handle.
    endpoint = Endpoint(MemoryDriver(store), ["A", "B"])
    endpoint.write("B", b"bbbbbbbb")
    fetched = endpoint.backend.fetch(endpoint.view("A"), None)
    offered = endpoint.backend.publish(endpoint.view("B"), None)
    for attempt in (fetched, offered):
        assert masks(endpoint.wait_outcome(attempt)) == [[True]]
        endpoint.wait_quiesced(attempt)
    endpoint.close()


# Driver failures.


def test_a_refusal_faults_the_call_and_ends_its_access(store: MemoryStore) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    writer.driver.fail("put", DriverRejected("x"))
    attempt = writer.backend.publish(writer.view("A"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False]] and outcome.fault == "group 0 row 0: refused: x"
    writer.wait_quiesced(attempt)  # nothing escaped
    assert writer.backend.stats()["calls_unprovable"] == 0
    writer.close()


@keeps_backend_threads
def test_any_other_raise_from_the_driver_holds_the_call_and_refuses_later_work(
    store: MemoryStore,
) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    writer.driver.fail("put", RuntimeError("x"))
    attempt = writer.backend.publish(writer.view("A"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False]]
    assert outcome.fault == "group 0 row 0: the store driver raised RuntimeError('x')"
    assert not attempt.quiesced()  # the rows may be accessed for good
    assert writer.backend.stats()["calls_unprovable"] == 1
    with pytest.raises(SubmissionRejected, match="broke its protocol: the store driver raised"):
        writer.backend.publish(writer.view("A"), None)
    assert attempt.label in writer.backend.shutdown(0)
    with pytest.raises(RuntimeError, match="region:0"):
        writer.binding.release_hold()
    assert not writer.hold.released


class NoHandleDriver(MemoryDriver):
    """Answers ``put`` with None in place of a handle."""

    def put(self, rows, *, on_change=None):
        return None


@keeps_backend_threads
def test_a_put_that_returns_no_handle_holds_the_call_and_refuses_later_work(
    store: MemoryStore,
) -> None:
    writer = Endpoint(NoHandleDriver(store), ["A"])
    attempt = writer.backend.publish(writer.view("A"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False]]
    assert outcome.fault == "group 0 row 0: the store driver returned no handle"
    assert not attempt.quiesced()  # nothing proves the driver stopped reading the slot
    assert writer.backend.stats()["calls_unprovable"] == 1
    with pytest.raises(SubmissionRejected, match="broke its protocol: .* returned no handle"):
        writer.backend.publish(writer.view("A"), None)
    assert writer.backend.shutdown(0) == frozenset({attempt.label, "region:0"})


@keeps_backend_threads
def test_a_protocol_violation_leaves_calls_already_in_the_driver_alone() -> None:
    store = KeyGatedStore(gets=True)
    commit_objects(store, {"X": b"xxxxxxxx"})
    endpoint = Endpoint(MemoryDriver(store, background=True), ["X", "Y"])
    endpoint.write("Y", b"yyyyyyyy")
    in_driver = endpoint.backend.fetch(endpoint.view("X"), None)
    try:
        store.wait_for_calls_at_gate(1)  # its only call is inside the store
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
def test_a_driver_raise_stops_the_rest_of_its_submission(store: MemoryStore) -> None:
    writer = three_rows(store, objects_in_store=False)
    writer.driver.fail("put", RuntimeError("lost"))
    broken = writer.backend.publish(writer.view("A", "B", "C"), None)
    outcome = writer.wait_outcome(broken)
    assert outcome.fault == "group 0 row 0: the store driver raised " + repr(RuntimeError("lost"))
    assert masks(outcome) == [[False, False, False]]
    assert store_calls(writer) == ["put"]  # the raise stopped the publish: B and C never went
    assert writer.backend.stats()["calls_awaiting_access_end"] == 1  # only A's
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
def test_a_poll_with_the_wrong_row_count_faults_the_call_and_keeps_it_held(
    store: MemoryStore,
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = ShortPollDriver(store)
    driver.short = True  # the first query of the operation, as get calls on_change
    reader = Endpoint(driver, ["A", "B", "C"])
    reader.write("C", b"cccccccc")
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[False, False]]
    assert "gave 1 results for 2 rows" in outcome.fault
    # Later answers have the right length and say access ended; after the violation the call is
    # unprovable instead: counted, its slots held.
    assert reader.backend.stats()["calls_unprovable"] == 1
    reader.driver.call_on_change(reader.op("A"))
    assert not attempt.quiesced()
    # The backend survived the bad answer and refuses all later work, naming why.
    assert all(thread.is_alive() for thread in reader.threads)
    with pytest.raises(SubmissionRejected, match="broke its protocol: .* gave 1 results for 2"):
        reader.backend.publish(reader.view("C"), None)
    assert store_calls(reader) == ["get"]
    assert reader.backend.shutdown(0) == frozenset({attempt.label, "region:0"})
    # The driver's own operation ended, so its drain answers True; the call the backend holds
    # still keeps the region registered.
    reader.driver.wait_for(lambda: reader.driver.calls_to("drain"), "a drain")
    time.sleep(PROMPT_S)  # a revocation would have come by now
    assert driver.live_registrations()


@keeps_backend_threads
def test_the_calls_after_a_protocol_violation_never_reach_the_driver(
    store: MemoryStore,
) -> None:
    capabilities = replace(DEFAULT_CAPABILITIES, max_rows_per_op=1)
    writer = Endpoint(MemoryDriver(store, capabilities=capabilities), ["X", "Y", "Z"])
    writer.driver.fail("put", RuntimeError("lost"))  # X's put
    attempt = writer.backend.publish(writer.view("X", "Y", "Z"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False, False, False]]
    assert outcome.fault == "group 0 row 0: the store driver raised " + repr(RuntimeError("lost"))
    assert store_calls(writer) == ["put"]  # Y and Z never reached the driver
    with pytest.raises(SubmissionRejected, match="broke its protocol"):
        writer.backend.publish(writer.view("X"), None)


@keeps_backend_threads
def test_a_violation_told_during_the_hand_over_stops_the_rest_of_it(store: MemoryStore) -> None:
    capabilities = replace(DEFAULT_CAPABILITIES, max_rows_per_op=1)
    writer = Endpoint(MemoryDriver(store, capabilities=capabilities), ["A", "B", "C"])
    for name in "ABC":
        writer.write(name, name.encode() * SLOT_BYTES)
    # A's put runs inside put and calls on_change there; that query raises.
    writer.driver.fail("access_ended", RuntimeError("flaky"))
    attempt = writer.backend.publish(writer.view("A", "B", "C"), None)
    outcome = writer.wait_outcome(attempt)
    assert store_calls(writer) == ["put"]  # B and C never reached the driver
    assert masks(outcome) == [[False, False, False]]
    assert outcome.fault == "group 0 row 0: a store driver query raised " + repr(
        RuntimeError("flaky")
    )
    with pytest.raises(SubmissionRejected, match="broke its protocol"):
        writer.backend.publish(writer.view("A"), None)


class BreaksItsProtocolDuringAPut(MemoryDriver):
    """Controlled. A put of ``trigger_key`` first tells of the earlier operation ``victim`` on
    another thread, with a poll one result short: a violation recorded while a submission is
    being handed over."""

    victim: MemoryOp | None = None
    trigger_key: str | None = None
    short = False

    def poll(self, op):
        results = super().poll(op)
        return results[:-1] if self.short and op is self.victim else results

    def put(self, rows, *, on_change=None):
        op = super().put(rows, on_change=on_change)
        if self.trigger_key in rows.keys():
            self.short = True
            teller = threading.Thread(target=self.victim.on_change, args=(self.victim,))
            teller.start()
            teller.join()
        return op


@keeps_backend_threads
def test_calls_cut_off_by_another_attempts_violation_end_their_access(store: MemoryStore) -> None:
    driver = BreaksItsProtocolDuringAPut(store, capabilities=ONE_ROW_PER_CALL, controlled=True)
    writer = Endpoint(driver, ["X", "A", "B", "C"])
    for name in "XABC":
        writer.write(name, name.encode() * SLOT_BYTES)
    first = writer.backend.publish(writer.view("X"), None)
    driver.victim, driver.trigger_key = writer.op("X"), store_key("A")
    second = writer.backend.publish(writer.view("A", "B", "C"), None)
    assert store_calls(writer) == ["put", "put"]  # X, then A; B and C never reached the driver
    op_a = writer.op("A")
    driver.complete(op_a)
    driver.end_access(op_a)
    outcome = writer.wait_outcome(second)
    assert masks(outcome) == [[True, False, False]]
    assert "group 0 row 1: not handed over: the store driver broke its protocol" in outcome.fault
    with pytest.raises(SubmissionRejected, match="broke its protocol"):
        writer.backend.publish(writer.view("B"), None)
    # B and C touched nothing, so the second attempt quiesces; only the first stays held.
    writer.wait_quiesced(second)
    assert writer.backend.shutdown(0) == frozenset({first.label, "region:0"})


@keeps_backend_threads
def test_a_log_line_that_blocks_never_holds_up_a_submission(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging_now, let_the_log_go = threading.Event(), threading.Event()

    def blocking_warning_once(*args, **kwargs) -> None:  # as a full log pipe would block
        logging_now.set()
        let_the_log_go.wait(10 * WAIT_S)  # outlasts the submitter's wait below

    monkeypatch.setattr(backend_module.logger, "warning_once", blocking_warning_once)
    driver = MemoryDriver(store, controlled=True)
    endpoint = Endpoint(driver, ["A", "B"])
    endpoint.write("A", b"aaaaaaaa")
    endpoint.write("B", b"bbbbbbbb")
    endpoint.backend.publish(endpoint.view("A"), None)
    op = endpoint.op("A")
    # Its on_change records the call as unprovable, then logs that.
    marker = threading.Thread(target=driver.mark_unprovable, args=(op,), daemon=True)
    marker.start()
    try:
        assert logging_now.wait(WAIT_S)
        published: list[object] = []
        submitter = threading.Thread(
            target=lambda: published.append(endpoint.backend.publish(endpoint.view("B"), None)),
            daemon=True,
        )
        submitter.start()
        submitter.join(WAIT_S)
        assert published, "a submission waited for a log line"
    finally:
        let_the_log_go.set()
        marker.join(WAIT_S)


@keeps_backend_threads
def test_a_protocol_violation_is_logged_once_and_never_under_the_lock(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged: list[str] = []
    logging_now, let_the_log_go = threading.Event(), threading.Event()

    def blocking_warning(message: str, *args, **kwargs) -> None:  # as a full log pipe would block
        logged.append(message)
        logging_now.set()
        let_the_log_go.wait(10 * WAIT_S)  # outlasts the waits below

    monkeypatch.setattr(backend_module.logger, "warning", blocking_warning)
    writer = Endpoint(MemoryDriver(store), ["A", "B"])
    writer.driver.fail("put", DriverRejected("busy"))
    writer.wait_outcome(writer.backend.publish(writer.view("A"), None))
    assert logged == []  # a refusal breaks no protocol
    writer.driver.fail("put", RuntimeError("bug"))
    violator = threading.Thread(
        target=lambda: writer.backend.publish(writer.view("B"), None), daemon=True
    )
    violator.start()
    try:
        assert logging_now.wait(WAIT_S)
        stats_read: list[object] = []
        reader = threading.Thread(
            target=lambda: stats_read.append(writer.backend.stats()), daemon=True
        )
        reader.start()
        reader.join(WAIT_S)
        assert stats_read, "the backend's lock was held while a violation was logged"
    finally:
        let_the_log_go.set()
        violator.join(WAIT_S)
    assert len(logged) == 1 and "broke its protocol" in logged[0]


def test_a_wrong_row_count_after_the_access_end_was_recorded_leaves_the_call_provable(
    store: MemoryStore,
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = ShortPollDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op = reader.op("A")
    driver.transfer(op)
    driver.end_access(op)
    wait_until(lambda: reader.backend.stats()["calls_awaiting_access_end"] == 0, "the access end")
    driver.short = True  # the next poll, now that the access end is recorded
    driver.publish_results(op)
    outcome = reader.wait_outcome(attempt)
    assert "gave 1 results for 2 rows" in outcome.fault
    assert attempt.quiesced()
    assert reader.backend.stats()["calls_unprovable"] == 0
    with pytest.raises(SubmissionRejected, match="broke its protocol"):  # a violation all the same
        reader.backend.fetch(reader.view("A"), None)
    reader.close()


@keeps_backend_threads
@pytest.mark.parametrize("failure", ["access_ended_raises", "poll_answers_no_sequence"])
def test_a_query_that_fails_holds_the_call_and_refuses_later_work(
    store: MemoryStore, failure: str
) -> None:
    writer = Endpoint(MemoryDriver(store), ["A", "B"])
    writer.write("A", b"aaaaaaaa")
    if failure == "access_ended_raises":
        writer.driver.fail("access_ended", RuntimeError("flaky"))
    else:
        writer.driver.answer("poll", None)  # the first query of the first operation
    attempt = writer.backend.publish(writer.view("A"), None)
    outcome = writer.wait_outcome(attempt)
    assert masks(outcome) == [[False]] and "a store driver query raised" in outcome.fault
    assert not attempt.quiesced()  # no later answer of the driver proves the access end
    assert writer.backend.stats()["calls_unprovable"] == 1
    with pytest.raises(SubmissionRejected, match="broke its protocol: a store driver query raised"):
        writer.backend.publish(writer.view("B"), None)
    assert writer.backend.shutdown(0) == frozenset({attempt.label, "region:0"})


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


def test_object_keys_in_any_memory_order_are_taken(store: MemoryStore) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    reader = Endpoint(MemoryDriver(store), ["A", "B"])
    rows = reader.rows(["A", "B"])
    view = CpuView([Rows(0, np.asfortranarray(rows.object_keys), rows.addresses, 0)])
    attempt = reader.backend.fetch(view, None)
    assert masks(reader.wait_outcome(attempt)) == [[True, True]]
    reader.wait_quiesced(attempt)
    reader.close()


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
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    capabilities = replace(DEFAULT_CAPABILITIES, hints=True, max_rows_per_op=2)
    driver = MemoryDriver(store, capabilities=capabilities)
    reader = Endpoint(driver, ["A", "B", "C"])
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


def test_submissions_call_put_and_get_and_queries_run_on_the_driver_threads(
    store: MemoryStore,
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, capabilities=ONE_ROW_PER_CALL, background=True)
    endpoint = Endpoint(driver, ["A", "B"])
    endpoint.write("B", b"bbbbbbbb")
    fetched = endpoint.backend.fetch(endpoint.view("A"), None)
    offered = endpoint.backend.publish(endpoint.view("B"), None)
    for attempt in (fetched, offered):
        assert masks(endpoint.wait_outcome(attempt)) == [[True]]
        endpoint.wait_quiesced(attempt)
    submitter = threading.current_thread().name
    handed_over = [c for c in endpoint.driver.calls if c.method in ("put", "get")]
    queries = [c for c in endpoint.driver.calls if c.method in ("poll", "access_ended")]
    assert len(handed_over) == 2 and all(c.thread == submitter for c in handed_over)
    assert queries and all(c.thread == "memory-driver-op" for c in queries)
    endpoint.close()


def test_submissions_return_while_the_driver_waits_on_the_store() -> None:
    store = KeyGatedStore(gets=True)
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = one_row_per_call(store, background=True)
    endpoint = Endpoint(driver, ["A", "B", "C", "D"])
    endpoint.write("C", b"cccccccc")
    blocked = endpoint.backend.fetch(endpoint.view("A", "B"), None)
    attempts = [blocked]
    try:
        store.wait_for_calls_at_gate(2)
        for submit in (
            lambda: attempts.append(endpoint.backend.fetch(endpoint.view("D"), None)),
            lambda: attempts.append(endpoint.backend.publish(endpoint.view("C"), None)),
            blocked.cancel,
        ):
            started = time.monotonic()
            submit()
            assert time.monotonic() - started < PROMPT_S
        # Both of the cancelled fetch's calls are still inside the store.
        assert blocked.quiesced() is False
        assert blocked.label in endpoint.backend.shutdown(0)
    finally:
        store.gate.set()
    for attempt in attempts:
        endpoint.wait_quiesced(attempt)
    endpoint.close()


# Shutdown.


def test_shutdown_forces_fault_outcomes_and_waits_out_its_budget(store: MemoryStore) -> None:
    commit_objects(store, {"B": b"short"})
    driver = one_row_per_call(store, controlled=True)
    driver.set_cancel_effective(False)  # the forwarded cancels end no access
    reader = Endpoint(driver, ["A", "B"])
    attempt = reader.backend.fetch(reader.view("A", "B"), None)
    op_a, op_b = reader.op("A"), reader.op("B")
    driver.transfer(op_a)  # running: the drain a shutdown starts cannot cancel it
    driver.complete(op_b)  # B's fault (the wrong size) lands; A has no result, neither access ends
    assert reader.backend.stats()["calls_awaiting_results"] == 1  # recorded as complete told of it
    started = time.monotonic()
    labels = reader.backend.shutdown(0.2)
    assert time.monotonic() - started >= 0.19  # both calls may still access their slots
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
    driver.publish_results(op_a)  # a late result
    driver.end_access(op_a)
    driver.end_access(op_b)
    reader.wait_quiesced(attempt)
    assert attempt.poll() is outcome
    reader.close()


@keeps_backend_threads
def test_shutdown_returns_once_every_open_attempt_has_an_unprovable_call(
    store: MemoryStore,
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    driver = one_row_per_call(store, controlled=True)
    driver.set_cancel_effective(False)  # the forwarded cancels end no access
    reader = Endpoint(driver, ["A", "B", "C"])
    never_quiesces = reader.backend.fetch(reader.view("A", "B"), None)
    other = reader.backend.fetch(reader.view("C"), None)
    op_a, op_b, op_c = reader.op("A"), reader.op("B"), reader.op("C")
    for op in (op_a, op_b, op_c):
        driver.transfer(op)  # running: the drain a shutdown starts cannot cancel them
    driver.mark_unprovable(op_a)  # its results still pending
    wait_until(lambda: reader.backend.stats()["calls_unprovable"] == 1, "the count")
    # C's attempt can still quiesce, so a shutdown waits for it; A's unprovable call keeps its
    # attempt from ever quiescing, whatever B does, so a shutdown does not wait for it.
    labels: list[frozenset[str]] = []
    shutter = threading.Thread(target=lambda: labels.append(reader.backend.shutdown(WAIT_S)))
    shutter.start()
    time.sleep(PROMPT_S)
    assert shutter.is_alive()
    driver.publish_results(op_c)
    driver.end_access(op_c)
    shutter.join(PROMPT_S)
    assert labels == [frozenset({never_quiesces.label, "region:0"})]
    assert other.quiesced() and not op_b.access_ended
    started = time.monotonic()
    assert reader.backend.shutdown(WAIT_S) == frozenset({never_quiesces.label, "region:0"})
    assert time.monotonic() - started < PROMPT_S
    # An unprovable call stays held whatever the driver says of it later, so the answer stands.
    for op in (op_a, op_b):
        driver.publish_results(op)
        driver.end_access(op)
    assert reader.backend.stats()["calls_awaiting_access_end"] == 1  # B's end, not A's
    assert not never_quiesces.quiesced()
    assert reader.backend.shutdown(0) == frozenset({never_quiesces.label, "region:0"})


@keeps_backend_threads
def test_shutdown_returns_once_a_call_in_the_driver_turns_unprovable(store: MemoryStore) -> None:
    driver = MemoryDriver(store, controlled=True)
    driver.set_cancel_effective(False)  # the forwarded cancel ends no access
    writer = Endpoint(driver, ["A"])
    writer.write("A", b"aaaaaaaa")
    attempt = writer.backend.publish(writer.view("A"), None)
    op = writer.op("A")
    driver.transfer(op)  # running: the drain a shutdown starts cannot cancel it
    labels: list[frozenset[str]] = []
    shutter = threading.Thread(target=lambda: labels.append(writer.backend.shutdown(WAIT_S)))
    shutter.start()
    time.sleep(PROMPT_S)
    assert shutter.is_alive()  # the put may still end its access
    driver.mark_unprovable(op)
    shutter.join(PROMPT_S)
    assert labels == [frozenset({attempt.label, "region:0"})]


@pytest.mark.parametrize("stop", ["cancel", "shutdown"])
def test_cancel_and_shutdown_ask_the_driver_to_cancel_the_open_calls(
    store: MemoryStore, stop: str
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb", "C": b"cccccccc"})
    driver = one_row_per_call(store, controlled=True)
    reader = Endpoint(driver, ["A", "B", "C"])
    attempt = reader.backend.fetch(reader.view("A", "B", "C"), None)
    ops = [reader.op(name) for name in "ABC"]
    if stop == "cancel":
        driver.complete(ops[0])
        driver.end_access(ops[0])
        wait_until(
            lambda: reader.backend.stats()["calls_awaiting_access_end"] == 2, "A's access end"
        )
        attempt.cancel()
    else:
        assert attempt.label in reader.backend.shutdown(0)
    reader.wait_quiesced(attempt)  # the driver cancelled the rows it had not started
    outcome = attempt.poll()
    cancelled = {id(call.op) for call in reader.driver.calls_to("cancel")}
    if stop == "cancel":  # A's row is delivered, without a fault
        assert outcome.fault is None and masks(outcome) == [[True, False, False]]
        assert cancelled == {id(op) for op in ops[1:]}  # only the calls the driver still held
    else:
        assert SHUTDOWN_FAULT in outcome.fault and masks(outcome) == [[False, False, False]]
        assert cancelled == {id(op) for op in ops}
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
    commit_objects(store, {"A": b"aaaaaaaa"})
    reader = Endpoint(MemoryDriver(store, background=True), ["A"])
    reader.backend.fetch(reader.view("A"), None)
    labels: list[frozenset[str]] = []
    # A daemon thread, so a shutdown that never returns fails the wait below instead of the run.
    shutter = threading.Thread(
        target=lambda: labels.append(reader.backend.shutdown(math.inf)), daemon=True
    )
    try:
        store.wait_for_calls_at_gate(1)
        shutter.start()
        time.sleep(PROMPT_S)
        assert shutter.is_alive()  # waiting, since the call is still in the store
    finally:
        store.gate.set()
    shutter.join(WAIT_S)
    assert labels == [frozenset()]
    reader.close()


def test_shutdown_never_calls_the_driver_and_unregisters_in_the_background(
    store: MemoryStore,
) -> None:
    endpoint = Endpoint(MemoryDriver(store), ["A"])
    revoke = endpoint.driver.gate("unregister")
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
    under the backend's lock would leave the helper blocked past its bound. Before ``attempt`` is
    known, as during the submission that wakes, it reads only the backend."""

    def __init__(self) -> None:
        self.attempt = None
        self.backend = None
        self.seen: list[dict] = []

    def __call__(self) -> None:
        attempt, backend, seen = self.attempt, self.backend, {}
        if backend is None:
            return

        def read() -> None:
            seen["stats"] = backend.stats()
            if attempt is not None:
                seen["outcome"] = attempt.poll()
                seen["quiesced"] = attempt.quiesced()
                attempt.cancel()
            seen["finished"] = True

        helper = threading.Thread(target=read, daemon=True)
        helper.start()
        helper.join(WAIT_S)
        self.seen.append(seen)


def test_an_outcome_that_lands_before_access_ends_wakes_the_holder(store: MemoryStore) -> None:
    commit_objects(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    attempt = reader.backend.fetch(reader.view("A"), None)
    op = reader.op("A")
    woken = reader.wake.count
    driver.complete(op)  # results, with access still open
    assert attempt.poll() is not None and not attempt.quiesced()
    assert reader.wake.count > woken
    driver.end_access(op)
    reader.wait_quiesced(attempt)
    reader.close()


def test_a_refused_call_wakes_the_holder(store: MemoryStore) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    writer.driver.fail("put", DriverRejected("busy"))
    attempt = writer.backend.publish(writer.view("A"), None)
    assert attempt.poll() is not None and attempt.quiesced()  # landed inside publish
    assert writer.wake.count >= 1
    writer.close()


@pytest.mark.parametrize("settled", ["at_hand_over", "later"])
def test_wake_runs_after_state_is_published_and_outside_the_lock_on_record(
    store: MemoryStore, settled: str
) -> None:
    commit_objects(store, {"A": b"aaaaaaaa"})
    probe = WakeProbe()
    # An automatic driver settles the operation inside ``get`` and tells of it there; a controlled
    # one settles it later, telling of each change as it comes.
    driver = MemoryDriver(store, controlled=settled == "later")
    reader = Endpoint(driver, ["A"], on_wake=probe)
    probe.backend = reader.backend
    probe.attempt = attempt = reader.backend.fetch(reader.view("A"), None)
    if settled == "later":
        op = reader.op("A")
        driver.complete(op)
        driver.end_access(op)
    reader.wait_quiesced(attempt)
    reader.wait_outcome(attempt)
    assert probe.seen and all(seen.get("finished") for seen in probe.seen)
    # A wake from inside ``get`` comes before fetch returned the attempt to the probe, so only a
    # later wake is sure to read it.
    if settled == "later":
        assert any(seen["outcome"] is not None for seen in probe.seen)
        assert any(seen["quiesced"] for seen in probe.seen)
    reader.close()


def test_wake_runs_after_state_is_published_and_outside_the_lock_on_forced_shutdown() -> None:
    store = KeyGatedStore(gets=True)
    commit_objects(store, {"A": b"aaaaaaaa"})
    probe = WakeProbe()
    reader = Endpoint(MemoryDriver(store, background=True), ["A"], on_wake=probe)
    probe.backend = reader.backend
    probe.attempt = reader.backend.fetch(reader.view("A"), None)
    labels = []
    shutter = threading.Thread(target=lambda: labels.append(reader.backend.shutdown(WAIT_S)))
    try:
        store.wait_for_calls_at_gate(1)
        shutter.start()
        reader.wake.wait_for(lambda: reader.wake.count >= 1, "a wake for the forced outcome")
        assert shutter.is_alive()  # it cannot return while the call is in the store
        (seen,) = probe.seen
        assert seen.get("finished") and SHUTDOWN_FAULT in seen["outcome"].fault
    finally:
        store.gate.set()
        shutter.join(WAIT_S)
    assert labels == [frozenset()]
    reader.close()


def test_the_progress_thread_never_keeps_the_process_alive(store: MemoryStore) -> None:
    endpoint = Endpoint(MemoryDriver(store), ["A"])
    (progress,) = [thread for thread in endpoint.threads if thread.name.endswith("-progress")]
    assert progress.daemon
    endpoint.close()


def test_an_idle_backend_runs_no_progress_round(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    rounds: list[float] = []
    forward_cancels = BlobBackend._forward_cancels

    def counted(backend: BlobBackend) -> None:
        rounds.append(time.monotonic())
        forward_cancels(backend)

    monkeypatch.setattr(BlobBackend, "_forward_cancels", counted)
    endpoint = Endpoint(MemoryDriver(store), ["A"])
    time.sleep(PROMPT_S)
    assert len(rounds) <= 1  # at most the round the thread starts with
    endpoint.close()


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
    assert binding.try_unregister_all() and driver.live_registrations() == []


def test_a_binding_with_no_region_to_register_is_rejected_and_lets_the_hold_go(
    store: MemoryStore,
) -> None:
    driver, hold = MemoryDriver(store, capabilities=HOST_ONLY), FakeHold()
    regions = (CpuRegion(SLOT_BYTES, 0), CpuRegion(SLOT_BYTES, 1, memory_kind=MemoryKind.DEVICE))
    with pytest.raises(ConstructionRejected, match="no region"):
        RegionBinding(regions, driver, hold).register()
    assert hold.released and driver.live_registrations() == []


def test_a_region_whose_layout_is_refused_registers_nothing_and_lets_the_hold_go(
    store: MemoryStore,
) -> None:
    driver, hold = MemoryDriver(store), FakeHold()
    refused = CpuRegion(SLOT_BYTES, 4)
    refused.slot_bytes = 0  # slots of no size: no layout can describe them
    with pytest.raises(ConstructionRejected, match="region:1"):
        RegionBinding((CpuRegion(SLOT_BYTES, 4), refused), driver, hold).register()
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
        reader.backend.driver_call_limit(1)
    assert store_calls(reader) == []
    reader.close()


# Repeated object keys.


REPEATED_KEYS = {
    # name: (the driver's max_rows_per_op, the keys the view names at slots 0, 1, 2, ..., rows in
    # each call)
    "no_limit": (None, "ABA", [2, 1]),
    "limit_of_two": (2, "AABC", [1, 2, 1]),
    "every_row_alike": (None, "AAA", [1, 1, 1]),
}


@pytest.mark.parametrize("direction", ["publish", "fetch"])
@pytest.mark.parametrize("case", REPEATED_KEYS)
def test_rows_naming_one_key_at_two_addresses_never_share_a_call(
    store: MemoryStore, case: str, direction: str
) -> None:
    """A view may name one key at two addresses; a RowBatch carries no key twice, so the group is
    cut there as well as at the driver's row limit, the rows keeping their order."""
    max_rows_per_op, keys, expected = REPEATED_KEYS[case]
    slots = [f"S{i}" for i in range(len(keys))]  # one slot per row, named apart from the keys
    capabilities = replace(DEFAULT_CAPABILITIES, max_rows_per_op=max_rows_per_op)
    endpoint = Endpoint(MemoryDriver(store, capabilities=capabilities), slots)
    if max_rows_per_op is None:
        assert endpoint.backend.driver_call_limit(0) is None  # no limit declared; still cut
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
        commit_objects(store, payload)
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


# Regions registered together.


def registration_calls(endpoint: Endpoint) -> list[str]:
    names = ("register", "unregister", "register_regions", "unregister_regions")
    return [call.method for call in endpoint.driver.calls if call.method in names]


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
    commit_objects(store, {"A": b"aaaaaaaa"})
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
        assert binding.try_unregister_all() and driver.live_registrations() == []
        binding.release_hold()
        assert hold.released
    else:  # nothing to revoke by: the hold is kept for good
        assert revocations == 0 and not binding.try_unregister_all()
        assert not hold.released


def test_a_busy_region_keeps_the_whole_batch_registered(store: MemoryStore) -> None:
    driver = TogetherDriver(store, controlled=True)
    endpoint = Endpoint(driver, ["A"], others=[CpuRegion(SLOT_BYTES, 1)])
    endpoint.write("A", b"aaaaaaaa")
    attempt = endpoint.backend.publish(endpoint.view("A"), None)
    op = endpoint.op("A")
    assert not endpoint.binding.try_unregister_all()  # refused: the put may still read A
    assert endpoint.binding.live_labels() == frozenset({"region:0", "region:1"})
    assert driver.live_batches()
    driver.complete(op)
    driver.end_access(op)
    endpoint.wait_quiesced(attempt)
    endpoint.close()
    assert driver.live_batches() == []


# A backend thread that cannot start.


@pytest.mark.parametrize("cleanup", ["succeeds", "fails"])
def test_a_backend_whose_thread_cannot_start_leaves_the_binding_to_its_owner(
    store: MemoryStore, cleanup: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The progress thread cannot start: the constructor raises, no backend thread survives, and
    the binding its owner registered before the build keeps the registrations and the hold for
    that owner to revoke and release. A revocation that fails keeps both, as after any build."""
    driver = MemoryDriver(store)
    region, hold = CpuRegion(SLOT_BYTES, 1), FakeHold()
    binding = RegionBinding((region,), driver, hold)
    binding.register()
    if cleanup == "fails":
        (handle,) = driver.live_registrations()
        driver.fail_next_unregister(handle, "device busy")

    def start(thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    before = set(threading.enumerate())
    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", start)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            BlobBackend(binding, lambda: None, BlobConfig())
    assert set(threading.enumerate()) == before
    assert binding.live_labels() == frozenset({"region:0"}) and not hold.released
    if cleanup == "fails":
        assert not binding.try_unregister_all() and driver.live_registrations()
        assert binding.live_labels() == frozenset({"region:0"})
        with pytest.raises(RuntimeError, match="region:0"):
            binding.release_hold()
        assert not hold.released
        return
    assert binding.try_unregister_all() and driver.live_registrations() == []
    binding.release_hold()
    assert hold.released
