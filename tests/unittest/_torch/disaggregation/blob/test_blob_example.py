# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend end to end on the CPU: two endpoints sharing objects through one store, a
lease holder's decisions and leases around the outcome and the quiescence of an attempt, the
region hold at shutdown, and two backends living in one process. Each test is also the smallest
working example of its part of the picture; ``README.md`` points here."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest
from cpu_fixtures import (
    SLOT_BYTES,
    WAIT_S,
    CpuRegion,
    Endpoint,
    FakeLeaseHolder,
    commit_object,
    masks,
    store_key,
    wait_until,
)

from tensorrt_llm._torch.disaggregation.blob.store import StoreError
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    DEFAULT_CAPABILITIES,
    MemoryDriver,
    MemoryStore,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


def test_two_endpoints_share_objects_through_the_store(store: MemoryStore) -> None:
    """A writer publishes two rows; a reader fetches them and a third that nobody published."""
    writer = Endpoint(MemoryDriver(store), ["A", "B"])
    reader = Endpoint(MemoryDriver(store), ["A", "B", "C"], fill=0x2D)
    writer.write("A", b"aaaaaaaa")
    writer.write("B", b"bbbbbbbb")
    publish = writer.backend.publish(writer.view("A", "B"), None)
    assert masks(writer.wait_outcome(publish)) == [[True, True]]
    writer.wait_quiesced(publish)
    assert writer.read("A") == b"aaaaaaaa"  # a publish reads its source and never writes it
    assert store.committed_payload(store_key("A")) == b"aaaaaaaa"

    fetch = reader.backend.fetch(reader.view("A", "B", "C"), None)
    outcome = reader.wait_outcome(fetch)
    assert masks(outcome) == [[True, True, False]] and outcome.fault is None
    assert reader.read("A") == b"aaaaaaaa" and reader.read("B") == b"bbbbbbbb"
    assert reader.read("C") == b"-" * SLOT_BYTES  # a miss leaves the slot alone
    reader.wait_quiesced(fetch)

    writer.write("A", b"xxxxxxxx")  # the store keeps its own copy: the source may change after
    again = reader.backend.fetch(reader.view("A"), None)
    assert masks(reader.wait_outcome(again)) == [[True]] and reader.read("A") == b"aaaaaaaa"
    reader.wait_quiesced(again)
    writer.close()
    reader.close()


def test_a_fault_with_a_late_write_keeps_the_lease_until_the_attempt_quiesced(
    store: MemoryStore,
) -> None:
    """The holder decides as soon as the outcome lands, but a faulted row's transfer may still
    be writing: the lease waits for the evidence that it stopped."""
    for name, data in {"A": b"aaaaaaaa", "B": b"bbbbbbbb"}.items():
        commit_object(store, name, data)
    one_row_per_call = replace(DEFAULT_CAPABILITIES, max_rows_per_op=1)
    driver = MemoryDriver(store, controlled=True, capabilities=one_row_per_call)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D)
    holder = FakeLeaseHolder(reader)
    attempt = holder.fetch("A", "B")
    op_a, op_b = reader.op("A"), reader.op("B")
    driver.complete(op_a)
    driver.end_access(op_a)
    driver.fail(op_b, "link reset")  # B's result is in; its access is not over
    reader.wait_outcome(attempt)
    holder.on_wake()
    assert holder.delivered(attempt) == [[True, False]]
    assert attempt.poll().fault == "group 0 row 1: link reset"
    assert attempt.label in holder.leased  # decided, and still leased
    driver.write_late(op_b, 0, b"late....")  # what the kept lease protects the slot from
    assert reader.read("B") == b"late...."
    driver.end_access(op_b)
    reader.wait_quiesced(attempt)
    holder.on_wake()
    assert attempt.label not in holder.leased
    assert holder.delivered(attempt) == [[True, False]]  # the decision stood
    reader.close()


def test_a_deadline_decides_from_nothing_and_a_late_outcome_changes_no_decision(
    store: MemoryStore,
) -> None:
    commit_object(store, "A", b"aaaaaaaa")
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"], fill=0x2D)
    holder = FakeLeaseHolder(reader)
    attempt = holder.fetch("A")
    op = reader.op("A")
    holder.decide(attempt)  # the holder's deadline: nothing has concluded
    assert holder.delivered(attempt) is None and attempt.label in holder.leased
    driver.complete(op)
    driver.end_access(op)
    reader.wait_outcome(attempt)
    reader.wait_quiesced(attempt)
    holder.on_wake()
    assert holder.delivered(attempt) is None  # decided once; the row landed but does not count
    assert reader.read("A") == b"aaaaaaaa"
    assert attempt.label not in holder.leased  # the lease still waited for the evidence
    reader.close()


def test_the_hold_goes_back_only_once_every_region_is_revoked(store: MemoryStore) -> None:
    """Two regions under one hold. Region 0's revocation fails once and its retry is held at a
    gate, so the state in between is fixed: region 1 revoked, region 0 not. The manager keeps
    the hold, and the binding, until the retry succeeded; nothing frees it on its own."""
    endpoint = Endpoint(MemoryDriver(store), ["A"], others=[CpuRegion(SLOT_BYTES, 2)])
    holder = FakeLeaseHolder(endpoint)
    endpoint.driver.fail("unregister", StoreError("device busy"), times=1)  # region 0, first try
    retry = endpoint.driver.gate("unregister", after=2)  # region 1 goes through; the retry waits
    try:
        labels = holder.shut_down(0)
        assert labels and labels <= {"region:0", "region:1"} and not endpoint.hold.released
        endpoint.driver.wait_for(
            lambda: len(endpoint.driver.calls_to("unregister")) == 3, "the retry at the gate"
        )
        assert endpoint.binding.live_labels() == frozenset({"region:0"})
        with pytest.raises(RuntimeError, match="region:0"):
            endpoint.binding.release_hold()  # one region still registered keeps the memory
        assert not endpoint.hold.released
    finally:
        retry.set()
    wait_until(lambda: not endpoint.binding.live_labels(), "the retried revocation")
    assert not endpoint.hold.released  # revoked, but the manager has not been told yet
    assert holder.shut_down(0) == frozenset() and endpoint.hold.released
    assert endpoint.hold.release_threads == [threading.get_ident()]
    for thread in endpoint.threads:
        thread.join(WAIT_S)
    assert not any(thread.is_alive() for thread in endpoint.threads)


def test_two_backends_in_one_process_serve_and_close_independently(store: MemoryStore) -> None:
    first = Endpoint(MemoryDriver(store), ["A", "B"], fill=0x2D)
    second = Endpoint(MemoryDriver(store), ["A", "B"], fill=0x2D)
    first.write("A", b"aaaaaaaa")
    second.write("B", b"bbbbbbbb")
    attempts = [
        first.backend.publish(first.view("A"), None),
        second.backend.publish(second.view("B"), None),
    ]
    for endpoint, attempt in zip((first, second), attempts, strict=True):
        assert masks(endpoint.wait_outcome(attempt)) == [[True]]
        endpoint.wait_quiesced(attempt)
    crossed = [
        first.backend.fetch(first.view("B"), None),  # what the other one published
        second.backend.fetch(second.view("A"), None),
    ]
    for endpoint, attempt in zip((first, second), crossed, strict=True):
        assert masks(endpoint.wait_outcome(attempt)) == [[True]]
        endpoint.wait_quiesced(attempt)
    assert first.read("B") == b"bbbbbbbb" and second.read("A") == b"aaaaaaaa"
    first.close()  # the other backend, with its own driver and threads, is untouched
    later = second.backend.fetch(second.view("B"), None)
    assert masks(second.wait_outcome(later)) == [[True]]
    second.wait_quiesced(later)
    second.close()
