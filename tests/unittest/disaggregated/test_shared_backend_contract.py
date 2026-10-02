# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the shared backend data and protocol surface.

The controllable backend supplies examples of independent logical and physical
completion. Those examples validate the test double, not provider conformance;
each real provider must establish these guarantees with its own transfer tests.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import FrozenInstanceError, fields
from inspect import Parameter, signature
from threading import Event
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from tensorrt_llm._torch.disaggregation.base import shared
from tensorrt_llm._torch.disaggregation.base.shared import (
    Attempt,
    CacheExtent,
    Cancelled,
    Delivered,
    Failed,
    Fetches,
    Outcome,
    Publishes,
    RegistersPools,
    Registration,
    Route,
    SubmissionRejected,
    Unit,
)

pytestmark = pytest.mark.cpu_only


class _ControlledAttempt:
    """An outcome and physical-access evidence advanced separately by a test."""

    def __init__(self) -> None:
        """Initialize a pending delivery with no evidence of ended access."""
        self._outcome: Outcome | None = None
        self.logical_ready = Event()
        self.access_ended = Event()

    def poll(self) -> Outcome | None:
        """Return the immutable logical outcome, if supplied by the test."""
        return self._outcome

    def answer(self, outcome: Outcome) -> None:
        """Commit one logical answer without changing access evidence.

        Args:
            outcome: Answer to expose on every subsequent poll.

        Raises:
            ValueError: An answer has already been committed.
        """
        if self._outcome is not None:
            raise ValueError("attempt already has an outcome")
        self._outcome = outcome
        self.logical_ready.set()


class _ControlledStore:
    """Single-source fetch double backed by local bytearrays and explicit events."""

    def __init__(self, memory: dict[tuple[int, int], bytearray]) -> None:
        """Initialize a double without populating remote content.

        Args:
            memory: Destination spans keyed by local group and region.
        """
        self.memory = memory
        self.content: dict[tuple[bytes, bytes], bytes] = {}
        self.attempts: list[_ControlledAttempt] = []
        self.extents: list[CacheExtent] = []
        self.hits = 0
        self.reject = False
        self.fail_after_write = False

    def fetch(self, extent: CacheExtent, *, route: Route | None = None) -> _ControlledAttempt:
        """Record submission, with controlled rejection or escaped failure.

        Args:
            extent: Names and destination coordinates.
            route: Unsupported by this single-source double.

        Returns:
            An attempt for every accepted submission.

        Raises:
            SubmissionRejected: Rejection was requested or a route was supplied;
                no destination, counter, or submitted-attempt state changes.
        """
        if self.reject or route is not None:
            raise SubmissionRejected("submission rejected before effects")
        attempt = _ControlledAttempt()
        self.attempts.append(attempt)
        self.extents.append(extent)
        if self.fail_after_write:
            unit = extent.units[0]
            self.memory[unit.local_group, unit.local][0] = 0
            attempt.answer(Failed("transport failed after touching destination"))
        return attempt

    def deliver(self, attempt: _ControlledAttempt) -> None:
        """Write complete hits and provide an answer without ending access.

        Args:
            attempt: Delivery previously submitted to this double.
        """
        extent = self.extents[self.attempts.index(attempt)]
        served: set[bytes] = set()
        for unit in extent.units:
            payload = self.content.get((extent.name, unit.name))
            if payload is not None:
                destination = self.memory[unit.local_group, unit.local]
                if len(destination) != len(payload):
                    attempt.answer(Failed("matching name has incompatible destination"))
                    return
                destination[:] = payload
                served.add(unit.name)
                self.hits += 1
        attempt.answer(Delivered(frozenset(served)))

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """Report only explicit no-future-access evidence supplied by the test.

        Args:
            attempts: Deliveries to inspect independently of logical outcomes.

        Returns:
            True if every delivery has ended access; False otherwise, with no
            guarantee that evidence will ever be supplied.
        """
        return all(
            self.attempts[self.attempts.index(attempt)].access_ended.is_set()
            for attempt in attempts
        )

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """Wait only for the supplied deliveries' logical outcomes.

        Args:
            attempts: Deliveries whose outcomes must exist on return.
        """
        for attempt in attempts:
            self.attempts[self.attempts.index(attempt)].logical_ready.wait()

    def probe(self, name: bytes, units: Sequence[bytes]) -> frozenset[bytes]:
        """Return held unit names without reserving them or touching destinations.

        Args:
            name: Extent name scoping the requested unit names.
            units: Candidate unit names.

        Returns:
            The requested unit names currently held by this double.
        """
        return frozenset(unit for unit in units if (name, unit) in self.content)

    def open_route(self, hint: Mapping[str, object]) -> Route:
        """Reject routing for this single-source backend.

        Args:
            hint: Unused deployment hint.

        Raises:
            NotImplementedError: Single-source backends have no route to open.
        """
        raise NotImplementedError("single-source backend")


def _extent() -> CacheExtent:
    """Return two units with equal region IDs in different local groups."""
    return CacheExtent(
        name=b"content",
        units=(
            Unit(name=b"first", local_group=0, local=0),
            Unit(name=b"second", local_group=1, local=0),
        ),
        is_last=True,
    )


def test_export_surface_and_fields_match_canonical_contract() -> None:
    """Keep incompatible content metadata out of the backend's field surface."""
    assert set(shared.__all__) == {
        "Unit",
        "CacheExtent",
        "Delivered",
        "Failed",
        "Cancelled",
        "Outcome",
        "Attempt",
        "SubmissionRejected",
        "Route",
        "Registration",
        "Fetches",
        "Publishes",
        "RegistersPools",
    }
    assert len(shared.__all__) == 13
    for cls, names in (
        (Unit, ["name", "local_group", "local"]),
        (CacheExtent, ["name", "units", "is_last"]),
        (Delivered, ["served"]),
        (Failed, ["reason"]),
        (Cancelled, ["by_peer"]),
    ):
        assert [field.name for field in fields(cls)] == names
    assert set(get_type_hints(Fetches.quiesce)) == {"attempts", "return"}
    assert get_type_hints(Fetches.quiesce)["return"] is bool
    assert get_type_hints(Publishes.quiesce)["return"] is bool
    route = signature(Fetches.fetch).parameters["route"]
    assert route.kind is Parameter.KEYWORD_ONLY and route.default is None
    assert set(signature(Publishes.publish).parameters) == {"self", "extent"}


@pytest.mark.parametrize("local_group,local", [(-1, 0), (0, -1), (-1, -1)])
def test_negative_coordinates_are_rejected(local_group: int, local: int) -> None:
    """A unit must resolve through nonnegative local coordinates."""
    with pytest.raises(ValueError, match="negative local coordinate"):
        Unit(name=b"opaque", local_group=local_group, local=local)


def test_extent_snapshots_sequence_and_rejects_duplicate_names() -> None:
    """Caller list mutation cannot change a lent extent or introduce aliases."""
    units = list(_extent().units)
    extent = CacheExtent(name=b"content", units=units, is_last=False)
    units.clear()
    assert extent.units == _extent().units
    with pytest.raises(ValueError, match="share a name"):
        CacheExtent(
            name=b"content",
            units=(extent.units[0], Unit(name=b"first", local_group=9, local=7)),
            is_last=True,
        )


@pytest.mark.parametrize(
    "value,attribute,new_value",
    [
        (Unit(name=b"first", local_group=0, local=0), "local", 1),
        (_extent(), "units", ()),
        (Delivered(frozenset({b"first"})), "served", frozenset()),
        (Failed("failure"), "reason", "replacement"),
        (Cancelled(False), "by_peer", True),
    ],
)
def test_descriptions_and_outcomes_are_frozen(
    value: object, attribute: str, new_value: object
) -> None:
    """Delivery descriptors and correctly typed outcomes reject reassignment."""
    with pytest.raises(FrozenInstanceError):
        setattr(value, attribute, new_value)


def test_empty_extent_is_valid_and_markers_are_explicit() -> None:
    """An empty delivery is distinct from an omitted sequence or cancel marker."""
    assert CacheExtent(name=b"", units=(), is_last=True).units == ()
    assert signature(CacheExtent).parameters["is_last"].default is Parameter.empty
    assert signature(Cancelled).parameters["by_peer"].default is Parameter.empty
    for parameter in signature(Unit).parameters.values():
        assert parameter.kind is Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("missing", ["fetch", "quiesce", "settle", "probe", "open_route"])
def test_fetch_requires_every_member(missing: str) -> None:
    """Unavailable probing and routing are answers, not omitted capabilities."""
    store = _ControlledStore({})
    members = {
        name: getattr(store, name)
        for name in ("fetch", "quiesce", "settle", "probe", "open_route")
        if name != missing
    }
    assert not isinstance(SimpleNamespace(**members), Fetches)
    assert isinstance(store, Fetches)
    assert not isinstance(store, Publishes)
    assert not isinstance(store, RegistersPools)
    assert isinstance(_ControlledAttempt(), Attempt)


def test_handles_do_not_advertise_runtime_protocol_checks() -> None:
    """Opaque handles are returned unchanged; structural checks are not promised."""
    for protocol in (Route, Registration):
        with pytest.raises(TypeError, match="runtime_checkable"):
            isinstance(object(), protocol)


@pytest.mark.parametrize(
    "hits",
    [frozenset(), frozenset({b"first"}), frozenset({b"first", b"second"})],
    ids=["miss", "partial", "complete"],
)
def test_controlled_store_leaves_unserved_destinations_untouched(hits: frozenset[bytes]) -> None:
    """The fixture expresses whole-unit answers with distinct local group lookup."""
    extent = _extent()
    store = _ControlledStore({(0, 0): bytearray(b"----"), (1, 0): bytearray(b"----")})
    store.content = {(extent.name, name): b"data" for name in hits}
    attempt = store.fetch(extent)
    assert attempt.poll() is None
    store.deliver(attempt)
    outcome = attempt.poll()
    assert isinstance(outcome, Delivered) and outcome.served == hits
    for unit in extent.units:
        assert store.memory[unit.local_group, unit.local] == (
            b"data" if unit.name in hits else b"----"
        )
    assert store.hits == len(hits)
    assert not store.quiesce([attempt])


def test_controlled_probe_is_advisory_and_single_source_routes_are_rejected() -> None:
    """A hit may disappear after probing; rejected routes never submit a fetch."""
    store = _ControlledStore({(0, 0): bytearray(b"----"), (1, 0): bytearray(b"----")})
    store.content[b"content", b"first"] = b"data"
    assert store.probe(b"content", [b"first", b"second"]) == frozenset({b"first"})
    store.content.clear()
    attempt = store.fetch(_extent())
    store.deliver(attempt)
    assert attempt.poll() == Delivered(frozenset())
    with pytest.raises(NotImplementedError):
        store.open_route({"source": "worker"})
    assert len(store.attempts) == 1


@pytest.mark.parametrize("escaped", [False, True], ids=["rejected", "escaped-failure"])
def test_controlled_submission_retains_a_handle_after_effects_escape(escaped: bool) -> None:
    """The fixture distinguishes no-effect rejection from a partially written failure."""
    store = _ControlledStore({(0, 0): bytearray(b"----"), (1, 0): bytearray(b"----")})
    store.reject, store.fail_after_write = not escaped, escaped
    if not escaped:
        with pytest.raises(SubmissionRejected):
            store.fetch(_extent())
        assert store.attempts == [] and store.hits == 0
        assert store.memory == {(0, 0): bytearray(b"----"), (1, 0): bytearray(b"----")}
    else:
        attempt = store.fetch(_extent())
        assert isinstance(attempt.poll(), Failed)
        assert store.memory[0, 0][0] == 0
        store.settle([attempt])
        assert not store.quiesce([attempt])
        attempt.access_ended.set()
        assert store.quiesce([attempt])


@pytest.mark.parametrize("physical_first", [False, True], ids=["outcome-first", "access-first"])
def test_controlled_outcome_and_access_end_are_independent(physical_first: bool) -> None:
    """Either event may precede the other, independently of unrelated attempts."""
    store = _ControlledStore({})
    extent = CacheExtent(name=b"empty", units=(), is_last=True)
    attempt, unrelated = store.fetch(extent), store.fetch(extent)
    outcome = Cancelled(by_peer=False)
    if physical_first:
        attempt.access_ended.set()
        assert store.quiesce([attempt])
        assert attempt.poll() is None
        attempt.answer(outcome)
    else:
        attempt.answer(outcome)
        store.settle([attempt])
        assert not store.quiesce([attempt])
        attempt.access_ended.set()
    for _ in range(2):
        store.settle([attempt])
        assert store.quiesce([attempt])
        assert attempt.poll() is outcome
    assert unrelated.poll() is None and not store.quiesce([unrelated])
    with pytest.raises(ValueError, match="already has an outcome"):
        attempt.answer(Failed("late failure"))
