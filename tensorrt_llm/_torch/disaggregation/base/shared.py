# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed cache backend contract.

Shared-cache consumers import this module explicitly. The paired transceiver
uses ``base.backend`` and its package re-exports; its request-scoped extents and
outcomes are distinct from the process-lifetime backend contract here.

Names encode content, reuse scope, layout, layer identity, and relevant sharding.
Backends treat them as opaque bytes; local coordinates resolve memory separately.
Logical outcomes never authorize memory reuse: only ``quiesce`` returning
``True`` proves that the listed deliveries have ended all caller-memory access.
These protocols describe obligations, not a mechanism that enforces them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
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
]


@dataclass(frozen=True, kw_only=True)
class Unit:
    """One independently named, indivisibly delivered piece of cache content.

    Args:
        name: Opaque identity, unique within the enclosing extent's name.
        local_group: Nonnegative process-local layer-group ordinal. It must not
            be sent to a peer or compared with a peer's group ordinal.
        local: Nonnegative region identifier within that local group. Together
            the two coordinates resolve one or more address-and-length spans.
    """

    name: bytes
    local_group: int
    local: int

    def __post_init__(self) -> None:
        """Validate local coordinates.

        Raises:
            ValueError: A coordinate is negative.
        """
        if self.local_group < 0 or self.local < 0:
            raise ValueError(f"negative local coordinate ({self.local_group}, {self.local})")


@dataclass(frozen=True, kw_only=True)
class CacheExtent:
    """Immutable description of one fetch destination or publish source.

    Name derivation must include every quantity whose cross-side mismatch could
    produce incorrect bytes. Layout, content, reuse scope, layer identity, token
    coverage, and relevant sharding therefore belong in names, not extra fields.
    This interface cannot validate that derivation.

    Args:
        name: Opaque content name scoping all unit names in this delivery.
        units: Units with distinct names. An empty extent is valid. Construction
            snapshots the supplied sequence into a tuple.
        is_last: Explicit sequence-end marker. Only sequence-aware backends may
            inspect it; a name-addressed store must not use it.
    """

    name: bytes
    units: tuple[Unit, ...]
    is_last: bool

    def __post_init__(self) -> None:
        """Snapshot the unit sequence and validate names within this extent.

        Raises:
            ValueError: Two units have the same name.
        """
        object.__setattr__(self, "units", tuple(self.units))
        names = [unit.name for unit in self.units]
        if len(set(names)) != len(names):
            raise ValueError("two units in one extent share a name")


@dataclass(frozen=True)
class Delivered:
    """A logical answer listing only fully served units from this delivery.

    Empty ``served`` is a fetch miss or an unaccepted publication, not an error.
    Unserved destinations must remain untouched. The caller correlates names to
    retained coverage records and performs readiness checks before consumption;
    this outcome establishes neither engine readiness nor physical quiescence.

    Args:
        served: Subset of the requested unit names whose delivery completed.
    """

    served: frozenset[bytes]


@dataclass(frozen=True)
class Failed:
    """Logical failure; destination contents are undefined until overwritten.

    A miss must not be reported as failure, nor a transport, capacity, or
    registration failure as a miss. Failure may precede physical quiescence.

    Args:
        reason: Human-readable failure description.
    """

    reason: str


@dataclass(frozen=True)
class Cancelled:
    """Logical cancellation without delivery, initiated outside this interface.

    There is no cancellation operation on this interface. Submitted work may
    continue accessing memory after cancellation is reported.

    Args:
        by_peer: Whether the peer initiated cancellation. Peer cancellation is
            a transfer error; local cancellation is ordinary termination.
    """

    by_peer: bool


Outcome = Delivered | Failed | Cancelled


@runtime_checkable
class Attempt(Protocol):
    """One delivery with a stable logical outcome, independent of memory access."""

    def poll(self) -> Outcome | None:
        """Inspect the current logical result without blocking.

        Returns:
            None while pending, otherwise an outcome that never changes on
            subsequent polls. No outcome proves physical quiescence.
        """
        ...


class SubmissionRejected(Exception):
    """Submission rejected before work, peer notification, or memory access escaped.

    The caller has nothing to quiesce. After any such effect escapes, submission
    must return an Attempt and report failure through its logical outcome.
    """


class Route(Protocol):
    """Opaque per-request source plan created by a backend; it retains no content."""

    def close(self) -> None:
        """End route preparation after the caller stops submitting along it.

        Existing deliveries continue and still require quiescence. Successful
        closure is idempotent. A raised exception leaves the route open and
        retryable; it must not be recorded as successfully closed.
        """
        ...


class Registration(Protocol):
    """Opaque handle for one non-overlapping registered memory span."""

    def close(self) -> None:
        """Revoke registration after all deliveries using it pass quiescence.

        This operation neither waits for access to end nor proves memory safety.
        Success is idempotent by handle: closing an old handle again must not
        revoke a later registration of the same address. If closure raises, the
        span remains registered and the same handle can be retried.
        """
        ...


@runtime_checkable
class Fetches(Protocol):
    """Process-lifetime fetching backend with all five methods required.

    Calls may overlap across requests, directions, and waits on the same
    Attempt. Runtime protocol checks establish member presence, not signature,
    concurrency, or semantic conformance. Observable per-unit hit counters are
    required operationally but their form is outside this interface.
    """

    def fetch(self, extent: CacheExtent, *, route: Route | None = None) -> Attempt:
        """Submit a fetch immediately without waiting for completion.

        Args:
            extent: Content and local destination coordinates.
            route: Unmodified open route produced by this backend, or None.

        Returns:
            Handle for every submission whose effects escaped, including failed
            submissions. For backends implementing RegistersPools, unregistered
            destinations must yield Failed.

        Raises:
            SubmissionRejected: Nothing escaped. This includes an unsupported,
                foreign, or closed route rejected before any submission effect.
        """
        ...

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """Wait for a memory-access determination for only the listed deliveries.

        Args:
            attempts: Submitted deliveries, possibly passed to this wait before.

        Returns:
            True only when all listed deliveries will never again access caller
            memory. False means this cannot be established, without promising
            that waiting longer will help. Retire their spans without freeing,
            reconstructing, or reusing them until a later call returns True.
            Neither answer implies that a logical outcome exists.
        """
        ...

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """Wait until every listed delivery has a stable logical outcome.

        This wait is unbounded and must not wait for unlisted deliveries. It
        does not establish physical quiescence; use timed polling for a bounded
        wait and quiesce for the separate memory-access question.

        Args:
            attempts: Submitted deliveries, possibly passed to this wait before.
        """
        ...

    def probe(self, name: bytes, units: Sequence[bytes]) -> frozenset[bytes] | None:
        """Report advisory hits without moving payload or reserving content.

        Args:
            name: Opaque content name scoping the requested units.
            units: Requested unit names.

        Returns:
            A subset of requested names, empty for no hits, or None if probing
            cannot answer more cheaply than fetching. A later fetch may miss
            previously reported hits. Probe failures must raise, never become
            an empty set or None.
        """
        ...

    def open_route(self, hint: Mapping[str, object]) -> Route:
        """Prepare a per-request route without waiting or touching cache memory.

        Preparation may initiate control-plane work but must not move payload
        or retain source content. A valid hint's transport error propagates
        unchanged; this operation must never raise SubmissionRejected.

        Args:
            hint: Deployment-defined routing information, opaque to this API.

        Returns:
            An opaque route belonging to this backend and request.

        Raises:
            NotImplementedError: This is a single-source backend.
            ValueError: The hint is unrecognized, incomplete, or names an
                unknown source.
        """
        ...


@runtime_checkable
class Publishes(Protocol):
    """Process-lifetime publication, independent of the Fetches capability.

    Concurrent calls and repeated waits must be supported. Publication is
    atomically visible per unit, not per extent. Repeated publication under a
    content name merges units without removing previously published units.
    """

    def publish(self, extent: CacheExtent) -> Attempt:
        """Submit publication immediately using already-readable source memory.

        Args:
            extent: Content and local source coordinates. No route is supplied:
                the backend is the store or answers the requesting peer.

        Returns:
            Handle for every submission whose effects escaped, including failed
            submissions. For backends implementing RegistersPools, unregistered
            sources must yield Failed.

        Raises:
            SubmissionRejected: No work, notification, or memory access escaped.
        """
        ...

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """Establish physical quiescence with the same semantics as Fetches.

        Args:
            attempts: Deliveries whose memory access must end; unlisted
                deliveries must not determine when this wait returns.

        Returns:
            True only if caller-memory access has ended permanently. False
            requires retaining all affected memory and promises no eventual
            True answer. Logical outcomes remain independent.
        """
        ...

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """Wait for logical outcomes with the same semantics as Fetches.

        Args:
            attempts: Deliveries that must have stable outcomes before return.
                Unlisted deliveries must not determine when this wait returns.
        """
        ...


@runtime_checkable
class RegistersPools(Protocol):
    """Optional pool registration, independent of local-coordinate resolution."""

    def register_pool(self, address: int, size: int) -> Registration:
        """Register one memory span before any delivery refers to it.

        Register each span once; overlapping live registrations must be
        rejected. Failure raises and must register nothing. Registration does
        not establish how local coordinates resolve to this address interval.

        Args:
            address: Starting memory address accessible to the backend.
            size: Span length in bytes.

        Returns:
            A handle that revokes exactly this registration after all referring
            deliveries are quiescent and new submissions have stopped.
        """
        ...
