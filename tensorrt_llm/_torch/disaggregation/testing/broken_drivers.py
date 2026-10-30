# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drivers that each break exactly one rule of the driver boundary.

Each is paired with the conformance check that guards its rule. A test runs the check against the
sound in-memory driver, which must pass, and against the broken one, which must fail; a check that
a weakened assertion lets both pass is caught there, before a real driver meets it.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Sequence
from typing import Any

from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    DriverRejected,
    Hint,
    OnChange,
    OpHandle,
    ProbeStatus,
    RegionHandle,
    RegionLayout,
    RegionSpec,
    RegistrationError,
    RowBatch,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreError,
    faulted,
)

from .memory_store import (
    DEFAULT_CAPABILITIES,
    Kind,
    MemoryDriver,
    MemoryOp,
    MemoryStore,
    ObjectNotReady,
    TogetherMemoryDriver,
    _Batch,
    _cancellable,
    _Registration,
    _scatter,
)

_MISS = RowResult(RowStatus.MISS)
_READY = RowResult(RowStatus.READY)
_CANCELLED = RowResult(RowStatus.CANCELLED)


class TouchesMemoryBeforeChecking(MemoryDriver):
    def get(
        self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None
    ) -> MemoryOp:
        for io in rows:
            for span in io.spans:
                ctypes.memset(span.address, 0, span.nbytes)
        return super().get(rows, hint, on_change=on_change)


class ChecksRowsNotExtents(MemoryDriver):
    def _check_extents(self, rows: Sequence[RowIO]) -> None:
        pass


class ChecksOnlyTheStartOfASpan(MemoryDriver):
    def _registered(self, span: Span) -> bool:
        return any(
            r.live and r.span.address <= span.address < r.span.end for r in self._registrations
        )


class MissTouchesTheDestination(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        result = super()._get_row(io)
        if result.status is RowStatus.MISS:
            _scatter(io.spans, bytes(io.nbytes))
        return result


class TruncatesLargerObjects(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        try:
            stored = self.store.get(io.key)
        except ObjectNotReady:  # another writer's: a sound get answers that
            return super()._get_row(io)
        if stored is not None and len(stored.payload) > io.nbytes:
            _scatter(io.spans, stored.payload[: io.nbytes])
            return _READY
        return super()._get_row(io)


class TruncatesIntoMultiSpanRows(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        try:
            stored = self.store.get(io.key)
        except ObjectNotReady:  # another writer's: a sound get answers that
            return super()._get_row(io)
        if len(io.spans) > 1 and stored is not None and len(stored.payload) > io.nbytes:
            _scatter(io.spans, stored.payload[: io.nbytes])
            return _READY
        return super()._get_row(io)


class CallsAWrongSizeAMiss(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        result = super()._get_row(io)
        return _MISS if result.status is RowStatus.FAULT else result


class CallsATimeoutAMiss(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        try:
            return super()._get_row(io)
        except StoreError:
            return _MISS


class ClobbersItsSource(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        result = super()._put_row(io)
        _scatter(io.spans, bytes(io.nbytes))
        return result


class OverwritesExistingObjects(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        if self.store.presence(io.key) is ProbeStatus.HIT:  # only an object that exists
            payload = b"".join(ctypes.string_at(span.address, span.nbytes) for span in io.spans)
            self.store.overwrite(io.key, payload, self._trailer_for(io, payload))
        return super()._put_row(io)


class WritesTheWholeRegion(MemoryDriver):
    """Clears the whole region a get row lies in, then delivers the row itself as it should."""

    def _get_row(self, io: RowIO) -> RowResult:
        for registration in self.live_registrations():
            if io.spans[0].within(registration.span):
                ctypes.memset(registration.span.address, 0, registration.span.nbytes)
        return super()._get_row(io)


class OrdersSpansByAddress(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        return super()._put_row(RowIO(io.key, tuple(sorted(io.spans, key=lambda s: s.address))))

    def _get_row(self, io: RowIO) -> RowResult:
        return super()._get_row(RowIO(io.key, tuple(sorted(io.spans, key=lambda s: s.address))))


class ChecksOnlyTheFirstSpan(MemoryDriver):
    def _check_registered(self, rows: Sequence[RowIO]) -> None:
        super()._check_registered([RowIO(io.key, io.spans[:1]) for io in rows])

    def _check_extents(self, rows: Sequence[RowIO]) -> None:
        pass  # nor the extents, which would catch the lapse above


class AcceptsOverlappingRegistrations(MemoryDriver):
    def register(self, address: int, nbytes: int, layout: RegionLayout) -> RegionHandle:
        with self._lock:
            registration = _Registration(Span(address, nbytes), layout)
            self._registrations.append(registration)
            return registration


class IgnoresUnregistration(MemoryDriver):
    def _registered(self, span: Span) -> bool:
        return any(span.within(r.span) for r in self._registrations)


class SecondUnregisterRaises(MemoryDriver):
    def unregister(self, handle: RegionHandle) -> None:
        with self._lock:
            if not self._registration(handle).live:
                raise DriverRejected("already unregistered")
        super().unregister(handle)


class UnregistersTheRegionOfAnUnprovableOperation(MemoryDriver):
    def _check_region_idle(self, registration: _Registration) -> None:
        if any(
            not op.access_ended
            and not op.unprovable
            and any(span.within(registration.span) for row in op.rows for span in row.io.spans)
            for op in self._ops
        ):
            raise DriverRejected("an operation may still access the region")


class DuplicatePutIsAFault(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        if self.store.presence(io.key) is ProbeStatus.HIT:
            return faulted("already exists")
        return super()._put_row(io)


class CallsAHeldPutReady(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        result = super()._put_row(io)
        return _READY if result.status is RowStatus.NOT_READY else result


class ProbeCallsAHeldKeyAMiss(MemoryDriver):
    def probe(self, keys: Sequence[str], budget_s: float, hint: Hint = None):
        answers = super().probe(keys, budget_s, hint)
        return tuple(ProbeStatus.MISS if a is ProbeStatus.NOT_READY else a for a in answers)


class ProbeCallsAMissAHit(MemoryDriver):
    def probe(self, keys: Sequence[str], budget_s: float, hint: Hint = None):
        answers = super().probe(keys, budget_s, hint)
        return tuple(ProbeStatus.HIT if a is ProbeStatus.MISS else a for a in answers)


class SplitsAMultiSpanRowEvenly(MemoryDriver):
    """Scatters a multi-span get as if every span had the same size."""

    def _get_row(self, io: RowIO) -> RowResult:
        if len(io.spans) == 1:
            return super()._get_row(io)
        stored = self.store.get(io.key)
        if stored is not None and len(stored.payload) == io.nbytes:
            share = io.nbytes // len(io.spans)
            for k, span in enumerate(io.spans):
                ctypes.memmove(
                    span.address, stored.payload[k * share : k * share + span.nbytes], span.nbytes
                )
            return _READY
        return super()._get_row(io)


class ResultsChangeAfterTheyConcluded(MemoryDriver):
    def poll(self, op: OpHandle) -> tuple[RowResult, ...]:
        results = super().poll(op)
        polls = self.__dict__.setdefault("_polls", {})
        polls[id(op)] = polls.get(id(op), 0) + 1
        if polls[id(op)] >= 3 and all(r.concluded for r in results):
            return tuple(faulted("changed") if r.status is RowStatus.READY else r for r in results)
        return results


class CancelRewritesConcludedRows(MemoryDriver):
    def _cancel_unstarted(self, op: MemoryOp) -> None:
        for row in op.rows:
            row.published = _CANCELLED
        self._settle_untouched(op)


class CancelSkipsOnChange(MemoryDriver):
    def cancel(self, op: OpHandle) -> None:
        with self._lock:
            if self._cancel_effective:
                self._cancel_unstarted(self._op(op))


class DrainSkipsOnChange(MemoryDriver):
    def drain(self, budget_s: float) -> bool:
        with self._lock:
            self._require_not_closed()
            self._start_draining()
            for op in self._ops:
                if any(_cancellable(row) for row in op.rows):
                    self._cancel_unstarted(op)
            self._lock.wait_for(self._drain_can_return, timeout=budget_s)
            return self._quiet()


class DrainNeedsABudget(MemoryDriver):
    """Turns True only inside a call with a budget, however idle it is: a driver that takes the
    budget as its one chance to wait, or one whose evidence must be newer than each call, which
    no evidence ever is. The backend only ever calls ``drain(0.0)``."""

    def drain(self, budget_s: float) -> bool:
        return super().drain(budget_s) and budget_s > 0


class DrainWaitsForTheRunningGet(MemoryDriver):
    """Waits for every operation to end whatever its budget says: a drain with no budget blocks
    for as long as a get sits in the store. The check fails it from a thread of its own instead
    of hanging behind it."""

    def drain(self, budget_s: float) -> bool:
        return super().drain(threading.TIMEOUT_MAX)


class DrainForcesAnEnd(MemoryDriver):
    """Once its budget ran out, declares the access of every operation still running ended."""

    def drain(self, budget_s: float) -> bool:
        drained = super().drain(budget_s)
        if not drained:
            with self._lock:
                for op in self._ops:
                    op.access_ended = op.access_ended or not op.unprovable
        return drained


class GetLetsACrashEscape(MemoryDriver):
    def _move(self, kind: Kind, io: RowIO) -> RowResult:
        if kind != "get":
            return super()._move(kind, io)
        try:
            return self._get_row(io)
        except StoreError as error:  # only what is not a store error escapes
            return faulted(str(error))


class GetWaitsForTheStore(MemoryDriver):
    """Runs a get on a thread of its own, but returns its handle only once the store answered."""

    def get(
        self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None
    ) -> MemoryOp:
        op = super().get(rows, hint, on_change=on_change)
        while not all(row.published.concluded for row in op.rows):
            time.sleep(0.001)
        return op


class IgnoresTheRowLimit(MemoryDriver):
    def _check_row_limit(self, rows: Sequence[RowIO]) -> None:
        pass


class IgnoresTheDirection(MemoryDriver):
    def _check_direction(self, kind: Kind) -> None:
        pass


class RegistersAnyMemoryKind(MemoryDriver):
    def _check_region_kind(self, layout: RegionLayout) -> None:
        pass


class NeverEndsAccess(MemoryDriver):
    def _execute(self, op: MemoryOp) -> None:
        super()._execute(op)
        with self._lock:
            op.access_ended = False  # the evidence is lost again


class CallsOnChangeBeforeTheChangeShows(MemoryDriver):
    def _run(self, op: MemoryOp, finished: threading.Event | None) -> None:
        self._call_on_change(op)  # nothing has changed yet, and on_change is not called again
        try:
            self._before_execute(op)
            self._execute(op)
        finally:
            if finished is not None:
                finished.set()
                with self._lock:
                    self._lock.notify_all()


class SkipsOnChangeWhenUnprovable(MemoryDriver):
    def mark_unprovable(self, op: MemoryOp) -> None:
        with self._lock:
            self._require_open(op)
            op.unprovable = True
            self._lock.notify_all()


class SkipsOnChangeWhileAccessIsOpen(MemoryDriver):
    def _call_on_change(self, op: MemoryOp) -> None:
        if op.access_ended or op.unprovable:
            super()._call_on_change(op)


class ChangesItsCapabilities(MemoryDriver):
    _flip = False

    @property
    def capabilities(self) -> DriverCapabilities:
        self._flip = not self._flip
        return DriverCapabilities(
            get=True,
            put=True,
            cancel=self._flip,
            probe=True,
            max_rows_per_op=None,
            absent_is_miss=True,
            device_memory=True,
            multi_span=True,
        )


class ProbeTakesItsTime(MemoryDriver):
    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        time.sleep(budget_s + 0.8)
        return super().probe(keys, budget_s, hint)


class ProbeIgnoresTheBudget(MemoryDriver):
    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        return tuple(self.store.presence(key) for key in keys)


class ProbeHidesStoreErrors(MemoryDriver):
    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        try:
            return super().probe(keys, budget_s, hint)
        except StoreError:
            return tuple(ProbeStatus.MISS for _ in keys)


class HoldsTheLockAcrossTheStore(MemoryDriver):
    def _execute(self, op: MemoryOp) -> None:
        with self._lock:
            super()._execute(op)


class UnregistersBusyRegions(MemoryDriver):
    def _check_region_idle(self, registration: object) -> None:
        pass


class CallsNotReadyAFault(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        result = super()._get_row(io)
        return faulted("not ready") if result.status is RowStatus.NOT_READY else result


class CallsDeclinedAFault(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        result = super()._put_row(io)
        return faulted("no space") if result.status is RowStatus.DECLINED else result


class LetsRaisesEscape(MemoryDriver):
    def _move(self, kind: Kind, io: RowIO) -> RowResult:
        try:
            return self._put_row(io) if kind == "put" else self._get_row(io)
        except StoreError as error:
            return faulted(str(error))


class DrainWaitsForEveryone(MemoryDriver):
    def _drain_can_return(self) -> bool:
        return self._quiet()


class KeepsAcceptingWhileDraining(MemoryDriver):
    def _check_not_draining(self) -> None:
        pass


class ClosesAnyTime(MemoryDriver):
    def _check_closable(self) -> None:
        pass


class UnregisterKeepsTheRange(MemoryDriver):
    def unregister(self, handle: RegionHandle) -> None:
        pass


class PutWaitsForTheStore(MemoryDriver):
    """Runs a put on a thread of its own, but returns its handle only once the store answered."""

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> MemoryOp:
        op = super().put(rows, on_change=on_change)
        while not all(row.published.concluded for row in op.rows):
            time.sleep(0.001)
        return op


class CancelWaitsForTheTransfer(MemoryDriver):
    def cancel(self, op: OpHandle) -> None:
        with self._lock:
            self._lock.wait_for(lambda: self._op(op).access_ended)
        super().cancel(op)


class CancelsStartedRows(MemoryDriver):
    def _cancel_unstarted(self, op: MemoryOp) -> None:
        for row in op.rows:
            if not row.published.concluded:
                row.published = _CANCELLED
        self._settle_untouched(op)


class EndsAccessOnceARowStarted(MemoryDriver):
    def access_ended(self, op: OpHandle) -> bool:
        with self._lock:
            recorded = self._op(op)
            return recorded.access_ended or any(row.started for row in recorded.rows)


class CancelEndsAccessWithTheRowInTheStore(MemoryDriver):
    """On cancel, a row the store holds is called FAULT and the access over, as if the transfer
    had been torn down; the transfer in the store goes on regardless."""

    def __init__(self, store: MemoryStore, **options: Any) -> None:
        super().__init__(store, **options)
        self._torn_down: set[str] = set()

    def cancel(self, op: OpHandle) -> None:
        with self._lock:
            recorded = self._op(op)
            if self._cancel_effective:
                self._cancel_unstarted(recorded)  # as a sound cancel does
            for row in recorded.rows:
                if row.started and not row.published.concluded:
                    row.published = faulted("torn down")
                    self._torn_down.add(row.io.key)
            recorded.access_ended = True
            self._lock.notify_all()
        self._call_on_change(recorded)

    def _move(self, kind: Kind, io: RowIO) -> RowResult:
        result = super()._move(kind, io)
        return faulted("torn down") if io.key in self._torn_down else result


LIMITED = DriverCapabilities(
    get=True, put=True, cancel=True, probe=True, max_rows_per_op=1, absent_is_miss=True
)
GET_ONLY = DriverCapabilities(
    get=True, put=False, cancel=True, probe=True, max_rows_per_op=None, absent_is_miss=True
)
PUT_ONLY = DriverCapabilities(
    get=False, put=True, cancel=True, probe=True, max_rows_per_op=None, absent_is_miss=True
)
HOST_ONLY = DriverCapabilities(
    get=True,
    put=True,
    cancel=True,
    probe=True,
    max_rows_per_op=None,
    absent_is_miss=True,
    multi_span=True,
)

# rule -> (driver breaking it, the capabilities to run with, the conformance check that must fail)
BROKEN_CONFORMANCE: dict[str, tuple[type[MemoryDriver], DriverCapabilities, str]] = {
    "rejection happens before any effect": (
        TouchesMemoryBeforeChecking,
        DEFAULT_CAPABILITIES,
        "test_a_row_outside_every_registration_is_rejected_before_any_effect",
    ),
    "a batch's extents are checked, not only its rows": (
        ChecksRowsNotExtents,
        DEFAULT_CAPABILITIES,
        "test_a_batch_whose_extent_leaves_its_registration_is_rejected_before_any_effect",
    ),
    "a span is checked to its end": (
        ChecksOnlyTheStartOfASpan,
        DEFAULT_CAPABILITIES,
        "test_a_span_leaving_or_outliving_its_registration_is_rejected",
    ),
    "a miss leaves the destination alone": (
        MissTouchesTheDestination,
        DEFAULT_CAPABILITIES,
        "test_get_reports_hit_hit_miss_and_leaves_the_miss_untouched",
    ),
    "a larger object is not truncated": (
        TruncatesLargerObjects,
        DEFAULT_CAPABILITIES,
        "test_an_object_larger_than_its_row_is_a_fault_not_a_truncated_ready",
    ),
    "a larger object is not truncated into spans": (
        TruncatesIntoMultiSpanRows,
        DEFAULT_CAPABILITIES,
        "test_a_larger_object_is_not_truncated_into_a_multi_span_row",
    ),
    "a smaller object is a fault": (
        CallsAWrongSizeAMiss,
        DEFAULT_CAPABILITIES,
        "test_an_object_smaller_than_its_row_is_a_fault",
    ),
    "a failed read is a fault": (
        CallsATimeoutAMiss,
        DEFAULT_CAPABILITIES,
        "test_a_failed_store_read_is_a_fault_not_a_miss",
    ),
    "a put leaves its source unchanged": (
        ClobbersItsSource,
        DEFAULT_CAPABILITIES,
        "test_put_then_get_moves_whole_rows_and_never_aliases_the_source",
    ),
    "a put never damages an existing object": (
        OverwritesExistingObjects,
        DEFAULT_CAPABILITIES,
        "test_a_put_under_an_existing_key_leaves_the_object_whole",
    ),
    "an operation stays within its rows": (
        WritesTheWholeRegion,
        DEFAULT_CAPABILITIES,
        "test_an_operation_leaves_the_rest_of_its_region_alone",
    ),
    "spans keep the order given": (
        OrdersSpansByAddress,
        DEFAULT_CAPABILITIES,
        "test_a_multi_span_row_is_stored_and_fetched_back_to_back",
    ),
    "the row limit is enforced": (
        IgnoresTheRowLimit,
        LIMITED,
        "test_rows_beyond_the_declared_limit_are_rejected",
    ),
    "an undeclared direction is rejected": (
        IgnoresTheDirection,
        GET_ONLY,
        "test_an_undeclared_direction_is_rejected",
    ),
    "a device region needs device_memory": (
        RegistersAnyMemoryKind,
        HOST_ONLY,
        "test_a_device_region_is_rejected_unless_device_memory_is_declared",
    ),
    "access ends with evidence": (
        NeverEndsAccess,
        DEFAULT_CAPABILITIES,
        "test_results_are_stable_and_access_ends_with_evidence",
    ),
    "on_change follows the change it tells of": (
        CallsOnChangeBeforeTheChangeShows,
        DEFAULT_CAPABILITIES,
        "test_on_change_is_called_after_each_operation_settles",
    ),
    "on_change tells of rows concluded before access ends": (
        SkipsOnChangeWhileAccessIsOpen,
        DEFAULT_CAPABILITIES,
        "test_on_change_is_called_once_every_row_concluded_while_access_stays_open",
    ),
    "on_change tells of an access turned unprovable": (
        SkipsOnChangeWhenUnprovable,
        DEFAULT_CAPABILITIES,
        "test_an_unprovable_get_calls_on_change_keeps_its_region_and_drain_returns_at_once",
    ),
    "capabilities are fixed": (
        ChangesItsCapabilities,
        DEFAULT_CAPABILITIES,
        "test_capabilities_do_not_change",
    ),
    "probe returns within its budget": (
        ProbeTakesItsTime,
        DEFAULT_CAPABILITIES,
        "test_probe_returns_within_its_budget",
    ),
    "probe answers unknown for a blocked key": (
        ProbeIgnoresTheBudget,
        DEFAULT_CAPABILITIES,
        "test_probe_answers_unknown_for_a_key_the_store_does_not_answer_in_time",
    ),
    "probe raises on a store error": (
        ProbeHidesStoreErrors,
        DEFAULT_CAPABILITIES,
        "test_probe_raises_on_a_store_error_rather_than_answering",
    ),
    "a blocked get delays nothing else": (
        HoldsTheLockAcrossTheStore,
        DEFAULT_CAPABILITIES,
        "test_a_get_blocked_in_the_store_delays_nothing_and_keeps_its_memory",
    ),
    "a busy region is not unregistered": (
        UnregistersBusyRegions,
        DEFAULT_CAPABILITIES,
        "test_a_get_blocked_in_the_store_delays_nothing_and_keeps_its_memory",
    ),
    "an unfinished write is not ready": (
        CallsNotReadyAFault,
        DEFAULT_CAPABILITIES,
        "test_a_key_held_by_an_unfinished_write_is_not_ready_for_get_put_and_probe",
    ),
    "no room is declined": (
        CallsDeclinedAFault,
        DEFAULT_CAPABILITIES,
        "test_a_row_with_no_room_is_declined_while_the_others_go_on",
    ),
    "a recorded operation never raises": (
        LetsRaisesEscape,
        PUT_ONLY,  # a one-direction driver: the check must run, not skip
        "test_a_raise_after_the_operation_is_recorded_becomes_a_row_fault",
    ),
    "drain returns at once on an unprovable operation": (
        DrainWaitsForEveryone,
        DEFAULT_CAPABILITIES,
        "test_an_unprovable_get_calls_on_change_keeps_its_region_and_drain_returns_at_once",
    ),
    "drain is one-way": (
        KeepsAcceptingWhileDraining,
        DEFAULT_CAPABILITIES,
        "test_drain_is_one_way",
    ),
    "drain turns true without a budget": (
        DrainNeedsABudget,
        DEFAULT_CAPABILITIES,
        "test_drain_polled_without_a_budget_turns_true_once_nothing_runs",
    ),
    "drain returns with no budget while a get is in the store": (
        DrainWaitsForTheRunningGet,
        DEFAULT_CAPABILITIES,
        "test_drain_polled_without_a_budget_turns_true_once_nothing_runs",
    ),
    "close needs a true drain": (
        ClosesAnyTime,
        DEFAULT_CAPABILITIES,
        "test_close_needs_a_true_drain_and_no_live_registrations",
    ),
    "unregister frees the range": (
        UnregisterKeepsTheRange,
        DEFAULT_CAPABILITIES,
        "test_a_registered_range_cannot_be_registered_again_until_unregistered",
    ),
    "registrations never overlap": (
        AcceptsOverlappingRegistrations,
        DEFAULT_CAPABILITIES,
        "test_a_registered_range_cannot_be_registered_again_until_unregistered",
    ),
    "every span is checked": (
        ChecksOnlyTheFirstSpan,
        DEFAULT_CAPABILITIES,
        "test_every_span_of_a_row_is_checked_against_the_registrations",
    ),
    "an unregistered span is rejected": (
        IgnoresUnregistration,
        DEFAULT_CAPABILITIES,
        "test_a_span_leaving_or_outliving_its_registration_is_rejected",
    ),
    "unregister is idempotent": (
        SecondUnregisterRaises,
        DEFAULT_CAPABILITIES,
        "test_unregister_is_idempotent",
    ),
    "an unprovable operation keeps its region": (
        UnregistersTheRegionOfAnUnprovableOperation,
        DEFAULT_CAPABILITIES,
        "test_an_unprovable_get_calls_on_change_keeps_its_region_and_drain_returns_at_once",
    ),
    "a duplicate put is ready": (
        DuplicatePutIsAFault,
        DEFAULT_CAPABILITIES,
        "test_a_duplicate_put_of_a_committed_object_is_ready",
    ),
    "a put of a held key is not ready": (
        CallsAHeldPutReady,
        DEFAULT_CAPABILITIES,
        "test_a_key_held_by_an_unfinished_write_is_not_ready_for_get_put_and_probe",
    ),
    "a probe of a held key is not ready": (
        ProbeCallsAHeldKeyAMiss,
        DEFAULT_CAPABILITIES,
        "test_a_key_held_by_an_unfinished_write_is_not_ready_for_get_put_and_probe",
    ),
    "probe tells hit from miss": (
        ProbeCallsAMissAHit,
        DEFAULT_CAPABILITIES,
        "test_probe_reports_hit_and_miss",
    ),
    "spans of a row keep their sizes": (
        SplitsAMultiSpanRowEvenly,
        DEFAULT_CAPABILITIES,
        "test_a_row_may_cover_parts_of_slots_across_regions",
    ),
    "results never change once concluded": (
        ResultsChangeAfterTheyConcluded,
        DEFAULT_CAPABILITIES,
        "test_results_are_stable_and_access_ends_with_evidence",
    ),
    "cancel leaves concluded rows alone": (
        CancelRewritesConcludedRows,
        DEFAULT_CAPABILITIES,
        "test_cancel_after_conclusion_changes_nothing",
    ),
    "a recorded get never raises": (
        GetLetsACrashEscape,
        GET_ONLY,  # a one-direction driver: the check must run, not skip
        "test_a_raise_after_the_operation_is_recorded_becomes_a_row_fault",
    ),
}

# The same, for checks that can only ask a driver that returns the handle before the store
# answers; the pairing test runs these against the asynchronous in-memory driver.
BROKEN_ASYNC_CONFORMANCE: dict[str, tuple[type[MemoryDriver], DriverCapabilities, str]] = {
    "put never waits for the store": (
        PutWaitsForTheStore,
        DEFAULT_CAPABILITIES,
        "test_a_put_returns_its_handle_before_the_store_answers",
    ),
    "cancel never waits for the store": (
        CancelWaitsForTheTransfer,
        DEFAULT_CAPABILITIES,
        "test_cancel_of_a_get_inside_the_store_neither_waits_nor_lies",
    ),
    "a started row is never cancelled": (
        CancelsStartedRows,
        DEFAULT_CAPABILITIES,
        "test_cancel_of_a_get_inside_the_store_neither_waits_nor_lies",
    ),
    "access does not end with a row in the store": (
        CancelEndsAccessWithTheRowInTheStore,
        DEFAULT_CAPABILITIES,
        "test_cancel_of_a_get_inside_the_store_neither_waits_nor_lies",
    ),
    "access stays open while a row is in the store": (
        EndsAccessOnceARowStarted,
        DEFAULT_CAPABILITIES,
        "test_a_get_blocked_in_the_store_delays_nothing_and_keeps_its_memory",
    ),
    "get never waits for the store": (
        GetWaitsForTheStore,
        DEFAULT_CAPABILITIES,
        "test_a_get_blocked_in_the_store_delays_nothing_and_keeps_its_memory",
    ),
    "drain never forces an end": (
        DrainForcesAnEnd,
        DEFAULT_CAPABILITIES,
        "test_a_get_blocked_in_the_store_delays_nothing_and_keeps_its_memory",
    ),
    "a cancel tells of the operation it ended": (
        CancelSkipsOnChange,
        DEFAULT_CAPABILITIES,
        "test_a_cancel_that_ends_an_operation_calls_its_on_change",
    ),
    "a drain tells of the operations it ended": (
        DrainSkipsOnChange,
        DEFAULT_CAPABILITIES,
        "test_a_drain_that_ends_an_operation_calls_its_on_change",
    ),
}


# Drivers breaking a rule of the optional RegistersRegionsTogether protocol.


class KeepsTheBatchAfterAFailure(TogetherMemoryDriver):
    """A failed ``register_regions`` that certifies nothing is registered while the store kept
    the batch: the memory is now registered with nothing to revoke it by."""

    def register_regions(self, specs: Sequence[RegionSpec]) -> _Batch:
        try:
            return super().register_regions(specs)
        except RegistrationError as error:
            if error.handle is None:
                with self._lock:
                    self._keep(_Batch([_Registration(spec.span, spec.layout) for spec in specs]))
            raise


class RevokesABatchOnlyOnce(TogetherMemoryDriver):
    def unregister_regions(self, handle: RegionHandle) -> None:
        with self._lock:
            if not self._batch(handle).live:
                raise StoreError("the batch was already revoked")
        super().unregister_regions(handle)


class RevocationKeepsTheRange(TogetherMemoryDriver):
    def unregister_regions(self, handle: RegionHandle) -> None:
        with self._lock:
            batch = self._batch(handle)
            for registration in batch.registrations:
                self._check_region_idle(registration)
            batch.live = False  # its registrations stay live: the range is never freed


class RegistersOnlyTheFirstRegion(TogetherMemoryDriver):
    def register_regions(self, specs: Sequence[RegionSpec]) -> _Batch:
        return super().register_regions(specs[:1])


class RevokesBusyBatches(TogetherMemoryDriver):
    def _check_region_idle(self, registration: _Registration) -> None:
        pass


class RevokesTheIdleRegionsOfABusyBatch(TogetherMemoryDriver):
    """Refuses a busy batch, but has already let its idle regions go."""

    def unregister_regions(self, handle: RegionHandle) -> None:
        with self._lock:
            busy = 0
            for registration in self._batch(handle).registrations:
                try:
                    self._check_region_idle(registration)
                except DriverRejected:
                    busy += 1
                else:
                    registration.live = False
            if busy:
                raise DriverRejected(f"{busy} region(s) of the batch are busy")
        super().unregister_regions(handle)


class RevokesTheIdleRegionsOfABusyBatchOneAtATime(RevokesTheIdleRegionsOfABusyBatch):
    """The same, in a driver that takes one batch at a time: it refuses a new registration
    whatever the idle regions' state, so only a row moved through them tells they are gone."""

    def register_regions(self, specs: Sequence[RegionSpec]) -> _Batch:
        with self._lock:
            if self.live_batches():
                raise DriverRejected("one batch at a time")
        return super().register_regions(specs)


class UnregisterRegionsWaitsForTheTransfer(TogetherMemoryDriver):
    """Revokes a busy batch once it is idle, instead of refusing: a caller holding something the
    transfer needs is stuck."""

    def unregister_regions(self, handle: RegionHandle) -> None:
        with self._lock:
            self._lock.wait_for(lambda: all(op.access_ended for op in self._ops))
        super().unregister_regions(handle)


# The same, for the checks of RegistersRegionsTogether; the pairing test runs these against the
# in-memory driver that registers in batches.
BROKEN_TOGETHER_CONFORMANCE: dict[
    str, tuple[type[TogetherMemoryDriver], DriverCapabilities, str]
] = {
    "a failed batch certifies only what is true": (
        KeepsTheBatchAfterAFailure,
        DEFAULT_CAPABILITIES,
        "test_a_failed_batch_registration_is_revocable_or_left_nothing_behind",
    ),
    "unregister_regions is idempotent": (
        RevokesABatchOnlyOnce,
        DEFAULT_CAPABILITIES,
        "test_regions_registered_together_are_revoked_together_and_the_range_freed",
    ),
    "unregister_regions frees the range": (
        RevocationKeepsTheRange,
        DEFAULT_CAPABILITIES,
        "test_regions_registered_together_are_revoked_together_and_the_range_freed",
    ),
    "every region of a batch is registered": (
        RegistersOnlyTheFirstRegion,
        DEFAULT_CAPABILITIES,
        "test_regions_registered_together_are_revoked_together_and_the_range_freed",
    ),
    "a busy batch is not revoked": (
        RevokesBusyBatches,
        DEFAULT_CAPABILITIES,
        "test_a_batch_is_not_revoked_while_any_of_its_regions_is_busy",
    ),
    "a busy batch keeps its idle regions": (
        RevokesTheIdleRegionsOfABusyBatch,
        DEFAULT_CAPABILITIES,
        "test_a_batch_is_not_revoked_while_any_of_its_regions_is_busy",
    ),
    "a busy batch keeps its idle regions, one batch at a time": (
        RevokesTheIdleRegionsOfABusyBatchOneAtATime,
        DEFAULT_CAPABILITIES,
        "test_a_batch_is_not_revoked_while_any_of_its_regions_is_busy",
    ),
    "unregister_regions never waits for the store": (
        UnregisterRegionsWaitsForTheTransfer,
        DEFAULT_CAPABILITIES,
        "test_a_batch_is_not_revoked_while_any_of_its_regions_is_busy",
    ),
}
