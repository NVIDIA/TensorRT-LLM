# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A store driver over Mooncake's distributed store, pinned to SDK 0.3.13.

Zero copy: each row is one object holding exactly the row's payload, written from and read into the
lender's registered host slots. A put starts a put session per key (the master reserves the
object), writes every row in one scatter, then ends the sessions of rows written whole (their
objects become readable) and revokes the rest. A get is one ``batch_get_into``; get sessions would
add two SDK calls, and every call waits for the GIL on return, which a busy executor thread makes
cost milliseconds. Every SDK return code is translated here and nowhere else.

When access ends differs by call. The put scatter (``batch_put_from_multi_buffer_ranges``) waits,
without a timeout, until every fragment it published is physically finished, so its return ends
access whatever it answers, TRANSFER_FAIL included. ``batch_get_into`` waits on one transfer
future per key, and a future can give up with TRANSFER_FAIL (after 60 s, on a status error, after
a partly posted RDMA batch) while its transfer still runs; every other answer comes after all
futures finished or from a step that submitted nothing. So a get ends access only when no answer
is TRANSFER_FAIL. In both, a raise or a wrong answer count leaves access unproven. Session,
existence and end calls touch master and local metadata, never row memory. This holds while the
wheel is built without TENT and calls pass the default config (one memory replica, no NoF), on
RDMA; the TCP transport's scatter completion was not checked line by line.

Every operation settles before ``put`` or ``get`` returns, so one whose access did not end then
never will by anything the driver can wait on: it is unprovable, and it keeps its regions from
being unregistered and the driver from draining.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from tensorrt_llm.logger import logger

from .store import (
    DECLINED,
    MISS,
    NOT_READY,
    PENDING,
    READY,
    DriverCapabilities,
    DriverRejected,
    Hint,
    ProbeStatus,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreError,
    faulted,
)

__all__ = ["MooncakeDriver"]


class _Code(IntEnum):
    """The SDK return codes the driver names (``mooncake-store/include/types.h``)."""

    OK = 0
    INTERNAL_ERROR = -1
    NO_AVAILABLE_HANDLE = -200
    INVALID_PARAMS = -600
    ILLEGAL_CLIENT = -601
    INVALID_WRITE = -700
    INVALID_REPLICA = -702
    REPLICA_IS_NOT_READY = -703
    OBJECT_NOT_FOUND = -704
    OBJECT_ALREADY_EXISTS = -705
    LEASE_EXPIRED = -707
    TRANSFER_FAIL = -800
    RPC_FAIL = -900
    RPC_TIMEOUT = -901
    CALL_FAILED = -100_000  # not an SDK code: a metadata call raised or miscounted (logged)


OBJECT_EXISTS, OBJECT_ABSENT = 1, 0  # batch_is_exist answers
MAX_ROW_BYTES = (1 << 31) - 1  # the put scatter answers each row's bytes as a 32-bit int
_CAPABILITIES = DriverCapabilities(
    get=True, put=True, cancel=False, probe=False, max_rows_per_op=None
)
# Why a row moved nothing and went wrong in nothing, and its result; ``stats()`` counts by both.
_UNTAKEN = {
    # get: no complete replica, as while it is written or once its replica's endpoint went stale
    "being_written": NOT_READY,
    "written_here": NOT_READY,  # put: this client holds the key's put session
    "written_elsewhere": NOT_READY,  # put: the key exists, but the master shows it uncommitted
    "pool_full": DECLINED,  # put: no space for the object
}
_PUT_START_UNTAKEN = {_Code.NO_AVAILABLE_HANDLE: "pool_full", _Code.INVALID_PARAMS: "written_here"}
_WHOLE_OBJECT = [0]  # the source or destination offset of a row's only range


@dataclass(eq=False)
class _Registration:
    span: Span
    live: bool = True


@dataclass(eq=False)
class _Op:
    """One settled put or get: its results and access end are known when the call returns."""

    results: tuple[RowResult, ...]
    access_ended: bool
    owner: MooncakeDriver


@dataclass(eq=False)
class _Rows:
    """One operation's rows in the SDK's terms, and their results so far."""

    keys: list[str]
    addresses: list[int]
    sizes: list[int]
    regions: list[_Registration]  # the registrations the rows lie in
    results: list[RowResult]
    untaken: dict[int, str]  # row -> a key of _UNTAKEN

    def untake(self, row: int, why: str) -> None:
        self.results[row] = _UNTAKEN[why]
        self.untaken[row] = why

    def fault(self, rows: Sequence[int], reason: str) -> None:
        for row in rows:
            self.results[row] = faulted(reason)

    def pick(self, rows: Sequence[int]) -> tuple[list[str], list[int], list[int]]:
        """The keys, addresses and sizes of ``rows``."""
        return (
            [self.keys[i] for i in rows],
            [self.addresses[i] for i in rows],
            [self.sizes[i] for i in rows],
        )


class _WrongAnswerCount(Exception):
    """The SDK answered for a different number of keys than it was given; its state is unknown."""


class MooncakeDriver:
    """A ``StoreDriver`` over one set-up ``mooncake.store.MooncakeDistributedStore``.

    The SDK keys put sessions by name, per client, not per caller, so the driver ends or revokes
    only the put sessions it started; one it cannot end or revoke is parked for ``drain``.

    Args:
        store: An SDK store whose ``setup`` returned 0.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        # Guards the fields below and wakes ``drain``; never held across an SDK call.
        self._lock = threading.Condition()
        self._registrations: list[_Registration] = []
        self._draining = False
        self._closed = False
        self._parked: set[str] = set()
        self._in_flight = 0  # put/get calls past admission and not yet returned
        self._unprovable = 0  # operations whose access never ended
        self._busy: Counter[_Registration] = Counter()  # operations whose access may touch it
        self._untaken: Counter[str] = Counter()

    @property
    def capabilities(self) -> DriverCapabilities:
        return _CAPABILITIES

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> _Registration:
        try:
            span = Span(address, nbytes)
        except ValueError as error:
            raise DriverRejected(str(error)) from None
        if layout.nbytes > nbytes:
            raise DriverRejected(f"layout needs {layout.nbytes} bytes, region has {nbytes}")
        with self._lock:
            self._require_open(DriverRejected)
            overlapping = [r.span for r in self._registrations if r.live and r.span.overlaps(span)]
        if overlapping:
            raise DriverRejected(f"region {span} overlaps a live registration {overlapping[0]}")
        registration = _Registration(span)
        try:
            code = self._store.register_buffer(address, nbytes)
        except Exception as error:
            self._add(registration)
            raise RegistrationError(f"register_buffer raised {error!r}", registration) from error
        if code != _Code.OK:  # the SDK rolled its partial registration back
            raise RegistrationError(f"register_buffer returned {_describe(code)}")
        self._add(registration)
        return registration

    def unregister(self, handle: _Registration) -> None:
        with self._lock:
            registration = self._registration(handle)
            if not registration.live:
                return
            if self._busy[registration]:
                raise DriverRejected(
                    f"{self._busy[registration]} operation(s) may still access the region"
                )
            registration.live = False  # admits no new rows while the SDK revokes it
        try:
            code = self._store.unregister_buffer(registration.span.address)
        except Exception as error:
            failure = StoreError(f"unregister_buffer raised {error!r}")
        else:
            if code == _Code.OK:
                return
            failure = StoreError(f"unregister_buffer returned {_describe(code)}")
        with self._lock:
            registration.live = True  # still registered: the call may be retried
        raise failure

    def put(self, rows: Sequence[RowIO]) -> _Op:
        batch, started = self._admit(rows), time.monotonic()  # the only raise is in _admit
        try:
            ended = self._put_rows(batch)
        except Exception as error:  # a fault past the SDK calls: keep the rows' memory held
            batch.fault(range(len(batch.keys)), f"put failed: {error!r}")
            ended = False
        return self._conclude("put", batch, ended, time.monotonic() - started)

    def get(self, rows: Sequence[RowIO], hint: Hint = None) -> _Op:
        if hint is not None:
            raise DriverRejected("this driver reads no hints")
        batch, started = self._admit(rows), time.monotonic()  # the only raise is in _admit
        try:
            ended = self._get_rows(batch, list(range(len(batch.keys))), reread=True)
        except Exception as error:  # a fault past the SDK calls: keep the rows' memory held
            batch.fault(range(len(batch.keys)), f"get failed: {error!r}")
            ended = False
        return self._conclude("get", batch, ended, time.monotonic() - started)

    def poll(self, op: _Op) -> tuple[RowResult, ...]:
        return self._op(op).results

    def access_ended(self, op: _Op) -> bool:
        return self._op(op).access_ended

    def access_unprovable(self, op: _Op) -> bool:
        return not self._op(op).access_ended  # it settled when the call returned

    def cancel(self, op: _Op) -> None:
        """Never called: the SDK has no cancel, so ``capabilities.cancel`` is false."""
        self._op(op)

    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        """Never called: ``capabilities.probe`` is false."""
        self._require_open()
        return tuple(ProbeStatus.UNKNOWN for _ in keys)

    def drain(self, budget_s: float) -> bool:
        """Refuses new work for good and waits up to ``budget_s`` for the calls in progress;
        False at once while an operation is unprovable. When it returns True it also revokes the
        parked put sessions once: no put runs any more, so none is in use. They hold no lender
        memory, so they never change the answer, and one whose revocation fails stays open until
        the master discards it."""
        with self._lock:
            self._require_open()
            self._draining = True
            self._lock.wait_for(lambda: self._unprovable or not self._in_flight, budget_s)
            quiet = not (self._in_flight or self._unprovable)
            parked = sorted(self._parked) if quiet else []
        if parked:
            codes = self._metadata_call(self._store.batch_put_session_revoke, parked)
            with self._lock:
                self._parked.difference_update(k for k, code in zip(parked, codes) if _gone(code))
        return quiet

    def close(self) -> None:
        """Closes the SDK client once a drain returned True and every region is unregistered.
        Never called in production: it holds the GIL through unmount RPCs; process exit frees it."""
        with self._lock:
            if self._closed:
                return
            if not self._draining or self._in_flight or self._unprovable:
                raise DriverRejected("close needs a drain that returned True")
            if any(r.live for r in self._registrations):
                raise DriverRejected("close needs every registration unregistered")
            self._closed = True
        self._store.close()

    def stats(self) -> dict[str, int]:
        """A snapshot for logs and tests; not part of the driver protocol."""
        with self._lock:
            return {
                **{
                    f"rows_{r.status.value}_{why}": self._untaken[why]
                    for why, r in _UNTAKEN.items()
                },
                "ops_unprovable": self._unprovable,
                "put_sessions_parked": len(self._parked),
            }

    # Put: start, one scatter, end what was written whole, revoke the rest.

    def _put_rows(self, batch: _Rows) -> bool:
        """Returns whether access ended."""
        starts = self._metadata_call(self._store.batch_put_session_start, batch.keys, batch.sizes)
        started, existing = [i for i, code in enumerate(starts) if code == _Code.OK], []
        unstarted = starts if len(started) < len(starts) else ()  # usually every row started
        for row, code in enumerate(unstarted):
            if code == _Code.OBJECT_ALREADY_EXISTS:
                existing.append(row)
            elif code in _PUT_START_UNTAKEN:
                batch.untake(row, _PUT_START_UNTAKEN[code])
            elif code != _Code.OK:
                batch.fault([row], f"put session start: {_describe(code)}")
        ended = self._write_and_close(batch, started) if started else True
        if existing:
            self._confirm_existing(batch, existing)
        return ended

    def _write_and_close(self, batch: _Rows, started: list[int]) -> bool:
        """Writes the rows whose sessions this put started and ends, revokes or parks each of
        those sessions. Returns whether access ended."""
        keys, addresses, sizes = batch.pick(started)
        ranges = ([[a] for a in addresses], [[n] for n in sizes], [_WHOLE_OBJECT] * len(keys))
        try:
            write = self._store.batch_put_from_multi_buffer_ranges
            answers = _answers(write(keys, *ranges), len(keys))
        except Exception as error:  # the scatter may still be reading the slots
            batch.fault(started, f"ranges raised {error!r}")
            self._park(keys)  # no end, no revoke
            return False
        whole = answers == sizes  # usually every row was written whole
        written, unwritten = (started, []) if whole else ([], [])
        for i, answer in zip(started, () if whole else answers):
            if answer == batch.sizes[i]:
                written.append(i)
            else:  # its object must never be marked complete
                unwritten.append(i)
                batch.fault([i], f"put ranges: {_moved(answer, batch.sizes[i])}")
        not_ended = self._end_put_sessions(batch, written)
        self._revoke_put_sessions([batch.keys[i] for i in unwritten + not_ended])
        return True

    def _end_put_sessions(self, batch: _Rows, written: list[int]) -> list[int]:
        """Ends the sessions of rows written whole; returns the rows whose session did not end."""
        if not written:
            return []
        end = self._store.batch_put_session_end
        keys = [batch.keys[i] for i in written]
        codes = self._metadata_call(end, keys)
        retry = [
            key for key, code in zip(keys, codes) if code in (_Code.RPC_FAIL, _Code.RPC_TIMEOUT)
        ]
        if retry:  # ending is idempotent on the master
            again = dict(zip(retry, self._metadata_call(end, retry)))
            codes = [again.get(key, code) for key, code in zip(keys, codes)]
        for i, code in zip(written, codes):
            batch.results[i] = (
                READY if code == _Code.OK else faulted(f"put session end: {_describe(code)}")
            )
        return [i for i, code in zip(written, codes) if code != _Code.OK]

    def _revoke_put_sessions(self, keys: list[str]) -> None:
        if keys:
            codes = self._metadata_call(self._store.batch_put_session_revoke, keys)
            self._park([key for key, code in zip(keys, codes) if not _gone(code)])

    def _park(self, keys: list[str]) -> None:
        if keys:
            logger.warning(f"mooncake: {len(keys)} put sessions left open, first {keys[0]}")
            with self._lock:
                self._parked.update(keys)

    def _confirm_existing(self, batch: _Rows, existing: list[int]) -> None:
        """A key that already exists is readable only once the master shows it committed."""
        keys = [batch.keys[i] for i in existing]
        for i, answer in zip(existing, self._metadata_call(self._store.batch_is_exist, keys)):
            if answer == OBJECT_EXISTS:
                batch.results[i] = READY
            elif answer == OBJECT_ABSENT:
                batch.untake(i, "written_elsewhere")
            else:
                batch.fault([i], f"existence check: {_describe(answer)}")

    # Get: one batch_get_into; rows whose read lease expired mid-read are read once more.

    def _get_rows(self, batch: _Rows, rows: list[int], reread: bool) -> bool:
        """Returns whether access ended."""
        keys, addresses, sizes = batch.pick(rows)
        try:
            answers = _answers(self._store.batch_get_into(keys, addresses, sizes), len(rows))
        except Exception as error:  # the transfers may still be writing the slots
            batch.fault(rows, f"batch_get_into raised {error!r}")
            return False
        if answers == sizes:  # usually every row was read whole
            for i in rows:
                batch.results[i] = READY
            return True
        for i, answer in zip(rows, answers):
            if answer == _Code.OBJECT_NOT_FOUND:
                batch.results[i] = MISS
            elif answer == _Code.REPLICA_IS_NOT_READY:
                batch.untake(i, "being_written")
            else:
                batch.results[i] = _get_result(answer, batch.sizes[i])
        ended = _Code.TRANSFER_FAIL not in answers
        expired = [i for i, answer in zip(rows, answers) if answer == _Code.LEASE_EXPIRED]
        if expired and reread:  # their bytes landed but may be torn: read them once more
            reread_ended = self._get_rows(batch, expired, reread=False)
            ended = ended and reread_ended  # a TRANSFER_FAIL of the first read stays
        return ended

    # Bookkeeping.

    def _admit(self, rows: Sequence[RowIO]) -> _Rows:
        """Checks ``rows`` and records the operation, or raises ``DriverRejected`` having done
        neither."""
        if not rows:
            raise DriverRejected("an operation needs at least one row")
        keys = [row.key for row in rows]
        # The SDK keys put sessions and get targets by name: a repeat would overwrite the other.
        if len(set(keys)) != len(keys):
            raise DriverRejected("a batch repeats a key")
        addresses = [row.payload.address for row in rows]
        sizes = [row.payload.nbytes for row in rows]
        if max(sizes) > MAX_ROW_BYTES:
            row = next(row for row in rows if row.payload.nbytes > MAX_ROW_BYTES)
            raise DriverRejected(f"a row holds at most {MAX_ROW_BYTES} bytes, got {row.payload}")
        low, high = min(addresses), max(map(sum, zip(addresses, sizes)))
        with self._lock:
            self._require_open(DriverRejected)
            if self._draining:
                raise DriverRejected("the driver is draining")
            live = [r for r in self._registrations if r.live]
            whole = next((r for r in live if r.span.address <= low and high <= r.span.end), None)
            regions = [whole] if whole is not None else _regions_of(rows, live)
            self._in_flight += 1
            self._busy.update(regions)
        return _Rows(keys, addresses, sizes, regions, [PENDING] * len(rows), {})

    def _metadata_call(self, call: Callable, keys: list[str], *args) -> list[int]:
        """A session or existence call. It touches no row memory, so a raise or a wrong answer
        count fails only its keys."""
        try:
            return _answers(call(keys, *args), len(keys))
        except Exception as error:
            logger.warning(f"mooncake: {getattr(call, '__name__', call)} failed: {error!r}")
            return [_Code.CALL_FAILED] * len(keys)

    def _conclude(self, direction: str, batch: _Rows, ended: bool, seconds: float) -> _Op:
        with self._lock:
            self._untaken.update(batch.untaken.values())
            self._in_flight -= 1
            if ended:
                self._busy.subtract(batch.regions)
            else:  # holds its regions for good
                self._unprovable += 1
            self._lock.notify_all()
        _report(direction, batch, ended, seconds)
        return _Op(tuple(batch.results), ended, self)

    def _add(self, registration: _Registration) -> None:
        with self._lock:
            self._registrations.append(registration)

    def _registration(self, handle: _Registration) -> _Registration:
        self._require_open()
        if not isinstance(handle, _Registration) or handle not in self._registrations:
            raise ValueError("handle does not belong to this driver")
        return handle

    def _op(self, op: _Op) -> _Op:
        self._require_open()
        if not isinstance(op, _Op) or op.owner is not self:
            raise ValueError("operation does not belong to this driver")
        return op

    def _require_open(self, error: type[Exception] = RuntimeError) -> None:
        """Raises ``error`` once closed: DriverRejected refuses new work, RuntimeError the rest."""
        if self._closed:
            raise error("the driver is closed")


def _regions_of(rows: Sequence[RowIO], live: list[_Registration]) -> list[_Registration]:
    """The registrations ``rows`` lie in, found row by row."""
    regions: list[_Registration] = []
    for row in rows:
        region = next((r for r in live if row.payload.within(r.span)), None)
        if region is None:
            raise DriverRejected(f"{row.payload} is outside every live registration")
        if region not in regions:
            regions.append(region)
    return regions


def _get_result(answer: int, size: int) -> RowResult:
    """A read answers the object's size, which must be the row's."""
    if answer == size:
        return READY
    if answer >= 0:
        return faulted(f"object is {answer} bytes, expected {size}")
    return faulted(f"batch_get_into: {_describe(answer)}")


def _gone(revoke_answer: int) -> bool:
    """Whether a revocation left no session: revoked, or none was left (INVALID_PARAMS)."""
    return revoke_answer in (_Code.OK, _Code.INVALID_PARAMS)


def _moved(answer: int, size: int) -> str:
    return f"moved {answer} of {size} bytes" if answer >= 0 else _describe(answer)


def _answers(answers: Sequence[int], count: int) -> list[int]:
    answers = list(answers)
    if len(answers) != count:
        raise _WrongAnswerCount(f"{len(answers)} answers for {count} keys")
    return answers


def _describe(code: int) -> str:
    try:
        return f"{_Code(code).name}({code})"
    except ValueError:
        return f"code({code})"


def _report(direction: str, batch: _Rows, ended: bool, seconds: float) -> None:
    """One line per call that left rows untaken or its access open; none for a clean call."""
    taken = (RowStatus.READY, RowStatus.MISS)
    untaken = [i for i, result in enumerate(batch.results) if result.status not in taken]
    if ended and not untaken:
        return
    first = ""
    if untaken:
        row = untaken[0]
        result = batch.results[row]
        why = result.fault or f"{result.status.value} ({batch.untaken[row]})"
        first = f"; first {batch.keys[row]}: {why}"
    access = "ended" if ended else "may still be in progress"
    logger.warning(
        f"mooncake {direction}: {len(untaken)} of {len(batch.keys)} rows not taken in "
        f"{seconds:.3f} s{first}; access {access}"
    )
