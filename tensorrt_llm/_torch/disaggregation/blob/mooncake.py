# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A store driver over Mooncake's distributed store, pinned to SDK 0.3.13.

Each row is one object holding exactly the row's payload, written from and read into the lender's
registered host slots without a copy. ``put`` and ``get`` run on the executor loop's thread: they
check a batch with a few comparisons, whatever its size, and record and queue the operation.
``workers`` threads, started with the driver and running until it drains, take the queued rows a
part at a time, at most ``rows_per_call`` rows and ``bytes_per_call`` payload bytes, and make each
part's SDK calls, which block their thread until the store answers. Gets go first, as they sit on
the time-to-first-token path, but once ``GETS_BEFORE_PUT_IS_DUE`` gets have gone ahead of a
waiting put, that put goes next once a put may run. Puts occupy at most ``workers - 1`` threads (at
least one), so with two workers or more a get never waits behind puts alone. Every SDK return code
is translated here and nowhere else.

A part's rows are READY unless an SDK answer says otherwise: a worker compares a call's answers
with what an all-OK call answers, and looks at rows one by one only when they differ.

Put, per part: ``batch_put_session_start`` per key (the master reserves the object), one
``batch_put_from_multi_buffer_ranges`` scatter for all its rows, then ``batch_put_session_end`` for
the rows written whole (their objects become readable) and ``batch_put_session_revoke`` for the
rest. Put uses sessions because this scatter has no timeout, so a failed row is revoked only once
nothing writes into its space; ``batch_put_from`` revokes a timed-out transfer at once, while its
RDMA writes may still land in space the master has given to another object.

Get, per part: one ``batch_get_into``. A get session would add two SDK calls, and every call waits
for the GIL on return, which a busy executor thread makes cost milliseconds.

When access ends. The rules below were read in the Mooncake sources at tag v0.3.13 (paths are in
that tree), for a wheel built without ``USE_TENT``, as the release wheels are
(``mooncake-common/common.cmake:174``, ``.github/workflows/_build-wheel.yaml:67-107``). They hold
only if whoever opens the SDK refuses ``MC_USE_TENT`` and ``MC_USE_TEV1``, which switch to the
newer engine at run time (``mooncake-transfer-engine/src/transfer_engine.cpp:393-410``); the SDK's
local hot cache, which could serve a torn read later; and TCP's legacy framing, which reports a
write before it lands and is also used toward a process of the pool that runs an SDK older than
0.3.12. The default replicate config is assumed: one memory replica, no NoF or SSD replica.

- The put scatter waits, without a timeout, until every task it published is terminal, also when
  submitting failed part way (``transfer_engine.cpp:987-993, 1236-1265, 1315-1319``). An RDMA task
  is FAILED only once all its slices ended (``mooncake-transfer-engine/src/transport/
  rdma_transport/rdma_transport.cpp:1023-1048``), and a TCP v2 write ends, either way, only after
  the body write released the caller's buffer (``mooncake-transfer-engine/src/transport/
  tcp_transport/tcp_transport_session_impl.h:818-905``). So the
  scatter's return ends access to the slots whatever it answers, TRANSFER_FAIL included.
- Over TCP a TRANSFER_FAIL may also mean the initiator gave up on a slow target: no
  acknowledgment within ``MC_TCP_STATUS_TIMEOUT_SEC`` (30 s by default;
  ``tcp_transport_session_impl.h:394-401, 447-470``) or no progress within
  ``MC_TCP_PROGRESS_TIMEOUT_SEC`` (``:404-414, 496-526``), while the target may still apply data
  it received into the object's space (``:838-840``). A revoke frees that space at once
  (``mooncake-store/src/master_service.cpp:4776, 4869``), so such a row's session is never revoked
  by the driver; the master frees a put left unfinished only after
  ``put_start_release_timeout_sec`` (600 s by default; ``master_service.cpp:4441-4450``).
- ``batch_get_into`` waits on one transfer future per key, and a future can give up with
  TRANSFER_FAIL (after 60 s, on a status error, after a partly posted RDMA batch;
  ``mooncake-store/src/transfer_task.cpp:746-905, 1239-1258``) while its transfer still runs;
  every other answer comes after all futures finished or from a step that posted nothing. So a get
  ends access only when no answer is TRANSFER_FAIL.
- In both, a raise or a wrong answer count leaves access unproven. Session, existence and end
  calls touch only master and local metadata, never row memory.

An operation can never prove its access ended once one of its parts returned from its SDK calls
with access not ended: it is ``access_unprovable``, and it keeps its region from being unregistered
and the driver from draining. Its other parts still run: each part's rows stand alone, and a cancel
or drain ends only the rows no worker took yet.
"""

from __future__ import annotations

import enum
import itertools
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, NamedTuple

from tensorrt_llm.logger import logger

from ..base.memory import MemoryKind
from .store import (
    DriverCapabilities,
    DriverRejected,
    Hint,
    OnChange,
    ProbeStatus,
    RegionLayout,
    RegistrationError,
    RowBatch,
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
    CHECKSUM_MISMATCH = -801  # with MOONCAKE_STORE_CHECKSUM: the bytes of a get landed, then failed
    RPC_FAIL = -900
    RPC_TIMEOUT = -901
    # Not an SDK code: the driver's own answer when a metadata call raised or miscounted (logged).
    METADATA_CALL_FAILED = -100_000


OBJECT_EXISTS, OBJECT_ABSENT = 1, 0  # batch_is_exist answers
DEFAULT_WORKERS = 4
DEFAULT_ROWS_PER_CALL = 256
DEFAULT_BYTES_PER_CALL = 128 << 20
# Gets taken while a put waits; then that put goes next, as soon as a put may run.
GETS_BEFORE_PUT_IS_DUE = 4
MAX_ROW_BYTES = (1 << 31) - 1  # the put scatter answers each row's bytes as a 32-bit int
_CAPABILITIES = DriverCapabilities(
    get=True,
    put=True,
    cancel=True,  # ends the rows no worker took; the SDK cannot stop a running call
    probe=False,
    max_rows_per_op=None,  # any number: the driver cuts an operation into parts itself
    # The master answers OBJECT_NOT_FOUND for a key it does not hold; failures keep their own
    # codes and stay FAULT.
    absent_is_miss=True,
    device_memory=False,  # registers host memory only: the lender stages rows there
    # One span per row: batch_get_into reads each object into one buffer; a get of several
    # spans would need batch_get_into_multi_buffers.
    multi_span=False,
)
# The results that carry no fault text, one shared object each.
_PENDING = RowResult(RowStatus.PENDING)
_READY = RowResult(RowStatus.READY)
_MISS = RowResult(RowStatus.MISS)
_NOT_READY = RowResult(RowStatus.NOT_READY)
_DECLINED = RowResult(RowStatus.DECLINED)
_CANCELLED = RowResult(RowStatus.CANCELLED)
_NO_WORKER_LEFT = faulted("no worker thread is left to run the row")
_instance_ids = itertools.count()  # names each driver's worker threads

# Why a row was skipped (nothing moved and nothing went wrong), and the result it gets; ``stats()``
# counts skipped rows by both, and a part's warning names its first row that faulted or was
# skipped, with the skip's reason.
_SKIP_RESULTS = {
    # get -703: no complete replica, as while the object is written or once its replica's
    # endpoint went stale
    "no_complete_replica": _NOT_READY,
    # put: a put of this driver is writing the key now, or a session it left open holds the key and
    # the master shows no complete object; no session is started
    "held_by_own_put": _NOT_READY,
    "uncommitted": _NOT_READY,  # put start -705, and the master shows the object uncommitted
    "pool_full": _DECLINED,  # put start -200: no space for the object
}

# Session-end answers after which the end may still reach the master, even if the retry is refused.
# Such a session is left for close and its key is never put again: the master matches an end by key
# and client only, so a late end would mark complete this client's next object of the key.
_END_OUTCOME_UNKNOWN = (_Code.RPC_FAIL, _Code.RPC_TIMEOUT, _Code.METADATA_CALL_FAILED)
# Session-end answers retried once, as the master ends a session idempotently; an end call that
# raised or miscounted is not retried.
_END_RETRIED = (_Code.RPC_FAIL, _Code.RPC_TIMEOUT)


class _LeftOpen(enum.Enum):
    """Who ends a put session this driver left open."""

    FOR_CLOSE = "for_close"  # close revokes it, as the owner's last step
    TO_MASTER = "to_master"  # the target may still write into its object: the master frees it


@dataclass(eq=False)
class _Registration:
    span: Span
    live: bool = True
    ops_accessing: int = 0  # operations whose access may touch it; unregister refuses it until 0


@dataclass(eq=False)
class _Op:
    """One ``put`` or ``get``; its fields change under the driver's lock."""

    owner: MooncakeDriver
    is_put: bool
    rows: RowBatch
    region: _Registration  # the registration every row lies in
    rows_per_part: int
    results: list[RowResult]  # per row; each part writes its slice when it ends
    # Called once every row concluded (its access ends then too, unless unprovable) and once
    # the operation turned unprovable.
    on_change: OnChange | None
    # Rows from here on wait in the operation's queue; once every row was taken, or ended
    # untaken, it is ``len(rows)`` and the operation has left the queue.
    first_untaken_row: int = 0
    parts_running: int = 0
    # A part returned with its access not proven ended: the region stays busy for good.
    unprovable: bool = False

    @property
    def all_taken(self) -> bool:
        return self.first_untaken_row == len(self.rows)

    @property
    def access_ended(self) -> bool:
        return self.all_taken and self.parts_running == 0 and not self.unprovable


@dataclass(frozen=True, eq=False)
class _Part:
    """Rows ``[start, stop)`` of ``op``: what a worker cut off the operation's queued rows when it
    took them, and runs in one go."""

    op: _Op
    start: int
    stop: int


class _Rows:
    """Rows of one part as SDK calls take them: keys, addresses, and the size every row has."""

    def __init__(self, keys: list[str], addresses: list[int], row_bytes: int) -> None:
        self.keys = keys
        self.addresses = addresses
        self.row_bytes = row_bytes
        # Also what a call answers when it moved every row whole.
        self.sizes = [row_bytes] * len(keys)

    def only(self, keys: list[str]) -> _Rows:
        """The rows of ``keys``, a subset of these in their order."""
        if len(keys) == len(self.keys):
            return self
        address_of = dict(zip(self.keys, self.addresses))
        return _Rows(keys, [address_of[key] for key in keys], self.row_bytes)


class _PartResults:
    """A part's row results, recorded only for the rows that did not end READY, by key (a batch
    never repeats a key), and why each skipped row was skipped (a key of ``_SKIP_RESULTS``)."""

    def __init__(self) -> None:
        self.recorded: dict[str, RowResult] = {}
        self.skip_reasons: dict[str, str] = {}

    @property
    def all_ready(self) -> bool:
        return not self.recorded

    def skip(self, key: str, reason: str) -> None:
        self.recorded[key] = _SKIP_RESULTS[reason]
        self.skip_reasons[key] = reason

    def fault(self, keys: Iterable[str], reason: str) -> None:
        result = faulted(reason)
        for key in keys:
            self.recorded[key] = result

    def in_order(self, keys: list[str]) -> list[RowResult]:
        """The results of ``keys``, READY where none was recorded."""
        if not self.recorded:
            return [_READY] * len(keys)
        return [self.recorded.get(key, _READY) for key in keys]


class _PartEnd(NamedTuple):
    """How a part's SDK calls ended: its rows' results in order, the reasons its rows were skipped,
    and whether their access ended."""

    results: list[RowResult]
    skip_reasons: tuple[str, ...]
    access_ended: bool


class _WrongAnswerCount(Exception):
    """The SDK answered for a different number of keys than it was given; its state is unknown."""


class MooncakeDriver:
    """A ``StoreDriver`` over one set-up ``mooncake.store.MooncakeDistributedStore``. It must be the
    only driver over its SDK client: the SDK names a put session by its key, per client.

    A put of a part's rows:

    1. Reserves each key that no put of this driver is writing and no session it left open holds.
       A key another put is writing is NOT_READY at once. A key whose session it left open is
       READY when the master shows its object complete, NOT_READY when it shows none, and FAULT
       when the existence check fails; it is never put again.
    2. Starts a session per reserved key, writes every started row in one scatter, ends the
       session of each row written whole (its object becomes readable) and revokes the rest.
    3. Leaves open each session it may have started and did not see ended or revoked:

       - to the master, never revoked here, since the target may still write into the object: a
         scatter that raised or miscounted, a row answered TRANSFER_FAIL. The master frees them
         after ``put_start_release_timeout_sec``.
       - for ``close`` to revoke, once no put runs: any other, such as a start that raised or
         miscounted, an end whose outcome is unknown (``_END_OUTCOME_UNKNOWN``), a revoke that
         failed, or a put stopped by a raise.

    Args:
        store: An SDK store whose ``setup`` returned 0. The driver takes it over: ``close``
            closes it.
        workers: The most threads running SDK calls; each blocks while its call waits on the
            store.
        rows_per_call: The most rows of one part, so of each SDK call.
        bytes_per_call: The most payload bytes of one part, so of each SDK call that moves rows;
            a row larger than this goes in a part of its own.
    """

    def __init__(
        self,
        store: Any,
        *,
        workers: int = DEFAULT_WORKERS,
        rows_per_call: int = DEFAULT_ROWS_PER_CALL,
        bytes_per_call: int = DEFAULT_BYTES_PER_CALL,
    ) -> None:
        for name, value in (
            ("workers", workers),
            ("rows_per_call", rows_per_call),
            ("bytes_per_call", bytes_per_call),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        self._store = store
        self._rows_per_call = rows_per_call
        self._bytes_per_call = bytes_per_call
        # Puts occupy at most this many threads, so with two workers or more a get never waits
        # behind puts alone.
        self._max_running_puts = max(1, workers - 1)
        # Guards the fields below and wakes ``drain``; never held across an SDK call or an
        # ``on_change``. Idle workers wait on ``_work_due``, which shares its lock.
        mutex = threading.Lock()
        self._lock = threading.Condition(mutex)
        self._work_due = threading.Condition(mutex)
        # Operations with rows no worker took yet, oldest first.
        self._queued_gets: deque[_Op] = deque()
        self._queued_puts: deque[_Op] = deque()
        self._workers_alive = 0  # worker threads started and not yet exited
        self._running_puts = 0  # put parts taken and not yet ended; at most _max_running_puts
        # Get parts taken while a put waited, since a put part was taken or a put started waiting
        # with none queued; at GETS_BEFORE_PUT_IS_DUE the put is due.
        self._gets_ahead_of_puts = 0
        self._registrations: list[_Registration] = []
        # Set by the first drain, whatever it returns, or when a worker thread fails to start.
        self._draining = False
        # A drain returned True: no part ran, no worker thread was left and none was unprovable.
        self._drained = False
        self._closed = False
        # Keys a running put part reserved, until that part ends: no two puts of this driver hold
        # a session of one key at once, so a -600 is never this driver's own.
        self._keys_being_put: set[str] = set()
        # The put sessions this driver left open (class docstring, step 3), by key. No key is in
        # both: a reserved key is left open only as its reservation ends, and is never reserved
        # again.
        self._sessions_left_open: dict[str, _LeftOpen] = {}
        self._in_flight_op_count = 0  # operations with rows queued or parts running
        self._unprovable_op_count = 0  # operations whose access can never be proven ended
        self._skipped_rows: Counter[str] = Counter()  # by reason
        # Started here, on the builder's thread: starting a thread waits for it to run, which on
        # the executor loop's thread would hand the GIL to every busy thread. If one cannot start,
        # those already started exit, as after a drain.
        name = f"mooncake-driver-{next(_instance_ids)}-worker"
        with self._lock:
            try:
                for _ in range(workers):
                    threading.Thread(target=self._worker_loop, name=name, daemon=True).start()
                    self._workers_alive += 1
            except BaseException:
                self._draining = True
                self._work_due.notify_all()
                raise

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
        if layout.memory_kind is not MemoryKind.HOST:
            raise DriverRejected("this driver registers host memory only")
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
        # The SDK undoes a partial registration before it answers (RdmaTransport::
        # registerLocalMemoryInternal), so a non-OK answer leaves nothing registered; a rollback
        # that itself fails is only logged by the SDK and cannot be seen here.
        if code != _Code.OK:
            raise RegistrationError(f"register_buffer returned {_describe(code)}")
        self._add(registration)
        return registration

    def unregister(self, handle: _Registration) -> None:
        with self._lock:
            registration = self._checked_registration(handle)
            if not registration.live:
                return
            if registration.ops_accessing:
                raise DriverRejected(
                    f"{registration.ops_accessing} operation(s) may still access the region"
                )
            registration.live = False  # admits no new rows while the SDK unregisters it
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

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> _Op:
        return self._admit(rows, is_put=True, on_change=on_change)

    def get(self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None) -> _Op:
        if hint is not None:
            raise DriverRejected("this driver reads no hints")
        return self._admit(rows, is_put=False, on_change=on_change)

    def poll(self, op: _Op) -> tuple[RowResult, ...]:
        with self._lock:
            return tuple(self._checked_op(op).results)

    def access_ended(self, op: _Op) -> bool:
        with self._lock:
            return self._checked_op(op).access_ended

    def access_unprovable(self, op: _Op) -> bool:
        with self._lock:
            return self._checked_op(op).unprovable

    def cancel(self, op: _Op) -> None:
        """Cancels the rows of ``op`` no worker took; the running parts run to their end."""
        with self._lock:
            done = self._end_untaken_rows([self._checked_op(op)], _CANCELLED)
        if done:
            self._call_on_change(op)

    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        """Never called: ``capabilities.probe`` is false."""
        self._require_open()
        return tuple(ProbeStatus.UNKNOWN for _ in keys)

    def drain(self, budget_s: float) -> bool:
        """Refuses new work for good, cancels the rows no worker took, ends the worker threads,
        and waits up to ``budget_s`` for the running parts to end and every worker thread to exit.
        Returns False at once while an operation is unprovable. Makes no SDK call, so it returns
        within ``budget_s``: the sessions left for close hold no lender memory, and ``close``
        revokes them."""
        with self._lock:
            self._require_open()
            self._draining = True
            done = self._end_untaken_rows(self._queued_ops(), _CANCELLED)  # none starts now
            self._work_due.notify_all()  # idle workers exit
        for op in done:
            self._call_on_change(op)
        with self._lock:
            self._lock.wait_for(
                lambda: self._unprovable_op_count
                or not (self._in_flight_op_count or self._workers_alive),
                min(max(0.0, budget_s), threading.TIMEOUT_MAX),
            )
            drained = not (
                self._in_flight_op_count or self._unprovable_op_count or self._workers_alive
            )
            if drained:
                self._drained = True
            return drained

    def close(self) -> None:
        """The owner's last step, once a drain returned True and every region is unregistered:
        tries once to revoke the sessions left for close, then closes the SDK client. Both wait on
        the master and the unmount with the GIL held, so the owner calls it off the executor loop's
        thread. A session whose revocation fails is freed by the master after
        ``put_start_release_timeout_sec``; sessions left to the master are never revoked here."""
        with self._lock:
            if self._closed:
                return
            if not self._drained:
                raise DriverRejected("close needs a drain that returned True")
            if any(r.live for r in self._registrations):
                raise DriverRejected("close needs every registration unregistered")
            self._closed = True
            left = self._sessions_left_open.items()
            to_revoke = [key for key, by in left if by is _LeftOpen.FOR_CLOSE]
        if to_revoke:  # no put runs any more, so none of them is in use
            codes = self._metadata_call(self._store.batch_put_session_revoke, to_revoke)
            with self._lock:
                for key, code in zip(to_revoke, codes):
                    if _session_gone(code):
                        del self._sessions_left_open[key]
        self._store.close()

    def stats(self) -> dict[str, int]:
        """A snapshot for logs and tests; not part of the driver protocol."""
        with self._lock:
            left = list(self._sessions_left_open.values())
            return {
                **{
                    f"rows_{r.status.value}_{reason}": self._skipped_rows[reason]
                    for reason, r in _SKIP_RESULTS.items()
                },
                "ops_unprovable": self._unprovable_op_count,
                "put_sessions_left_for_close": left.count(_LeftOpen.FOR_CLOSE),
                "put_sessions_left_to_master": left.count(_LeftOpen.TO_MASTER),
            }

    # Admission, on the caller's thread.

    def _admit(self, rows: RowBatch, is_put: bool, on_change: OnChange | None) -> _Op:
        """Checks ``rows`` and queues the operation, or raises ``DriverRejected`` having done
        neither. The checks look at the batch once, whatever its size: its rows are laid out alike,
        in the registration that holds their extent. The only work per row is the operation's list
        of pending results, built at C speed."""
        if not len(rows):
            raise DriverRejected("an operation needs at least one row")
        if rows.spans_per_row != 1:
            raise DriverRejected("this driver takes rows of one span")
        if rows.row_bytes > MAX_ROW_BYTES:
            raise DriverRejected(f"a row holds at most {MAX_ROW_BYTES} bytes")
        per_part = max(1, min(self._rows_per_call, self._bytes_per_call // rows.row_bytes))
        with self._lock:
            self._require_open(DriverRejected)
            if self._draining:
                raise DriverRejected("the driver is draining")
            if not self._workers_alive:  # every worker was stopped by an exception
                raise DriverRejected("no worker thread is left to run the operation")
            (extent,) = rows.extents
            region = next(
                (r for r in self._registrations if r.live and extent.within(r.span)), None
            )
            if region is None:
                raise DriverRejected(f"{extent} is outside every live registration")
            op = _Op(self, is_put, rows, region, per_part, [_PENDING] * len(rows), on_change)
            parts = -(-len(rows) // per_part)
            if is_put:
                if not self._queued_puts:  # the wait GETS_BEFORE_PUT_IS_DUE measures starts now
                    self._gets_ahead_of_puts = 0
                self._queued_puts.append(op)
                put_room = max(0, self._max_running_puts - self._running_puts)
                self._work_due.notify(min(parts, put_room))
            else:
                self._queued_gets.append(op)
                self._work_due.notify(parts)
            self._in_flight_op_count += 1
            region.ops_accessing += 1
        return op

    # The workers.

    def _part_is_takeable(self) -> bool:
        """Under the lock: whether a worker could take a part now."""
        put_room = self._running_puts < self._max_running_puts
        return bool(self._queued_gets) or (put_room and bool(self._queued_puts))

    def _worker_loop(self) -> None:
        """A worker thread: runs the parts ``_take`` gives it and waits while there is none; exits
        once the driver drains. After each take it wakes an idle worker for a part still takeable.
        The last worker to exit ends the rows no worker took, since none will."""
        try:
            while True:
                with self._lock:
                    part = self._take()
                    while part is None and not self._draining:
                        self._work_due.wait()
                        part = self._take()
                    if part is None:
                        return
                    if self._part_is_takeable():
                        self._work_due.notify()
                self._run_part(part)
        finally:
            done: list[_Op] = []
            with self._lock:
                self._workers_alive -= 1
                # The last worker to exit ends the rows still queued, since no one will take
                # them. After a drain nothing is queued; otherwise the others died of exceptions.
                if not self._workers_alive:
                    done = self._end_untaken_rows(self._queued_ops(), _NO_WORKER_LEFT)
                self._lock.notify_all()  # drain waits for every worker thread to exit
            for op in done:
                self._call_on_change(op)

    def _take(self) -> _Part | None:
        """Under the lock: the next part to run, or None when none may run now. Gets go first, but
        once ``GETS_BEFORE_PUT_IS_DUE`` gets have gone ahead of a waiting put, that put is due and
        goes next once a put may run; puts occupy at most ``_max_running_puts`` threads."""
        put_may_run = bool(self._queued_puts) and self._running_puts < self._max_running_puts
        put_is_due = put_may_run and self._gets_ahead_of_puts >= GETS_BEFORE_PUT_IS_DUE
        if self._queued_gets and not put_is_due:
            if self._queued_puts:  # this get goes ahead of a waiting put
                self._gets_ahead_of_puts += 1
            return self._next_part(self._queued_gets)
        if put_may_run:
            self._gets_ahead_of_puts = 0
            self._running_puts += 1
            return self._next_part(self._queued_puts)
        return None

    @staticmethod
    def _next_part(queue: deque[_Op]) -> _Part:
        """Under the lock: cuts the next part off the operation at the head of ``queue``; the
        operation leaves the queue with its last rows."""
        op = queue[0]
        start = op.first_untaken_row
        op.first_untaken_row = min(start + op.rows_per_part, len(op.rows))
        op.parts_running += 1
        if op.all_taken:
            queue.popleft()
        return _Part(op, start, op.first_untaken_row)

    def _run_part(self, part: _Part) -> None:
        """Runs ``part``'s SDK calls and records how they ended; calls the operation's
        ``on_change`` once it is done or turned unprovable. A part whose calls stop before they
        return its results still ends, its rows FAULT and its access unproven."""
        ended: _PartEnd | None = None
        try:
            ended = self._make_sdk_calls(part)
        finally:
            on_change_due = self._end_part(part, ended)
            if on_change_due:
                self._call_on_change(part.op)

    def _make_sdk_calls(self, part: _Part) -> _PartEnd:
        direction = "put" if part.op.is_put else "get"
        batch = part.op.rows[part.start : part.stop]
        try:
            rows = _Rows(batch.keys(), batch.addresses[:, 0].tolist(), batch.row_bytes)
        except Exception as error:  # no SDK call was made, so nothing touches the slots
            failed = [faulted(f"{direction} failed before any SDK call: {error!r}")] * len(batch)
            logger.warning(f"mooncake {direction}: {failed[0].fault}")
            return _PartEnd(failed, (), access_ended=True)
        started_s = time.monotonic()
        try:
            if part.op.is_put:
                results, access_ended = self._put_part(rows)
            else:
                results, access_ended = self._get_part(rows)
        except Exception as error:  # a bug, not an SDK answer: a transfer may still be running
            results, access_ended = _PartResults(), False
            results.fault(rows.keys, f"{direction} failed: {error!r}")
        if not results.all_ready or not access_ended:
            try:
                _warn(direction, rows, results, access_ended, time.monotonic() - started_s)
            except Exception as error:  # a log line must not change how the part ended
                logger.warning(f"mooncake: warning about a part raised {error!r}")
        skip_reasons = tuple(results.skip_reasons.values())
        return _PartEnd(results.in_order(rows.keys), skip_reasons, access_ended)

    def _end_part(self, part: _Part, ended: _PartEnd | None) -> bool:
        """Records how ``part`` ended (``None``: its calls stopped before returning its results,
        so its rows FAULT and its access is unproven); a part whose access did not end makes its
        operation unprovable. True when the operation is done or turned unprovable here."""
        op = part.op
        if ended is None:
            stopped = faulted("the worker stopped before the row was decided")
            ended = _PartEnd([stopped] * (part.stop - part.start), (), access_ended=False)
        access_ended = ended.access_ended
        with self._lock:
            op.results[part.start : part.stop] = ended.results
            self._skipped_rows.update(ended.skip_reasons)
            if op.is_put:
                self._running_puts -= 1
                if self._part_is_takeable():  # a waiting put may run now
                    self._work_due.notify()
            turned_unprovable = not access_ended and not op.unprovable
            if turned_unprovable:  # holds its region for good
                op.unprovable = True
                self._unprovable_op_count += 1
            op.parts_running -= 1
            return self._end_op_if_done(op) or turned_unprovable

    def _queued_ops(self) -> list[_Op]:
        """Under the lock: every operation with rows no worker took, gets first."""
        return [*self._queued_gets, *self._queued_puts]

    def _end_untaken_rows(self, ops: list[_Op], result: RowResult) -> list[_Op]:
        """Under the lock: ends the rows of ``ops`` no worker took as ``result``, taking each
        operation out of its queue; returns those this left done."""
        done = []
        for op in [op for op in ops if not op.all_taken]:
            (self._queued_puts if op.is_put else self._queued_gets).remove(op)
            untaken = len(op.rows) - op.first_untaken_row
            op.results[op.first_untaken_row :] = [result] * untaken
            op.first_untaken_row = len(op.rows)
            if self._end_op_if_done(op):
                done.append(op)
        return done

    def _end_op_if_done(self, op: _Op) -> bool:
        """Under the lock, after a part of ``op`` ended or its untaken rows did: once no row waits
        and no part runs, the operation is done: every row concluded and its access ended, unless
        unprovable. It then leaves ``_in_flight_op_count`` and, unless unprovable, releases its
        region. Wakes ``drain``; True when the operation is done."""
        self._lock.notify_all()
        if not op.all_taken or op.parts_running:
            return False
        self._in_flight_op_count -= 1
        if not op.unprovable:
            op.region.ops_accessing -= 1
        return True

    def _call_on_change(self, op: _Op) -> None:
        """Outside the lock, as ``OnChange`` asks."""
        if op.on_change is None:
            return
        try:
            op.on_change(op)
        except Exception as error:  # it must not raise; a worker must live on if it does
            logger.warning(f"mooncake: on_change raised {error!r}")

    # Put: start, one scatter, end what was written whole, revoke the rest.

    def _put_part(self, rows: _Rows) -> tuple[_PartResults, bool]:
        """Returns the part's results and whether access ended. Whatever stops the calls, each
        reserved key whose session they did not see closed is left open (class docstring,
        step 3)."""
        results = _PartResults()
        reserved, held_by_left_session = self._reserve_keys(rows.keys, results)
        # Reserved keys known to hold no session (never opened, or ended or revoked), and those
        # whose object the target may still write into.
        closed: set[str] = set()
        to_master: set[str] = set()
        try:
            if held_by_left_session:  # never put again, but another client may have written it
                self._confirm_existing(held_by_left_session, results, "held_by_own_put")
            if not reserved:
                return results, True
            access_ended = self._put_reserved(rows.only(reserved), results, closed, to_master)
            return results, access_ended
        finally:
            self._release_keys_and_record_left_open(reserved, closed, to_master)

    def _reserve_keys(self, keys: list[str], results: _PartResults) -> tuple[list[str], list[str]]:
        """Reserves every key that no put of this driver is writing and no session it left open
        holds, and returns them, then the keys a left-open session holds. A key another put of
        this driver is writing is NOT_READY."""
        with self._lock:
            being_put, left_open = self._keys_being_put, self._sessions_left_open.keys()
            if being_put.isdisjoint(keys) and left_open.isdisjoint(keys):  # usually
                reserved, held_by_left_session = keys, []
            else:
                reserved, held_by_left_session = [], []
                for key in keys:
                    if key in being_put:
                        results.skip(key, "held_by_own_put")
                    elif key in left_open:
                        held_by_left_session.append(key)
                    else:
                        reserved.append(key)
            self._keys_being_put.update(reserved)
        return reserved, held_by_left_session

    def _release_keys_and_record_left_open(
        self, reserved: list[str], closed: set[str], to_master: set[str]
    ) -> None:
        """Ends the reservation of ``reserved`` and records each session not ``closed`` as left
        open: for the master when its key is in ``to_master``, otherwise for ``close``."""
        left: dict[str, _LeftOpen] = {}
        if not closed.issuperset(reserved):  # usually every session ended
            left = {
                key: _LeftOpen.TO_MASTER if key in to_master else _LeftOpen.FOR_CLOSE
                for key in reserved
                if key not in closed
            }
        with self._lock:
            self._keys_being_put.difference_update(reserved)
            self._sessions_left_open.update(left)
        if left:
            first, why = next(iter(left.items()))
            logger.warning(
                f"mooncake: {len(left)} put sessions left open, first {first} {why.value}"
            )

    def _put_reserved(
        self, rows: _Rows, results: _PartResults, closed: set[str], to_master: set[str]
    ) -> bool:
        """Starts a session per row, writes and finishes the started ones, and confirms the rows
        whose object exists already. Returns whether access ended."""
        starts = self._metadata_call(self._store.batch_put_session_start, rows.keys, rows.sizes)
        started, existing = rows.keys, []
        if any(starts):  # not every session started (OK is 0)
            started = []
            for key, code in zip(rows.keys, starts):
                if code == _Code.OK:
                    started.append(key)
                    continue
                if code != _Code.METADATA_CALL_FAILED:  # the SDK answered: no session opened
                    closed.add(key)
                if code == _Code.OBJECT_ALREADY_EXISTS:
                    existing.append(key)
                elif code == _Code.NO_AVAILABLE_HANDLE:
                    results.skip(key, "pool_full")
                else:
                    # Any other code faults the row. That includes -600: this driver holds no
                    # session of the key, so it is something else, such as a cachelib master's
                    # object cap (16 MiB - 16 B) or a replicate config the session path refuses
                    # (the driver passes none); the master's log says which.
                    results.fault([key], f"put session start: {_describe(code)}")
        access_ended = True
        if started:
            access_ended = self._write_and_finish(rows.only(started), results, closed, to_master)
        if existing:  # readable only once the master shows it committed
            self._confirm_existing(existing, results, "uncommitted")
        return access_ended

    def _write_and_finish(
        self, rows: _Rows, results: _PartResults, closed: set[str], to_master: set[str]
    ) -> bool:
        """Writes the rows whose put session started in one scatter, then ends the session of each
        row written whole and revokes the rest, except those the target may still write into.
        Returns whether access ended."""
        count = len(rows.keys)
        try:
            # One range per row: the whole payload, written at offset 0 of its object.
            answers = _one_answer_per_key(
                self._store.batch_put_from_multi_buffer_ranges(
                    rows.keys,
                    [[a] for a in rows.addresses],
                    [[rows.row_bytes]] * count,
                    [[0]] * count,
                ),
                count,
            )
        except Exception as error:  # the scatter may still be reading the slots
            # Neither end (the object may be torn) nor revoke (the master would give its space
            # to another object while the scatter may still write into it).
            results.fault(rows.keys, f"batch_put_from_multi_buffer_ranges raised {error!r}")
            to_master.update(rows.keys)
            return False
        written, to_revoke = rows.keys, []
        if answers != rows.sizes:  # rare: some row was not written whole
            written = []
            for key, answer in zip(rows.keys, answers):
                if answer == rows.row_bytes:
                    written.append(key)
                    continue
                # Its object must never be marked complete. After a TRANSFER_FAIL a TCP target
                # may still write into the object's space (module docstring).
                results.fault([key], f"put ranges: {_describe_moved(answer, rows.row_bytes)}")
                if answer == _Code.TRANSFER_FAIL:
                    to_master.add(key)
                else:
                    to_revoke.append(key)
        if written:
            to_revoke += self._end_sessions(written, results, closed)
        if to_revoke:
            codes = self._metadata_call(self._store.batch_put_session_revoke, to_revoke)
            closed.update(key for key, code in zip(to_revoke, codes) if _session_gone(code))
        return True

    def _end_sessions(self, keys: list[str], results: _PartResults, closed: set[str]) -> list[str]:
        """Ends the sessions of rows written whole. Returns the keys whose end was refused, to
        revoke; a session whose end may still reach the master, answered ``_END_OUTCOME_UNKNOWN``
        first, stays open even when the retry was refused."""
        end = self._store.batch_put_session_end
        codes = self._metadata_call(end, keys)
        if not any(codes):  # every session ended (OK is 0)
            closed.update(keys)
            return []
        final = dict(zip(keys, codes))
        retry = [key for key, code in final.items() if code in _END_RETRIED]
        if retry:
            final.update(zip(retry, self._metadata_call(end, retry)))
        refused = []
        for key, first in zip(keys, codes):
            code = final[key]
            if code == _Code.OK:
                closed.add(key)
                continue
            results.fault([key], f"put session end: {_describe(code)}")
            if first not in _END_OUTCOME_UNKNOWN:
                refused.append(key)
        return refused

    def _confirm_existing(
        self, keys: list[str], results: _PartResults, reason_if_absent: str
    ) -> None:
        """Leaves READY the rows whose objects the master shows complete; skips the others for
        ``reason_if_absent``, or faults them when the master could not tell."""
        for key, answer in zip(keys, self._metadata_call(self._store.batch_is_exist, keys)):
            if answer == OBJECT_ABSENT:
                results.skip(key, reason_if_absent)
            elif answer != OBJECT_EXISTS:
                results.fault([key], f"existence check: {_describe(answer)}")

    # Get: one batch_get_into; rows whose read lease expired mid-read are read once more.

    def _get_part(self, rows: _Rows) -> tuple[_PartResults, bool]:
        results = _PartResults()
        return results, self._get_into(rows, results, is_reread=False)

    def _get_into(self, rows: _Rows, results: _PartResults, is_reread: bool) -> bool:
        """Reads ``rows`` and records each one not read whole; returns whether access ended."""
        try:
            answers = _one_answer_per_key(
                self._store.batch_get_into(rows.keys, rows.addresses, rows.sizes), len(rows.keys)
            )
        except Exception as error:  # the transfers may still be writing the slots
            results.fault(rows.keys, f"batch_get_into raised {error!r}")
            return False
        if answers == rows.sizes:  # usually every row was read whole
            return True
        expired = []
        for key, answer in zip(rows.keys, answers):
            if answer == rows.row_bytes:
                continue
            if answer == _Code.LEASE_EXPIRED and not is_reread:
                expired.append(key)
            elif answer == _Code.REPLICA_IS_NOT_READY and not is_reread:
                results.skip(key, "no_complete_replica")
            else:
                results.recorded[key] = _result_of_get(answer, rows.row_bytes, is_reread)
        # A TRANSFER_FAIL future may give up while its RDMA still runs (module docstring).
        access_ended = _Code.TRANSFER_FAIL not in answers
        if expired:
            # The read lease expired before the transfer finished, so the master may have reused
            # the object's space mid-read: read once more. The reread is READY only if it reads
            # the whole object again; anything else, a second expiry included, is a FAULT. A
            # TRANSFER_FAIL of the first read keeps access open whatever the reread does.
            reread_access_ended = self._get_into(rows.only(expired), results, is_reread=True)
            access_ended = access_ended and reread_access_ended
        return access_ended

    # Helpers.

    def _metadata_call(self, call: Callable, keys: list[str], *args) -> list[int]:
        """A session or existence call. It touches no row memory, so a raise or a wrong answer
        count fails only its keys."""
        try:
            return _one_answer_per_key(call(keys, *args), len(keys))
        except Exception as error:
            logger.warning(f"mooncake: {getattr(call, '__name__', call)} failed: {error!r}")
            return [_Code.METADATA_CALL_FAILED] * len(keys)

    def _add(self, registration: _Registration) -> None:
        with self._lock:
            self._registrations.append(registration)

    def _checked_registration(self, handle: _Registration) -> _Registration:
        self._require_open()
        if not isinstance(handle, _Registration) or handle not in self._registrations:
            raise ValueError("handle does not belong to this driver")
        return handle

    def _checked_op(self, op: _Op) -> _Op:
        self._require_open()
        if not isinstance(op, _Op) or op.owner is not self:
            raise ValueError("operation does not belong to this driver")
        return op

    def _require_open(self, error: type[Exception] = RuntimeError) -> None:
        """Raises ``error`` once closed: DriverRejected refuses new work, RuntimeError the rest."""
        if self._closed:
            raise error("the driver is closed")


def _result_of_get(answer: int, size: int, is_reread: bool) -> RowResult:
    """The result of a row ``batch_get_into`` did not read whole."""
    if answer >= 0:  # a read answers the object's size, which must be the row's
        return faulted(f"object is {answer} bytes, expected {size}")
    if is_reread and answer in (_Code.OBJECT_NOT_FOUND, _Code.REPLICA_IS_NOT_READY):
        # The first read already wrote this slot, and MISS or NOT_READY would claim that nothing
        # landed there.
        return faulted(f"reread after LEASE_EXPIRED: {_describe(answer)}")
    if answer == _Code.OBJECT_NOT_FOUND:
        return _MISS
    return faulted(f"batch_get_into: {_describe(answer)}")


def _session_gone(revoke_answer: int) -> bool:
    """Whether a revocation left no session: revoked, or none was left (INVALID_PARAMS)."""
    return revoke_answer in (_Code.OK, _Code.INVALID_PARAMS)


def _describe_moved(answer: int, size: int) -> str:
    return f"moved {answer} of {size} bytes" if answer >= 0 else _describe(answer)


def _one_answer_per_key(answers: Sequence[int], count: int) -> list[int]:
    answers = list(answers)
    if len(answers) != count:
        raise _WrongAnswerCount(f"{len(answers)} answers for {count} keys")
    return answers


def _describe(code: int) -> str:
    try:
        return f"{_Code(code).name}({code})"
    except ValueError:
        return f"code({code})"


def _warn(
    direction: str, rows: _Rows, results: _PartResults, access_ended: bool, seconds: float
) -> None:
    """One warning per part that failed or skipped a row, or left access open; a MISS is
    neither."""
    went_wrong = [(key, result) for key, result in results.recorded.items() if result is not _MISS]
    if access_ended and not went_wrong:
        return
    first = ""
    if went_wrong:
        key, result = went_wrong[0]
        why = result.fault or f"{result.status.value} ({results.skip_reasons[key]})"
        first = f"; first {key}: {why}"
    access = "ended" if access_ended else "may still be in progress"
    logger.warning(
        f"mooncake {direction}: {len(went_wrong)} of {len(rows.keys)} rows failed or skipped in "
        f"{seconds:.3f} s{first}; access {access}"
    )
