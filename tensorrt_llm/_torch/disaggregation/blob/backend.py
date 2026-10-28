# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: the public ``base.backend.Backend`` over a private store driver.

The backend translates and keeps the books; the driver owns how its store's work runs: its queue,
the order in which gets and puts run, and its threads. A submission is checked and cut into calls,
one per layer group (split where the driver limits the rows of one call, and at a repeated object
key, which no call carries twice). A call is one driver operation, its ``OpHandle``; a driver may
split it further. Each call goes to the driver at once, on the caller's thread, as one ``put`` or
``get`` of a ``RowBatch``: the driver only records and queues it. The backend queues nothing, so
it has no order of its own to keep.

Each call carries an ``on_change`` callback, which the driver calls once the operation's rows all
concluded, once its access ended and once it turned unprovable; it is the backend's only news of
the call. The callback queries the operation through the driver and records what the queries
show, on the thread the driver called it from. A record only ever adds evidence (results, an access
end, unprovability), so records may come in any order, on any thread, more than once. All
bookkeeping is under one lock; neither the driver nor ``wake`` is ever called under it.

A call settles once its row results are recorded and its access end is recorded or can never be
proven ("unprovable"). An unprovable call holds its slots for good: its attempt never quiesces and
the regions stay registered.

An attempt's outcome lands once every call has its results; it quiesces once every call's access
ended. A cancel, a fetch's first fault, a publish's whole-call failure or a shutdown stops it: when
the driver declares ``cancel``, the progress thread asks it to cancel the calls whose rows have not
all concluded. That, and the teardown after a shutdown (draining the driver at once, revoking the
registrations once every attempt quiesced), is all the progress thread does.

A submission hands its calls over up to the first one the driver does not take, or until a
protocol violation is recorded; the calls after that never reach the driver. A driver that breaks
its protocol (a raise other than ``DriverRejected`` from ``put`` or ``get``, no handle, a raise
from a query, or a wrong result count) leaves its call held as unprovable, unless the call's access
end was recorded already, and every later submission is rejected.
"""

from __future__ import annotations

import functools
import itertools
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, NoReturn, Protocol, cast

import numpy as np

from tensorrt_llm.logger import logger

from ..base.backend import (
    Capabilities,
    Hint,
    InvalidSubmission,
    LayerGroupRows,
    Outcome,
    SlotRegion,
    SubmissionRejected,
    View,
    WakeFn,
)
from ..base.memory import MemoryKind
from .keys import check_key_prefix, key_stem
from .store import (
    DriverRejected,
    OnChange,
    OpHandle,
    RegionHandle,
    RegionLayout,
    RegionSpec,
    RegistersRegionsTogether,
    RegistrationError,
    RowBatch,
    RowResult,
    RowStatus,
    Span,
    StoreDriver,
    concluded,
)

__all__ = [
    "BlobAttempt",
    "BlobBackend",
    "BlobConfig",
    "ConstructionRejected",
    "DriverCallLimit",
    "RegionBinding",
    "RegionHold",
]

# After a shutdown, the progress thread retries what is left of the teardown (an attempt not
# quiesced yet, a drain that answered False, a revocation that failed) at an interval that starts
# at TEARDOWN_RETRY_MIN_S, doubles up to TEARDOWN_RETRY_MAX_S, and drops back to
# TEARDOWN_RETRY_MIN_S whenever the thread is poked.
TEARDOWN_RETRY_MIN_S = 0.001
TEARDOWN_RETRY_MAX_S = 0.1
SHUTDOWN_FAULT = "blob backend shut down"

# Numbers each backend and binding: it names a backend's progress thread and keys each instance's
# warning_once calls, so one instance's warning never silences another's.
_instance_ids = itertools.count()


@dataclass(frozen=True)
class BlobConfig:
    """The ``blob`` section of a store config.

    How a store's work is batched, scheduled and overlapped is its driver's own, so it is set with
    the driver, for Mooncake in the ``mooncake`` section.

    Attributes:
        key_prefix: The deployment's key namespace; matches ``[A-Za-z0-9_-]{1,64}``.
    """

    key_prefix: str = "trtllm-kv"

    def __post_init__(self) -> None:
        if not isinstance(self.key_prefix, str):
            raise ValueError(f"key_prefix must be a str, got {self.key_prefix!r}")
        check_key_prefix(self.key_prefix)


class ConstructionRejected(RuntimeError):
    """``RegionBinding.register`` failed and left nothing registered, held or accessible: no cleanup
    is needed, and a backend factory maps it to the runtime's certified rejection.

    Any other exception from ``register`` may leave a registration live; the binding then keeps the
    hold, and ``release_hold`` refuses until every registration is revoked.
    """


class RegionHold(Protocol):
    """What this package uses of ``pyexecutor.kv_cache.sharing.RegionHold``, restated because blob
    modules may not import ``pyexecutor``."""

    def release(self) -> None: ...


class RegionBinding:
    """The lender's region hold and the registrations made under it.

    Registers, once each and on the builder thread, the regions the driver can access; revokes the
    registrations only when nothing can access them, on the backend's progress thread (or on the
    builder thread when ``register`` rolls back a failed registration); releases the hold only once
    no registration may be live: through ``release_hold`` on the manager's thread, or inside
    ``register`` on the builder thread when it raises ``ConstructionRejected``, after which the
    caller must not call ``release_hold``. The driver is never called under the binding's lock, so
    reading the live registrations never waits on a revocation.

    A driver implementing ``RegistersRegionsTogether`` takes every region in one
    ``register_regions`` call and gives them back in one ``unregister_regions``; every region has
    an owner here before the store is asked, so a failure of any shape leaves them accounted for.

    Args:
        regions: The lender's regions, in its order; region ``i`` is labelled ``region:<i>``.
        driver: The store driver that registers them.
        hold: The lender's hold, taken before registering.
    """

    def __init__(
        self, regions: Sequence[SlotRegion], driver: StoreDriver, hold: RegionHold
    ) -> None:
        self.regions = tuple(regions)
        self.driver = driver
        device_memory = driver.capabilities.device_memory
        # The regions register() registers, by index: those with slots in memory the driver can
        # access. A row in any other region is one the backend does not recognize.
        self.served_region_indices = frozenset(
            index
            for index, region in enumerate(self.regions)
            if region.slot_count > 0 and (region.memory_kind is MemoryKind.HOST or device_memory)
        )
        self._hold = hold
        # The driver as a RegistersRegionsTogether, taking the regions as one batch; else None.
        self._together = driver if isinstance(driver, RegistersRegionsTogether) else None
        self._name = f"region-binding-{next(_instance_ids)}"
        self._lock = threading.Lock()
        # Region index -> its handle, or None for a registration that may exist without one.
        self._live_registrations: dict[int, RegionHandle | None] = {}

    def register(self) -> None:
        """Registers the regions in ``served_region_indices``.

        Raises:
            ConstructionRejected: Nothing is registered and the hold is released: there is no region
                to register, a region's layout is refused, or a registration failed with nothing
                escaping and the earlier ones were revoked.
            RuntimeError: A registration may be live; the hold is kept.
        """
        if not self.served_region_indices:  # a backend that could serve no row
            self._hold.release()
            raise ConstructionRejected("no region has slots in memory the store driver can access")
        layouts = {}
        for index in sorted(self.served_region_indices):
            region = self.regions[index]
            try:  # every layout is checked before any region is registered
                layouts[index] = RegionLayout(
                    region.slot_bytes, region.slot_count, region.layout_label, region.memory_kind
                )
            except ValueError as error:
                self._hold.release()
                raise ConstructionRejected(f"region:{index}: {error}") from error
        if self._together is not None:
            self._register_together(self._together, layouts)
            return
        for index, layout in layouts.items():
            region = self.regions[index]
            try:
                handle = self.driver.register(region.address, region.nbytes, layout)
            except Exception as error:
                self._roll_back_if_possible_and_raise(index, error)
            else:
                self._track_live_registration(index, handle)

    def try_unregister_all(self) -> bool:
        """Revokes every live registration it can. True when none is left; a revocation that failed
        stays live for the next call."""
        if self._together is not None:
            return self._unregister_together(self._together)
        with self._lock:
            revocable = [
                (i, handle) for i, handle in self._live_registrations.items() if handle is not None
            ]
        for index, handle in revocable:
            try:
                self.driver.unregister(handle)
            except Exception as error:
                logger.warning_once(
                    f"blob backend: revoking region:{index} failed, will retry: {error!r}",
                    key=(self._name, "unregister", index),
                )
                continue
            with self._lock:
                del self._live_registrations[index]
        with self._lock:
            return not self._live_registrations

    def release_hold(self) -> None:
        """Releases the lender's hold, on the manager's thread.

        Raises:
            RuntimeError: A registration may still be live; the hold is kept.
        """
        live = self.live_labels()
        if live:
            raise RuntimeError(f"the region hold is kept: {sorted(live)} may still be registered")
        self._hold.release()

    def live_labels(self) -> frozenset[str]:
        """``region:<i>`` for every region that may still be registered."""
        with self._lock:
            return frozenset(f"region:{index}" for index in self._live_registrations)

    def _roll_back_if_possible_and_raise(self, index: int, error: Exception) -> NoReturn:
        """Raises for region ``index``'s failed registration. When nothing escaped, unregisters
        every earlier region; once none is left, releases the lender's hold and raises
        ``ConstructionRejected``. Otherwise (the failed registration may be live, or an earlier
        region stays registered) keeps the hold and raises ``RuntimeError``."""
        handle = error.handle if isinstance(error, RegistrationError) else None
        nothing_escaped = isinstance(error, DriverRejected) or (
            isinstance(error, RegistrationError) and handle is None
        )
        if not nothing_escaped:
            self._track_live_registration(index, handle)
            raise RuntimeError(f"region:{index} may be registered after {error!r}") from error
        if not self.try_unregister_all():
            raise RuntimeError(
                f"registering region:{index} failed and an earlier region stays registered: {error}"
            ) from error
        # Nothing is registered or accessible, and construction precedes the executor loop, so the
        # builder thread may still give the hold back.
        self._hold.release()
        raise ConstructionRejected(f"registering region:{index} failed: {error}") from error

    def _track_live_registration(self, index: int, handle: RegionHandle | None) -> None:
        with self._lock:
            self._live_registrations[index] = handle

    def _register_together(
        self, driver: RegistersRegionsTogether, layouts: dict[int, RegionLayout]
    ) -> None:
        try:
            specs = [
                RegionSpec(self.regions[index].address, self.regions[index].nbytes, layout)
                for index, layout in layouts.items()
            ]
        except ValueError as error:  # a region the boundary cannot describe; the store not asked
            self._hold.release()
            raise ConstructionRejected(f"bad region geometry: {error}") from error
        with self._lock:  # every region has an owner before the store is asked
            self._live_registrations = dict.fromkeys(layouts)
        try:
            handle = driver.register_regions(specs)
        except Exception as error:
            self._raise_for_failed_batch(error)
        self._keep_all(handle)

    def _unregister_together(self, driver: RegistersRegionsTogether) -> bool:
        with self._lock:
            handles: list[RegionHandle] = []
            for handle in self._live_registrations.values():
                if handle is not None and not any(handle is seen for seen in handles):
                    handles.append(handle)
        for handle in handles:  # the one batch handle every region shares
            try:
                driver.unregister_regions(handle)
            except Exception as error:
                logger.warning_once(
                    f"blob backend: revoking the regions together failed, will retry: {error!r}",
                    key=(self._name, "unregister_regions"),
                )
                continue
            with self._lock:
                self._live_registrations = {
                    i: h for i, h in self._live_registrations.items() if h is not handle
                }
        with self._lock:
            return not self._live_registrations

    def _raise_for_failed_batch(self, error: Exception) -> NoReturn:
        handle = error.handle if isinstance(error, RegistrationError) else None
        nothing_escaped = isinstance(error, DriverRejected) or (
            isinstance(error, RegistrationError) and handle is None
        )
        if nothing_escaped:
            with self._lock:
                self._live_registrations.clear()
            self._hold.release()
            raise ConstructionRejected(
                f"registering the regions together failed: {error}"
            ) from error
        if handle is None:  # some may be registered, with nothing to revoke them by
            raise RuntimeError(f"the regions may be registered after {error!r}") from error
        self._keep_all(handle)
        if not self.try_unregister_all():
            raise RuntimeError(
                f"registering the regions together failed and some may stay registered: {error}"
            ) from error
        self._hold.release()
        raise ConstructionRejected(f"registering the regions together failed: {error}") from error

    def _keep_all(self, handle: RegionHandle) -> None:
        with self._lock:
            for index in self._live_registrations:
                self._live_registrations[index] = handle


class DriverCallLimit(NamedTuple):
    """The most rows of a region, and their bytes, that one driver ``put`` or ``get`` carries."""

    rows: int
    nbytes: int


class _GroupRows(NamedTuple):
    """One group's arrays, taken from the view at submission."""

    object_keys: np.ndarray  # uint8 [rows, width]
    addresses: np.ndarray  # int64 [rows, regions]
    region_indices: tuple[int, ...]


@dataclass(eq=False)
class _Call:
    """Rows ``[start, stop)`` of one group of an attempt, handed to the driver as one operation: one
    ``put`` or ``get``, one ``OpHandle``."""

    attempt: BlobAttempt
    group_position: int  # in ``View.groups``, not the group's ``layer_group_index``
    start: int
    stop: int
    # The fields below change only under the backend's lock.
    op: OpHandle | None = None  # the driver's handle, once the whole submission was handed over
    results_recorded: bool = False
    access_end_recorded: bool = False
    unprovable: bool = False  # the driver can never prove its access ended
    cancel_forwarded: bool = False  # the driver was asked to cancel it

    def __len__(self) -> int:
        return self.stop - self.start


class _CallResults(NamedTuple):
    """A concluded call's results as an attempt records them: whether each row is READY, and the
    first row that faulted (its index in the call) with its reason."""

    ready: list[bool]
    first_fault: tuple[int, str] | None

    @classmethod
    def none_delivered(cls, rows: int, fault: str | None = None) -> _CallResults:
        """Every row undelivered: each faults with ``fault`` when given (the call failed as a
        whole), otherwise none does (CANCELLED)."""
        return cls([False] * rows, None if fault is None else (0, fault))

    @classmethod
    def of(cls, results: Sequence[RowResult]) -> _CallResults:
        ready = [result.status is RowStatus.READY for result in results]
        if all(ready):
            return cls(ready, None)
        faults = ((row, r.fault) for row, r in enumerate(results) if r.fault is not None)
        return cls(ready, next(faults, None))


class _Recorded(NamedTuple):
    """What a record under the backend's lock leaves to do once the lock is released."""

    wake_due: bool  # the attempt's outcome landed or it quiesced
    newly_unprovable: bool  # the call turned unprovable with this record


class BlobAttempt:
    """The handle of one submission: a ``base.backend.Attempt``.

    Its state changes only under the backend's lock; ``poll`` and ``quiesced`` read one reference
    without it, so they never block.

    Two counters, both set to the number of calls at submission; each call decrements each at most
    once:

    - ``_calls_awaiting_results``: calls whose row results are not recorded yet.
    - ``_calls_awaiting_access_end``: calls whose access end is not recorded yet. An unprovable
      call never leaves it.

    The outcome lands once ``_calls_awaiting_results`` is 0; the attempt quiesces once
    ``_calls_awaiting_access_end`` is 0.
    """

    def __init__(
        self,
        backend: BlobBackend,
        label: str,
        is_fetch: bool,
        groups: list[_GroupRows],
        call_ranges: list[tuple[int, int, int]],
        hint: Hint = None,
    ) -> None:
        self.label = label
        """The attempt's label among ``BlobBackend.shutdown``'s, ``attempt:<n>``."""
        self.is_fetch = is_fetch
        self._backend = backend
        self._hint = hint  # a fetch's source, for every driver get; immutable, so kept as given
        self._masks = [np.zeros(len(rows.addresses), dtype=bool) for rows in groups]
        self._calls = [_Call(self, *call_range) for call_range in call_ranges]
        self._calls_awaiting_results = self._calls_awaiting_access_end = len(self._calls)
        # The first row fault; _record_results decides whether it stops the attempt.
        self._first_fault: str | None = None
        # Set by a cancel, a stopping fault or a shutdown; when the driver declares cancel, the
        # progress thread then asks it to cancel the calls not concluded.
        self._stopped = False
        self._outcome: Outcome | None = None
        self._quiesced = False

    def poll(self) -> Outcome | None:
        return self._outcome

    def quiesced(self) -> bool:
        return self._quiesced

    def cancel(self) -> None:
        """Stops the attempt unless its outcome landed. When the driver declares ``cancel``, the
        progress thread asks it to cancel the calls whose rows have not all concluded; rows already
        delivered stay delivered."""
        self._backend._cancel(self)

    @property
    def _has_unprovable_call(self) -> bool:
        """A call can never prove its access ended, so the attempt never quiesces."""
        return any(call.unprovable for call in self._calls)

    def _record_results(self, call: _Call, results: _CallResults, whole_call_failed: bool) -> bool:
        """Writes ``call``'s results into the masks and keeps the first fault. True when that fault
        stops the attempt: when the driver declares ``cancel``, every call not concluded yet is
        cancelled, earlier ones included, and rows already delivered stay delivered. Any fault stops
        a fetch: the request it serves has a gap to fill some other way. A publish's rows stand
        alone, so only a call that failed as a whole stops it."""
        self._masks[call.group_position][call.start : call.stop] = results.ready
        if results.first_fault is None:
            return False
        row, fault = results.first_fault
        where = f"group {call.group_position} row {call.start + row}"  # indexes the outcome's masks
        self._first_fault = self._first_fault or f"{where}: {fault}"
        return self.is_fetch or whole_call_failed

    def _conclude_or_quiesce(self) -> bool:
        """Lands the outcome and turns quiesced once their conditions hold; True when either did."""
        changed = False
        # The outcome waits for every call's results even after a fault: calls conclude in any
        # order, so the first fault seen can come from a call issued after one that is still
        # delivering rows a prefix reuse needs.
        if self._outcome is None and self._calls_awaiting_results == 0:
            for mask in self._masks:  # final, so the outcome keeps them without a copy
                mask.flags.writeable = False
            self._outcome = Outcome(tuple(self._masks), self._first_fault)
            changed = True
        if not self._quiesced and self._calls_awaiting_access_end == 0:
            self._quiesced = True
            changed = True
        return changed

    def _force_outcome(self, reason: str) -> None:
        first = self._first_fault
        fault = reason if first is None else f"{reason}; first fault: {first}"
        # Still writable, so ``Outcome`` copies them: calls still open go on writing their results
        # into these masks.
        self._outcome = Outcome(tuple(self._masks), fault)


class BlobBackend:
    """Moves a view's rows between the lender's registered regions and a byte store: a
    ``base.backend.Backend``. A process may hold several, one per lender, each with its own driver.

    Args:
        binding: The lender's regions, already registered with ``binding.driver``. The backend
            revokes the registrations once nothing can access them after shutdown.
        wake: Called outside every lock whenever an attempt's outcome lands or it quiesces, once
            the change shows through ``poll`` or ``quiesced``.
        config: The store config's ``blob`` section.
    """

    def __init__(self, binding: RegionBinding, wake: WakeFn, config: BlobConfig) -> None:
        # Fixed once __init__ returns.
        self._binding = binding
        self._driver = binding.driver
        self._regions = binding.regions
        # Each region's whole extent, which a RowBatch names as holding all its rows.
        self._region_extents = {
            index: Span(region.address, region.nbytes)
            for index, region in enumerate(binding.regions)
            if index in binding.served_region_indices
        }
        self._wake = wake
        self._key_stem = key_stem(config.key_prefix)
        self._name = name = f"blob-backend-{next(_instance_ids)}"
        driver_capabilities = self._driver.capabilities
        self._max_rows_per_op = driver_capabilities.max_rows_per_op
        self._driver_multi_span = driver_capabilities.multi_span
        self._driver_reads_hints = driver_capabilities.hints
        self._driver_cancels = driver_capabilities.cancel
        # Every call goes to the driver as it is submitted, so the backend can cancel only by
        # asking the driver: it declares cancel exactly when the driver does.
        self._capabilities = Capabilities(
            fetch=driver_capabilities.get,
            publish=driver_capabilities.put,
            cancel=driver_capabilities.cancel,
        )
        self._attempt_ids = itertools.count()  # its next() is atomic: submissions take it unlocked
        self._lock = threading.Lock()
        # The progress thread waits on this between rounds; _poke_progress notifies it.
        self._progress_due = threading.Condition(self._lock)
        # ``shutdown`` waits on this; notified whenever its answer may have changed: an attempt
        # concluded or quiesced, or the progress thread finished the teardown.
        self._shutdown_recheck = threading.Condition(self._lock)

        # Guarded by _lock.
        # Set by _poke_progress, cleared as the next round starts: run a round now.
        self._progress_poked = False
        self._attempts: set[BlobAttempt] = set()  # until concluded and quiesced
        # Set once and never cleared. The progress loop also reads it without the lock: a round
        # that misses it runs again, since setting it pokes the progress thread.
        self._shutdown_requested = False
        self._teardown_done = False  # drained and every registration revoked
        self._protocol_violation: str | None = None  # how the driver first broke its protocol

        # Used by the progress thread only.
        self._drained = False  # the driver's drain returned True; it is not drained again
        # Used by ``shutdown`` only: the labels the last shutdown logged, which it logs again only
        # once they changed.
        self._last_logged_labels: frozenset[str] | None = None

        self._progress_thread = threading.Thread(
            target=self._progress_loop, name=f"{name}-progress", daemon=True
        )
        self._progress_thread.start()

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def fetch(self, view: View, source: Hint) -> BlobAttempt:
        """``source`` goes unchanged to every driver ``get`` of the fetch when the driver declares
        ``hints``; any other driver takes only None."""
        return self._submit(view, source, is_fetch=True)

    def publish(self, view: View, destination: Hint) -> BlobAttempt:
        """``destination`` must be None: a driver ``put`` takes no hint."""
        return self._submit(view, destination, is_fetch=False)

    def shutdown(self, timeout: float) -> frozenset[str]:
        """Forces outcomes, then waits up to ``timeout`` for the progress thread to finish cleanly,
        or only until the answer can no longer change: every attempt not yet quiesced holds a call
        whose access can never be proven to end.

        Never calls the driver: the progress thread forwards cancels, drains the driver and, once
        every attempt quiesced, revokes the registrations, since revocation can be slow (RDMA
        deregisters on every NIC) and a shutdown with no budget left must still return at once.
        """
        # Bounded so that the waits below accept it: an infinite budget must not raise.
        deadline = time.monotonic() + min(max(0.0, timeout), threading.TIMEOUT_MAX)
        if self._refuse_new_work_and_force_outcomes():
            self._call_wake()
        with self._lock:
            self._shutdown_recheck.wait_for(
                self._shutdown_answer_is_final, max(0.0, deadline - time.monotonic())
            )
            teardown_done = self._teardown_done
        if teardown_done:  # the progress thread is returning
            self._progress_thread.join(max(0.0, deadline - time.monotonic()))
        labels = self._still_accessible_labels()
        self._log_still_accessible(labels)
        return labels

    def driver_call_limit(self, region_index: int) -> DriverCallLimit | None:
        """The most rows of region ``region_index`` the driver takes in one ``put`` or ``get`` (its
        ``max_rows_per_op``), and their bytes; None when the driver declares no limit on the rows
        of one call. A group is still cut at a repeated object key, which no call carries twice.

        An attempt quiesces only once every call proved its access ended, so one call that never
        proves it keeps the slots of its whole submission; a holder bounds that by submitting at
        most this many rows of the region at a time.

        Raises:
            ValueError: The backend did not register the region.
        """
        if region_index not in self._binding.served_region_indices:
            raise ValueError(f"region {region_index} is not registered")
        rows = self._max_rows_per_op
        if rows is None:
            return None
        return DriverCallLimit(rows=rows, nbytes=rows * self._regions[region_index].slot_bytes)

    def stats(self) -> dict[str, int]:
        """A snapshot of the work in flight, for logs and tests."""
        with self._lock:
            calls = [call for attempt in self._attempts for call in attempt._calls]
            return {
                "attempts_active": len(self._attempts),
                "calls_awaiting_results": sum(not call.results_recorded for call in calls),
                "calls_awaiting_access_end": sum(not call.access_end_recorded for call in calls),
                "calls_unprovable": sum(call.unprovable for call in calls),
            }

    # Submission, on the executor loop's thread: validates, then hands each call to the driver,
    # which only records and queues it.

    def _submit(self, view: View, hint: Hint, is_fetch: bool) -> BlobAttempt:
        if hint is not None and not is_fetch:
            raise InvalidSubmission(f"a publish takes only a None destination, got {hint!r}")
        if hint is not None and not self._driver_reads_hints:
            raise InvalidSubmission(f"the store driver reads no hints, got {hint!r}")
        if not (self._capabilities.fetch if is_fetch else self._capabilities.publish):
            raise InvalidSubmission(f"the store driver cannot {'get' if is_fetch else 'put'}")
        groups = [self._checked_group_rows(group) for group in view.groups]
        if is_fetch:
            _refuse_shared_destinations(groups, self._regions)
        label = f"attempt:{next(self._attempt_ids)}"
        attempt = BlobAttempt(self, label, is_fetch, groups, self._call_ranges(groups), hint)
        batches = [self._row_batch(groups, call) for call in attempt._calls]
        with self._lock:
            if self._shutdown_requested:
                raise SubmissionRejected("the blob backend is shut down")
            if self._protocol_violation is not None:
                raise SubmissionRejected(
                    f"the store driver broke its protocol: {self._protocol_violation}"
                )
            self._attempts.add(attempt)
            wake_due = self._conclude_or_quiesce_and_retire(attempt)  # an empty view ends here
        if wake_due:
            self._call_wake()
        self._hand_over(attempt, batches)
        return attempt

    def _checked_group_rows(self, group: LayerGroupRows) -> _GroupRows:
        where = f"layer group {group.layer_group_index}"
        if group.object_keys is None or group.slots is None or group.region_indices is None:
            raise InvalidSubmission(f"{where}: rows need object keys, slots and regions")
        keys = np.asarray(group.object_keys)
        slots = np.asarray(group.slots)
        rows = len(group)
        if keys.dtype != np.uint8 or keys.ndim != 2 or keys.shape[0] != rows or keys.shape[1] == 0:
            raise InvalidSubmission(
                f"{where}: object keys must be uint8 [{rows}, width > 0], got {keys.dtype} "
                f"{keys.shape}"
            )
        keys = np.ascontiguousarray(keys)  # one key twice is allowed: _call_bounds cuts there
        if slots.dtype != np.int64 or slots.shape != (rows,):
            raise InvalidSubmission(
                f"{where}: slots must be int64 [{rows}], got {slots.dtype} {slots.shape}"
            )
        indices = group.region_indices
        if (
            not isinstance(indices, tuple)
            or not indices
            or any(type(index) is not int for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise InvalidSubmission(
                f"{where}: region indices must be a nonempty tuple of distinct ints"
            )
        if len(indices) > 1 and not self._driver_multi_span:
            raise InvalidSubmission(f"{where}: the store driver cannot transfer multiple spans")
        # Refused, never skipped: the driver must not be handed memory it never registered.
        regions: list[SlotRegion] = []
        for index in indices:
            if index not in self._binding.served_region_indices:
                raise InvalidSubmission(f"{where}: region {index} is not registered")
            region = self._regions[index]
            if np.any((slots < 0) | (slots >= region.slot_count)):
                raise InvalidSubmission(f"{where}: a slot is outside region {index}")
            if region.address + region.slot_bytes * region.slot_count > np.iinfo(np.int64).max:
                raise InvalidSubmission(f"{where}: region {index} exceeds int64 addresses")
            regions.append(region)
        addresses = np.column_stack(
            [region.address + slots * region.slot_bytes for region in regions]
        )
        for column, region in enumerate(regions):
            for previous, other in enumerate(regions[:column]):
                if np.any(
                    (addresses[:, column] < addresses[:, previous] + other.slot_bytes)
                    & (addresses[:, previous] < addresses[:, column] + region.slot_bytes)
                ):
                    raise InvalidSubmission(f"{where}: a row's spans overlap")
        return _GroupRows(keys, addresses, indices)

    def _call_ranges(self, groups: list[_GroupRows]) -> list[tuple[int, int, int]]:
        """``(group position, start, stop)`` of each call: one per group, split where the driver
        limits the rows of one call and at a repeated object key."""
        ranges = []
        for position, group_rows in enumerate(groups):
            rows_per_call = self._max_rows_per_op or max(len(group_rows.addresses), 1)
            ranges += [
                (position, start, stop)
                for start, stop in _call_bounds(group_rows.object_keys, rows_per_call)
            ]
        return ranges

    def _row_batch(self, groups: list[_GroupRows], call: _Call) -> RowBatch:
        group_rows = groups[call.group_position]
        rows = slice(call.start, call.stop)
        indices = group_rows.region_indices
        return RowBatch.of_spans(
            self._key_stem,
            group_rows.object_keys[rows],
            group_rows.addresses[rows],
            tuple(self._regions[index].slot_bytes for index in indices),
            tuple(self._region_extents[index] for index in indices),
        )

    def _hand_over(self, attempt: BlobAttempt, batches: list[RowBatch]) -> None:
        """Hands ``attempt``'s calls to the driver in order, up to the first one the driver does
        not take, or the first after a protocol violation was recorded (an earlier call's
        ``on_change`` may record one meanwhile); that call fails as a whole, and the calls after it
        never reach the driver and conclude CANCELLED."""
        taken: list[tuple[_Call, OpHandle]] = []
        for call, rows in zip(attempt._calls, batches):
            # Read without the lock: a violation recorded a moment later lets one more call
            # through, which is then held like any call already in the driver.
            violation = self._protocol_violation
            if violation is not None:
                reason = f"not handed over: the store driver broke its protocol ({violation})"
                self._fail_call(call, reason, this_call_broke_protocol=False)
                break
            op = self._hand_over_call(call, rows)
            if op is None:
                break
            taken.append((call, op))
        for call in attempt._calls[len(taken) + 1 :]:  # past the one the driver did not take
            self._record(call, _CallResults.none_delivered(len(call)), access_ended=True)
        with self._lock:
            for call, op in taken:
                call.op = op
            # The progress thread cancels only calls whose handle it knows, so a stop that came
            # while the calls were being handed over is forwarded from here.
            if attempt._stopped and self._driver_cancels:
                self._poke_progress()

    def _hand_over_call(self, call: _Call, rows: RowBatch) -> OpHandle | None:
        """The driver's handle for ``call``, or None when the driver did not take it; the failure
        is then recorded."""
        on_change: OnChange = functools.partial(self._query_and_record, call)
        attempt = call.attempt
        try:
            if not attempt.is_fetch:
                op = self._driver.put(rows, on_change=on_change)
            elif attempt._hint is None:
                op = self._driver.get(rows, on_change=on_change)
            else:
                op = self._driver.get(rows, attempt._hint, on_change=on_change)
        except DriverRejected as error:  # certifies that nothing escaped
            self._fail_call(call, f"refused: {error}", this_call_broke_protocol=False)
            return None
        except Exception as error:  # outside the driver protocol: the rows may be accessed for good
            self._fail_call(
                call, f"the store driver raised {error!r}", this_call_broke_protocol=True
            )
            return None
        if op is None:  # outside the driver protocol too: nothing to query or cancel
            self._fail_call(
                call, "the store driver returned no handle", this_call_broke_protocol=True
            )
        return op

    # Recording, on the thread that calls ``on_change``, or on the submitting one for a call the
    # driver did not take.

    def _query_and_record(self, call: _Call, op: OpHandle) -> None:
        """``call``'s ``on_change``: queries its operation through the driver and records what the
        queries show. A record can stop the attempt, land its outcome and quiescence, and latch a
        protocol violation."""
        try:
            results = self._driver.poll(op)
            access_ended = self._driver.access_ended(op)
            unprovable = not access_ended and self._driver.access_unprovable(op)
            count, all_concluded = len(results), concluded(results)
        except Exception as error:  # the queries never raise for a handle the driver issued
            self._fail_call(
                call, f"a store driver query raised {error!r}", this_call_broke_protocol=True
            )
            return
        if count != len(call):
            reason = f"the store driver gave {count} results for {len(call)} rows"
            self._fail_call(call, reason, this_call_broke_protocol=True)
            return
        # Read row by row here, outside the lock the submitting thread takes.
        call_results = _CallResults.of(results) if all_concluded else None
        self._record(call, call_results, access_ended, unprovable)

    def _fail_call(self, call: _Call, reason: str, this_call_broke_protocol: bool) -> None:
        """Records ``call`` as failed as a whole: every row faults with ``reason``.

        A call the driver refused, or never saw, touched nothing, so its access ended. A call on
        which the driver broke its protocol is held as unprovable (unless its access end was
        recorded already): nothing the driver says about it later is taken as evidence. Calls
        already in the driver keep being recorded from its answers, and every later submission is
        rejected."""
        with self._lock:
            first_violation = this_call_broke_protocol and self._protocol_violation is None
            if first_violation:
                self._protocol_violation = reason
            recorded = self._record_locked(
                call,
                _CallResults.none_delivered(len(call), reason),
                access_ended=not this_call_broke_protocol,
                unprovable=this_call_broke_protocol,
                whole_call_failed=True,
            )
        if first_violation:
            logger.warning(
                f"blob backend: the store driver broke its protocol ({reason}); every later "
                "submission is rejected"
            )
        self._after_record(call.attempt, recorded)

    def _record(
        self,
        call: _Call,
        results: _CallResults | None,
        access_ended: bool,
        unprovable: bool = False,
    ) -> None:
        with self._lock:
            recorded = self._record_locked(call, results, access_ended, unprovable)
        self._after_record(call.attempt, recorded)

    def _after_record(self, attempt: BlobAttempt, recorded: _Recorded) -> None:
        """What a record leaves to do once the lock is released: a log line for a call that turned
        unprovable, and the holder's ``wake``. Neither runs under the lock, so a slow log sink or
        ``wake`` never holds up the submitting thread."""
        if recorded.newly_unprovable:
            logger.warning_once(
                f"blob backend: a store call of {attempt.label} can never prove its access "
                "ended; its slots stay held",
                key=(self._name, "unprovable"),
            )
        if recorded.wake_due:
            self._call_wake()

    # Bookkeeping, under the lock.

    def _record_locked(
        self,
        call: _Call,
        results: _CallResults | None,
        access_ended: bool,
        unprovable: bool,
        whole_call_failed: bool = False,
    ) -> _Recorded:
        """Records ``call``'s results (None while not all concluded), its access end and whether
        that end can never be proven. Only adds evidence, so a stale or repeated record changes
        nothing; a recorded access end stands against a later claim that it can never be proven,
        and an unprovable call stays held whatever the driver says of it later. The caller passes
        what it returns to ``_after_record`` once it released the lock."""
        attempt = call.attempt
        if results is not None and not call.results_recorded:
            call.results_recorded = True
            attempt._calls_awaiting_results -= 1
            fault_stops_attempt = attempt._record_results(call, results, whole_call_failed)
            if fault_stops_attempt:
                self._stop(attempt)
        if access_ended and not call.access_end_recorded and not call.unprovable:
            call.access_end_recorded = True
            attempt._calls_awaiting_access_end -= 1
        newly_unprovable = unprovable and not call.unprovable and not call.access_end_recorded
        if newly_unprovable:
            call.unprovable = True
        return _Recorded(self._conclude_or_quiesce_and_retire(attempt), newly_unprovable)

    def _conclude_or_quiesce_and_retire(self, attempt: BlobAttempt) -> bool:
        """Under the lock: lands ``attempt``'s outcome and quiescence once due, and drops it from
        ``_attempts`` once it has both. True when the caller is to call ``wake``."""
        wake_due = attempt._conclude_or_quiesce()
        if attempt._outcome is not None and attempt._quiesced:
            self._attempts.discard(attempt)
            if self._shutdown_requested:  # the teardown may be one round away
                self._poke_progress()
        # Every attempt change a shutdown waits for passes through here; the progress thread
        # notifies the end of the teardown itself.
        self._shutdown_recheck.notify_all()
        return wake_due

    def _cancel(self, attempt: BlobAttempt) -> None:
        """Stops ``attempt`` unless its outcome already landed. Stopping changes no counter, so it
        lands neither the outcome nor quiescence itself."""
        with self._lock:
            if attempt._outcome is None:
                self._stop(attempt)

    def _stop(self, attempt: BlobAttempt) -> None:
        """Under the lock: none of ``attempt``'s calls is worth finishing any more. When the driver
        declares ``cancel``, the progress thread asks it to cancel those still running."""
        if not attempt._stopped:
            attempt._stopped = True
            if self._driver_cancels:
                self._poke_progress()

    def _poke_progress(self) -> None:
        self._progress_poked = True
        self._progress_due.notify()

    # The progress thread: forwards cancels and, after a shutdown, tears down.

    def _progress_loop(self) -> None:
        retry_s = TEARDOWN_RETRY_MIN_S
        while True:
            try:
                self._forward_cancels()
                if self._shutdown_requested and self._try_teardown():
                    with self._lock:
                        self._teardown_done = True
                        self._shutdown_recheck.notify_all()
                    return
            except Exception as error:  # the thread must live on, or the labels never clear
                logger.warning_once(
                    f"blob backend: a progress round raised {error!r}",
                    key=(self._name, "progress", type(error)),
                )
            retry_s = self._wait_for_next_round(retry_s)

    def _forward_cancels(self) -> None:
        """Asks the driver, once per call, to cancel the calls of stopped attempts whose rows have
        not all concluded."""
        if not self._driver_cancels:
            return
        with self._lock:
            due = [
                call
                for attempt in self._attempts
                if attempt._stopped
                for call in attempt._calls
                if call.op is not None and not call.results_recorded and not call.cancel_forwarded
            ]
            for call in due:
                call.cancel_forwarded = True
        for call in due:
            try:
                self._driver.cancel(call.op)
            except Exception as error:  # advisory: the call runs on
                logger.warning_once(
                    f"blob backend: cancelling a store call raised {error!r}",
                    key=(self._name, "cancel", type(error)),
                )

    def _wait_for_next_round(self, retry_s: float) -> float:
        """Waits until poked or, while a teardown may still finish, for at most ``retry_s``; returns
        the retry interval for the wait after it."""
        with self._lock:
            if not self._progress_poked:
                tearing_down = self._shutdown_requested and not self._shutdown_answer_is_final()
                self._progress_due.wait(retry_s if tearing_down else None)
            poked, self._progress_poked = self._progress_poked, False
        return TEARDOWN_RETRY_MIN_S if poked else min(2 * retry_s, TEARDOWN_RETRY_MAX_S)

    def _try_teardown(self) -> bool:
        """True once the driver is drained and the registrations revoked; a step that is not done
        yet is retried next round. The driver is drained from the first round on, as nothing is
        submitted any more: queued work is cancelled and the driver's idle threads exit, even when
        an attempt never quiesces. The registrations are revoked only once every attempt quiesced.
        The driver is never closed here: closing a store client may block for long, and it is
        the step of the owner that opened it, once shutdown returned no labels and the hold went
        back."""
        if not self._drained:
            self._drained = self._driver.drain(0.0)
        with self._lock:
            if any(not attempt._quiesced for attempt in self._attempts):
                return False
        return self._drained and self._binding.try_unregister_all()

    # Shutdown, on the shutdown thread.

    def _refuse_new_work_and_force_outcomes(self) -> bool:
        """Refuses every later submission, forces the shutdown outcome on each attempt not
        concluded and stops it (its calls are cancelled when the driver declares cancel), and pokes
        the progress thread to start the teardown. True when an outcome was forced, so the caller
        calls ``wake``."""
        with self._lock:
            self._shutdown_requested = True
            forced = [attempt for attempt in self._attempts if attempt._outcome is None]
            for attempt in forced:
                self._stop(attempt)
                attempt._force_outcome(SHUTDOWN_FAULT)
                self._conclude_or_quiesce_and_retire(attempt)
            self._poke_progress()  # the progress thread forwards the cancels and tears down
        return bool(forced)

    def _shutdown_answer_is_final(self) -> bool:
        """Under the lock, once closing: the progress thread finished, or the labels are final. An
        attempt with a call whose access can never be proven to end never quiesces, and the
        registrations are revoked only once every attempt quiesced, so when every attempt not yet
        quiesced has such a call, neither its label nor a region's can go any more."""
        if self._teardown_done:
            return True
        unquiesced = [attempt for attempt in self._attempts if not attempt._quiesced]
        return bool(unquiesced) and all(attempt._has_unprovable_call for attempt in unquiesced)

    def _still_accessible_labels(self) -> frozenset[str]:
        with self._lock:
            attempts = {attempt.label for attempt in self._attempts if not attempt._quiesced}
        return frozenset(attempts) | self._binding.live_labels()

    def _log_still_accessible(self, labels: frozenset[str]) -> None:
        if labels == self._last_logged_labels:
            return
        self._last_logged_labels = labels
        if labels:
            logger.warning(f"blob backend: shutting down, still accessible: {sorted(labels)}")
        else:
            logger.info("blob backend: shut down; nothing is accessible any more")

    def _call_wake(self) -> None:
        try:
            self._wake()
        except Exception as error:  # a raising wake must not kill a backend thread
            logger.warning_once(
                f"blob backend: wake raised {error!r}", key=(self._name, "wake", type(error))
            )


def _call_bounds(keys: np.ndarray, step: int) -> list[tuple[int, int]]:
    """Consecutive ``[start, stop)`` runs of at most ``step`` rows, none naming a key twice: a view
    may name one key at two addresses, while a ``RowBatch`` promises the driver no key twice (a get
    of one key into two slots would answer READY for both and write one). Keys are compared as
    bytes in a Python set, not with ``np.unique``, whose sort lets go of the GIL and leaves this
    thread waiting a switch interval behind any busy thread. Distinct keys, the usual case, cut
    into whole steps."""
    count = len(keys)
    if count == 0:
        return []
    rows = cast(
        list[bytes],
        np.ascontiguousarray(keys).view(np.dtype((np.void, keys.shape[1]))).ravel().tolist(),
    )
    if len(set(rows)) == count:
        return [(start, min(start + step, count)) for start in range(0, count, step)]
    bounds: list[tuple[int, int]] = []
    seen: set[bytes] = set()
    start = 0
    for index, key in enumerate(rows):
        if index - start == step or key in seen:
            bounds.append((start, index))
            start, seen = index, set()
        seen.add(key)
    bounds.append((start, count))
    return bounds


def _refuse_shared_destinations(groups: list[_GroupRows], regions: Sequence[SlotRegion]) -> None:
    """Two fetched rows in one slot would let one report READY over the other's bytes. A publish
    only reads its slots, so only a fetch is checked. Spans are checked across all regions and
    groups before any driver call."""
    spans = sorted(
        (address, address + regions[index].slot_bytes)
        for group in groups
        for column, index in enumerate(group.region_indices)
        for address in group.addresses[:, column].tolist()
    )
    if any(start < previous_end for (_, previous_end), (start, _) in zip(spans, spans[1:])):
        raise InvalidSubmission("two rows of a fetch write the same slot")
