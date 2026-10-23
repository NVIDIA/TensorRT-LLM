# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: the public ``base.backend.Backend`` over a private store driver.

Threads. The caller's thread validates a submission and cuts it into batches, one driver call
each and no key twice in one: a view may name one key at two addresses, and a driver refuses a
key that appears twice in one call. ``BlobConfig.workers`` worker threads take batches, fetches
first, and make the driver calls.
One progress thread observes the calls that returned before they settled and, once a shutdown was
asked and every attempt quiesced, drains the driver and revokes the registrations. All bookkeeping
is under one lock; neither the driver nor ``wake`` is ever called under it.

A batch goes: queued -> taken by a worker -> its driver call returns -> settled, at once or later
on the progress thread. It is settled once its row results are recorded and its access end is
recorded or can never be proven ("unprovable"). An unprovable batch holds its slots for good: its
attempt never quiesces and the regions stay registered.

An attempt's outcome lands once none of its batches is queued and every started one has its
results; it quiesces once none is queued and every started one has its access end. A cancel, a
fetch's first fault, a publish's whole-call failure, a driver protocol violation or a shutdown
stops it: its queued batches are dropped, and the started ones run on. A protocol violation (a
driver raise other than ``DriverRejected``, or a wrong result count) also refuses every later
submission.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, NoReturn, Protocol

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
from .keys import key_stem
from .store import (
    DriverRejected,
    OpHandle,
    RegionHandle,
    RegionLayout,
    RegionSpec,
    RegistersRegionsTogether,
    RegistrationError,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreDriver,
    concluded,
    faulted,
)

__all__ = [
    "BlobAttempt",
    "BlobBackend",
    "BlobConfig",
    "ConstructionRejected",
    "RegionBinding",
    "RegionHold",
]

# Fetches sit on the time-to-first-token path and go first. Once this many fetch batches were taken
# while a publish batch waited, that publish batch goes next, so publishes, whose leases pin the
# lender's slots, never starve.
FETCH_BATCHES_PER_PUBLISH_BATCH = 4
PROGRESS_MIN_DELAY_S = 0.001
PROGRESS_MAX_DELAY_S = 0.1
# A driver call may block for long: a store may wait on its transfers without a timeout, or for as
# long as its own transfer timeout. A call past this is logged once; its submission keeps holding
# its slots meanwhile.
STUCK_CALL_WARNING_S = 60.0
SHUTDOWN_FAULT = "blob backend shut down"

_instance_ids = itertools.count()  # names each backend and binding in thread names and logs
_Results = Sequence[RowResult] | None  # None: not concluded yet, or no new evidence


def _warn_once(message: str, key: tuple[object, ...]) -> None:
    """One warning per ``key`` for the process; the logger's own API is untyped."""
    logger.warning_once(message, key=key)  # type: ignore[no-untyped-call]


class _Observation(NamedTuple):
    """What one look at a batch's driver call found."""

    results: _Results  # one per row once concluded; None while not, or when the look failed
    ended: bool  # the call's access to the rows' memory has ended, with evidence
    unprovable: bool  # the driver can never prove that access ended
    # Set when the call failed as a whole and every row carries this reason as its fault.
    whole_call_fault: str | None = None
    # The failure is outside the driver protocol: the backend sends the driver no more work.
    breaks_protocol: bool = False


@dataclass(frozen=True)
class BlobConfig:
    """How the backend cuts submissions into driver calls and runs them; the ``blob`` section of a
    store config.

    The defaults assume nothing about the driver: one worker, and no size limit of the backend's
    own on a call; the driver's ``max_rows_per_op`` still applies, and a repeated key still splits
    a group into several calls. How many calls to overlap and how large to make them depends on
    the store (whether its calls block, what a call costs), so a store's values come from its
    store config.

    Attributes:
        key_prefix: The deployment's key namespace; matches ``[A-Za-z0-9_-]{1,64}``.
        workers: Driver calls at once, each on a backend thread of its own. A driver whose calls
            block until their transfers end needs several to overlap them. Publishes take at most
            ``max(1, workers - 1)`` of them, so unless there is one worker, one is always left for
            fetches.
        rows_per_batch: Most rows in one driver call, or None for no limit of the backend's own.
        bytes_per_batch: Most payload bytes in one driver call, or None for no limit of the
            backend's own; a call always carries at least one row, however large its slot. A call
            holds its submission's slots until it returns, or for good when it can never prove its
            access ended (``BlobBackend.batch_size``).
        max_queued_batches: Most batches waiting for a worker, over every attempt. A submission is
            refused (``SubmissionRejected``) while that many are queued, so a holder that submits
            faster than the store drains meets back-pressure instead of an unbounded backlog. A
            submission accepted below the limit may push the count past it, so no view is too
            large to submit; the batches of a stopped attempt leave the queues at once.
    """

    key_prefix: str = "trtllm-kv"
    workers: int = 1
    rows_per_batch: int | None = None
    bytes_per_batch: int | None = None
    max_queued_batches: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.key_prefix, str):
            raise ValueError(f"key_prefix must be a str, got {self.key_prefix!r}")
        key_stem(self.key_prefix)
        for name in ("workers", "max_queued_batches"):
            if not _is_count(getattr(self, name)):
                raise ValueError(
                    f"{name} must be an int of at least 1, got {getattr(self, name)!r}"
                )
        for name in ("rows_per_batch", "bytes_per_batch"):
            value = getattr(self, name)
            if value is not None and not _is_count(value):
                raise ValueError(f"{name} must be None or an int of at least 1, got {value!r}")


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
    builder thread when ``register`` rolls back a failed registration); releases the hold only after
    every registration is confirmed revoked, on the manager's thread. The driver is never called
    under the binding's lock, so reading the live registrations never waits on a revocation.

    A driver that implements ``RegistersRegionsTogether`` gets every region in one
    ``register_regions`` call and gives them back in one ``unregister_regions``. The regions count
    as live, and the hold stays, from before that call until the batch is confirmed revoked.

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
        self._together = driver if isinstance(driver, RegistersRegionsTogether) else None
        device_memory = driver.capabilities.device_memory
        # The regions register() registers, by index: those with slots in memory the driver can
        # access. A row in any other region is one the backend does not recognize.
        self.served_regions = frozenset(
            index
            for index, region in enumerate(self.regions)
            if region.slot_count > 0 and (region.memory_kind is MemoryKind.HOST or device_memory)
        )
        self._hold = hold
        self._name = f"region-binding-{next(_instance_ids)}"
        self._lock = threading.Lock()
        # Region index -> its handle, or None for a registration that may exist without one.
        self._live: dict[int, RegionHandle | None] = {}

    def register(self) -> None:
        """Registers the regions in ``served_regions``: one ``register`` per region, or one
        ``register_regions`` for all of them when the driver registers regions together.

        Raises:
            ConstructionRejected: Nothing is registered and the hold is released: there is no region
                to register, or a registration failed with nothing escaping and whatever was
                registered before it was revoked.
            RuntimeError: A registration may be live; the hold is kept. ``live_labels`` names the
                regions, ``unregister`` may be called again to revoke them, and ``release_hold``
                gives the hold back once none is left.
        """
        if not self.served_regions:  # a backend that could serve no row
            self._hold.release()
            raise ConstructionRejected("no region has slots in memory the store driver can access")
        if self._together is not None:
            self._register_together(self._together)
            return
        for index in sorted(self.served_regions):
            region = self.regions[index]
            layout = RegionLayout(
                region.slot_bytes, region.slot_count, region.layout_label, region.memory_kind
            )
            try:
                handle = self.driver.register(region.address, region.nbytes, layout)
            except Exception as error:
                self._raise_for_failed_registration(index, error)
            self._keep(index, handle)

    def unregister(self) -> bool:
        """Revokes every live registration it can. True when none is left; a revocation that failed
        stays live for the next call."""
        if self._together is not None:
            return self._unregister_together(self._together)
        with self._lock:
            revocable = [(i, handle) for i, handle in self._live.items() if handle is not None]
        for index, handle in revocable:
            try:
                self.driver.unregister(handle)
            except Exception as error:
                _warn_once(
                    f"blob backend: revoking region:{index} failed, will retry: {error!r}",
                    (self._name, "unregister", index),
                )
                continue
            with self._lock:
                del self._live[index]
        with self._lock:
            return not self._live

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
            return frozenset(f"region:{index}" for index in self._live)

    def _raise_for_failed_registration(self, index: int, error: Exception) -> NoReturn:
        handle = error.handle if isinstance(error, RegistrationError) else None
        nothing_escaped = isinstance(error, DriverRejected) or (
            isinstance(error, RegistrationError) and handle is None
        )
        if not nothing_escaped:
            self._keep(index, handle)
            raise RuntimeError(f"region:{index} may be registered after {error!r}") from error
        if not self.unregister():
            raise RuntimeError(
                f"registering region:{index} failed and an earlier region stays registered: {error}"
            ) from error
        # Nothing is registered or accessible, and construction precedes the executor loop, so the
        # builder thread may still give the hold back.
        self._hold.release()
        raise ConstructionRejected(f"registering region:{index} failed: {error}") from error

    def _keep(self, index: int, handle: RegionHandle | None) -> None:
        with self._lock:
            self._live[index] = handle

    # The batch path, for a driver that registers regions together.

    def _register_together(self, driver: RegistersRegionsTogether) -> None:
        indices = sorted(self.served_regions)
        try:
            specs = [self._spec(index) for index in indices]
        except ValueError as error:  # a region the boundary cannot describe; the store not asked
            self._hold.release()
            raise ConstructionRejected(f"bad region geometry: {error}") from error
        owned: dict[int, RegionHandle | None] = dict.fromkeys(indices)
        with self._lock:  # every region has an owner before the store is asked
            self._live = owned
        try:
            handle = driver.register_regions(specs)
        except Exception as error:
            self._raise_for_failed_batch(error)
        self._keep_all(handle)

    def _unregister_together(self, driver: RegistersRegionsTogether) -> bool:
        with self._lock:
            handles: list[RegionHandle] = []
            for handle in self._live.values():
                if handle is not None and not any(handle is seen for seen in handles):
                    handles.append(handle)
        for handle in handles:  # the one batch handle every region shares
            try:
                driver.unregister_regions(handle)
            except Exception as error:
                _warn_once(
                    f"blob backend: revoking the regions together failed, will retry: {error!r}",
                    (self._name, "unregister_regions"),
                )
                continue
            with self._lock:
                self._live = {i: h for i, h in self._live.items() if h is not handle}
        with self._lock:
            return not self._live

    def _raise_for_failed_batch(self, error: Exception) -> NoReturn:
        handle = error.handle if isinstance(error, RegistrationError) else None
        nothing_escaped = isinstance(error, DriverRejected) or (
            isinstance(error, RegistrationError) and handle is None
        )
        if nothing_escaped:
            with self._lock:
                self._live.clear()
            self._hold.release()
            raise ConstructionRejected(
                f"registering the regions together failed: {error}"
            ) from error
        if handle is None:  # some may be registered, with nothing to revoke them by
            raise RuntimeError(f"the regions may be registered after {error!r}") from error
        self._keep_all(handle)
        if not self.unregister():
            raise RuntimeError(
                f"registering the regions together failed and some may stay registered: {error}"
            ) from error
        self._hold.release()
        raise ConstructionRejected(f"registering the regions together failed: {error}") from error

    def _spec(self, index: int) -> RegionSpec:
        region = self.regions[index]
        layout = RegionLayout(
            region.slot_bytes, region.slot_count, region.layout_label, region.memory_kind
        )
        return RegionSpec(region.address, region.nbytes, layout)

    def _keep_all(self, handle: RegionHandle) -> None:
        with self._lock:
            for index in self._live:
                self._live[index] = handle


class _GroupRows(NamedTuple):
    """One group's arrays, taken from the view at submission."""

    object_keys: np.ndarray  # uint8 [rows, width]
    addresses: np.ndarray  # int64 [rows]
    region: SlotRegion


@dataclass(eq=False)
class _Batch:
    """Rows ``[start, stop)`` of one group of an attempt; the unit of one driver call."""

    attempt: BlobAttempt
    group: int  # position in the view's groups
    start: int
    stop: int
    op: OpHandle | None = None
    results_recorded: bool = False  # set once, under the backend's lock
    access_recorded: bool = False  # set once, under the backend's lock
    unprovable: bool = False  # the driver can never prove its access ended
    cancel_forwarded: bool = False  # the driver's cancel was called for op; at most once

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def settled(self) -> bool:
        """Nothing more to learn from the driver: results recorded, and access ended or never
        provable. An unprovable batch keeps its access open, so its attempt never quiesces."""
        return self.results_recorded and (self.access_recorded or self.unprovable)


class BlobAttempt:
    """The handle of one submission: a ``base.backend.Attempt``.

    Its state changes only under the backend's lock; ``poll`` and ``quiesced`` read one reference
    without it, so they never block.

    Three counters track its batches:

    - ``_queued_batches``: still in a queue. A stop zeroes it at once and takes the batches out
      of the queues (``BlobBackend._drop_queued``), so nothing of a stopped attempt is left for a
      worker.
    - ``_results_pending``: started, row results not recorded yet.
    - ``_access_open``: started, access end not recorded yet. An unprovable batch never leaves it.

    The outcome lands once ``_queued_batches`` and ``_results_pending`` are 0; the attempt quiesces
    once ``_queued_batches`` and ``_access_open`` are 0.
    """

    def __init__(
        self,
        backend: BlobBackend,
        label: str,
        is_fetch: bool,
        groups: list[_GroupRows],
        hint: Hint = None,
    ) -> None:
        self.label = label
        """The attempt's label among ``BlobBackend.shutdown``'s, ``attempt:<n>``."""
        self.is_fetch = is_fetch
        self._backend = backend
        self._groups: list[_GroupRows] | None = groups
        self._hint = hint  # a fetch's source, for every driver get; immutable, so kept as given
        self._masks = [np.zeros(len(rows.addresses), dtype=bool) for rows in groups]
        self._queued_batches = 0  # see the class docstring for the three counters
        self._results_pending = 0
        self._access_open = 0
        # A batch can never prove its access ended, so this attempt never quiesces.
        self._has_unprovable_batch = False
        self._fault: str | None = None  # the first fault; _record_results says when it stops
        # Set by a cancel, a stopping fault, a driver protocol violation or a shutdown.
        self._stopped = False
        self._cancel_requested = False  # set by a cancel only; forwarded to started batches
        self._outcome: Outcome | None = None
        self._quiesced = False

    def poll(self) -> Outcome | None:
        return self._outcome

    def quiesced(self) -> bool:
        return self._quiesced

    def cancel(self) -> None:
        """Stops dispatching the attempt's queued batches and, when the driver declares
        ``cancel``, forwards the cancel once to every started batch without a result yet, a
        batch whose handle arrives later included. Started batches otherwise run on.

        Advisory either way: a cancel is no access-end evidence, and the attempt quiesces only
        once every started batch proved its access ended.
        """
        self._backend._cancel(self)

    def _record_results(
        self, batch: _Batch, results: Sequence[RowResult], whole_call_failed: bool
    ) -> None:
        self._masks[batch.group][batch.start : batch.stop] = [
            result.status is RowStatus.READY for result in results
        ]
        for row, result in enumerate(results, batch.start):
            if result.fault is not None:
                self._fault = self._fault or f"group {batch.group} row {row}: {result.fault}"
                # A fetch stops at its first fault: a prefix reuse has no use for the rows past
                # it. A publish's rows stand alone, so only a call that failed as a whole stops it.
                if self.is_fetch or whole_call_failed:
                    self._stop()
                return

    def _stop(self) -> None:
        self._stopped = True
        self._queued_batches = 0

    def _conclude_or_quiesce(self) -> bool:
        """Lands the outcome and turns quiesced once their conditions hold; True when either did."""
        changed = False
        # After a fault the outcome still waits for the results of batches already started: with
        # several workers a later batch can fail first, and freezing then would drop the earlier
        # rows that a prefix reuse needs.
        if self._outcome is None and self._queued_batches == 0 and self._results_pending == 0:
            for mask in self._masks:  # final, so the outcome keeps them without a copy
                mask.flags.writeable = False
            self._outcome = Outcome(tuple(self._masks), self._fault)
            changed = True
        if not self._quiesced and self._queued_batches == 0 and self._access_open == 0:
            self._quiesced = True
            self._groups = None  # nothing reads the view's arrays once the attempt quiesced
            changed = True
        return changed

    def _force_outcome(self, reason: str) -> None:
        self._stop()
        fault = reason if self._fault is None else f"{reason}; first fault: {self._fault}"
        # Still writable, so ``Outcome`` copies them: batches already started go on writing
        # their results into these masks.
        self._outcome = Outcome(tuple(self._masks), fault)


class BlobBackend:
    """Moves a view's rows between the lender's registered regions and a byte store: a
    ``base.backend.Backend``. A process may hold several, one per lender, each with its own driver.

    Args:
        binding: The lender's regions, already registered with ``binding.driver``. The backend
            revokes the registrations once nothing can access them after shutdown.
        wake: Called outside every lock whenever an attempt's outcome lands or it quiesces, once
            the change shows through ``poll`` or ``quiesced``.
        config: How submissions are cut into driver calls.

    Raises:
        RuntimeError: A backend thread could not start. Nothing of the backend runs afterwards:
            the workers already started have left. The binding keeps its registrations and the
            hold for whoever built it to revoke (``RegionBinding.unregister``) and release.
    """

    def __init__(self, binding: RegionBinding, wake: WakeFn, config: BlobConfig) -> None:
        self._binding = binding
        self._driver = binding.driver
        self._regions = binding.regions
        self._wake = wake
        self._config = config
        self._key_stem = key_stem(config.key_prefix)
        self._lock = threading.Lock()
        self._work_ready = threading.Condition(self._lock)
        self._progress_due = threading.Condition(self._lock)
        # ``shutdown`` waits on this; notified whenever its answer may have changed: an attempt
        # concluded or quiesced, or the progress thread finished the teardown.
        self._shutdown_recheck = threading.Condition(self._lock)
        self._progress_poked = False
        self._fetch_queue: deque[_Batch] = deque()
        self._publish_queue: deque[_Batch] = deque()
        self._fetches_in_a_row = 0
        self._watched: set[_Batch] = set()
        self._attempts: set[BlobAttempt] = set()  # until concluded and quiesced
        self._attempt_ids = itertools.count()
        self._shutdown_requested = False
        self._drained = False
        self._teardown_done = False  # drained and every registration revoked
        self._protocol_violation: str | None = None  # how the driver first broke its protocol
        self._driver_call_started: dict[_Batch, float] = {}  # batch -> when its driver call started
        self._reported_labels: frozenset[str] | None = None
        self._name = name = f"blob-backend-{next(_instance_ids)}"
        driver_capabilities = self._driver.capabilities
        self._max_rows_per_op = driver_capabilities.max_rows_per_op
        self._driver_reads_hints = driver_capabilities.hints
        self._driver_cancels = driver_capabilities.cancel
        self._capabilities = Capabilities(
            fetch=driver_capabilities.get, publish=driver_capabilities.put, cancel=True
        )
        self._unprovable_batches = 0
        self._workers_alive = config.workers
        self._publish_limit = max(1, config.workers - 1)
        self._publishes_running = 0
        self._workers = [
            threading.Thread(target=self._worker_loop, name=f"{name}-worker-{i}", daemon=True)
            for i in range(config.workers)
        ]
        self._progress = threading.Thread(
            target=self._progress_loop, name=f"{name}-progress", daemon=True
        )
        started: list[threading.Thread] = []
        try:
            for thread in (*self._workers, self._progress):
                thread.start()
                started.append(thread)
        except RuntimeError:  # the system refused a thread: the backend never runs
            self._abandon(started)
            raise

    def _abandon(self, started: list[threading.Thread]) -> None:
        """Ends the workers started before a later thread failed to start. The progress thread
        starts last, so it is never among them: no teardown runs, and the binding stays with its
        owner, registrations and hold included."""
        with self._lock:
            self._shutdown_requested = True
            self._workers_alive = len(started)
            self._work_ready.notify_all()
        for thread in started:
            thread.join()

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
        or only until the answer can no longer change: every attempt still open holds a batch whose
        access can never be proven to end.

        Never calls the driver: the progress thread drains and revokes the registrations once every
        worker has left and every attempt quiesced, since revocation can be slow (RDMA deregisters
        on every NIC) and a call with no budget left must still return at once.
        """
        # Bounded so that the waits below accept it: an infinite budget must not raise.
        deadline = time.monotonic() + min(max(0.0, timeout), threading.TIMEOUT_MAX)
        if self._close_and_force_outcomes():
            self._call_wake()
        with self._lock:
            self._shutdown_recheck.wait_for(
                self._shutdown_answer_is_final, max(0.0, deadline - time.monotonic())
            )
            teardown_done = self._teardown_done
        if teardown_done:  # the progress thread is returning
            self._progress.join(max(0.0, deadline - time.monotonic()))
        labels = self._labels()
        self._report(labels)
        return labels

    def batch_size(self, region_index: int) -> tuple[int, int] | None:
        """The most rows of region ``region_index`` one driver call carries, and their bytes; None
        means no configured or driver-imposed size limit, while a repeated key may still split a
        group into several calls.

        An attempt quiesces only once every batch it started proved its access ended, so one call
        that never proves it keeps the slots of its whole submission; a holder bounds that by
        submitting at most this many rows of the region at a time.

        Raises:
            ValueError: The backend did not register the region.
        """
        if region_index not in self._binding.served_regions:
            raise ValueError(f"region {region_index} is not registered")
        region = self._regions[region_index]
        rows = self._rows_per_batch(region)
        return None if rows is None else (rows, rows * region.slot_bytes)

    def stats(self) -> dict[str, int]:
        """A snapshot of the work in flight, for logs and tests."""
        now = time.monotonic()
        with self._lock:
            return {
                "attempts_active": len(self._attempts),
                "batches_queued": sum(a._queued_batches for a in self._attempts),
                "batches_in_queues": len(self._fetch_queue) + len(self._publish_queue),
                "batches_access_open": sum(a._access_open for a in self._attempts),
                "batches_unprovable": self._unprovable_batches,
                "oldest_driver_call_ms": int(
                    1e3 * (now - min(self._driver_call_started.values(), default=now))
                ),
            }

    # Submission, in the call sequence.

    def _submit(self, view: View, hint: Hint, is_fetch: bool) -> BlobAttempt:
        if hint is not None and not is_fetch:
            raise InvalidSubmission(f"a publish takes only a None destination, got {hint!r}")
        if hint is not None and not self._driver_reads_hints:
            raise InvalidSubmission(f"the store driver reads no hints, got {hint!r}")
        if not (self._capabilities.fetch if is_fetch else self._capabilities.publish):
            raise InvalidSubmission(f"the store driver cannot {'get' if is_fetch else 'put'}")
        groups = [self._group_rows(group) for group in view.groups]
        if is_fetch:
            _refuse_shared_destinations(groups)
        label = f"attempt:{next(self._attempt_ids)}"
        attempt = BlobAttempt(self, label, is_fetch, groups, hint)
        batches = self._cut(attempt, groups)
        with self._lock:
            if self._shutdown_requested:
                raise SubmissionRejected("the blob backend is shut down")
            if self._protocol_violation is not None:
                raise SubmissionRejected(
                    f"the store driver broke its protocol: {self._protocol_violation}"
                )
            queued = sum(a._queued_batches for a in self._attempts)
            if queued >= self._config.max_queued_batches:
                raise SubmissionRejected(
                    f"{queued} batches are waiting for a worker (max_queued_batches="
                    f"{self._config.max_queued_batches})"
                )
            self._attempts.add(attempt)
            attempt._queued_batches = len(batches)
            (self._fetch_queue if is_fetch else self._publish_queue).extend(batches)
            self._work_ready.notify(len(batches))
            wake_due = self._conclude_or_quiesce(attempt)
        if wake_due:
            self._call_wake()
        return attempt

    def _group_rows(self, group: LayerGroupRows) -> _GroupRows:
        where = f"layer group {group.layer_group_index}"
        if group.object_keys is None or group.addresses is None or group.region_index is None:
            raise InvalidSubmission(f"{where}: rows need object keys, addresses and a region")
        keys = np.asarray(group.object_keys)
        addresses = np.asarray(group.addresses)
        rows = len(group)
        if keys.dtype != np.uint8 or keys.ndim != 2 or keys.shape[0] != rows or keys.shape[1] == 0:
            raise InvalidSubmission(
                f"{where}: object keys must be uint8 [{rows}, width > 0], got {keys.dtype} "
                f"{keys.shape}"
            )
        if addresses.dtype != np.int64 or addresses.shape != (rows,):
            raise InvalidSubmission(
                f"{where}: addresses must be int64 [{rows}], got {addresses.dtype} "
                f"{addresses.shape}"
            )
        if group.region_index not in self._binding.served_regions:  # refused, never skipped
            raise InvalidSubmission(f"{where}: region {group.region_index} is not registered")
        region = self._regions[group.region_index]
        slot, misalignment = np.divmod(addresses - region.address, region.slot_bytes)
        if np.any((slot < 0) | (slot >= region.slot_count) | (misalignment != 0)):
            raise InvalidSubmission(f"{where}: an address is not a slot of its region")
        return _GroupRows(keys, addresses, region)

    def _cut(self, attempt: BlobAttempt, groups: list[_GroupRows]) -> list[_Batch]:
        batches = []
        for group, group_rows in enumerate(groups):
            count = len(group_rows.addresses)
            step = self._rows_per_batch(group_rows.region) or max(count, 1)  # None: no size limit
            batches += [
                _Batch(attempt, group, start, stop)
                for start, stop in _batch_bounds(group_rows.object_keys, step)
            ]
        return batches

    def _rows_per_batch(self, region: SlotRegion) -> int | None:
        """The most rows of ``region`` in one driver call, at least one; None when nothing limits
        them."""
        bytes_per_batch = self._config.bytes_per_batch
        limits = [
            limit
            for limit in (
                self._config.rows_per_batch,
                None if bytes_per_batch is None else bytes_per_batch // region.slot_bytes,
                self._max_rows_per_op,
            )
            if limit is not None
        ]
        return max(1, min(limits)) if limits else None

    def _cancel(self, attempt: BlobAttempt) -> None:
        with self._lock:
            if attempt._outcome is not None:
                return
            attempt._cancel_requested = True
            attempt._stop()
            self._drop_queued(attempt)
            # Started batches whose handle is back and whose result is not: the driver may still
            # spare their rows. They sit in _driver_call_started until their worker finishes the
            # call, then in _watched; a batch whose handle is not back yet gets its cancel when it
            # lands.
            started = [
                batch.op
                for batch in itertools.chain(self._watched, self._driver_call_started)
                if batch.attempt is attempt and self._take_cancel(batch)
            ]
            wake_due = self._conclude_or_quiesce(attempt)
        for op in started:
            self._forward_cancel(op)
        if wake_due:
            self._call_wake()

    def _take_cancel(self, batch: _Batch) -> bool:
        """Under the lock: whether this call forwards the cancel for ``batch``, once per batch and
        only to a driver that declares cancel, for a batch with a handle and no result yet."""
        if not self._driver_cancels or batch.op is None:
            return False
        if batch.results_recorded or batch.cancel_forwarded:
            return False
        batch.cancel_forwarded = True
        return True

    def _forward_cancel(self, op: OpHandle) -> None:
        try:
            self._driver.cancel(op)
        except Exception as error:  # noqa: BLE001 - advisory: a failed cancel changes nothing
            _warn_once(
                f"blob backend: cancelling a store operation raised {error!r}",
                (self._name, "cancel", type(error)),
            )

    def _drop_queued(self, attempt: BlobAttempt) -> None:
        """Under the lock: takes a stopped attempt's batches out of the queues, so submitting and
        cancelling over and over cannot pile up entries the workers would only skip."""
        for queue in (self._fetch_queue, self._publish_queue):
            if any(batch.attempt is attempt for batch in queue):
                kept = [batch for batch in queue if batch.attempt is not attempt]
                queue.clear()
                queue.extend(kept)

    # Workers: up to ``config.workers`` driver calls at once, each on a thread of its own.

    def _worker_loop(self) -> None:
        try:
            while (batch := self._take()) is not None:
                self._run(batch)
        finally:
            with self._lock:
                self._workers_alive -= 1
                self._poke_progress()

    def _take(self) -> _Batch | None:
        with self._lock:
            while True:
                batch = self._pop_next()
                if batch is None:
                    if self._shutdown_requested:
                        return None
                    self._work_ready.wait()
                    continue
                attempt = batch.attempt
                # A stopped attempt has no batch left here: the stop dropped them under this lock.
                if not attempt.is_fetch:
                    self._publishes_running += 1
                # Counts only the fetches a waiting publish let go first.
                publish_waits = bool(self._publish_queue)
                fetch_ahead = attempt.is_fetch and publish_waits
                self._fetches_in_a_row = self._fetches_in_a_row + 1 if fetch_ahead else 0
                # Counted as started before the driver call: while the call runs, _access_open
                # alone keeps the attempt from quiescing once a stop has zeroed _queued_batches.
                attempt._queued_batches -= 1
                attempt._results_pending += 1
                attempt._access_open += 1
                self._driver_call_started[batch] = time.monotonic()
                return batch

    def _pop_next(self) -> _Batch | None:
        publish_due = (
            self._publish_queue
            and self._publishes_running < self._publish_limit
            and (not self._fetch_queue or self._fetches_in_a_row >= FETCH_BATCHES_PER_PUBLISH_BATCH)
        )
        if publish_due:
            return self._publish_queue.popleft()
        return self._fetch_queue.popleft() if self._fetch_queue else None

    def _run(self, batch: _Batch) -> None:
        with self._lock:
            violation = self._protocol_violation
        if violation is not None:  # taken before the break was recorded: never dispatched
            reason = f"the store driver broke its protocol: {violation}"
            self._finish(batch, _refused(batch, reason))
            return
        try:
            row_ios = self._row_ios(batch)
        except Exception as error:  # the driver was never called: nothing escaped
            self._finish(batch, _refused(batch, f"refused: {error}"))
            return
        try:
            op = self._call_driver(batch.attempt, row_ios)
        except DriverRejected as error:  # certifies that nothing escaped
            self._finish(batch, _refused(batch, f"refused: {error}"))
            return
        except Exception as error:  # outside the driver protocol: the rows may be accessed for good
            reason = f"the store driver raised {error!r}"
            self._finish(
                batch,
                _Observation(
                    _faults(batch, reason),
                    ended=False,
                    unprovable=True,
                    whole_call_fault=reason,
                    breaks_protocol=True,
                ),
            )
            return
        with self._lock:
            # The handle lands under the lock, so a cancel from now on finds it in
            # _driver_call_started; a cancel that came while the call was out is forwarded here.
            batch.op = op
            forward = batch.attempt._cancel_requested and self._take_cancel(batch)
        if forward:
            self._forward_cancel(op)
        self._finish(batch, self._observe(batch))

    def _call_driver(self, attempt: BlobAttempt, row_ios: list[RowIO]) -> OpHandle:
        if not attempt.is_fetch:
            return self._driver.put(row_ios)
        if attempt._hint is None:
            return self._driver.get(row_ios)
        return self._driver.get(row_ios, attempt._hint)

    def _row_ios(self, batch: _Batch) -> list[RowIO]:
        groups = batch.attempt._groups
        assert groups is not None  # cleared only once the attempt quiesced, after every batch
        group_rows = groups[batch.group]
        slot_bytes, stem = group_rows.region.slot_bytes, self._key_stem
        names = group_rows.object_keys[batch.start : batch.stop]
        hexed, width = names.tobytes().hex(), 2 * names.shape[1]  # one pass for the batch
        addresses = group_rows.addresses[batch.start : batch.stop].tolist()
        return [
            RowIO(stem + hexed[i * width : (i + 1) * width], (Span(address, slot_bytes),))
            for i, address in enumerate(addresses)
        ]

    def _observe(self, batch: _Batch) -> _Observation:
        """The driver's results (when concluded), access end and unprovability for ``batch``."""
        try:
            results = self._driver.poll(batch.op)
            ended = self._driver.access_ended(batch.op)
            unprovable = not ended and self._driver.access_unprovable(batch.op)
            count, done = len(results), concluded(results)
        except Exception as error:  # no new evidence this time
            _warn_once(
                f"blob backend: observing a store operation raised {error!r}",
                (self._name, "observe", type(error)),
            )
            return _NO_NEW_EVIDENCE
        if count != len(batch):  # outside the driver protocol: no later answer proves an access end
            reason = f"the store driver gave {count} results for {len(batch)} rows"
            return _Observation(
                _faults(batch, reason),
                ended=False,
                unprovable=not batch.access_recorded,
                whole_call_fault=reason,
                breaks_protocol=True,
            )
        return _Observation(results if done else None, ended=ended, unprovable=unprovable)

    def _finish(self, batch: _Batch, observed: _Observation) -> None:
        with self._lock:
            del self._driver_call_started[batch]
            # No notify when a publish ends: this worker calls _take next and uses the publish room
            # it freed itself; an idle worker waits only while nothing it may take is queued.
            if not batch.attempt.is_fetch:
                self._publishes_running -= 1
            wake_due = self._record(batch, observed)
            if batch.op is not None and not batch.settled:
                self._watched.add(batch)
                self._poke_progress()
        if wake_due:
            self._call_wake()

    # Bookkeeping, under the lock.

    def _record(self, batch: _Batch, observed: _Observation) -> bool:
        attempt = batch.attempt
        if observed.results is not None and not batch.results_recorded:
            batch.results_recorded = True
            attempt._results_pending -= 1
            whole_call_failed = observed.whole_call_fault is not None
            stopped_before = attempt._stopped
            attempt._record_results(batch, observed.results, whole_call_failed=whole_call_failed)
            if attempt._stopped and not stopped_before:  # a stopping fault: its queue goes too
                self._drop_queued(attempt)
        # An unprovable batch is held for good, whatever the driver says of it later: its attempt
        # never quiesces, which is what lets a shutdown answer before its budget runs out.
        if observed.ended and not batch.access_recorded and not batch.unprovable:
            batch.access_recorded = True
            attempt._access_open -= 1
        if observed.unprovable and not batch.unprovable:  # counted; polled again only for results
            batch.unprovable = True
            attempt._has_unprovable_batch = True
            self._unprovable_batches += 1
            _warn_once(
                f"blob backend: a store call of {attempt.label} can never prove its access "
                "ended; its slots stay held",
                (self._name, "unprovable"),
            )
        if batch.settled:
            self._watched.discard(batch)
        others_changed = False
        if observed.breaks_protocol and self._protocol_violation is None:
            violation = observed.whole_call_fault or "an unnamed protocol violation"
            others_changed = self._stop_after_protocol_violation(violation)
        return self._conclude_or_quiesce(attempt) or others_changed

    def _stop_after_protocol_violation(self, violation: str) -> bool:
        """The driver broke its protocol: refuses every later submission and stops every attempt
        that still has batches queued, so none of them reaches the driver; that attempt's outcome
        carries the violation as its fault unless it already had one. Batches already in the
        driver run on and still record their rows. True when an attempt concluded or quiesced."""
        self._protocol_violation = violation
        logger.warning(  # type: ignore[no-untyped-call]
            f"blob backend: the store driver broke its protocol ({violation}); every later "
            "submission is refused and queued batches are dropped"
        )
        fault = f"the store driver broke its protocol: {violation}"
        changed = False
        for attempt in list(self._attempts):  # _conclude_or_quiesce may remove from the set
            if attempt._queued_batches:
                attempt._fault = attempt._fault or fault
                attempt._stop()
                self._drop_queued(attempt)
                changed = self._conclude_or_quiesce(attempt) or changed
        return changed

    def _conclude_or_quiesce(self, attempt: BlobAttempt) -> bool:
        wake_due = attempt._conclude_or_quiesce()
        if attempt._outcome is not None and attempt._quiesced:
            self._attempts.discard(attempt)
        self._shutdown_recheck.notify_all()  # every change a shutdown waits for passes through here
        return wake_due

    def _poke_progress(self) -> None:
        self._progress_poked = True
        self._progress_due.notify()

    # The progress thread: the only one that observes an operation after its worker let go.

    def _progress_loop(self) -> None:
        delay = PROGRESS_MIN_DELAY_S
        while True:
            try:
                self._observe_watched()
                self._report_stuck_calls()
                if self._shutdown_requested and self._try_teardown():
                    with self._lock:
                        self._teardown_done = True
                        self._shutdown_recheck.notify_all()
                    return
            except Exception as error:  # the thread must live on, or the labels never clear
                _warn_once(
                    f"blob backend: a progress round raised {error!r}",
                    (self._name, "progress", type(error)),
                )
            delay = self._wait_for_next_round(delay)

    def _observe_watched(self) -> None:
        with self._lock:
            batches = list(self._watched)
        if not batches:
            return
        observations = [(batch, self._observe(batch)) for batch in batches]
        with self._lock:
            wake_due = [self._record(batch, observed) for batch, observed in observations]
        if any(wake_due):
            self._call_wake()

    def _report_stuck_calls(self) -> None:
        now = time.monotonic()
        with self._lock:
            stuck = [
                b
                for b, started in self._driver_call_started.items()
                if now - started > STUCK_CALL_WARNING_S
            ]
        for batch in stuck:
            _warn_once(
                f"blob backend: a store call of {batch.attempt.label} has run for over "
                f"{STUCK_CALL_WARNING_S:g} s; its slots stay held until it returns",
                (self._name, "stuck", batch.attempt.label, batch.group, batch.start),
            )

    def _wait_for_next_round(self, delay: float) -> float:
        """Waits ``delay`` before the next progress round, or less when poked; returns the next
        delay. It doubles, up to PROGRESS_MAX_DELAY_S, while watched batches stay unsettled, and a
        poke resets it."""
        with self._lock:
            if not self._progress_poked:
                # Idle, the thread wakes only as often as stuck driver calls need reporting.
                busy = self._watched or self._shutdown_requested
                self._progress_due.wait(delay if busy else STUCK_CALL_WARNING_S)
            poked, self._progress_poked = self._progress_poked, False
        return PROGRESS_MIN_DELAY_S if poked else min(2 * delay, PROGRESS_MAX_DELAY_S)

    def _try_teardown(self) -> bool:
        """True once the driver is drained and the registrations revoked; a step that is not done
        yet is retried next round. The driver is never closed here: whoever opened it closes it,
        after this revocation, since a close may hold the GIL or send RPCs and is the opener's to
        schedule."""
        with self._lock:
            if self._workers_alive or any(not attempt._quiesced for attempt in self._attempts):
                return False
        for worker in self._workers:
            worker.join()  # each has left its loop already
        if not self._drained:
            self._drained = self._driver.drain(0.0)
        return self._drained and self._binding.unregister()

    # Shutdown, on the shutdown thread.

    def _close_and_force_outcomes(self) -> bool:
        with self._lock:
            self._shutdown_requested = True
            forced = [attempt for attempt in self._attempts if attempt._outcome is None]
            for attempt in forced:
                attempt._force_outcome(SHUTDOWN_FAULT)
                self._conclude_or_quiesce(attempt)
            self._fetch_queue.clear()  # every attempt is stopped: nothing queued will run
            self._publish_queue.clear()
            self._work_ready.notify_all()
            self._poke_progress()
        return bool(forced)

    def _shutdown_answer_is_final(self) -> bool:
        """Under the lock, once closing: the progress thread finished, or the labels are final. An
        attempt pinned by a batch whose access can never be proven to end never quiesces, and the
        registrations are revoked only once every attempt quiesced, so when every open attempt is
        pinned, neither its label nor a region's can go any more."""
        if self._teardown_done:
            return True
        open_attempts = [attempt for attempt in self._attempts if not attempt._quiesced]
        return bool(open_attempts) and all(
            attempt._has_unprovable_batch for attempt in open_attempts
        )

    def _labels(self) -> frozenset[str]:
        with self._lock:
            attempts = {attempt.label for attempt in self._attempts if not attempt._quiesced}
        return frozenset(attempts) | self._binding.live_labels()

    def _report(self, labels: frozenset[str]) -> None:
        if labels == self._reported_labels:
            return
        self._reported_labels = labels
        if labels:
            message = f"blob backend: shutting down, still accessible: {sorted(labels)}"
            logger.warning(message)  # type: ignore[no-untyped-call]
        else:
            message = "blob backend: shut down; nothing is accessible any more"
            logger.info(message)  # type: ignore[no-untyped-call]

    def _call_wake(self) -> None:
        try:
            self._wake()
        except Exception as error:  # a raising wake must not kill a backend thread
            _warn_once(f"blob backend: wake raised {error!r}", (self._name, "wake", type(error)))


def _batch_bounds(keys: np.ndarray, step: int) -> list[tuple[int, int]]:
    """Consecutive ``[start, stop)`` runs of at most ``step`` rows, none naming a key twice: a
    driver refuses a key that appears twice in one call, while a view may name one key at two
    addresses. Distinct keys, the usual case, cut into whole steps without a pass over the rows."""
    count = len(keys)
    if count == 0:
        return []
    if np.unique(keys, axis=0).shape[0] == count:
        return [(start, min(start + step, count)) for start in range(0, count, step)]
    bounds: list[tuple[int, int]] = []
    seen: set[bytes] = set()
    start = 0
    for index in range(count):
        key = keys[index].tobytes()
        if index - start == step or key in seen:
            bounds.append((start, index))
            start, seen = index, set()
        seen.add(key)
    bounds.append((start, count))
    return bounds


def _refuse_shared_destinations(groups: list[_GroupRows]) -> None:
    """Two fetched rows in one slot would let one report READY over the other's bytes. A publish
    only reads its slots, so only a fetch is checked."""
    addresses = np.concatenate([rows.addresses for rows in groups]) if groups else np.empty(0)
    if np.unique(addresses).size != addresses.size:
        raise InvalidSubmission("two rows of a fetch write the same slot")


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _faults(batch: _Batch, reason: str) -> tuple[RowResult, ...]:
    return (faulted(reason),) * len(batch)


def _refused(batch: _Batch, reason: str) -> _Observation:
    """A call that failed as a whole with nothing escaped: every row faults, access ended."""
    return _Observation(
        _faults(batch, reason), ended=True, unprovable=False, whole_call_fault=reason
    )


_NO_NEW_EVIDENCE = _Observation(results=None, ended=False, unprovable=False)
