# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Out-of-band MPI failure-detection propagation for WideEP fault tolerance.

The model-forward thread can be stuck in an AlltoAll kernel when a peer fails,
so it cannot participate in recovery consensus. :class:`MpiFtSubcomm` owns a
dedicated EP-scoped MPI communicator and progresses only nonblocking
point-to-point requests on an independent CPU thread.

The component propagates one rank-failure detection and updates only the
process-local :class:`DetectedRankState`. It never receives or updates the
committed ``EPGroupHealth`` execution mask. Multi-rank suspect/confirm
consensus, commit authorization, and communicator reconstruction are
intentionally left to later WideEP FT phases.

Every survivor echoes a newly observed failure. Receiving one echo from every
active survivor proves only that they propagated the same detection. Detection
reconciliation does **not** authorize request resume, EPLB changes, NCCL
reconfiguration or use, or committed-mask mutation. The future 1c.4b recovery
coordinator consumes the detection callback and owns that authorization.

ULFM revoke is reserved for terminal control-plane aborts because MPI does not
carry enough information to distinguish a successful commit revoke from an
emergency abort at remote ranks. Without ULFM, terminal state is echoed to every
survivor; failure to confirm that echo within a bounded interval uses
``MPI_Abort`` on the world-spanning FT communicator as a fail-stop fallback.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping


_FAILURE_TAG = 0
_MESSAGE_WORDS = 2


class _MessageKind(IntEnum):
    FAILURE = 1
    ABORT = 2


_PROCESS_LIFETIME_REFS: list[object] = []
_PROCESS_LIFETIME_REFS_LOCK = threading.Lock()


def _retain_for_process_lifetime(*references: object) -> None:
    """Keep active MPI objects reachable until interpreter teardown."""
    with _PROCESS_LIFETIME_REFS_LOCK:
        _PROCESS_LIFETIME_REFS.extend(references)


class _MpiRequest(Protocol):
    """Subset of mpi4py request operations used by the progress loop."""

    def Cancel(self) -> None: ...

    def Test(self) -> bool: ...


class _MpiComm(Protocol):
    """Subset of mpi4py communicator operations used by this component."""

    def Get_rank(self) -> int: ...

    def Get_size(self) -> int: ...

    def Irecv(self, buffer: np.ndarray, source: int, tag: int) -> _MpiRequest: ...

    def Isend(self, buffer: np.ndarray, dest: int, tag: int) -> _MpiRequest: ...

    def Is_revoked(self) -> bool: ...

    def Revoke(self) -> None: ...

    def Abort(self, errorcode: int) -> None: ...

    def Set_errhandler(self, errhandler: object) -> None: ...


class _MpiModule(Protocol):
    """Subset of the mpi4py ``MPI`` module used by this component."""

    ERRORS_RETURN: object
    ERR_PROC_FAILED: int
    ERR_UNKNOWN: int
    THREAD_MULTIPLE: int
    Exception: type[BaseException]

    def Query_thread(self) -> int: ...


FailureDetectedCallback = Callable[[int, int, float], None]
# Compatibility name for callers of the original draft API. New integrations
# should use ``FailureDetectedCallback`` because this callback carries detected,
# never committed, state.
FailureReceivedCallback = FailureDetectedCallback


@dataclass(frozen=True)
class DetectedRankStateSnapshot:
    """Atomic snapshot of rank-failure evidence for one communicator epoch."""

    mask: int
    failed_ranks: frozenset[int]
    generation: int


class DetectedRankState:
    """Thread-safe, process-local rank-failure evidence.

    This state is deliberately distinct from ``EPGroupHealth``. It is an
    observation owned by :class:`MpiFtSubcomm`, not a committed execution mask,
    and it is monotonic for the lifetime of one FT communicator epoch.

    Args:
        ep_size: Number of EP-local ranks represented by this state.

    Raises:
        ValueError: If ``ep_size`` is not positive.
    """

    def __init__(self, ep_size: int) -> None:
        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got {ep_size}")
        self._ep_size = ep_size
        self._all_active_mask = (1 << ep_size) - 1
        self._active_mask = self._all_active_mask
        self._failed_ranks: set[int] = set()
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def ep_size(self) -> int:
        """Number of ranks represented by this immutable communicator epoch."""
        return self._ep_size

    @property
    def generation(self) -> int:
        """Number of effective detected-failure transitions."""
        with self._lock:
            return self._generation

    def record_failure(self, rank: int) -> bool:
        """Record one failed rank, returning whether evidence changed."""
        self._validate_rank(rank)
        bit = 1 << rank
        with self._lock:
            if not self._active_mask & bit:
                return False
            self._active_mask &= ~bit
            self._failed_ranks.add(rank)
            self._generation += 1
            return True

    def is_active(self, rank: int) -> bool:
        """Return whether no failure has been recorded for ``rank``."""
        self._validate_rank(rank)
        with self._lock:
            return bool(self._active_mask & (1 << rank))

    def get_mask(self) -> int:
        """Return the active-rank mask derived from detected evidence."""
        with self._lock:
            return self._active_mask

    def get_failed_ranks(self) -> frozenset[int]:
        """Return an immutable snapshot of ranks with failure evidence."""
        with self._lock:
            return frozenset(self._failed_ranks)

    def all_active(self) -> bool:
        """Return whether no rank failure has been detected."""
        with self._lock:
            return self._active_mask == self._all_active_mask

    def snapshot(self) -> DetectedRankStateSnapshot:
        """Return one coherent mask, failure-set, and generation snapshot."""
        with self._lock:
            return DetectedRankStateSnapshot(
                mask=self._active_mask,
                failed_ranks=frozenset(self._failed_ranks),
                generation=self._generation,
            )

    def _validate_rank(self, rank: int) -> None:
        if not 0 <= rank < self._ep_size:
            raise ValueError(f"rank must be in [0, {self._ep_size}), got {rank}")


@dataclass(frozen=True)
class MpiFtSubcommConfig:
    """Runtime-independent tuning for the FT progress thread.

    Args:
        poll_interval_sec: Maximum delay between MPI progress passes. The
            default leaves margin below the 100 ms agreement budget.
        startup_timeout_sec: Maximum time to wait for receive requests to be
            posted when :meth:`MpiFtSubcomm.start` is called.
        stop_timeout_sec: Default bounded join timeout used by
            :meth:`MpiFtSubcomm.stop`.
        reconcile_timeout_sec: Maximum time for every active survivor to
            announce the same failure before the control plane fails closed.
        unattributed_error_timeout_sec: Maximum time to wait for the host
            watchdog to identify a rank after non-ULFM MPI reports a generic
            transport error. The default leaves margin for the design's
            five-second watchdog bound while keeping terminal escalation
            inside the ten-second recovery budget.
        abort_timeout_sec: Maximum time to reconcile a relayed terminal ABORT
            before falling back to communicator-wide ``MPI_Abort`` when ULFM
            is unavailable.
    """

    poll_interval_sec: float = 0.01
    startup_timeout_sec: float = 2.0
    stop_timeout_sec: float = 2.0
    reconcile_timeout_sec: float = 1.0
    unattributed_error_timeout_sec: float = 6.0
    abort_timeout_sec: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("poll_interval_sec", self.poll_interval_sec),
            ("startup_timeout_sec", self.startup_timeout_sec),
            ("stop_timeout_sec", self.stop_timeout_sec),
            ("reconcile_timeout_sec", self.reconcile_timeout_sec),
            ("unattributed_error_timeout_sec", self.unattributed_error_timeout_sec),
            ("abort_timeout_sec", self.abort_timeout_sec),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")


@dataclass
class _PendingSend:
    request: _MpiRequest
    buffer: np.ndarray
    destination: int
    failed_rank: int
    kind: _MessageKind


@dataclass
class _PendingReceive:
    request: _MpiRequest
    buffer: np.ndarray
    source: int


@dataclass(frozen=True)
class _OutboundMessage:
    kind: _MessageKind
    failed_rank: int


class _Lifecycle(Enum):
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class MpiFtSubcomm:
    """Broadcast EP-rank failure detections on a dedicated MPI control plane.

    Construction creates the FT communicator collectively on the caller's
    thread. Call :meth:`start` only after every rank has constructed the
    component. During normal operation, the progress thread owns MPI calls on
    this communicator; :meth:`report_detected_failure` updates only local
    detected state and enqueues a report, so it never waits for a peer or MPI
    progress. The watchdog must call this method instead of pre-recording evidence.
    Because ``MPI.THREAD_MULTIPLE`` is required, a caller whose start/stop
    deadline proves that thread is wedged may issue the terminal Revoke/Abort
    fallback.

    Args:
        mapping: Distributed mapping whose ``moe_ep_group`` defines the FT
            communicator and whose ``moe_ep_rank`` defines the local rank.
        detected_state: Process-local EP detection evidence updated by sent and
            received failure reports. ``EPGroupHealth`` is rejected because it
            is the committed execution-mask state owned by the recovery
            coordinator.
        config: Progress-loop timing configuration.
        on_failure_detected: Optional callback invoked as
            ``(failed_rank, source_rank, monotonic_time)`` when a local or
            received report first changes detected state. This is the handoff
            to the future 1c.4b recovery coordinator; it is detection evidence,
            not commit authorization. Delivery runs on a dedicated daemon
            thread so callback latency cannot block MPI progress.
        comm: Pre-created FT communicator. Intended for tests; production
            callers should let the component create it from ``mapping``.
        mpi_module: MPI module paired with ``comm``. Intended for tests.

    Raises:
        RuntimeError: If MPI support or thread support is insufficient, or if
            the communicator does not match ``mapping``.
        TypeError: If ``detected_state`` is not a :class:`DetectedRankState`.
        ValueError: If ``detected_state`` and ``mapping`` describe different
            EP groups, or if both callback keyword names are provided.
    """

    def __init__(
        self,
        mapping: Mapping,
        detected_state: DetectedRankState,
        config: MpiFtSubcommConfig | None = None,
        on_failure_detected: FailureDetectedCallback | None = None,
        *,
        comm: _MpiComm | None = None,
        mpi_module: _MpiModule | None = None,
        on_failure_received: FailureReceivedCallback | None = None,
    ) -> None:
        if not isinstance(detected_state, DetectedRankState):
            raise TypeError(
                "detected_state must be a DetectedRankState; committed "
                "EPGroupHealth cannot be used as failure-detection evidence"
            )
        if on_failure_detected is not None and on_failure_received is not None:
            raise ValueError(
                "Specify only on_failure_detected; on_failure_received is a compatibility alias"
            )
        if on_failure_detected is None:
            on_failure_detected = on_failure_received
        if mpi_module is None:
            if MPI is None:
                raise RuntimeError("mpi4py is required for WideEP failure propagation")
            mpi_module = cast(_MpiModule, MPI)
        self._mpi = mpi_module

        self._config = config or MpiFtSubcommConfig()
        comm_created_collectively = comm is None
        collective_setup = None
        if comm_created_collectively:
            # Keep the collective startup validation and Split on the
            # constructing thread. In particular, do not raise a rank-local
            # mapping error before peers have entered the same collective
            # validation path.
            from tensorrt_llm._torch.distributed.communicator import create_mpi_ft_subcomm

            collective_setup = create_mpi_ft_subcomm(
                mapping,
                health_size=detected_state.ep_size,
            )
            comm = cast(_MpiComm, collective_setup.comm)

        assert comm is not None
        self._comm = comm
        if comm_created_collectively:
            # create_mpi_ft_subcomm already reconciled thread support, mapping,
            # health size, communicator rank/size, and ERRORS_RETURN across the
            # parent communicator. Repeating any of those MPI operations here
            # would reintroduce a rank-local failure after collective startup.
            assert collective_setup is not None
            self._local_rank = collective_setup.local_rank
            self._ep_size = collective_setup.ep_size
        else:
            if self._mpi.Query_thread() < self._mpi.THREAD_MULTIPLE:
                raise RuntimeError(
                    "WideEP FT requires MPI.THREAD_MULTIPLE because its control-plane "
                    "thread overlaps other MPI traffic"
                )

            ep_group = tuple(mapping.moe_ep_group)
            if len(ep_group) != mapping.moe_ep_size:
                raise ValueError(
                    "mapping.moe_ep_group size must match mapping.moe_ep_size, "
                    f"got {len(ep_group)} and {mapping.moe_ep_size}"
                )
            expected_world = set(range(mapping.world_size))
            if mapping.moe_ep_size != mapping.world_size or set(ep_group) != expected_world:
                raise ValueError(
                    "WideEP FT MVP requires one MoE EP group spanning the full MPI world; "
                    f"got world_size={mapping.world_size}, moe_ep_group={ep_group}"
                )
            if detected_state.ep_size != mapping.moe_ep_size:
                raise ValueError(
                    "DetectedRankState size must match mapping.moe_ep_size, "
                    f"got {detected_state.ep_size} and {mapping.moe_ep_size}"
                )
            self._local_rank = mapping.moe_ep_rank
            self._ep_size = mapping.moe_ep_size
            # Install the non-fatal handler before inspecting the injected
            # communicator so even validation errors are returned to Python.
            self._comm.Set_errhandler(self._mpi.ERRORS_RETURN)
            actual_rank = self._comm.Get_rank()
            actual_size = self._comm.Get_size()
            if actual_rank != self._local_rank or actual_size != self._ep_size:
                raise RuntimeError(
                    "WideEP FT communicator does not match the mapping: "
                    f"rank={actual_rank}, size={actual_size}, "
                    f"expected rank={self._local_rank}, size={self._ep_size}"
                )

        self._detected_state = detected_state
        self._on_failure_detected = on_failure_detected
        self._outbound_reports: queue.SimpleQueue[_OutboundMessage] = queue.SimpleQueue()
        self._announced_failures: set[int] = set()
        self._failure_reporters: dict[int, set[int]] = {}
        self._reconcile_deadlines: dict[int, float] = {}
        self._unattributed_error_deadlines: dict[int, float] = {}
        self._failure_fanout_posted: set[int] = set()
        self._failure_fanout_complete: set[int] = set()
        self._abort_reporters: set[int] = set()
        self._abort_expected_reporters: frozenset[int] = frozenset()
        self._abort_deadline: float | None = None
        self._abort_payload = -1
        self._abort_announced = False
        self._abort_fanout_posted = False
        self._abort_fanout_complete = False
        self._protocol_lock = threading.Lock()
        self._failure_claim_lock = threading.Lock()
        self._accepted_failed_rank: int | None = None
        self._accepted_failure_generation: int | None = None
        self._stop_event = threading.Event()
        self._abort_requested = threading.Event()
        self._wake_event = threading.Event()
        self._ready_event = threading.Event()
        self._deadline_monitor_stop_event = threading.Event()
        self._deadline_monitor_wake_event = threading.Event()
        self._deadline_monitor_ready_event = threading.Event()
        self._transport_poisoned = threading.Event()
        self._progress_failed = threading.Event()
        self._lifecycle = _Lifecycle.CREATED
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._deadline_monitor_thread: threading.Thread | None = None
        self._last_error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._retained_requests: list[_PendingSend | _PendingReceive] = []
        self._process_lifetime_retained = comm_created_collectively
        if comm_created_collectively:
            # The FT communicator is a process-lifetime control-plane
            # resource. Letting mpi4py destroy it when one broadcaster is
            # garbage-collected would call a collective Free at rank-local
            # and nondeterministic times.
            _retain_for_process_lifetime(self._comm)
        self._shutdown_deadline: float | None = None
        self._callback_queue: queue.SimpleQueue[tuple[int, int, float] | None] = queue.SimpleQueue()
        self._callback_stop_event = threading.Event()
        self._callback_thread: threading.Thread | None = None
        self._terminal_action_lock = threading.Lock()
        self._revoke_attempted = False
        self._world_abort_attempted = False
        self._ulfm_available = (
            collective_setup.ulfm_available if collective_setup is not None else self._probe_ulfm()
        )

    @property
    def ulfm_available(self) -> bool:
        """Whether this MPI build implements the ULFM communicator methods."""
        with self._terminal_action_lock:
            return self._ulfm_available

    @property
    def last_error(self) -> BaseException | None:
        """First terminal control-plane error, if any."""
        with self._error_lock:
            return self._last_error

    def start(self) -> None:
        """Start the dedicated MPI progress thread.

        This method is not restartable. A second call, including one after
        :meth:`stop`, raises ``RuntimeError``.
        """
        thread_start_error: BaseException | None = None
        with self._lifecycle_lock:
            if self._lifecycle is not _Lifecycle.CREATED:
                raise RuntimeError(
                    f"WideEP FT broadcaster cannot start from {self._lifecycle.name.lower()} state"
                )
            self._lifecycle = _Lifecycle.STARTING
            try:
                if self._on_failure_detected is not None:
                    callback_thread = threading.Thread(
                        target=self._callback_loop,
                        name="wide-ep-ft-callback",
                        daemon=True,
                    )
                    callback_thread.start()
                    # Store only a successfully started thread. stop() may join
                    # every stored thread and joining an unstarted Thread raises.
                    self._callback_thread = callback_thread
                deadline_monitor_thread = threading.Thread(
                    target=self._deadline_monitor_loop,
                    name="wide-ep-ft-deadline-monitor",
                    daemon=True,
                )
                deadline_monitor_thread.start()
                self._deadline_monitor_thread = deadline_monitor_thread
                progress_thread = threading.Thread(
                    target=self._progress_loop,
                    name="wide-ep-ft-broadcast",
                    daemon=True,
                )
                progress_thread.start()
                self._thread = progress_thread
            except Exception as error:
                # Publish the error before FAILED so concurrent observers never
                # see a terminal lifecycle without its cause.
                self._record_error(error)
                self._progress_failed.set()
                self._lifecycle = _Lifecycle.FAILED
                thread_start_error = error

        if thread_start_error is not None:
            self._request_deadline_monitor_stop()
            self._request_callback_stop()
            self._retain_requests([])
            # There is no progress thread to relay ABORT, so fail closed on the
            # caller thread. This is the only thread that can own the FT comm
            # after progress-thread creation itself failed.
            self._fail_closed_immediately(thread_start_error)
            callback_thread = self._callback_thread
            if callback_thread is not None and callback_thread.is_alive():
                callback_thread.join(self._config.stop_timeout_sec)
            deadline_monitor_thread = self._deadline_monitor_thread
            if deadline_monitor_thread is not None and deadline_monitor_thread.is_alive():
                deadline_monitor_thread.join(self._config.stop_timeout_sec)
            raise RuntimeError(
                "WideEP FT control-plane thread failed to start"
            ) from thread_start_error

        startup_deadline = time.monotonic() + self._config.startup_timeout_sec
        progress_ready = self._ready_event.wait(max(0.0, startup_deadline - time.monotonic()))
        deadline_monitor_ready = self._deadline_monitor_ready_event.wait(
            max(0.0, startup_deadline - time.monotonic())
        )
        if not progress_ready or not deadline_monitor_ready:
            timeout_error = TimeoutError(
                "WideEP FT control-plane threads did not start before the timeout"
            )
            with self._lifecycle_lock:
                self._lifecycle = _Lifecycle.FAILED
            # Stop callback delivery before the terminal MPI action. Real MPI_Abort
            # does not return, while test doubles and Revoke do; either way no
            # callback worker should be left waiting for progress-thread cleanup.
            self._request_deadline_monitor_stop()
            self._request_callback_stop()
            # The progress thread may be wedged inside the MPI call whose
            # startup deadline just expired, so it cannot execute the terminal
            # action itself. MPI.THREAD_MULTIPLE is a construction invariant;
            # use the caller thread to issue the terminal communicator action.
            self._fail_closed_immediately(timeout_error)
            # The progress thread may be stuck inside the first MPI Irecv and
            # therefore unable to reach its ``finally`` block. Do not leave the
            # independent callback worker alive after start() has failed.
            callback_thread = self._callback_thread
            if callback_thread is not None and callback_thread is not threading.current_thread():
                callback_thread.join(self._config.stop_timeout_sec)
            deadline_monitor_thread = self._deadline_monitor_thread
            if (
                deadline_monitor_thread is not None
                and deadline_monitor_thread is not threading.current_thread()
            ):
                deadline_monitor_thread.join(self._config.stop_timeout_sec)
            raise timeout_error
        with self._lifecycle_lock:
            startup_error = self.last_error
            lifecycle = self._lifecycle
            if startup_error is not None:
                self._lifecycle = _Lifecycle.FAILED
            elif lifecycle is not _Lifecycle.RUNNING:
                raise RuntimeError(
                    "WideEP FT broadcaster did not reach running state during startup "
                    f"(state={lifecycle.name.lower()})"
                )
        if startup_error is not None:
            self._request_deadline_monitor_stop()
            deadline_monitor_thread = self._deadline_monitor_thread
            if (
                deadline_monitor_thread is not None
                and deadline_monitor_thread is not threading.current_thread()
            ):
                deadline_monitor_thread.join(self._config.stop_timeout_sec)
            raise RuntimeError("WideEP FT progress thread failed during startup") from startup_error

    def stop(self, timeout: float | None = None) -> None:
        """Stop progress without blocking on MPI requests or freeing the comm.

        Healthy requests are cancelled and polled within the same bounded
        timeout. After a peer or transport failure, requests and their NumPy
        buffers are retained until process teardown because cancelling,
        waiting for, or freeing them can hang inside MPI.

        If a known failure is not yet reconciled, the progress thread keeps
        running until agreement or the existing reconciliation deadline drives
        the terminal fail-stop path.

        Args:
            timeout: Maximum join time. Defaults to
                :attr:`MpiFtSubcommConfig.stop_timeout_sec`.

        Raises:
            ValueError: If ``timeout`` is not finite and positive.
            TimeoutError: If the progress thread does not exit in time.
        """
        timeout = self._config.stop_timeout_sec if timeout is None else timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"timeout must be finite and > 0, got {timeout}")

        stop_deadline = time.monotonic() + timeout
        with self._lifecycle_lock:
            if self._lifecycle is _Lifecycle.CREATED:
                self._lifecycle = _Lifecycle.STOPPED
                self._request_deadline_monitor_stop()
                self._request_callback_stop()
                return
            callback_thread = self._callback_thread
            deadline_monitor_thread = self._deadline_monitor_thread
            if (
                self._lifecycle is _Lifecycle.STOPPED
                and (callback_thread is None or not callback_thread.is_alive())
                and (deadline_monitor_thread is None or not deadline_monitor_thread.is_alive())
            ):
                return
            thread = self._thread
            if self._lifecycle in (_Lifecycle.STARTING, _Lifecycle.RUNNING):
                self._lifecycle = _Lifecycle.STOPPING
            join_grace = min(0.01, timeout / 2)
            self._shutdown_deadline = stop_deadline - join_grace
            self._stop_event.set()
            self._wake_event.set()
            self._request_callback_stop()

        if thread is not None:
            thread.join(max(0.0, stop_deadline - time.monotonic()))
            if thread.is_alive():
                timeout_error = TimeoutError(
                    "WideEP FT progress thread did not stop before the timeout"
                )
                with self._lifecycle_lock:
                    self._lifecycle = _Lifecycle.FAILED
                # Do not enqueue fail-stop work onto the thread that this
                # timeout proves is not making MPI progress.
                self._request_deadline_monitor_stop()
                self._fail_closed_immediately(timeout_error)
                raise timeout_error

        self._request_deadline_monitor_stop()
        if (
            deadline_monitor_thread is not None
            and deadline_monitor_thread is not threading.current_thread()
        ):
            deadline_monitor_thread.join(max(0.0, stop_deadline - time.monotonic()))
            if deadline_monitor_thread.is_alive():
                timeout_error = TimeoutError(
                    "WideEP FT deadline monitor did not stop before the timeout"
                )
                self._fail_closed_immediately(timeout_error)
                raise timeout_error

        if callback_thread is not None and callback_thread is not threading.current_thread():
            callback_thread.join(max(0.0, stop_deadline - time.monotonic()))
            if callback_thread.is_alive():
                timeout_error = TimeoutError(
                    "WideEP FT callback thread did not stop before the timeout"
                )
                # The detection handoff does not own MPI state. A stuck future
                # coordinator callback may make this stop call time out, but it
                # must not make one rank take the poisoned-world process-exit
                # path.
                raise timeout_error

        with self._lifecycle_lock:
            if self._lifecycle is not _Lifecycle.FAILED:
                self._lifecycle = _Lifecycle.STOPPED

    def report_detected_failure(self, failed_rank: int) -> bool:
        """Record and asynchronously announce one detected EP-rank failure.

        This is the nonblocking seam invoked by the host watchdog. The
        broadcaster owns the detected-state transition; callers must not record
        the failure before invoking it. This method performs no MPI
        calls and never waits for the progress thread. Repeated reports for the
        same rank are coalesced after the first local announcement.

        Args:
            failed_rank: EP-local rank in ``[0, ep_size)``.

        Returns:
            ``True`` if this call changed local detected state, else ``False``.
        """
        self._validate_rank(failed_rank)
        with self._lifecycle_lock:
            if self._lifecycle is not _Lifecycle.RUNNING or self._progress_failed.is_set():
                raise RuntimeError(
                    f"WideEP FT broadcaster is not running (state={self._lifecycle.name.lower()})"
                )
            start_time = time.monotonic()
            try:
                self._claim_failure(failed_rank)
            except RuntimeError as error:
                self._lifecycle = _Lifecycle.FAILED
                self._request_terminal_abort(error, failed_rank, self._local_rank)
                raise
            changed = self._detected_state.record_failure(failed_rank)
            enqueued = self._observe_failure(failed_rank, self._local_rank)
        detected_time = time.monotonic()
        logger.warning(
            f"WideEP FT local failure rank={failed_rank} changed={changed} "
            f"enqueued={enqueued} elapsed_ms={(detected_time - start_time) * 1000.0:.3f}"
        )
        if changed:
            self._invoke_detection_callback(failed_rank, self._local_rank, detected_time)
        return changed

    def pre_failover(self, failed_rank: int) -> bool:
        """Compatibility alias for :meth:`report_detected_failure`.

        Despite the historical name, this method does not commit failover and
        callers must not pre-record detected evidence.
        """
        return self.report_detected_failure(failed_rank)

    def broadcast_failure(self, failed_rank: int) -> bool:
        """Compatibility alias for :meth:`report_detected_failure`."""
        return self.report_detected_failure(failed_rank)

    def world_is_poisoned(self) -> bool:
        """Return whether this communicator epoch has ever observed failure."""
        return (
            self._transport_poisoned.is_set()
            or self._accepted_failure_rank() is not None
            or not self._detected_state.all_active()
        )

    def failure_detection_is_reconciled(self, failed_rank: int) -> bool:
        """Return whether every active survivor announced the same detection.

        This is detection-plane evidence only. A ``True`` result does not
        authorize request resume, EPLB changes, NCCL reconfiguration or use, or
        committed-mask mutation. The future 1c.4b recovery coordinator owns
        those decisions. This method performs no MPI calls, and a terminal
        progress error always fails closed.
        """
        self._validate_rank(failed_rank)
        with self._lifecycle_lock:
            if self._lifecycle is not _Lifecycle.RUNNING:
                return False
            if self.last_error is not None or self._progress_failed.is_set():
                return False
            snapshot = self._detected_state.snapshot()
            if failed_rank not in snapshot.failed_ranks:
                return False
            accepted_failure, accepted_generation = self._accepted_failure_state()
            with self._protocol_lock:
                if not self._detection_reconciliation_state_is_valid_locked(
                    snapshot, accepted_failure, accepted_generation
                ):
                    return False
                return self._failure_detection_is_reconciled_locked(failed_rank, snapshot.mask)

    def detected_state_is_reconciled(self) -> bool:
        """Return whether survivors propagated every local failure detection.

        This is not a commit gate. A ``True`` result does not authorize request
        resume, EPLB changes, NCCL reconfiguration or use, or committed-mask
        mutation. The future 1c.4b recovery coordinator owns those decisions.
        """
        with self._lifecycle_lock:
            if self._lifecycle is not _Lifecycle.RUNNING:
                return False
            if self.last_error is not None or self._progress_failed.is_set():
                return False
            snapshot = self._detected_state.snapshot()
            accepted_failure, accepted_generation = self._accepted_failure_state()
            if not snapshot.failed_ranks:
                return accepted_failure is None and not self._transport_poisoned.is_set()
            with self._protocol_lock:
                if not self._detection_reconciliation_state_is_valid_locked(
                    snapshot, accepted_failure, accepted_generation
                ):
                    return False
                return all(
                    self._failure_detection_is_reconciled_locked(failed_rank, snapshot.mask)
                    for failed_rank in snapshot.failed_ranks
                )

    def detected_health_is_reconciled(self) -> bool:
        """Compatibility alias for :meth:`detected_state_is_reconciled`."""
        return self.detected_state_is_reconciled()

    def failure_is_reconciled(self, failed_rank: int) -> bool:
        """Compatibility alias for :meth:`failure_detection_is_reconciled`."""
        return self.failure_detection_is_reconciled(failed_rank)

    def health_is_reconciled(self) -> bool:
        """Compatibility alias for :meth:`detected_state_is_reconciled`."""
        return self.detected_state_is_reconciled()

    def _validate_rank(self, rank: int) -> None:
        if not 0 <= rank < self._ep_size:
            raise ValueError(f"failed_rank must be in [0, {self._ep_size}), got {rank}")

    def _claim_failure(self, failed_rank: int) -> None:
        """Atomically enforce the single-distinct-failure MVP contract."""
        with self._failure_claim_lock:
            snapshot = self._detected_state.snapshot()
            known_failures = snapshot.failed_ranks
            conflicting_evidence = known_failures - {failed_rank}
            accepted = self._accepted_failed_rank
            if (accepted is not None and accepted != failed_rank) or conflicting_evidence:
                first_failure = accepted
                if first_failure is None:
                    first_failure = min(conflicting_evidence)
                raise RuntimeError(
                    "WideEP FT MVP supports exactly one distinct failed rank; "
                    f"already accepted rank {first_failure}, received rank {failed_rank}"
                )
            if accepted is None:
                if failed_rank in known_failures:
                    raise RuntimeError(
                        "WideEP FT detected state was mutated before the failure "
                        "broadcaster accepted the report; the watchdog must call "
                        "report_detected_failure() without pre-recording evidence"
                    )
                self._accepted_failed_rank = failed_rank
                # Exactly one effective record_failure() must be the next
                # detected-state transition. Capturing that generation here
                # prevents an independent mutation in the claim-to-observe
                # window from becoming the accepted epoch baseline.
                self._accepted_failure_generation = snapshot.generation + 1

    def _accepted_failure_rank(self) -> int | None:
        with self._failure_claim_lock:
            return self._accepted_failed_rank

    def _accepted_failure_state(self) -> tuple[int | None, int | None]:
        with self._failure_claim_lock:
            return self._accepted_failed_rank, self._accepted_failure_generation

    def _observe_failure(self, failed_rank: int, reporter: int) -> bool:
        """Record one detection report and enqueue this rank's echo once."""
        now = time.monotonic()
        snapshot = self._detected_state.snapshot()
        message: _OutboundMessage | None = None
        with self._protocol_lock:
            unattributed_deadline = self._unattributed_error_deadlines.get(failed_rank)
            if unattributed_deadline != math.inf:
                self._unattributed_error_deadlines.pop(failed_rank, None)
            reporters = self._failure_reporters.setdefault(failed_rank, set())
            reporters.update((reporter, self._local_rank))
            self._reconcile_deadlines.setdefault(
                failed_rank, now + self._config.reconcile_timeout_sec
            )
            failure_enqueued = failed_rank not in self._announced_failures
            if failure_enqueued:
                self._announced_failures.add(failed_rank)
                message = _OutboundMessage(_MessageKind.FAILURE, failed_rank)
            if self._failure_detection_is_reconciled_locked(failed_rank, snapshot.mask):
                self._reconcile_deadlines.pop(failed_rank, None)
        if message is not None:
            self._outbound_reports.put(message)
            self._wake_event.set()
        self._deadline_monitor_wake_event.set()
        return failure_enqueued

    def _detection_reconciliation_state_is_valid_locked(
        self,
        snapshot: DetectedRankStateSnapshot,
        accepted_failure: int | None,
        accepted_generation: int | None,
    ) -> bool:
        """Reject unresolved transport errors and same-epoch rank restoration."""
        if self._unattributed_error_deadlines:
            return False
        if accepted_failure is None:
            return not snapshot.failed_ranks and accepted_generation is None
        return (
            snapshot.failed_ranks == frozenset({accepted_failure})
            and accepted_generation == snapshot.generation
        )

    def _request_terminal_abort(
        self,
        error: BaseException,
        payload: int,
        reporter: int,
    ) -> None:
        """Relay terminal state before the progress thread fails closed.

        The relay is best effort because it uses the same FT communicator. If
        survivor echoes do not converge before the bounded deadline, the
        progress thread escalates to ULFM revoke or communicator-wide
        ``MPI_Abort``. This method performs no MPI calls.
        """
        self._record_error(error)
        snapshot = self._detected_state.snapshot()
        enqueue_abort = False
        now = time.monotonic()
        with self._protocol_lock:
            if self._abort_deadline is None:
                self._abort_payload = payload
                self._abort_expected_reporters = frozenset(
                    rank for rank in range(self._ep_size) if snapshot.mask & (1 << rank)
                )
                self._abort_deadline = now + self._config.abort_timeout_sec
            self._abort_reporters.update((reporter, self._local_rank))
            if not self._abort_announced:
                self._abort_announced = True
                enqueue_abort = True
        self._abort_requested.set()
        if enqueue_abort:
            self._outbound_reports.put(_OutboundMessage(_MessageKind.ABORT, self._abort_payload))
        self._wake_event.set()
        self._deadline_monitor_wake_event.set()

    def _failure_detection_is_reconciled_locked(self, failed_rank: int, active_mask: int) -> bool:
        required_reporters = {rank for rank in range(self._ep_size) if active_mask & (1 << rank)}
        if not required_reporters:
            # There is no survivor that can participate in later recovery.
            # Do not let the usual ``empty set is a subset`` rule claim
            # detection propagation completed in this terminal case.
            return False
        reporters = self._failure_reporters.get(failed_rank, set())
        return failed_rank in self._failure_fanout_complete and required_reporters.issubset(
            reporters
        )

    def _probe_ulfm(self) -> bool:
        if not callable(getattr(self._comm, "Is_revoked", None)) or not callable(
            getattr(self._comm, "Revoke", None)
        ):
            return False
        try:
            already_revoked = self._comm.Is_revoked()
        except Exception as error:
            if isinstance(error, NotImplementedError) or self._is_unsupported_ulfm_error(error):
                logger.debug(f"WideEP FT ULFM is unavailable: {error}")
                return False
            self._record_error(error)
            self._retain_requests([])
            raise
        if already_revoked:
            error = RuntimeError("WideEP FT communicator is already revoked at construction")
            self._record_error(error)
            self._retain_requests([])
            raise error
        return True

    def _is_unsupported_ulfm_error(self, error: BaseException) -> bool:
        if not isinstance(error, self._mpi.Exception):
            return False
        get_error_class = getattr(error, "Get_error_class", None)
        if not callable(get_error_class):
            return False
        unsupported_classes = {
            value
            for value in (
                getattr(self._mpi, "ERR_NOT_SUPPORTED", None),
                getattr(self._mpi, "ERR_UNSUPPORTED_OPERATION", None),
            )
            if value is not None
        }
        try:
            error_class = get_error_class()
        except Exception:
            # Error inspection is advisory. Preserve the original probe error
            # as the construction failure when the runtime cannot classify it.
            return False
        return error_class in unsupported_classes

    def _progress_loop(self) -> None:
        pending_sends: list[_PendingSend] = []
        pending_receives: dict[int, _PendingReceive] = {}
        try:
            for peer in range(self._ep_size):
                if peer != self._local_rank:
                    pending_receives[peer] = self._post_receive(peer)
            with self._lifecycle_lock:
                if self._lifecycle is _Lifecycle.STARTING:
                    self._lifecycle = _Lifecycle.RUNNING
            self._ready_event.set()

            while True:
                if self._progress_failed.is_set():
                    terminal_error = self.last_error
                    if terminal_error is None:
                        terminal_error = RuntimeError("WideEP FT progress was asked to fail closed")
                    raise terminal_error
                self._wake_event.clear()
                self._drain_outbound_reports(pending_sends)
                self._progress_sends(pending_sends)
                self._progress_receives(pending_receives)
                self._check_accepted_failure_epoch()
                self._check_unattributed_transport_deadlines()
                self._check_reconciliation_deadlines()
                with self._lifecycle_lock:
                    terminal_pending = (
                        self._abort_requested.is_set() or self._lifecycle is _Lifecycle.FAILED
                    )
                    stop_requested = self._stop_event.is_set() and not terminal_pending
                if terminal_pending:
                    if self._progress_terminal_abort():
                        break
                    self._wake_event.wait(self._config.poll_interval_sec)
                    continue
                if stop_requested:
                    safe_to_stop, stop_error = self._stop_reconciliation_state()
                    if stop_error is not None:
                        with self._lifecycle_lock:
                            self._lifecycle = _Lifecycle.FAILED
                        self._request_terminal_abort(
                            stop_error,
                            -1,
                            self._local_rank,
                        )
                    elif safe_to_stop:
                        break
                self._wake_event.wait(self._config.poll_interval_sec)
        except Exception as error:
            self._fail_closed_immediately(error)
            logger.error(f"WideEP FT progress thread failed: {error}")
        finally:
            self._ready_event.set()
            self._request_deadline_monitor_stop()
            self._request_callback_stop()
            try:
                self._cleanup_requests(pending_sends, pending_receives)
            except Exception as error:
                self._retain_requests([*pending_sends, *pending_receives.values()])
                logger.error(f"WideEP FT request cleanup failed: {error}")
                self._fail_closed_after_cleanup(error)
            with self._lifecycle_lock:
                if self._last_error is not None:
                    self._lifecycle = _Lifecycle.FAILED
                elif self._stop_event.is_set():
                    self._lifecycle = _Lifecycle.STOPPED

    def _post_receive(self, source: int) -> _PendingReceive:
        buffer = np.empty(_MESSAGE_WORDS, dtype=np.int64)
        request = self._comm.Irecv(buffer, source=source, tag=_FAILURE_TAG)
        return _PendingReceive(request=request, buffer=buffer, source=source)

    def _drain_outbound_reports(self, pending_sends: list[_PendingSend]) -> None:
        while True:
            try:
                message = self._outbound_reports.get_nowait()
            except queue.Empty:
                return

            failed_rank = message.failed_rank
            snapshot = self._detected_state.snapshot()
            if message.kind is _MessageKind.ABORT:
                # Terminal state must reach every rank still believed active.
                # If one of them is actually the second failed rank, send
                # failure falls through to the fail-stop MPI_Abort fallback.
                peers = [
                    peer
                    for peer in range(self._ep_size)
                    if peer != self._local_rank and snapshot.mask & (1 << peer)
                ]
            else:
                peers = [
                    peer
                    for peer in range(self._ep_size)
                    if peer != self._local_rank
                    and peer != failed_rank
                    and snapshot.mask & (1 << peer)
                ]
            if not peers:
                with self._protocol_lock:
                    if message.kind is _MessageKind.ABORT:
                        self._abort_fanout_posted = True
                        self._abort_fanout_complete = True
                    else:
                        self._failure_fanout_posted.add(failed_rank)
                        self._failure_fanout_complete.add(failed_rank)
                continue
            for peer in peers:
                buffer = np.asarray([int(message.kind), failed_rank], dtype=np.int64)
                request = self._comm.Isend(buffer, dest=peer, tag=_FAILURE_TAG)
                pending_sends.append(
                    _PendingSend(
                        request=request,
                        buffer=buffer,
                        destination=peer,
                        failed_rank=failed_rank,
                        kind=message.kind,
                    )
                )
            with self._protocol_lock:
                if message.kind is _MessageKind.ABORT:
                    # Posting is not completion: rendezvous sends can still
                    # require this thread to call Test before peers can receive
                    # the relay.
                    self._abort_fanout_posted = True
                else:
                    self._failure_fanout_posted.add(failed_rank)

    def _progress_sends(self, pending_sends: list[_PendingSend]) -> None:
        incomplete: list[_PendingSend] = []
        for pending in pending_sends:
            completed = pending.request.Test()
            if not completed:
                incomplete.append(pending)
        pending_sends[:] = incomplete
        incomplete_failure_ranks = {
            pending.failed_rank for pending in incomplete if pending.kind is _MessageKind.FAILURE
        }
        abort_incomplete = any(pending.kind is _MessageKind.ABORT for pending in incomplete)
        with self._protocol_lock:
            self._failure_fanout_complete.update(
                self._failure_fanout_posted - incomplete_failure_ranks
            )
            if self._abort_fanout_posted and not abort_incomplete:
                self._abort_fanout_complete = True

    def _progress_receives(self, pending_receives: dict[int, _PendingReceive]) -> None:
        for source, pending in list(pending_receives.items()):
            try:
                completed = pending.request.Test()
            except Exception as error:
                if not isinstance(error, self._mpi.Exception):
                    raise
                self._transport_poisoned.set()
                # A request that raised may still own its receive buffer. Keep
                # both alive until process teardown instead of assuming the
                # implementation completed it while returning the error.
                self._retain_requests([pending])
                if self._is_failed_peer_error(error):
                    del pending_receives[source]
                    self._handle_failed_peer(source, source)
                    continue
                if not self.ulfm_available:
                    # Without ULFM, MPI implementations may surface the dead
                    # source as a generic error. Do not guess the failed rank;
                    # drop only this fixed-source receive and keep polling the
                    # other survivors for a rank-attributed detection report.
                    del pending_receives[source]
                    self._track_unattributed_transport_error(source)
                    logger.warning(
                        f"WideEP FT receive from EP rank {source} failed without ULFM: {error}"
                    )
                    continue
                raise
            if not completed:
                continue

            del pending_receives[source]
            message_kind_value = int(pending.buffer[0])
            failed_rank = int(pending.buffer[1])
            try:
                message_kind = _MessageKind(message_kind_value)
            except ValueError:
                logger.warning(
                    f"WideEP FT ignored invalid message kind {message_kind_value} "
                    f"from EP rank {source}"
                )
            else:
                if message_kind is _MessageKind.ABORT and -1 <= failed_rank < self._ep_size:
                    self._handle_received_abort(failed_rank, source)
                elif message_kind is _MessageKind.FAILURE and 0 <= failed_rank < self._ep_size:
                    self._handle_received_report(failed_rank, source)
                else:
                    logger.warning(
                        f"WideEP FT ignored invalid {message_kind.name.lower()} payload "
                        f"{failed_rank} from EP rank {source}"
                    )

            if self._detected_state.is_active(source) and not self._stop_event.is_set():
                try:
                    pending_receives[source] = self._post_receive(source)
                except Exception as error:
                    if isinstance(error, self._mpi.Exception) and not self.ulfm_available:
                        self._transport_poisoned.set()
                        self._track_unattributed_transport_error(source)
                        logger.warning(
                            f"WideEP FT could not repost receive from EP rank {source} "
                            f"without ULFM: {error}"
                        )
                    else:
                        raise

    def _handle_received_report(self, failed_rank: int, source: int) -> None:
        start_time = time.monotonic()
        changed = self._accept_received_failure(
            failed_rank,
            reporter=source,
            transport_poisoned=False,
        )
        if changed is None:
            return
        if not changed:
            return
        received_time = time.monotonic()
        logger.warning(
            f"WideEP FT received failure rank={failed_rank} source={source} "
            f"elapsed_ms={(received_time - start_time) * 1000.0:.3f}"
        )
        self._invoke_detection_callback(failed_rank, source, received_time)

    def _handle_failed_peer(self, failed_rank: int, source: int) -> None:
        changed = self._accept_received_failure(
            failed_rank,
            reporter=self._local_rank,
            transport_poisoned=True,
        )
        if changed is None:
            return
        if changed:
            detected_time = time.monotonic()
            logger.warning(
                f"WideEP FT observed failed EP rank {failed_rank} through the FT communicator"
            )
            self._invoke_detection_callback(failed_rank, source, detected_time)

    def _accept_received_failure(
        self,
        failed_rank: int,
        *,
        reporter: int,
        transport_poisoned: bool,
    ) -> bool | None:
        """Atomically claim, mark, and observe one progress-thread failure."""
        claim_error: RuntimeError | None = None
        with self._lifecycle_lock:
            try:
                self._claim_failure(failed_rank)
            except RuntimeError as error:
                self._lifecycle = _Lifecycle.FAILED
                claim_error = error
            else:
                changed = self._detected_state.record_failure(failed_rank)
                if transport_poisoned:
                    self._transport_poisoned.set()
                self._observe_failure(failed_rank, reporter)
                return changed

        assert claim_error is not None
        self._request_terminal_abort(
            claim_error,
            failed_rank,
            self._local_rank,
        )
        return None

    def _handle_received_abort(self, payload: int, source: int) -> None:
        error = RuntimeError(
            f"WideEP FT received terminal ABORT from EP rank {source} (payload={payload})"
        )
        with self._lifecycle_lock:
            self._lifecycle = _Lifecycle.FAILED
            self._request_terminal_abort(error, payload, source)

    def _invoke_detection_callback(self, failed_rank: int, source: int, event_time: float) -> None:
        if self._on_failure_detected is None or self._callback_stop_event.is_set():
            return
        self._callback_queue.put((failed_rank, source, event_time))

    def _callback_loop(self) -> None:
        """Deliver detection handoffs without blocking MPI progress."""
        while True:
            event = self._callback_queue.get()
            if event is None:
                return
            failed_rank, source, event_time = event
            try:
                callback = self._on_failure_detected
                if callback is not None:
                    callback(failed_rank, source, event_time)
            except Exception as error:
                # A coordinator callback must never terminate either
                # control-plane thread.
                logger.warning(f"WideEP FT failure callback raised: {error}")

    def _request_callback_stop(self) -> None:
        if self._callback_stop_event.is_set():
            return
        self._callback_stop_event.set()
        self._callback_queue.put(None)

    def _request_deadline_monitor_stop(self) -> None:
        self._deadline_monitor_stop_event.set()
        self._deadline_monitor_wake_event.set()

    def _deadline_monitor_loop(self) -> None:
        """Enforce Python-side deadlines even when an MPI call is wedged.

        Normal communicator traffic remains owned by the progress thread. This
        monitor only observes protected protocol state. If a terminal relay
        itself reaches its deadline, ``MPI.THREAD_MULTIPLE`` permits this thread
        to issue the existing one-shot Revoke/Abort fail-stop action.
        """
        self._deadline_monitor_ready_event.set()
        try:
            while True:
                # Clear before checking stop/deadline state so a concurrent
                # setter cannot be erased between the check and wait below.
                self._deadline_monitor_wake_event.clear()
                if self._deadline_monitor_stop_event.is_set():
                    return
                if self._progress_failed.is_set():
                    return

                self._check_accepted_failure_epoch()
                self._check_unattributed_transport_deadlines()
                self._check_reconciliation_deadlines()

                if self._progress_failed.is_set():
                    return
                now = time.monotonic()
                with self._protocol_lock:
                    abort_deadline = self._abort_deadline
                    abort_reconciled = (
                        self._abort_fanout_complete
                        and self._abort_expected_reporters.issubset(self._abort_reporters)
                    )
                if self._abort_requested.is_set() and abort_deadline is not None:
                    if abort_reconciled:
                        # The relay reached every expected survivor. The progress
                        # thread will break cleanly when it resumes. Do not mark
                        # progress failed here: its exception path would otherwise
                        # escalate a reconciled non-ULFM terminal relay to
                        # MPI_Abort.
                        self._wake_event.set()
                        return
                    if now >= abort_deadline:
                        self._fail_closed_immediately(
                            TimeoutError(
                                "WideEP FT terminal relay did not complete before "
                                "the abort deadline"
                            )
                        )
                        self._wake_event.set()
                        return

                self._deadline_monitor_wake_event.wait(self._config.poll_interval_sec)
        except Exception as error:
            self._fail_closed_immediately(error)
            self._wake_event.set()
        finally:
            # start() also uses this event to distinguish a scheduled monitor
            # from a thread that failed before its loop became observable.
            self._deadline_monitor_ready_event.set()

    def _is_failed_peer_error(self, error: BaseException) -> bool:
        get_error_class = getattr(error, "Get_error_class", None)
        if not callable(get_error_class):
            return False
        try:
            error_class = get_error_class()
        except Exception:
            # Preserve the original transport failure as the terminal cause (or
            # let the non-ULFM watchdog identify the source) when an MPI runtime
            # cannot even classify its own exception.
            return False
        unknown = getattr(self._mpi, "ERR_UNKNOWN", None)
        failure_classes = {
            value
            for value in (getattr(self._mpi, "ERR_PROC_FAILED", None),)
            if value is not None and value != unknown
        }
        return error_class in failure_classes

    def _track_unattributed_transport_error(self, source: int) -> None:
        """Bound how long a generic non-ULFM peer error can remain unresolved."""
        # The watchdog report and the MPI error race independently. If this
        # source is already the accepted failed rank, the transport error is
        # explained and must not arm a stale deadline after _observe_failure()
        # has already cleared the error-first ordering.
        deadline = time.monotonic() + self._config.unattributed_error_timeout_sec
        with self._protocol_lock:
            if self._accepted_failure_rank() == source:
                return
            self._unattributed_error_deadlines.setdefault(source, deadline)
        self._deadline_monitor_wake_event.set()

    def _check_accepted_failure_epoch(self) -> None:
        """Fail closed when detected state leaves the accepted failure epoch."""
        error: RuntimeError | None = None
        accepted_failure: int | None = None
        with self._lifecycle_lock:
            if self._lifecycle not in (_Lifecycle.RUNNING, _Lifecycle.STOPPING):
                return
            snapshot = self._detected_state.snapshot()
            accepted_failure, accepted_generation = self._accepted_failure_state()
            if accepted_failure is None:
                if not snapshot.failed_ranks:
                    return
                error = RuntimeError(
                    "WideEP FT detected state changed outside the failure "
                    "broadcaster; the watchdog must call "
                    "report_detected_failure() without pre-recording evidence "
                    f"(snapshot={snapshot})"
                )
            elif (
                snapshot.failed_ranks == frozenset({accepted_failure})
                and snapshot.generation == accepted_generation
            ):
                return
            else:
                error = RuntimeError(
                    "WideEP FT accepted failure epoch changed before communicator "
                    "reconstruction "
                    f"(accepted_rank={accepted_failure}, "
                    f"expected_generation={accepted_generation}, snapshot={snapshot})"
                )
            self._lifecycle = _Lifecycle.FAILED

        assert error is not None
        payload = accepted_failure if accepted_failure is not None else -1
        self._request_terminal_abort(error, payload, self._local_rank)

    def _check_unattributed_transport_deadlines(self) -> None:
        now = time.monotonic()
        # Match the public reconciliation gate's lifecycle -> protocol lock
        # order. Publish FAILED while both locks exclude a concurrent gate or
        # local broadcast, then latch the expired poison before either can run.
        with self._lifecycle_lock:
            if self._lifecycle not in (_Lifecycle.RUNNING, _Lifecycle.STOPPING):
                return
            with self._protocol_lock:
                expired_sources = [
                    source
                    for source, deadline in self._unattributed_error_deadlines.items()
                    if now >= deadline
                ]
                if expired_sources:
                    self._lifecycle = _Lifecycle.FAILED
                    # Retaining these few per-source entries for this
                    # communicator epoch is harmless and permanently keeps the
                    # protocol gate closed. The infinity latch also prevents
                    # repeated terminal requests on later progress passes.
                    for source in expired_sources:
                        self._unattributed_error_deadlines[source] = math.inf
        if not expired_sources:
            return
        error = TimeoutError(
            "WideEP FT could not attribute non-ULFM transport errors before "
            f"the watchdog deadline: {expired_sources}"
        )
        self._request_terminal_abort(error, -1, self._local_rank)

    def _check_reconciliation_deadlines(self) -> None:
        now = time.monotonic()
        expired: list[int] = []
        with self._lifecycle_lock:
            if self._lifecycle not in (_Lifecycle.RUNNING, _Lifecycle.STOPPING):
                return
            snapshot = self._detected_state.snapshot()
            with self._protocol_lock:
                for failed_rank, deadline in list(self._reconcile_deadlines.items()):
                    if self._failure_detection_is_reconciled_locked(failed_rank, snapshot.mask):
                        del self._reconcile_deadlines[failed_rank]
                    elif now >= deadline:
                        expired.append(failed_rank)
            if expired:
                self._lifecycle = _Lifecycle.FAILED
        if expired:
            error = TimeoutError(
                f"WideEP FT did not reconcile failed EP ranks before the timeout: {expired}"
            )
            self._request_terminal_abort(error, expired[0], self._local_rank)

    def _stop_reconciliation_state(self) -> tuple[bool, BaseException | None]:
        """Return whether shutdown can stop MPI progress without splitting ranks."""
        snapshot = self._detected_state.snapshot()
        failed_ranks = snapshot.failed_ranks
        accepted_failure, accepted_generation = self._accepted_failure_state()
        with self._protocol_lock:
            reconciliation_state_is_valid = self._detection_reconciliation_state_is_valid_locked(
                snapshot,
                accepted_failure,
                accepted_generation,
            )
        if not reconciliation_state_is_valid:
            return False, RuntimeError(
                "WideEP FT cannot stop after the accepted failure epoch changed "
                f"(accepted_rank={accepted_failure}, "
                f"expected_generation={accepted_generation}, "
                f"snapshot={snapshot})"
            )
        if not failed_ranks:
            if self._transport_poisoned.is_set():
                return False, RuntimeError(
                    "WideEP FT cannot stop safely after an unattributed transport error"
                )
            return True, None

        with self._protocol_lock:
            unreconciled = [
                failed_rank
                for failed_rank in failed_ranks
                if not self._failure_detection_is_reconciled_locked(failed_rank, snapshot.mask)
            ]
            untracked = [
                failed_rank
                for failed_rank in unreconciled
                if failed_rank not in self._reconcile_deadlines
            ]
        if not unreconciled:
            return True, None
        if untracked:
            return False, RuntimeError(
                "WideEP FT cannot stop with locally failed ranks that were never announced: "
                f"{untracked}"
            )
        # Their existing reconciliation deadlines bound how long stop keeps
        # progressing. Expiry enters the normal terminal ABORT path.
        return False, None

    def _progress_terminal_abort(self) -> bool:
        """Return ``True`` after terminal state is globally safe to stop."""
        if self.ulfm_available:
            # Revoke is itself a communicator-wide terminal notification; it
            # does not depend on rendezvous Isends reaching completion.
            self._progress_failed.set()
            with self._lifecycle_lock:
                self._lifecycle = _Lifecycle.FAILED
            try:
                self._try_revoke()
            except Exception as error:
                logger.warning(f"WideEP FT terminal revoke failed: {error}")
                self._try_world_abort()
                return True
            if self.ulfm_available:
                return True
            # A runtime unsupported result disables ULFM and falls through to
            # the relayed-ABORT protocol. Clear the flag so the next loop pass
            # can continue progressing those requests.
            self._progress_failed.clear()

        now = time.monotonic()
        with self._protocol_lock:
            reconciled = self._abort_fanout_complete and self._abort_expected_reporters.issubset(
                self._abort_reporters
            )
            deadline = self._abort_deadline
        expired = deadline is not None and now >= deadline
        if not reconciled and not expired:
            return False

        self._progress_failed.set()
        with self._lifecycle_lock:
            self._lifecycle = _Lifecycle.FAILED
        # A final echo can race the deadline check. Once every expected
        # survivor has acknowledged ABORT, fail-stop escalation is no longer
        # needed even if the wall-clock deadline has just elapsed.
        if expired and not reconciled:
            self._try_world_abort()
        return True

    def _fail_closed_immediately(self, error: BaseException) -> None:
        """Abort globally after an error makes bounded relay unsafe."""
        with self._lifecycle_lock:
            self._record_error(error)
            self._progress_failed.set()
            self._lifecycle = _Lifecycle.FAILED
        if self.ulfm_available:
            try:
                self._try_revoke()
                if self.ulfm_available:
                    return
            except Exception as revoke_error:
                logger.warning(f"WideEP FT could not revoke after terminal error: {revoke_error}")
        self._try_world_abort()

    def _fail_closed_after_cleanup(self, error: BaseException) -> None:
        """Terminate the world when healthy MPI requests cannot be quiesced.

        At this point the progress loop has already stopped, so a relayed ABORT
        cannot be made reliable and ULFM revoke alone would not update peers'
        process-local shutdown state. Communicator-wide abort is the only
        deterministic way to prevent rank-asymmetric MPI_Finalize behavior.
        """
        self._record_error(error)
        self._progress_failed.set()
        with self._lifecycle_lock:
            self._lifecycle = _Lifecycle.FAILED
        self._try_world_abort()

    def _try_world_abort(self) -> None:
        """Last-resort fail-stop when terminal state cannot be reconciled."""
        with self._terminal_action_lock:
            if self._world_abort_attempted:
                return
            # MPI_Abort normally never returns. Mark the attempt before calling
            # it so test doubles, broken runtimes, or a later-resuming progress
            # thread cannot issue the communicator-wide action twice.
            self._world_abort_attempted = True
        abort = getattr(self._comm, "Abort", None)
        if not callable(abort):
            logger.error(
                "WideEP FT communicator has no MPI_Abort fallback; "
                "survivors must terminate through the poisoned-world shutdown hook"
            )
            return
        try:
            abort(1)
        except Exception as error:
            # Real MPI_Abort does not return. Test doubles and broken runtimes
            # may do so or raise; the local process remains terminal either way.
            logger.error(f"WideEP FT communicator-wide abort failed: {error}")

    def _retain_requests(self, requests: list[_PendingSend | _PendingReceive]) -> None:
        self._retained_requests.extend(requests)
        references: list[object] = list(requests)
        if not self._process_lifetime_retained:
            self._process_lifetime_retained = True
            references.append(self._comm)
        _retain_for_process_lifetime(*references)

    def _cleanup_requests(
        self,
        pending_sends: list[_PendingSend],
        pending_receives: dict[int, _PendingReceive],
    ) -> None:
        requests: list[_PendingSend | _PendingReceive] = [
            *pending_sends,
            *pending_receives.values(),
        ]
        # Never cancel, wait for, or free MPI operations after a peer or the
        # communicator has failed. Keeping each request together with its
        # NumPy buffer is the only bounded shutdown behavior in that state;
        # the later model-engine shutdown hook skips MPI_Finalize entirely.
        if self.last_error is not None or self.world_is_poisoned():
            self._retain_requests(requests)
            return

        if not requests:
            return

        # Healthy shutdown can locally cancel the preposted receives/sends.
        # Poll Test rather than calling Wait so stop() remains bounded even on
        # an MPI implementation with surprising cancellation behavior.
        for pending in requests:
            try:
                pending.request.Cancel()
            except Exception as error:
                if not isinstance(error, self._mpi.Exception):
                    raise
                logger.warning(f"WideEP FT could not cancel a healthy MPI request: {error}")

        remaining = requests
        deadline = self._shutdown_deadline or time.monotonic()
        while remaining and time.monotonic() < deadline:
            next_remaining: list[_PendingSend | _PendingReceive] = []
            for pending in remaining:
                try:
                    if not pending.request.Test():
                        next_remaining.append(pending)
                except Exception as error:
                    if not isinstance(error, self._mpi.Exception):
                        raise
                    next_remaining.append(pending)
            remaining = next_remaining
            if remaining:
                time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))

        if remaining:
            raise TimeoutError(
                f"{len(remaining)} WideEP FT MPI requests remained active after "
                "the healthy shutdown timeout"
            )

    def _record_error(self, error: BaseException) -> None:
        self._transport_poisoned.set()
        with self._error_lock:
            if self._last_error is None:
                self._last_error = error

    def _try_revoke(self) -> None:
        # Claim the one permitted Revoke attempt, but never hold a Python lock
        # across MPI. A caller-thread timeout and a later-resuming progress
        # thread may enter terminal handling concurrently; the claimant owns
        # any unsupported/error fallback.
        with self._terminal_action_lock:
            if not self._ulfm_available or self._revoke_attempted:
                return
            self._revoke_attempted = True
        try:
            self._comm.Revoke()
        except Exception as error:
            if isinstance(error, NotImplementedError) or self._is_unsupported_ulfm_error(error):
                with self._terminal_action_lock:
                    self._ulfm_available = False
                logger.warning(f"WideEP FT revoke is unavailable at runtime: {error}")
                return
            raise
