# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import math
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional

from tensorrt_llm._utils import ENABLE_MULTI_DEVICE, mpi_comm, mpi_disabled, print_all_stacks
from tensorrt_llm.logger import logger

# 137 == 128 + SIGKILL(9): the exit code a shell reports for a SIGKILL'd process.
_HARD_KILL_EXIT_CODE = 137

# Grace (seconds) between a rank's executor-loop crash and the hard kill of the
# whole world. Negative disables the kill entirely (escape hatch).
RANK_CRASH_KILL_GRACE_ENV = "TLLM_RANK_CRASH_HARD_KILL_GRACE"
_RANK_CRASH_KILL_GRACE_DEFAULT = 10.0


def _best_effort_flush_streams() -> None:
    """Flush stdout/stderr without ever raising; diagnostics must not block hard kill."""
    for stream in (sys.stderr, sys.stdout):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass


def _best_effort_log_error(message: str) -> None:
    """Log at error level without ever raising; diagnostics must not block hard kill."""
    try:
        logger.error(message)
    except Exception:  # noqa: BLE001 - diagnostics must not block hard kill
        pass


def _best_effort_log_debug(message: str) -> None:
    """Log at debug level without ever raising; diagnostics must not block hard kill."""
    try:
        logger.debug(message)
    except Exception:  # noqa: BLE001 - diagnostics must not block hard kill
        pass


def propagate_hard_kill(exit_code: int = _HARD_KILL_EXIT_CODE) -> None:
    """Hard-kill this rank and propagate the kill to peer ranks.

    Cross-rank propagation is the load-bearing part: a peer blocked in an NCCL
    collective would otherwise hold its GPU until the job's wall-clock pod-kill.

    - Preferred (when safe): ``MPI_Abort`` aborts the whole MPI job in one call.
      Only safe from the detector's daemon thread when MPI was initialized with
      ``MPI_THREAD_MULTIPLE``; guarded by ``Query_thread``.
    - Fallback: self-``SIGKILL``. The launcher (``mpirun`` propagates by default;
      ``srun`` needs ``--kill-on-bad-exit``) then tears down peers.

    All flushing and logging is best-effort: a closed/broken stdout, stderr, or
    logger must never prevent reaching ``MPI_Abort`` or ``os.kill``.
    """
    _best_effort_flush_streams()
    try:
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            from mpi4py import MPI

            if MPI.Is_initialized() and MPI.Query_thread() == MPI.THREAD_MULTIPLE:
                _best_effort_log_error(
                    "HangDetector: propagating hard-kill to all ranks via MPI_Abort."
                )
                mpi_comm().Abort(exit_code)
                return  # not reached; Abort does not return
    except Exception as e:  # noqa: BLE001 - last-resort path must not raise
        _best_effort_log_error(
            f"HangDetector: MPI_Abort propagation failed ({e}); falling back to self-SIGKILL."
        )
    _best_effort_log_error(
        "HangDetector: self-SIGKILL; relying on the launcher to propagate to peer ranks."
    )
    os.kill(os.getpid(), signal.SIGKILL)


def _rank_crash_kill_grace() -> Optional[float]:
    """Resolve the crash-kill grace period; ``None`` means the kill is disabled."""
    raw = os.environ.get(RANK_CRASH_KILL_GRACE_ENV)
    if raw is None:
        return _RANK_CRASH_KILL_GRACE_DEFAULT
    try:
        grace = float(raw)
    except ValueError:
        _best_effort_log_error(
            f"Invalid {RANK_CRASH_KILL_GRACE_ENV}={raw!r}; "
            f"using default {_RANK_CRASH_KILL_GRACE_DEFAULT}s"
        )
        return _RANK_CRASH_KILL_GRACE_DEFAULT
    if not math.isfinite(grace):
        # nan slips past `grace < 0` (every nan comparison is False) and then
        # collapses to a ZERO grace downstream: max(0.0, nan) returns 0.0, and
        # the direct path's `remaining > 0` guard is False too. That kills
        # instantly, destroying the very window the grace exists to provide.
        # inf is the mirror case: a watchdog thread that never fires, silently
        # equivalent to -1 but costing a live thread. Reject both.
        _best_effort_log_error(
            f"Non-finite {RANK_CRASH_KILL_GRACE_ENV}={raw!r}; "
            f"using default {_RANK_CRASH_KILL_GRACE_DEFAULT}s"
        )
        return _RANK_CRASH_KILL_GRACE_DEFAULT
    return None if grace < 0 else grace


def _remaining_kill_grace(grace: float, deadline: Optional[float]) -> float:
    """Time left before the kill must fire.

    ``deadline`` (a ``time.monotonic()`` stamp) lets a kill that was already
    armed elsewhere keep its ORIGINAL fire time when it is handed over to
    another waiter, so the handover cannot push the kill out by a second
    grace.

    The ``max(0.0, ...)`` is belt-and-braces only: it keeps the return value
    meaningful as "time left" for callers and logs. It is NOT what stops a
    negative sleep -- ``_wait_out_kill_grace`` does that with its
    ``remaining > 0`` guard (and ``Event.wait`` returns immediately for a
    negative timeout anyway). Do not drop that guard on the strength of this
    clamp.
    """
    if deadline is None:
        return grace
    return max(0.0, deadline - time.monotonic())


def _wait_out_kill_grace(remaining: float, cancelled: Optional[threading.Event]) -> bool:
    """Sleep out the crash-kill grace; return False if the kill was cancelled.

    ``cancelled`` makes the wait interruptible so the timer can be handed
    over to another waiter instead of two clocks running at once.

    The ``remaining > 0`` guard is load-bearing: a deadline already in the
    past must fire the kill now, and ``time.sleep`` of a negative duration
    would raise into ``hard_kill_on_rank_crash``'s blanket except and drop
    the kill -- precisely in the case (cleanup outlasted the grace) the
    watchdog exists for.
    """
    if cancelled is None:
        if remaining > 0:
            time.sleep(remaining)
        return True
    return not cancelled.wait(remaining)


def hard_kill_on_rank_crash(
    world_size: int,
    deadline: Optional[float] = None,
    cancelled: Optional[threading.Event] = None,
    error_delivered: Optional[threading.Event] = None,
) -> bool:
    """Hard-kill the whole world after this rank's executor loop crashed.

    A rank whose executor loop died on an exception can never rejoin its
    peers' collectives: without an explicit kill, every peer blocks in its
    next collective until its own HangDetector fires (300 s), and the whole
    test session burns that long for an error that was already known.

    The grace sleep before the kill is load-bearing: it gives the crashed
    rank's cleaner error paths time to win the race, so the client reports
    the ORIGINAL exception instead of a bare worker death —
    - rank-local response waiters woken by the executor-loop cleanup read
      the stashed error and surface it through the response path;
    - during init, the worker's ready handshake returns the real error to
      the proxy before the abort tears the world down;
    - the worker main thread returning lets its mpi4py future complete with
      the original exception.

    Never raises (it runs in a ``finally`` where an exception would mask the
    original loop error). Returns True when the kill path was taken — only
    observable in tests, where ``propagate_hard_kill`` is stubbed; in
    production that call does not return. Returns False when the kill does
    not apply (single rank, disabled by env) or was cancelled during the
    grace.
    """
    try:
        if world_size <= 1:
            # No peers to unblock; the worker's own death already completes
            # its future/handshake with the original exception.
            return False
        grace = _rank_crash_kill_grace()
        if grace is None:
            return False
        remaining = _remaining_kill_grace(grace, deadline)
        _best_effort_log_error(
            f"Executor loop crashed on this rank; hard-killing all "
            f"{world_size} ranks in {remaining:g}s (peers cannot make progress "
            f"without this rank). Set {RANK_CRASH_KILL_GRACE_ENV}=-1 to disable."
        )
        if not _wait_out_kill_grace(remaining, cancelled):
            # Debug, not error: the only caller that cancels does so to take
            # the same kill over on the same deadline. Logging "cancelled" at
            # ERROR right before the world is SIGKILLed reads during triage
            # as "the kill was called off", which is the opposite of what
            # happens.
            _best_effort_log_debug(
                "Rank-crash hard kill timer disarmed (handed over or no "
                "longer needed); this timer will not fire."
            )
            return False
        # The grace has elapsed. Before killing, check whether the crash
        # already reached the client.
        #
        # `crashed` upstream means "the loop raised before its break", which
        # is broader than "peers are stranded". In a SYMMETRIC crash -- a
        # deterministic Python error, a bad config, an OOM at the same batch --
        # every rank raises, nobody is stranded, and every rank arms this kill.
        # Firing then would replace N clean tracebacks with a bare exit 137.
        #
        # The kill exists to stop peers blocking in a collective forever. If
        # the stashed error has already been surfaced to the client, the
        # failure is diagnosable and the kill buys nothing, so skip it and let
        # the process exit normally with its original exception.
        if error_delivered is not None and error_delivered.is_set():
            _best_effort_log_error(
                "Rank-crash hard kill NOT fired: the executor-loop error "
                "already reached the client, so the failure is reportable "
                "without killing the world. Peers that are genuinely stranded "
                "are still covered -- in that case nothing consumes the error "
                "and this kill fires as before."
            )
            return False
        propagate_hard_kill()
        return True
    except Exception as e:  # noqa: BLE001 - must not mask the loop's original error
        _best_effort_log_error(f"hard_kill_on_rank_crash failed (ignored): {e!r}")
        return False


class RankCrashKillWatchdog(threading.Thread):
    """Daemon thread that hard-kills the world once the crash grace elapses.

    A plain ``Thread`` for backwards compatibility (callers may still join it
    or inspect ``daemon``), plus the two things needed to hand the timer over
    to another waiter instead of running two clocks:

    - ``cancel()`` disarms THIS timer. It is a bookkeeping aid, not a safety
      net: the only caller cancels in order to take the same kill over on the
      same deadline one line later, so cancelling does not spare a rank. What
      decides whether a rank is killed at all is the ``crashed`` predicate in
      ``PyExecutor._event_loop_wrapper``.
    - ``deadline`` exposes the original fire time so the caller that takes
      over still fires at crash + grace rather than restarting the clock.
    """

    def __init__(
        self, world_size: int, grace: float, error_delivered: Optional[threading.Event] = None
    ):
        super().__init__(name="rank_crash_kill_watchdog", daemon=True)
        self._world_size = world_size
        self.deadline = time.monotonic() + max(0.0, grace)
        self._cancelled = threading.Event()
        self._error_delivered = error_delivered

    def cancel(self) -> None:
        """Disarm the kill. Never raises; safe to call more than once."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:
        hard_kill_on_rank_crash(
            self._world_size,
            deadline=self.deadline,
            cancelled=self._cancelled,
            error_delivered=self._error_delivered,
        )


def start_rank_crash_kill_watchdog(
    world_size: int,
    error_delivered: Optional[threading.Event] = None,
) -> Optional[RankCrashKillWatchdog]:
    """Arm a daemon thread that hard-kills the world once the grace elapses.

    Must be armed BEFORE executor-loop cleanup: cleanup can block without
    bound (e.g. ``wait()`` on a pending PP send handle wedged by the crash),
    and a kill placed after it would never be reached — leaving peers to
    burn in their own 300 s HangDetectors, the exact failure this kill
    exists to avoid. The thread reuses ``hard_kill_on_rank_crash``, so the
    kill fires at crash + grace whether cleanup finishes, blocks, or raises.

    The returned watchdog is the ONLY timer while cleanup runs; the caller
    is expected to ``cancel()`` it once cleanup returns and carry the kill
    (with the same ``deadline``) itself, so the two paths never race with
    two independent clocks.

    Never raises. Returns the armed watchdog, or ``None`` when the kill is
    not applicable (single rank, disabled by env) or the thread could not
    be started — in that case the caller's post-cleanup kill remains the
    only mechanism.
    """
    try:
        if world_size <= 1:
            return None
        grace = _rank_crash_kill_grace()
        if grace is None:
            return None
        watchdog = RankCrashKillWatchdog(world_size, grace, error_delivered)
        watchdog.start()
        return watchdog
    except Exception as e:  # noqa: BLE001 - must not mask the loop's original error
        _best_effort_log_error(f"failed to arm rank-crash kill watchdog (ignored): {e!r}")
        return None


class HangDetector:
    """Watchdog that fires when the executor loop stops checkpointing.

    Contract:

    - ``timeout`` seconds without a ``checkpoint()`` dumps all thread stacks for
      diagnosis and runs ``on_detected`` (the hard-kill + cross-rank
      propagation path).
    - Continued checkpointing never fires it. A false positive hard-kills a
      healthy job, so this bound is as load-bearing as detection itself.
    - ``start()`` leaves detection disarmed; the first ``checkpoint()`` arms it,
      so the start-to-first-checkpoint window is not hang-eligible.
    - ``pause()`` suppresses detection in scope and re-arms on exit. It does
      not nest: leaving an inner ``pause()`` re-arms while an outer one is
      still open.
    - Detection never stops while active: not after firing, and not if
      ``on_detected`` raises an ``Exception``. ``on_detected`` is not
      idempotent, so a single lapse invokes it once.
    - ``checkpoint()`` is one clock read and one float store, and does no
      cross-thread work. The executor loop calls it three times per iteration.
    """

    def __init__(
        self, timeout: Optional[int] = None, on_detected: Optional[Callable[[], None]] = None
    ):
        self.timeout = timeout if timeout is not None else 300
        assert self.timeout > 0, "timeout must be greater than 0"
        self.on_detected = on_detected or (lambda: None)
        self.task = None
        self.loop = None
        self.loop_thread = None
        self.lock = threading.Lock()
        self.active = False
        self._detected = False
        self._status_providers: list[Callable[[], str]] = []
        # Monotonic stamp the watcher compares against; ``inf`` means disarmed.
        # A plain float store is the entire cost of ``checkpoint()``.
        self._deadline = math.inf

    def start(self):
        """Enable hang detection."""

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        with self.lock:
            # Locked, not a bare check: concurrent callers could both observe
            # ``active`` false and schedule a watcher, and watchers share
            # ``_deadline``, so a second one reports the same lapse twice and
            # propagates two hard kills.
            if self.active:
                _best_effort_log_error(
                    "HangDetector.start() called while already active; ignoring."
                )
                return
            # Disarmed until the first checkpoint so startup does not lapse.
            # Stored before ``active`` is published so a checkpoint racing this
            # call cannot have its arm overwritten here.
            self._deadline = math.inf
            self.active = True

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=run_loop, daemon=True, name="hang_detector_loop")
        self.loop_thread.start()
        # One long-lived watcher, scheduled once; the hot path only moves
        # ``_deadline``.
        self.task = asyncio.run_coroutine_threadsafe(self._watch(), self.loop)

    def register_status_provider(self, provider: Callable[[], str]) -> None:
        """Register a nonblocking callable that returns status to dump on hang detection."""
        with self.lock:
            self._status_providers.append(provider)

    async def _watch(self) -> None:
        """Sleep until the deadline lapses, report, and keep watching.

        Waking early is normal: ``checkpoint()`` pushes ``_deadline`` forward
        without touching this task, so each wake-up either finds time left and
        sleeps again, or finds the deadline passed and reports. Every sleep is
        clamped to ``timeout`` because ``checkpoint()`` only stores a float and
        never wakes this loop, so an unclamped sleep would not notice a later
        arm.

        This task never writes ``_deadline``; the lapse it last reported is
        watcher-local.

        This task outlives a report, and outlives a report that raises. A
        watchdog that quietly stopped watching would be the exact failure it
        exists to catch, and ``on_detected`` is the cross-rank hard kill, which
        can itself fail on an already-degraded job.
        """
        # The deadline whose lapse already ran ``on_detected``. Compared by
        # identity, not equality: ``checkpoint()`` publishes a fresh float, so a
        # re-arm that lands on the same value still reports.
        reported = None
        while self.active:
            # Clock first: a checkpoint can land between these two reads, and
            # whichever is read first is the stale one. A stale deadline fires
            # at work that was checkpointed in time; a stale clock only defers.
            now = time.monotonic()
            deadline = self._deadline
            remaining = deadline - now
            if remaining > 0:
                await asyncio.sleep(min(remaining, self.timeout))
                continue
            if deadline is reported:
                # ``on_detected`` is not idempotent, so one lapse runs it once.
                # Nothing wakes this loop, so poll for the next arm.
                await asyncio.sleep(self.timeout)
                continue
            reported = deadline
            try:
                await self._report_hang()
            except Exception as error:  # noqa: BLE001 - the watcher must survive
                _best_effort_log_error(
                    f"HangDetector: reporting failed with {type(error).__name__}: {error}"
                )

    async def _report_hang(self) -> None:
        with self.lock:
            status_providers = tuple(self._status_providers)

        # All diagnostics are best-effort: nothing may prevent on_detected()
        # (hard-kill propagation) from firing.
        _best_effort_log_error(f"Hang detected after {self.timeout} seconds.")
        for provider in status_providers:
            try:
                status = provider()
                if status:
                    _best_effort_log_error(status)
            except Exception as error:  # noqa: BLE001 - isolate diagnostic providers
                _best_effort_log_error(
                    f"HangDetector: status provider failed with {type(error).__name__}: {error}"
                )
        try:
            print_all_stacks()
        except Exception:  # noqa: BLE001 - stack dump must not block hard kill
            pass

        # Set _detected last so observers (and tests) see it only once
        # diagnostics are done and on_detected is about to fire.
        with self.lock:
            self._detected = True
        self.on_detected()

    def detected(self):
        """Return True if hang is detected."""
        with self.lock:
            return self._detected

    def checkpoint(self):
        """Reset hang detection timer."""
        if self.active:
            self._deadline = time.monotonic() + self.timeout

    def disarm(self) -> None:
        """Disarm hang detection until the next checkpoint."""
        self._deadline = math.inf

    def cancel_task(self) -> None:
        """Compatibility alias for :meth:`disarm`.

        The watcher is long-lived and has no task to cancel, but the old name is
        load-bearing for the cache-transceiver precheck and its SLURM example.
        Delegating rather than aliasing keeps a subclass override of ``disarm``
        effective through this name.
        """
        self.disarm()

    @contextmanager
    def pause(self):
        """Pause hang detection in scope."""
        self.disarm()
        try:
            yield
        finally:
            self.checkpoint()

    def stop(self):
        """Stop hang detection."""
        self.active = False
        self.disarm()
        if self.loop is not None:
            # Cancel all pending tasks before stopping the loop
            def cancel_all_tasks():
                for task in asyncio.all_tasks(self.loop):
                    if not task.done():
                        task.cancel()
                self.loop.call_soon(self.loop.stop)

            self.loop.call_soon_threadsafe(cancel_all_tasks)

            if self.loop_thread is not None and self.loop_thread.is_alive():
                self.loop_thread.join()

            self.loop = None
            self.loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
