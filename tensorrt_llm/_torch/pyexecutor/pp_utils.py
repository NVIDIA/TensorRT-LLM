# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import threading
from enum import IntEnum
from queue import Empty, Queue
from typing import Callable, Optional, Tuple, TypeVar

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger


class PPCommTag(IntEnum):
    """
    Unique tags for pipeline parallelism communication.
    """

    TERMINATION = 20000
    SCHEDULE_RESULT = 20001
    EXECUTED_BATCH_NUM = 20002
    SAMPLE_STATE = 20003
    # Reserved and never sent, so the MPI progress pump's MPI_Iprobe always
    # misses. Kept well apart from the tags above, which share its
    # communicator.
    MPI_PROGRESS_PROBE = 20100


# Idle-time MPI progress for pipeline parallelism.
#
# MPI only progresses on threads that call into MPI, and the PP executor drives
# all PP messages from one thread. While that thread blocks in a CUDA
# synchronization, a rendezvous-protocol send it already issued cannot
# complete, which can deadlock the PP ring when pp_size >= 3. The sample-state
# relay thread therefore polls its queue with a timeout and issues a
# non-matching MPI_Iprobe between polls. A native call that holds the GIL
# across a whole CUDA synchronization can still starve it.
MPI_PROGRESS_POLL_MS_ENV_VAR_NAME = "TLLM_PP_MPI_PROGRESS_POLL_MS"
# CPython's default thread switch interval.
DEFAULT_MPI_PROGRESS_POLL_MS = 5.0
# A tiny period spins a core and competes with the executor thread for the
# GIL; a huge one can overflow Queue.get(timeout=...).
MIN_MPI_PROGRESS_POLL_MS = 0.5
MAX_MPI_PROGRESS_POLL_MS = 1000.0

# (pump, poll interval in seconds); see make_mpi_progress_pump.
MpiProgressPump = Tuple[Callable[[], bool], float]

_T = TypeVar("_T")


def resolve_mpi_progress_poll_interval_ms() -> float:
    """Resolve the MPI progress poll period from the environment, in ms.

    Returns the default when the variable is unset or not a finite number. A
    value <= 0 is returned unchanged and means "disabled"; any other value is
    clamped into ``[MIN_MPI_PROGRESS_POLL_MS, MAX_MPI_PROGRESS_POLL_MS]``.

    Messages are logged at ERROR because only the leader rank lowers the log
    level, so a WARNING would be dropped on every other rank by default.
    """
    raw = os.environ.get(MPI_PROGRESS_POLL_MS_ENV_VAR_NAME)
    if raw is None:
        return DEFAULT_MPI_PROGRESS_POLL_MS
    try:
        poll_interval_ms = float(raw)
    except ValueError:
        poll_interval_ms = math.nan
    if not math.isfinite(poll_interval_ms):
        logger.error(
            f"Ignoring malformed {MPI_PROGRESS_POLL_MS_ENV_VAR_NAME}={raw!r}; "
            f"using {DEFAULT_MPI_PROGRESS_POLL_MS} ms."
        )
        return DEFAULT_MPI_PROGRESS_POLL_MS
    if poll_interval_ms <= 0:
        return poll_interval_ms
    clamped_ms = min(max(poll_interval_ms, MIN_MPI_PROGRESS_POLL_MS), MAX_MPI_PROGRESS_POLL_MS)
    if clamped_ms != poll_interval_ms:
        logger.error(
            f"{MPI_PROGRESS_POLL_MS_ENV_VAR_NAME}={raw!r} is outside "
            f"[{MIN_MPI_PROGRESS_POLL_MS}, {MAX_MPI_PROGRESS_POLL_MS}] ms; "
            f"using {clamped_ms} ms instead."
        )
    return clamped_ms


def make_mpi_progress_pump(
    comm, stop_event: threading.Event, quiesced_event: threading.Event
) -> Optional[MpiProgressPump]:
    """Build the idle-time MPI progress pump for the sample-state relay thread.

    Args:
        comm: The communicator owned by the relay thread, never the executor
            thread's.
        stop_event: Set to retire the pump.
        quiesced_event: Set by the pump whenever it returns ``False``, that
            is, once it will issue no further MPI call.

    Returns:
        A ``(pump, poll_interval_s)`` pair, or ``None`` when the pump is
        disabled or unavailable. ``pump()`` never raises and returns
        ``False`` once it must not be called again.

    The probe is an ``MPI_Iprobe`` on ``PPCommTag.MPI_PROGRESS_PROBE``, which
    always misses and only drives the progress engine. It must stay a plain
    ``Iprobe``: ``Improbe``/``Mprobe`` remove the matched message and would
    race pkl5's ``Probe``-then-``Recv`` on the same communicator. For the same
    reason the pump never calls ``Test``/``Wait`` on requests owned by the
    executor thread.
    """
    poll_interval_ms = resolve_mpi_progress_poll_interval_ms()
    if poll_interval_ms <= 0:
        logger.error(
            f"{MPI_PROGRESS_POLL_MS_ENV_VAR_NAME}={poll_interval_ms}: "
            f"the pipeline parallelism MPI progress pump is disabled."
        )
        return None
    if mpi_disabled():
        return None

    from mpi4py import MPI

    # The relay thread calls MPI concurrently with the executor thread.
    provided = MPI.Query_thread()
    if provided < MPI.THREAD_MULTIPLE:
        logger.error(
            f"MPI thread level {provided} is below MPI_THREAD_MULTIPLE "
            f"({int(MPI.THREAD_MULTIPLE)}); the pipeline parallelism MPI "
            f"progress pump is disabled."
        )
        return None

    any_source = MPI.ANY_SOURCE
    probe_tag = int(PPCommTag.MPI_PROGRESS_PROBE)

    def pump() -> bool:
        """Drive one progress tick.  Never raises; returns still-armed."""
        # Nothing may escape and kill the relay thread. A torn-down MPI can
        # also fail with non-MPI exceptions, so any failure disarms the pump.
        try:
            if stop_event.is_set():
                quiesced_event.set()
                return False
            # A finalize that is still running on another thread is covered
            # by the exit hook from make_mpi_progress_exit_hook.
            if MPI.Is_finalized():
                quiesced_event.set()
                return False
            comm.Iprobe(source=any_source, tag=probe_tag)
            return True
        except Exception as e:  # noqa: BLE001 - see the comment above
            try:
                logger.error(
                    f"The pipeline parallelism MPI progress pump hit an error "
                    f"and is now disabled: {e!r}"
                )
            except Exception:  # noqa: BLE001 - logging must not kill the relay
                pass
            quiesced_event.set()
            return False

    return pump, poll_interval_ms / 1000.0


def make_mpi_progress_exit_hook(
    stop_event: threading.Event, quiesced_event: threading.Event, poll_interval_s: float
) -> Callable[[], None]:
    """Build an ``atexit`` hook that retires the pump before MPI_Finalize.

    Some exit paths never reach ``PyExecutor.shutdown()``, so the daemon relay
    thread could still be inside ``MPI_Iprobe`` when mpi4py finalizes MPI. A
    hook registered after mpi4py was imported runs before that finalization.
    It sets ``stop_event`` and waits, bounded, for ``quiesced_event``. It must
    not hold a reference to the executor.
    """
    quiesce_timeout_s = min(max(20.0 * poll_interval_s, 0.1), 1.0)

    def _retire_mpi_progress_pump() -> None:
        stop_event.set()
        quiesced_event.wait(timeout=quiesce_timeout_s)

    return _retire_mpi_progress_pump


def get_with_mpi_progress(
    queue: "Queue[_T]", progress: Optional[MpiProgressPump], stop_event: threading.Event
) -> _T:
    """Get the next item from ``queue``, running the pump while waiting.

    Without a pump this is a plain blocking ``queue.get()``. Once the pump
    retires, ``stop_event`` is set and the timed wait continues without it.
    """
    if progress is None:
        return queue.get()
    pump, poll_interval_s = progress
    while True:
        try:
            return queue.get(timeout=poll_interval_s)
        except Empty:
            pass
        if not pump():
            stop_event.set()
