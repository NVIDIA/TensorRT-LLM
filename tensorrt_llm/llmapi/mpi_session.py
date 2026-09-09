# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import itertools
import math
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import wait as futures_wait
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import zmq

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.logger import logger

from .._utils import global_mpi_rank, mpi_barrier, mpi_rank
from .utils import logger_debug, print_colored

if ENABLE_MULTI_DEVICE:
    import mpi4py
    from mpi4py.futures import MPICommExecutor, MPIPoolExecutor

    from tensorrt_llm._utils import global_mpi_size, mpi_world_size

T = TypeVar("T")

_FLASHINFER_WORKSPACE_ROOT = "~/.cache/tensorrt_llm/flashinfer"
_FLASHINFER_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"
_FLASHINFER_WORKSPACE_MANAGED_ENV = "TRTLLM_FLASHINFER_WORKSPACE_MANAGED"
_FLASHINFER_WORKER_BOOTSTRAP = """
import fcntl
import os
import sys
import tempfile
from pathlib import Path

from mpi4py import MPI

workspace_lock = None
temporary_workspace = None
rank = "unknown"
# ``MPIPoolExecutor(env=...)`` overlays the inherited environment; omitting a
# variable does not unset it. This bootstrap is selected only when automatic
# isolation is required, so replace any launcher-managed parent workspace.
os.environ.pop("FLASHINFER_WORKSPACE_BASE", None)
try:
    rank = MPI.COMM_WORLD.Get_rank()
    workspace_root = Path(sys.argv[1]).expanduser()
    slot = rank
    slot_stride = MPI.COMM_WORLD.Get_size()
    # Reuse the rank's cache when possible. Concurrent pools with the same
    # rank skip locked slots in world-size strides, keeping every worker apart.
    # Slots intentionally persist for JIT cache reuse, so the workspace root
    # can grow to the high-water mark of concurrent pools.
    while True:
        workspace = workspace_root / f"rank-{slot}"
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_lock = (workspace / ".lock").open("a")
        try:
            fcntl.flock(workspace_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            workspace_lock.close()
            workspace_lock = None
            slot += slot_stride

    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(workspace)
except Exception as error:  # noqa: BLE001
    if workspace_lock is not None:
        try:
            workspace_lock.close()
        except Exception as close_error:  # noqa: BLE001
            print(
                f"[trtllm] rank {rank} could not close a failed FlashInfer "
                f"workspace lock ({close_error})",
                file=sys.stderr,
            )
    workspace_lock = None

    try:
        temporary_workspace = tempfile.TemporaryDirectory(
            prefix=f"trtllm-flashinfer-rank-{rank}-"
        )
    except Exception as temporary_error:  # noqa: BLE001
        raise RuntimeError(
            f"rank {rank} could not create an isolated FlashInfer workspace; "
            f"persistent setup failed with {error} and temporary setup "
            f"failed with {temporary_error}. Configure "
            f"FLASHINFER_WORKSPACE_BASE to a writable process-unique path, "
            f"or set TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS=0 to disable "
            f"automatic isolation if the shared-workspace risk is acceptable"
        ) from temporary_error
    os.environ["FLASHINFER_WORKSPACE_BASE"] = temporary_workspace.name
    print(
        f"[trtllm] rank {rank} could not use a persistent FlashInfer "
        f"workspace ({error}); using temporary workspace "
        f"{temporary_workspace.name}",
        file=sys.stderr,
    )

os.environ["TRTLLM_FLASHINFER_WORKSPACE_MANAGED"] = "1"

# Preserve FlashInfer's default cubin cache without importing
# flashinfer.jit.env. The environment must be fully configured before that
# module initializes its workspace constants.
if "FLASHINFER_CUBIN_DIR" not in os.environ and os.environ.get("HOME"):
    os.environ["FLASHINFER_CUBIN_DIR"] = str(
        Path(os.environ["HOME"]) / ".cache" / "flashinfer" / "cubins"
    )

from mpi4py.futures.server import main

# Hold the lock for the server's lifetime. The kernel also releases it when a
# worker exits abnormally, so a later launch can safely reuse the cache slot.
try:
    main()
finally:
    if workspace_lock is not None:
        try:
            fcntl.flock(workspace_lock, fcntl.LOCK_UN)
        except OSError as error:
            print(
                f"[trtllm] rank {rank} could not unlock the FlashInfer "
                f"workspace ({error})",
                file=sys.stderr,
            )
        try:
            workspace_lock.close()
        except OSError as error:
            print(
                f"[trtllm] rank {rank} could not close the FlashInfer "
                f"workspace lock ({error})",
                file=sys.stderr,
            )
    if temporary_workspace is not None:
        try:
            temporary_workspace.cleanup()
        except OSError as error:
            print(
                f"[trtllm] rank {rank} could not remove the temporary "
                f"FlashInfer workspace ({error})",
                file=sys.stderr,
            )
"""


class MPINodeState:
    """MPINodeState acts as a central global state shares between tasks on MPI node.

    An example:
        def task():
            if MPINodeState.state is None:
                MPINodeState.state = 0
            MPINodeState.state += 1
            return MPINodeState.state

        n_workers = 4
        with MPIPoolExecutor(max_workers=n_workers) as executor:
            for i in range(2):
                futures = [executor.submit(task) for i in range(n_workers)]

        This should produce the following output:
        - [1, 1, 1, 1]
        - [2, 2, 2, 2]
    """

    state = None
    # Global MPICommExecutor instance to be reused across multiple MpiCommSession instances
    # This is necessary because MPICommExecutor can only be created once per MPI process
    _global_comm_executor = None
    _global_mpi_pool = None

    @staticmethod
    def is_initialized() -> bool:
        return MPINodeState.state is not None


def external_mpi_comm_available(model_world_size: int) -> bool:
    """Check if the current process is launched by mpirun and does not use MPIPoolExecutor to spawn processes.
    e.g. mpirun -np 4 python script.py
    """
    if ENABLE_MULTI_DEVICE:
        return (get_mpi_world_size() == model_world_size
                and model_world_size > 1) or (global_mpi_size()
                                              > get_mpi_world_size())
    else:
        return False


def need_spawn_mpi_workers(model_world_size: int) -> bool:
    """Check if the current process needs to spawn MPI workers."""
    if ENABLE_MULTI_DEVICE:
        return get_mpi_world_size() == 1 and model_world_size > 1
    else:
        return False


def set_mpi_session_cpp(comm):
    if ENABLE_MULTI_DEVICE:
        comm_fortran = comm.py2f()
        from tensorrt_llm.bindings import MpiComm
        MpiComm.set_raw_mpi_session_by_fortran_handle(comm_fortran)


def validate_session_world_size(mpi_session, model_world_size: int) -> None:
    """Fail loudly when an external session cannot serve ``model_world_size``.

    ``submit()`` launches one worker task per pool worker, so an externally
    provided session must match the model's world size exactly; otherwise the
    wrong number of executors would start.
    """
    external_workers = getattr(mpi_session, "n_workers", None)
    if external_workers is not None and external_workers != model_world_size:
        raise ValueError(
            f"External MPI session has {external_workers} workers but "
            f"the model needs world_size={model_world_size}.")


class MpiSession(abc.ABC):

    @abc.abstractmethod
    def submit(self, task: Callable[..., T], *args,
               **kwargs) -> List[Future[T]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        raise NotImplementedError()

    @abc.abstractmethod
    def shutdown(self, wait=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def abort(self):
        raise NotImplementedError()

    def is_comm_session(self) -> bool:
        return isinstance(self, (MpiCommSession, RemoteMpiCommSessionClient))

    def _abort_on_timeout(self, fut: Future, timeout: float, reason=None):
        try:
            fut.result(timeout=timeout)
        except TimeoutError:
            logger.critical(f"MpiSession shutdown timeout after {timeout}s; "
                            "calling MPI_Abort to force-kill stuck ranks...")
            if reason is not None:
                logger.info(f"Reason to shutdown: {reason!r}")
            self.abort()
            logger.critical("MpiSession MPI_Abort returned")

    def shutdown_abort(self, grace: float = 60, reason=None):
        if sys.is_finalizing():
            # cannot start thread at interpreter shutdown
            # simply don't wait to avoid hang
            return self.shutdown(wait=False)

        logger.info(
            f"MpiSession.shutdown_abort: waiting up to {grace}s for workers to exit"
        )
        fut = Future()
        killer = threading.Thread(group=None,
                                  target=self._abort_on_timeout,
                                  name="MpiSessionTimeoutKiller",
                                  args=(fut, grace, reason))
        killer.start()
        self.shutdown()
        logger.info("MpiSession.shutdown_abort: workers exited cleanly")
        fut.set_result(None)
        killer.join()

    def release_exit_joins(self):
        """Mark the worker world dead and release anything that would join it.

        Non-destructive, so it may be called by a component that does not
        own the session. Must not tear the session down -- only ensure that
        nothing (interpreter exit, a later blocking ``shutdown()`` by the
        owner) waits forever on the dead world. Default: no-op.
        """

    def abandon(self):
        """Tear the session down without waiting on a dead worker world."""
        self.release_exit_joins()
        self.shutdown(wait=False)


def _abandon_mpi_pool_threads(mpi_pool) -> None:
    """Let interpreter exit proceed despite a wedged pool manager thread.

    When the worker world dies abruptly, the ``MPIPoolExecutor`` manager
    thread stays blocked in an MPI call forever, and process exit hangs on
    it twice: mpi4py's exit hook joins every registered manager thread, and
    CPython joins every non-daemon thread. Deregister the thread from both;
    it is reaped with the process.

    Best-effort: the touched names are private to mpi4py (``THREADS_QUEUES``
    in ``_lib``/3.x and ``_core``/4.x) and CPython
    (``threading._shutdown_locks``, 3.9-3.12). Where a name is absent, that
    mechanism is left alone and exit may still block on it.
    """
    thread = getattr(getattr(mpi_pool, '_pool', None), 'thread', None)
    if thread is None:
        return
    # mpi4py's own exit hook (joins all registered manager threads).
    for mod_name in ('mpi4py.futures._lib', 'mpi4py.futures._core'):
        mod = sys.modules.get(mod_name)
        registry = getattr(mod, 'THREADS_QUEUES', None) if mod else None
        if registry is not None:
            try:
                registry.pop(thread, None)
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.debug(f"THREADS_QUEUES cleanup failed (ignored): {e!r}")
    # CPython's non-daemon thread join at interpreter shutdown.
    tstate_lock = getattr(thread, '_tstate_lock', None)
    shutdown_locks = getattr(threading, '_shutdown_locks', None)
    if tstate_lock is not None and shutdown_locks is not None:
        try:
            shutdown_locks.discard(tstate_lock)
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            logger.debug(f"_shutdown_locks cleanup failed (ignored): {e!r}")


def _process_start_time(pid: int) -> Optional[bytes]:
    """Kernel start time (jiffies since boot) of ``pid``, or None if gone.

    PIDs are recycled by the OS, but the (pid, start_time) pair uniquely
    identifies a process incarnation — comparing it prevents waiting on an
    unrelated process that inherited a dead worker's PID.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read()
        # Field 2 (comm) may contain spaces/parens; parse after the last ')'.
        return stat.rsplit(b")", 1)[1].split()[19]  # field 22 overall
    except OSError:
        return None


_DEFAULT_IDENTITY_TIMEOUT = 300.0


def _identity_barrier_timeout() -> float:
    """Deadline for the ``wait_shutdown`` worker-identity barrier, in seconds.

    The barrier itself completes in milliseconds, but it is the first work ever
    submitted to a freshly built ``MPIPoolExecutor``, and mpi4py spawns lazily
    from its manager thread — so this deadline really bounds the whole worker
    bootstrap: process spawn plus ``import tensorrt_llm``, measured at ~50-65s
    on an idle node and up to ~117s on a contended one. Hence a ceiling sized
    against bootstrap cost rather than barrier latency. The test-session
    prefetcher derives its own wait budget from this value so it cannot abandon
    a bootstrap that this layer still considers healthy.
    ``TRTLLM_MPI_IDENTITY_TIMEOUT`` overrides it.
    """
    raw = os.environ.get("TRTLLM_MPI_IDENTITY_TIMEOUT")
    if not raw:
        return _DEFAULT_IDENTITY_TIMEOUT
    try:
        value = float(raw)
        if math.isfinite(value) and value > 0:
            return value
    except ValueError:
        pass
    logger.warning(f"Ignoring invalid TRTLLM_MPI_IDENTITY_TIMEOUT={raw!r}; "
                   f"using {_DEFAULT_IDENTITY_TIMEOUT}s")
    return _DEFAULT_IDENTITY_TIMEOUT


def _worker_identity_barrier():
    """Runs inside a pool worker; module-level so it is picklable.

    The leading barrier pins the ``n_workers`` submitted tasks one-per-worker
    (a worker holding one task blocks until every other worker holds its own,
    so no worker can drain a second one), collecting every worker's identity
    exactly once. The workers' ``MPI_COMM_WORLD`` is the spawned worker world
    (the parent process is not a member).
    """
    from mpi4py import MPI
    MPI.COMM_WORLD.barrier()
    pid = os.getpid()
    return (pid, _process_start_time(pid))


class MpiPoolSession(MpiSession):

    def __init__(self,
                 n_workers: int,
                 wait_shutdown: bool = False,
                 env_overrides: Optional[Dict[str, str]] = None):
        """Spawn a pool of MPI worker processes.

        Args:
            n_workers: number of MPI workers to spawn.
            wait_shutdown: when True, ``shutdown()`` blocks until the spawned
                worker processes have actually exited.
                ``MPIPoolExecutor.shutdown`` returns at disconnect, but a
                worker's GPU memory is only released when its process exits;
                callers that start new GPU work right after ``shutdown()``
                (e.g. CI test infrastructure handing a pre-spawned pool to the
                next test) race that release and can OOM. Off by default:
                production teardown does not need the barrier and keeps its
                current latency.
            env_overrides: extra environment variables to set in the WORKERS at
                spawn, on top of the TRTLLM*/TLLM* variables forwarded from the
                parent. The parent process environment is never touched — this
                replaces the racy "set os.environ around the spawn, then
                restore" pattern for callers that spawn pools from background
                threads.
        """
        self.n_workers = n_workers
        self._wait_shutdown = wait_shutdown
        self._env_overrides = dict(env_overrides) if env_overrides else {}
        self._worker_identities: Tuple = ()
        self.mpi_pool: Optional[MPIPoolExecutor] = None
        self._start_mpi_pool()
        if wait_shutdown:
            self._worker_identities = self._collect_worker_identities()
        if ENABLE_MULTI_DEVICE:
            self.comm = mpi4py.MPI.COMM_WORLD

    def get_comm(self):
        return self.comm

    def submit(self, task: Callable[..., T], *args,
               **kwargs) -> List[Future[T]]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self, wait=True):
        if getattr(self, '_pool_dead', False):
            # A dead pool can never be joined; never block on it, no matter
            # what the caller asked for.
            wait = False
        if self.mpi_pool is not None:
            logger.info(
                f"MpiPoolSession.shutdown: joining {self.n_workers} worker(s) "
                f"(wait={wait})")
            self.mpi_pool.shutdown(wait=wait)
            logger.info("MpiPoolSession.shutdown: done")
            self.mpi_pool = None
            if self._wait_shutdown:
                self._wait_workers_exit()

    def _collect_worker_identities(self) -> Tuple:
        """(pid, start_time) of every worker, recorded right after spawn.

        FAIL-CLOSED (review requirement): ``wait_shutdown=True`` is a
        contract — shutdown blocks until the workers exited. A pool without
        complete identities cannot honor it, and returning it anyway would
        silently downgrade to the old non-waiting behavior (the timeout can
        trip on a slow-but-healthy bootstrap, and ``futures_wait`` does not
        cancel the pending tasks). Instead of handing out such a pool, tear
        it down and raise; callers fall back to a fresh spawn.
        """
        timeout = _identity_barrier_timeout()
        try:
            futures = [
                self.mpi_pool.submit(_worker_identity_barrier)
                for _ in range(self.n_workers)
            ]
            done, not_done = futures_wait(futures, timeout=timeout)
            identities = tuple(f.result() for f in done)
        except Exception as e:
            self._teardown_unidentified_pool(())
            raise RuntimeError(
                f"MpiPoolSession(wait_shutdown=True): worker identity "
                f"collection failed ({e}); pool torn down") from e
        if (not_done or len(identities) != self.n_workers
                or len({pid
                        for pid, _ in identities}) != self.n_workers
                or any(start is None for _, start in identities)):
            self._teardown_unidentified_pool(identities)
            raise RuntimeError(
                "MpiPoolSession(wait_shutdown=True): worker identity "
                f"collection incomplete ({len(identities)}/{self.n_workers} "
                "valid identities); pool torn down instead of handing out a "
                "session that cannot honor the wait_shutdown contract. Raise "
                "TRTLLM_MPI_IDENTITY_TIMEOUT if worker bootstrap is merely "
                f"slow (deadline was {timeout}s)")
        return identities

    def _teardown_unidentified_pool(self, partial_identities: Tuple) -> None:
        """Dispose of a pool whose identity collection failed.

        The workers may be stuck in the collection barrier (one of them
        never picked up its task), so a graceful blocking shutdown could
        hang; disconnect without waiting and SIGKILL the workers we did
        identify (with the pid-recycling guard). Workers we never identified
        exit with the MPI runtime teardown; if one is truly wedged it leaks
        until job end — the same bounded leak class as any wedged pool.
        """
        import signal

        try:
            self.mpi_pool.shutdown(wait=False)
        except Exception:
            pass
        self.mpi_pool = None
        for pid, start in partial_identities:
            if start is None or _process_start_time(pid) != start:
                continue  # gone already, or the PID was recycled
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _wait_workers_exit(self, timeout: float = 30.0) -> None:
        """Block until the spawned worker processes have actually exited.

        Bounded: a wedged worker stops blocking the caller after ``timeout``
        (its memory is not coming back anyway; the caller's own recovery —
        e.g. an OOM retry or a fresh spawn — takes over from there).
        """
        deadline = time.monotonic() + timeout
        for pid, start in self._worker_identities:
            if start is None:
                continue
            while _process_start_time(pid) == start:
                if time.monotonic() >= deadline:
                    logger.warning(
                        f"MpiPoolSession.shutdown: worker pid {pid} still "
                        f"alive after {timeout}s; not waiting further")
                    return
                time.sleep(0.05)

    def release_exit_joins(self):
        if self.mpi_pool is not None:
            _abandon_mpi_pool_threads(self.mpi_pool)
        self._pool_dead = True

    def abort(self):
        self.get_comm().Abort(1)

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        env = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("TRTLLM") or key.startswith("TLLM") or key in (
                "FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUBIN_DIR")
        }
        workspace_managed = env.get(_FLASHINFER_WORKSPACE_MANAGED_ENV) == "1"
        env.update(self._env_overrides)
        explicit_workspace_override = (_FLASHINFER_WORKSPACE_ENV
                                       in self._env_overrides)
        isolate_workspace = (
            (self.n_workers > 1 or workspace_managed)
            and env.get("TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS", "1") != "0"
            and (_FLASHINFER_WORKSPACE_ENV not in env or
                 (workspace_managed and not explicit_workspace_override)))
        if isolate_workspace:
            env.pop(_FLASHINFER_WORKSPACE_ENV, None)
        elif explicit_workspace_override:
            # The override is user-owned, including in any further nested pool.
            env.pop(_FLASHINFER_WORKSPACE_MANAGED_ENV, None)
        python_args = ([
            "-c", _FLASHINFER_WORKER_BOOTSTRAP, _FLASHINFER_WORKSPACE_ROOT
        ] if isolate_workspace else None)
        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers,
                                        path=sys.path,
                                        env=env,
                                        python_args=python_args)

    def __del__(self):
        self.shutdown_abort()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


class MpiCommSession(MpiSession):

    def __init__(self, comm=None, n_workers: int = 1):
        self.comm = comm
        self.n_workers = n_workers
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.mpi_pool: Optional[MPIPoolExecutor] = None
        self.owns_mpi_pool = False  # Track if this instance owns the mpi_pool

        if n_workers <= 0:
            raise ValueError(
                f'n_workers must be non-negative, but got {n_workers}')

        if ENABLE_MULTI_DEVICE:
            if not self.comm:
                self.comm = mpi4py.MPI.COMM_WORLD

            if self.comm.Get_rank() != 0:
                raise RuntimeError(
                    f'only rank 0 can start multi-node session, got {self.comm.Get_rank()}'
                )

            if self.comm.Get_size() != n_workers:
                raise ValueError(
                    f'n_workers must be equal to the number of processes in MPI, got {n_workers} vs {get_mpi_world_size()}'
                )

        self._start_mpi_pool()

    def get_comm(self):
        return self.comm

    def submit(self, task: Callable[..., T], *args,
               **kwargs) -> List[Future[T]]:
        """Submit a task to MPI workers.

        Args:
            task: The task to be submitted.
            args: Positional arguments for the task.
            kwargs: Keyword arguments for the task.
        """
        assert self.mpi_pool is not None, 'MPI session not started'
        worker_futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers - 1)
        ]

        rank0_future = self.thread_pool.submit(task, *args, **kwargs)
        return [rank0_future] + worker_futures

    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        futures = self.submit(task, *args, **kwargs)
        return [future.result() for future in futures]

    def shutdown(self, wait=True):
        # Only shutdown the mpi_pool if this instance created it
        # For shared global mpi_pool, we don't shut it down
        if self.mpi_pool is not None and self.owns_mpi_pool:
            logger.info(
                f"MpiCommSession.shutdown: joining {self.n_workers - 1} worker(s) "
                f"(wait={wait})")
            self.mpi_pool.shutdown(wait=wait)
            logger.info("MpiCommSession.shutdown: mpi_pool done")
        self.mpi_pool = None
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=wait)
            self.thread_pool = None

    def abort(self):
        self.get_comm().Abort(1)

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Use global MPICommExecutor if using COMM_WORLD
        # This is necessary because MPICommExecutor can only be created once per MPI process
        logger_debug(
            f"_start_mpi_pool: ENABLE_MULTI_DEVICE={ENABLE_MULTI_DEVICE}, self.comm={self.comm}\n",
            "grey")
        if ENABLE_MULTI_DEVICE:
            logger_debug(
                f"_start_mpi_pool: Checking if self.comm == mpi4py.MPI.COMM_WORLD: {self.comm == mpi4py.MPI.COMM_WORLD}\n",
                "grey")
        if ENABLE_MULTI_DEVICE and self.comm == mpi4py.MPI.COMM_WORLD:
            if MPINodeState._global_comm_executor is None:
                logger_debug("Creating global MPICommExecutor for COMM_WORLD\n",
                             "yellow")
                MPINodeState._global_comm_executor = MPICommExecutor(self.comm)
                MPINodeState._global_mpi_pool = MPINodeState._global_comm_executor.__enter__(
                )
            else:
                logger_debug("Reusing global MPICommExecutor for COMM_WORLD\n",
                             "yellow")
            self.mpi_pool = MPINodeState._global_mpi_pool
            self.owns_mpi_pool = False
        else:
            logger_debug(
                "_start_mpi_pool: Creating new MPICommExecutor (not COMM_WORLD or ENABLE_MULTI_DEVICE=False)\n",
                "grey")
            # For non-COMM_WORLD communicators, create a new executor
            comm_executor = MPICommExecutor(self.comm)
            self.mpi_pool = comm_executor.__enter__()
            self.owns_mpi_pool = True

    def __del__(self):
        self.shutdown_abort()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


class RemoteTask(NamedTuple):
    task: Callable[..., T]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    sync: bool = False  # if True, the result will be sent back to the client


class RemoteWorkerDeath(NamedTuple):
    """Worker-death notification forwarded by RemoteMpiCommSessionServer.

    Async (fire-and-forget) task submissions have no result channel, so a
    crashed worker would otherwise be invisible to the client and pending
    requests would block forever. The exception is carried as strings (not
    the exception object) because arbitrary exceptions may not pickle.
    """
    exc_type: str
    message: str

    @classmethod
    def from_exception(cls, e: BaseException) -> "RemoteWorkerDeath":
        return cls(exc_type=type(e).__name__, message=str(e))

    def to_exception(self) -> RuntimeError:
        return RuntimeError(
            f"Remote MPI worker died: {self.exc_type}: {self.message}")


class RemoteMpiCommSessionClient(MpiSession):
    """RemoteMpiCommSessionClient is a variant of MpiCommSession that is used to connect to a remote MPI pool.

    Note: This class uses a global singleton pattern because ZeroMQ PAIR sockets only support
    one connection at a time. Multiple LLM instances will reuse the same client connection.
    """
    _global_instance = None
    _global_instance_lock = threading.Lock()

    def __new__(cls, addr: str, hmac_key: bytes):
        # Implement singleton pattern to reuse the same client connection
        # for multiple LLM instances, since PAIR sockets only support one connection
        with cls._global_instance_lock:
            if cls._global_instance is None or cls._global_instance.addr != addr:
                logger_debug(
                    f"Creating new global RemoteMpiCommSessionClient for {addr}\n",
                    "yellow")
                instance = super().__new__(cls)
                cls._global_instance = instance
                instance._initialized = False
            else:
                logger_debug(
                    f"Reusing existing global RemoteMpiCommSessionClient for {addr}\n",
                    "yellow")
            return cls._global_instance

    def __init__(self, addr: str, hmac_key: bytes):
        # Only initialize once
        if self._initialized:
            return

        # FIXME: this is a hack to avoid circular import, resolve later
        from tensorrt_llm.executor.ipc import ZeroMqQueue
        self.addr = addr
        logger_debug(f"RemoteMpiCommSessionClient connecting to {addr}\n",
                     "yellow")
        self.queue = ZeroMqQueue((addr, hmac_key),
                                 is_server=False,
                                 socket_type=zmq.PAIR,
                                 use_hmac_encryption=True)
        self._is_shutdown = False
        # Non-error messages consumed by check_worker_error() while scanning
        # for RemoteWorkerDeath are buffered here for poll() (submit_sync).
        self._pending_responses: list = []
        self._initialized = True

    def submit(self,
               task: Callable[..., T],
               *args,
               sync: bool = False,
               **kwargs) -> list:
        """Submit a task to the remote MPI pool."""
        if self._is_shutdown:
            logger_debug("RemoteMpiCommSessionClient is already shut down\n",
                         "yellow")
            return []
        logger_debug(
            f"RemoteMpiCommSessionClient [rank{global_mpi_rank()}] sending task {task} to {self.addr}\n",
            "yellow")
        self.queue.put(RemoteTask(task, args, kwargs, sync=sync))
        return []

    SYNC_IDLE_INTERVAL = 8

    def submit_sync(self, task, *args, **kwargs) -> List[T]:
        """Submit a task to the remote MPI pool and wait for task completion."""
        self.submit(task, *args, sync=True, **kwargs)

        while not ((res := self.poll()) or self._is_shutdown):
            logger_debug(f"Waiting for task completion... {res}\n", "grey")
            time.sleep(self.SYNC_IDLE_INTERVAL)

        logger_debug(
            f"rank{global_mpi_rank()} RemoteMpiCommSessionClient.send_sync received results: {res}\n",
            "green")

        if not res:
            raise RuntimeError(
                "RemoteMpiCommSessionClient received unexpected response")
        return res

    def poll(self) -> bool:
        """Poll the queue for a response.

        Returns:
            True if a response is received, False otherwise.
        """
        if self._is_shutdown:
            return False
        if self._pending_responses:
            return self._pending_responses.pop(0)
        response = self.queue.poll(0.1)
        if response:
            return self.queue.get()  # should get a True if success
        return False

    def check_worker_error(self) -> Optional[BaseException]:
        """Non-blockingly fetch a worker-death notification, if any.

        RemoteMpiCommSessionServer forwards a RemoteWorkerDeath when an async
        (fire-and-forget) worker future fails -- the only error channel in
        this mode, since submit() returns no futures for the client to watch.
        Non-error messages encountered while scanning are buffered for poll().
        """
        if self._is_shutdown:
            return None
        try:
            while self.queue.poll(0):
                msg = self.queue.get()
                if isinstance(msg, RemoteWorkerDeath):
                    return msg.to_exception()
                self._pending_responses.append(msg)
        except Exception as e:
            logger_debug(f"check_worker_error poll failed: {e}\n", "grey")
        return None

    def abort(self):
        self.shutdown()

    def shutdown(self, wait=True):
        # NOTE: We do NOT close the queue or mark as shutdown for the singleton instance.
        # The RemoteMpiCommSessionClient is a global singleton that's reused across multiple
        # LLM instances. Marking it as shutdown would prevent subsequent LLM instances from
        # using it. The connection stays open for the entire lifetime of the mgmn setup.
        logger_debug(
            "RemoteMpiCommSessionClient.shutdown() called (no-op for singleton)\n",
            "grey")

    def shutdown_abort(self, grace: float = 60, reason=None):
        self.shutdown()


class RemoteMpiCommSessionServer():
    """RemoteMpiCommSessionServer is a variant of MpiCommSession that is used to create a remote MPI pool.
    """

    def __init__(self,
                 hmac_key: bytes,
                 n_workers: int = 0,
                 addr: str = 'tcp://127.0.0.1:*',
                 comm=None,
                 is_comm: bool = False):
        # FIXME: this is a hack to avoid circular import, resolve later
        from tensorrt_llm.executor.ipc import ZeroMqQueue
        self.addr = addr
        self.queue = ZeroMqQueue((addr, hmac_key),
                                 is_server=True,
                                 socket_type=zmq.PAIR,
                                 use_hmac_encryption=True)
        self.comm = comm
        self.results = []  # the results may arrive in any order

        if self.comm is not None:
            self.session = MpiCommSession(n_workers=self.comm.Get_size(),
                                          comm=self.comm)
        else:
            self.session = MpiCommSession(
                n_workers=n_workers) if is_comm else MpiPoolSession(
                    n_workers=n_workers)

    @staticmethod
    def task_wrapper(task: Callable[..., T], *args, **kwargs) -> T:
        logger_debug(
            f"MpiCommSession rank{mpi_rank()} with world_size {mpi_world_size()}\n",
            "green")
        logger_debug(
            f"MpiCommSession rank{mpi_rank()} start task [{task}] with args: {args} and kwargs: {kwargs}\n",
            "green")

        # wait for all ranks to start the task
        mpi_barrier()

        try:
            return task(*args, **kwargs)
        except Exception as e:
            print_colored(
                f"MpiCommSession rank{mpi_rank()} task [{task}] failed with exception: {e}\n",
                "red")
            traceback.print_exc()
            raise e
        finally:
            logger_debug(
                f"MpiCommSession rank{mpi_rank()} task [{task}] finished\n",
                "green")
            mpi_barrier()

    def serve(self):
        logger_debug(f"RemoteMpiCommSessionServer listening on {self.addr}\n",
                     "yellow")
        pending_futures = []
        while True:
            # Wait for any pending futures from previous tasks to complete
            # This ensures all ranks are ready before accepting the next task
            if pending_futures:
                logger_debug(
                    f"RemoteMpiCommSessionServer waiting for {len(pending_futures)} pending futures to complete\n",
                    "grey")
                n_failed = 0
                first_exc = None
                # Use as_completed so that failures are logged as soon as
                # they occur rather than blocking behind a stuck future.
                for future in as_completed(pending_futures):
                    try:
                        future.result()  # Wait for completion
                    except Exception as e:
                        n_failed += 1
                        if first_exc is None:
                            first_exc = e
                        print_colored(
                            f"RemoteMpiCommSessionServer: MPI worker future "
                            f"failed: {type(e).__name__}: {e}\n", "red")
                        if n_failed == len(pending_futures):
                            # All workers failed — no point waiting further.
                            break
                if n_failed:
                    logger.error(
                        f"RemoteMpiCommSessionServer: {n_failed}/"
                        f"{len(pending_futures)} MPI worker(s) failed. "
                        f"First error: {first_exc}")
                pending_futures.clear()
                logger_debug(
                    "RemoteMpiCommSessionServer all pending futures completed\n",
                    "grey")

            message: Optional[RemoteTask] = self.queue.get()
            if message is None:
                logger_debug(
                    f"RemoteMpiCommSessionServer [rank{global_mpi_rank()}] received shutdown signal\n",
                    "green")
                self.session.shutdown_abort()
                break
            else:
                logger_debug(
                    f"RemoteMpiCommSessionServer [rank{global_mpi_rank()}] received task [{message.task}] from {self.addr}\n",
                    "green")
                futures = self.session.submit(
                    RemoteMpiCommSessionServer.task_wrapper, message.task,
                    *message.args, **message.kwargs)
                self.num_results = self.session.n_workers
                assert len(futures) == self.num_results == mpi_world_size()
                # Store futures to wait for them before the next task
                pending_futures = list(futures)
                for future in futures:
                    if message.sync:
                        future.add_done_callback(self.mpi_future_callback)
                    else:
                        # Fire-and-forget tasks have no result channel, but a
                        # crashed worker must still reach the client (the
                        # client-side session has no futures to watch); see
                        # RemoteWorkerDeath.
                        future.add_done_callback(self.mpi_async_error_callback)

    def mpi_async_error_callback(self, future):
        """Forward a worker exception to the client for async tasks.

        Runs on the executor's callback thread, like the existing sync-path
        mpi_future_callback (same pre-existing cross-thread ZMQ-put pattern).
        Best-effort: the socket may already be closed at shutdown.
        """
        if future.cancelled():
            return
        exc = future.exception()
        if exc is None:
            return
        print_colored(
            f"RemoteMpiCommSessionServer: async MPI worker failed, forwarding "
            f"to client: {type(exc).__name__}: {exc}\n", "red")
        try:
            self.queue.put(RemoteWorkerDeath.from_exception(exc))
        except Exception as e:
            logger_debug(f"Failed to forward worker death to client: {e}\n",
                         "red")

    def mpi_future_callback(self, future):
        logger_debug(f"rank{global_mpi_rank()} got future: {future}\n", "red")
        if future.exception() is not None:
            logger_debug(
                f"mpi_future got exception: {future.exception()}, quitting\n",
                "red")
            self.queue.put(future.exception())
            return

        result = future.result()
        self.results.append(result)
        logger_debug(
            f"RemoteMpiCommSessionServer working status: {len(self.results)}/{self.num_results}\n",
            "grey")
        if len(self.results) == self.num_results:
            logger_debug(
                "RemoteMpiCommSessionServer received all results, sending to client\n",
                "green")
            try:
                self.queue.put_noblock(self.results, retry=2)
            except zmq.ZMQError as e:
                # The client could be shutdown first.
                if e.errno == zmq.EAGAIN:
                    pass
                else:
                    raise e

            logger_debug("RemoteMpiCommSessionServer sent results to client\n",
                         "green")
            self.results.clear()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def find_free_ipc_addr() -> str:
    import os
    import tempfile
    import uuid
    return f'ipc://{os.path.join(tempfile.gettempdir(), "rpc_" + str(uuid.uuid4()))}'


def get_mpi_world_size() -> int:
    # avoid cyclic import
    from ..executor.utils import get_spawn_proxy_process_env

    # If the proxy process is spawned, the MPI-related env will be cleaned in the proxy process, thus we made another env for the mpi_world_size
    if get_spawn_proxy_process_env():
        return int(os.getenv("tllm_mpi_size") or 1)
    else:
        return mpi_world_size()


def split_mpi_env(mpi_env_keys: List[str] | None = None) -> Tuple[dict, dict]:
    """Splits the environment variables into MPI-related and non-MPI-related dictionaries.

    Args:
        mpi_env_keys: Additional environment variables to be considered as MPI-related.

    Returns:
        Tuple[dict, dict]: (non_mpi_env, mpi_env)
            - non_mpi_env: Environment dictionary without MPI-related variables
            - mpi_env: Environment dictionary containing only MPI-related variables
    """
    current_env = os.environ.copy()

    # Identify MPI-related variables
    mpi_vars = set(
        itertools.chain([
            var for var in current_env if var.startswith((
                'MPI_',
                'OMPI_',
                'PMIX_',
                'PMI_',
                'OMPI_',
                'PMIX_',
                'PMI_',
                'SLURM_',
                'MPI_',
                'UCX_',
                'I_MPI_',
                'HYDRA_',
                'KMP_',
                'MPICH_',
                'MV2_',
                'CRAY_',
            ))
        ], mpi_env_keys or []))

    # Split into two dictionaries
    non_mpi_env = {k: v for k, v in current_env.items() if k not in mpi_vars}
    mpi_env = {k: v for k, v in current_env.items() if k in mpi_vars}

    return non_mpi_env, mpi_env
