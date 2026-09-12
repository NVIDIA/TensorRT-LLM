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
"""Launch bootstrap for VisualGen worker ranks.

Resolves how the current process was launched into a :class:`LaunchPlan`
(topology, torch rendezvous, IPC host) and brings the worker ranks up behind a
uniform :class:`WorkerHandle`, so neither ``VisualGen`` nor
``DiffusionRemoteClient`` branches on the launch environment.
"""

import asyncio
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch.multiprocessing as mp

from tensorrt_llm.executor.utils import create_mpi_comm_session, get_spawn_proxy_process_env
from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.args import VisualGenArgs

# Grace period (seconds) for a worker process or thread to exit before it is
# escalated to SIGTERM and then SIGKILL.
WORKER_TIMEOUT = 2.0
# Bound on how long shutdown waits for an in-flight ``Process.start()``.
WORKER_SPAWN_SHUTDOWN_TIMEOUT = 5.0

# Module-local seams keep lifecycle tests from monkeypatching process-wide
# module objects used by unrelated threads and tests.
_Event = threading.Event
_Thread = threading.Thread
_get_mp_context = mp.get_context
_get_process_id = os.getpid

# Default ``torch.distributed`` process-group timeout (seconds) for MGMN
# workers. Paired with TORCH_NCCL_ASYNC_ERROR_HANDLING it bounds how long a
# surviving rank blocks in a collective whose peers died, while staying
# generous enough for slow model-load barriers. Overridable per run via
# TLLM_VG_MGMN_PG_TIMEOUT_SEC.
MGMN_PG_TIMEOUT_SEC = 1800

# Minimum interval (seconds) between MGMN worker-death polls.
MGMN_ERROR_POLL_INTERVAL = 1.0


class LaunchMode(str, Enum):
    """How the current process's worker ranks are brought up."""

    SPAWN = "spawn"
    EXTERNAL = "external"
    MGMN = "mgmn"


@dataclass(frozen=True)
class LaunchPlan:
    """Resolved launch topology for one VisualGen run.

    ``rank``/``local_rank`` describe the process that resolved the plan. In
    SPAWN and MGMN modes that process is the client, which is rank 0 by
    construction; in EXTERNAL mode it is whichever launcher rank is running.

    ``master_addr``/``master_port`` are ``None`` in MGMN mode only: the
    launcher strips the client's MPI environment, so the rendezvous is
    resolved by the worker ranks themselves and agreed over MPI.
    """

    mode: LaunchMode
    world_size: int
    rank: int
    local_rank: int
    ipc_host: str
    master_addr: Optional[str] = None
    master_port: Optional[int] = None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_ip_address() -> str:
    """Get local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _detect_external_launch() -> Optional[tuple]:
    """Detect whether the process was launched by an external distributed launcher.

    Checks for torchrun (``RANK`` + ``WORLD_SIZE``) and then SLURM
    (``SLURM_PROCID`` + ``SLURM_NTASKS``).  Returns a
    ``(rank, local_rank, world_size, master_addr, master_port)`` tuple when a
    multi-process launcher is detected (world_size > 1), or ``None`` for
    single-process / single-node ``mp.Process`` mode.
    """
    # torchrun / torchelastic sets RANK and WORLD_SIZE
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            master_addr = os.environ.get("MASTER_ADDR")
            if master_addr is None:
                raise RuntimeError(
                    "MASTER_ADDR must be set for multi-node torchrun runs. "
                    "Add --master-addr=<node0-ip> to your torchrun command, or set "
                    "MASTER_ADDR in the environment before launching."
                )
            master_port = int(os.environ.get("MASTER_PORT", 29500))
            return rank, local_rank, world_size, master_addr, master_port

    # SLURM: srun --ntasks-per-node=GPUS_PER_NODE sets SLURM_PROCID / SLURM_NTASKS
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        if world_size > 1:
            local_rank = int(os.environ.get("SLURM_LOCALID", rank))
            master_addr = os.environ.get("MASTER_ADDR")
            if master_addr is None:
                raise RuntimeError(
                    "MASTER_ADDR must be set for multi-node SLURM runs. "
                    "Add to your sbatch script:\n"
                    "  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)"
                )
            master_port = int(os.environ.get("MASTER_PORT", 29500))
            return rank, local_rank, world_size, master_addr, master_port

    return None


def resolve_launch_plan(n_workers: int) -> LaunchPlan:
    """Resolve the launch mode and topology for a run of ``n_workers`` ranks.

    Raises ``ValueError`` when a launcher's world size disagrees with
    ``n_workers``. Resolve once per run: in SPAWN mode this picks the
    rendezvous port, so two calls yield two different plans.
    """
    # MGMN is checked first: trtllm-llmapi-launch strips SLURM_*/OMPI_* from
    # the wrapped program but not a stale RANK/WORLD_SIZE pair, which
    # _detect_external_launch() would misread as a torchrun launch.
    if get_spawn_proxy_process_env():
        mpi_world_size = get_mpi_world_size()
        if mpi_world_size != n_workers:
            raise ValueError(
                f"MGMN launcher world size ({mpi_world_size}) does not match "
                f"n_workers ({n_workers}). Launch exactly n_workers MPI ranks, "
                f"e.g. `mpirun -n {n_workers} trtllm-llmapi-launch <program>` or "
                f"`srun --mpi=pmix --ntasks={n_workers} ... "
                f"trtllm-llmapi-launch <program>`."
            )
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            logger.info(
                "VisualGen: MGMN launcher detected; ignoring stale "
                "RANK/WORLD_SIZE environment variables"
            )
        return LaunchPlan(
            mode=LaunchMode.MGMN,
            world_size=n_workers,
            rank=0,
            local_rank=0,
            ipc_host=get_ip_address(),
        )

    ext = _detect_external_launch()
    if ext is not None:
        rank, local_rank, world_size, master_addr, master_port = ext
        if world_size != n_workers:
            raise ValueError(
                f"Launcher world_size ({world_size}) does not match "
                f"n_workers ({n_workers}). "
                "Launch exactly n_workers tasks."
            )
        return LaunchPlan(
            mode=LaunchMode.EXTERNAL,
            world_size=n_workers,
            rank=rank,
            local_rank=local_rank,
            ipc_host=master_addr,
            master_addr=master_addr,
            master_port=master_port,
        )

    return LaunchPlan(
        mode=LaunchMode.SPAWN,
        world_size=n_workers,
        rank=0,
        local_rank=0,
        ipc_host=get_ip_address(),
        master_addr="127.0.0.1",
        master_port=find_free_port(),
    )


def is_external_worker_rank() -> bool:
    """Whether this process is a non-leader rank of an external (SPMD) launch.

    Such a rank must become a worker rather than bind the HTTP port, or every
    rank on a multi-GPU node races the same port. Always False under the MGMN
    launcher, which runs user code on rank 0 only — checked first because the
    launcher leaves a stale RANK/WORLD_SIZE pair behind that the external-launch
    detector would misread.

    Answers the question without resolving a full plan, which must happen only
    once per run.
    """
    if get_spawn_proxy_process_env():
        return False
    ext = _detect_external_launch()
    return ext is not None and ext[0] != 0


def run_worker_and_exit_if_not_leader(plan: LaunchPlan, args: "VisualGenArgs") -> None:
    """Turn a non-leader external-launch rank into a worker; never returns for it.

    Only EXTERNAL mode runs user code on every rank, so only there can this
    process be a non-leader. Returns immediately in every other case.
    """
    if plan.mode is not LaunchMode.EXTERNAL or plan.rank == 0:
        return

    from tensorrt_llm._torch.visual_gen.executor import run_diffusion_worker

    logger.info(
        f"VisualGen: rank {plan.rank}/{plan.world_size}, local_rank {plan.local_rank} — "
        "starting as worker (external launch mode)"
    )
    run_diffusion_worker(
        rank=plan.rank,
        world_size=plan.world_size,
        master_addr=plan.master_addr,
        master_port=plan.master_port,
        # unused: non-zero ranks receive requests via dist.broadcast_object_list
        request_queue_addr=None,
        response_queue_addr=None,  # unused: only rank 0 sends responses over ZMQ
        visual_gen_args=args,
        req_hmac_key=None,
        resp_hmac_key=None,
        local_rank=plan.local_rank,
    )
    raise SystemExit(0)


def _reap_worker_process(process: mp.Process) -> bool:
    worker_pid = process.pid
    if worker_pid is None:
        return False
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        logger.warning(f"DiffusionClient: Terminating worker {worker_pid} with SIGTERM")
        process.terminate()
        process.join(timeout=WORKER_TIMEOUT)
        if process.is_alive():
            logger.warning(f"DiffusionClient: Force killing worker {worker_pid} with SIGKILL")
            process.kill()
            process.join(timeout=WORKER_TIMEOUT)
    return True


class _WorkerProcessSpawner:
    """Spawn workers off the signal-handling thread and reap late starts."""

    def __init__(self, processes: List[mp.Process]):
        self._processes = processes
        self._spawn_cancelled = _Event()
        self._spawn_complete = _Event()
        self._thread_entered = _Event()
        self._reap_locks = {id(process): threading.Lock() for process in processes}
        self._reaped_process_ids: set = set()
        self._spawn_error: Optional[BaseException] = None
        self._thread = _Thread(
            target=self._run,
            name="visualgen-worker-process-spawner",
            daemon=True,
        )

    def start(self) -> None:
        try:
            self._thread.start()
        except BaseException as e:
            # A main-thread signal can interrupt Thread.start() after the OS
            # thread exists but before start() returns. Give that thread a
            # chance to publish its entry before declaring the batch inert.
            self._thread_entered.wait(timeout=WORKER_SPAWN_SHUTDOWN_TIMEOUT)
            if not self._thread_entered.is_set():
                self._spawn_error = e
                self._spawn_complete.set()
            raise

    def cancel_spawn(self) -> None:
        self._spawn_cancelled.set()

    def wait_for_spawn(self, timeout: Optional[float] = None, *, raise_error: bool = True) -> bool:
        if not self._spawn_complete.wait(timeout=timeout):
            if raise_error:
                raise TimeoutError(
                    f"VisualGen worker process spawn did not complete within {timeout:.0f}s"
                )
            return False
        if raise_error and self._spawn_error is not None:
            raise self._spawn_error
        return True

    def reap_started_processes(self) -> None:
        for process in self._processes:
            process_id = id(process)
            with self._reap_locks[process_id]:
                if process_id in self._reaped_process_ids:
                    continue
                if _reap_worker_process(process):
                    self._reaped_process_ids.add(process_id)

    def _run(self) -> None:
        try:
            self._thread_entered.set()
            for process in self._processes:
                if self._spawn_cancelled.is_set():
                    break
                process.start()
        except BaseException as e:
            self._spawn_error = e
            logger.error(f"VisualGen worker process spawn failed: {e}")
        finally:
            self._thread_entered.set()
            self._spawn_complete.set()
            # Process.start() may publish its pid after shutdown's bounded
            # wait. Reap that late worker before this short-lived thread exits.
            if self._spawn_cancelled.is_set():
                self.reap_started_processes()


def run_diffusion_worker_mgmn(
    world_size: int,
    request_queue_addr: str,
    response_queue_addr: str,
    visual_gen_args: "VisualGenArgs",
    req_hmac_key: Optional[bytes] = None,
    resp_hmac_key: Optional[bytes] = None,
    log_level: str = "info",
):
    """Entry point for MGMN (``trtllm-llmapi-launch``) workers.

    Dispatched once via ``MpiSession.submit`` and executed on every
    pre-spawned MPI rank: ranks >= 1 inside the launcher's persistent
    ``mgmn_worker_node`` processes, rank 0 inside the MGMN leader process on
    the client's node. Rank/topology discovery uses raw mpi4py rather than
    the ``_utils`` helpers, which honor ``TLLM_DISABLE_MPI`` and would
    silently return wrong values if that env leaked into the persistent rank
    process.
    """
    from tensorrt_llm._torch.visual_gen.executor import run_diffusion_worker

    # Set log level before any other work so the MGMN preamble is visible.
    logger.set_level(log_level)

    # Never trust inherited state: a leaked TLLM_DISABLE_MPI=1 (from the user
    # environment or a previous task in this persistent rank process) would
    # corrupt rank resolution for any MPI-gated helper sharing the process.
    os.environ.pop("TLLM_DISABLE_MPI", None)

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != world_size:
        raise RuntimeError(
            f"MPI world size ({comm.Get_size()}) does not match "
            f"parallel_config.n_workers ({world_size}). Launch exactly "
            f"n_workers MPI ranks under trtllm-llmapi-launch."
        )
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()

    # Deterministic, launcher-agnostic node id consumed by
    # mapping._get_host_id via GROUP_RANK: the first-occurrence index of this
    # rank's processor name among the unique names. Identical on every rank
    # because the allgather result is.
    names = comm.allgather(MPI.Get_processor_name())
    host_id = sorted(set(names), key=names.index).index(names[rank])
    os.environ["GROUP_RANK"] = str(host_id)

    # torch rendezvous: rank-0-authoritative, agreed over an UNCONDITIONAL
    # bcast. Every rank participates regardless of its own env, so a
    # MASTER_ADDR/MASTER_PORT present on only some ranks can neither diverge
    # the collective nor fork the value; rank 0's env override wins. The
    # address is by construction resolved on (and reachable as) the rank-0
    # worker's host, and the port is bound-tested on that same host.
    root_addr = (os.environ.get("MASTER_ADDR") or socket.getfqdn()) if rank == 0 else None
    master_addr = comm.bcast(root_addr, root=0)
    root_port = (int(os.environ.get("MASTER_PORT") or find_free_port())) if rank == 0 else None
    master_port = comm.bcast(root_port, root=0)

    # Surviving ranks must not block forever in collectives whose peers died:
    # abort NCCL collectives on async errors, paired with the bounded
    # process-group timeout passed below.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    pg_timeout = timedelta(
        seconds=int(os.environ.get("TLLM_VG_MGMN_PG_TIMEOUT_SEC", str(MGMN_PG_TIMEOUT_SEC)))
    )

    logger.info(
        f"MGMN worker rank {rank}/{world_size} (local_rank {local_rank}, "
        f"host_id {host_id}): rendezvous {master_addr}:{master_port}"
    )

    run_diffusion_worker(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        request_queue_addr=request_queue_addr if rank == 0 else None,
        response_queue_addr=response_queue_addr if rank == 0 else None,
        visual_gen_args=visual_gen_args,
        log_level=log_level,
        req_hmac_key=req_hmac_key if rank == 0 else None,
        resp_hmac_key=resp_hmac_key if rank == 0 else None,
        local_rank=local_rank,
        disable_mpi_env=False,
        pg_timeout=pg_timeout,
    )


class WorkerHandle(ABC):
    """The launched worker ranks of one run, as the client observes them.

    Every method is abstract even where a mode has nothing to do, so that
    "this mode needs no cleanup here" is a decision each handle writes down
    rather than one it inherits by accident.
    """

    #: True when the client can neither kill these workers nor outlive them,
    #: so the only way to stop them is to send them the shutdown signal.
    must_be_signaled_to_exit = False

    @abstractmethod
    def liveness_failure(self) -> Optional[str]:
        """A message naming a worker that exited, or ``None``."""

    @abstractmethod
    def abort(self) -> None:
        """Kill and reap the group after a terminal failure."""

    @abstractmethod
    def begin_shutdown(self) -> None:
        """Stop bringing further workers up. Bounded; runs before the client's
        coordinator thread is joined."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release the workers. Must be safe to call more than once."""


class _SpawnWorkers(WorkerHandle):
    def __init__(self, processes: List[mp.Process], spawner: _WorkerProcessSpawner):
        self._processes = processes
        self._spawner = spawner

    def liveness_failure(self) -> Optional[str]:
        dead = [(p.pid, p.exitcode) for p in self._processes if not p.is_alive()]
        if not dead:
            return None
        statuses = ", ".join(f"pid={pid}, exitcode={code}" for pid, code in dead)
        return f"DiffusionClient: local worker processes exited: {statuses}"

    def abort(self) -> None:
        # A surviving rank may be blocked in a collective that can never
        # complete after another rank exits, so containment is immediate.
        for process in self._processes:
            if process.is_alive():
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        self._spawner.reap_started_processes()

    def begin_shutdown(self) -> None:
        self._spawner.cancel_spawn()
        # Process.start() runs on the spawner thread and cannot be interrupted
        # by Python's main-thread signal handlers. Give the current start a
        # short bounded chance to finish, then reap every registered process
        # whose pid has been published.
        if not self._spawner.wait_for_spawn(
            timeout=WORKER_SPAWN_SHUTDOWN_TIMEOUT, raise_error=False
        ):
            logger.error(
                "VisualGen worker spawn batch did not complete within "
                f"{WORKER_SPAWN_SHUTDOWN_TIMEOUT:.0f}s during shutdown; "
                "continuing to reap every worker with a published pid"
            )

    def shutdown(self) -> None:
        self._spawner.reap_started_processes()


class _ExternalRank0Worker(WorkerHandle):
    def __init__(self, thread: threading.Thread):
        self._thread = thread

    def liveness_failure(self) -> Optional[str]:
        if self._thread.is_alive():
            return None
        return "DiffusionClient: external-launch worker thread exited"

    def abort(self) -> None:
        # This worker is a daemon thread of the client process and goes with
        # it; the external launcher owns the sibling ranks.
        return

    def begin_shutdown(self) -> None:
        # The thread is already running: there is no spawn batch to cancel.
        return

    def shutdown(self) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=WORKER_TIMEOUT)


class _MgmnWorkers(WorkerHandle):
    # The workers live in launcher-owned processes with no handle to kill, and
    # the rank-0 worker blocks in a request recv that has no timeout.
    must_be_signaled_to_exit = True

    def __init__(self, session: Any):
        self.session = session
        self._failure: Optional[str] = None
        self._last_poll = 0.0

    def liveness_failure(self) -> Optional[str]:
        """Poll the MGMN session for a worker death forwarded by the leader.

        ``RemoteMpiCommSessionClient.submit`` returns no futures, so worker
        exceptions reach the client only as ``RemoteWorkerDeath`` messages on
        the session's control socket. The client polls this on every pass of
        its coordinator loop, so the socket read is rate-limited. Sticky: once
        observed, the failure stays.
        """
        if self._failure is not None:
            return self._failure
        now = time.monotonic()
        if now - self._last_poll < MGMN_ERROR_POLL_INTERVAL:
            return None
        self._last_poll = now
        check = getattr(self.session, "check_worker_error", None)
        if check is None:
            return None
        try:
            error = check()
        except Exception as exc:  # noqa: BLE001 - the watchdog must not die
            logger.debug(f"DiffusionClient: check_worker_error failed (ignored): {exc!r}")
            return None
        if error is not None:
            self._failure = f"DiffusionClient: MGMN worker death reported: {error!r}"
        return self._failure

    def abort(self) -> None:
        # These workers have no handle to kill, which is why the client signals
        # them instead -- see must_be_signaled_to_exit.
        return

    def begin_shutdown(self) -> None:
        # submit() is fire-and-forget: there is no spawn batch to cancel.
        return

    def shutdown(self) -> None:
        # The session is owned by the launcher and shared process-wide;
        # RemoteMpiCommSessionClient.shutdown() is a documented no-op that
        # never touches the session socket, so calling it off the event-loop
        # thread that owns all session I/O is safe.
        self.session.shutdown()


def start_workers(
    plan: LaunchPlan,
    *,
    worker_fn: Callable,
    worker_kwargs: Dict[str, Any],
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> WorkerHandle:
    """Bring up the worker ranks for ``plan``.

    ``worker_kwargs`` carries the mode-independent worker arguments (IPC
    addresses, args, HMAC keys, log level); this function adds the topology.

    ``event_loop`` is the client's coordinator loop, required in MGMN mode:
    the launcher session it dispatches on owns a socket that only that thread
    may touch.
    """
    if plan.mode is LaunchMode.MGMN:
        if event_loop is None:
            raise ValueError("start_workers() requires event_loop in MGMN mode")
        return _start_mgmn_workers(plan, worker_kwargs, event_loop)
    if plan.mode is LaunchMode.SPAWN:
        return _start_spawn_workers(plan, worker_fn, worker_kwargs)
    return _start_external_rank0_worker(plan, worker_fn, worker_kwargs)


def _start_spawn_workers(
    plan: LaunchPlan, worker_fn: Callable, worker_kwargs: Dict[str, Any]
) -> WorkerHandle:
    logger.info(f"DiffusionClient: Launching {plan.world_size} workers")
    ctx = _get_mp_context("spawn")
    parent_pid = _get_process_id()
    processes = [
        ctx.Process(
            target=worker_fn,
            kwargs={
                **worker_kwargs,
                "rank": rank,
                "world_size": plan.world_size,
                "master_addr": plan.master_addr,
                "master_port": plan.master_port,
                "local_rank": rank,
                "parent_pid": parent_pid,
            },
        )
        for rank in range(plan.world_size)
    ]

    # Process.start() can block during spawn bootstrap. Run the finite spawn
    # batch away from Python's signal-handling thread so shutdown can remain
    # bounded. That thread exits as soon as spawning finishes; worker lifetime
    # follows the coordinator process through the native watchdog instead.
    spawner = _WorkerProcessSpawner(processes)
    handle = _SpawnWorkers(processes, spawner)
    spawner.start()
    spawner.wait_for_spawn()
    return handle


def _start_external_rank0_worker(
    plan: LaunchPlan, worker_fn: Callable, worker_kwargs: Dict[str, Any]
) -> WorkerHandle:
    # Only rank 0 reaches here; ranks 1..N-1 already became workers and their
    # ZMQ clients connect once these server sockets bind. The external launcher
    # owns sibling-rank cleanup: torchrun terminates its worker group, and srun
    # deployments must enable KillOnBadExit.
    thread = _Thread(
        target=worker_fn,
        kwargs={
            **worker_kwargs,
            "rank": plan.rank,
            "world_size": plan.world_size,
            "master_addr": plan.master_addr,
            "master_port": plan.master_port,
            "local_rank": plan.local_rank,
            "in_client_process": True,
        },
        daemon=True,
    )
    thread.start()
    return _ExternalRank0Worker(thread)


def _start_mgmn_workers(
    plan: LaunchPlan,
    worker_kwargs: Dict[str, Any],
    event_loop: asyncio.AbstractEventLoop,
) -> WorkerHandle:
    logger.info(f"DiffusionClient: Dispatching {plan.world_size} MGMN workers")

    handle: Dict[str, WorkerHandle] = {}

    # The session wraps a single ZMQ PAIR socket that is not thread-safe, so
    # it is created and used only on the background event-loop thread — this
    # dispatch and the later liveness_failure() polls alike. submit() is
    # fire-and-forget (returns no futures); readiness is signaled by the
    # workers' READY handshake and death by a forwarded RemoteWorkerDeath.
    async def _dispatch():
        session = create_mpi_comm_session(plan.world_size)
        session.submit(
            run_diffusion_worker_mgmn,
            **worker_kwargs,
            world_size=plan.world_size,
        )
        handle["h"] = _MgmnWorkers(session)

    asyncio.run_coroutine_threadsafe(_dispatch(), event_loop).result()
    return handle["h"]
