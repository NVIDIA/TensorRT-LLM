# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent rank-process pool for visual_gen multi-GPU (``torch.distributed``) tests.

The tests in this package run their logic as ``fn(rank, world_size)`` on
``world_size`` processes joined by ``dist.init_process_group``. Spawning those
processes per test pays a full ``tensorrt_llm`` import per rank (~1.5 min on
CI) for a test body that runs in seconds. This module keeps one pool of rank
processes per ``(world_size, backend)`` alive across tests and dispatches test
functions to it; ``fn`` runs on every rank exactly as it would under
``mp.spawn`` and is subject to the same pickling rule (module-level function).

Failure contract -- the part that is not just a work queue:

- A rank whose test raises reports the traceback. The parent does NOT wait for
  the other ranks: they may be blocked in a collective waiting for the rank
  that bailed and would otherwise sit there until the NCCL timeout. After a
  short grace period for stragglers' reports, the whole pool is killed and the
  failure is raised with every traceback that arrived; ranks that never
  reported are listed as killed, not as failures of their own.
- A rank that dies without reporting (segfault, CUDA fault, OOM-kill) or a test
  that outlives its deadline likewise kills the pool.
- Every failure retires the pool; the next test spawns a fresh one. A failure
  can cost time, never a cascade of false failures in later tests.
- After a kill, the parent checks that the ranks are gone and, for NCCL pools,
  that no rank pid still holds GPU memory; a rank the kernel will not reap is
  reported instead of letting the next test die of an unexplained OOM.
- Ranks die with the parent: ``PR_SET_PDEATHSIG`` at rank start (kernel
  delivered SIGKILL, works even when the rank is wedged in a collective) and,
  once ``tensorrt_llm`` is loaded in the rank (``preload`` or the first test
  module import), the library's native coordinator watchdog on top.

Between tests each rank resets the process state a test can leave behind:
``os.environ`` back to its post-spawn contents, RNG seeds, the logger's
log-once memory, Dynamo caches, the visual_gen mesh singletons, every
non-default process group and torch's mesh cache, the CUDA allocator cache, and
finally a barrier so a collective leaked by one test surfaces in that test's
deadline rather than in the next test.

A pool is also retired when the parent's env / ``sys.path`` no longer match
what the ranks froze at spawn. How long a pool otherwise lives is
``TRTLLM_TEST_VISUALGEN_POOL_LIFETIME``: ``session`` (default: the whole
pytest run), ``module`` (the package conftest shuts pools down at each test
file's teardown), or ``test`` (fresh ranks for every test, i.e. the old
spawn-per-test behaviour, for state-bleed debugging and CI A/B timing).
``run_distributed(..., reuse=False)`` gives one test fresh ranks and drains the
cache first. Under pytest-xdist workers the lifetime is forced to ``test``,
since parallel pytest processes would multiply live pools on the same GPUs.

Parent/rank plumbing is ``mp.Pipe`` connections, not ``mp.Queue``: queues run
a feeder thread in the parent for the life of the queue, which pytest-threadleak
charges to whichever test left a pool cached.
"""

import atexit
import importlib
import multiprocessing as mp
import multiprocessing.util  # registers multiprocessing's atexit hook; see atexit.register below
import os
import pickle
import random
import signal
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import wait as _wait_connections
from typing import Any, Callable

from test_common._session_utils import _spawn_snapshot

# Rank -> parent report tags.
_READY = "ready"
_OK = "ok"
_ERROR = "error"

# The one timing constant: how long the parent waits for a rank to react on
# its own before escalating. Used for: the other ranks' reports after the
# first failure, exit after the shutdown sentinel, reaping after SIGKILL, and
# the driver releasing a killed rank's GPU memory. The pool's margin before
# pytest-timeout is three of these (reports, reap, GPU release).
_GRACE_SECONDS = 10.0
_RNG_SEED_BETWEEN_TESTS = 0
_LIFETIMES = ("session", "module", "test")

# Mutable module state.
# Cached live pools keyed by (world_size, backend, preload).
_POOLS: dict[tuple[int, str, tuple], "_Pool"] = {}
# The current test's effective pytest-timeout, set per test by the conftest.
_pytest_timeout: float | None = None
# The xdist notice is printed once per session.
_xdist_noted = False


def lifetime() -> str:
    """``session``, ``module`` or ``test``; see the module docstring."""
    global _xdist_noted
    if os.environ.get("PYTEST_XDIST_WORKER"):
        if not _xdist_noted:
            _xdist_noted = True
            _log("pytest-xdist worker: pool lifetime forced to one test")
        return "test"
    value = os.environ.get("TRTLLM_TEST_VISUALGEN_POOL_LIFETIME", "session").lower()
    if value not in _LIFETIMES:
        raise ValueError(
            f"TRTLLM_TEST_VISUALGEN_POOL_LIFETIME={value!r}; expected one of {_LIFETIMES}"
        )
    return value


def set_pytest_timeout(seconds: float | None) -> None:
    """Record the current test's effective pytest-timeout (conftest calls this).

    The pool's deadline for a test (spawn included) is that value minus the
    time the kill path needs, so the pool fails one test before pytest-timeout
    kills the whole pytest process. Without pytest-timeout there is no
    deadline, as with the mp.spawn harness before.
    """
    global _pytest_timeout
    _pytest_timeout = seconds or None


def _default_timeout() -> float | None:
    if _pytest_timeout is None:
        return None
    margin = 3 * _GRACE_SECONDS
    # A pytest-timeout too short to leave a margin: use it whole and let
    # pytest-timeout win the race, which is the test author's choice.
    return _pytest_timeout - margin if _pytest_timeout > margin else _pytest_timeout


def _log(msg: str) -> None:
    print(f"[visualgen-pool] {msg}", flush=True)


class WorkerPoolError(RuntimeError):
    """A test run on the pool failed, timed out, or lost a rank."""


# ---------------------------------------------------------------------------
# Rank process side
# ---------------------------------------------------------------------------


def _bind_to_parent(parent_pid: int) -> None:
    """Make the kernel SIGKILL this rank when its parent dies.

    ``PR_SET_PDEATHSIG`` is tied to the thread that created the child; the
    ranks are started from the parent's main thread, so that is the parent's
    lifetime. The ``getppid`` check closes the window where the parent died
    before the flag was set (compare with the recorded pid, not with 1: under
    a subreaper an orphan's parent is the subreaper).
    """
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)  # 1 == PR_SET_PDEATHSIG
        except OSError:
            pass
    if os.getppid() != parent_pid:
        os._exit(1)


def _start_native_watchdog(parent_pid: int) -> bool:
    """Add the library's coordinator watchdog once ``tensorrt_llm`` is loaded.

    Only called when the package is already in ``sys.modules`` (the test
    module imported it), so this import is cheap. Returns True once started.
    """
    try:
        from tensorrt_llm.bindings.internal import start_coordinator_watchdog

        warning = start_coordinator_watchdog(parent_pid)
    except Exception as e:
        print(f"[visualgen-pool] rank could not start the native watchdog: {e}", flush=True)
        return True  # do not retry every task
    if warning is not None:
        print(f"[visualgen-pool] native watchdog: {warning}", flush=True)
    return True


def _destroy_extra_process_groups() -> None:
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    default = c10d._get_default_group()
    for pg in list(c10d._world.pg_map):
        if pg is not default:
            dist.destroy_process_group(pg)
    mesh_mod = sys.modules.get("torch.distributed.device_mesh")
    resources = getattr(mesh_mod, "_mesh_resources", None)
    if resources is not None:
        # Every attribute of the mesh registry is a cache keyed by meshes whose
        # groups were just destroyed.
        for value in getattr(resources, "__dict__", {}).values():
            if isinstance(value, (dict, list, set)):
                value.clear()


def _reset_rank_state(env_snapshot: dict) -> None:
    import torch
    import torch.distributed as dist

    os.environ.clear()
    os.environ.update(env_snapshot)

    random.seed(_RNG_SEED_BETWEEN_TESTS)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(_RNG_SEED_BETWEEN_TESTS)
    torch.manual_seed(_RNG_SEED_BETWEEN_TESTS)

    # Only touch singletons of modules the test actually loaded; importing
    # them here would pull the whole library into a rank that never used it.
    logger_mod = sys.modules.get("tensorrt_llm.logger")
    if logger_mod is not None:
        logger_mod.logger._appeared_keys.clear()
    dynamo = sys.modules.get("torch._dynamo")
    if dynamo is not None:
        dynamo.reset()
    device_mesh = sys.modules.get("tensorrt_llm._torch.device_mesh")
    if device_mesh is not None:
        device_mesh.DeviceMeshTopologyImpl.device_mesh = None
        device_mesh.DeviceMeshTopologyImpl.tp_mesh = None
    mapping = sys.modules.get("tensorrt_llm._torch.visual_gen.mapping")
    if mapping is not None:
        mapping.VisualGenMapping.seq_mesh = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _destroy_extra_process_groups()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Fence: a collective leaked by the test blocks here, where the parent's
    # deadline turns it into a failure of the right test.
    dist.barrier()


def _rank_main(
    rank: int,
    world_size: int,
    backend: str,
    port: int,
    parent_pid: int,
    preload: tuple,
    task_conn,
    result_conn,
) -> None:
    _bind_to_parent(parent_pid)
    # A rank is a torch.distributed process by definition, whatever the parent's
    # env says at spawn time (a module-scope fixture may have popped the flag).
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # not "localhost": it may resolve to ::1 first
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        import torch
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        for name in preload:
            importlib.import_module(name)
    except Exception:
        result_conn.send((rank, _ERROR, traceback.format_exc()))
        return
    env_snapshot = dict(os.environ)
    native_watchdog = False
    if "tensorrt_llm" in sys.modules:
        native_watchdog = _start_native_watchdog(parent_pid)
    result_conn.send((rank, _READY, None))

    while True:
        try:
            payload = task_conn.recv()
        except EOFError:
            break
        if payload is None:
            break
        try:
            fn, args, kwargs = pickle.loads(payload)
            if not native_watchdog and "tensorrt_llm" in sys.modules:
                native_watchdog = _start_native_watchdog(parent_pid)
            fn(rank, world_size, *args, **kwargs)
            _reset_rank_state(env_snapshot)
        except Exception:
            result_conn.send((rank, _ERROR, traceback.format_exc()))
            # Stay parked: the parent kills the whole pool. Exiting here would
            # run NCCL destructors against peers that may be wedged.
            continue
        result_conn.send((rank, _OK, None))

    try:
        dist.destroy_process_group()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def _gpu_holders(pids: set[int], deadline_s: float) -> list[int]:
    """Pids from ``pids`` that nvidia-smi still lists as holding GPU memory."""
    end = time.monotonic() + deadline_s
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        listed = {int(tok) for tok in out.split() if tok.strip().isdigit()}
        held = sorted(pids & listed)
        if not held or time.monotonic() >= end:
            return held
        time.sleep(1.0)


class _Pool:
    def __init__(self, world_size: int, backend: str, preload: tuple):
        self.world_size = world_size
        self.backend = backend
        self.preload = preload
        self.snapshot = _spawn_snapshot()
        self._ctx = mp.get_context("spawn")
        self._procs: list = []
        self._task_conns: list = []  # parent's send end, one per rank
        self._result_conns: list = []  # parent's receive end, one per rank

    # ---- lifecycle ----

    def start(self, deadline_at: float | None) -> None:
        from ._visual_gen_dist_utils import spawn_with_retry

        def attempt(port: int) -> None:
            self._launch(port)
            # On failure _collect has already killed the ranks. A rendezvous
            # EADDRINUSE is retried with a fresh port by spawn_with_retry;
            # anything else (CUDA init failure included) is raised at once.
            self._collect(_READY, deadline_at, f"spawn on port {port}")

        spawn_with_retry(attempt, retryable=(WorkerPoolError,))

    def _launch(self, port: int) -> None:
        self._procs, self._task_conns, self._result_conns = [], [], []
        for rank in range(self.world_size):
            task_r, task_w = self._ctx.Pipe(duplex=False)
            result_r, result_w = self._ctx.Pipe(duplex=False)
            p = self._ctx.Process(
                target=_rank_main,
                args=(
                    rank,
                    self.world_size,
                    self.backend,
                    port,
                    os.getpid(),
                    self.preload,
                    task_r,
                    result_w,
                ),
                name=f"visualgen-pool-{self.world_size}x{self.backend}-rank{rank}",
            )
            p.start()
            # The child holds its own copies; closing ours gives EOF semantics.
            task_r.close()
            result_w.close()
            self._procs.append(p)
            self._task_conns.append(task_w)
            self._result_conns.append(result_r)

    def run(self, fn: Callable, args: tuple, kwargs: dict, deadline_at: float | None) -> None:
        # Pickle up front so an unpicklable fn fails here with a clear error.
        payload = pickle.dumps((fn, args, kwargs))
        try:
            for conn in self._task_conns:
                conn.send(payload)
            self._collect(_OK, deadline_at, getattr(fn, "__name__", repr(fn)))
        except BaseException:
            self.kill()
            raise

    def shutdown(self) -> str:
        """Ask the ranks to exit, then kill whatever has not left in time.

        Returns the kill report (empty when every rank is gone and holds no GPU
        memory).
        """
        for conn in self._task_conns:
            try:
                conn.send(None)
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + _GRACE_SECONDS
        for p in self._procs:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        return self.kill()

    def kill(self) -> str:
        """SIGKILL every rank still alive, drop the connections, verify release.

        The ranks are direct children, so ``Process.kill`` is already safe
        against pid reuse (it is a no-op once the child has been reaped). No
        graceful stop: a discarded pool may hold ranks wedged in a collective.
        Ranks are not daemonic (test bodies may fork compile workers), so this
        and ``shutdown`` are the only things that end them.

        Returns a report of anything that did not go away: a rank the kernel
        did not reap in time (stuck in the driver), or rank pids that
        nvidia-smi still lists as holding GPU memory. Empty on success.
        """
        pids = {p.pid for p in self._procs if p.pid is not None}
        for p in self._procs:
            if p.is_alive():
                p.kill()
        for p in self._procs:
            p.join(timeout=_GRACE_SECONDS)
        for conn in [*self._task_conns, *self._result_conns]:
            conn.close()
        problems = []
        alive = sorted(p.pid for p in self._procs if p.is_alive())
        if alive:
            problems.append(f"ranks not reaped {_GRACE_SECONDS:.0f}s after SIGKILL (pids {alive})")
        if self.backend == "nccl" and pids:
            held = _gpu_holders(pids, _GRACE_SECONDS)
            if held:
                problems.append(
                    f"rank pids {held} still hold GPU memory {_GRACE_SECONDS:.0f}s after SIGKILL"
                )
        self._procs, self._task_conns, self._result_conns = [], [], []
        return "\n".join(problems)

    # ---- result collection ----

    def _collect(self, expect: str, deadline_at: float | None, what: str) -> None:
        """Wait for one report per rank; raise unless all of them are ``expect``.

        Sleeps on the ranks' result pipes and their death sentinels at once, so
        a report or a dead rank wakes the parent immediately. Stops waiting
        early once any rank has failed or died, after a grace period for the
        other ranks' reports -- the survivors may be blocked in a collective
        and would never report on their own.
        """
        reports: dict[int, tuple[str, Any]] = {}
        first_failure_at: float | None = None

        while len(reports) < self.world_size:
            now = time.monotonic()
            if deadline_at is not None and now >= deadline_at:
                msg = self._describe(what, reports, "deadline reached", "no report by the deadline")
                raise WorkerPoolError(self._append_kill_report(msg))
            if first_failure_at is not None and now - first_failure_at >= _GRACE_SECONDS:
                break
            pending = [r for r in range(self.world_size) if r not in reports]
            waitables = [self._result_conns[r] for r in pending]
            waitables += [self._procs[r].sentinel for r in pending if self._procs[r].is_alive()]
            if len(waitables) == len(pending) and first_failure_at is not None:
                break  # every pending rank is dead; nothing more will arrive
            budgets = []
            if deadline_at is not None:
                budgets.append(deadline_at - now)
            if first_failure_at is not None:
                budgets.append(_GRACE_SECONDS - (now - first_failure_at))
            ready = _wait_connections(waitables, timeout=min(budgets) if budgets else None)
            for obj in ready:
                if obj in self._result_conns:
                    try:
                        rank, status, detail = obj.recv()
                    except EOFError:  # rank died before reporting
                        if first_failure_at is None:
                            first_failure_at = time.monotonic()
                        continue
                    reports[rank] = (status, detail)
                    if status != expect and first_failure_at is None:
                        first_failure_at = time.monotonic()
                elif first_failure_at is None:
                    first_failure_at = time.monotonic()  # a sentinel fired: rank died

        if len(reports) == self.world_size and all(s == expect for s, _ in reports.values()):
            return
        failed = [r for r, (s, _) in reports.items() if s != expect]
        dead = self._dead_ranks(reports)
        parts = []
        if failed:
            parts.append(f"ranks {failed} failed")
        if dead:
            parts.append(f"ranks {dead} died")
        msg = self._describe(
            what,
            reports,
            ", ".join(parts) or "incomplete reports",
            "no report (killed after peer failure)",
        )
        raise WorkerPoolError(self._append_kill_report(msg))

    def _append_kill_report(self, msg: str) -> str:
        problems = self.kill()
        return f"{msg}\n{problems}" if problems else msg

    def _dead_ranks(self, reports: dict) -> list[int]:
        return [
            rank for rank, p in enumerate(self._procs) if rank not in reports and not p.is_alive()
        ]

    def _describe(self, what: str, reports: dict, headline: str, silent_alive: str) -> str:
        lines = [f"{self.world_size}-rank {self.backend} pool, {what}: {headline}"]
        for rank, p in enumerate(self._procs):
            if rank in reports:
                status, detail = reports[rank]
                if detail:
                    lines.append(f"--- rank {rank}: {status} ---\n{detail.rstrip()}")
                else:
                    lines.append(f"--- rank {rank}: {status} ---")
            elif not p.is_alive():
                lines.append(
                    f"--- rank {rank}: exited with code {p.exitcode} without reporting ---"
                )
            else:
                lines.append(f"--- rank {rank}: {silent_alive} ---")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pool cache
# ---------------------------------------------------------------------------


def _retire(pool: _Pool, reason: str) -> None:
    _log(f"retiring {pool.world_size}-rank {pool.backend} pool: {reason}")
    problems = pool.shutdown()
    if problems:
        raise WorkerPoolError(f"retired {pool.world_size}-rank {pool.backend} pool:\n{problems}")


def _acquire(world_size: int, backend: str, preload: tuple, deadline_at: float | None) -> _Pool:
    key = (world_size, backend, preload)
    pool = _POOLS.pop(key, None)
    if pool is not None and pool.snapshot != _spawn_snapshot():
        _retire(pool, "env/sys.path changed since spawn")
        pool = None
    if pool is None:
        _log(f"spawning {world_size}-rank {backend} pool")
        t0 = time.monotonic()
        pool = _Pool(world_size, backend, preload)
        pool.start(deadline_at)
        _log(f"{world_size}-rank {backend} pool ready in {time.monotonic() - t0:.0f}s")
    return pool


def _release(pool: _Pool) -> None:
    if lifetime() == "test":
        _retire(pool, "pool lifetime is one test")
    else:
        _POOLS[(pool.world_size, pool.backend, pool.preload)] = pool


def shutdown_all() -> None:
    """Shut every cached pool down; problems are printed, not raised (hook/atexit path)."""
    while _POOLS:
        _, pool = _POOLS.popitem()
        problems = pool.shutdown()
        if problems:
            _log(f"ERROR shutting down {pool.world_size}-rank {pool.backend} pool:\n{problems}")


# multiprocessing's own exit hook joins non-daemon children, which would block
# forever on ranks parked at their task connection. atexit runs hooks last-in
# first-out; importing multiprocessing.util above registered that hook first,
# so ours runs before it.
atexit.register(shutdown_all)


def run_distributed(
    world_size: int,
    fn: Callable,
    *args: Any,
    backend: str = "nccl",
    timeout: float | None = None,
    reuse: bool = True,
    preload: tuple[str, ...] = (),
    **kwargs: Any,
) -> None:
    """Run ``fn(rank, world_size, *args, **kwargs)`` on every rank of a pool.

    Raises :class:`WorkerPoolError` carrying every rank's traceback if any rank
    fails, dies, or the run (spawn included) exceeds ``timeout`` seconds. The
    default deadline is the test's pytest-timeout minus the kill path's budget,
    so the pool fails one test instead of pytest-timeout killing the whole
    process; without pytest-timeout there is no deadline. ``fn`` and its
    arguments must be picklable; ``fn`` must be a module-level function.

    ``reuse=False`` drains every cached pool first and runs ``fn`` on fresh
    ranks that are shut down afterwards, for a test that depends on process
    state no earlier test may have touched.

    ``preload`` names modules each rank imports before reporting ready, so a
    slow library import is paid under the spawn deadline and shows in the
    pool's ready time instead of inflating the first test's time.
    """
    budget = _default_timeout() if timeout is None else timeout
    deadline_at = None if budget is None else time.monotonic() + budget
    if not reuse:
        shutdown_all()
        pool = _acquire(world_size, backend, preload, deadline_at)
        try:
            pool.run(fn, args, kwargs, deadline_at)
        finally:
            _retire(pool, "test asked for fresh ranks")
        return
    pool = _acquire(world_size, backend, preload, deadline_at)
    pool.run(fn, args, kwargs, deadline_at)
    _release(pool)
