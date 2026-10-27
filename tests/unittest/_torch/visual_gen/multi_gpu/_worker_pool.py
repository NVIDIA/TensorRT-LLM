# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent rank-process pool for the multi-GPU (``torch.distributed``) tests in this package.

The tests run their logic as ``fn(rank, world_size)`` on ``world_size``
processes joined by ``dist.init_process_group``. Spawning those processes per
test pays a full ``tensorrt_llm`` import per rank for a test body that runs in
seconds. This module keeps one pool of rank processes per
``(world_size, backend, preload)`` alive across tests and dispatches test
functions to it; ``fn`` runs on every rank exactly as it would under
``mp.spawn`` and is subject to the same pickling rule (module-level function).
``run_distributed`` must be called from a thread that outlives the pool (pytest's
main thread): the ranks are bound to the lifetime of the thread that spawned
them, see ``_bind_to_parent``.

Failure contract
----------------
- A rank whose test raises reports the traceback. The parent does not wait for
  the other ranks: they may be blocked in a collective waiting for the rank
  that bailed and would otherwise sit there until the NCCL timeout. After one
  grace period for stragglers' reports, the whole pool is killed and the
  failure is raised with every traceback that arrived; ranks that never
  reported are listed as killed, not as failures of their own.
- A rank that dies without reporting (segfault, CUDA fault, OOM-kill) or a test
  that reaches its deadline likewise kills the pool.
- Every failure retires the pool and the next test starts from fresh
  processes. What a failed test cannot leak into later tests is process state;
  GPU memory held by a rank the kernel will not reap can still affect them, and
  is reported (see ``_Pool.kill``).
- After a kill, ranks get one grace period to be reaped and, for NCCL pools,
  one more for the driver to release their GPU memory. Memory still held is
  part of the raised error; a release check that could not run is logged as
  unverified, never counted as success; unreaped ranks are kept on record and
  re-checked at every later shutdown.
- Ranks die with the parent: ``PR_SET_PDEATHSIG`` at rank start (kernel
  delivered SIGKILL, works even when the rank is wedged in a collective) and,
  when ``preload`` brought ``tensorrt_llm`` in, the library's native
  coordinator watchdog on top.

Deadline
--------
The package conftest records each test's deadline at test setup from its
effective pytest-timeout, minus three grace periods (reports, reap, GPU
release) so the pool fails one test before pytest-timeout kills the whole
pytest process. Every pool call in that test, spawn included, is bounded by
the time remaining; an explicit ``timeout=`` can only shorten it. Without
pytest-timeout there is no deadline.

Reset between tests
-------------------
Ranks are seeded to ``_RNG_SEED`` before their first task. After a passing test
body each rank restores ``os.environ`` to its post-spawn contents, re-seeds
``random``/``numpy``/``torch``/CUDA, clears the logger's log-once memory and
Dynamo caches, resets the visual_gen mesh singletons and empties the CUDA
allocator cache. Nothing in the reset is collective: a test may let some ranks
return early while others keep computing, and the parent's wait for every
rank's report is the fence (a rank stuck in a collective a test leaked never
reports, and that test fails at its deadline). The process groups a test
created on top of the world group, and torch's current-mesh context stack, are
destroyed at the start of the next task instead, when every rank is known to
be idle: destroying a NCCL sub-group waits for the group's peers. Not covered:
torch's default dtype, grad mode and backend flags, and any other module
global a test mutates. A test that changes such state runs with
``reuse=False``.

Lifetime
--------
``TRTLLM_TEST_VISUALGEN_POOL_LIFETIME``: ``session`` (default, the whole
pytest run), ``module`` (the conftest shuts pools down at each test file's
teardown), or ``test`` (fresh ranks for every ``run_distributed`` call, which
is what ``mp.spawn`` did; a pytest test making several calls spawns several
times). A pool is also retired when the parent's env / ``sys.path`` no longer
match what the ranks froze at spawn. Under pytest-xdist workers the lifetime
is forced to ``test``, since parallel pytest processes would multiply live
pools on the same GPUs.

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
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as _wait_connections
from typing import Any, Callable

from test_common._session_utils import _spawn_snapshot

# Rank -> parent report tags.
_READY = "ready"
_OK = "ok"
_ERROR = "error"

# Seconds the parent waits for ranks to react on their own before escalating;
# the only timing constant. See the module docstring for where it applies.
_GRACE_SECONDS = 10.0
_RNG_SEED = 0
_LIFETIMES = ("session", "module", "test")

# Cached live pools keyed by (world_size, backend, preload).
_POOLS: dict[tuple[int, str, tuple], "_Pool"] = {}
# Ranks that did not die within a grace period of SIGKILL; re-checked at every shutdown.
_UNREAPED: list[mp.Process] = []
# Absolute (monotonic) deadline of the current test, set per test by the conftest.
_test_deadline_at: float | None = None


def lifetime() -> str:
    """``session``, ``module`` or ``test``; see the module docstring."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return "test"
    value = os.environ.get("TRTLLM_TEST_VISUALGEN_POOL_LIFETIME", "session").lower()
    if value not in _LIFETIMES:
        raise ValueError(
            f"TRTLLM_TEST_VISUALGEN_POOL_LIFETIME={value!r}; expected one of {_LIFETIMES}"
        )
    return value


def set_test_deadline(pytest_timeout: float | None) -> None:
    """Record the current test's deadline from its effective pytest-timeout in
    seconds (``None``/0: no deadline). A pytest-timeout too short to leave the
    kill path its margin is used whole, which lets pytest-timeout win the race."""
    global _test_deadline_at
    if not pytest_timeout:
        _test_deadline_at = None
        return
    margin = 3 * _GRACE_SECONDS
    budget = pytest_timeout - margin if pytest_timeout > margin else pytest_timeout
    _test_deadline_at = time.monotonic() + budget


def _deadline_for(timeout: float | None) -> float | None:
    """Absolute deadline of one pool call: the test's remaining time, shortened
    further by an explicit ``timeout``."""
    if timeout is None:
        return _test_deadline_at
    call_deadline_at = time.monotonic() + timeout
    if _test_deadline_at is None:
        return call_deadline_at
    return min(call_deadline_at, _test_deadline_at)


def _log(msg: str) -> None:
    print(f"[visualgen-pool] {msg}", flush=True)


class WorkerPoolError(RuntimeError):
    """A test run on the pool failed, timed out, or lost a rank."""


# ---------------------------------------------------------------------------
# Rank process side
# ---------------------------------------------------------------------------


def _bind_to_parent(parent_pid: int) -> None:
    """Make the kernel SIGKILL this rank when its parent dies.

    ``PR_SET_PDEATHSIG`` fires when the thread that spawned the child exits, so
    the ranks live as long as the thread that called ``run_distributed``. The
    ``getppid`` check closes the window where the parent died before the flag
    was set (compare with the recorded pid, not with 1: under a subreaper an
    orphan's parent is the subreaper).
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


def _try_start_native_watchdog(parent_pid: int) -> None:
    """Add the library's coordinator watchdog; only called when ``tensorrt_llm``
    is already in ``sys.modules`` (the import is then cheap). Failure is
    reported, not fatal: PDEATHSIG already covers parent death."""
    try:
        from tensorrt_llm.bindings.internal import start_coordinator_watchdog

        warning = start_coordinator_watchdog(parent_pid)
    except Exception as e:
        print(f"[visualgen-pool] rank could not start the native watchdog: {e}", flush=True)
        return
    if warning is not None:
        print(f"[visualgen-pool] native watchdog: {warning}", flush=True)


def _reseed() -> None:
    import numpy
    import torch

    random.seed(_RNG_SEED)
    numpy.random.seed(_RNG_SEED)
    torch.manual_seed(_RNG_SEED)  # CPU and every CUDA device


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
        # A `with mesh:` context leaked by a test would make the next test's
        # current mesh wrong. The registry's other fields are either caches
        # keyed by this test's (now unreferenced) mesh objects, harmless once
        # their groups are destroyed, or user-set options that must survive.
        resources.mesh_stack.clear()


def _reset_rank_state(env_snapshot: dict[str, str]) -> None:
    import torch

    os.environ.clear()
    os.environ.update(env_snapshot)
    _reseed()

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
        torch.cuda.empty_cache()
    # Nothing collective here: ranks may finish a test at different times (a
    # test can let other ranks return while rank 0 runs a single-GPU reference
    # for minutes). Destroying a NCCL sub-group is a finalize that waits for its
    # peers, so the previous test's sub-groups are destroyed at the start of the
    # next task, once the parent has seen every rank finish.


def _rank_main(
    rank: int,
    world_size: int,
    backend: str,
    port: int,
    parent_pid: int,
    preload: tuple[str, ...],
    task_conn: Connection,
    result_conn: Connection,
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
        _reseed()
    except Exception:
        result_conn.send((_ERROR, traceback.format_exc()))
        return
    env_snapshot = dict(os.environ)
    if "tensorrt_llm" in sys.modules:  # brought in by preload
        _try_start_native_watchdog(parent_pid)
    result_conn.send((_READY, None))

    while True:
        try:
            payload = task_conn.recv_bytes()
        except EOFError:
            break
        if not payload:  # shutdown sentinel
            break
        try:
            _destroy_extra_process_groups()  # the previous test's; all ranks are idle now
            fn, args, kwargs = pickle.loads(payload)
            fn(rank, world_size, *args, **kwargs)
            _reset_rank_state(env_snapshot)
        except Exception:
            result_conn.send((_ERROR, traceback.format_exc()))
            # Stay parked: the parent kills the whole pool. Exiting here would
            # run NCCL destructors against peers that may be wedged.
            continue
        result_conn.send((_OK, None))

    try:
        dist.destroy_process_group()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def _gpu_holders(pids: set[int], deadline_at: float) -> tuple[list[int], str | None]:
    """``(held, problem)``: pids from ``pids`` that nvidia-smi still lists as
    holding GPU memory at the deadline, and why the check could not be trusted
    (``None`` when it ran)."""
    while True:
        remaining = deadline_at - time.monotonic()
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=max(1.0, remaining),
            )
        except FileNotFoundError:
            return [], "nvidia-smi not found"
        except subprocess.TimeoutExpired:
            return [], "nvidia-smi did not answer in time"
        if proc.returncode != 0:
            return [], f"nvidia-smi exited with {proc.returncode}: {proc.stderr.strip()}"
        listed = {int(tok) for tok in proc.stdout.split() if tok.strip().isdigit()}
        held = sorted(pids & listed)
        remaining = deadline_at - time.monotonic()
        if not held or remaining <= 0:
            return held, None
        time.sleep(min(1.0, remaining))


@dataclass
class _Rank:
    proc: mp.Process
    task_conn: Connection  # parent's send end
    result_conn: Connection  # parent's receive end


class _Pool:
    def __init__(self, world_size: int, backend: str, preload: tuple[str, ...]):
        self.world_size = world_size
        self.backend = backend
        self.preload = preload
        self.snapshot = _spawn_snapshot()
        self._ranks: list[_Rank] = []

    # ---- lifecycle ----

    def start(self, deadline_at: float | None) -> None:
        """Spawn the ranks and wait for every READY."""
        from ._visual_gen_dist_utils import spawn_with_retry

        def attempt(port: int) -> None:
            self._owning(lambda: self._launch(port), _READY, deadline_at, f"spawn on port {port}")

        # A rendezvous EADDRINUSE is retried with a fresh port; anything else
        # (CUDA init failure included) is raised at once.
        spawn_with_retry(attempt, retryable=(WorkerPoolError,))

    def _launch(self, port: int) -> None:
        ctx = mp.get_context("spawn")
        self._ranks = []
        for rank in range(self.world_size):
            task_r, task_w = ctx.Pipe(duplex=False)
            result_r, result_w = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_rank_main,
                args=(rank, self.world_size, self.backend, port, os.getpid(), self.preload),
                kwargs={"task_conn": task_r, "result_conn": result_w},
                name=f"visualgen-pool-{self.world_size}x{self.backend}-rank{rank}",
            )
            proc.start()
            # The child holds its own copies; closing ours gives EOF semantics
            # in both directions when either side dies.
            task_r.close()
            result_w.close()
            self._ranks.append(_Rank(proc, task_w, result_r))

    def run(self, payload: bytes, what: str, deadline_at: float | None) -> None:
        """Run one pre-pickled task on every rank."""

        def dispatch() -> None:
            for r in self._ranks:
                r.task_conn.send_bytes(payload)

        self._owning(dispatch, _OK, deadline_at, what)

    def _owning(
        self, action: Callable[[], None], expect: str, deadline_at: float | None, what: str
    ) -> None:
        """Run ``action`` and collect one ``expect`` report per rank, as the single
        owner of cleanup for this attempt: on a reported failure the ranks are
        killed and the kill report is appended to the error; on any other
        exception (interrupts included) they are killed and it propagates."""
        try:
            action()
            self._collect(expect, deadline_at, what)
        except WorkerPoolError as e:
            problems = self.kill()
            raise WorkerPoolError(f"{e}\n{problems}" if problems else str(e)) from None
        except BaseException:
            self.kill()
            raise

    def shutdown(self) -> str:
        """Ask the ranks to exit, give them one grace period, then ``kill``."""
        for r in self._ranks:
            try:
                r.task_conn.send_bytes(b"")  # shutdown sentinel
            except (OSError, ValueError):
                pass
        deadline_at = time.monotonic() + _GRACE_SECONDS
        for r in self._ranks:
            r.proc.join(timeout=max(0.0, deadline_at - time.monotonic()))
        return self.kill()

    def kill(self) -> str:
        """SIGKILL every rank still alive, drop the connections, check release.

        Ranks are direct children, so ``Process.kill`` is already safe against
        pid reuse (a no-op once the child has been reaped). They are not
        daemonic (test bodies may fork compile workers), so this and
        ``shutdown`` are the only things that end them.

        Returns the problems detected: ranks not reaped within one grace
        period (also appended to ``_UNREAPED``), and rank pids that nvidia-smi
        still lists after one more. Empty when nothing was detected; a release
        check that could not run is logged, not returned.
        """
        pids = {r.proc.pid for r in self._ranks if r.proc.pid is not None}
        for r in self._ranks:
            if r.proc.is_alive():
                r.proc.kill()
        deadline_at = time.monotonic() + _GRACE_SECONDS
        for r in self._ranks:
            r.proc.join(timeout=max(0.0, deadline_at - time.monotonic()))
        for r in self._ranks:
            r.task_conn.close()
            r.result_conn.close()
        problems = []
        unreaped = [r.proc for r in self._ranks if r.proc.is_alive()]
        if unreaped:
            _UNREAPED.extend(unreaped)
            problems.append(
                f"ranks not reaped {_GRACE_SECONDS:.0f}s after SIGKILL "
                f"(pids {sorted(p.pid for p in unreaped)})"
            )
        if self.backend == "nccl" and pids:
            held, problem = _gpu_holders(pids, time.monotonic() + _GRACE_SECONDS)
            if problem is not None:
                _log(
                    f"WARNING: could not verify GPU memory release of pids {sorted(pids)}: "
                    f"{problem}"
                )
            elif held:
                problems.append(
                    f"rank pids {held} still hold GPU memory {_GRACE_SECONDS:.0f}s after SIGKILL"
                )
        self._ranks = []
        return "\n".join(problems)

    # ---- result collection ----

    def _collect(self, expect: str, deadline_at: float | None, what: str) -> None:
        """Wait for one report per rank; raise ``WorkerPoolError`` unless all of
        them are ``expect``. Does not kill anything; see ``_owning``.

        Sleeps on the pending ranks' result pipes and death sentinels at once.
        Once any rank has failed or died, waits one more grace period for the
        others' reports, then gives up on them: survivors may be blocked in a
        collective and would never report on their own.
        """
        pending = {r.result_conn: i for i, r in enumerate(self._ranks)}
        reports: dict[int, tuple[str, Any]] = {}
        first_failure_at: float | None = None

        def note_failure() -> None:
            nonlocal first_failure_at
            if first_failure_at is None:
                first_failure_at = time.monotonic()

        while pending:
            now = time.monotonic()
            if deadline_at is not None and now >= deadline_at:
                raise WorkerPoolError(
                    self._describe(what, reports, "deadline reached", "no report by the deadline")
                )
            if first_failure_at is not None and now - first_failure_at >= _GRACE_SECONDS:
                break
            sentinels = [
                self._ranks[i].proc.sentinel
                for i in pending.values()
                if self._ranks[i].proc.is_alive()
            ]
            budgets = []
            if deadline_at is not None:
                budgets.append(deadline_at - now)
            if first_failure_at is not None:
                budgets.append(_GRACE_SECONDS - (now - first_failure_at))
            for obj in _wait_connections(
                [*pending, *sentinels], timeout=min(budgets, default=None)
            ):
                if obj not in pending:
                    note_failure()  # a death sentinel fired; its pipe reports EOF next
                    continue
                try:
                    status, detail = obj.recv()
                except EOFError:  # rank died before reporting
                    del pending[obj]
                    note_failure()
                    continue
                reports[pending.pop(obj)] = (status, detail)
                if status != expect:
                    note_failure()

        if not pending and all(s == expect for s, _ in reports.values()):
            return
        failed = [i for i, (s, _) in reports.items() if s != expect]
        dead = [i for i, r in enumerate(self._ranks) if i not in reports and not r.proc.is_alive()]
        parts = []
        if failed:
            parts.append(f"ranks {failed} failed")
        if dead:
            parts.append(f"ranks {dead} died")
        raise WorkerPoolError(
            self._describe(
                what,
                reports,
                ", ".join(parts) or "incomplete reports",
                "no report (killed after peer failure)",
            )
        )

    def _describe(self, what: str, reports: dict, headline: str, silent_alive: str) -> str:
        lines = [f"{self.world_size}-rank {self.backend} pool, {what}: {headline}"]
        for i, r in enumerate(self._ranks):
            if i in reports:
                status, detail = reports[i]
                lines.append(
                    f"--- rank {i}: {status} ---" + (f"\n{detail.rstrip()}" if detail else "")
                )
            elif not r.proc.is_alive():
                lines.append(
                    f"--- rank {i}: exited with code {r.proc.exitcode} without reporting ---"
                )
            else:
                lines.append(f"--- rank {i}: {silent_alive} ---")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pool cache
# ---------------------------------------------------------------------------


def _retire(pool: _Pool, reason: str) -> None:
    _log(f"retiring {pool.world_size}-rank {pool.backend} pool: {reason}")
    problems = pool.shutdown()
    if problems:
        raise WorkerPoolError(f"retired {pool.world_size}-rank {pool.backend} pool:\n{problems}")


def _acquire(
    world_size: int, backend: str, preload: tuple[str, ...], deadline_at: float | None
) -> _Pool:
    key = (world_size, backend, preload)
    pool = _POOLS.pop(key, None)
    if pool is not None and not all(r.proc.is_alive() for r in pool._ranks):
        # Died while parked (OOM killer, NCCL watchdog): not this test's failure.
        _retire(pool, "rank died while idle")
        pool = None
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


def _recheck_unreaped() -> None:
    """Join ranks that survived an earlier SIGKILL for one grace period; keep
    the ones still alive on record and report them."""
    if not _UNREAPED:
        return
    deadline_at = time.monotonic() + _GRACE_SECONDS
    for p in _UNREAPED:
        p.join(timeout=max(0.0, deadline_at - time.monotonic()))
    _UNREAPED[:] = [p for p in _UNREAPED if p.is_alive()]
    if _UNREAPED:
        _log(
            f"ERROR: ranks still not reaped after SIGKILL (pids {sorted(p.pid for p in _UNREAPED)})"
        )


def shutdown_all() -> None:
    """Shut every cached pool down and re-check unreaped ranks. Called after a
    failed test, at module teardown under lifetime ``module``, at session end
    and at interpreter exit; problems are printed, not raised."""
    while _POOLS:
        _, pool = _POOLS.popitem()
        problems = pool.shutdown()
        if problems:
            _log(f"ERROR shutting down {pool.world_size}-rank {pool.backend} pool:\n{problems}")
    _recheck_unreaped()


# multiprocessing's own exit hook joins non-daemon children, which would block
# forever on ranks parked at their task connection. atexit runs hooks last-in
# first-out; importing multiprocessing.util above registered that hook first,
# so ours runs before it.
atexit.register(shutdown_all)


def run_distributed(
    world_size: int,
    fn: Callable,
    *args: Any,
    dist_backend: str = "nccl",
    timeout: float | None = None,
    reuse: bool = True,
    preload: tuple[str, ...] = (),
    **kwargs: Any,
) -> None:
    """Run ``fn(rank, world_size, *args, **kwargs)`` on every rank of a pool.

    ``dist_backend`` is the torch.distributed backend of the pool (``nccl`` or
    ``gloo``); it is not called ``backend`` because the test functions here take
    an attention ``backend`` kwarg that travels through ``**kwargs``.

    Raises :class:`WorkerPoolError` carrying every rank's traceback if any rank
    fails, dies, or the run (spawn included) reaches its deadline (module
    docstring, Deadline); ``timeout`` in seconds can only shorten it. ``fn``
    and its arguments must be picklable; ``fn`` must be a module-level
    function.

    ``reuse=False`` drains every cached pool first and runs ``fn`` on fresh
    ranks that are shut down afterwards (module docstring, Reset between
    tests). ``preload`` names modules each rank imports before reporting
    ready, so their import time is attributed to the spawn, not to the first
    test.
    """
    # Everything that can fail without a rank existing comes first.
    payload = pickle.dumps((fn, args, kwargs))
    what = getattr(fn, "__name__", repr(fn))
    mode = lifetime()
    deadline_at = _deadline_for(timeout)

    if not reuse:
        shutdown_all()
    pool = _acquire(world_size, dist_backend, preload, deadline_at)  # kills its ranks on failure
    pool.run(payload, what, deadline_at)  # likewise
    if not reuse:
        _retire(pool, "test asked for fresh ranks")
    elif mode == "test":
        _retire(pool, "pool lifetime is one call")
    else:
        _POOLS[(world_size, dist_backend, preload)] = pool
