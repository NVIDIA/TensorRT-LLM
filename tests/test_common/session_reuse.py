# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automatic MPI session reuse for bare ``LLM(...)`` tests — zero test changes.

Instead of destroying its ``MpiPoolSession`` at ``LLM`` shutdown, the pool is
returned to a per-size cache and handed to the NEXT bare ``LLM(...)`` of the
same size, saving the ~50-65s worker spawn+import per reuse. This delivers the
same reuse the explicit fixtures in this PR provide, but through shared test
infrastructure only: no test signature changes, no wrapper functions.

Eligibility is automatic:
- size mismatch            -> new pool (cache keeps the old one for later)
- env/sys.path mismatch    -> cached pool retired (workers froze that state
                              at spawn; a stale pool would silently miss it)
- RPC executors            -> keep a private, never-cached pool; their seam
                              drains the cache first (their engine build
                              cannot share GPUs with cached idle pools)
- tests passing their own ``_mpi_session`` (the explicit fixtures) -> never
  reach the patched seam
- ``@pytest.mark.private_mpi_session`` -> explicit opt-out: the cache is
  drained and the test gets an untracked fresh pool
- use-count cap            -> pool retired after N handouts (default 16),
                              bounding worker state accumulation

Between handouts every worker runs a torch.compile/Dynamo reset (rank-verified
via ``grouped_test_utils.submit_sync_per_worker``) and the handover waits for
the previous worker's GPU memory to actually be released (NVML settle barrier)
— both failure modes were observed in validation, not hypothetical.

Enable/disable with ``TRTLLM_TEST_REUSE_SESSION`` (default on; ``0`` disables).
Disabled under pytest-xdist workers (parallel tests would multiply live pools).
"""

import os
import sys
import threading
import time

from test_common.grouped_test_utils import reset_worker_torch_compile_state, submit_sync_per_worker

# The only places in the library that construct MpiPoolSession for a bare
# LLM(...); tests passing their own _mpi_session never reach these lines.
_PATCH_TARGETS = (
    "tensorrt_llm.executor.proxy",
    "tensorrt_llm.llmapi.llm",
)
# RPC executors keep a PRIVATE pool (never cached), but their engine build
# cannot share GPUs with cached idle pools (observed init hang), so their
# seam gets a drain-then-build factory instead of the cache.
_RPC_PATCH_TARGET = "tensorrt_llm.executor.rpc_proxy"
_ALL_PATCH_TARGETS = _PATCH_TARGETS + (_RPC_PATCH_TARGET,)

# Worker-side HF weight cache for cache-managed pools only: set (if absent)
# around the spawn so the workers freeze it, then restored — private/RPC
# pools keep the production default (cache off).
_WEIGHT_CACHE_ENV = {
    "TRTLLM_HF_WEIGHT_CACHE": "1",
    "TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES": "1",
}

# Workers freeze the parent environment AND sys.path at spawn time, so a
# cached pool must not be handed to a test that changed either (silently
# stale env / unimportable monkeypatched modules). Process bookkeeping that
# legitimately drifts between tests is ignored; a false mismatch only costs
# one synchronous rebuild.
_ENV_IGNORE = frozenset(
    {
        "PYTEST_CURRENT_TEST",
        "COLUMNS",
        "LINES",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
    }
)


def _spawn_snapshot():
    """The worker-visible state a pool freezes at spawn: env + sys.path."""
    return (
        {k: v for k, v in os.environ.items() if k not in _ENV_IGNORE},
        list(sys.path),
    )


# GPU-memory settle barrier at handover: a reused live pool skips the ~50s
# synchronous spawn that used to give the previous LLM's worker time to exit;
# its CUDA memory is only released when the process actually exits. Building
# the next model into that race fails with "insufficient GPU memory".
_SETTLE_MIN_FREE_FRAC = 0.85
_SETTLE_POLL_S = 0.5
_SETTLE_FLAT_POLLS = 3
_SETTLE_EPSILON = 256 << 20
_SETTLE_TIMEOUT_S = 30.0


def _visible_gpu_indices(count: int):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return list(range(count))
    indices = []
    for token in visible.split(","):
        token = token.strip()
        if not token.isdigit() or int(token) >= count:
            return list(range(count))  # UUID/MIG form: fall back to all GPUs
        indices.append(int(token))
    return indices or list(range(count))


def wait_gpu_memory_settle() -> None:
    """Wait until visible GPUs are mostly free or free memory stops rising.

    Never raises: on any NVML problem the handover proceeds as before.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return
    try:
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in _visible_gpu_indices(pynvml.nvmlDeviceGetCount())
        ]

        def _free_total():
            infos = [pynvml.nvmlDeviceGetMemoryInfo(h) for h in handles]
            return [i.free for i in infos], [i.total for i in infos]

        t0 = time.monotonic()
        flat, prev = 0, None
        while True:
            free, total = _free_total()
            if all(f >= _SETTLE_MIN_FREE_FRAC * t for f, t in zip(free, total)):
                break
            if prev is not None and all(f - p < _SETTLE_EPSILON for f, p in zip(free, prev)):
                flat += 1
                if flat >= _SETTLE_FLAT_POLLS:
                    break  # not increasing: that memory is legitimately in use
            else:
                flat = 0
            if time.monotonic() - t0 >= _SETTLE_TIMEOUT_S:
                break
            prev = free
            time.sleep(_SETTLE_POLL_S)
        waited = time.monotonic() - t0
        if waited >= _SETTLE_POLL_S:
            print(
                f"[session-reuse] waited {waited:.1f}s before handover for GPU memory release",
                flush=True,
            )
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _proc_start_time(pid: int):
    """Kernel start time (jiffies since boot) of ``pid``, or None if gone.

    PIDs are recycled by the OS, but the (pid, start_time) pair is unique:
    verifying it right before SIGKILL prevents killing an unrelated process
    (e.g. a replacement pool's worker) that inherited a dead worker's PID.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read()
        # Field 2 (comm) may contain spaces/parens; parse after the last ')'.
        return stat.rsplit(b")", 1)[1].split()[19]  # field 22 overall
    except OSError:
        return None


def _get_worker_pid() -> tuple:
    """Runs inside a worker; module-level so it is picklable."""
    pid = os.getpid()
    return (pid, _proc_start_time(pid))


def _collect_worker_pids(real, n_workers: int) -> tuple:
    """Record the worker PIDs of a freshly spawned pool.

    ``_retire`` uses them to SIGKILL wedged workers: a graceful shutdown
    blocks forever on a broken pool and ``shutdown_abort`` would MPI_Abort
    the parent test process too. Records (pid, start_time) pairs so the kill
    can verify the PID was not recycled. Best effort — a missing PID just
    means that worker is left for the graceful-shutdown fallback.
    """
    pids: set = set()
    try:
        for _ in range(4):
            pids.update(real.submit_sync(_get_worker_pid))
            if len(pids) >= n_workers:
                break
    except Exception:
        pass
    return tuple(sorted(pids))


def _describe_mismatch(spawn_snap, now_snap, uses, max_uses):
    """One line naming WHY a cached pool cannot be handed out (observability)."""
    if uses >= max_uses:
        return f"lifetime cap reached ({uses}/{max_uses} uses)"
    spawn_env, spawn_path = spawn_snap
    now_env, now_path = now_snap
    changed = [k for k in set(spawn_env) | set(now_env) if spawn_env.get(k) != now_env.get(k)]
    if changed:
        return f"env changed since spawn: {sorted(changed)[:6]}"
    if spawn_path != now_path:
        added = [p for p in now_path if p not in spawn_path]
        removed = [p for p in spawn_path if p not in now_path]
        return f"sys.path changed since spawn: +{added[:3]} -{removed[:3]}"
    return "snapshot mismatch"


class _ReusableSession:
    """A pool wrapper whose ``shutdown()`` returns it to the cache.

    Everything else delegates to the real ``MpiPoolSession``. ``shutdown_abort``
    marks the pool dead so a crashed pool is never handed out again.
    """

    def __init__(self, real, cache):
        self._real = real
        self._cache = cache
        self._dead = False
        self._released = False

    def __getattr__(self, name):
        # Only fires for names NOT set on the wrapper (plain attribute reads
        # of _real/_cache/... resolve normally, no recursion hazard). Reads
        # after release stay delegated (harmless); destructive calls are
        # gated in shutdown()/shutdown_abort() below.
        return getattr(self.__dict__["_real"], name)

    def shutdown(self):
        if self._dead or self._released:
            return
        self._released = True
        self._cache._release(self._real)

    def shutdown_abort(self, *args, **kwargs):
        if self._released:
            # The pool went back to the cache at shutdown() and may already
            # belong to the NEXT test: never kill it from a late error path.
            return None
        self._dead = True
        self._cache._forget(self._real)
        return self._real.shutdown_abort(*args, **kwargs)


class SessionReuseCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._pools = {}  # n_workers -> MpiPoolSession
        self._patched = set()
        self._suspended = False

    @property
    def enabled(self) -> bool:
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return False  # parallel workers would multiply live pools
        return os.environ.get("TRTLLM_TEST_REUSE_SESSION", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @property
    def max_uses(self) -> int:
        return int(os.environ.get("TRTLLM_TEST_REUSE_MAX_USES", "16"))

    @staticmethod
    def _retire(real, broken: bool = False):
        """Dispose of a pool in the background without blocking the test.

        Healthy retires (lifetime cap, stale env snapshot, duplicate cache
        slot) use a graceful ``shutdown()``: the workers are idle and exit
        cleanly, and killing MPI-spawned children abnormally can upset the
        MPI runtime in the parent process.

        ``broken=True`` (failed health probe) means the workers may be wedged
        in a collective: a graceful shutdown would block forever and leak
        their GPU memory into subsequent tests, and ``shutdown_abort`` calls
        ``MPI_COMM_WORLD.Abort``, which kills the parent test process too.
        Instead SIGKILL the worker PIDs recorded at spawn (a discarded pool
        needs no graceful stop; the driver reclaims GPU memory on process
        death) and then reap the client side.
        """
        pids = getattr(real, "_reuse_worker_pids", ()) if broken else ()

        def _dispose():
            import signal

            for pid, start_time in pids:
                # Guard against PID recycling: only kill if the process at
                # this PID is still the worker we recorded at spawn.
                if start_time is None or _proc_start_time(pid) != start_time:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                real.shutdown()
            except Exception:
                pass

        threading.Thread(target=_dispose, daemon=True, name="session-reuse-retire").start()

    # ---- factory installed at the pool-creation seam ----

    def install_pool_factory_if_loaded(self) -> None:
        """Lazily patch the pool-creation seams (idempotent).

        Only patches target modules ALREADY imported by the test suite, so
        suites that never create MPI pools pay nothing — not even the
        tensorrt_llm import. Called from ``pytest_runtest_setup``.
        """
        if len(self._patched) == len(_ALL_PATCH_TARGETS):
            return  # fully installed: skip the env reads and module scan
        if not self.enabled:
            return
        pending = [n for n in _ALL_PATCH_TARGETS if n in sys.modules and n not in self._patched]
        if not pending:
            return
        from tensorrt_llm.llmapi.mpi_session import MpiPoolSession as real_cls

        cache = self

        def factory(n_workers, *args, **kwargs):
            if args or kwargs:  # unknown calling convention: stay out of the way
                print(
                    "[session-reuse] bypassing reuse: MpiPoolSession called with "
                    "unexpected arguments (library signature changed?)",
                    flush=True,
                )
                return real_cls(n_workers, *args, **kwargs)
            return cache.acquire(real_cls, n_workers)

        def rpc_factory(n_workers, *args, **kwargs):
            # Fires exactly when an RPC executor is constructed, whatever the
            # test is named — no name heuristics.
            cache.drain()
            return real_cls(n_workers, *args, **kwargs)

        for name in pending:
            mod = sys.modules[name]
            if getattr(mod, "MpiPoolSession", None) is real_cls:
                mod.MpiPoolSession = rpc_factory if name == _RPC_PATCH_TARGET else factory
            self._patched.add(name)

    # ---- cache operations ----

    def acquire(self, real_cls, n_workers):
        """Hand out a cached same-size pool (reset + settled) or build one."""
        if self._suspended or not self.enabled:
            # Opt-out test (private_mpi_session) or the kill switch flipped
            # after the seams were patched: untracked fresh pool that the LLM
            # owns and destroys normally.
            return real_cls(n_workers=n_workers)
        with self._lock:
            real = self._pools.pop(n_workers, None)
        if real is not None:
            # Compare against the state FROZEN INTO the workers at spawn time:
            # if the current test expects different env/sys.path, the cached
            # workers would silently miss it.
            snap = _spawn_snapshot()
            if real._reuse_spawn_snapshot != snap or real._reuse_uses >= self.max_uses:
                print(
                    "[session-reuse] retiring cached pool: "
                    + _describe_mismatch(
                        real._reuse_spawn_snapshot, snap, real._reuse_uses, self.max_uses
                    ),
                    flush=True,
                )
                self._retire(real)  # stale worker state or lifetime cap
            else:
                try:
                    submit_sync_per_worker(real, reset_worker_torch_compile_state)
                    wait_gpu_memory_settle()
                    print(
                        f"[session-reuse] reusing {n_workers}-worker pool "
                        f"(use #{real._reuse_uses + 1})",
                        flush=True,
                    )
                    return _ReusableSession(real, self)
                except Exception as e:  # unhealthy pool: discard, build fresh
                    print(
                        f"[session-reuse] cached pool failed reset, rebuilding: {e}",
                        flush=True,
                    )
                    self._retire(real, broken=True)
        return _ReusableSession(self._spawn_fresh(real_cls, n_workers), self)

    def _spawn_fresh(self, real_cls, n_workers):
        """Spawn a cache-managed pool with the worker-side HF weight cache on.

        The cache env vars must be visible at spawn (workers freeze the env)
        and are removed right after, so non-managed pools (private/RPC) and
        the rest of the suite keep the production default. The spawn snapshot
        is taken BEFORE adding them so later acquire-time comparisons (which
        see the restored env) still match. An explicit user setting of either
        var is respected and left untouched.
        """
        snapshot = _spawn_snapshot()
        added = [k for k in _WEIGHT_CACHE_ENV if k not in os.environ]
        for k in added:
            os.environ[k] = _WEIGHT_CACHE_ENV[k]
        try:
            real = real_cls(n_workers=n_workers)
        finally:
            for k in added:
                os.environ.pop(k, None)
        real._reuse_uses = 0
        real._reuse_spawn_snapshot = snapshot
        real._reuse_worker_pids = _collect_worker_pids(real, n_workers)
        return real

    def _release(self, real):
        real._reuse_uses += 1
        with self._lock:
            prior = self._pools.get(real.n_workers)
            if prior is not None and prior is not real:
                self._retire(real)  # a pool of this size is already cached
                return
            self._pools[real.n_workers] = real

    def _forget(self, real):
        with self._lock:
            if self._pools.get(real.n_workers) is real:
                del self._pools[real.n_workers]

    def suspend(self, suspended: bool) -> None:
        """Bypass the cache for the current test (private_mpi_session)."""
        self._suspended = suspended

    def drain(self) -> None:
        """Shut down all cached pools in parallel (frees GPU/CPU footprint)."""
        with self._lock:
            pools, self._pools = list(self._pools.values()), {}
        if not pools:
            return
        threads = [
            # daemon: a wedged pool shutdown must not keep the interpreter
            # alive at exit (a non-daemon thread would hang the CI stage).
            threading.Thread(target=p.shutdown, name="session-reuse-drain", daemon=True)
            for p in pools
        ]
        for t in threads:
            t.start()
        for t in threads:
            # Bounded wait: one wedged pool shutdown must not turn a drain at
            # a shared seam (sessionfinish / RPC construction) into a
            # suite-wide hang; a leaked wedged pool is the lesser evil.
            t.join(timeout=60)
            if t.is_alive():
                print(
                    "[session-reuse] WARNING: pool shutdown did not finish within 60s", flush=True
                )
        print(f"[session-reuse] drained {len(pools)} cached pool(s)", flush=True)


REUSE = SessionReuseCache()
