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
- torch.compile tests      -> private pool; the process-global Userbuffers
                              Manager assumes one Engine per process
- use-count cap            -> pool retired after N handouts (default 16),
                              bounding worker state accumulation

Between eligible handouts every worker runs a health probe and defensive
torch.compile/Dynamo reset (exactly once per worker, barrier-pinned:
``grouped_test_utils.submit_sync_per_worker``). Handover cannot race the
previous worker's GPU-memory release: every pool these layers build is
constructed with ``wait_shutdown=True``, so its shutdown blocks until the
workers actually exited.

Cache misses (first pool of a size, post-drain rebuild, post-retire
replacement) take a shadow pool pre-spawned by the session-prefetch layer
when it is wired (``_prefetcher``), hiding the ~50s spawn; each miss restocks
one shadow for the next.

Enable/disable with ``TRTLLM_TEST_REUSE_SESSION`` (default on; ``0`` disables).
Disabled under pytest-xdist workers (parallel tests would multiply live pools).
"""

import os
import sys
import threading
import time
from typing import Protocol

# The spawn snapshot is shared with the session-prefetch layer (both hand a
# live pool to a test that did not spawn it — same invariant).
from test_common._session_utils import _isinstance_transparent_shim, _spawn_snapshot
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


_RETIRE_THREADS: list[threading.Thread] = []
_RETIRE_LOCK = threading.Lock()


class _PoolSession(Protocol):
    _reuse_worker_pids: tuple[tuple[int, bytes | None], ...]

    def shutdown(self) -> None: ...

    def release_exit_joins(self) -> None: ...


def _reap_retires(timeout: float = 60.0) -> None:
    """Join in-flight retire threads (bounded); no-op when none are running.

    A retired pool's workers hold their (full-model) GPU memory until they
    exit; the retire thread blocks on that exit (``wait_shutdown=True``), but
    it is a BACKGROUND thread — the test hot path never waits on it. Before
    an instant cached-pool handover, joining in-flight retires is what makes
    the handover safe against a corpse still releasing (e.g. the duplicate
    retired by ``_release`` moments earlier); every other path spawns fresh
    (~50s), which outlasts the release naturally. Also called at drain
    rendezvous points so disposals cannot leak past the session.
    """
    with _RETIRE_LOCK:
        in_flight, _RETIRE_THREADS[:] = list(_RETIRE_THREADS), []
    for t in in_flight:
        t.join(timeout=timeout)
        if t.is_alive():
            print(
                "[session-reuse] WARNING: pool retirement did not finish within 60s",
                flush=True,
            )


def _worker_start_time(pid: int) -> bytes | None:
    """Read a worker's kernel start time without loading TRT-LLM eagerly."""
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    return _process_start_time(pid)


def _kill_recorded_workers(real: _PoolSession) -> int:
    """SIGKILL this pool's recorded workers, guarded against PID reuse.

    Where the kernel supports it, the signal goes through a pidfd. Opening the
    pidfd binds this loop to one exact process, so the start-time recheck below
    it can no longer be invalidated by the PID being recycled before the signal
    lands. Without pidfd the start-time recheck alone still guards the kill: that
    leaves a microsecond-wide window, but this is the path that reaps wedged
    workers, so refusing to signal at all would strand them on exactly the
    platforms the reaper exists for.
    """
    import signal

    send_via_pidfd = getattr(signal, "pidfd_send_signal", None)
    open_pidfd = getattr(os, "pidfd_open", None)

    killed = 0
    for pid, start_time in getattr(real, "_reuse_worker_pids", ()):
        if start_time is None:
            continue
        handle = None
        if send_via_pidfd is not None and open_pidfd is not None:
            try:
                handle = open_pidfd(pid)
            except (OSError, ValueError):
                handle = None
        try:
            # Recheck identity AFTER pinning the handle: only kill if the
            # process at this PID is still the worker recorded at spawn.
            if _worker_start_time(pid) != start_time:
                continue
            try:
                if handle is not None:
                    send_via_pidfd(handle, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
                killed += 1
            except (ProcessLookupError, PermissionError, OSError):
                pass
        finally:
            if handle is not None:
                os.close(handle)
    return killed


def _prefetcher():
    """The session-prefetch singleton when that layer is wired, else None.

    Mirror of the prefetcher's own reuse probe: coordination goes through
    sys.modules so neither layer imports the other at module load (a suite
    wired with only one layer pays nothing for the other). The prefetcher
    yields the pool SEAMS to reuse; reuse in turn consumes prefetched
    shadows on its cache misses — the two layers compose, not compete.
    """
    mod = sys.modules.get("test_common.session_prefetcher")
    return getattr(mod, "PREFETCHER", None)


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

    def is_active(self) -> bool:
        """Public probe for sibling layers: does reuse own the pool seams?

        The session prefetcher yields the ``MpiPoolSession`` seams when this
        returns True (reuse eliminates the respawn outright; prefetch could
        only hide it). Deliberately ignores ``_suspended``: a per-test
        cache bypass (``private_mpi_session``) does not change seam
        ownership.
        """
        return self.enabled

    @property
    def max_uses(self) -> int:
        return int(os.environ.get("TRTLLM_TEST_REUSE_MAX_USES", "16"))

    @staticmethod
    def _retire(real: _PoolSession, broken: bool = False) -> None:
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

        def _dispose() -> None:
            if broken:
                _kill_recorded_workers(real)
            try:
                real.shutdown()
            except Exception:
                pass

        t = threading.Thread(target=_dispose, daemon=True, name="session-reuse-retire")
        t.start()
        # Track in-flight disposals so drain() can reap them at natural
        # rendezvous points (failure fence / session finish) with a bounded
        # join; the hot path stays non-blocking and daemon=True still
        # guarantees a wedged disposal cannot hang interpreter exit.
        with _RETIRE_LOCK:
            _RETIRE_THREADS.append(t)

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
            if args or kwargs:
                return real_cls(n_workers, *args, **kwargs)
            # wait_shutdown: the private pool dies at LLM shutdown; block
            # there until its workers exited so the next pool (often handed
            # over instantly from the cache) cannot race the GPU release.
            return real_cls(n_workers=n_workers, wait_shutdown=True)

        for name in pending:
            mod = sys.modules[name]
            if getattr(mod, "MpiPoolSession", None) is real_cls:
                mod.MpiPoolSession = _isinstance_transparent_shim(
                    real_cls, rpc_factory if name == _RPC_PATCH_TARGET else factory
                )
            self._patched.add(name)

    # ---- cache operations ----

    def acquire(self, real_cls, n_workers):
        """Hand out a cached same-size pool (workers reset) or build one."""
        if self._suspended or not self.enabled:
            # Opt-out test (private_mpi_session) or the kill switch flipped
            # after the seams were patched: untracked fresh pool that the LLM
            # owns and destroys normally (wait_shutdown: its shutdown blocks
            # until the workers exited, so the next handover cannot race the
            # GPU-memory release).
            return real_cls(n_workers=n_workers, wait_shutdown=True)
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
                        real._reuse_spawn_snapshot,
                        snap,
                        real._reuse_uses,
                        self.max_uses,
                    ),
                    flush=True,
                )
                self._retire(real)  # stale worker state or lifetime cap
            else:
                try:
                    # An instant handover must not race a corpse still
                    # releasing its GPU memory. Retire threads block on the
                    # workers' exit (wait_shutdown=True) but run in the
                    # BACKGROUND, so join any in flight (a duplicate retired
                    # by _release moments ago held full model memory). No-op
                    # on the common path; every non-cached path spawns fresh
                    # (~50s), which outlasts the release naturally.
                    _reap_retires()
                    submit_sync_per_worker(real, reset_worker_torch_compile_state)
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
        """Obtain a cache-managed pool: prefetched if one is armed, else spawn.

        Every cache miss lands here (first pool of a size, post-drain
        rebuild, post-retire replacement). When the session-prefetch layer is
        wired, a shadow pool armed at the PREVIOUS miss is taken instantly —
        hiding the ~50s spawn the miss would otherwise pay — and a
        replacement shadow is armed for the next miss of this size. Without
        the prefetch layer (or on a shadow miss) the synchronous spawn is
        unchanged.

        The worker-side HF weight cache env is frozen into the workers via
        the library's ``env_overrides`` channel (parent env untouched, so
        acquire-time snapshot comparisons still match); an explicit user
        setting of either var is respected and left untouched. Prefetched
        shadows were armed with the same overlay. wait_shutdown: shutdown of
        this pool blocks until its workers exited, so a successor cannot
        race the GPU-memory release.
        """
        snapshot = _spawn_snapshot()
        overrides = {k: v for k, v in _WEIGHT_CACHE_ENV.items() if k not in os.environ}
        real = None
        prefetcher = _prefetcher()
        if prefetcher is not None:
            # A timeout must fail closed: starting a synchronous replacement
            # would create two MPI pools concurrently on the same allocation.
            # Unexpected prefetcher errors also propagate instead of silently
            # hiding lifecycle bugs behind a synchronous fallback.
            real = prefetcher.take(n_workers)
        if real is None:
            # One attempt gets the full worker-bootstrap deadline. If identity
            # collection still fails, unidentified workers may remain alive;
            # an immediate retry would overlap another MPI bootstrap with
            # them, so propagate the fail-closed error.
            real = real_cls(n_workers=n_workers, wait_shutdown=True, env_overrides=overrides)
        if prefetcher is not None:
            try:
                # Restock only after the current pool is ready. On a shadow
                # miss, scheduling before the synchronous spawn would make
                # two MPI pools bootstrap concurrently on the same GPUs.
                prefetcher.schedule_shadow(n_workers, env_overlay=overrides)
            except Exception:
                pass
        real._reuse_uses = 0
        real._reuse_spawn_snapshot = snapshot
        # (pid, start_time) per worker, recorded by the library at spawn
        # (wait_shutdown=True above). _retire uses them to SIGKILL wedged
        # workers; best effort — empty means graceful shutdown only.
        real._reuse_worker_pids = getattr(real, "_worker_identities", ())
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

    def drain(self, timeout: float = 60.0) -> None:
        """Shut down all cached pools in parallel (frees GPU/CPU footprint).

        Also reaps in-flight retire threads: drain runs at natural rendezvous
        points (failure fence, opt-out, session finish), so waiting here keeps
        disposals from leaking past the session without ever blocking the
        per-test hot path. The join is bounded for the same reason as below.
        """
        _reap_retires()
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

        # Bound the whole parallel drain, rather than waiting ``timeout`` for
        # every pool in sequence. A healthy shutdown remains graceful.
        deadline = time.monotonic() + timeout
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))

        wedged = [(pool, thread) for pool, thread in zip(pools, threads) if thread.is_alive()]
        for pool, _ in wedged:
            killed = _kill_recorded_workers(pool)
            print(
                "[session-reuse] WARNING: pool shutdown did not finish within "
                f"{timeout:g}s; sent SIGKILL to {killed} recorded worker(s)",
                flush=True,
            )

        # Killing a wedged worker should release its GPU allocation and let
        # the already-running shutdown return. Give that original thread a
        # short bounded reap window; never start a concurrent second shutdown.
        reap_deadline = time.monotonic() + min(max(timeout, 1.0), 5.0)
        for _, t in wedged:
            t.join(timeout=max(0.0, reap_deadline - time.monotonic()))

        still_alive = [(pool, thread) for pool, thread in wedged if thread.is_alive()]
        for pool, _ in still_alive:
            pool.release_exit_joins()

        # release_exit_joins() *unblocks* the wedged shutdown (it drops the
        # exit joins the thread is parked on) rather than merely marking it
        # abandoned, so the thread normally finishes just after. Join it here
        # instead of returning immediately: drain runs inside a test (RPC
        # construction seam, opt-out setup, failure fence), so a thread that
        # terminates a moment later takes its transport threads with it across
        # the test boundary, and pytest-threadleak charges the leak to whatever
        # test happens to be running next.
        release_deadline = time.monotonic() + min(max(timeout, 1.0), 30.0)
        for _, t in still_alive:
            t.join(timeout=max(0.0, release_deadline - time.monotonic()))

        leaked = [pool for pool, thread in still_alive if thread.is_alive()]
        if leaked:
            print(
                f"[session-reuse] WARNING: {len(leaked)} pool shutdown thread(s) "
                "remain after worker termination",
                flush=True,
            )
        else:
            print(f"[session-reuse] drained {len(pools)} cached pool(s)", flush=True)


REUSE = SessionReuseCache()
