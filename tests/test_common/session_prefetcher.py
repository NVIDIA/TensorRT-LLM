# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in background prefetch of the NEXT test's session and model page cache.

Multi-GPU LLM-API tests pay ~50-65s per test to spawn an MPI pool whose
workers import ``tensorrt_llm``, plus a cold read of the model weights. Both
are pure CPU/IO work, so while the CURRENT test is running on the GPUs the
next test's pool can be spawned and its weight files pre-read in background
threads — hiding those costs behind the previous test's runtime. Prefetched
workers run no kernels and allocate nothing before handover; depending on
the library version, importing tensorrt_llm may leave an idle CUDA context
(~a few hundred MB), which safely coexists with the running test.

Enabled by default; set ``TRTLLM_TEST_PREFETCH_SESSION=0`` to disable. With
no test declaring a spec/model marker the prefetcher never fires, so plain
suites are unaffected either way.

Wiring (see tests/unittest/llmapi/conftest.py):
- ``pytest_collection_modifyitems`` -> ``PREFETCHER.on_collection(items)``
- ``pytest_runtest_setup``          -> ``PREFETCHER.on_test_setup(item)``
- session-scoped fixtures call ``PREFETCHER.take(n_workers)`` and fall back
  to building synchronously when it returns ``None``.

Tests declare their session spec with ``@pytest.mark.prefetch_session(N)``
(or a class attribute ``n_gpus``). Consecutive tests with the SAME spec are
assumed to share one session (that is what makes the fixture reuse safe), so
prefetch only fires at a spec boundary.

Weight page-cache warming (opt-in per test): mark tests with
``@pytest.mark.prefetch_model_dir("/path/to/model")``; when the NEXT test's
model differs from the current one, its weight files are read in a background
thread so the kernel page cache is hot by the time that test loads weights.
This fires even between tests that share a session (pool reuse does not
cover model IO). Page cache is reclaimable memory, so warming cannot OOM the
host; a wasted warm (test skipped or reordered) costs only IO bandwidth.
"""

import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# The only places in the library that construct MpiPoolSession for a bare
# LLM(...); tests passing their own _mpi_session never reach these lines.
_PATCH_TARGETS = (
    "tensorrt_llm.executor.proxy",
    "tensorrt_llm.executor.rpc_proxy",
    "tensorrt_llm.llmapi.llm",
)

_WEIGHT_GLOBS = ("*.safetensors", "*.bin")
_READ_CHUNK = 64 << 20  # 64MB

# MpiPoolSession workers freeze their environment AND sys.path at spawn time
# (they inherit the whole parent env at MPI spawn, plus MPIPoolExecutor's
# explicit TRTLLM*/TLLM* overrides and path=sys.path), so a pool prefetched
# during test A must not be handed to test B if B changed ANY env var (proven
# silent-failure class: OVERRIDE_QUANT_ALGO and other non-prefixed test knobs
# are read inside workers) or prepended to sys.path (proven hard-failure
# class: test_modeling_out_of_tree monkeypatches sys.path before LLM(), CI
# build 46175 "Executor worker died during initialization" on 4 platforms).
# The snapshot therefore covers the FULL environment, minus process
# bookkeeping that legitimately drifts between tests without affecting
# workers. A false mismatch only costs a synchronous rebuild.
_ENV_IGNORE = frozenset(
    {
        "PYTEST_CURRENT_TEST",  # changes every test phase by design
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


def _spec_of(item):
    """A test item's session spec: ``prefetch_session`` marker or class ``n_gpus``."""
    marker = item.get_closest_marker("prefetch_session")
    if marker is not None and marker.args:
        return marker.args[0]
    return getattr(getattr(item, "cls", None), "n_gpus", None)


def _model_dir_of(item):
    """A test item's model dir from its ``prefetch_model_dir`` marker, or None."""
    marker = item.get_closest_marker("prefetch_model_dir")
    return marker.args[0] if marker is not None and marker.args else None


def _worker_import_report_cuda() -> bool:
    """Import tensorrt_llm (the expensive part) and report CUDA state.

    Some library versions initialize a CUDA context at import time; that
    idle context (~a few hundred MB, no kernels/allocations) is acceptable
    and coexists with the running test, so it is reported, not asserted.
    """
    import torch

    import tensorrt_llm  # noqa: F401

    return torch.cuda.is_initialized()


# GPU-memory settle barrier at handover (see wait_gpu_memory_settle).
_SETTLE_MIN_FREE_FRAC = 0.85  # "GPU is essentially free" — stop waiting
_SETTLE_POLL_S = 0.5
_SETTLE_FLAT_POLLS = 3  # consecutive non-increasing polls => memory is in use
_SETTLE_EPSILON = 256 << 20  # free-memory delta below this counts as flat
_SETTLE_TIMEOUT_S = 30.0


def _visible_gpu_indices(count: int):
    """NVML indices of the GPUs this process may touch (CUDA_VISIBLE_DEVICES)."""
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


def wait_gpu_memory_settle(timeout: float = _SETTLE_TIMEOUT_S) -> None:
    """Wait for a dying previous worker to release its GPU memory.

    A handed-over live pool skips the ~50s synchronous spawn that used to
    give the previous LLM's worker process time to exit: MPI pool shutdown
    returns at disconnect, but the child's CUDA memory is only released when
    the process actually exits. Building the next model into that race fails
    with "Executor creation failed due to insufficient GPU memory" (CI build
    46175: nemotron_h/nano_v2_vl/qwen3_lora, free 16-19 GiB at model load).

    Polls NVML (context-free) until every visible GPU is mostly free, or its
    free memory stops increasing (memory legitimately in use — e.g. another
    fixture's live LLM), or ``timeout``. Fast no-op on an already-free GPU.
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
            if time.monotonic() - t0 >= timeout:
                break
            prev = free
            time.sleep(_SETTLE_POLL_S)
        waited = time.monotonic() - t0
        if waited >= _SETTLE_POLL_S:
            print(
                f"[session-prefetch] waited {waited:.1f}s before handover for GPU "
                f"memory release (free {min(f / t for f, t in zip(free, total)):.0%})",
                flush=True,
            )
    except Exception as e:
        print(f"[session-prefetch] GPU settle check skipped: {e}", flush=True)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def warm_page_cache(model_dir: str) -> float:
    """Read ``model_dir``'s weight files to keep them in the OS page cache.

    The next LLM create then loads the weights from RAM, not disk. Pure file
    IO — never touches CUDA, safe to run while another test owns the GPUs.
    Returns the number of GiB read.
    """
    files = sorted(f for pat in _WEIGHT_GLOBS for f in glob.glob(os.path.join(model_dir, pat)))
    t0 = time.monotonic()

    def _read(path):
        n = 0
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_READ_CHUNK)
                if not chunk:
                    return n
                n += len(chunk)

    # thread_name_prefix keeps the IO workers inside the pytest.ini
    # threadleak_exclude pattern (session-prefetch-\w+): a large warm can
    # legitimately still be reading during the next test's threadleak check.
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="session-prefetch-io") as ex:
        total = sum(ex.map(_read, files))
    gib = total / (1 << 30)
    print(
        f"[session-prefetch] warmed page cache: {gib:.1f} GiB from "
        f"{model_dir} in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return gib


class SessionPrefetcher:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._build_gen = 0  # bumped when a pending build is abandoned
        self._built_spec = None
        self._built_session = None
        self._built_snapshot = None
        self._items = []
        self._warmed_dirs = set()
        self._patched = set()

    @property
    def enabled(self) -> bool:
        # Under pytest-xdist every worker sees the FULL collection but runs a
        # scheduler-assigned subset: collection-order lookahead would prefetch
        # pools/models for tests that run in other workers, and N workers
        # would each hold a live pool plus a spare. Disable in xdist workers.
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return False
        return os.environ.get("TRTLLM_TEST_PREFETCH_SESSION", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def on_collection(self, items) -> None:
        if self.enabled:
            self._items = list(items)

    def on_test_setup(self, item) -> None:
        """Prefetch the next block's pool / next model's weights.

        If ``item`` is the last test of its session block, start building the
        next block's pool in a background thread. Independently, if the next
        test declaring a model uses a DIFFERENT model than ``item``, start
        warming that model's page cache.
        """
        if not self.enabled or not self._items:
            return
        try:
            idx = self._items.index(item)
        except ValueError:
            return

        # Model-weight warming: independent of the pool trigger, so it also
        # fires between tests that share a session but switch models.
        cur_model = _model_dir_of(item)
        nxt_model = next(
            (_model_dir_of(it) for it in self._items[idx + 1 :] if _model_dir_of(it) is not None),
            None,
        )
        if nxt_model and nxt_model != cur_model:
            with self._lock:
                start_warm = nxt_model not in self._warmed_dirs
                self._warmed_dirs.add(nxt_model)
            if start_warm:
                threading.Thread(
                    target=self._warm,
                    args=(nxt_model,),
                    daemon=True,
                    name="session-prefetch-warm",
                ).start()

        cur = _spec_of(item)
        nxt = next(
            (_spec_of(it) for it in self._items[idx + 1 :] if _spec_of(it) is not None), None
        )
        if nxt is None or nxt == cur:
            return
        with self._lock:
            if self._thread is not None or self._built_spec == nxt:
                return  # already building / built
            self._thread = threading.Thread(
                target=self._build,
                args=(nxt, self._build_gen),
                daemon=True,
                name="session-prefetch-build",
            )
            self._thread.start()

    def _warm(self, model_dir: str) -> None:
        try:
            warm_page_cache(model_dir)
        except Exception as e:  # warming must never break the tests
            print(f"[session-prefetch] page-cache warm failed (harmless): {e}", flush=True)

    def _build(self, spec: int, gen: int) -> None:
        try:
            from tensorrt_llm._utils import mpi_disabled

            if mpi_disabled():
                return
            from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

            snapshot = _spawn_snapshot()  # workers freeze env+sys.path at spawn
            session = MpiPoolSession(n_workers=spec)
            cuda_state = session.submit_sync(_worker_import_report_cuda)
            if any(cuda_state) if isinstance(cuda_state, (list, tuple)) else cuda_state:
                print(
                    "[session-prefetch] note: tensorrt_llm import initialized an idle "
                    "CUDA context in the prefetched workers (library version behavior)",
                    flush=True,
                )
            self._publish(spec, session, snapshot, gen)
        except Exception as e:  # prefetch must never break the tests
            print(
                f"[session-prefetch] background build failed (falling back to synchronous): {e}",
                flush=True,
            )

    def _publish(self, spec, session, snapshot, gen: int) -> None:
        """Publish a finished background build, unless it was abandoned.

        ``take()``/``dispose()`` bump ``_build_gen`` when a build outlives its
        join timeout; such a late build must shut its pool down instead of
        publishing (a late publish would overwrite — and leak — a newer pool,
        or hand a stale pool to a future test). The empty-slot check likewise
        prevents overwriting an unconsumed pool.
        """
        with self._lock:
            if gen == self._build_gen and self._built_session is None:
                self._built_spec = spec
                self._built_session = session
                self._built_snapshot = snapshot
                return
        print("[session-prefetch] discarding abandoned background build", flush=True)
        session.shutdown()

    def take(self, spec: int):
        """Return a prefetched session for ``spec``, or None to build sync."""
        if not self.enabled:
            return None
        # Read _thread under the lock: schedule_shadow() assigns-then-starts
        # inside its critical section, and an unlocked read here can observe
        # the assigned-but-not-yet-started thread ("cannot join thread before
        # it is started", CI build 46175, tests creating LLMs concurrently).
        with self._lock:
            thread = self._thread
        if thread is not None:
            # Slowest legitimate build measured is ~117s (busy node); 180s
            # gives 1.5x margin. On a genuine hang we give up and fall back
            # to a synchronous build instead of stalling the suite.
            thread.join(timeout=180)
        session = None
        with self._lock:
            if thread is not None and thread.is_alive():
                # Abandon the overdue build: bump the generation so its late
                # _publish() shuts the pool down instead of landing.
                self._build_gen += 1
            self._thread = None
            if (
                self._built_spec == spec
                and self._built_session is not None
                and self._built_snapshot == _spawn_snapshot()
            ):
                session, self._built_session, self._built_spec = (self._built_session, None, None)
            # Spec/env/sys.path mismatch (test skipped, reordered, or changed
            # state the frozen workers would not see): discard, build sync.
            elif self._built_session is not None:
                stale, self._built_session, self._built_spec = (self._built_session, None, None)
                threading.Thread(
                    target=stale.shutdown, daemon=True, name="session-prefetch-discard"
                ).start()
        if session is not None:
            # An instant handover skips the ~50s synchronous spawn that used
            # to give the PREVIOUS LLM's worker time to exit and release its
            # GPU memory; don't start the next model build into that race.
            wait_gpu_memory_settle()
            print(f"[session-prefetch] handing over prefetched {spec}-worker pool", flush=True)
        return session

    # ---- shadow mode: zero-test-change prefetch at the pool-creation seam ----

    def schedule_shadow(self, spec: int) -> None:
        """Start building a spare ``spec``-worker pool in the background.

        Heuristic: the next test most likely needs a pool of the same size as
        the current one. A miss is discarded at ``take()`` and the sync build
        is no slower than without prefetch.
        """
        if not self.enabled or spec < 1:
            return
        with self._lock:
            if self._thread is not None or self._built_spec == spec:
                return  # already building / built
            self._thread = threading.Thread(
                target=self._build,
                args=(spec, self._build_gen),
                daemon=True,
                name="session-prefetch-build",
            )
            self._thread.start()

    def _make_factory(self, real_cls):
        """A drop-in for ``MpiPoolSession`` that consumes and re-arms the shadow."""

        def factory(n_workers, *args, **kwargs):
            if args or kwargs:
                return real_cls(n_workers, *args, **kwargs)
            # n_workers == 1 included: the default single-GPU path also spawns
            # a 1-worker pool (executor.py -> proxy.py) costing ~50s of
            # spawn+import, the same as multi-GPU pools.
            session = self.take(n_workers) or real_cls(n_workers=n_workers)
            self.schedule_shadow(n_workers)  # re-arm for the NEXT test
            return session

        return factory

    def install_pool_factory_if_loaded(self) -> None:
        """Lazily patch the pool-creation seams for zero-test-change prefetch.

        Only patches target modules ALREADY imported by the test suite, so
        suites that never touch tensorrt_llm pay nothing (not even the
        import). Idempotent — called from ``pytest_runtest_setup``.
        Only the ``mpi_session is None`` branches construct ``MpiPoolSession``
        directly, so tests passing their own session (shared/grouped pools)
        are never intercepted.
        """
        if not self.enabled:
            return
        pending = [n for n in _PATCH_TARGETS if n in sys.modules and n not in self._patched]
        if not pending:
            return
        from tensorrt_llm.llmapi.mpi_session import MpiPoolSession as real_cls

        factory = self._make_factory(real_cls)
        for name in pending:
            mod = sys.modules[name]
            if getattr(mod, "MpiPoolSession", None) is real_cls:
                mod.MpiPoolSession = factory
            self._patched.add(name)

    def dispose(self) -> None:
        """Shut down any unconsumed shadow pool (end-of-session cleanup)."""
        with self._lock:  # same assigned-but-unstarted hazard as take()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=60)
        with self._lock:
            if thread is not None and thread.is_alive():
                self._build_gen += 1  # abandon: late publish must not land
            self._thread = None
            stale, self._built_session, self._built_spec = (self._built_session, None, None)
        if stale is not None:
            stale.shutdown()


PREFETCHER = SessionPrefetcher()
