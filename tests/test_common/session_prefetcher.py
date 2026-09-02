# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background prefetch of the NEXT test's MPI session — zero test changes.

Multi-GPU LLM-API tests pay ~50-65s per bare ``LLM(...)`` to spawn an MPI
pool whose workers import ``tensorrt_llm``. That is pure CPU/IO work, so
while the CURRENT test runs on the GPUs a spare pool for the next test can
be spawned in a background thread — hiding the spawn cost behind the
previous test's runtime. Prefetched workers run no kernels and allocate
nothing before handover; depending on the library version, importing
tensorrt_llm may leave an idle CUDA context (~a few hundred MB), which
safely coexists with the running test.

Mechanism (wired by ``tests/test_common/session_prefetcher_hooks.py``,
loaded from each test tree's top-level conftest): ``pytest_runtest_setup``
lazily patches the library seams that construct ``MpiPoolSession`` for a
bare ``LLM(...)`` with a factory that (a) hands over the prefetched pool
when its size and spawn-time env/sys.path still match, and (b) re-arms a
spare pool of the same size for the next test. A miss falls back to the
normal synchronous spawn, so a wrong prefetch can only cost time, never
correctness.

Weight page-cache warming: when the NEXT test's model differs from the
current one, its weight files are read in a background thread so the kernel
page cache is hot by the time that test loads weights. The next model is
discovered automatically from the accuracy-harness ``MODEL_PATH`` class
attribute or a ``model_folder``-style test parameter (modeling unit tests),
or declared explicitly with
``@pytest.mark.prefetch_model_dir("/path/to/model")``. This complements
pool prefetch (pool reuse does not cover model IO). Page cache is
reclaimable memory, so warming cannot OOM the host; a wasted warm (test
skipped or reordered) costs only IO bandwidth.

Coexistence with MPI session reuse: when ``test_common/session_reuse.py`` is
wired and enabled it owns the same pool-creation seams and eliminates the
respawn outright, so the prefetcher automatically stays off the seams (see
``_reuse_layer_active``). Weight warming stays active, and reuse consumes
this layer's shadow pools on its cache misses (first pool of a size,
post-drain rebuild, post-retire replacement) via ``take``/``schedule_shadow``
— the two layers compose: reuse covers the steady state, prefetch covers the
misses.

Enabled by default; ``TRTLLM_TEST_PREFETCH_SESSION=0`` disables BOTH pool
prefetch and weight warming (one kill switch for the whole plugin). Suites
that never import tensorrt_llm's executor modules pay nothing — not even
the tensorrt_llm import.
"""

import glob
import math
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

# The spawn snapshot is shared with the session-reuse layer (both hand a
# live pool to a test that did not spawn it — same invariant: workers freeze
# the FULL env + sys.path at spawn).
from test_common._session_utils import _isinstance_transparent_shim, _spawn_snapshot

# The only places in the library that construct MpiPoolSession for a bare
# LLM(...); tests passing their own _mpi_session never reach these lines.
# test_patch_targets_cover_all_library_construction_sites keeps this list
# honest against new construction sites appearing in the library.
_PATCH_TARGETS = (
    "tensorrt_llm.executor.proxy",
    "tensorrt_llm.executor.rpc_proxy",
    "tensorrt_llm.llmapi.llm",
)

# Identity collection is followed by one lightweight diagnostic submitted to
# the workers. Keep enough room for that hand-off after the lower-level
# bootstrap deadline expires.
_SHADOW_BUILD_FINISH_GRACE = 30.0
_FALLBACK_IDENTITY_TIMEOUT = 300.0
# Mirrors mpi_session._IDENTITY_BARRIER_TIMEOUT: the identity gate spends its
# bootstrap budget first and only then runs the barrier, so the fallback total
# has to include both phases.
_FALLBACK_BARRIER_TIMEOUT = 60.0


def _fallback_identity_timeout() -> float:
    """Mirror the lower-level env contract while its module is still loading."""
    raw = os.environ.get("TRTLLM_MPI_IDENTITY_TIMEOUT")
    if raw:
        try:
            value = float(raw)
            if math.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
    return _FALLBACK_IDENTITY_TIMEOUT


def _shadow_build_wait_timeout() -> float:
    """Upper-level wait budget derived from the MPI identity-gate budget.

    Do not import TensorRT-LLM here: this plugin must stay usable by pure-logic
    tests and suites without built bindings. A real shadow build imports
    ``mpi_session`` before it can construct a pool, so the live lower-level
    setting is present by the time ``take()`` waits on that build.
    """
    mpi_session = sys.modules.get("tensorrt_llm.llmapi.mpi_session")
    budget_fn = getattr(mpi_session, "identity_gate_budget", None)
    if budget_fn is not None:
        gate_budget = budget_fn()
    else:
        gate_budget = _fallback_identity_timeout() + _FALLBACK_BARRIER_TIMEOUT
    return gate_budget + _SHADOW_BUILD_FINISH_GRACE


def _reuse_layer_active() -> bool:
    """True when the MPI session-reuse layer owns the pool-creation seams.

    ``test_common.session_reuse`` keeps pools alive across tests at the SAME
    seams this module would patch, and saves the whole respawn rather than
    just hiding it — strictly better where it applies. When it is wired and
    enabled, the prefetcher must stay off the seams so the two factories
    don't fight over them (whoever patches first would silently disable the
    other). Weight page-cache warming is orthogonal and stays on either way.
    """
    mod = sys.modules.get("test_common.session_reuse")
    if mod is None:
        return False  # not wired into this suite: seams are ours to patch
    try:
        return bool(mod.REUSE.is_active())
    except Exception:
        return True  # loaded but unreadable: err on staying out of the way


_READ_CHUNK = 64 << 20  # 64MB


def _weight_files(model_dir: str):
    """The weight files the loader will actually read, in loader order.

    Mirrors HfWeightLoader.load_weights' selection: safetensors first —
    minus "consolidated" copies, which the loader deliberately skips (they
    duplicate the shards and can be enormous) — else *.bin, else *.pth.
    Warming anything else is pure wasted IO.
    """
    files = [
        f
        for f in glob.glob(os.path.join(model_dir, "*.safetensors"))
        if "consolidated" not in os.path.basename(f)
    ]
    for fallback in ("*.bin", "*.pth"):
        if files:
            break
        files = glob.glob(os.path.join(model_dir, fallback))
    return sorted(files)


def _available_host_memory():
    """MemAvailable from /proc/meminfo in bytes, or None when unreadable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


# Parametrized-test convention: the model lives in a parameter with one of
# these names (e.g. test_modeling_* files), holding either an absolute path
# or a directory name under LLM_MODELS_ROOT.
_MODEL_PARAM_NAMES = ("model_folder", "model_dir", "model_path")


def _models_root():
    """The models root the tests themselves resolve against, or None."""
    root = os.environ.get("LLM_MODELS_ROOT")
    if root:
        return root
    try:  # same fallback the test suites use (CI default scratch path)
        from test_common.llm_data import llm_models_root

        root = llm_models_root()
        return str(root) if root else None
    except Exception:
        return None


def _model_dir_of(item):
    """A test item's model dir: marker, else class or parameter convention.

    Discovery order: the explicit ``prefetch_model_dir`` marker; the accuracy
    harness's ``MODEL_PATH`` class attribute (120+ classes across
    tests/integration/defs/accuracy); a ``model_folder``-style test parameter
    (modeling unit tests), resolved under LLM_MODELS_ROOT unless absolute.
    All of it is guess-tolerant: a value that is not a real directory of
    weight files (e.g. an HF model id) makes ``warm_page_cache`` a silent
    no-op, so a wrong guess costs nothing.
    """
    marker = item.get_closest_marker("prefetch_model_dir")
    if marker is not None and marker.args:
        return marker.args[0]
    model_path = getattr(getattr(item, "cls", None), "MODEL_PATH", None)
    if isinstance(model_path, str):
        return model_path
    params = getattr(getattr(item, "callspec", None), "params", None) or {}
    for name in _MODEL_PARAM_NAMES:
        value = params.get(name)
        if isinstance(value, str) and value:
            if os.path.isabs(value):
                return value
            root = _models_root()
            if root:
                return os.path.join(root, value)
    return None


def warm_page_cache(model_dir: str) -> float:
    """Read ``model_dir``'s weight files to keep them in the OS page cache.

    The next LLM create then loads the weights from RAM, not disk. Pure file
    IO — never touches CUDA, safe to run while another test owns the GPUs.
    Returns the number of GiB read.
    """
    files = _weight_files(model_dir)
    if not files:
        return 0.0  # not a local weight dir (e.g. an HF model id): nothing to warm
    total_bytes = sum(os.stat(f).st_size for f in files)
    available = _available_host_memory()
    if available is not None and total_bytes > available:
        # Larger than RAM: pages would be evicted before the test loads them —
        # pure filer traffic with zero benefit (e.g. multi-hundred-GB models).
        print(
            f"[session-prefetch] skipping warm of {model_dir}: {total_bytes >> 30} GiB "
            f"exceeds available host memory ({available >> 30} GiB)",
            flush=True,
        )
        return 0.0
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


def _worker_import_report_cuda() -> bool:
    """Import tensorrt_llm (the expensive part) and report CUDA state.

    Some library versions initialize a CUDA context at import time; that
    idle context (~a few hundred MB, no kernels/allocations) is acceptable
    and coexists with the running test, so it is reported, not asserted.
    """
    import torch

    import tensorrt_llm  # noqa: F401

    return torch.cuda.is_initialized()


class _Built(NamedTuple):
    """A finished background build: everything published (and consumed) together."""

    spec: int
    session: object
    snapshot: object


def _prefetched_workers_alive(session: object) -> bool:
    """Whether every recorded worker is still the process spawned for this pool.

    ``MpiPoolSession(wait_shutdown=True)`` records a complete set of
    ``(pid, start_time)`` identities before a pool can be published. Checking
    both values rejects workers that exited after publication without
    mistaking a recycled PID for the original process.
    """
    identities = getattr(session, "_worker_identities", ())
    if len(identities) != getattr(session, "n_workers", None):
        return False
    mpi_session = sys.modules.get("tensorrt_llm.llmapi.mpi_session")
    process_start_time = getattr(mpi_session, "_process_start_time", None)
    if process_start_time is None:
        return False
    return all(
        start_time is not None and process_start_time(pid) == start_time
        for pid, start_time in identities
    )


def _reap_dead_pool(session: object) -> None:
    """Abandon a part-dead pool and SIGKILL whichever workers are still up.

    ``abandon()`` only disconnects the parent side. With a rank already gone the
    manager thread stays wedged in MPI, so the survivors are never told to exit
    and keep their GPU memory while the replacement pool spawns on top of them.
    Mirrors ``MpiPoolSession._teardown_unidentified_pool``, recycling guard
    included.
    """
    import signal

    try:
        session.abandon()
    except Exception as exc:  # noqa: BLE001
        # Broad on purpose: this pool is already known part-dead, so abandon()
        # reaches into mpi4py with a rank missing and can fail in ways MPI does
        # not enumerate. Swallowing is required -- the SIGKILL sweep below is
        # what actually reclaims the survivors' GPU memory, and it must run
        # whatever abandon() did. Surface it instead of passing silently.
        print(
            f"[session-prefetch] WARNING: abandon() failed on a part-dead pool: {exc!r}",
            flush=True,
        )
    mpi_session = sys.modules.get("tensorrt_llm.llmapi.mpi_session")
    process_start_time = getattr(mpi_session, "_process_start_time", None)
    if process_start_time is None:
        return
    for pid, start_time in getattr(session, "_worker_identities", ()):
        if start_time is None or process_start_time(pid) != start_time:
            continue  # already gone, or the PID was recycled
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class SessionPrefetcher:
    def __init__(self):
        self._lock = threading.Lock()
        self._drain_lock = threading.Lock()
        self._thread = None
        self._building_spec = None  # spec of the in-flight build, while _thread is set
        self._build_timed_out = False
        self._build_gen = 0  # bumped when a pending build is abandoned
        self._built = None  # Optional[_Built], set only by _publish()
        self._patched = set()
        self._next_model = None  # item -> next model dir; built lazily
        self._warmed_dirs = set()
        self._disposed = False
        # Activity counters, reported once per session by dispose(). pytest
        # captures per-test stdout (swallowing the per-event prints for
        # passing tests), but pytest_sessionfinish runs OUTSIDE capture, so
        # the summary is the one line guaranteed to reach the CI console.
        self.stats = Counter()
        self._warmed_gib = 0.0

    @property
    def enabled(self) -> bool:
        # Under pytest-xdist every worker sees the FULL collection but runs a
        # scheduler-assigned subset, and N workers would each hold a live
        # pool plus a spare. Disable in xdist workers.
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return False
        return os.environ.get("TRTLLM_TEST_PREFETCH_SESSION", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @staticmethod
    def _next_model_map(items):
        """Per item, the model dir of the NEXT test declaring one.

        One reverse pass, one lookup per item — O(n) once; ``on_test_setup``
        then costs a single dict lookup per test. (The naive alternative —
        scanning the remaining collection at every test setup — is O(n^2)
        lookups per session, seconds to minutes on large suites.)
        """
        next_model, mapping = None, {}
        for item in reversed(items):
            mapping[item] = next_model
            model = _model_dir_of(item)
            if model is not None:
                next_model = model
        return mapping

    def on_test_setup(self, item) -> None:
        """Warm the NEXT test's model weights while this test runs.

        Fires when the next model differs from the current one — including
        between tests that share a pool (pool prefetch does not cover model
        IO). The next-model map is built lazily on the FIRST test setup, from
        ``session.items``: by then every reordering/deselecting plugin
        (pytest-split runs trylast, --test-list filtering, -k/-m) has produced
        the final run order, which a ``pytest_collection_modifyitems`` hook
        could not guarantee.
        """
        if self._next_model is None:
            items = getattr(item.session, "items", None) or [item]
            self._next_model = self._next_model_map(items) if self.enabled else {}
        nxt = self._next_model.get(item)
        if not nxt or nxt == _model_dir_of(item):
            return
        # Main-pytest-thread only (pytest_runtest_setup): no lock needed.
        if nxt in self._warmed_dirs:
            return  # already warmed (or being warmed) this session
        self._warmed_dirs.add(nxt)
        threading.Thread(
            target=self._warm, args=(nxt,), daemon=True, name="session-prefetch-warm"
        ).start()

    def _warm(self, model_dir: str) -> None:
        try:
            gib = warm_page_cache(model_dir)
            with self._lock:
                if gib > 0:
                    self.stats["warms"] += 1
                    self._warmed_gib += gib
                else:
                    self.stats["warm_noops"] += 1  # no local weights / RAM guard
        except Exception as e:  # warming must never break the tests
            print(f"[session-prefetch] page-cache warm failed (harmless): {e}", flush=True)

    def schedule_shadow(self, spec: int, env_overlay=None) -> None:
        """Start building a spare ``spec``-worker pool in the background.

        Heuristic: the next test most likely needs a pool of the same size as
        the current one. A mismatched in-flight build is drained before a
        synchronous miss to preserve allocation-wide single-flight. This can
        add latency when the size prediction is wrong, but avoids two MPI
        bootstraps contending on the same allocation.

        ``env_overlay``: extra env vars to freeze into the WORKERS at spawn
        (session_reuse restocks shadows with its worker-side weight cache
        on). Passed through the library's worker-env channel, so the parent
        process environment — and therefore the take()-time snapshot
        comparison — is never touched.
        """
        if not self.enabled or spec < 1:
            return
        with self._lock:
            if self._thread is not None or (self._built is not None and self._built.spec == spec):
                return  # already building / built
            self._building_spec = spec
            self._build_timed_out = False
            self._thread = threading.Thread(
                target=self._build,
                args=(spec, self._build_gen, env_overlay),
                daemon=True,
                name="session-prefetch-build",
            )
            self._thread.start()

    def _build(self, spec: int, gen: int, env_overlay=None) -> None:
        try:
            from tensorrt_llm._utils import mpi_disabled

            if mpi_disabled():
                return
            from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

            snapshot = _spawn_snapshot()  # workers freeze env+sys.path at spawn
            # wait_shutdown: see _make_factory — every pool this layer hands
            # out blocks its shutdown on actual worker exit.
            session = MpiPoolSession(n_workers=spec, wait_shutdown=True, env_overrides=env_overlay)
            if any(session.submit_sync(_worker_import_report_cuda)):
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

        ``_drain()`` bumps ``_build_gen`` when a build outlives its join
        timeout; such a late build must shut its pool down instead of
        publishing (a late publish would overwrite — and leak — a newer pool,
        or hand a stale pool to a future test). The empty-slot check likewise
        prevents overwriting an unconsumed pool.
        """
        with self._lock:
            if gen == self._build_gen and self._built is None:
                self._built = _Built(spec, session, snapshot)
                self.stats["pools_built"] += 1
                return
            self.stats["pools_discarded_superseded"] += 1
        print("[session-prefetch] discarding superseded background build", flush=True)
        session.shutdown()

    def _drain(self, timeout: float | None = None) -> _Built | None:
        """Join a pending build and pop the completed shadow slot.

        A live build is kept registered and marked terminal on timeout.
        Callers fail closed instead of starting a second pool while the first
        bootstrap is still running; later calls fail immediately until the
        thread exits, then clear the terminal state.
        """
        # Serialize drains so concurrent LLM construction cannot make multiple
        # callers wait through the full deadline before one records the
        # terminal timeout.
        with self._drain_lock:
            # Read _thread under the lock: schedule_shadow() assigns-then-starts
            # inside its critical section, and an unlocked read here can observe
            # the assigned-but-not-yet-started thread ("cannot join thread before
            # it is started" when a test creates LLMs concurrently).
            with self._lock:
                thread = self._thread
                build_timed_out = self._build_timed_out
            if thread is not None:
                if build_timed_out and thread.is_alive():
                    raise TimeoutError(
                        "session-prefetch shadow build previously timed out and "
                        "is still running; refusing to start a concurrent MPI pool"
                    )
                if not build_timed_out:
                    if timeout is None:
                        timeout = _shadow_build_wait_timeout()
                    thread.join(timeout=timeout)
            with self._lock:
                if thread is not None and thread.is_alive():
                    # Invalidate a late publish but retain _thread so
                    # schedule_shadow() cannot start another build alongside it.
                    if not self._build_timed_out:
                        self._build_timed_out = True
                        self._build_gen += 1
                        self.stats["pool_build_timeouts"] += 1
                    raise TimeoutError(
                        "session-prefetch shadow build did not finish within "
                        f"{timeout}s; refusing to start a concurrent MPI pool"
                    )
                self._thread = None
                self._building_spec = None
                self._build_timed_out = False
                built, self._built = self._built, None
            return built

    def take(self, spec: int) -> object | None:
        """Return a prefetched session for ``spec``, or None to build sync."""
        if not self.enabled:
            return None
        # The upper-level deadline is derived from the identity barrier's
        # bootstrap deadline plus a small post-bootstrap diagnostic grace.
        # It must never expire while the lower layer still considers the
        # in-flight pool healthy. A wrong-size build is also drained before
        # returning a miss: starting the requested size alongside it would
        # recreate the same concurrent-bootstrap contention.
        built = self._drain()
        if built is None:
            return None
        if not _prefetched_workers_alive(built.session):
            # A worker may die after the background build publishes its pool.
            # Never hand that unusable executor to the next test: its queued
            # initialization task would have no worker and can wait until the
            # outer pytest timeout. The dead world cannot be joined, so
            # abandon it before the synchronous fallback.
            self.stats["pools_discarded_dead"] += 1
            print(
                "[session-prefetch] discarding prefetched pool with dead worker",
                flush=True,
            )
            threading.Thread(
                target=_reap_dead_pool,
                args=(built.session,),
                daemon=True,
                name="session-prefetch-discard-dead",
            ).start()
            return None
        if built.spec == spec and built.snapshot == _spawn_snapshot():
            # An instant handover is safe against the previous worker's GPU
            # memory: every pool these layers hand out is built with
            # wait_shutdown=True, so its shutdown() blocked until the workers
            # actually exited (and released their memory).
            self.stats["pools_handed_over"] += 1
            print(f"[session-prefetch] handing over prefetched {spec}-worker pool", flush=True)
            return built.session
        # Spec/env/sys.path mismatch (test skipped, reordered, or changed
        # state the frozen workers would not see): discard.
        self.stats["pools_discarded_stale"] += 1
        threading.Thread(
            target=built.session.shutdown, daemon=True, name="session-prefetch-discard"
        ).start()
        return None

    def _make_factory(self, real_cls):
        """A drop-in for ``MpiPoolSession`` that consumes and re-arms the shadow."""

        def factory(n_workers, *args, **kwargs):
            if args or kwargs:
                return real_cls(n_workers, *args, **kwargs)
            # n_workers == 1 included: the default single-GPU path also spawns
            # a 1-worker pool (executor.py -> proxy.py) costing ~50s of
            # spawn+import, the same as multi-GPU pools.
            # wait_shutdown: this pool's shutdown must not return until its
            # workers exited (and released GPU memory) — the NEXT pool is
            # handed over instantly, without the ~50s sync spawn that used to
            # hide the release window. Such spawns fail closed when identity
            # collection cannot complete. Do not immediately retry or degrade
            # to a plain pool: unidentified workers may still be exiting, and
            # wait_shutdown=False cannot protect the next handover.
            session = self.take(n_workers)
            if session is None:
                session = real_cls(n_workers=n_workers, wait_shutdown=True)
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
        if len(self._patched) == len(_PATCH_TARGETS):
            return  # everything already patched: per-test fast path
        if _reuse_layer_active():
            # session_reuse owns the seams: skip MPI-pool prefetch entirely
            # (reuse eliminates the respawn; prefetch could only hide it).
            self.stats["mpi_yielded_to_reuse"] = 1
            return
        pending = [n for n in _PATCH_TARGETS if n in sys.modules and n not in self._patched]
        if not pending:
            return
        from tensorrt_llm.llmapi.mpi_session import MpiPoolSession as real_cls

        factory = self._make_factory(real_cls)
        for name in pending:
            mod = sys.modules[name]
            if getattr(mod, "MpiPoolSession", None) is real_cls:
                # A real type, not a bare function: library code may run
                # isinstance(x, MpiPoolSession) against the patched attribute
                # (proxy.py's killed-worker detection did) — a function there
                # raises TypeError and kills every LLM creation.
                mod.MpiPoolSession = _isinstance_transparent_shim(real_cls, factory)
            self._patched.add(name)

    def dispose(self) -> None:
        """Shut down any unconsumed shadow pool (end-of-session cleanup).

        Uses the same coordinated deadline as ``take()``. Ending the pytest
        session is not permission to abandon a bootstrap while the lower
        layer still considers it healthy. Idempotent: a repository-root run
        dispatches sessionfinish from both the repo-root and the subtree
        conftest.
        """
        if self._disposed:
            return
        self._disposed = True
        try:
            built = self._drain()
        except TimeoutError as e:
            # Do not turn pytest_sessionfinish into an internal error. The
            # build remains registered, preventing another shadow from being
            # launched; the daemon thread is bounded by the lower-level
            # identity timeout unless the MPI runtime itself is wedged.
            print(f"[session-prefetch] cleanup timed out: {e}", flush=True)
            built = None
        if built is not None:
            built.session.shutdown()
        # One line per session, emitted OUTSIDE pytest's per-test capture
        # (pytest_sessionfinish) so it reaches the CI console: the per-event
        # prints above are swallowed for passing tests. Silent when the
        # prefetcher never did anything (non-LLM suites).
        if self.stats:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items()))
            if self._warmed_gib:
                parts += f", warmed_gib={self._warmed_gib:.1f}"
            print(f"[session-prefetch] session summary: {parts}", flush=True)


PREFETCHER = SessionPrefetcher()
