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
attribute, or declared explicitly with
``@pytest.mark.prefetch_model_dir("/path/to/model")``. This complements
pool prefetch (pool reuse does not cover model IO). Page cache is
reclaimable memory, so warming cannot OOM the host; a wasted warm (test
skipped or reordered) costs only IO bandwidth.

Enabled by default; set ``TRTLLM_TEST_PREFETCH_SESSION=0`` to disable.
Suites that never import tensorrt_llm's executor modules pay nothing —
not even the tensorrt_llm import.
"""

import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

# The only places in the library that construct MpiPoolSession for a bare
# LLM(...); tests passing their own _mpi_session never reach these lines.
# test_patch_targets_cover_all_library_construction_sites keeps this list
# honest against new construction sites appearing in the library.
_PATCH_TARGETS = (
    "tensorrt_llm.executor.proxy",
    "tensorrt_llm.executor.rpc_proxy",
    "tensorrt_llm.llmapi.llm",
)

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


def _model_dir_of(item):
    """A test item's model dir: explicit marker, else the class convention.

    The accuracy-test harness declares the model as a ``MODEL_PATH`` class
    attribute (120+ classes across tests/integration/defs/accuracy), so those
    suites get weight warming automatically, with the marker as the explicit
    override for everything else. A value that is not a real directory of
    weight files makes ``warm_page_cache`` a silent no-op.
    """
    marker = item.get_closest_marker("prefetch_model_dir")
    if marker is not None and marker.args:
        return marker.args[0]
    model_path = getattr(getattr(item, "cls", None), "MODEL_PATH", None)
    return model_path if isinstance(model_path, str) else None


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


# GPU-memory settle barrier at handover (see wait_gpu_memory_settle).
_SETTLE_MIN_FREE_FRAC = 0.85  # "GPU is essentially free" — stop waiting
_SETTLE_POLL_S = 0.5
_SETTLE_FLAT_POLLS = 3  # consecutive non-increasing polls => memory is in use
_SETTLE_EPSILON = 256 << 20  # free-memory delta below this counts as flat
_SETTLE_TIMEOUT_S = 30.0


def _visible_gpu_handles(pynvml):
    """NVML handles of the GPUs this process may touch (CUDA_VISIBLE_DEVICES).

    Handles both the index form ("0,1") and the UUID form ("GPU-..."/"MIG-...")
    used on shared CI nodes — falling back to all devices there would make the
    settle barrier poll GPUs owned by other jobs. An explicitly EMPTY value
    means no GPUs are visible: nothing to wait for.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        return []
    count = pynvml.nvmlDeviceGetCount()
    if visible is None:
        return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    handles = []
    for token in visible.split(","):
        token = token.strip()
        try:
            if token.startswith(("GPU-", "MIG-")):
                handles.append(pynvml.nvmlDeviceGetHandleByUUID(token))
            else:
                index = int(token)
                if index >= count:
                    raise ValueError(token)
                handles.append(pynvml.nvmlDeviceGetHandleByIndex(index))
        except Exception:
            # Unparsable form: fall back to all GPUs (the caller's
            # flat-detection keeps a too-wide scan bounded).
            return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    return handles


def wait_gpu_memory_settle(timeout: float = _SETTLE_TIMEOUT_S) -> None:
    """Wait for a dying previous worker to release its GPU memory.

    A handed-over live pool skips the ~50s synchronous spawn that used to
    give the previous LLM's worker process time to exit: MPI pool shutdown
    returns at disconnect, but the child's CUDA memory is only released when
    the process actually exits. Building the next model into that race fails
    with "Executor creation failed due to insufficient GPU memory" (CI build
    46175: nemotron_h/nano_v2_vl/qwen3_lora, free 16-19 GiB at model load).

    TODO: the deeper fix is MpiPoolSession.shutdown(wait=True) not returning
    until the spawned worker processes have exited — that would close the
    race for every consumer (back-to-back LLM() creations included) and
    retire this heuristic. Library-behavior change, tracked separately.

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
        handles = _visible_gpu_handles(pynvml)
        if not handles:
            return

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


class _Built(NamedTuple):
    """A finished background build: everything published (and consumed) together."""

    spec: int
    session: object
    snapshot: object


class SessionPrefetcher:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._build_gen = 0  # bumped when a pending build is abandoned
        self._built = None  # Optional[_Built], set only by _publish()
        self._patched = set()
        self._next_model = None  # item -> next model dir; built lazily
        self._warmed_dirs = set()

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
            warm_page_cache(model_dir)
        except Exception as e:  # warming must never break the tests
            print(f"[session-prefetch] page-cache warm failed (harmless): {e}", flush=True)

    def schedule_shadow(self, spec: int) -> None:
        """Start building a spare ``spec``-worker pool in the background.

        Heuristic: the next test most likely needs a pool of the same size as
        the current one. A miss is discarded at ``take()`` and the sync build
        is no slower than without prefetch.
        """
        if not self.enabled or spec < 1:
            return
        with self._lock:
            if self._thread is not None or (self._built is not None and self._built.spec == spec):
                return  # already building / built
            self._thread = threading.Thread(
                target=self._build,
                args=(spec, self._build_gen),
                daemon=True,
                name="session-prefetch-build",
            )
            self._thread.start()

    def _build(self, spec: int, gen: int) -> None:
        try:
            from tensorrt_llm._utils import mpi_disabled

            if mpi_disabled():
                return
            from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

            snapshot = _spawn_snapshot()  # workers freeze env+sys.path at spawn
            session = MpiPoolSession(n_workers=spec)
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
                return
        print("[session-prefetch] discarding abandoned background build", flush=True)
        session.shutdown()

    def _drain(self, timeout: float):
        """Join a pending build (abandoning it on timeout) and pop the slot."""
        # Read _thread under the lock: schedule_shadow() assigns-then-starts
        # inside its critical section, and an unlocked read here can observe
        # the assigned-but-not-yet-started thread ("cannot join thread before
        # it is started", CI build 46175, tests creating LLMs concurrently).
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            if thread is not None and thread.is_alive():
                # Abandon the overdue build: bump the generation so its late
                # _publish() shuts the pool down instead of landing.
                self._build_gen += 1
            self._thread = None
            built, self._built = self._built, None
        return built

    def take(self, spec: int):
        """Return a prefetched session for ``spec``, or None to build sync."""
        if not self.enabled:
            return None
        # Slowest legitimate build measured is ~117s (busy node); 180s gives
        # 1.5x margin. On a genuine hang we give up and fall back to a
        # synchronous build instead of stalling the suite.
        built = self._drain(timeout=180)
        if built is None:
            return None
        if built.spec == spec and built.snapshot == _spawn_snapshot():
            # An instant handover skips the ~50s synchronous spawn that used
            # to give the PREVIOUS LLM's worker time to exit and release its
            # GPU memory; don't start the next model build into that race.
            wait_gpu_memory_settle()
            print(f"[session-prefetch] handing over prefetched {spec}-worker pool", flush=True)
            return built.session
        # Spec/env/sys.path mismatch (test skipped, reordered, or changed
        # state the frozen workers would not see): discard, build sync.
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
        if len(self._patched) == len(_PATCH_TARGETS):
            return  # everything already patched: per-test fast path
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
        built = self._drain(timeout=60)
        if built is not None:
            built.session.shutdown()


PREFETCHER = SessionPrefetcher()
