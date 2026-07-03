# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in background prefetch of the NEXT test's session and model page cache.

Multi-GPU LLM-API tests pay ~50-65s per test to spawn an MPI pool whose
workers import ``tensorrt_llm``, plus a cold read of the model weights. Both
are pure CPU/IO work, so while the CURRENT test is running on the GPUs the
next test's pool can be spawned and its weight files pre-read in background
threads — hiding those costs behind the previous test's runtime. The
prefetched pool must NOT touch CUDA until it is handed over (asserted at
``take()``).

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_WEIGHT_GLOBS = ("*.safetensors", "*.bin")
_READ_CHUNK = 64 << 20  # 64MB

# MpiPoolSession workers freeze their environment at spawn time, so a pool
# prefetched during test A must not be handed to test B if B changed any env
# var the workers care about (proven silent-failure class).
_ENV_PREFIXES = ("TRTLLM", "TLLM", "NCCL_", "CUDA_", "UCX_", "OMPI_")


def _env_snapshot():
    return {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES)}


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


def _worker_import_and_assert_cuda_clean() -> bool:
    import torch

    import tensorrt_llm  # noqa: F401  (the expensive part we want to hide)

    assert not torch.cuda.is_initialized(), "prefetched pool worker touched CUDA before handover"
    return True


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

    with ThreadPoolExecutor(max_workers=4) as ex:
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
        self._built_spec = None
        self._built_session = None
        self._built_env = None
        self._items = []
        self._warmed_dirs = set()

    @property
    def enabled(self) -> bool:
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
                threading.Thread(target=self._warm, args=(nxt_model,), daemon=True).start()

        cur = _spec_of(item)
        nxt = next(
            (_spec_of(it) for it in self._items[idx + 1 :] if _spec_of(it) is not None), None
        )
        if nxt is None or nxt == cur:
            return
        with self._lock:
            if self._thread is not None or self._built_spec == nxt:
                return  # already building / built
            self._thread = threading.Thread(target=self._build, args=(nxt,), daemon=True)
            self._thread.start()

    def _warm(self, model_dir: str) -> None:
        try:
            warm_page_cache(model_dir)
        except Exception as e:  # warming must never break the tests
            print(f"[session-prefetch] page-cache warm failed (harmless): {e}", flush=True)

    def _build(self, spec: int) -> None:
        try:
            from tensorrt_llm._utils import mpi_disabled

            if mpi_disabled():
                return
            from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

            env = _env_snapshot()  # workers freeze env at spawn: snapshot now
            session = MpiPoolSession(n_workers=spec)
            session.submit_sync(_worker_import_and_assert_cuda_clean)
            with self._lock:
                self._built_spec = spec
                self._built_session = session
                self._built_env = env
        except Exception as e:  # prefetch must never break the tests
            print(
                f"[session-prefetch] background build failed (falling back to synchronous): {e}",
                flush=True,
            )

    def take(self, spec: int):
        """Return a prefetched session for ``spec``, or None to build sync."""
        if not self.enabled:
            return None
        thread = self._thread
        if thread is not None:
            # Slowest legitimate build measured is ~117s (busy node); 180s
            # gives 1.5x margin. On a genuine hang we give up and fall back
            # to a synchronous build instead of stalling the suite.
            thread.join(timeout=180)
        with self._lock:
            self._thread = None
            if (
                self._built_spec == spec
                and self._built_session is not None
                and self._built_env == _env_snapshot()
            ):
                session, self._built_session, self._built_spec = (self._built_session, None, None)
                print(f"[session-prefetch] handing over prefetched {spec}-worker pool", flush=True)
                return session
            # Spec/env mismatch (test skipped, reordered, or changed env vars
            # the frozen workers would not see): discard, build synchronously.
            if self._built_session is not None:
                stale, self._built_session, self._built_spec = (self._built_session, None, None)
                threading.Thread(target=stale.shutdown, daemon=True).start()
        return None

    # ---- shadow mode: zero-test-change prefetch at the pool-creation seam ----

    def schedule_shadow(self, spec: int) -> None:
        """Start building a spare ``spec``-worker pool in the background.

        Heuristic: the next test most likely needs a pool of the same size as
        the current one. A miss is discarded at ``take()`` and the sync build
        is no slower than without prefetch.
        """
        if not self.enabled or spec <= 1:
            return
        with self._lock:
            if self._thread is not None or self._built_spec == spec:
                return  # already building / built
            self._thread = threading.Thread(target=self._build, args=(spec,), daemon=True)
            self._thread.start()

    def _make_factory(self, real_cls):
        """A drop-in for ``MpiPoolSession`` that consumes and re-arms the shadow."""

        def factory(n_workers, *args, **kwargs):
            if args or kwargs or n_workers <= 1:
                return real_cls(n_workers, *args, **kwargs)
            session = self.take(n_workers) or real_cls(n_workers=n_workers)
            self.schedule_shadow(n_workers)  # re-arm for the NEXT test
            return session

        return factory

    def install_pool_factory(self) -> None:
        """Patch the pool-creation seams for zero-test-change prefetch.

        Bare ``LLM(...)`` tests then consume prefetched pools automatically.
        Only the ``mpi_session is None`` branches construct ``MpiPoolSession``
        directly, so tests passing their own session (shared/grouped pools)
        are never intercepted.
        """
        import importlib

        from tensorrt_llm.llmapi.mpi_session import MpiPoolSession as real_cls

        factory = self._make_factory(real_cls)
        for name in (
            "tensorrt_llm.executor.proxy",
            "tensorrt_llm.executor.rpc_proxy",
            "tensorrt_llm.llmapi.llm",
        ):
            mod = importlib.import_module(name)
            if getattr(mod, "MpiPoolSession", None) is real_cls:
                mod.MpiPoolSession = factory

    def dispose(self) -> None:
        """Shut down any unconsumed shadow pool (end-of-session cleanup)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=60)
        with self._lock:
            self._thread = None
            stale, self._built_session, self._built_spec = (self._built_session, None, None)
        if stale is not None:
            stale.shutdown()


PREFETCHER = SessionPrefetcher()
