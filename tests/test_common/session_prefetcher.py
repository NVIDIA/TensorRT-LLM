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

Enable with ``TRTLLM_TEST_PREFETCH_SESSION=1``; off by default so CI behavior
is unchanged until the mechanism has soaked.

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
        self._items = []
        self._warmed_dirs = set()

    @property
    def enabled(self) -> bool:
        return os.environ.get("TRTLLM_TEST_PREFETCH_SESSION", "0").lower() in (
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

            session = MpiPoolSession(n_workers=spec)
            session.submit_sync(_worker_import_and_assert_cuda_clean)
            with self._lock:
                self._built_spec = spec
                self._built_session = session
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
            thread.join(timeout=600)
        with self._lock:
            self._thread = None
            if self._built_spec == spec and self._built_session is not None:
                session, self._built_session, self._built_spec = (self._built_session, None, None)
                print(f"[session-prefetch] handing over prefetched {spec}-worker pool", flush=True)
                return session
            # Spec mismatch (e.g. the expected test was skipped): discard.
            if self._built_session is not None:
                stale, self._built_session, self._built_spec = (self._built_session, None, None)
                threading.Thread(target=stale.shutdown, daemon=True).start()
        return None


PREFETCHER = SessionPrefetcher()
