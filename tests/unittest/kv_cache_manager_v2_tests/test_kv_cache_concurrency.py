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

"""Concurrency regression tests for KVCacheManagerV2.

The rest of the Python suite is single-threaded, so it cannot see the failure mode these cover: a
binding that blocks on the C++ API lock while still holding the GIL. That deadlocks against a lock
holder which needs the GIL to run a Python callback, and it stalls every other Python thread for the
duration of a resize even when it does not deadlock outright.

Every test here runs under a faulthandler watchdog, because the bug they guard against manifests as
a hang rather than as a wrong answer. The per-thread join timeouts below are not sufficient on their
own: in the deadlock the *main* thread is the one parked (it holds the API lock and wants the GIL),
so it never reaches the join.
"""

import faulthandler
import threading

import pytest
import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    AttentionLayerConfig,
    BufferConfig,
    CudaStream,
    GpuCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Generous relative to the work each test does (milliseconds); tight enough that a deadlock fails
# the run in reasonable time rather than hanging it.
TIMEOUT_S = 60.0
# Must exceed TIMEOUT_S so a stuck worker is reported as a clean assertion failure first; the
# watchdog is the backstop for a deadlocked *main* thread, which never reaches that assertion.
WATCHDOG_S = 120.0


@pytest.fixture(scope="module", autouse=True)
def initialize_cuda_context() -> None:
    torch.empty(1, device="cuda")


@pytest.fixture(autouse=True)
def deadlock_watchdog():
    """Aborts the process (with every thread's stack) if a test deadlocks.

    faulthandler is the only option that works here: pytest-timeout and any other Python-level
    watchdog runs on a Python thread and so needs the GIL, which is exactly what the deadlock holds
    forever. faulthandler's timer lives in a C thread that never touches the GIL.
    """
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _make_config() -> KVCacheManagerConfig:
    return KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[GpuCacheTierConfig(quota=8 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=0,
                buffers=[BufferConfig(role="key", size=4096)],
            )
        ],
        enable_stats=True,
    )


def _worker(target):
    """Wraps a thread body so it binds the CUDA device and captures any exception.

    KVCM2 calls the CUDA driver API, which needs a current context on *each* thread; a fresh Python
    thread has none, so the first driver call fails with "invalid device context".
    """
    errors: list[BaseException] = []

    def run() -> None:
        try:
            torch.cuda.set_device(0)
            target()
        except BaseException as exc:  # noqa: BLE001 - re-raised by _join below
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.errors = errors  # type: ignore[attr-defined]
    return thread


def _join(*threads: threading.Thread) -> None:
    """Join with a timeout and re-raise whatever the thread captured.

    A deadlocked thread cannot be killed, so on timeout we fail and leave it parked -- strictly
    better than hanging the whole run.
    """
    for thread in threads:
        thread.join(TIMEOUT_S)
        assert not thread.is_alive(), (
            f"timed out after {TIMEOUT_S}s -- likely a GIL/API-lock deadlock"
        )
    for thread in threads:
        for error in getattr(thread, "errors", []):
            raise error


def test_stats_queries_do_not_block_a_concurrent_resize() -> None:
    """Locked read APIs must release the GIL before blocking on the API lock.

    `resize()` holds the exclusive lock with the GIL released. If a stats getter blocks on that lock
    *without* releasing the GIL, it stops every Python thread until the resize finishes -- and
    deadlocks outright when the resize needs the GIL back for a priority callback.
    """
    manager = KVCacheManager(_make_config())
    try:
        stop = threading.Event()
        counts = {"n": 0}

        def poll_stats() -> None:
            while not stop.is_set():
                manager.get_and_reset_iteration_stats()
                manager.get_committed_stats()
                manager.get_dirty_stats_kv_cache_ids()
                manager.get_quota(GPU_LEVEL)
                counts["n"] += 1

        poller = _worker(poll_stats)
        poller.start()
        try:
            for _ in range(50):
                manager.resize(GPU_LEVEL, 8 << 20)
        finally:
            stop.set()
            _join(poller)

        assert counts["n"] > 0
    finally:
        manager.shutdown()


def test_probe_reuse_runs_concurrently_with_a_background_resize() -> None:
    """The motivating scenario: serve prefix-match probes while a resize runs elsewhere."""
    manager = KVCacheManager(_make_config())
    try:
        tokens = list(range(manager.tokens_per_block * 4))
        stop = threading.Event()
        probes = {"n": 0}

        def probe() -> None:
            while not stop.is_set():
                manager.probe_reuse(None, tokens)
                probes["n"] += 1

        prober = _worker(probe)
        prober.start()
        try:
            for _ in range(50):
                manager.resize(GPU_LEVEL, 8 << 20)
        finally:
            stop.set()
            _join(prober)

        assert probes["n"] > 0
    finally:
        manager.shutdown()


def test_many_threads_probe_reuse_concurrently() -> None:
    """Concurrent probes share the lock, so they must not corrupt the radix tree.

    Matching is read-only; were it to drain the pending root erases, two probes could double-free a
    `SharedPtr<RootBlock>`, whose refcount is non-atomic.
    """
    manager = KVCacheManager(_make_config())
    try:
        tokens = list(range(manager.tokens_per_block * 8))
        results: list[int] = []
        lock = threading.Lock()

        def probe() -> None:
            local = [manager.probe_reuse(None, tokens) for _ in range(500)]
            with lock:
                results.extend(local)

        threads = [_worker(probe) for _ in range(8)]
        for thread in threads:
            thread.start()
        _join(*threads)

        assert len(results) == 8 * 500
        assert all(value == 0 for value in results)
    finally:
        manager.shutdown()


def test_priority_callback_under_the_lock_does_not_deadlock_stats_queries() -> None:
    """The exact GIL/API-lock inversion.

    `create_kv_cache`/`resize` hold the exclusive lock with the GIL released, then reacquire the GIL
    to invoke `custom_priority_callback`. A stats getter that blocks on the API lock *while holding
    the GIL* is then unkillable: it waits for the mutex held by the allocating thread, which is
    waiting for the GIL it holds. This test hangs on bindings that do not release the GIL.
    """
    manager = KVCacheManager(_make_config())
    try:
        callback_hits = {"n": 0}

        def priority(block_ordinal: int, life_cycle: object) -> int:
            callback_hits["n"] += 1
            return 0

        stream = CudaStream(torch.cuda.Stream().cuda_stream)
        stop = threading.Event()
        stats_polls = {"n": 0}

        def poll_stats() -> None:
            while not stop.is_set():
                # Exclusive-locked and shared-locked getters, hammered from Python.
                manager.get_and_reset_iteration_stats()
                manager.get_quota(GPU_LEVEL)
                manager.is_stats_excluded(None)
                stats_polls["n"] += 1

        poller = _worker(poll_stats)
        poller.start()
        try:
            tokens = list(range(manager.tokens_per_block * 4))
            for _ in range(200):
                cache = manager.create_kv_cache(None, tokens, custom_priority_callback=priority)
                # resume() needs an explicit stream: the no-stream default leaves the cache in a
                # state that aborts in SharedPageLock::unlock() at suspend() time.
                cache.resume(stream)
                cache.resize(len(tokens))
                cache.suspend()
                cache.close()
        finally:
            stop.set()
            _join(poller)

        assert callback_hits["n"] > 0, "priority callback never ran; test proves nothing"
        assert stats_polls["n"] > 0
    finally:
        manager.shutdown()
