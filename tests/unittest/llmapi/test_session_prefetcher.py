# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for the session prefetcher — no MPI, no GPU."""

import pytest
from test_common import session_prefetcher
from test_common.session_prefetcher import SessionPrefetcher, warm_page_cache


class _FakeMarker:
    def __init__(self, *args):
        self.args = args


class _FakeItem:
    def __init__(self, spec=None, model_dir=None):
        self._markers = {}
        if spec is not None:
            self._markers["prefetch_session"] = _FakeMarker(spec)
        if model_dir is not None:
            self._markers["prefetch_model_dir"] = _FakeMarker(model_dir)

    def get_closest_marker(self, name):
        return self._markers.get(name)


@pytest.fixture
def prefetcher(monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_PREFETCH_SESSION", "1")
    p = SessionPrefetcher()
    # Record trigger calls instead of spawning MPI pools / reading weights.
    built, warmed = [], []
    monkeypatch.setattr(SessionPrefetcher, "_build", lambda self, spec: built.append(spec))
    monkeypatch.setattr(SessionPrefetcher, "_warm", lambda self, d: warmed.append(d))
    # No real NVML polling in pure-logic tests.
    monkeypatch.setattr(session_prefetcher, "wait_gpu_memory_settle", lambda: None)
    p.built, p.warmed = built, warmed
    return p


def _run_setup_and_join(p, item):
    p.on_test_setup(item)
    for attr in ("_thread",):
        th = getattr(p, attr)
        if th is not None:
            th.join(timeout=10)


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("TRTLLM_TEST_PREFETCH_SESSION", raising=False)
    assert SessionPrefetcher().enabled


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_PREFETCH_SESSION", "0")
    p = SessionPrefetcher()
    p.on_collection([_FakeItem(spec=2)])
    assert p._items == []
    assert p.take(2) is None


def test_same_spec_does_not_build(prefetcher):
    items = [_FakeItem(spec=2), _FakeItem(spec=2)]
    prefetcher.on_collection(items)
    _run_setup_and_join(prefetcher, items[0])
    assert prefetcher.built == []


def test_spec_boundary_builds_next(prefetcher):
    items = [_FakeItem(spec=2), _FakeItem(spec=2), _FakeItem(spec=4)]
    prefetcher.on_collection(items)
    _run_setup_and_join(prefetcher, items[0])  # next spec == 2 -> no-op
    assert prefetcher.built == []
    _run_setup_and_join(prefetcher, items[1])  # next spec == 4 -> build
    assert prefetcher.built == [4]


def test_model_switch_triggers_warm_even_within_block(prefetcher):
    items = [
        _FakeItem(spec=2, model_dir="/models/a"),
        _FakeItem(spec=2, model_dir="/models/b"),
    ]
    prefetcher.on_collection(items)
    prefetcher.on_test_setup(items[0])
    # Warm threads are fire-and-forget; poll briefly for the recorded call.
    import time

    for _ in range(100):
        if prefetcher.warmed:
            break
        time.sleep(0.05)
    assert prefetcher.warmed == ["/models/b"]
    # Same next-model again: deduplicated.
    prefetcher.on_test_setup(items[0])
    time.sleep(0.2)
    assert prefetcher.warmed == ["/models/b"]


def test_take_spec_mismatch_returns_none(prefetcher):
    prefetcher._built_spec = 4
    prefetcher._built_session = None
    assert prefetcher.take(2) is None


def test_cls_n_gpus_fallback():
    class _Cls:
        n_gpus = 8

    item = _FakeItem()
    item.cls = _Cls
    assert session_prefetcher._spec_of(item) == 8


class _FakePool:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.shut = False

    def shutdown(self):
        self.shut = True


@pytest.mark.prefetch_session(2)
def test_fixture_hands_over_prefetched_pool(request, monkeypatch):
    pool = _FakePool(2)
    monkeypatch.setattr(session_prefetcher.PREFETCHER, "take", lambda spec: pool)
    assert request.getfixturevalue("prefetched_mpi_session") is pool


@pytest.mark.prefetch_session(2)
def test_fixture_falls_back_to_sync_build(request, monkeypatch):
    import tensorrt_llm.llmapi.mpi_session as mpi_session_mod

    monkeypatch.setattr(session_prefetcher.PREFETCHER, "take", lambda spec: None)
    monkeypatch.setattr(mpi_session_mod, "MpiPoolSession", _FakePool)
    session = request.getfixturevalue("prefetched_mpi_session")
    assert isinstance(session, _FakePool) and session.n_workers == 2


def test_factory_miss_builds_sync_and_arms_shadow(prefetcher, monkeypatch):
    # _build is stubbed by the fixture to record specs instead of spawning.
    factory = prefetcher._make_factory(_FakePool)
    session = factory(4)  # nothing prefetched yet -> sync build
    assert isinstance(session, _FakePool) and session.n_workers == 4
    prefetcher._thread.join(timeout=10)
    assert prefetcher.built == [4]  # shadow armed for the next test


def test_factory_hit_hands_over_shadow(prefetcher, monkeypatch):
    pool = _FakePool(4)
    prefetcher._built_spec, prefetcher._built_session = 4, pool
    prefetcher._built_snapshot = session_prefetcher._spawn_snapshot()
    factory = prefetcher._make_factory(_FakePool)
    assert factory(4) is pool  # prefetched pool handed over


def test_take_discards_on_env_mismatch(prefetcher, monkeypatch):
    pool = _FakePool(4)
    prefetcher._built_spec, prefetcher._built_session = 4, pool
    prefetcher._built_snapshot = session_prefetcher._spawn_snapshot()
    monkeypatch.setenv("TLLM_TEST_ONLY_FLAG", "changed-after-spawn")
    assert prefetcher.take(4) is None  # frozen workers would miss the new env
    for _ in range(100):
        if pool.shut:
            break
        import time

        time.sleep(0.05)
    assert pool.shut  # stale shadow torn down in the background


def test_take_discards_on_syspath_mismatch(prefetcher, monkeypatch):
    # CI build 46175: test_modeling_out_of_tree monkeypatches sys.path before
    # LLM(); pool workers freeze sys.path at spawn (MPIPoolExecutor(path=...)),
    # so a pool spawned earlier can't import the out-of-tree module and dies
    # during initialization. sys.path must be part of the handover guard.
    pool = _FakePool(4)
    prefetcher._built_spec, prefetcher._built_session = 4, pool
    prefetcher._built_snapshot = session_prefetcher._spawn_snapshot()
    monkeypatch.syspath_prepend("/oot/example/path")
    assert prefetcher.take(4) is None  # frozen workers would miss the new path


def test_take_does_not_join_unstarted_shadow_thread(prefetcher, monkeypatch):
    # CI build 46175: a test creating LLMs concurrently (ThreadPoolExecutor)
    # raced take()'s unlocked read of _thread against schedule_shadow()'s
    # assign-then-start critical section, joining a thread that had not been
    # started yet ("cannot join thread before it is started"). A slow start()
    # widens the assign->start window deterministically.
    import threading
    import time

    class _SlowStartThread(threading.Thread):
        def start(self):
            time.sleep(0.3)  # hold the assigned-but-unstarted state visible
            super().start()

    monkeypatch.setattr(session_prefetcher.threading, "Thread", _SlowStartThread)
    errors = []

    def _taker():
        time.sleep(0.1)  # let schedule_shadow enter its critical section first
        try:
            prefetcher.take(1)
        except RuntimeError as e:  # pre-fix: "cannot join thread before it is started"
            errors.append(e)

    taker = threading.Thread(target=_taker)
    taker.start()
    prefetcher.schedule_shadow(1)
    taker.join(timeout=10)
    assert errors == []


def test_factory_single_worker_also_prefetches(prefetcher):
    # The default single-GPU path spawns a 1-worker pool too (executor.py ->
    # proxy.py), paying the same ~50s spawn+import: it must benefit as well.
    factory = prefetcher._make_factory(_FakePool)
    session = factory(1)
    assert isinstance(session, _FakePool)
    prefetcher._thread.join(timeout=10)
    assert prefetcher.built == [1]  # shadow armed for the next 1-GPU test


def test_handover_waits_for_gpu_settle(prefetcher, monkeypatch):
    # The instant handover must first let the previous test's dying worker
    # release its GPU memory (CI build 46175 OOM class).
    pool = _FakePool(4)
    prefetcher._built_spec, prefetcher._built_session = 4, pool
    prefetcher._built_snapshot = session_prefetcher._spawn_snapshot()
    settled = []
    monkeypatch.setattr(session_prefetcher, "wait_gpu_memory_settle", lambda: settled.append(1))
    assert prefetcher.take(4) is pool
    assert settled == [1]


_GIB = 1 << 30


class _FakeMem:
    def __init__(self, free, total):
        self.free, self.total = free, total


def _fake_pynvml(free_sequence, total):
    """A pynvml stand-in whose reported free memory follows ``free_sequence``."""
    import types

    mod = types.ModuleType("pynvml")
    calls = {"n": 0}
    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetCount = lambda: 1
    mod.nvmlDeviceGetHandleByIndex = lambda i: i

    def _mem_info(handle):
        free = free_sequence[min(calls["n"], len(free_sequence) - 1)]
        calls["n"] += 1
        return _FakeMem(free, total)

    mod.nvmlDeviceGetMemoryInfo = _mem_info
    mod.calls = calls
    return mod


def test_settle_noop_when_gpu_free(monkeypatch):
    import sys as _sys
    import time

    fake = _fake_pynvml([75 * _GIB], 80 * _GIB)
    monkeypatch.setitem(_sys.modules, "pynvml", fake)
    t0 = time.monotonic()
    session_prefetcher.wait_gpu_memory_settle()
    assert time.monotonic() - t0 < 0.2  # no polling on an already-free GPU
    assert fake.calls["n"] == 1


def test_settle_waits_for_previous_worker_release(monkeypatch):
    import sys as _sys

    # Free memory rising as the previous worker exits: 17 -> 40 -> 75 GiB.
    fake = _fake_pynvml([17 * _GIB, 40 * _GIB, 75 * _GIB], 80 * _GIB)
    monkeypatch.setitem(_sys.modules, "pynvml", fake)
    monkeypatch.setattr(session_prefetcher, "_SETTLE_POLL_S", 0.01)
    session_prefetcher.wait_gpu_memory_settle()
    assert fake.calls["n"] == 3  # polled until the release completed


def test_settle_gives_up_when_memory_stays_in_use(monkeypatch):
    import sys as _sys
    import time

    fake = _fake_pynvml([17 * _GIB], 80 * _GIB)  # flat: legitimately in use
    monkeypatch.setitem(_sys.modules, "pynvml", fake)
    monkeypatch.setattr(session_prefetcher, "_SETTLE_POLL_S", 0.01)
    t0 = time.monotonic()
    session_prefetcher.wait_gpu_memory_settle(timeout=5)
    assert time.monotonic() - t0 < 1.0  # flat-detection, not the full timeout


def test_settle_without_pynvml_is_noop(monkeypatch):
    import builtins
    import sys as _sys

    monkeypatch.delitem(_sys.modules, "pynvml", raising=False)
    real_import = builtins.__import__

    def _no_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pynvml)
    session_prefetcher.wait_gpu_memory_settle()  # must not raise


def test_warm_page_cache_reads_weight_files(tmp_path):
    payload = b"x" * (1 << 20)
    (tmp_path / "model-00001.safetensors").write_bytes(payload)
    (tmp_path / "pytorch_model.bin").write_bytes(payload)
    (tmp_path / "config.json").write_bytes(b"{}")  # not a weight file
    gib = warm_page_cache(str(tmp_path))
    assert gib == pytest.approx(2 / 1024, rel=1e-3)
