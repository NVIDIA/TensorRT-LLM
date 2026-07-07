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
    def __init__(self, model_dir=None):
        self._marker = _FakeMarker(model_dir) if model_dir else None

    def get_closest_marker(self, name):
        return self._marker if name == "prefetch_model_dir" else None


@pytest.fixture
def prefetcher(monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_PREFETCH_SESSION", "1")
    # These pure-logic tests may themselves run under xdist; pin the worker
    # marker off so the prefetcher's xdist guard does not disable it here.
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    p = SessionPrefetcher()
    # Record build/warm triggers instead of spawning MPI pools/reading weights.
    built, warmed = [], []
    monkeypatch.setattr(SessionPrefetcher, "_build", lambda self, spec, gen: built.append(spec))
    monkeypatch.setattr(SessionPrefetcher, "_warm", lambda self, d: warmed.append(d))
    # No real NVML polling in pure-logic tests.
    monkeypatch.setattr(session_prefetcher, "wait_gpu_memory_settle", lambda: None)
    p.built, p.warmed = built, warmed
    return p


class _FakePool:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.shut = False

    def shutdown(self):
        self.shut = True


def _arm(prefetcher, pool, spec=4):
    """Publish ``pool`` into the shadow slot through the real API."""
    prefetcher._publish(spec, pool, session_prefetcher._spawn_snapshot(), prefetcher._build_gen)


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("TRTLLM_TEST_PREFETCH_SESSION", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    assert SessionPrefetcher().enabled


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_PREFETCH_SESSION", "0")
    p = SessionPrefetcher()
    p.schedule_shadow(2)
    assert p._thread is None
    assert p.take(2) is None


def test_disabled_in_xdist_worker(monkeypatch):
    # Under xdist each worker runs a scheduler-assigned subset; N workers
    # would each hold a live pool plus a spare.
    monkeypatch.setenv("TRTLLM_TEST_PREFETCH_SESSION", "1")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    assert not SessionPrefetcher().enabled


def test_factory_miss_builds_sync_and_arms_shadow(prefetcher):
    # _build is stubbed by the fixture to record specs instead of spawning.
    factory = prefetcher._make_factory(_FakePool)
    session = factory(4)  # nothing prefetched yet -> sync build
    assert isinstance(session, _FakePool) and session.n_workers == 4
    prefetcher._thread.join(timeout=10)
    assert prefetcher.built == [4]  # shadow armed for the next test


def test_factory_single_worker_also_prefetches(prefetcher):
    # The default single-GPU path spawns a 1-worker pool too (executor.py ->
    # proxy.py), paying the same ~50s spawn+import: it must benefit as well.
    factory = prefetcher._make_factory(_FakePool)
    session = factory(1)
    assert isinstance(session, _FakePool)
    prefetcher._thread.join(timeout=10)
    assert prefetcher.built == [1]  # shadow armed for the next 1-GPU test


def test_factory_hit_hands_over_shadow(prefetcher):
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    factory = prefetcher._make_factory(_FakePool)
    assert factory(4) is pool  # prefetched pool handed over


def test_take_spec_mismatch_returns_none(prefetcher):
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    assert prefetcher.take(2) is None  # wrong size: sync fallback


def test_take_discards_on_env_mismatch(prefetcher, monkeypatch):
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    monkeypatch.setenv("TLLM_TEST_ONLY_FLAG", "changed-after-spawn")
    assert prefetcher.take(4) is None  # frozen workers would miss the new env
    import time

    for _ in range(100):
        if pool.shut:
            break
        time.sleep(0.05)
    assert pool.shut  # stale shadow torn down in the background


def test_take_discards_on_nonprefixed_env_mismatch(prefetcher, monkeypatch):
    # Workers inherit the WHOLE parent env at spawn; test knobs outside any
    # TRTLLM*/TLLM* prefix (e.g. OVERRIDE_QUANT_ALGO, read inside workers by
    # model_config.py) must also invalidate a prefetched pool, else workers
    # silently run with stale env (review finding on the prefix allowlist).
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    monkeypatch.setenv("OVERRIDE_QUANT_ALGO", "W4A16_MXFP4")
    assert prefetcher.take(4) is None


def test_pytest_current_test_drift_does_not_discard(prefetcher, monkeypatch):
    # PYTEST_CURRENT_TEST changes every test phase by design; it must not
    # invalidate the snapshot or no prefetched pool would ever be handed over.
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_a (call)")
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_b (setup)")
    assert prefetcher.take(4) is pool


def test_take_discards_on_syspath_mismatch(prefetcher, monkeypatch):
    # CI build 46175: test_modeling_out_of_tree monkeypatches sys.path before
    # LLM(); pool workers freeze sys.path at spawn (MPIPoolExecutor(path=...)),
    # so a pool spawned earlier can't import the out-of-tree module and dies
    # during initialization. sys.path must be part of the handover guard.
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
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


def test_abandoned_build_publish_discards_pool(prefetcher):
    # A build that outlives _drain()'s join timeout is abandoned (generation
    # bump); its late _publish() must shut the pool down, not land it —
    # landing would overwrite (and leak) a newer pool or hand stale state
    # to a future test.
    pool = _FakePool(4)
    gen = prefetcher._build_gen
    prefetcher._build_gen += 1  # what _drain() does on abandonment
    prefetcher._publish(4, pool, session_prefetcher._spawn_snapshot(), gen)
    assert pool.shut
    assert prefetcher._built is None


def test_publish_never_overwrites_unconsumed_pool(prefetcher):
    first, second = _FakePool(4), _FakePool(4)
    _arm(prefetcher, first, spec=4)
    _arm(prefetcher, second, spec=4)
    assert prefetcher._built.session is first  # slot kept
    assert second.shut and not first.shut  # newcomer discarded, not the slot


def test_handover_waits_for_gpu_settle(prefetcher, monkeypatch):
    # The instant handover must first let the previous test's dying worker
    # release its GPU memory (CI build 46175 OOM class).
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    settled = []
    monkeypatch.setattr(session_prefetcher, "wait_gpu_memory_settle", lambda: settled.append(1))
    assert prefetcher.take(4) is pool
    assert settled == [1]


def test_model_switch_triggers_warm_and_dedups(prefetcher):
    items = [_FakeItem("/models/a"), _FakeItem("/models/b")]
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


def test_same_next_model_does_not_warm(prefetcher):
    # Consecutive tests on the same model: its weights are already hot.
    items = [_FakeItem("/models/a"), _FakeItem("/models/a")]
    prefetcher.on_collection(items)
    prefetcher.on_test_setup(items[0])
    import time

    time.sleep(0.2)
    assert prefetcher.warmed == []


def test_no_marker_suite_never_warms(prefetcher):
    # Suites without prefetch_model_dir markers must stay O(1) per test:
    # on_collection keeps an empty map (no per-setup collection scans).
    items = [_FakeItem(), _FakeItem(), _FakeItem()]
    prefetcher.on_collection(items)
    assert prefetcher._next_model == {}
    prefetcher.on_test_setup(items[0])
    assert prefetcher.warmed == []


def test_warm_page_cache_reads_weight_files(tmp_path):
    payload = b"x" * (1 << 20)
    (tmp_path / "model-00001.safetensors").write_bytes(payload)
    (tmp_path / "pytorch_model.bin").write_bytes(payload)
    (tmp_path / "config.json").write_bytes(b"{}")  # not a weight file
    gib = warm_page_cache(str(tmp_path))
    assert gib == pytest.approx(2 / 1024, rel=1e-3)


def test_warm_io_thread_names_covered_by_threadleak_exclude():
    # Both pytest.ini threadleak_exclude lists contain r"session-prefetch-\w+";
    # the warm executor's thread_name_prefix must keep its IO workers inside
    # that pattern (a large warm can outlive the test that started it).
    import re

    assert re.fullmatch(r"session-prefetch-\w+", "session-prefetch-io_0")


def test_patch_targets_cover_all_library_construction_sites():
    # The factory only intercepts the modules listed in _PATCH_TARGETS. If the
    # library grows another MpiPoolSession(...) construction site, prefetch
    # would silently stop covering it (armed spare pools would idle next to
    # directly-constructed ones) — turn that drift into a red test.
    import re
    from pathlib import Path

    import tensorrt_llm

    root = Path(tensorrt_llm.__file__).parent
    # mpi_session.py defines the class (and the MGMN server path, which
    # legitimately builds its own pool outside the bare-LLM() seams).
    exempt = {"tensorrt_llm.llmapi.mpi_session"}
    offenders = []
    for py in root.rglob("*.py"):
        if re.search(r"(?<![\w.])MpiPoolSession\(", py.read_text(errors="ignore")):
            module = "tensorrt_llm." + ".".join(py.relative_to(root).with_suffix("").parts)
            if module not in session_prefetcher._PATCH_TARGETS and module not in exempt:
                offenders.append(module)
    assert not offenders, (
        f"MpiPoolSession constructed outside _PATCH_TARGETS: {offenders} — "
        "add the module(s) to _PATCH_TARGETS (or the exempt list above) so "
        "the prefetch factory keeps covering every bare-LLM() pool spawn"
    )


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


@pytest.fixture
def no_visible_devices_mask(monkeypatch):
    # The settle helper resolves CUDA_VISIBLE_DEVICES; pin it unset so the
    # fake single-GPU pynvml is used as-is.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_settle_noop_when_gpu_free(monkeypatch, no_visible_devices_mask):
    import sys as _sys
    import time

    fake = _fake_pynvml([75 * _GIB], 80 * _GIB)
    monkeypatch.setitem(_sys.modules, "pynvml", fake)
    t0 = time.monotonic()
    session_prefetcher.wait_gpu_memory_settle()
    assert time.monotonic() - t0 < 0.2  # no polling on an already-free GPU
    assert fake.calls["n"] == 1


def test_settle_waits_for_previous_worker_release(monkeypatch, no_visible_devices_mask):
    import sys as _sys

    # Free memory rising as the previous worker exits: 17 -> 40 -> 75 GiB.
    fake = _fake_pynvml([17 * _GIB, 40 * _GIB, 75 * _GIB], 80 * _GIB)
    monkeypatch.setitem(_sys.modules, "pynvml", fake)
    monkeypatch.setattr(session_prefetcher, "_SETTLE_POLL_S", 0.01)
    session_prefetcher.wait_gpu_memory_settle()
    assert fake.calls["n"] == 3  # polled until the release completed


def test_settle_gives_up_when_memory_stays_in_use(monkeypatch, no_visible_devices_mask):
    import sys as _sys
    import time

    fake = _fake_pynvml([17 * _GIB], 80 * _GIB)  # flat: legitimately in use
    monkeypatch.setitem(_sys.modules, "pynvml", fake)
    monkeypatch.setattr(session_prefetcher, "_SETTLE_POLL_S", 0.01)
    t0 = time.monotonic()
    session_prefetcher.wait_gpu_memory_settle(timeout=5)
    assert time.monotonic() - t0 < 1.0  # flat-detection, not the full timeout


def test_settle_skips_when_no_gpus_visible(monkeypatch):
    import sys as _sys

    # CUDA_VISIBLE_DEVICES="" means NO GPUs are visible to this process:
    # nothing to wait for (and no polling of other jobs' GPUs).
    fake = _fake_pynvml([17 * _GIB], 80 * _GIB)
    monkeypatch.setitem(_sys.modules, "pynvml", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    session_prefetcher.wait_gpu_memory_settle()
    assert fake.calls["n"] == 0  # no memory queries at all


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
