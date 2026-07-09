# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for the session prefetcher — no MPI, no GPU."""

import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from test_common import session_prefetcher
from test_common.session_prefetcher import SessionPrefetcher, warm_page_cache


class _FakeMarker:
    def __init__(self, *args):
        self.args = args


class _FakeItem:
    def __init__(self, model_dir=None, cls=None):
        self._marker = _FakeMarker(model_dir) if model_dir else None
        self.cls = cls

    def get_closest_marker(self, name):
        return self._marker if name == "prefetch_model_dir" else None


def _as_session(*items):
    """Link fake items into a fake pytest session (final run order)."""
    session = types.SimpleNamespace(items=list(items))
    for item in items:
        item.session = session
    return list(items)


def _wait_for(cond, timeout=5.0):
    """Poll ``cond`` (fire-and-forget background work) until true or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


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
    # No real NVML polling in pure-logic tests (True = safe to hand over).
    monkeypatch.setattr(session_prefetcher, "wait_gpu_memory_settle", lambda: True)
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
    assert _wait_for(lambda: pool.shut)  # stale shadow torn down in background


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
    # test_modeling_out_of_tree monkeypatches sys.path before
    # LLM(); pool workers freeze sys.path at spawn (MPIPoolExecutor(path=...)),
    # so a pool spawned earlier can't import the out-of-tree module and dies
    # during initialization. sys.path must be part of the handover guard.
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    monkeypatch.syspath_prepend("/oot/example/path")
    assert prefetcher.take(4) is None  # frozen workers would miss the new path


def test_take_does_not_join_unstarted_shadow_thread(prefetcher, monkeypatch):
    # A test creating LLMs concurrently (ThreadPoolExecutor)
    # raced take()'s unlocked read of _thread against schedule_shadow()'s
    # assign-then-start critical section, joining a thread that had not been
    # started yet ("cannot join thread before it is started"). A slow start()
    # widens the assign->start window deterministically.
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
    # release its GPU memory (insufficient-GPU-memory failure class).
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    settled = []
    monkeypatch.setattr(
        session_prefetcher, "wait_gpu_memory_settle", lambda: settled.append(1) or True
    )
    assert prefetcher.take(4) is pool
    assert settled == [1]


def test_session_summary_counters_and_emission(prefetcher, capfd):
    # Handover / stale-discard / busy-discard / superseded each count once;
    # dispose() emits ONE summary line (outside pytest capture in real runs,
    # the only guaranteed console-visible record of prefetch activity).
    hit = _FakePool(4)
    _arm(prefetcher, hit, spec=4)
    assert prefetcher.take(4) is hit
    stale = _FakePool(4)
    _arm(prefetcher, stale, spec=4)
    assert prefetcher.take(2) is None  # spec mismatch -> stale discard
    busy = _FakePool(4)
    _arm(prefetcher, busy, spec=4)
    prefetcher._build_gen += 0  # no-op, keep gen valid
    import test_common.session_prefetcher as sp

    orig = sp.wait_gpu_memory_settle
    sp.wait_gpu_memory_settle = lambda: False
    try:
        assert prefetcher.take(4) is None  # busy refusal
    finally:
        sp.wait_gpu_memory_settle = orig
    late = _FakePool(4)
    prefetcher._publish(4, late, session_prefetcher._spawn_snapshot(), prefetcher._build_gen - 1)
    assert prefetcher.stats["pools_handed_over"] == 1
    assert prefetcher.stats["pools_discarded_stale"] == 1
    assert prefetcher.stats["pools_discarded_gpu_busy"] == 1
    assert prefetcher.stats["pools_discarded_superseded"] == 1
    prefetcher.dispose()
    out = capfd.readouterr().out
    assert "[session-prefetch] session summary:" in out
    assert "pools_handed_over=1" in out


def test_no_summary_when_prefetch_never_fired(prefetcher, capfd):
    prefetcher.dispose()
    assert "session summary" not in capfd.readouterr().out


def test_warm_counters_track_gib(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_PREFETCH_SESSION", "1")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    p = SessionPrefetcher()  # real _warm (fixture would stub it)
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * (1 << 20))
    p._warm(str(tmp_path))
    assert p.stats["warms"] == 1 and p._warmed_gib > 0
    p._warm("not/a/real/dir")
    assert p.stats["warm_noops"] == 1


def test_model_switch_triggers_warm_and_dedups(prefetcher):
    items = _as_session(_FakeItem("/models/a"), _FakeItem("/models/b"))
    prefetcher.on_test_setup(items[0])
    # Warm threads are fire-and-forget; poll briefly for the recorded call.
    assert _wait_for(lambda: prefetcher.warmed)
    assert prefetcher.warmed == ["/models/b"]
    # Same next-model again: deduplicated.
    prefetcher.on_test_setup(items[0])
    time.sleep(0.2)
    assert prefetcher.warmed == ["/models/b"]


def test_same_next_model_does_not_warm(prefetcher):
    # Consecutive tests on the same model: its weights are already hot.
    items = _as_session(_FakeItem("/models/a"), _FakeItem("/models/a"))
    prefetcher.on_test_setup(items[0])
    time.sleep(0.2)
    assert prefetcher.warmed == []


def test_no_marker_suite_never_warms(prefetcher):
    items = _as_session(_FakeItem(), _FakeItem(), _FakeItem())
    prefetcher.on_test_setup(items[0])
    time.sleep(0.2)
    assert prefetcher.warmed == []


def test_auto_model_dir_from_accuracy_class_attr(prefetcher):
    # Accuracy-harness classes declare MODEL_PATH; warming must pick it up
    # automatically, with no marker on the test.
    class _TestLlama:
        MODEL_PATH = "/models/llama"

    class _TestQwen:
        MODEL_PATH = "/models/qwen"

    items = _as_session(_FakeItem(cls=_TestLlama), _FakeItem(cls=_TestQwen))
    prefetcher.on_test_setup(items[0])
    assert _wait_for(lambda: prefetcher.warmed)
    assert prefetcher.warmed == ["/models/qwen"]


def test_marker_overrides_class_model_path():
    class _TestCls:
        MODEL_PATH = "/models/from-class"

    item = _FakeItem(model_dir="/models/from-marker", cls=_TestCls)
    assert session_prefetcher._model_dir_of(item) == "/models/from-marker"


def test_accuracy_harness_still_declares_model_path():
    # _model_dir_of auto-discovers models via the MODEL_PATH class attribute of
    # the accuracy harnesses (accuracy_core.py); renaming that attribute would
    # silently kill warming repo-wide. Textual check — importing accuracy_core
    # would drag integration-only dependencies into this unit test.
    core = Path(__file__).parents[2] / "integration" / "defs" / "accuracy" / "accuracy_core.py"
    assert re.search(r"^\s+MODEL_PATH\s*=", core.read_text(), re.MULTILINE), (
        "accuracy_core.py no longer declares MODEL_PATH — update "
        "session_prefetcher._model_dir_of to the harness's new convention"
    )


def test_warm_selects_files_like_the_weight_loader(tmp_path):
    # Selection must mirror HfWeightLoader.load_weights: safetensors first
    # (minus huge "consolidated" copies the loader skips), so the .bin copy
    # and the consolidated file must NOT be read here.
    payload = b"x" * (1 << 20)
    (tmp_path / "model-00001.safetensors").write_bytes(payload)
    (tmp_path / "consolidated.safetensors").write_bytes(payload * 4)
    (tmp_path / "pytorch_model.bin").write_bytes(payload)
    (tmp_path / "config.json").write_bytes(b"{}")  # not a weight file
    assert warm_page_cache(str(tmp_path)) == pytest.approx(1 / 1024, rel=1e-3)


def test_warm_falls_back_to_bin_then_pth(tmp_path):
    payload = b"x" * (1 << 20)
    bin_dir, pth_dir = tmp_path / "bin", tmp_path / "pth"
    bin_dir.mkdir(), pth_dir.mkdir()
    (bin_dir / "pytorch_model.bin").write_bytes(payload)
    (pth_dir / "model.pth").write_bytes(payload)
    assert warm_page_cache(str(bin_dir)) == pytest.approx(1 / 1024, rel=1e-3)
    assert warm_page_cache(str(pth_dir)) == pytest.approx(1 / 1024, rel=1e-3)


def test_warm_page_cache_ignores_non_weight_dirs(tmp_path):
    # MODEL_PATH may be an HF model id or a dir without local weights: no-op.
    assert warm_page_cache(str(tmp_path)) == 0.0
    assert warm_page_cache("not/a/real/dir") == 0.0


def test_warm_skips_models_larger_than_host_memory(tmp_path, monkeypatch):
    # Warming a model bigger than free RAM is pure filer traffic: the pages
    # would be evicted before the test loads them (DeepSeek-R1-class dirs).
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * (1 << 20))
    monkeypatch.setattr(session_prefetcher, "_available_host_memory", lambda: 1 << 10)
    assert warm_page_cache(str(tmp_path)) == 0.0


def test_warm_io_thread_names_covered_by_threadleak_exclude():
    # Both pytest.ini threadleak_exclude lists contain r"session-prefetch-\w+";
    # the warm executor's thread_name_prefix must keep its IO workers inside
    # that pattern (a large warm can outlive the test that started it).
    assert re.fullmatch(r"session-prefetch-\w+", "session-prefetch-io_0")


def test_patch_targets_cover_all_library_construction_sites():
    # The factory only intercepts the modules listed in _PATCH_TARGETS. If the
    # library grows another MpiPoolSession(...) construction site, prefetch
    # would silently stop covering it (armed spare pools would idle next to
    # directly-constructed ones) — turn that drift into a red test.
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
def unset_cuda_visible_devices(monkeypatch):
    # The settle helper resolves CUDA_VISIBLE_DEVICES; pin it unset so the
    # fake single-GPU pynvml is used as-is.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_settle_noop_when_gpu_free(monkeypatch, unset_cuda_visible_devices):
    fake = _fake_pynvml([75 * _GIB], 80 * _GIB)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    t0 = time.monotonic()
    assert session_prefetcher.wait_gpu_memory_settle() is True
    assert time.monotonic() - t0 < 0.2  # no polling on an already-free GPU
    assert fake.calls["n"] == 1


def test_settle_waits_for_previous_worker_release(monkeypatch, unset_cuda_visible_devices):
    # Free memory rising as the previous worker exits: 17 -> 40 -> 75 GiB.
    fake = _fake_pynvml([17 * _GIB, 40 * _GIB, 75 * _GIB], 80 * _GIB)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(session_prefetcher, "_SETTLE_POLL_S", 0.01)
    assert session_prefetcher.wait_gpu_memory_settle() is True
    assert fake.calls["n"] == 3  # polled until the release completed


def test_settle_gives_up_when_memory_stays_in_use(monkeypatch, unset_cuda_visible_devices):
    fake = _fake_pynvml([17 * _GIB], 80 * _GIB)  # flat: legitimately in use
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(session_prefetcher, "_SETTLE_POLL_S", 0.01)
    t0 = time.monotonic()
    # 17/80 GiB free (21%) is below the handover threshold: refuse.
    assert session_prefetcher.wait_gpu_memory_settle(timeout=5) is False
    assert time.monotonic() - t0 < 1.0  # flat-detection, not the full timeout


def test_settle_flat_but_half_free_still_hands_over(monkeypatch, unset_cuda_visible_devices):
    # Memory flat at 60% free (e.g. a small resident fixture): plenty of room
    # for the next model — hand over rather than pay the sync spawn.
    fake = _fake_pynvml([48 * _GIB], 80 * _GIB)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(session_prefetcher, "_SETTLE_POLL_S", 0.01)
    assert session_prefetcher.wait_gpu_memory_settle(timeout=5) is True


def test_take_discards_pool_when_gpu_stays_busy(prefetcher, monkeypatch):
    # When the settle wait gives up on a still-mostly-used GPU, the
    # handed-over build would OOM in the worker. A busy GPU must force the
    # sync fallback (whose ~50s spawn restores the release window) instead.
    pool = _FakePool(4)
    _arm(prefetcher, pool, spec=4)
    monkeypatch.setattr(session_prefetcher, "wait_gpu_memory_settle", lambda: False)
    assert prefetcher.take(4) is None
    assert _wait_for(lambda: pool.shut)  # pool discarded in the background


def test_settle_skips_when_no_gpus_visible(monkeypatch):
    # CUDA_VISIBLE_DEVICES="" means NO GPUs are visible to this process:
    # nothing to wait for (and no polling of other jobs' GPUs).
    fake = _fake_pynvml([17 * _GIB], 80 * _GIB)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert session_prefetcher.wait_gpu_memory_settle() is True
    assert fake.calls["n"] == 0  # no memory queries at all


def test_settle_without_pynvml_is_noop(monkeypatch):
    import builtins

    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    real_import = builtins.__import__

    def _no_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pynvml)
    # Must not raise, and must fail open (hand over as before the barrier).
    assert session_prefetcher.wait_gpu_memory_settle() is True
