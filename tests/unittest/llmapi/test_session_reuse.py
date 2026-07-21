# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for automatic MPI session reuse — no MPI, no GPU."""

import pytest
from test_common import session_reuse
from test_common.session_reuse import SessionReuseCache


class _FakePool:
    def __init__(self, n_workers, wait_shutdown=False, env_overrides=None):
        self.n_workers = n_workers
        self.wait_shutdown = wait_shutdown
        self.env_overrides = dict(env_overrides or {})
        self.shut = False
        import os

        # What the workers freeze at spawn: TRTLLM* forwarded from the parent
        # env plus explicit overrides (mirrors MpiPoolSession._start_mpi_pool).
        self.spawn_env_weight_cache = self.env_overrides.get(
            "TRTLLM_HF_WEIGHT_CACHE", os.environ.get("TRTLLM_HF_WEIGHT_CACHE")
        )

    def shutdown(self):
        self.shut = True

    def shutdown_abort(self, *args, **kwargs):
        self.shut = True


@pytest.fixture
def reuse_cache(monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_REUSE_SESSION", "1")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    cache = SessionReuseCache()
    # No real MPI / NVML in pure-logic tests: record the calls instead.
    resets = []
    monkeypatch.setattr(session_reuse, "submit_sync_per_worker", lambda s, fn: resets.append(s))
    cache.resets = resets

    # Hermetic: the REAL prefetcher singleton must not start background MPI
    # builds from these tests; vend/record through a fake instead.
    class _FakePrefetcher:
        def __init__(self):
            self.shadow = None  # pool vended on the next take()
            self.restocks = []

        def take(self, n):
            pool, self.shadow = self.shadow, None
            return pool

        def schedule_shadow(self, n, env_overlay=None):
            self.restocks.append((n, env_overlay))

    cache.prefetch = _FakePrefetcher()
    monkeypatch.setattr(session_reuse, "_prefetcher", lambda: cache.prefetch)
    return cache


def _wait(pred, timeout=5.0):
    import time

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_reuse_hands_back_same_pool(reuse_cache):
    s1 = reuse_cache.acquire(_FakePool, 2)
    real = s1._real
    # Cache-managed pools block their (real) shutdown on worker exit, so a
    # replacement spawned right after a retire cannot race the GPU release.
    assert real.wait_shutdown
    s1.shutdown()  # released to cache, not killed
    assert not real.shut
    s2 = reuse_cache.acquire(_FakePool, 2)
    assert s2._real is real  # the SAME pool, reused
    assert reuse_cache.resets  # workers were reset between handouts


def test_cached_handover_reaps_in_flight_retires(reuse_cache):
    # Two same-size pools released back-to-back (concurrent LLMs in one
    # test): the duplicate is retired in a BACKGROUND thread while holding
    # full model GPU memory until its workers exit. The next instant
    # cached-pool handover must join that retire first — the corpse race the
    # deleted NVML settle barrier used to cover.
    s1 = reuse_cache.acquire(_FakePool, 2)
    s2 = reuse_cache.acquire(_FakePool, 2)  # cache miss: second pool
    kept, dup = s1._real, s2._real
    s1.shutdown()  # released: cached
    s2.shutdown()  # duplicate slot: retired in background
    assert session_reuse._RETIRE_THREADS  # retire in flight (or just done)
    s3 = reuse_cache.acquire(_FakePool, 2)
    assert s3._real is kept
    assert not session_reuse._RETIRE_THREADS  # handover joined the retire
    assert dup.shut  # corpse fully disposed before the handover returned


def test_cache_miss_takes_prefetched_shadow(reuse_cache):
    # A shadow armed at the PREVIOUS miss is consumed instantly on this one
    # (no synchronous spawn), and a replacement is restocked for the next
    # miss with the worker-side weight-cache overlay.
    shadow = _FakePool(2, wait_shutdown=True, env_overrides={"TRTLLM_HF_WEIGHT_CACHE": "1"})
    reuse_cache.prefetch.shadow = shadow
    s = reuse_cache.acquire(_FakePool, 2)
    assert s._real is shadow
    assert s._real._reuse_uses == 0  # adopted as a cache-managed pool
    n, overlay = reuse_cache.prefetch.restocks[-1]
    assert n == 2 and overlay.get("TRTLLM_HF_WEIGHT_CACHE") == "1"


def test_cache_miss_falls_back_to_sync_spawn_and_restocks(reuse_cache):
    s = reuse_cache.acquire(_FakePool, 2)  # nothing armed: sync spawn
    assert isinstance(s._real, _FakePool) and s._real.wait_shutdown
    assert s._real.env_overrides.get("TRTLLM_HF_WEIGHT_CACHE") == "1"
    assert reuse_cache.prefetch.restocks  # shadow armed for the NEXT miss


def test_spawn_failure_retries_once(reuse_cache):
    # wait_shutdown spawns fail closed (identity collection must complete);
    # one loud retry absorbs a transient slow node, a second failure means
    # the node is genuinely broken and must propagate.
    calls = []

    class _FlakyPool(_FakePool):
        def __init__(self, n_workers, wait_shutdown=False, env_overrides=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("identity collection incomplete")
            super().__init__(n_workers, wait_shutdown, env_overrides)

    s = reuse_cache.acquire(_FlakyPool, 2)
    assert isinstance(s._real, _FlakyPool) and len(calls) == 2


def test_reuse_size_mismatch_builds_new(reuse_cache):
    s1 = reuse_cache.acquire(_FakePool, 2)
    s1.shutdown()
    s2 = reuse_cache.acquire(_FakePool, 4)
    assert s2._real.n_workers == 4 and s2._real is not s1._real


def test_reuse_env_change_retires_pool(reuse_cache, monkeypatch):
    s1 = reuse_cache.acquire(_FakePool, 2)
    real = s1._real
    s1.shutdown()
    monkeypatch.setenv("SOME_TEST_KNOB", "changed-after-spawn")
    s2 = reuse_cache.acquire(_FakePool, 2)
    assert s2._real is not real  # stale-env pool not handed out
    assert _wait(lambda: real.shut)  # and it was retired in the background


def test_reuse_syspath_change_retires_pool(reuse_cache, monkeypatch):
    s1 = reuse_cache.acquire(_FakePool, 2)
    real = s1._real
    s1.shutdown()
    monkeypatch.syspath_prepend("/oot/example/path")
    s2 = reuse_cache.acquire(_FakePool, 2)
    assert s2._real is not real  # frozen workers could not import from it


def test_reuse_max_uses_retires_pool(reuse_cache, monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_REUSE_MAX_USES", "2")
    s = reuse_cache.acquire(_FakePool, 2)
    first = s._real
    s.shutdown()
    s = reuse_cache.acquire(_FakePool, 2)  # use #2 (cap)
    assert s._real is first
    s.shutdown()
    s = reuse_cache.acquire(_FakePool, 2)  # over the cap -> fresh pool
    assert s._real is not first
    assert _wait(lambda: first.shut)


def test_reuse_abort_never_recycles(reuse_cache):
    s1 = reuse_cache.acquire(_FakePool, 2)
    real = s1._real
    s1.shutdown_abort()
    s2 = reuse_cache.acquire(_FakePool, 2)
    assert s2._real is not real


def test_reuse_failed_reset_rebuilds(reuse_cache, monkeypatch):
    s1 = reuse_cache.acquire(_FakePool, 2)
    real = s1._real
    s1.shutdown()

    def _boom(session, fn):
        raise RuntimeError("worker died")

    monkeypatch.setattr(session_reuse, "submit_sync_per_worker", _boom)
    s2 = reuse_cache.acquire(_FakePool, 2)  # health probe fails -> fresh pool
    assert s2._real is not real
    assert _wait(lambda: real.shut)


def test_suspended_gives_private_untracked_pool(reuse_cache):
    reuse_cache.suspend(True)
    s = reuse_cache.acquire(_FakePool, 2)
    assert isinstance(s, _FakePool)  # raw pool: the LLM owns and destroys it
    reuse_cache.suspend(False)


def test_disabled_under_xdist(monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_REUSE_SESSION", "1")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    assert not SessionReuseCache().enabled


def test_weight_cache_env_scoped_to_spawn(reuse_cache, monkeypatch):
    import os

    monkeypatch.delenv("TRTLLM_HF_WEIGHT_CACHE", raising=False)
    s = reuse_cache.acquire(_FakePool, 2)
    # Workers froze the cache env at spawn...
    assert s._real.spawn_env_weight_cache == "1"
    # ...but the suite's environment is untouched afterwards.
    assert "TRTLLM_HF_WEIGHT_CACHE" not in os.environ


def test_weight_cache_env_respects_user_setting(reuse_cache, monkeypatch):
    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "0")
    s = reuse_cache.acquire(_FakePool, 2)
    assert s._real.spawn_env_weight_cache == "0"  # explicit user value wins


def test_abort_after_release_never_kills_cached_pool(reuse_cache):
    # Late executor error paths can call shutdown_abort AFTER shutdown already
    # returned the pool to the cache; the pool may belong to the NEXT test by
    # then and must not be killed through the stale wrapper.
    s1 = reuse_cache.acquire(_FakePool, 2)
    real = s1._real
    s1.shutdown()  # released to cache
    s1.shutdown_abort()  # late abort on the stale wrapper: must be a no-op
    assert not real.shut
    s2 = reuse_cache.acquire(_FakePool, 2)
    assert s2._real is real  # still reusable


def test_disable_after_patch_bypasses_cache(reuse_cache, monkeypatch):
    # The kill switch must keep meaning "off" even after the seams were
    # patched: acquire() consults it on every call.
    s1 = reuse_cache.acquire(_FakePool, 2)
    s1.shutdown()
    monkeypatch.setenv("TRTLLM_TEST_REUSE_SESSION", "0")
    s2 = reuse_cache.acquire(_FakePool, 2)
    assert isinstance(s2, _FakePool)  # raw private pool, cache untouched


def test_drain_shuts_cached_pools(reuse_cache):
    s = reuse_cache.acquire(_FakePool, 2)
    real = s._real
    s.shutdown()
    reuse_cache.drain()
    assert real.shut


def test_autodeploy_nodeids_are_private():
    from test_common.session_reuse_hooks import _is_private_nodeid

    assert _is_private_nodeid(
        "accuracy/test_llm_api_autodeploy.py::TestModelRegistryAccuracy::"
        "test_autodeploy_from_registry[m-True]"
    )
    assert _is_private_nodeid(
        "examples/test_ad_guided_decoding.py::test_autodeploy_guided_decoding_main_json"
    )
    assert _is_private_nodeid("unittest/_torch/auto_deploy/unit/singlegpu/test_x.py::test_y")
    assert not _is_private_nodeid(
        "accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4_4gpus[a]"
    )
    assert not _is_private_nodeid(
        "unittest/_torch/speculative/test_eagle3.py::test_llama_eagle3[x]"
    )


def test_failed_item_fences_cached_pools(reuse_cache, monkeypatch):
    """After a failed item the next acquire must NOT reuse the cached pool."""
    from test_common import session_reuse_hooks as hooks

    monkeypatch.setattr(hooks, "REUSE", reuse_cache)

    s = reuse_cache.acquire(_FakePool, 2)
    real = s._real
    s.shutdown()  # pool returned to the cache

    class _FailedReport:
        failed = True
        nodeid = "tests/foo.py::test_bar"

    hooks.pytest_runtest_logreport(_FailedReport)
    hooks.pytest_runtest_logfinish(_FailedReport.nodeid, None)

    assert real.shut  # fence drained the cached pool
    s2 = reuse_cache.acquire(_FakePool, 2)
    assert s2._real is not real  # fresh pool, no reuse across the failure


class _FakeBarrierSession:
    """Fakes MpiPoolSession.submit for master-side submit_sync_per_worker tests.

    Worker-side behaviour (the barrier itself) needs real MPI and is covered
    by the multi-GPU validation runs; here each entry stands for one worker's
    future: a (rank, result) tuple, an Exception, or None (never completes,
    i.e. a wedged worker).
    """

    def __init__(self, outcomes):
        import concurrent.futures

        self.n_workers = len(outcomes)
        self._futures = []
        for outcome in outcomes:
            f = concurrent.futures.Future()
            if isinstance(outcome, Exception):
                f.set_exception(outcome)
            elif outcome is not None:
                f.set_result(outcome)
            self._futures.append(f)

    def submit(self, task, *args):
        return self._futures


def test_probe_returns_results_ordered_by_rank():
    from test_common.grouped_test_utils import submit_sync_per_worker

    session = _FakeBarrierSession([(2, "c"), (0, "a"), (1, "b")])
    assert submit_sync_per_worker(session, lambda: None) == ["a", "b", "c"]


def test_probe_times_out_on_wedged_worker():
    from test_common.grouped_test_utils import submit_sync_per_worker

    session = _FakeBarrierSession([(0, "a"), None])  # rank 1 never returns
    with pytest.raises(RuntimeError, match="timed out"):
        submit_sync_per_worker(session, lambda: None, timeout=0.2)


def test_probe_propagates_worker_exception():
    from test_common.grouped_test_utils import submit_sync_per_worker

    session = _FakeBarrierSession([(0, "a"), ValueError("worker died")])
    with pytest.raises(ValueError, match="worker died"):
        submit_sync_per_worker(session, lambda: None)


def test_probe_flags_missing_rank():
    from test_common.grouped_test_utils import submit_sync_per_worker

    session = _FakeBarrierSession([(0, "a"), (0, "a")])  # rank 1 never covered
    with pytest.raises(RuntimeError, match=r"ranks \[1\]"):
        submit_sync_per_worker(session, lambda: None)


def test_passing_item_keeps_reuse(reuse_cache, monkeypatch):
    """The fence must not fire for passing items; reuse stays intact."""
    from test_common import session_reuse_hooks as hooks

    monkeypatch.setattr(hooks, "REUSE", reuse_cache)

    s = reuse_cache.acquire(_FakePool, 2)
    real = s._real
    s.shutdown()

    class _PassedReport:
        failed = False
        nodeid = "tests/foo.py::test_ok"

    hooks.pytest_runtest_logreport(_PassedReport)
    hooks.pytest_runtest_logfinish(_PassedReport.nodeid, None)

    s2 = reuse_cache.acquire(_FakePool, 2)
    assert s2._real is real  # pool survived and was reused


def test_seam_shim_is_isinstance_transparent():
    # Reproduces the #16338 breakage class: library code doing
    # isinstance(x, MpiPoolSession) against the PATCHED seam attribute. A
    # plain function there raised TypeError and killed every LLM creation;
    # the shim must behave as a type that answers for the real class.
    class _Real:
        pass

    made = []
    shim = session_reuse._isinstance_transparent_shim(
        _Real, lambda *a, **k: (made.append((a, k)), _Real())[1]
    )
    obj = shim(2, key="v")  # construction still routed to the factory
    assert made == [((2,), {"key": "v"})]
    assert isinstance(obj, shim)  # instance checks answer for the real class
    assert isinstance(_Real(), shim)
    assert not isinstance(object(), shim)
    assert issubclass(_Real, shim)


def test_patched_library_seam_survives_isinstance(reuse_cache):
    # End to end against the real seam: whatever currently sits on
    # tensorrt_llm.executor.proxy.MpiPoolSession (the real class, or the
    # shim installed by the session-reuse plugin active in this test
    # session), the proxy.py isinstance pattern must not raise. On the
    # pre-fix code (function at the seam) this raises TypeError.
    proxy = pytest.importorskip("tensorrt_llm.executor.proxy")
    reuse_cache.install_pool_factory_if_loaded()
    assert isinstance(object(), proxy.MpiPoolSession) in (False, True)
    assert issubclass(type(object()), proxy.MpiPoolSession) in (False, True)
