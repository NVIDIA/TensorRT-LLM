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
"""KvHostHelper windows and the KVCacheManagerV2 work it runs (deferred closes, context claims ahead,
context prefetches), on mock KvCaches. No GPU required."""

import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import host_helper as host_helper_mod
from tensorrt_llm._torch.pyexecutor.kv_cache.host_helper import (
    CLAIM,
    PREFETCH,
    RELEASE,
    KvHostHelper,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL

pytestmark = pytest.mark.cpu_only


def _helper() -> KvHostHelper:
    """A helper driven by `run_one` on the test thread."""
    return KvHostHelper(None, start_thread=False)


# A protective timeout for the threaded tests: far beyond what any of them needs.
_TIMEOUT_S = 30


class _ObservedLock:
    """The helper's item lock, signalling when a thread other than the helper's starts a blocking
    acquire: a test releases the running item only once the executor-side call waits for it."""

    def __init__(self, helper_thread_name: str) -> None:
        self._lock = threading.Lock()
        self._helper = helper_thread_name
        self.waiting = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking and threading.current_thread().name != self._helper:
            self.waiting.set()
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _release_when_waiting(lock: _ObservedLock, release: threading.Event) -> threading.Thread:
    def run():
        lock.waiting.wait(_TIMEOUT_S)
        release.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class _Clock:
    """time.monotonic for the helper, moved by the test."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# KvHostHelper
# ---------------------------------------------------------------------------


def test_nothing_runs_outside_a_window() -> None:
    helper = _helper()
    ran = []
    for kind in (RELEASE, CLAIM, PREFETCH):
        helper.submit(kind, lambda kind=kind: ran.append(kind))

    assert helper.run_one() is False
    helper.begin_wait()
    helper.end_wait()
    helper.begin_forward(may_capture=True)
    assert helper.run_one() is False
    helper.end_forward()
    assert ran == []


def test_the_wait_window_runs_releases_only() -> None:
    helper = _helper()
    ran = []
    helper.submit(CLAIM, lambda: ran.append("claim"))
    helper.submit(PREFETCH, lambda: ran.append("prefetch"))
    helper.submit(RELEASE, lambda: ran.append("release"))

    helper.begin_wait()
    while helper.run_one():
        pass

    assert ran == ["release"]
    assert helper.num_pending(CLAIM) == 1 and helper.num_pending(PREFETCH) == 1


def test_a_forward_without_capture_runs_every_kind_in_priority_order() -> None:
    helper = _helper()
    ran = []
    helper.submit(PREFETCH, lambda: ran.append("prefetch"))
    helper.submit(CLAIM, lambda: ran.append("claim"))
    helper.submit(RELEASE, lambda: ran.append("release"))

    helper.begin_forward(may_capture=False)
    while helper.run_one():
        pass

    assert ran == ["release", "claim", "prefetch"]
    assert helper.num_run == [1, 1, 1]


@pytest.mark.parametrize("kind", [CLAIM, PREFETCH])
def test_claims_and_prefetches_start_only_early_in_a_forward(monkeypatch, kind) -> None:
    clock = _Clock()
    monkeypatch.setattr(host_helper_mod, "time", SimpleNamespace(monotonic=clock.monotonic))
    helper = KvHostHelper(None, start_thread=False, forward_start_fraction=0.5)
    helper.begin_forward(may_capture=False)
    clock.now += 0.01  # one 10 ms forward: the cutoff is 5 ms into the next
    helper.end_forward()
    assert helper._forward_ema == pytest.approx(0.01)
    ran = []
    helper.submit(kind, lambda: ran.append("early"))
    helper.submit(RELEASE, lambda: ran.append("release"))

    helper.begin_forward(may_capture=False)
    clock.now += 0.005  # at the cutoff: too late for it
    while helper.run_one():
        pass
    assert ran == ["release"]
    helper.end_forward()  # a 5 ms forward: the average is now 9 ms, the cutoff 4.5 ms in

    helper.begin_forward(may_capture=False)
    clock.now += 0.0044  # just before the cutoff
    assert helper.run_one() is True
    assert ran == ["release", "early"]


def test_a_forward_that_may_capture_waits_for_the_running_item() -> None:
    helper = KvHostHelper(None, name="kv-host-helper-test")
    helper._item_lock = lock = _ObservedLock("kv-host-helper-test")
    started, release = threading.Event(), threading.Event()
    done = []

    def item():
        started.set()
        release.wait(_TIMEOUT_S)
        done.append(True)

    try:
        helper.submit(RELEASE, item)
        helper.begin_wait()
        assert started.wait(_TIMEOUT_S)
        helper.end_wait()
        _release_when_waiting(lock, release)

        helper.begin_forward(may_capture=True)

        assert done == [True]
        assert helper.num_forward_waits == 1
    finally:
        helper.stop()


def test_an_item_error_is_raised_on_the_executor_thread() -> None:
    helper = _helper()

    def failing():
        raise RuntimeError("close failed")

    helper.submit(RELEASE, failing)
    helper.begin_wait()
    assert helper.run_one() is True

    with pytest.raises(RuntimeError, match="close failed"):
        helper.begin_forward(may_capture=True)
    helper.begin_forward(may_capture=True)  # raised once


def test_drain_runs_inline_and_drops_the_queued_items() -> None:
    helper = _helper()
    ran = []
    helper.submit(RELEASE, lambda: ran.append("ticket"))

    helper.drain(RELEASE, lambda: ran.append("inline"))

    assert ran == ["inline"]
    assert helper.num_pending(RELEASE) == 0


def test_the_thread_runs_items_and_stops() -> None:
    helper = KvHostHelper(None)
    done = threading.Event()
    try:
        helper.submit(RELEASE, done.set)
        helper.begin_wait()
        assert done.wait(_TIMEOUT_S)
    finally:
        helper.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        helper.submit(RELEASE, done.set)


# ---------------------------------------------------------------------------
# Deferred closes on the helper
# ---------------------------------------------------------------------------


def _manager(request_ids=(), is_draft=False, cls=KVCacheManagerV2):
    """A manager whose release and claim paths run for real on mock KvCaches."""
    manager = object.__new__(cls)
    caches = {rid: Mock(beam_width=1, locality_domain_id=0) for rid in request_ids}
    manager.kv_cache_map = dict(caches)
    manager.impl = Mock()
    manager.index_mapper = Mock()
    manager.index_mapper.num_free_slots.return_value = 8
    manager.num_pools = 1
    manager.sparse_metadata_batch = None
    manager.is_draft = is_draft
    manager.kv_connector_manager = None
    manager.conversation_manager = None
    manager.enable_block_reuse = True
    manager.tokens_per_block = 4
    manager._has_cp_helix = False
    manager._early_freed_index_requests = set()
    manager._mixed_precision_low_prefix_len = {}
    manager._deferred_closes = deque()
    manager._pending_preemption = {}
    manager._allocated_draft_lens = {}
    manager._inflight_draft_lens = {}
    manager._request_stats_enabled_ids = set()
    manager._fresh_pages_filled = {}
    manager._disagg_receive_ready = {}
    manager._on_allocation_stream = set()
    manager._allocation_ready = {}
    return manager, caches


def _finished(request_id, *, generation_only=False, context_only=False):
    return Mock(
        py_request_id=request_id,
        is_generation_only_request=generation_only,
        is_context_only_request=context_only,
        is_dummy=False,
    )


def test_deferred_closes_run_on_the_helper_in_a_wait() -> None:
    manager, caches = _manager([7])
    manager.enable_deferred_close(True)
    helper = manager.enable_host_helper(None, releases=True, start_thread=False)

    manager.free_resources(_finished(7, generation_only=True))

    caches[7].close.assert_not_called()
    assert helper.num_pending(RELEASE) == 1
    helper.begin_wait()
    assert helper.run_one() is True
    caches[7].close.assert_called_once_with()
    manager.impl.clear_stats_excluded.assert_called_once_with(7)
    # Its pages were back before a scheduler could run out: a flush does not report it, so a
    # context request out of pages preempts instead of waiting for pages that are not coming.
    assert manager.flush_deferred_closes() is False


def test_a_flush_reports_the_close_the_helper_was_running() -> None:
    manager, caches = _manager([7])
    manager.enable_deferred_close(True)
    helper = manager.enable_host_helper(None, releases=True)
    helper._item_lock = lock = _ObservedLock(helper._name)
    started, release = threading.Event(), threading.Event()

    def close():
        started.set()
        release.wait(_TIMEOUT_S)

    caches[7].close.side_effect = close
    try:
        manager.free_resources(_finished(7, generation_only=True))
        helper.begin_wait()
        assert started.wait(_TIMEOUT_S)
        helper.end_wait()
        _release_when_waiting(lock, release)

        # The close finished while the flush waited, so its pages are new to the caller.
        assert manager.flush_deferred_closes() is True
        assert manager.flush_deferred_closes() is False
    finally:
        manager.stop_host_helper()


def test_a_flush_closes_what_the_helper_has_not_started() -> None:
    manager, caches = _manager([7, 8])
    manager.enable_deferred_close(True)
    helper = manager.enable_host_helper(None, releases=True, start_thread=False)
    manager.free_resources(_finished(7, generation_only=True))
    manager.free_resources(_finished(8, generation_only=True))

    assert manager.flush_deferred_closes() is True

    caches[7].close.assert_called_once_with()
    caches[8].close.assert_called_once_with()
    helper.begin_wait()
    assert helper.run_one() is False  # its tickets were dropped
    assert not manager._deferred_closes


def test_context_only_requests_are_deferred_only_when_asked() -> None:
    manager, caches = _manager([7, 8])
    manager.enable_deferred_close(True)
    manager.free_resources(_finished(7, context_only=True))
    caches[7].close.assert_called_once_with()

    manager.enable_deferred_close(True, context_only=True)
    manager.free_resources(_finished(8, context_only=True))
    caches[8].close.assert_not_called()
    assert [rid for rid, _ in manager._deferred_closes] == [8]


def test_context_only_deferral_alone_leaves_generation_requests_immediate() -> None:
    manager, caches = _manager([7, 8])
    manager.enable_deferred_close(True, context_only=True, generation_only=False)

    manager.free_resources(_finished(7, generation_only=True))
    manager.free_resources(_finished(8, context_only=True))

    caches[7].close.assert_called_once_with()
    caches[8].close.assert_not_called()
    assert [rid for rid, _ in manager._deferred_closes] == [8]


def test_a_deferred_release_forgets_the_request_at_once() -> None:
    manager, _ = _manager([7])
    manager.impl.num_locality_domains = 2  # read through the impl
    manager._locality_domain_load = [1, 0]
    manager._mixed_precision_low_prefix_len[7] = 16
    manager.enable_deferred_close(True)

    manager.free_resources(_finished(7, generation_only=True))

    assert manager._locality_domain_load == [0, 0]
    assert 7 not in manager._mixed_precision_low_prefix_len


def test_the_helper_does_not_release_for_a_cache_lender() -> None:
    manager, _ = _manager([7])
    manager._sharing = Mock()
    manager.enable_deferred_close(True)

    assert manager.enable_host_helper(None, releases=True, start_thread=False) is None
    assert manager._helper_releases is False


def test_stopping_the_helper_runs_the_pending_closes() -> None:
    manager, caches = _manager([7])
    manager.enable_deferred_close(True)
    manager.enable_host_helper(None, releases=True, start_thread=False)
    manager.free_resources(_finished(7, generation_only=True))

    manager.stop_host_helper()

    caches[7].close.assert_called_once_with()
    assert manager._host_helper is None


# ---------------------------------------------------------------------------
# Context claims ahead
# ---------------------------------------------------------------------------


def _context_request(request_id, num_tokens=16):
    req = Mock(
        py_request_id=request_id,
        is_first_context_chunk=True,
        is_dummy=False,
        is_disagg_generation_init_state=False,
        lora_task_id=None,
        cache_salt=None,
        return_perf_metrics=False,
        kv_cache_retention_config=None,
    )
    req.tokens = list(range(num_tokens))
    return req


def _claim_manager(limit=2):
    manager, _ = _manager()
    manager.mixed_precision_reuse = "strict"
    manager._context_reuse_tokens = lambda req, limit=None: req.tokens
    manager.context_reuse_claim_limit = lambda req: None
    manager._context_create_args = lambda req: ("scope", {"expected_prompt_length": 17})
    helper = manager.enable_host_helper(None, claim_ahead=limit, start_thread=False)
    manager._claim_stream = Mock(cuda_stream=42)
    return manager, helper


def test_a_claim_is_created_and_resumed_on_the_helper() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True

    assert manager.claim_context_ahead([_context_request(5)]) == 1
    manager.impl.create_kv_cache.assert_not_called()
    helper.begin_forward(may_capture=False)
    assert helper.run_one() is True

    manager.impl.create_kv_cache.assert_called_once_with(
        "scope", list(range(16)), id=5, expected_prompt_length=17
    )
    kv_cache.resume.assert_called_once_with(42)
    assert manager._take_claim(5, 16) is kv_cache
    assert 5 not in manager._claims


def test_a_claim_for_another_lookup_is_closed() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()

    assert manager._take_claim(5, 12) is None

    kv_cache.close.assert_called_once_with()
    manager.impl.clear_stats_excluded.assert_called_once_with(5)


def test_a_queued_claim_is_withdrawn_when_its_request_is_scheduled_first() -> None:
    manager, helper = _claim_manager()
    manager.claim_context_ahead([_context_request(5)])

    assert manager._take_claim(5, 16) is None

    assert helper.num_pending(CLAIM) == 0  # its item left the helper's queue
    helper.begin_forward(may_capture=False)
    assert helper.run_one() is False
    manager.impl.create_kv_cache.assert_not_called()


def test_a_claim_that_cannot_resume_leaves_the_request_to_the_scheduler() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = False
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()

    assert manager._take_claim(5, 16) is None
    kv_cache.close.assert_called_once_with()


def test_claims_are_bounded_and_age_out() -> None:
    manager, helper = _claim_manager(limit=2)
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    requests = [_context_request(rid) for rid in (1, 2, 3)]

    assert manager.claim_context_ahead(requests) == 2
    assert sorted(manager._claims) == [1, 2]
    helper.begin_forward(may_capture=False)
    while helper.run_one():
        pass
    for _ in range(4):
        manager.claim_context_ahead([])
    assert sorted(manager._claims) == [1, 2]

    manager.claim_context_ahead([])  # a fifth pass

    assert manager._claims == {}
    assert kv_cache.close.call_count == 2
    # A request whose claim aged out (a demoted long prefill) is not claimed again.
    assert manager.claim_context_ahead(requests) == 1
    assert sorted(manager._claims) == [3]


def test_releasing_claims_gives_their_pages_back() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()

    assert manager.release_claims_ahead() is True
    kv_cache.close.assert_called_once_with()
    assert manager.release_claims_ahead() is False


def test_claims_released_for_pages_are_not_made_again_at_once() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = False  # no pages: the claim holds nothing
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()

    # A failed claim gave nothing back: the scheduler preempts instead of retrying.
    assert manager.release_claims_ahead() is False
    for _ in range(4):
        assert manager.claim_context_ahead([_context_request(6)]) == 0
    assert manager.claim_context_ahead([_context_request(5), _context_request(6)]) == 1
    assert sorted(manager._claims) == [6]


def test_the_scheduler_reads_the_claim_room_and_state() -> None:
    """What scheduler_config.context_claim_hold reads: the room left, a parked claim, a
    given-up request, and no room while claims are paused after a release for pages."""
    manager, _ = _manager()
    assert manager.context_claim_limit == 0 and manager.context_claim_room() == 0
    manager, _ = _claim_manager(limit=3)
    assert manager.context_claim_limit == 3 and manager.context_claim_room() == 3
    manager.claim_context_ahead([_context_request(5)])
    assert manager.context_claim_room() == 2
    assert manager.has_context_claim(5) and not manager.has_context_claim(6)
    manager.release_claims_ahead()
    assert manager.context_claim_given_up(5) and not manager.has_context_claim(5)
    assert manager.context_claim_room() == 0  # paused


def test_committed_prompt_depth() -> None:
    manager, _ = _manager()  # tokens_per_block 4
    assert manager.committed_prompt_depth(10) == 8
    manager.enable_block_reuse = False
    assert manager.committed_prompt_depth(10) == 0


def test_a_running_claim_does_not_age_out_under_a_forward() -> None:
    manager, _ = _claim_manager()
    manager.claim_context_ahead([_context_request(5)])
    claim = manager._claims[5]
    claim.state = 1  # running on the helper

    for _ in range(6):
        manager.claim_context_ahead([])
    assert 5 in manager._claims

    claim.state = 3  # it failed
    manager.claim_context_ahead([])
    assert manager._claims == {}


def test_a_freed_request_drops_its_claim() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()

    manager.free_resources(_finished(5, context_only=True))

    kv_cache.close.assert_called_once_with()
    assert manager._claims == {}


def test_create_kv_cache_adopts_a_claim() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()
    manager.impl.create_kv_cache.reset_mock()
    manager.enable_stats = True
    manager._set_page_index_bufs = Mock()

    adopted = manager._create_kv_cache(5, None, list(range(16)))

    assert adopted is kv_cache
    manager.impl.create_kv_cache.assert_not_called()
    assert manager.kv_cache_map[5] is kv_cache
    manager.index_mapper.add_new_sequence.assert_called_once_with(5)
    manager._set_page_index_bufs.assert_called_once_with(5, kv_cache)


@pytest.mark.parametrize(
    "setting",
    [
        {"mixed_precision_reuse": "recompute"},
        {"impl.num_locality_domains": 2},
        {"_has_cp_helix": True},
        {"kv_connector_manager": Mock()},
        {"enable_block_reuse": False},
    ],
)
def test_claims_are_refused_where_a_pass_claims_differently(setting) -> None:
    manager, _ = _manager()
    manager.mixed_precision_reuse = "strict"
    for name, value in setting.items():
        owner, _, attr = name.rpartition(".")
        setattr(manager.impl if owner else manager, attr, value)

    assert manager.enable_host_helper(None, claim_ahead=2, start_thread=False) is None
    assert manager._claim_ahead_limit == 0


class _ReplayBacktrackManager(KVCacheManagerV2):
    """Creates and resumes context caches its own way, as CSA2 does."""

    def _create_kv_cache(self, request_id, lora_task_id, input_tokens, **kwargs):
        kwargs["optional_reuse_max_backtrack"] = 64
        return super()._create_kv_cache(request_id, lora_task_id, input_tokens, **kwargs)

    def _resume_context_cache(self, req, cache):
        return super()._resume_context_cache(req, cache)


class _LoggingManager(KVCacheManagerV2):
    """Only wraps the base create, as MambaHybridCacheManagerV2 does."""

    _context_claim_safe_overrides = ("_create_kv_cache",)

    def _create_kv_cache(self, request_id, lora_task_id, input_tokens, **kwargs):
        return super()._create_kv_cache(request_id, lora_task_id, input_tokens, **kwargs)


class _LoggingManagerWithOwnCreate(_LoggingManager):
    """Inherits the allowance, but overrides the create again: not covered by it."""

    def _create_kv_cache(self, request_id, lora_task_id, input_tokens, **kwargs):
        kwargs["optional_reuse_max_backtrack"] = 64
        return super()._create_kv_cache(request_id, lora_task_id, input_tokens, **kwargs)


@pytest.mark.parametrize(
    ("cls", "claim_ahead"),
    [
        (KVCacheManagerV2, 2),
        (_LoggingManager, 2),
        (_ReplayBacktrackManager, 0),
        (_LoggingManagerWithOwnCreate, 0),
    ],
)
def test_claims_need_the_base_context_cache_path(cls, claim_ahead) -> None:
    manager, _ = _manager(cls=cls)
    manager.mixed_precision_reuse = "strict"

    manager.enable_host_helper(None, claim_ahead=2, start_thread=False)

    assert manager._claim_ahead_limit == claim_ahead


def test_the_hybrid_mamba_manager_keeps_claims() -> None:
    """The serving manager of hybrid models (Qwen3.8) only logs in its create override."""
    mamba = pytest.importorskip(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2"
    )
    manager = object.__new__(mamba.MambaHybridCacheManagerV2)

    assert manager._context_claim_path_overrides() == []


def test_the_hybrid_mamba_manager_commits_at_its_from_end_snapshot() -> None:
    """A later prompt reuses a hybrid model's prefill up to the snapshot its from-end offset places
    (context_demote_no_walk's record depth); without one, nothing is taken as committed."""
    mamba = pytest.importorskip(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2"
    )
    from types import SimpleNamespace

    manager = object.__new__(mamba.MambaHybridCacheManagerV2)
    state = SimpleNamespace(additional_snapshot_offsets_from_end=[2, 5])
    manager.kv_cache_config = SimpleNamespace(enable_block_reuse=True, mamba_state_config=state)
    assert manager.committed_prompt_depth(3000) == 2998
    state.additional_snapshot_offsets_from_end = []
    assert manager.committed_prompt_depth(3000) == 0


# ---------------------------------------------------------------------------
# Context prefetch
# ---------------------------------------------------------------------------


def _prefetch_manager(limit):
    manager, _ = _manager()
    manager._context_reuse_tokens = lambda req, limit=None: req.tokens
    manager._derive_reuse_salt = lambda salt: None
    manager._can_evict = True  # a host or disk tier below the GPU
    helper = manager.enable_host_helper(None, prefetch_blocks=limit, start_thread=False)
    return manager, helper


def test_a_prefetch_brings_the_prefix_to_the_gpu_tier_in_a_forward() -> None:
    manager, helper = _prefetch_manager(limit=8)

    assert manager.offer_context_prefetch([_context_request(5, num_tokens=16)]) == 1
    helper.begin_wait()
    assert helper.run_one() is False  # not in a GPU wait
    helper.begin_forward(may_capture=False)
    assert helper.run_one() is True

    (scope, tokens, level), _ = manager.impl.prefetch_reuse.call_args
    assert tokens == list(range(16)) and level == GPU_LEVEL
    assert manager._prefetch_blocks == 4


def test_prefetched_blocks_stay_within_the_bound_until_claimed() -> None:
    manager, helper = _prefetch_manager(limit=6)
    first, second = _context_request(5, num_tokens=16), _context_request(6, num_tokens=16)

    assert manager.offer_context_prefetch([first, second]) == 1  # 4 + 4 blocks > 6
    manager._end_prefetch(5)  # its cache was created
    assert manager._prefetch_blocks == 0
    assert manager.offer_context_prefetch([second]) == 1


def test_a_prefetch_of_a_request_that_started_is_skipped() -> None:
    manager, helper = _prefetch_manager(limit=8)
    manager.offer_context_prefetch([_context_request(5)])

    manager.free_resources(_finished(5, context_only=True))
    helper.begin_forward(may_capture=False)
    helper.run_one()

    manager.impl.prefetch_reuse.assert_not_called()
    assert manager._prefetch_blocks == 0


def test_nothing_is_prefetched_without_a_tier_below_the_gpu() -> None:
    manager, _ = _prefetch_manager(limit=8)
    manager._can_evict = False

    assert manager.offer_context_prefetch([_context_request(5)]) == 0
    assert manager._prefetch_blocks == 0


def test_stopping_the_helper_closes_the_claims_and_forgets_the_prefetches() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()

    manager.stop_host_helper()

    assert kv_cache.mock_calls[-1] == call.close()
    assert manager._claims == {} and manager._claim_ahead_limit == 0


def test_stopping_the_helper_clears_a_live_prefetch_reservation() -> None:
    manager, _ = _manager()
    manager.mixed_precision_reuse = "strict"
    manager._context_reuse_tokens = lambda req, limit=None: req.tokens
    manager._derive_reuse_salt = lambda salt: None
    manager._can_evict = True
    helper = manager.enable_host_helper(None, claim_ahead=2, prefetch_blocks=8, start_thread=False)
    assert manager.offer_context_prefetch([_context_request(5, num_tokens=16)]) == 1
    assert manager._prefetch_blocks == 4 and helper.num_pending(PREFETCH) == 1
    prefetch = manager._prefetches[5]

    manager.stop_host_helper()

    assert manager._prefetches == {} and manager._prefetch_blocks == 0
    assert prefetch.cancelled
    manager.impl.prefetch_reuse.assert_not_called()


def test_creating_the_cache_cancels_a_queued_prefetch() -> None:
    manager, helper = _prefetch_manager(limit=8)
    manager.enable_stats = True
    manager._set_page_index_bufs = Mock()
    manager._impl_create_args = lambda *args, **kwargs: ("scope", {})
    manager.offer_context_prefetch([_context_request(5, num_tokens=16)])

    manager._create_kv_cache(5, None, list(range(16)))

    assert manager._prefetch_blocks == 0 and manager._prefetches == {}
    assert helper.num_pending(PREFETCH) == 0
    helper.begin_forward(may_capture=False)
    assert helper.run_one() is False
    manager.impl.prefetch_reuse.assert_not_called()


@pytest.mark.parametrize("outcome", ["nothing", "raises"])
def test_a_failed_prefetch_gives_its_blocks_back(outcome) -> None:
    manager, helper = _prefetch_manager(limit=4)
    if outcome == "raises":
        manager.impl.prefetch_reuse.side_effect = RuntimeError("no pages")
    else:
        manager.impl.prefetch_reuse.return_value = False
    first, second = _context_request(5, num_tokens=16), _context_request(6, num_tokens=16)
    assert manager.offer_context_prefetch([first, second]) == 1  # 4 blocks: the whole bound
    helper.begin_forward(may_capture=False)
    assert helper.run_one() is True

    # The second request takes the budget while the first one still waits.
    assert manager.offer_context_prefetch([first, second]) == 1
    assert sorted(manager._prefetches) == [5, 6]
    assert manager._prefetch_blocks == 4
    manager._end_prefetch(5)  # the failed one leaves: nothing more is given back
    assert manager._prefetch_blocks == 4


def test_a_successful_prefetch_keeps_its_blocks() -> None:
    manager, helper = _prefetch_manager(limit=4)
    manager.impl.prefetch_reuse.return_value = True
    first, second = _context_request(5, num_tokens=16), _context_request(6, num_tokens=16)
    manager.offer_context_prefetch([first])
    helper.begin_forward(may_capture=False)
    helper.run_one()

    assert manager.offer_context_prefetch([second]) == 0
    assert manager._prefetch_blocks == 4


def test_cancelled_claims_and_prefetches_leave_the_helper_queue() -> None:
    manager, _ = _manager()
    manager.mixed_precision_reuse = "strict"
    manager._context_reuse_tokens = lambda req, limit=None: req.tokens
    manager.context_reuse_claim_limit = lambda req: None
    manager._context_create_args = lambda req: ("scope", {})
    manager._derive_reuse_salt = lambda salt: None
    manager._can_evict = True
    helper = manager.enable_host_helper(None, claim_ahead=2, prefetch_blocks=8, start_thread=False)
    manager._claim_stream = Mock(cuda_stream=42)
    # Offered and freed again and again with no forward window in between.
    for rid in range(100):
        req = _context_request(rid)
        manager.claim_context_ahead([req])
        manager.offer_context_prefetch([req])
        manager.free_resources(_finished(rid, context_only=True))
        assert helper.num_pending(CLAIM) == 0 and helper.num_pending(PREFETCH) == 0

    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    live = _context_request(100)
    manager.claim_context_ahead([live])
    manager.offer_context_prefetch([live])
    helper.begin_forward(may_capture=False)
    while helper.run_one():
        pass
    manager.impl.create_kv_cache.assert_called_once()
    assert manager.impl.create_kv_cache.call_args.kwargs["id"] == 100
    manager.impl.prefetch_reuse.assert_called_once()


def test_a_claim_whose_resume_and_close_both_raise_is_not_adopted() -> None:
    manager, helper = _claim_manager()
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.side_effect = RuntimeError("resume failed")
    kv_cache.close.side_effect = RuntimeError("close failed")
    manager.claim_context_ahead([_context_request(5)])
    claim = manager._claims[5]
    helper.begin_forward(may_capture=False)
    assert helper.run_one() is True

    assert claim.state == 3 and claim.kv_cache is None  # _CLAIM_FAILED
    kv_cache.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="close failed"):
        helper.raise_pending_error()
    # The lookup creates its own cache.
    assert manager._take_claim(5, 16) is None


def test_an_expired_claim_that_the_helper_then_starts_does_nothing() -> None:
    """Claim expiry cancels a pending claim under the claim condition, so a helper that picks
    its item up right after the check finds it cancelled instead of being waited for."""
    manager, _ = _claim_manager()
    manager.claim_context_ahead([_context_request(5)])
    item = manager._claims[5].item
    for _ in range(4):
        manager.claim_context_ahead([])
    assert 5 in manager._claims

    manager.claim_context_ahead([])  # a fifth pass: expired, not waited for

    assert manager._claims == {} and manager.context_claim_given_up(5)
    item()  # the helper's late start
    manager.impl.create_kv_cache.assert_not_called()


def test_stopping_the_helper_finishes_after_a_failing_release() -> None:
    manager, caches = _manager([7, 8])
    manager.mixed_precision_reuse = "strict"
    manager._context_reuse_tokens = lambda req, limit=None: req.tokens
    manager.context_reuse_claim_limit = lambda req: None
    manager._context_create_args = lambda req: ("scope", {})
    manager.enable_deferred_close(True)
    helper = manager.enable_host_helper(None, releases=True, claim_ahead=2, start_thread=False)
    manager._claim_stream = Mock(cuda_stream=42)
    kv_cache = manager.impl.create_kv_cache.return_value
    kv_cache.resume.return_value = True
    manager.claim_context_ahead([_context_request(5)])
    helper.begin_forward(may_capture=False)
    helper.run_one()  # the claim is parked with its cache
    caches[7].close.side_effect = RuntimeError("release failed")
    manager.free_resources(_finished(7, generation_only=True))
    manager.free_resources(_finished(8, generation_only=True))

    with pytest.raises(RuntimeError, match="release failed"):
        manager.stop_host_helper()

    assert manager._host_helper is None and helper._stopped
    caches[8].close.assert_called_once_with()  # the close after the failing one still ran
    kv_cache.close.assert_called_once_with()  # and the parked claim was closed
    assert manager._claims == {} and not manager._deferred_closes
    manager.stop_host_helper()  # nothing left: a second call is a no-op


def test_stopping_the_helper_closes_every_claim_when_one_close_fails() -> None:
    manager, helper = _claim_manager(limit=2)
    caches = [Mock(), Mock()]
    manager.impl.create_kv_cache.side_effect = caches
    for cache in caches:
        cache.resume.return_value = True
    caches[0].close.side_effect = RuntimeError("close failed")
    manager.claim_context_ahead([_context_request(5), _context_request(6)])
    helper.begin_forward(may_capture=False)
    while helper.run_one():
        pass

    with pytest.raises(RuntimeError, match="close failed"):
        manager.stop_host_helper()

    caches[1].close.assert_called_once_with()
    assert manager._claims == {}


def test_a_helper_whose_device_setup_fails_does_not_start(monkeypatch) -> None:
    import torch

    def set_device(device):
        raise RuntimeError("no such device")

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    with pytest.raises(RuntimeError, match="could not start") as raised:
        KvHostHelper(0)
    assert "no such device" in str(raised.value.__cause__)
