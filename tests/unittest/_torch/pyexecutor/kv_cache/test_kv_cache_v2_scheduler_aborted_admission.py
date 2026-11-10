# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVCacheV2Scheduler: a failed first-chunk admission must not feed the pool-ratio auto-tuner.

`release_context_admission` frees the caches set up for an admission that did not go through. The
auto-tuner samples a KvCache when it closes, unless its id is stats-excluded (kvCache.cpp, close()).
Sampled, every retry of a waiting long prompt adds the aborted attempt's small capacity to the
averages, which can shrink the target for exactly the pools that prompt waits for, so it waits
for good while short requests keep arriving. The fake manager below mirrors the real teardown order:
free_resources -> _remove_kv_cache -> close() (samples unless excluded) -> clear_stats_excluded.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler, ScheduleAction


class _FakeImpl:
    """The stats-exclusion part of the C++ KvCacheManager."""

    def __init__(self):
        self.excluded: set[int] = set()

    def mark_stats_excluded(self, rid):
        self.excluded.add(rid)

    def clear_stats_excluded(self, rid):
        self.excluded.discard(rid)

    def is_stats_excluded(self, rid):
        return rid in self.excluded


class _FakeManager:
    """KVCacheManagerV2 as far as the release path sees it."""

    def __init__(self, fail_free: bool = False):
        self.impl = _FakeImpl()
        self.kv_cache_map: dict[int, object] = {}
        self.sampled: list[int] = []  # ids whose close() fed the tuner
        self.unsampled: list[int] = []  # ids closed while stats-excluded
        self.fail_free = fail_free

    def free_resources(self, req):
        rid = req.py_request_id
        if self.fail_free:
            raise RuntimeError(f"Cannot release KV cache while request {rid} is loading")
        # _remove_kv_cache: pop, close (sample unless excluded), clear the exclusion.
        self.kv_cache_map.pop(rid)
        (self.unsampled if self.impl.is_stats_excluded(rid) else self.sampled).append(rid)
        self.impl.clear_stats_excluded(rid)


class _Req:
    def __init__(self, rid: int, first_chunk: bool = True):
        self.py_request_id = rid
        self.is_first_context_chunk = first_chunk


def _scheduler(managers):
    """A KVCacheV2Scheduler with only what the admission/release path reads."""
    sched = object.__new__(KVCacheV2Scheduler)
    sched.kv_cache_manager, sched.draft_kv_cache_manager, sched.cross_kv_cache_manager = managers
    sched.tokens_per_block = 32
    sched.chunking_enabled = False
    sched._ctx_shortest_first = False
    return sched


@pytest.fixture
def rewinds(monkeypatch):
    calls = []
    monkeypatch.setattr(
        scheduler_v2,
        "rewind_context_after_cache_drop",
        lambda req, tpb: calls.append((req.py_request_id, tpb)),
    )
    return calls


@pytest.mark.parametrize("with_draft", [False, True])
def test_release_closes_caches_stats_excluded(rewinds, with_draft):
    target = _FakeManager()
    draft = _FakeManager() if with_draft else None
    sched = _scheduler((target, draft, None))
    for m in (target, draft):
        if m is not None:
            m.kv_cache_map[7] = object()

    sched.release_context_admission(_Req(7))

    for m in (target, draft):
        if m is not None:
            assert m.unsampled == [7] and m.sampled == []
            assert 7 not in m.kv_cache_map
            assert not m.impl.is_stats_excluded(7), "the mark must not outlive the cache"
    assert rewinds == [(7, 32)]


def test_release_skips_pools_without_a_cache(rewinds):
    target, cross = _FakeManager(), _FakeManager()
    target.kv_cache_map[3] = object()
    sched = _scheduler((target, None, cross))

    sched.release_context_admission(_Req(3))

    assert target.unsampled == [3]
    assert cross.unsampled == [] and cross.sampled == [] and not cross.impl.excluded
    assert rewinds == [(3, 32)]


def test_readmitted_request_is_sampled_at_completion(rewinds):
    """Only the aborted attempt is excluded: the same request's real close later still feeds the tuner."""
    target = _FakeManager()
    sched = _scheduler((target, None, None))
    target.kv_cache_map[5] = object()
    sched.release_context_admission(_Req(5))
    target.kv_cache_map[5] = object()  # admitted on a later iteration
    target.free_resources(_Req(5))  # completes: normal teardown, not via release_context_admission
    assert target.unsampled == [5] and target.sampled == [5]


def test_failed_free_does_not_leave_the_mark(rewinds):
    target = _FakeManager(fail_free=True)
    target.kv_cache_map[9] = object()
    sched = _scheduler((target, None, None))

    with pytest.raises(RuntimeError, match="loading"):
        sched.release_context_admission(_Req(9))

    assert 9 in target.kv_cache_map
    assert not target.impl.is_stats_excluded(9), "a kept cache's real close must still be sampled"
    assert rewinds == []


def test_failed_free_keeps_a_preexisting_mark(rewinds):
    target = _FakeManager(fail_free=True)
    target.kv_cache_map[9] = object()
    target.impl.mark_stats_excluded(9)  # e.g. a dummy cache
    sched = _scheduler((target, None, None))

    with pytest.raises(RuntimeError):
        sched.release_context_admission(_Req(9))

    assert target.impl.is_stats_excluded(9)


@pytest.mark.parametrize(
    "action, first_chunk, sampled_expected",
    [
        (ScheduleAction.SKIP, True, False),  # failed first-chunk admission: released, not sampled
        (ScheduleAction.STOP, True, False),
        (ScheduleAction.SCHEDULED, True, None),  # admitted: cache kept
        (ScheduleAction.SKIP, False, None),  # later chunk: not released here
    ],
)
def test_try_schedule_context_releases_failed_first_admission_unsampled(
    rewinds, monkeypatch, action, first_chunk, sampled_expected
):
    target = _FakeManager()
    target.kv_cache_map[11] = object()
    sched = _scheduler((target, None, None))
    monkeypatch.setattr(
        KVCacheV2Scheduler,
        "_try_schedule_context_full",
        lambda self, req, budget, preempt: (action, 0, False),
    )

    result = sched._try_schedule_context(
        _Req(11, first_chunk), budget=None, preempt_for_pages=lambda r: False
    )

    assert result[0] is action
    if sampled_expected is None:
        assert 11 in target.kv_cache_map and target.unsampled == [] and rewinds == []
    else:
        assert 11 not in target.kv_cache_map
        assert target.unsampled == [11] and target.sampled == []
        assert rewinds == [(11, 32)]
