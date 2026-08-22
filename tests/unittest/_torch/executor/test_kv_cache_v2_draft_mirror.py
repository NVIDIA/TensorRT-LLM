# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draft-KV-mirror admission veto for KVCacheManagerV2 (nvbugs/6621358).

The one-model draft mirror only allocates in ``_prepare_draft_resources``, which
runs *after* scheduling where a shortfall can no longer be deferred.  These
tests pin the two halves of the fix: admission must refuse a request the mirror
cannot afford (covering create, resume *and* growth), and the post-scheduling
call site must degrade to a deferral instead of raising.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

# Room for exactly this many tokens in the mirror; resize() beyond it fails the
# way the real pool does when it is out of pages.
_POOL_LIMIT = 64


def _manager(*, kv_reserve_draft_tokens: int = 0) -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.kv_compression_manages_history = False
    manager._kv_reserve_draft_tokens = kv_reserve_draft_tokens
    manager.max_total_draft_tokens = 3
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = {}
    manager._allocated_draft_lens = {}
    manager._probed_draft_gen_tokens = 0
    manager._stream = SimpleNamespace(cuda_stream=7)
    manager._restore_page_index_bufs = MagicMock()
    return manager


def _request(request_id: int = 1, *, draft_tokens: int = 3, prompt_len: int = 4096):
    """A generation request whose chunk cursors follow the *target* pair.

    ``use_draft_model`` flips ``context_current_position`` / ``context_chunk_size``
    to a second, draft-side pair that a one-model drafter never writes (it stays
    at ``prompt_len``); this double reproduces that switch so a reader of the
    wrong pair sizes itself to the whole prompt.
    """

    class _Req:
        def __init__(self):
            self.py_request_id = request_id
            self.py_draft_tokens = [0] * draft_tokens
            self.lora_task_id = None
            self.cache_salt = None
            self.is_dummy = False
            self.is_first_context_chunk = True
            self.use_draft_model = False
            self.py_disable_speculative_decoding = False
            self.is_disagg_generation_transmission_complete = False
            self.context_phase_params = None
            self._target_position = 0
            self._target_chunk = 32
            self._prompt_len = prompt_len

        @property
        def context_current_position(self):
            return self._prompt_len if self.use_draft_model else self._target_position

        @property
        def context_chunk_size(self):
            return self._prompt_len if self.use_draft_model else self._target_chunk

    return _Req()


def _cache(*, capacity: int = 0, active: bool = True, resumable: bool = True) -> MagicMock:
    """A mirror cache that refuses to grow past ``_POOL_LIMIT``."""
    cache = MagicMock()
    cache.capacity = capacity
    cache.is_active = active

    def _resize(target, history=None):
        if target > _POOL_LIMIT:
            return False
        cache.capacity = target
        return True

    cache.resize.side_effect = _resize
    cache.resume.return_value = resumable
    return cache


def _batch(*, context=(), generation=()):
    return SimpleNamespace(
        context_requests=list(context),
        generation_requests=list(generation),
        all_requests=lambda: list(context) + list(generation),
    )


# ---- admission veto: every failure mode prepare_resources can hit ----


def test_admission_refuses_when_mirror_cannot_be_created() -> None:
    manager = _manager()
    manager._create_kv_cache = MagicMock(return_value=None)

    assert manager.can_reserve_draft_generation(_request()) is False


def test_admission_refuses_when_mirror_resume_is_refused() -> None:
    """resume() returning False is GPU-pressure back-pressure, not an error."""
    manager = _manager()
    request = _request()
    manager.kv_cache_map[request.py_request_id] = _cache(capacity=8, active=False, resumable=False)

    assert manager.can_reserve_draft_generation(request) is False


def test_admission_refuses_when_mirror_cannot_grow() -> None:
    manager = _manager()
    request = _request()
    manager.kv_cache_map[request.py_request_id] = _cache(capacity=_POOL_LIMIT)

    assert manager.can_reserve_draft_generation(request) is False


def test_admission_admits_and_leaves_mirror_unchanged() -> None:
    """The check is a probe: prepare_resources still does the real allocation."""
    manager = _manager()
    request = _request()
    cache = _cache(capacity=8)
    manager.kv_cache_map[request.py_request_id] = cache

    assert manager.can_reserve_draft_generation(request) is True
    assert cache.capacity == 8


def test_probe_tally_stops_sequential_probes_overcommitting() -> None:
    """N probes must not all pass against the same free pages."""
    manager = _manager()
    caches = []
    for req_id in range(3):
        # Each step needs 4 more tokens (1 + 3 draft) on top of 56, and the pool
        # tops out at 64, so only the first two probes can be honoured.
        cache = _cache(capacity=56)
        caches.append(cache)
        manager.kv_cache_map[req_id] = cache

    verdicts = [manager.can_reserve_draft_generation(_request(i)) for i in range(3)]

    # Without the running tally each probe would see the same free pages and
    # every request would be admitted.
    assert verdicts == [True, True, False]
    assert [c.capacity for c in caches] == [56, 56, 56]


def test_begin_admission_pass_clears_the_tally() -> None:
    manager = _manager()
    manager._probed_draft_gen_tokens = 512

    manager.begin_draft_admission_pass()

    assert manager._probed_draft_gen_tokens == 0
    # A request that only fits with a cleared tally now passes.
    request = _request()
    manager.kv_cache_map[request.py_request_id] = _cache(capacity=56)
    assert manager.can_reserve_draft_generation(request) is True


def test_context_reservation_refuses_a_chunk_the_mirror_cannot_hold() -> None:
    manager = _manager()
    request = _request()
    cache = _cache(capacity=_POOL_LIMIT)
    manager.kv_cache_map[request.py_request_id] = cache

    # A chunk larger than the pool can hold; the scheduler suspends on False.
    assert manager.reserve_draft_context(request, _POOL_LIMIT * 2) is False
    assert cache.capacity == _POOL_LIMIT


def test_context_reservation_admits_within_capacity() -> None:
    manager = _manager()
    request = _request()
    cache = _cache(capacity=0)
    manager.kv_cache_map[request.py_request_id] = cache

    assert manager.reserve_draft_context(request, 32) is True
    assert cache.capacity == 32


def test_saturated_index_mapper_defers_instead_of_admitting() -> None:
    """A mirror that does not exist must not be scheduled as if it did."""
    manager = _manager()
    manager._create_kv_cache = MagicMock(return_value=None)

    assert manager._mirror_context_capacity(_request(), 32) is False


def test_created_mirror_carries_the_manager_stream() -> None:
    manager = _manager()
    cache = _cache(capacity=0)
    manager._create_kv_cache = MagicMock(return_value=cache)

    assert manager._mirror_context_capacity(_request(), 32) is True
    assert cache.cuda_stream == manager._stream.cuda_stream
    cache.stop_committing.assert_called_once()


# ---- post-scheduling call site: defer, never raise ----


def test_prepare_resources_sizes_context_from_the_target_cursors() -> None:
    """Must follow the target's chunking, not the draft pair's prompt_len."""
    manager = _manager()
    request = _request(prompt_len=4096)
    cache = _cache(capacity=0)
    manager.kv_cache_map[request.py_request_id] = cache

    manager._prepare_draft_resources(_batch(context=[request]))

    # position 0 + chunk 32 + 3 draft tokens; the whole 4096-token prompt would
    # have exceeded the pool outright.
    assert cache.capacity == 35


def test_prepare_resources_defers_a_context_shortfall_without_raising() -> None:
    manager = _manager()
    # A chunk cursor already past the pool limit, so the mirror cannot be sized.
    request = _request()
    request._target_position = _POOL_LIMIT * 2
    cache = _cache(capacity=8)
    manager.kv_cache_map[request.py_request_id] = cache

    manager._prepare_draft_resources(_batch(context=[request]))

    assert cache.capacity == 8


def test_prepare_resources_defers_a_generation_shortfall_without_raising() -> None:
    manager = _manager()
    request = _request()
    cache = _cache(capacity=_POOL_LIMIT)
    manager.kv_cache_map[request.py_request_id] = cache

    manager._prepare_draft_resources(_batch(generation=[request]))

    assert cache.capacity == _POOL_LIMIT


def test_prepare_resources_defers_a_resume_refusal_without_raising() -> None:
    """The third crash site: a suspended mirror refusing to resume."""
    manager = _manager()
    request = _request()
    manager.kv_cache_map[request.py_request_id] = _cache(capacity=8, active=False, resumable=False)

    manager._prepare_draft_resources(_batch(generation=[request]))


def test_prepare_resources_creates_a_missing_generation_mirror() -> None:
    manager = _manager()
    request = _request()
    cache = _cache(capacity=0)
    manager._create_kv_cache = MagicMock(return_value=cache)

    manager._prepare_draft_resources(_batch(generation=[request]))

    manager._create_kv_cache.assert_called_once()
    assert cache.capacity == 4


def test_prepare_resources_restores_use_draft_model_after_deferring() -> None:
    manager = _manager()
    request = _request()
    manager.kv_cache_map[request.py_request_id] = _cache(capacity=_POOL_LIMIT)

    manager._prepare_draft_resources(_batch(generation=[request]))

    assert request.use_draft_model is False


@pytest.mark.parametrize("reserve", [0, 8], ids=["no_reserve", "with_reserve"])
def test_generation_growth_includes_the_draft_reserve(reserve: int) -> None:
    manager = _manager(kv_reserve_draft_tokens=reserve)
    request = _request(draft_tokens=3)

    # 1 + 3 draft tokens, padded up to the configured reserve.
    assert manager._draft_gen_growth(request) == 4 + max(reserve - 3, 0)
