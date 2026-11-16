# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation-step KV bookkeeping under the overlap executor.

The overlap loop schedules step N+1 (``try_allocate_generation``) before it
runs ``update_resources`` for step N. Each update must reclaim the unaccepted
draft slots of its own step, not of the step scheduled after it; otherwise a
change of draft width between consecutive steps leaves capacity below history
and ``KvCache::resize`` throws "History length cannot exceed capacity".
"""

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

MAX_DRAFT = 7
PROMPT = 1000


class _KvCache:
    """Enforces the two checks of the C++ ``KvCache::resize``."""

    def __init__(self, tokens: int) -> None:
        self.capacity = tokens
        self.history_length = tokens
        self.is_active = True

    def enter_decode(self) -> bool:
        return True

    def resume(self, *args, **kwargs) -> bool:
        self.is_active = True
        return True

    def resize(self, capacity=None, history_length=None) -> bool:
        new_capacity = self.capacity if capacity is None else capacity
        new_history = self.history_length if history_length is None else history_length
        if new_history < self.history_length:
            raise ValueError("History length cannot be decreased")
        if new_capacity < new_history:
            raise ValueError("History length cannot exceed capacity")
        self.capacity, self.history_length = new_capacity, new_history
        return True


def _manager() -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.kv_cache_type = CacheType.SELF
    manager.is_draft = False
    manager._has_cp_helix = False
    manager.kv_compression_manages_history = False
    manager._kv_reserve_draft_tokens = 0
    manager.max_total_draft_tokens = MAX_DRAFT
    manager._allocated_draft_lens = {}
    manager.kv_cache_map = {}
    manager._sharing = None
    manager._stream = SimpleNamespace(cuda_stream=0)
    manager._ensure_generation_beam_width = lambda req, kv_cache: True
    manager._fill_fresh_kv_pages = lambda request_id: None
    manager._log_window_crossing = lambda *args, **kwargs: None
    manager._restore_page_index_bufs = lambda *args, **kwargs: None
    return manager


def _request(request_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        py_draft_tokens=[],
        py_rewind_len=0,
        py_num_accepted_draft_tokens=0,
        py_num_accepted_draft_tokens_indices=None,
        max_beam_num_tokens=PROMPT + 1,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        is_disagg_generation_transmission_complete=False,
        context_phase_params=None,
        py_disable_speculative_decoding=False,
        is_dummy_request=False,
    )


@pytest.fixture(autouse=True)
def _disable_draft_token_relocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kv_cache_v2_module, "_update_kv_cache_draft_token_location", MagicMock())


class _OverlapDriver:
    """One request through the overlap executor's per-iteration order.

    ``step(width, accepted)``: schedule with ``width`` drafts, then update the
    previous step. ``revert(width)``: schedule, then the attention-DP vote
    skips the forward (revert, previous step stays pending). ``idle()``: not
    scheduled; the previous step is still updated. After every update the
    capacity must equal history plus ``1 + width`` per step still in flight.
    """

    def __init__(self) -> None:
        self.manager = _manager()
        self.request = _request()
        self.kv_cache = _KvCache(PROMPT)
        self.manager.kv_cache_map[self.request.py_request_id] = self.kv_cache
        self.batch = SimpleNamespace(generation_requests=[self.request], context_requests=[])
        self.pending = None
        self.in_flight = []

    def _allocate(self, width: int) -> None:
        self.request.py_draft_tokens = [0] * width
        assert self.manager.try_allocate_generation(self.request)
        self.in_flight.append(width)

    def _update(self, width: int, accepted: int) -> None:
        self.request.max_beam_num_tokens += 1 + accepted
        self.request.py_num_accepted_draft_tokens = accepted
        self.request.py_rewind_len = width - accepted
        self.request.py_draft_tokens = [0] * MAX_DRAFT
        self.manager.update_resources(self.batch)
        self.in_flight.pop(0)
        assert self.kv_cache.capacity == self.kv_cache.history_length + sum(
            1 + w for w in self.in_flight
        )

    def _flush(self) -> None:
        if self.pending is not None:
            self._update(*self.pending)
            self.pending = None

    def step(self, width: int, accepted: int) -> None:
        self._allocate(width)
        self._flush()
        self.pending = (width, accepted)

    def revert(self, width: int) -> None:
        self._allocate(width)
        self.manager.revert_allocate_generation(self.request)
        self.in_flight.pop()

    def idle(self) -> None:
        self._flush()

    def finish(self) -> None:
        self._flush()
        assert self.kv_cache.capacity == self.kv_cache.history_length


def test_draft_width_increase_reclaims_each_steps_own_slots() -> None:
    driver = _OverlapDriver()
    # Step 1 runs with no drafts; step 2, scheduled before step 1 is updated,
    # reserves 7. Updating step 1 must not reclaim step 2's 7 slots.
    driver.step(0, 0)
    driver.step(MAX_DRAFT, 0)
    assert driver.kv_cache.capacity == PROMPT + 1 + 1 + MAX_DRAFT
    driver.finish()
    assert driver.kv_cache.history_length == PROMPT + 2


def test_draft_width_decrease_does_not_leak_slots() -> None:
    driver = _OverlapDriver()
    driver.step(MAX_DRAFT, 0)
    driver.step(3, 1)
    driver.step(0, 0)
    driver.finish()


def test_reverted_step_drops_only_its_own_width() -> None:
    driver = _OverlapDriver()
    driver.step(3, 1)
    driver.revert(MAX_DRAFT)
    driver.step(MAX_DRAFT, 2)
    driver.finish()


def test_suspended_update_consumes_its_steps_width() -> None:
    driver = _OverlapDriver()
    driver.step(3, 0)
    driver.request.py_draft_tokens = [0] * MAX_DRAFT
    assert driver.manager.try_allocate_generation(driver.request)
    # Victim eviction between scheduling and the update of step 1: the update
    # skips the suspended cache, so step 1's slots stay reserved.
    driver.kv_cache.is_active = False
    driver.request.max_beam_num_tokens += 1
    driver.manager.update_resources(driver.batch)
    driver.kv_cache.is_active = True
    driver.request.max_beam_num_tokens += 1
    driver.request.py_num_accepted_draft_tokens = 0
    driver.request.py_rewind_len = MAX_DRAFT
    driver.manager.update_resources(driver.batch)
    # Step 2 reclaims its own 7 unaccepted slots; only step 1's 3 stay behind.
    assert driver.kv_cache.capacity == driver.kv_cache.history_length + 3
    assert not driver.manager._step_draft_lens[driver.request.py_request_id]


_EVENTS = (
    ("step", MAX_DRAFT, 0),
    ("step", MAX_DRAFT, MAX_DRAFT),
    ("step", 3, 1),
    ("step", 0, 0),
    ("idle",),
    ("revert", MAX_DRAFT),
    ("revert", 3),
)


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_every_short_schedule_keeps_capacity_exact(length: int) -> None:
    for schedule in itertools.product(_EVENTS, repeat=length):
        driver = _OverlapDriver()
        for kind, *args in schedule:
            getattr(driver, kind)(*args)
        driver.finish()


def test_unscheduled_manager_falls_back_to_runtime_draft_len() -> None:
    manager = _manager()
    request = _request()
    request.py_rewind_len = 3
    kv_cache = _KvCache(PROMPT)
    kv_cache.capacity = PROMPT + 1 + 3
    manager.kv_cache_map[request.py_request_id] = kv_cache
    request.max_beam_num_tokens += 1

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    assert (kv_cache.capacity, kv_cache.history_length) == (PROMPT + 1, PROMPT + 1)
