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
"""Session leases (block_reuse_config.max_session_leases).

The cpu_only tests drive the lease bookkeeping with fake caches. The others run
context-only turns of a conversation through a real cache manager and scheduler,
in PyExecutor order, and allocate device memory pools.
"""

import gc
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.session_lease import SessionLease, SessionLeases
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig

# ---------------------------------------------------------------------------
# Bookkeeping (CPU)
# ---------------------------------------------------------------------------


class _FakeCache:
    def __init__(self) -> None:
        self.closed = 0


class _Pool:
    """Utilization source and close sink for SessionLeases."""

    def __init__(self) -> None:
        self.utilization = 0.0
        self.closed: list[int] = []

    def close(self, request_id: int, kv_cache: _FakeCache) -> None:
        kv_cache.closed += 1
        self.closed.append(request_id)


def _leases(capacity: int = 4, max_utilization: float = 0.8) -> tuple[SessionLeases, _Pool]:
    pool = _Pool()
    return SessionLeases(capacity, max_utilization, lambda: pool.utilization, pool.close), pool


def _lease(leases: SessionLeases, key: str, request_id: int, tokens: list[int]) -> _FakeCache:
    """Admit and release a turn, then lease its cache with ``tokens`` as the committed prefix."""
    leases.admit(key, request_id, tokens + [0])
    assert leases.release(request_id) == key
    cache = _FakeCache()
    leases.put(key, SessionLease(request_id, cache, np.array(tokens, dtype=np.int32)))
    return cache


@pytest.mark.cpu_only
def test_turn_extending_the_leased_prefix_takes_it_over() -> None:
    leases, pool = _leases()
    cache = _lease(leases, "conv", 1, list(range(8)))

    prompt = list(range(8)) + [100, 101, 102]
    # peek is side-effect free.
    assert leases.peek("conv", prompt) == 8
    assert leases.peek("conv", prompt) == 8
    assert leases.peek("other", prompt) is None
    assert leases.peek(None, prompt) is None

    lease = leases.admit("conv", 2, prompt)
    assert lease is not None and lease.kv_cache is cache
    assert len(leases) == 0
    assert cache.closed == 0 and pool.closed == []
    assert leases.is_admitted(2)
    assert leases.counters["taken_over"] == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "prompt",
    [
        [0, 1, 2, 3, 50, 51, 52, 53, 54],  # diverges inside the leased prefix
        list(range(8)),  # no new token
        list(range(4)),  # shorter than the leased prefix
    ],
)
def test_prompt_that_does_not_extend_the_lease_closes_it(prompt: list[int]) -> None:
    leases, pool = _leases()
    cache = _lease(leases, "conv", 1, list(range(8)))

    assert leases.peek("conv", prompt) is None
    assert leases.admit("conv", 2, prompt) is None
    assert cache.closed == 1 and pool.closed == [1]
    assert len(leases) == 0
    assert leases.counters["mismatch"] == 1
    # The request still registers, so its own cache may lease when it finishes.
    assert leases.is_admitted(2)


@pytest.mark.cpu_only
def test_utilization_guard_closes_oldest_first() -> None:
    leases, pool = _leases(max_utilization=0.5)
    first = _lease(leases, "a", 1, [1, 2])
    second = _lease(leases, "b", 2, [3, 4])

    pool.utilization = 0.6
    # Every admission sheds before it looks for a lease.
    leases.admit("c", 3, [5, 6, 7])
    assert pool.closed == [1, 2]
    assert first.closed == 1 and second.closed == 1
    assert leases.counters["shed"] == 2

    # put sheds too, so a lease is never held above the bound.
    _lease(leases, "c", 4, [5, 6])
    assert len(leases) == 0
    assert pool.closed == [1, 2, 4]


@pytest.mark.cpu_only
def test_capacity_closes_oldest_first() -> None:
    leases, pool = _leases(capacity=2)
    for request_id, key in enumerate("abc", start=1):
        _lease(leases, key, request_id, [request_id])
    assert len(leases) == 2
    assert pool.closed == [1]
    assert leases.peek("a", [1, 9]) is None
    assert leases.peek("c", [3, 9]) == 1


@pytest.mark.cpu_only
def test_close_oldest_reports_whether_it_closed_one() -> None:
    leases, pool = _leases()
    _lease(leases, "a", 1, [1])
    _lease(leases, "b", 2, [2])
    assert leases.close_oldest()
    assert leases.close_oldest()
    assert not leases.close_oldest()
    assert pool.closed == [1, 2]


@pytest.mark.cpu_only
def test_only_the_newest_turn_of_a_conversation_may_lease() -> None:
    leases, _ = _leases()
    leases.admit("conv", 1, [1, 2, 3])
    leases.admit("conv", 2, [1, 2, 3, 4])
    # Turn 1 finishes after turn 2 was admitted: turn 2 holds the newer history.
    assert leases.release(1) is None
    assert leases.counters["stale"] == 1
    assert leases.release(2) == "conv"
    # Released once: a second release is a no-op.
    assert leases.release(2) is None
    assert leases.release(99) is None


@pytest.mark.cpu_only
def test_put_replaces_an_older_lease_of_the_same_conversation() -> None:
    leases, pool = _leases()
    old = _lease(leases, "conv", 1, [1, 2])
    new = _lease(leases, "conv", 2, [7, 8])
    assert old.closed == 1 and new.closed == 0
    assert len(leases) == 1
    assert leases.peek("conv", [7, 8, 9]) == 2


@pytest.mark.cpu_only
def test_close_all_closes_every_lease_and_forgets_admissions() -> None:
    leases, pool = _leases()
    _lease(leases, "a", 1, [1])
    _lease(leases, "b", 2, [2])
    leases.admit("c", 3, [3, 4])
    leases.close_all()
    assert len(leases) == 0
    assert sorted(pool.closed) == [1, 2]
    assert not leases.is_admitted(3)


@pytest.mark.cpu_only
def test_block_reuse_config_session_lease_fields() -> None:
    config = BlockReuseConfig()
    assert config.max_session_leases == 0
    assert config.session_lease_max_utilization == 0.8
    assert BlockReuseConfig(max_session_leases=8).max_session_leases == 8
    for policy in ("per_request", "per_conversation"):
        with pytest.raises(ValueError, match="max_session_leases"):
            BlockReuseConfig(policy=policy, max_session_leases=1)
    for utilization in (0.0, 1.5):
        with pytest.raises(ValueError):
            BlockReuseConfig(session_lease_max_utilization=utilization)


# ---------------------------------------------------------------------------
# Real cache manager (GPU)
# ---------------------------------------------------------------------------

TOKENS_PER_BLOCK = 32
MAX_SEQ_LEN = 512
MAX_NUM_TOKENS = 512

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)


@contextmanager
def real_manager(max_session_leases: int, max_utilization: float = 0.8) -> Iterator:
    import tensorrt_llm.bindings
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            enable_block_reuse=True,
            enable_partial_reuse=False,
            block_reuse_config=BlockReuseConfig(
                max_session_leases=max_session_leases,
                session_lease_max_utilization=max_utilization,
            ),
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
        vocab_size=32000,
        is_disagg=True,
    )
    try:
        yield manager
    finally:
        manager.shutdown()
        del manager
        gc.collect()
        torch.cuda.empty_cache()


def make_scheduler(manager):
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

    return KVCacheV2Scheduler(
        max_batch_size=4,
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )


def make_turn(
    request_id: int,
    tokens: list[int],
    conversation_id: str | None = "conv",
    context_only: bool = True,
):
    from tensorrt_llm._torch.pyexecutor.llm_request import (
        LlmRequest,
        LlmRequestType,
        SamplingConfig,
    )
    from tensorrt_llm.conversation_params import ConversationParams

    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
            if context_only
            else LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION
        ),
    )
    if conversation_id is not None:
        request.py_conversation_params = ConversationParams(conversation_id=conversation_id)
    return request


def prefill(manager, scheduler, request) -> int:
    """Run one request's context phase; return where its first chunk started (its reuse)."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    first_chunk_start = None
    while request.context_remaining_length > 0:
        output = scheduler.schedule_request([request], set())
        assert [r.py_request_id for r in output.context_requests] == [request.py_request_id]
        batch = ScheduledRequests()
        batch.append_context_request(request)
        manager.prepare_resources(batch)
        if first_chunk_start is None:
            first_chunk_start = request.context_current_position
        request.move_to_next_context_chunk()
        manager.update_context_resources(batch)
    return first_chunk_start


def finish(manager, request) -> None:
    """Release a context-only request the way the executor does once its KV was sent."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

    if request.is_context_only_request:
        request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
    manager.free_resources(request)


def leased_cache(manager, conversation_id: str = "conv"):
    leases = manager._session_leases._leases
    lease = next(lease for key, lease in leases.items() if key[0] == conversation_id)
    return lease.kv_cache


@requires_cuda
def test_next_turn_takes_over_the_leased_cache() -> None:
    with real_manager(max_session_leases=4) as manager:
        scheduler = make_scheduler(manager)
        turn1 = list(range(100))
        r1 = make_turn(1, turn1)
        assert prefill(manager, scheduler, r1) == 0
        finish(manager, r1)

        # Leased with whole blocks only; the partial last block stays uncommitted.
        assert len(manager._session_leases) == 1
        assert r1.py_request_id not in manager.kv_cache_map
        cache = leased_cache(manager)
        assert cache.is_active
        assert cache.num_committed_tokens == 96

        turn2 = turn1 + list(range(1000, 1040))
        r2 = make_turn(2, turn2)
        assert manager.probe_context_reuse(r2) == 96
        assert prefill(manager, scheduler, r2) == 96
        assert manager.kv_cache_map[r2.py_request_id] is cache
        assert manager._session_leases.counters["taken_over"] == 1
        finish(manager, r2)
        assert cache.num_committed_tokens == 128

        # Turn 3 differs inside turn 2's partial last block and still takes over.
        turn3 = turn2[:128] + list(range(2000, 2020))
        r3 = make_turn(3, turn3)
        assert prefill(manager, scheduler, r3) == 128
        assert manager.kv_cache_map[r3.py_request_id] is cache
        assert manager._session_leases.counters["taken_over"] == 2
        finish(manager, r3)


@requires_cuda
def test_prefix_mismatch_closes_the_lease_and_reuses_from_the_tree() -> None:
    with real_manager(max_session_leases=4) as manager:
        scheduler = make_scheduler(manager)
        turn1 = list(range(100))
        r1 = make_turn(1, turn1)
        prefill(manager, scheduler, r1)
        finish(manager, r1)
        cache = leased_cache(manager)

        # Diverges at token 40: the committed block [0, 32) still matches.
        turn2 = turn1[:40] + list(range(5000, 5060))
        r2 = make_turn(2, turn2)
        assert manager.probe_context_reuse(r2) == 32
        assert prefill(manager, scheduler, r2) == 32
        assert manager.kv_cache_map[r2.py_request_id] is not cache
        assert not cache.is_active
        assert manager._session_leases.counters["mismatch"] == 1
        finish(manager, r2)
        # Turn 2 leases its own cache in turn 1's place.
        assert leased_cache(manager).num_committed_tokens == 96


@requires_cuda
def test_utilization_guard_closes_leases_and_the_tree_keeps_their_blocks() -> None:
    with real_manager(max_session_leases=4, max_utilization=1e-6) as manager:
        scheduler = make_scheduler(manager)
        turn1 = list(range(100))
        r1 = make_turn(1, turn1)
        prefill(manager, scheduler, r1)
        finish(manager, r1)
        assert len(manager._session_leases) == 0
        assert manager._session_leases.counters["shed"] == 1

        r2 = make_turn(2, turn1 + list(range(1000, 1040)))
        assert prefill(manager, scheduler, r2) == 96
        assert manager._session_leases.counters["taken_over"] == 0
        finish(manager, r2)


@requires_cuda
def test_capacity_closes_the_oldest_conversation() -> None:
    with real_manager(max_session_leases=1) as manager:
        scheduler = make_scheduler(manager)
        for request_id, conversation_id in ((1, "a"), (2, "b")):
            request = make_turn(
                request_id, [request_id * 1000 + i for i in range(64)], conversation_id
            )
            prefill(manager, scheduler, request)
            finish(manager, request)
        assert [key[0] for key in manager._session_leases._leases] == ["b"]
        assert manager._session_leases.counters["shed"] == 1


@requires_cuda
@pytest.mark.parametrize(
    ("conversation_id", "context_only"),
    [(None, True), ("conv", False)],
    ids=["no_conversation_params", "aggregated_request"],
)
def test_requests_outside_the_lease_conditions_close_as_before(
    conversation_id: str | None, context_only: bool
) -> None:
    with real_manager(max_session_leases=4) as manager:
        scheduler = make_scheduler(manager)
        request = make_turn(1, list(range(100)), conversation_id, context_only)
        prefill(manager, scheduler, request)
        assert not manager._session_leases.is_admitted(request.py_request_id)
        finish(manager, request)
        assert len(manager._session_leases) == 0
        assert request.py_request_id not in manager.kv_cache_map


@requires_cuda
def test_session_leases_are_off_by_default() -> None:
    with real_manager(max_session_leases=0) as manager:
        assert manager._session_leases is None
        scheduler = make_scheduler(manager)
        turn1 = list(range(100))
        r1 = make_turn(1, turn1)
        prefill(manager, scheduler, r1)
        finish(manager, r1)
        assert r1.py_request_id not in manager.kv_cache_map

        r2 = make_turn(2, turn1 + list(range(1000, 1040)))
        assert prefill(manager, scheduler, r2) >= 96
        finish(manager, r2)
