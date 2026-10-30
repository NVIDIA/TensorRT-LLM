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
"""A new context request waits for a prefix another request is prefilling.

The first tests drive PrefixWaitPlanner against a stand-in for the cache
manager's reuse probe and snapshot request. The rest run a real
KVCacheManagerV2 and KVCacheV2Scheduler in PyExecutor order (schedule,
prepare_resources, advance the context chunk, update_context_resources) and
allocate device memory pools.
"""

import gc
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.prefix_wait_planner import (
    MAX_WAIT_PASSES,
    MIN_GAIN_BLOCKS,
    PREFIX_UNIT_BLOCKS,
    PrefixWaitPlanner,
)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import (
    BlockReuseConfig,
    CapacitySchedulerPolicy,
    ContextChunkingPolicy,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

# ---------------------------------------------------------------------------
# PrefixWaitPlanner
# ---------------------------------------------------------------------------

PLANNER_TOKENS_PER_BLOCK = 4
UNIT_TOKENS = PREFIX_UNIT_BLOCKS * PLANNER_TOKENS_PER_BLOCK
MIN_GAIN_TOKENS = MIN_GAIN_BLOCKS * PLANNER_TOKENS_PER_BLOCK
SHARED = list(range(1, 2 * MIN_GAIN_TOKENS + 1))


class _CacheManager:
    """The reuse probe and snapshot request the planner asks the cache manager for."""

    tokens_per_block = PLANNER_TOKENS_PER_BLOCK

    def __init__(self, reusable=0, snapshot_ok=True):
        self.kv_cache_map = {}
        self.reusable = reusable
        self.snapshot_ok = snapshot_ok
        self.requested = []

    def probe_context_reuse(self, req):
        return self.reusable

    def request_prefix_snapshot(self, owner, num_tokens):
        self.requested.append((owner.py_request_id, num_tokens))
        return self.snapshot_ok


def _request(request_id, tokens, salt=None):
    return SimpleNamespace(
        py_request_id=request_id,
        get_tokens_view=lambda beam: tokens,
        prompt_len=len(tokens),
        context_current_position=0,
        context_remaining_length=len(tokens),
        is_first_context_chunk=True,
        is_dummy_request=False,
        lora_task_id=None,
        cache_salt=salt,
        multimodal_hashes=None,
    )


def _prefilling_owner(manager, planner, tokens=None):
    """A request the planner saw and the scheduler admitted, one chunk in."""
    owner = _request(1, SHARED + [7] * 40 if tokens is None else tokens)
    assert planner.plan([owner]) == {}
    manager.kv_cache_map[owner.py_request_id] = object()
    planner.add_owner(owner)
    owner.is_first_context_chunk = False
    owner.context_current_position = UNIT_TOKENS
    owner.context_remaining_length -= UNIT_TOKENS
    return owner


def test_waits_for_an_owner_and_requests_its_snapshot():
    manager = _CacheManager()
    planner = PrefixWaitPlanner(manager)
    owner = _prefilling_owner(manager, planner)

    assert planner.plan([owner, _request(2, SHARED + [9] * 40)]) == {2: None}
    assert manager.requested == [(1, len(SHARED))]


def test_waits_for_a_new_request_ahead_in_the_same_pass():
    manager = _CacheManager()
    planner = PrefixWaitPlanner(manager)
    first = _request(1, SHARED + [7] * 40)
    second = _request(2, SHARED + [9] * 40)
    third = _request(3, SHARED + [5] * 40)

    assert planner.plan([first, second, third]) == {2: 1, 3: 1}
    assert {owner for owner, _ in manager.requested} == {1}


@pytest.mark.parametrize(
    "reusable,snapshot_ok,salt,waits",
    [
        (0, True, None, True),
        # The reuse tree already offers most of the shared prefix.
        (len(SHARED) - MIN_GAIN_TOKENS + 1, True, None, False),
        # The owner cannot make the shared depth reusable.
        (0, False, None, False),
        # A different reuse scope never shares a prefix.
        (0, True, "other", False),
    ],
)
def test_wait_conditions(reusable, snapshot_ok, salt, waits):
    manager = _CacheManager(reusable=reusable, snapshot_ok=snapshot_ok)
    planner = PrefixWaitPlanner(manager)
    _prefilling_owner(manager, planner)

    assert (planner.plan([_request(2, SHARED + [9] * 40, salt=salt)]) == {2: None}) is waits


def test_short_shared_prefix_does_not_wait():
    manager = _CacheManager()
    planner = PrefixWaitPlanner(manager)
    short = list(range(1, MIN_GAIN_TOKENS))
    _prefilling_owner(manager, planner, tokens=short + [7] * 400)

    assert planner.plan([_request(2, short + [9] * 400)]) == {}


def test_wait_is_bounded():
    manager = _CacheManager()
    planner = PrefixWaitPlanner(manager)
    _prefilling_owner(manager, planner)
    waiter = _request(2, SHARED + [9] * 40)

    for _ in range(MAX_WAIT_PASSES):
        assert planner.plan([waiter]) == {2: None}
    assert planner.plan([waiter]) == {}


def test_owner_that_ended_its_context_is_not_waited_for_while_it_decodes():
    manager = _CacheManager()
    planner = PrefixWaitPlanner(manager)
    owner = _prefilling_owner(manager, planner)
    owner.context_remaining_length = 0

    # The owner still holds its cache claim.
    assert owner.py_request_id in manager.kv_cache_map
    assert planner.plan([_request(2, SHARED + [9] * 40)]) == {}


def test_requests_with_a_claim_or_a_later_chunk_never_wait():
    manager = _CacheManager()
    planner = PrefixWaitPlanner(manager)
    _prefilling_owner(manager, planner)
    claimed = _request(2, SHARED + [9] * 40)
    manager.kv_cache_map[2] = object()
    continuing = _request(3, SHARED + [5] * 40)
    continuing.is_first_context_chunk = False

    assert planner.plan([claimed, continuing]) == {}


# ---------------------------------------------------------------------------
# Real cache manager and scheduler
# ---------------------------------------------------------------------------

TOKENS_PER_BLOCK = 32
WINDOW = 64
MAX_SEQ_LEN = 4096
# Nine prefix units: deep enough to wait for.
SHARED_LEN = 9 * PREFIX_UNIT_BLOCKS * TOKENS_PER_BLOCK
TAIL_LEN = 256
# A shorter prefix that an earlier request leaves in the reuse tree. Once the
# owner commits it, the rest of the shared prefix is still worth waiting for.
FORK = SHARED_LEN - MIN_GAIN_BLOCKS * TOKENS_PER_BLOCK
# Below the prompt length, so a prefill takes budget-sized chunks.
CHUNK_BUDGET = 1024
# Layer 0 slides; layer 1 is full attention (a window of max_seq_len normalizes to None).
SWA_AND_FULL = [WINDOW, MAX_SEQ_LEN]
FULL_ONLY = [MAX_SEQ_LEN, MAX_SEQ_LEN]

SWA_SCRATCH_REUSE = dict(
    max_attention_window=SWA_AND_FULL,
    policy="all_reusable",
    swa_scratch_reuse=True,
    enable_branch_snapshot=True,
)


@contextmanager
def real_manager(
    *,
    max_attention_window: list[int],
    policy: str,
    swa_scratch_reuse: bool = False,
    enable_branch_snapshot: bool = False,
    enable_block_reuse: bool = True,
) -> Iterator[KVCacheManagerV2]:
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=16384,
            enable_block_reuse=enable_block_reuse,
            max_attention_window=max_attention_window,
            enable_swa_scratch_reuse=swa_scratch_reuse,
            block_reuse_config=BlockReuseConfig(
                policy=policy, enable_branch_snapshot=enable_branch_snapshot
            ),
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
    )
    try:
        yield manager
    finally:
        manager.shutdown()
        del manager
        gc.collect()
        torch.cuda.empty_cache()


def make_request(request_id: int, shared_len: int = SHARED_LEN, tail_len: int = TAIL_LEN):
    tail = [20000 + 1000 * request_id + i for i in range(tail_len)]
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(shared_len)) + tail,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


class Executor:
    """Runs scheduler passes over a real manager in PyExecutor order."""

    def __init__(self, manager: KVCacheManagerV2, max_num_tokens: int = CHUNK_BUDGET):
        self.manager = manager
        # The executor selects FORCE_CHUNK when enable_branch_snapshot is set;
        # every case uses it, so they differ only in the manager. Without the
        # prefix-aware skip, which defers a request by one pass behind a request
        # that computes the same block, only the prefix wait defers a request.
        self.scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(ContextChunkingPolicy.FORCE_CHUNK, TOKENS_PER_BLOCK),
            enable_prefix_aware_scheduling=False,
        )
        self.chunk_starts = defaultdict(list)
        self.claimed = {}

    def step(self, requests, inflight=frozenset(), run_forward=True) -> list[int]:
        """Schedule one pass and, with ``run_forward``, run its context chunks.

        Returns the ids of the context requests it admitted. Without
        ``run_forward`` the chunks stay allocated but not computed, as for a
        batch whose forward is still in flight.
        """
        pending = [r for r in requests if r.context_remaining_length > 0]
        admitted = self.scheduler.schedule_request(pending, set(inflight)).context_requests
        self.claimed.update((r.py_request_id, r) for r in admitted)
        if run_forward:
            batch = ScheduledRequests()
            for request in admitted:
                batch.append_context_request(request)
            self.manager.prepare_resources(batch)
            for request in admitted:
                self.chunk_starts[request.py_request_id].append(request.context_current_position)
                request.move_to_next_context_chunk()
            self.manager.update_context_resources(batch)
        return [r.py_request_id for r in admitted]

    def prefill(self, request) -> None:
        while request.context_remaining_length > 0:
            assert self.step([request]) == [request.py_request_id]

    def free_all(self) -> None:
        for request_id, request in self.claimed.items():
            if request_id in self.manager.kv_cache_map:
                self.manager.free_resources(request)


@requires_cuda
@pytest.mark.parametrize(
    ("config", "with_fork", "owner_ends_chunk_at_shared"),
    [
        # Every computed block is committed, at the latest at the owner's
        # context end; the sibling waits until then.
        (dict(max_attention_window=FULL_ONLY, policy="per_request"), False, False),
        # Only the SWA window at a chunk end gets real pages; the owner ends a
        # chunk at the shared depth.
        (SWA_SCRATCH_REUSE, False, True),
        # Only the SWA window at a prompt end or a branch point is committed.
        # With a fork, the owner also ends a chunk at its own branch point.
        (
            dict(
                max_attention_window=SWA_AND_FULL, policy="per_request", enable_branch_snapshot=True
            ),
            False,
            True,
        ),
        (
            dict(
                max_attention_window=SWA_AND_FULL, policy="per_request", enable_branch_snapshot=True
            ),
            True,
            True,
        ),
    ],
    ids=["full-attention", "swa-scratch-reuse", "swa-per-request", "swa-per-request-fork"],
)
def test_sibling_waits_for_the_owner_and_reuses_the_shared_prefix(
    config, with_fork, owner_ends_chunk_at_shared
):
    with real_manager(**config) as manager:
        assert manager.can_share_inflight_prefix
        # A budget above the shared depth, so an owner chunk ends there only for
        # a snapshot point: under all_reusable every chunk end is committed, and
        # earlier chunk ends would leave less than the minimum gain to wait for.
        executor = Executor(manager, max_num_tokens=4 * CHUNK_BUDGET)
        try:
            if with_fork:
                # Leaves the first FORK tokens in the reuse tree but not the SWA
                # window there, so the owner's lookup records a branch point.
                executor.prefill(make_request(3, shared_len=FORK))
            owner, sibling = make_request(1), make_request(2)

            # The sibling waits for the owner admitted ahead of it.
            assert executor.step([owner, sibling]) == [1]
            sibling_admitted_after_owner_context = None
            for _ in range(16):
                owner_context_done = owner.context_remaining_length == 0
                admitted = executor.step([owner, sibling])
                if 2 in admitted and sibling_admitted_after_owner_context is None:
                    sibling_admitted_after_owner_context = owner_context_done
                if sibling.context_remaining_length == owner.context_remaining_length == 0:
                    break
            assert sibling.context_remaining_length == owner.context_remaining_length == 0

            owner_starts = executor.chunk_starts[1]
            assert (SHARED_LEN in owner_starts) is owner_ends_chunk_at_shared
            assert (FORK in owner_starts) is with_fork
            assert sibling_admitted_after_owner_context is not owner_ends_chunk_at_shared
            # The sibling starts from the whole shared prefix.
            assert executor.chunk_starts[2][0] == SHARED_LEN
        finally:
            executor.free_all()


@requires_cuda
@pytest.mark.parametrize(
    ("config", "shares"),
    [
        (dict(max_attention_window=FULL_ONLY, policy="per_request"), True),
        (dict(max_attention_window=FULL_ONLY, policy="all_reusable", swa_scratch_reuse=True), True),
        # Every SWA block keeps its pages and is committed.
        (dict(max_attention_window=SWA_AND_FULL, policy="all_reusable"), True),
        (SWA_SCRATCH_REUSE, True),
        (
            dict(max_attention_window=FULL_ONLY, policy="all_reusable", enable_block_reuse=False),
            False,
        ),
        # Attention-only SWA whose window reaches the radix tree only at a chunk
        # end, without branch snapshots to end a chunk at the shared depth.
        (
            dict(max_attention_window=SWA_AND_FULL, policy="all_reusable", swa_scratch_reuse=True),
            False,
        ),
        (dict(max_attention_window=SWA_AND_FULL, policy="per_request"), False),
    ],
    ids=[
        "full-per-request",
        "full-scratch-reuse",
        "swa-all-reusable",
        "swa-scratch-reuse-branch-snapshot",
        "no-block-reuse",
        "swa-scratch-reuse-no-branch-snapshot",
        "swa-per-request-no-branch-snapshot",
    ],
)
def test_siblings_wait_only_where_the_cache_can_share_an_inflight_prefix(config, shares):
    with real_manager(**config) as manager:
        assert manager.can_share_inflight_prefix is shares
        executor = Executor(manager, max_num_tokens=4 * CHUNK_BUDGET)
        try:
            admitted = executor.step([make_request(1), make_request(2)])
            assert sorted(admitted) == ([1] if shares else [1, 2])
        finally:
            executor.free_all()


@requires_cuda
def test_a_released_owner_no_longer_holds_a_sibling_back():
    with real_manager(max_attention_window=FULL_ONLY, policy="per_request") as manager:
        executor = Executor(manager)
        try:
            owner, sibling = make_request(1), make_request(2)
            assert executor.step([owner]) == [1]
            assert executor.step([owner, sibling]) == [1]

            # The owner is cancelled mid-prefill and releases its cache.
            manager.free_resources(owner)
            assert executor.step([sibling]) == [2]
        finally:
            executor.free_all()


@requires_cuda
@pytest.mark.parametrize("owner_state", ["before", "passed", "chunk-allocated"])
def test_sibling_waits_only_while_the_owner_can_end_a_chunk_at_the_shared_depth(owner_state):
    """Under SWA scratch reuse the owner must end a chunk at the shared depth.
    It no longer can once it has computed past that depth, or holds the pages of
    a chunk across it whose forward has not run yet."""
    with real_manager(**SWA_SCRATCH_REUSE) as manager:
        executor = Executor(manager)
        try:
            owner = make_request(1, tail_len=3 * CHUNK_BUDGET + CHUNK_BUDGET // 2 - SHARED_LEN)
            sibling = make_request(2)
            # Two budget-sized chunks, below the shared depth.
            for _ in range(2):
                assert executor.step([owner]) == [1]
            assert owner.context_current_position < SHARED_LEN
            inflight = set()
            if owner_state == "passed":
                assert executor.step([owner]) == [1]
                assert owner.context_current_position > SHARED_LEN
            elif owner_state == "chunk-allocated":
                # The next chunk, across the shared depth, holds its pages while
                # its forward is in flight.
                assert executor.step([owner], run_forward=False) == [1]
                inflight = {owner.py_request_id}

            admitted = executor.step([owner, sibling], inflight=inflight, run_forward=False)
            assert (2 in admitted) is (owner_state != "before")
        finally:
            executor.free_all()
