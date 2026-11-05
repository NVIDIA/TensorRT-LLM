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
"""Aggregated helix under KV memory pressure, with real managers and schedulers.

Each simulated CP rank owns a real KVCacheManagerV2 and KVCacheV2Scheduler and
its own copy of every request, as the executor does. The driver stands in for
the executor's forward pass, so the test covers scheduling, allocation,
preemption and request state, not attention.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm._torch.utils import helix_local_len
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    ContextChunkingPolicy,
    KvCacheConfig,
)
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

CP_SIZE = 2
TOKENS_PER_BLOCK = 32
MAX_NEW_TOKENS = 64
# Pools are carved in fixed-size chunks, so the page size is what keeps the
# pool small: these dimensions give an 8-page pool per rank.
NUM_LAYERS = 4
NUM_KV_HEADS = 8
HEAD_DIM = 128
POOL_TOKENS = 256
POOL_PAGES = 8
MAX_ITERATIONS = 1000
MAX_PREEMPTIONS_PER_REQUEST = 3


@dataclass(frozen=True)
class Scenario:
    prompt_lens: dict
    max_num_tokens: int
    arrivals: dict = field(default_factory=dict)
    chunk_unit: Optional[int] = None
    expect_context_preemption: bool = False
    max_new_tokens: Optional[dict] = None
    max_batch_size: Optional[int] = None

    def new_tokens(self, request_id: int) -> int:
        return MAX_NEW_TOKENS if self.max_new_tokens is None else self.max_new_tokens[request_id]

    def batch_size(self) -> int:
        return self.max_batch_size or len(self.prompt_lens)


SCENARIOS = {
    # The four prefills fill the pool. Request 1 has the longest prompt, so it
    # is the first decode to need a new page, and the decode ladder preempts
    # the newest request on its behalf.
    "decode_pressure": Scenario(
        prompt_lens={1: 120, 2: 96, 3: 96, 4: 96},
        max_num_tokens=408,
    ),
    # Chunked prefill, as aggregated helix requires: request 3's long prompt
    # runs out of pages mid-prefill and preempts the newer request 4 on the
    # context path.
    "chunked_context_pressure": Scenario(
        prompt_lens={1: 96, 2: 96, 3: 224, 4: 96},
        max_num_tokens=64,
        arrivals={4: 8},
        chunk_unit=TOKENS_PER_BLOCK,
        expect_context_preemption=True,
    ),
    # More requests than batch slots, arriving over time. A newer request that
    # runs out of pages mid-prefill while an older one sits out the pass must
    # not preempt the older one: it would retake the pages next pass and
    # preempt the newer one in turn, over and over.
    "chunked_staggered_arrivals": Scenario(
        prompt_lens={1: 223, 2: 230, 3: 173, 4: 83, 5: 125},
        max_new_tokens={1: 124, 2: 71, 3: 139, 4: 20, 5: 140},
        max_num_tokens=128,
        max_batch_size=3,
        arrivals={1: 26, 2: 20, 4: 52},
        chunk_unit=TOKENS_PER_BLOCK,
        expect_context_preemption=True,
    ),
}


def _make_rank(cp_rank: int, scenario: Scenario) -> tuple[KVCacheManagerV2, KVCacheV2Scheduler]:
    distributed = Mock()
    distributed.allreduce.side_effect = lambda value, op: value
    with patch.object(Distributed, "get", return_value=distributed):
        manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=POOL_TOKENS, enable_block_reuse=False),
            CacheType.SELF,
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max(scenario.prompt_lens.values())
            + max(map(scenario.new_tokens, scenario.prompt_lens))
            + TOKENS_PER_BLOCK,
            max_batch_size=scenario.batch_size(),
            mapping=Mapping(
                world_size=CP_SIZE,
                rank=cp_rank,
                cp_size=CP_SIZE,
                cp_config={"cp_type": CpType.HELIX},
            ),
            dtype=DataType.HALF,
            is_disagg=False,
        )
    scheduler = KVCacheV2Scheduler(
        max_batch_size=scenario.batch_size(),
        max_num_tokens=scenario.max_num_tokens,
        kv_cache_manager=manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        ctx_chunk_config=(
            None
            if scenario.chunk_unit is None
            else (ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, scenario.chunk_unit)
        ),
        enable_recompute_pause=True,
    )
    return manager, scheduler


def _make_request(request_id: int, scenario: Scenario) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=scenario.new_tokens(request_id),
        input_tokens=list(range(request_id, request_id + scenario.prompt_lens[request_id])),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def _check_helix_fields(request: LlmRequest, cp_rank: int) -> None:
    """This step's token sits at the global position after the last token, and
    only its owner rank holds it."""
    pos = request.get_num_tokens(0) - 1
    owner = (pos // TOKENS_PER_BLOCK) % CP_SIZE
    assert request.py_helix_is_inactive_rank == (owner != cp_rank), (request.py_request_id, pos)
    assert request.seqlen_this_rank_cp == helix_local_len(
        pos, TOKENS_PER_BLOCK, CP_SIZE, cp_rank
    ) + (owner == cp_rank), (request.py_request_id, pos)


def _step(
    cp_rank: int,
    manager: KVCacheManagerV2,
    scheduler: KVCacheV2Scheduler,
    active: list,
    scenario: Scenario,
) -> tuple:
    """One executor iteration on one rank; returns what the rank decided."""
    out = scheduler.schedule_request(active, set())
    for request in out.recompute_paused_requests:
        request.reset_for_recompute(1 << 20)
    for request in out.context_requests:
        assert request.total_input_len_cp == request.prompt_len
    for request in out.generation_requests:
        _check_helix_fields(request, cp_rank)
    decided = (
        tuple((r.py_request_id, r.context_chunk_size) for r in out.context_requests),
        tuple(r.py_request_id for r in out.generation_requests),
        tuple(r.py_request_id for r in out.paused_requests),
        tuple(r.py_request_id for r in out.recompute_paused_requests),
    )

    batch = ScheduledRequests()
    for request in out.context_requests:
        batch.append_context_request(request)
    batch.generation_requests = list(out.generation_requests)

    # Forward pass: prefill chunks advance, every finished prefill and every
    # decode step samples one token.
    manager.update_context_resources(batch)
    for request in out.context_requests:
        request.move_to_next_context_chunk()
        if request.context_remaining_length == 0:
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.add_new_token(0, 0)
    for request in out.generation_requests:
        request.add_new_token(0, 0)
        request.py_decoding_iter += 1
    manager.update_resources(batch)

    for request in list(active):
        request_id = request.py_request_id
        if request.get_num_tokens(0) >= scenario.prompt_lens[request_id] + scenario.new_tokens(
            request_id
        ):
            request.state = LlmRequestState.GENERATION_COMPLETE
            manager.free_resources(request)
            active.remove(request)
    return decided


@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_agg_helix_preempts_identically_and_re_prefills_correctly(scenario_name: str) -> None:
    scenario = SCENARIOS[scenario_name]
    ranks = [_make_rank(cp_rank, scenario) for cp_rank in range(CP_SIZE)]
    context_preemptions = [0] * CP_SIZE
    for cp_rank, (manager, scheduler) in enumerate(ranks):
        # Aggregated helix provisions no tier below GPU, so preemption is the
        # pressure valve.
        assert not manager.has_cache_tier_below_gpu
        assert manager.get_kv_cache_stats().max_num_blocks == POOL_PAGES

        def record_context_preemption(
            *args, _preempt=scheduler._try_preempt_for_pages, _rank=cp_rank
        ):
            preempted = _preempt(*args)
            context_preemptions[_rank] += preempted
            return preempted

        scheduler._try_preempt_for_pages = record_context_preemption
    pending = [{i: _make_request(i, scenario) for i in scenario.prompt_lens} for _ in ranks]
    active = [[] for _ in ranks]

    preempted_at_tokens: dict[int, int] = {}
    preemptions: Counter = Counter()
    re_prefilled: set[int] = set()
    decoded_after_re_prefill: set[int] = set()
    for iteration in range(MAX_ITERATIONS):
        for rank_pending, rank_active in zip(pending, active):
            for request_id in list(rank_pending):
                if scenario.arrivals.get(request_id, 0) <= iteration:
                    rank_active.append(rank_pending.pop(request_id))
        if not any(active) and not any(pending):
            break
        decisions = [
            _step(cp_rank, manager, scheduler, rank_active, scenario)
            for cp_rank, ((manager, scheduler), rank_active) in enumerate(zip(ranks, active))
        ]
        assert all(d == decisions[0] for d in decisions), decisions
        ctx, gen, paused, recompute_paused = decisions[0]
        assert paused == ()

        for request in active[0]:
            if request.py_request_id in recompute_paused:
                preempted_at_tokens[request.py_request_id] = request.get_num_tokens(0)
                preemptions[request.py_request_id] += 1
                # A repeat preemption replays a longer prompt; check that one too.
                re_prefilled.discard(request.py_request_id)
                decoded_after_re_prefill.discard(request.py_request_id)
        for request_id, _ in ctx:
            if request_id in preempted_at_tokens and request_id not in re_prefilled:
                for rank_active in active:
                    request = next(r for r in rank_active if r.py_request_id == request_id)
                    # The replayed prompt is every token the request held when
                    # it was preempted, generated ones included, in full on
                    # every rank.
                    assert request.prompt_len == preempted_at_tokens[request_id]
                re_prefilled.add(request_id)
        decoded_after_re_prefill.update(re_prefilled.intersection(gen))

    assert not any(active) and not any(pending), "requests did not finish"
    assert preempted_at_tokens, "the pool never ran out, so nothing was preempted"
    assert decoded_after_re_prefill == set(preempted_at_tokens)
    assert context_preemptions[0] == context_preemptions[1]
    assert (context_preemptions[0] > 0) == scenario.expect_context_preemption
    # Requests taking turns preempting each other would preempt the same
    # ones over and over; each preemption here has to make lasting room.
    assert max(preemptions.values()) <= MAX_PREEMPTIONS_PER_REQUEST, preemptions
