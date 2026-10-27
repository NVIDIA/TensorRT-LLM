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
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
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
from tensorrt_llm.runtime.kv_cache_manager_v2 import CACHE_LEVEL1 as HOST_LEVEL
from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL

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


def _make_rank(
    cp_rank: int, scenario: Scenario, host_cache_size: int = 0
) -> tuple[KVCacheManagerV2, KVCacheV2Scheduler]:
    distributed = Mock()
    distributed.allreduce.side_effect = lambda value, op: value
    distributed.local_world_size = 1
    with patch.object(Distributed, "get", return_value=distributed):
        manager = KVCacheManagerV2(
            KvCacheConfig(
                max_tokens=POOL_TOKENS, host_cache_size=host_cache_size, enable_block_reuse=False
            ),
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
        # host_cache_size=0 leaves no tier below GPU, so recompute-pause is the
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


# The smallest host tier the manager provisions: four pages.
HOST_TIER_BYTES = 2 << 20


@pytest.mark.parametrize("cp_rank", range(CP_SIZE))
def test_suspend_frees_gpu_pages_until_the_host_tier_is_full(cp_rank: int) -> None:
    """Suspending a request moves its pages to the host tier and frees them on
    GPU, so a new request fits. Once the host tier is full, a suspended
    request's pages stay on GPU, and only recompute-pause can free them."""
    scenario = Scenario(prompt_lens={i: 96 for i in range(1, 17)}, max_num_tokens=96)
    manager, _ = _make_rank(cp_rank, scenario, host_cache_size=HOST_TIER_BYTES)
    requests = {}

    def allocate(request_id: int) -> bool:
        request = requests.setdefault(request_id, _make_request(request_id, scenario))
        return manager.prepare_context(request) and manager.resize_context(
            request, request.prompt_len
        )

    def free_gpu_pages() -> int:
        return manager.impl.get_storage_statistics(GPU_LEVEL)[0].free

    def free_host_pages() -> int:
        return manager.impl.get_storage_statistics(HOST_LEVEL)[0].free

    try:
        # Fill the GPU pool, measuring one request's pages on the first.
        gpu_pages = free_gpu_pages()
        host_pages = free_host_pages()
        assert allocate(1)
        pages_per_request = gpu_pages - free_gpu_pages()
        num_resident = gpu_pages // pages_per_request
        for request_id in range(2, num_resident + 1):
            assert allocate(request_id)
        assert free_gpu_pages() < pages_per_request
        # The host tier must fill before every resident request is suspended.
        num_suspendable = host_pages // pages_per_request
        assert 0 < num_suspendable < num_resident

        # Room on the host tier: each suspend makes room for one new request.
        newcomer = num_resident + 1
        for victim in range(num_resident, num_resident - num_suspendable, -1):
            assert not allocate(newcomer)
            manager.suspend_request(requests[victim])
            assert allocate(newcomer)
            newcomer += 1
        assert free_gpu_pages() < pages_per_request
        assert free_host_pages() < pages_per_request

        # Host tier full: the victim's pages have nowhere to go.
        assert not allocate(newcomer)
        manager.suspend_request(requests[num_resident - num_suspendable])
        assert not allocate(newcomer)
    finally:
        for request in requests.values():
            manager.free_resources(request)
        manager.shutdown()


# Enough host pages that every suspend in decode_pressure finds room.
ROOMY_HOST_TIER_BYTES = 8 << 20


def test_agg_helix_suspends_and_resumes_identically() -> None:
    """Under decode pressure with room on the host tier, every rank suspends
    the same requests and resumes them later, and a resumed request decodes
    on from where it stopped with consistent helix fields. The generation
    ladder can still recompute-pause a suspended request before the requester
    suspends itself; that request re-prefills instead of resuming."""
    scenario = SCENARIOS["decode_pressure"]
    ranks = [
        _make_rank(cp_rank, scenario, host_cache_size=ROOMY_HOST_TIER_BYTES)
        for cp_rank in range(CP_SIZE)
    ]
    for manager, _ in ranks:
        assert manager.has_cache_tier_below_gpu
    active = [[_make_request(i, scenario) for i in scenario.prompt_lens] for _ in ranks]

    # Token count of each request when it was last suspended, until it resumes.
    suspended_at_tokens: dict[int, int] = {}
    resumed: set[int] = set()
    for _ in range(MAX_ITERATIONS):
        if not any(active):
            break
        decisions = [
            _step(cp_rank, manager, scheduler, rank_active, scenario)
            for cp_rank, ((manager, scheduler), rank_active) in enumerate(zip(ranks, active))
        ]
        assert all(d == decisions[0] for d in decisions), decisions
        _, gen, paused, recompute_paused = decisions[0]
        for request_id in recompute_paused:
            suspended_at_tokens.pop(request_id, None)

        for request_id in gen:
            if request_id in suspended_at_tokens:
                tokens = suspended_at_tokens.pop(request_id)
                for rank_active in active:
                    request = next(r for r in rank_active if r.py_request_id == request_id)
                    # _step already sampled this step's token.
                    assert request.get_num_tokens(0) == tokens + 1
                resumed.add(request_id)
        for request in active[0]:
            if request.py_request_id in paused:
                suspended_at_tokens[request.py_request_id] = request.get_num_tokens(0)

    assert not any(active), "requests did not finish"
    assert resumed, "the pool never ran out, so nothing was suspended"
    assert not suspended_at_tokens, f"never resumed: {sorted(suspended_at_tokens)}"


# Request 1 decodes and needs its third page while request 2 is mid-prefill and
# the pool is full, so it suspends request 2 to the host tier. Once request 1
# finishes, request 2 resumes chunking.
MID_PREFILL_SUSPEND = Scenario(
    prompt_lens={1: 120, 2: 448},
    max_new_tokens={1: 20, 2: 8},
    max_num_tokens=128,
    chunk_unit=TOKENS_PER_BLOCK,
)

HOST_TIER_CHUNKED_SCENARIOS = {
    "chunked_context_pressure": SCENARIOS["chunked_context_pressure"],
    "chunked_staggered_arrivals": SCENARIOS["chunked_staggered_arrivals"],
    "mid_prefill_suspend": MID_PREFILL_SUSPEND,
}


@pytest.mark.parametrize("scenario_name", HOST_TIER_CHUNKED_SCENARIOS)
def test_agg_helix_chunked_prefill_with_host_tier(scenario_name: str) -> None:
    """Chunked prefill under pressure with a host tier: every rank suspends the
    same requests and lets the same prompts wait, and a request suspended
    mid-prefill resumes chunking from where it stopped."""
    scenario = HOST_TIER_CHUNKED_SCENARIOS[scenario_name]
    ranks = [
        _make_rank(cp_rank, scenario, host_cache_size=ROOMY_HOST_TIER_BYTES)
        for cp_rank in range(CP_SIZE)
    ]
    pending = [{i: _make_request(i, scenario) for i in scenario.prompt_lens} for _ in ranks]
    active = [[] for _ in ranks]

    # Prefill position of each request suspended mid-prefill, until it resumes
    # chunking or is recompute-paused.
    suspended_at_position: dict[int, int] = {}
    resumed_mid_prefill: set[int] = set()
    num_suspends = 0
    suspended_while_prompt_pending = False
    had_mixed_step = False
    for iteration in range(MAX_ITERATIONS):
        for rank_pending, rank_active in zip(pending, active):
            for request_id in list(rank_pending):
                if scenario.arrivals.get(request_id, 0) <= iteration:
                    rank_active.append(rank_pending.pop(request_id))
        if not any(active) and not any(pending):
            break
        # Prefill position of every request still in context, per rank.
        context_positions = [
            {
                r.py_request_id: r.context_current_position
                for r in rank_active
                if r.state == LlmRequestState.CONTEXT_INIT
            }
            for rank_active in active
        ]
        decisions = [
            _step(cp_rank, manager, scheduler, rank_active, scenario)
            for cp_rank, ((manager, scheduler), rank_active) in enumerate(zip(ranks, active))
        ]
        assert all(d == decisions[0] for d in decisions), decisions
        ctx, gen, paused, recompute_paused = decisions[0]
        had_mixed_step = had_mixed_step or bool(ctx and gen)

        for request_id in recompute_paused:
            suspended_at_position.pop(request_id, None)
        for request_id, _ in ctx:
            if request_id in suspended_at_position:
                position = suspended_at_position.pop(request_id)
                for positions in context_positions:
                    assert positions[request_id] == position, (request_id, positions)
                resumed_mid_prefill.add(request_id)
        for request_id in paused:
            if request_id in context_positions[0]:
                suspended_at_position[request_id] = context_positions[0][request_id]
        num_suspends += len(paused)
        # A prompt other than the suspended ones was admitted or left waiting.
        if paused and set(context_positions[0]) - set(paused):
            suspended_while_prompt_pending = True

    assert not any(active) and not any(pending), "requests did not finish"
    assert num_suspends > 0, "the pool never ran out, so nothing was suspended"
    assert had_mixed_step, "no step scheduled prefill and decode together"
    if scenario_name == "chunked_staggered_arrivals":
        assert suspended_while_prompt_pending
    if scenario_name == "mid_prefill_suspend":
        assert resumed_mid_prefill == {2}


# Two prompts of seven ledger blocks each; four blocks of each fill the pool.
PREFILL_ONLY_PRESSURE = Scenario(
    prompt_lens={1: 448, 2: 448},
    max_new_tokens={1: 8, 2: 8},
    max_num_tokens=128,
    chunk_unit=TOKENS_PER_BLOCK,
)


def _fill_pool_with_prefills(ranks: list) -> list:
    """Leave both requests half-prefilled in a full pool on every rank, with no
    decode request holding pages.

    FCFS chunking gives the whole token budget to the oldest prompt, which then
    finishes its prefill before a newer one gets far. Leaving request 1 out of
    the active list while request 2 prefills is what lets both stop halfway.
    """
    scenario = PREFILL_ONLY_PRESSURE
    requests = [{i: _make_request(i, scenario) for i in scenario.prompt_lens} for _ in ranks]
    for request_id in (1, 2):
        for _ in range(2):
            decisions = [
                _step(cp_rank, manager, scheduler, [rank_requests[request_id]], scenario)
                for cp_rank, ((manager, scheduler), rank_requests) in enumerate(
                    zip(ranks, requests)
                )
            ]
            assert all(d == decisions[0] for d in decisions), decisions
    for (manager, _), rank_requests in zip(ranks, requests):
        assert manager.impl.get_storage_statistics(GPU_LEVEL)[0].free == 0
        for request in rank_requests.values():
            assert request.state == LlmRequestState.CONTEXT_INIT
            assert request.context_current_position == 256
    return [[rank_requests[1], rank_requests[2]] for rank_requests in requests]


def test_prefill_only_pressure_recompute_pauses_with_host_tier() -> None:
    """Two half-prefilled prompts fill the pool and no decode request holds
    pages, so the generation path has nothing to suspend. The context path
    recompute-pauses the newer prompt, every rank picks the same one, and both
    finish.

    The starting state is built directly; how often serving reaches it (for
    example after the decode requests are suspended) needs a multi-GPU run.
    """
    scenario = PREFILL_ONLY_PRESSURE
    ranks = [
        _make_rank(cp_rank, scenario, host_cache_size=ROOMY_HOST_TIER_BYTES)
        for cp_rank in range(CP_SIZE)
    ]
    active = _fill_pool_with_prefills(ranks)

    first_recompute_paused = None
    for _ in range(MAX_ITERATIONS):
        if not any(active):
            break
        decisions = [
            _step(cp_rank, manager, scheduler, rank_active, scenario)
            for cp_rank, ((manager, scheduler), rank_active) in enumerate(zip(ranks, active))
        ]
        assert all(d == decisions[0] for d in decisions), decisions
        recompute_paused = decisions[0][3]
        if first_recompute_paused is None and recompute_paused:
            first_recompute_paused = recompute_paused

    assert not any(active), "requests did not finish"
    assert first_recompute_paused == (2,)


# Kimi K3 keeps a fixed-size KDA recurrent state per request next to its
# attention KV. These shapes match K3's per-rank KDA state (6 heads, K = V = 128,
# conv width 4), so a state slot is ~2.25 MiB and the state pool is small.
NUM_KDA_LAYERS = 6
NUM_HYBRID_REQUESTS = 12
HYBRID_PROMPT_LEN = 60
HYBRID_NEW_TOKENS = 40

# Paged attention next to the KDA state: plain multi-head KV, and K3's MLA
# latent cache (one K-only head of kv_lora_rank + qk_rope_head_dim = 512 + 64).
HYBRID_KV_LAYOUTS = {
    "mha": dict(kv_cache_type=CacheType.SELF, num_layers=1, num_kv_heads=8, head_dim=128),
    "mla": dict(kv_cache_type=CacheType.SELFKONLY, num_layers=4, num_kv_heads=1, head_dim=576),
}


def _make_hybrid_rank(
    cp_rank: int, kv_layout: str, host_cache_size: int
) -> tuple[MambaHybridCacheManagerV2, KVCacheV2Scheduler]:
    """A real KDA + attention hybrid manager and scheduler for one CP rank,
    with a KV pool small enough that the requests run out of pages."""
    layout = HYBRID_KV_LAYOUTS[kv_layout]
    num_attention_layers = layout["num_layers"]
    distributed = Mock()
    distributed.allreduce.side_effect = lambda value, op: value
    distributed.local_world_size = 1
    with patch.object(Distributed, "get", return_value=distributed):
        manager = MambaHybridCacheManagerV2(
            mamba_d_state=128,
            mamba_d_conv=4,
            mamba_num_heads=6,
            mamba_n_groups=6,
            mamba_head_dim=128,
            mamba_num_layers=NUM_KDA_LAYERS,
            mamba_layer_mask=[True] * NUM_KDA_LAYERS + [False] * num_attention_layers,
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=torch.float32,
            kv_cache_config=KvCacheConfig(
                max_tokens=768, host_cache_size=host_cache_size, enable_block_reuse=False
            ),
            kv_cache_type=layout["kv_cache_type"],
            num_layers=num_attention_layers,
            num_kv_heads=layout["num_kv_heads"],
            head_dim=layout["head_dim"],
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=128,
            max_batch_size=NUM_HYBRID_REQUESTS,
            mapping=Mapping(
                world_size=CP_SIZE,
                rank=cp_rank,
                cp_size=CP_SIZE,
                cp_config={"cp_type": CpType.HELIX},
            ),
            dtype=DataType.HALF,
            layer_mask=[False] * NUM_KDA_LAYERS + [True] * num_attention_layers,
            vocab_size=1024,
            conv_state_layout="q_k_v",
            is_disagg=False,
        )
    scheduler = KVCacheV2Scheduler(
        max_batch_size=NUM_HYBRID_REQUESTS,
        max_num_tokens=128,
        kv_cache_manager=manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TOKENS_PER_BLOCK),
        enable_recompute_pause=True,
    )
    return manager, scheduler


def _kda_states(manager: MambaHybridCacheManagerV2, request_id: int) -> list:
    """The request's recurrent state in every KDA layer, as the kernels see it."""
    slot = manager.get_state_indices([request_id])[0]
    return [manager.get_ssm_states(layer)[slot] for layer in range(NUM_KDA_LAYERS)]


def _hybrid_step(
    manager: MambaHybridCacheManagerV2,
    scheduler: KVCacheV2Scheduler,
    active: list,
    markers: dict,
) -> tuple:
    """One executor iteration on one rank; returns what the rank decided.

    Once a request finishes its prefill, its KDA state is filled with a marker
    (`markers` maps request ID to it), and every later decode step checks the
    marker is still there, whether or not the request was suspended between.
    The forward pass does not run, so nothing else writes the state.
    """
    out = scheduler.schedule_request(active, set())
    decided = (
        tuple(
            (r.py_request_id, r.context_current_position, r.context_chunk_size)
            for r in out.context_requests
        ),
        tuple(r.py_request_id for r in out.generation_requests),
        tuple(r.py_request_id for r in out.paused_requests),
        tuple(r.py_request_id for r in out.recompute_paused_requests),
    )
    for request in out.recompute_paused_requests:
        request.reset_for_recompute(1 << 20)
        # The re-prefill rebuilds the state from the prompt.
        markers.pop(request.py_request_id, None)

    batch = ScheduledRequests()
    for request in out.context_requests:
        batch.append_context_request(request)
    batch.generation_requests = list(out.generation_requests)
    # Assigns every scheduled request a state slot; a resumed request's state
    # is back on GPU by now.
    manager.prepare_resources(batch)
    for request in out.generation_requests:
        request_id = request.py_request_id
        if request_id in markers:
            for state in _kda_states(manager, request_id):
                assert bool((state == markers[request_id]).all()), request_id

    # Forward pass: prefill chunks advance, every finished prefill and every
    # decode step samples one token.
    manager.update_context_resources(batch)
    for request in out.context_requests:
        request.move_to_next_context_chunk()
        if request.context_remaining_length == 0:
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.add_new_token(0, 0)
            markers[request.py_request_id] = float(request.py_request_id)
            for state in _kda_states(manager, request.py_request_id):
                state.fill_(markers[request.py_request_id])
    for request in out.generation_requests:
        request.add_new_token(0, 0)
        request.py_decoding_iter += 1
    manager.update_resources(batch)

    for request in list(active):
        if request.get_num_tokens(0) >= HYBRID_PROMPT_LEN + HYBRID_NEW_TOKENS:
            request.state = LlmRequestState.GENERATION_COMPLETE
            manager.free_resources(request)
            markers.pop(request.py_request_id, None)
            active.remove(request)
    return decided


@pytest.mark.parametrize(
    "host_cache_size", [0, ROOMY_HOST_TIER_BYTES], ids=["gpu_only", "host_tier"]
)
@pytest.mark.parametrize("kv_layout", HYBRID_KV_LAYOUTS)
def test_agg_helix_kda_state_survives_suspend_and_restarts_on_recompute(
    kv_layout: str, host_cache_size: int
) -> None:
    """KDA state under KV pressure on a K3-like hybrid manager: every rank
    makes the same choices, a suspended request resumes with its KDA state
    intact, and a recompute-paused request re-prefills from the first token,
    where the KDA prefill starts without initial state."""
    ranks = [_make_hybrid_rank(cp_rank, kv_layout, host_cache_size) for cp_rank in range(CP_SIZE)]
    active = [
        [
            LlmRequest(
                request_id=i,
                max_new_tokens=HYBRID_NEW_TOKENS,
                input_tokens=list(range(i, i + HYBRID_PROMPT_LEN)),
                sampling_config=SamplingConfig(1),
                is_streaming=False,
            )
            for i in range(1, NUM_HYBRID_REQUESTS + 1)
        ]
        for _ in ranks
    ]
    markers = [{} for _ in ranks]

    # Requests suspended after their prefill, until they decode again.
    suspended_with_state: set[int] = set()
    resumed_with_state: set[int] = set()
    # Recompute-paused requests, until their re-prefill starts.
    awaiting_re_prefill: set[int] = set()
    num_re_prefills = 0
    num_suspends = 0
    for _ in range(MAX_ITERATIONS):
        if not any(active):
            break
        decisions = [
            _hybrid_step(manager, scheduler, rank_active, rank_markers)
            for (manager, scheduler), rank_active, rank_markers in zip(ranks, active, markers)
        ]
        assert all(d == decisions[0] for d in decisions), decisions
        ctx, gen, paused, recompute_paused = decisions[0]

        for request_id, start_position, _ in ctx:
            if request_id in awaiting_re_prefill:
                assert start_position == 0, (request_id, start_position)
                awaiting_re_prefill.remove(request_id)
                num_re_prefills += 1
        resumed_with_state.update(suspended_with_state.intersection(gen))
        suspended_with_state.difference_update(gen)
        for request_id in recompute_paused:
            suspended_with_state.discard(request_id)
            awaiting_re_prefill.add(request_id)
        # A request suspended mid-prefill has no marker yet.
        suspended_with_state.update(r for r in paused if r in markers[0])
        num_suspends += len(paused)

    assert not any(active), "requests did not finish"
    assert not awaiting_re_prefill
    if host_cache_size:
        assert resumed_with_state, "no request resumed with its KDA state"
    else:
        # Without a tier below GPU, suspending frees nothing, so helix never
        # suspends and recompute-pause makes all the room.
        assert num_suspends == 0
        assert num_re_prefills > 0
