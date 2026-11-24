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
"""The KV cache manager V2 scheduler of a DKV group across real ranks: every rank takes the same
decisions although the ranks' radix trees differ.

Each rank runs a real ``KVCacheManagerV2(dkv=True)`` and ``KVCacheV2Scheduler`` over the same
context-only requests, with the per-request chunk cap and parked conversation caches on. Before
that, one rank commits the shared prompt deeper than the others and many siblings under its first
DKV block, so a reuse lookup cut at the group's agreed depth claims less there than on the others
(the radix tree skips its partial-match pass at a node with 32 or more children). The test checks
that the ranks would claim different depths without the pins' check, then that every scheduling
pass schedules, sizes, claims, parks and closes alike on every rank. Needs one node with
``WORLD_SIZE`` GPUs; ``python3 test_dkv_scheduler_tp.py`` under ``mpirun -n 4`` runs the body.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _swa_branch_snapshot_point,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    LlmRequestType,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import (
    BlockReuseConfig,
    CapacitySchedulerPolicy,
    ContextChunkCapConfig,
    ContextChunkingPolicy,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope, TokenId

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

TPB = 16  # physical page; a DKV block of 4 ranks is 64 tokens
CAP = 128


def _shared(n: int) -> list:
    return [1000 + i for i in range(n)]


def _commit(manager: KVCacheManagerV2, tokens: list) -> None:
    """Commit ``tokens`` to this rank's radix tree only, outside any request."""
    stream = torch.cuda.current_stream().cuda_stream
    kv = manager.impl.create_kv_cache(ReuseScope(lora_id=None, salt=None))
    assert kv.resume(stream)
    kv.capacity = len(tokens)
    kv.commit([TokenId(t) for t in tokens])
    kv.close()


def _request(rid: int, tokens: list, conversation: str) -> LlmRequest:
    req = LlmRequest(
        request_id=rid,
        max_new_tokens=1,
        input_tokens=list(tokens),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    req.py_conversation_params = ConversationParams(conversation_id=conversation)
    return req


def _same(comm, value, what: str) -> None:
    everyone = comm.allgather(value)
    assert all(v == everyone[0] for v in everyone), (what, everyone)


def _body(world: int, rank: int, comm) -> None:
    torch.cuda.set_device(rank % torch.cuda.device_count())
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=16384,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            block_reuse_config=BlockReuseConfig(max_parked_conversation_caches=2),
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=TPB,
        max_seq_len=2048,
        max_batch_size=8,
        mapping=Mapping(world_size=world, tp_size=world, rank=rank),
        dtype=DataType.HALF,
        vocab_size=32000,
        dkv=True,
    )
    try:
        block = manager.dkv.dkv_tokens_per_block
        assert block == 64
        # Every rank but the last holds the shared prompt to 100 tokens (a partial second block).
        # The last holds it to 200 and 40 other prompts that leave it after its first block.
        if rank == world - 1:
            _commit(manager, _shared(200))
            for i in range(40):
                _commit(manager, _shared(block) + [10**5 * (i + 1) + j for j in range(block)])
        else:
            _commit(manager, _shared(100))

        conversation = _request(1, _shared(150) + [7] * 50, "c1")
        others = [_request(rid, [rid * 10**4 + i for i in range(300)], f"c{rid}") for rid in (2, 3)]

        # The group agrees on 100 tokens (the shallowest rank); a lookup cut there claims only the
        # first block on the last rank. Without the pins' check the ranks would diverge.
        manager.agree_dkv_context_reuse([conversation])
        agreed = manager._dkv_agreed[1]
        impl, pin, pin_id = manager._pin_reuse_prefix(conversation, agreed, 0)
        cut_claim = int(pin.num_committed_tokens)
        pin.close()
        impl.clear_stats_excluded(pin_id)
        manager.release_dkv_reuse_pins()
        _same(comm, agreed, "agreed depth")
        claims = comm.allgather(cut_claim)
        assert agreed == 100 and claims == [100] * (world - 1) + [64], claims

        scheduler = KVCacheV2Scheduler(
            max_batch_size=8,
            max_num_tokens=512,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TPB),
            context_chunk_cap=ContextChunkCapConfig(
                max_tokens_per_request=CAP, min_waiting_requests=2
            ),
            enable_recompute_pause=False,
        )
        active = [conversation, *others]
        reused, capped = {}, False
        turn2 = None
        for step in range(12):
            if not active:
                break
            out = scheduler.schedule_request(active, set())
            batch = ScheduledRequests()
            for req in out.context_requests:
                batch.append_context_request(req)
                if req.py_request_id not in reused:
                    reused[req.py_request_id] = int(
                        manager.kv_cache_map[req.py_request_id].num_committed_tokens
                    )
            capped = capped or any(r.context_chunk_size == CAP for r in out.context_requests)
            decisions = [
                (r.py_request_id, r.context_current_position, r.context_chunk_size)
                for r in out.context_requests
            ]
            manager.prepare_resources(batch)
            for req in out.context_requests:
                req.move_to_next_context_chunk()
            manager.update_context_resources(batch)
            for req in list(out.context_requests):
                if req.context_remaining_length == 0:
                    req.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
                    manager.free_resources(req)
                    active.remove(req)
            if turn2 is None and not active:
                # The next turn of the third conversation takes over its parked cache.
                turn2 = _request(4, others[1].get_tokens(0) + [5] * 30, "c3")
                active.append(turn2)
            _same(
                comm,
                (decisions, list(manager._parked_kv_caches), dict(manager._parked_counts)),
                f"step {step}",
            )
        assert not active, "requests left after 12 steps"
        _same(comm, reused, "reused tokens")
        # Every rank claims what the group verified: the first DKV block, not the agreed 100.
        assert reused[1] == 64 and reused[4] == 300, reused
        assert capped, "the chunk cap never applied"
        assert manager._parked_counts["taken_over"] == 1, manager._parked_counts
        if rank == 0:
            print(f"DKV scheduler: claims {reused}, parked {manager._parked_counts}", flush=True)
    finally:
        manager._close_parked_kv_caches()
        manager.shutdown()


def _fork_body(world: int, rank: int, comm) -> None:
    """Branch snapshots under DKV: each rank's tree holds a request's prompt to a different depth,
    so each rank's own fork differs; the group's fork, its lookup length and the branch point every
    rank derives from them are the same on all ranks."""
    torch.cuda.set_device(rank % torch.cuda.device_count())
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=16384,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            block_reuse_config=BlockReuseConfig(enable_branch_snapshot=True),
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=TPB,
        max_seq_len=2048,
        max_batch_size=8,
        mapping=Mapping(world_size=world, tp_size=world, rank=rank),
        dtype=DataType.HALF,
        vocab_size=32000,
        dkv=True,
    )
    try:
        _commit(manager, _shared(100 + 30 * rank))
        req = _request(11, _shared(250) + [9] * 50, "f1")
        own = comm.allgather(manager._probe_dkv_fork(req))
        assert [fork[0] for fork in own] == [100 + 30 * r for r in range(world)], own
        manager.agree_dkv_context_reuse([req])
        manager.pin_dkv_context_reuse([req])
        fork, reused = manager._dkv_forks[11], manager._dkv_agreed[11]
        assert fork == (100, 100, req.prompt_len - 1) and reused == 100, (fork, reused)
        point = _swa_branch_snapshot_point(
            fork[0], fork[2], reused, req.prompt_len, manager._fork_tokens_per_block
        )
        _same(comm, (fork, reused, point, manager._fork_tokens_per_block), "fork")
        manager.release_dkv_reuse_pins()
        if rank == 0:
            print(f"DKV fork: own {own}, group {fork}, point {point}", flush=True)
    finally:
        manager.shutdown()


def _run(world: int) -> bool:
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
    try:
        _body(world, rank, comm)
        _fork_body(world, rank, comm)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("world_size", [4], ids=lambda n: f"tp{n}")
def test_every_rank_schedules_alike(world_size: int) -> None:
    from mpi4py.futures import MPIPoolExecutor

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs, has {torch.cuda.device_count()}")
    with MPIPoolExecutor(max_workers=world_size) as executor:
        assert all(executor.map(_run, [world_size] * world_size))


if __name__ == "__main__":
    _run(MPI.COMM_WORLD.Get_size())
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("DKV scheduler TP test passed", flush=True)
