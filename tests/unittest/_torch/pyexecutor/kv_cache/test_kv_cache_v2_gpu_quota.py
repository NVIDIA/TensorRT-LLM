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
"""Resizing the GPU pool of KVCacheManagerV2 while serving, and deferring its reuse commits."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig, KvPoolRebalanceConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

MIB = 1 << 20


@pytest.fixture
def manager():
    # 32768 tokens of 8 KiB each make a 256 MiB pool, which allocates in 2 MiB units.
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=32768, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=4,
        num_kv_heads=4,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=1024,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    yield manager
    manager.shutdown()


def test_resize_releases_memory_keeps_kv(manager):
    (request,) = manager.add_dummy_requests([0], [512])
    (pages,) = manager.get_batch_cache_indices([0], 0)
    manager.get_buffers(0)[pages[-1]].fill_(1.5)
    torch.cuda.synchronize()
    quota = manager.gpu_quota

    # The pool may move pages, so every KV cache must be suspended first.
    with pytest.raises(RuntimeError, match="SUSPENDED"):
        manager.resize_gpu_quota(quota - 64 * MIB)
    # A quota that rounds down to the pool's own changes nothing, so it needs no suspension.
    assert manager.resize_gpu_quota(quota + MIB) == quota

    manager.suspend_request(request)
    free = torch.cuda.mem_get_info()[0]
    # Rounded down to the pool's 2 MiB allocation unit.
    assert manager.resize_gpu_quota(quota - 64 * MIB - 1) == quota - 66 * MIB
    assert manager.gpu_quota == quota - 66 * MIB
    assert torch.cuda.mem_get_info()[0] >= free + 66 * MIB

    assert manager.resize_gpu_quota(quota) == quota
    assert manager.resume_request(request)
    (pages,) = manager.get_batch_cache_indices([0], 0)
    assert manager.get_buffers(0)[pages[-1]].eq(1.5).all()


def test_resize_refuses_growth_past_construction(manager):
    """Refuse to grow the pool past its size at construction, the size its views cover."""
    quota = manager.gpu_quota
    num_slots = [pool_group.num_slots for pool_group in manager.impl.pool_group_descs]
    with pytest.raises(ValueError, match="cannot grow to .* past its size at construction"):
        manager.resize_gpu_quota(quota + 64 * MIB)
    assert manager.gpu_quota == quota
    assert [pool_group.num_slots for pool_group in manager.impl.pool_group_descs] == num_slots

    # Growing back to that size after a shrink is allowed.
    assert manager.resize_gpu_quota(quota - 64 * MIB) == quota - 64 * MIB
    assert manager.resize_gpu_quota(quota) == quota


def test_kv_pools_suspended_resumes_after_error(manager):
    """Keep every request active with its KV when a restore shrinks the pool and then fails."""
    (request,) = manager.add_dummy_requests([0], [512])
    (pages,) = manager.get_batch_cache_indices([0], 0)
    manager.get_buffers(0)[pages[-1]].fill_(1.5)
    torch.cuda.synchronize()
    executor = SimpleNamespace(
        kv_cache_manager=manager,
        active_requests=[request],
        model_engine=SimpleNamespace(cuda_graph_runner=None),
        _read_kv_cache_capacity=lambda: None,
        _defer_generation_only_release=False,
    )
    executor._suspend_for_rebalance = partial(PyExecutor._suspend_for_rebalance, executor)
    executor._resume_after_rebalance = partial(PyExecutor._resume_after_rebalance, executor)
    executor._suspend_padding_dummies_for_rebalance = partial(
        PyExecutor._suspend_padding_dummies_for_rebalance, executor
    )
    executor._resume_padding_dummies_after_rebalance = partial(
        PyExecutor._resume_padding_dummies_after_rebalance, executor
    )

    with pytest.raises(RuntimeError, match="a peer exited"):
        with PyExecutor.kv_pools_suspended(executor) as kv_cache_manager:
            kv_cache_manager.resize_gpu_quota(kv_cache_manager.gpu_quota - 64 * MIB)
            raise RuntimeError("a peer exited during the remap")
    assert manager.is_request_active(request.py_request_id)
    (pages,) = manager.get_batch_cache_indices([0], 0)
    assert manager.get_buffers(0)[pages[-1]].eq(1.5).all()
    manager.free_resources(request)


def _pool_bytes(pool, slots: range) -> torch.Tensor:
    """Return the bytes of ``slots`` in ``pool``."""
    return convert_to_torch_tensor(
        TensorWrapper(
            pool.base_address + slots.start * pool.slot_bytes,
            torch.uint8,
            [len(slots) * pool.slot_bytes],
        )
    )


def test_resize_zeroes_added_slots():
    """Zero the slots that a grow adds, which can hold the bytes they held before a shrink."""
    # The pool of the layer with one KV head fits in one 2 MiB allocation unit. Such a pool keeps
    # its memory mapped through a shrink, so the slots that the shrink drops come back from the
    # grow still holding the bytes written below.
    manager = KVCacheManagerV2(
        KvCacheConfig(max_gpu_total_bytes=18 * MIB, host_cache_size=0, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=[8, 1],
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    try:
        num_slots = [pool_group.num_slots for pool_group in manager.impl.pool_group_descs]
        assert any(
            count * pool.slot_bytes <= 2 * MIB
            for pool_group, count in zip(manager.impl.pool_group_descs, num_slots)
            for pool in pool_group.pools
        )
        for pool_group, count in zip(manager.impl.pool_group_descs, num_slots):
            for pool in pool_group.pools:
                _pool_bytes(pool, range(count)).fill_(0xFF)
        quota = manager.gpu_quota
        manager.resize_gpu_quota(quota // 2)
        kept = [pool_group.num_slots for pool_group in manager.impl.pool_group_descs]
        manager.resize_gpu_quota(quota)
        torch.cuda.synchronize()

        pool_groups = manager.impl.pool_group_descs
        assert any(count < pool_group.num_slots for pool_group, count in zip(pool_groups, kept))
        for pool_group, count in zip(pool_groups, kept):
            for pool in pool_group.pools:
                assert _pool_bytes(pool, range(count)).eq(0xFF).all()
                assert not _pool_bytes(pool, range(count, pool_group.num_slots)).any()
    finally:
        manager.shutdown()


def _two_pool_group_manager(enable_kv_pool_rebalance: bool) -> KVCacheManagerV2:
    """Return a manager with one pool group per layer, sized for 1024-token sequences.

    Layers share a pool group only when their life cycle's slot sizes match, so the layers differ
    in KV head count as well as in attention window. The pools are sized for one context chunk of
    max_num_tokens beside decoding requests of 1024 tokens; with a small max_num_tokens the
    decoding requests dominate. Each needs a few blocks of the sliding-window layer of 8 KV heads
    and many of the full-attention layer of 1 KV head, which splits the pool's bytes evenly. A
    32-token request needs one block of each, so short requests move the auto-tuner's target
    split to 8 to 1. The auto-tuner may act at once, on any change of its target.
    """
    return KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 * MIB,
            host_cache_size=0,
            enable_block_reuse=False,
            max_attention_window=[64, 1024],
            avg_seq_len=1024,
            enable_kv_pool_rebalance=enable_kv_pool_rebalance,
            kv_pool_rebalance_config=KvPoolRebalanceConfig(
                min_sampled_kv_caches=0,
                cooldown_secs=0,
                target_ratio_update_interval=1,
                ratio_tolerance=0.01,
            ),
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=[8, 1],
        head_dim=128,
        tokens_per_block=CHUNK,
        max_seq_len=1024,
        max_batch_size=4,
        max_num_tokens=4 * CHUNK,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
        vocab_size=32000,
    )


def _request(request_id: int, num_tokens: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(1000, 1000 + num_tokens)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def _run_short_requests(manager: KVCacheManagerV2) -> None:
    """Run and free one-block requests, each a sample for the auto-tuner."""
    for request_id in range(64):
        request = _request(request_id, CHUNK)
        _run_context_chunk(manager, request)
        manager.free_resources(request)


def test_resize_keeps_pool_groups_within_construction():
    """Keep every pool group within its size at construction through a shrink and a staged grow.

    The short requests move the auto-tuner's target split, which nothing may act on while
    rebalancing is off.
    """
    # With rebalancing on, the same requests move the target far enough to ask for an adjustment.
    tuned = _two_pool_group_manager(enable_kv_pool_rebalance=True)
    try:
        assert len(tuned.impl.pool_group_descs) == 2
        assert not tuned.impl.need_adjustment
        _run_short_requests(tuned)
        assert tuned.impl.need_adjustment
    finally:
        tuned.shutdown()

    manager = _two_pool_group_manager(enable_kv_pool_rebalance=False)

    def num_slots():
        return [pool_group.num_slots for pool_group in manager.impl.pool_group_descs]

    try:
        construction = num_slots()
        assert len(construction) == 2
        _run_short_requests(manager)
        held = _request(64, 4 * CHUNK)
        while held.context_remaining_length > 0:
            _run_context_chunk(manager, held)
        manager.suspend_request(held)

        quota = manager.gpu_quota
        # Shrink with the held request's pages in use, then grow back in steps.
        for target in [quota // 2, *range(quota // 2 + quota // 8, quota, quota // 8), quota]:
            manager.resize_gpu_quota(target)
            assert all(count <= max_count for count, max_count in zip(num_slots(), construction))
        assert num_slots() == construction
        assert not manager.impl.need_adjustment

        assert manager.resume_request(held)
        manager.free_resources(held)
    finally:
        manager.shutdown()


CHUNK = 32  # one KV block


@pytest.fixture
def reuse_manager():
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=1024,
            enable_block_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=CHUNK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
        vocab_size=32000,
    )
    yield manager
    manager.shutdown()


def _run_context_chunk(manager, request):
    """Allocate and run the request's next context chunk; return its batch."""
    batch = ScheduledRequests()
    batch.append_context_request(request)
    if request.context_current_position == 0:
        assert manager.prepare_context(request)
    request.context_chunk_size = min(CHUNK, request.context_remaining_length)
    assert manager.resize_context(request, request.context_chunk_size)
    manager.prepare_resources(batch)
    request.move_to_next_context_chunk()
    return batch


def test_deferred_context_commits_wait_for_commit(reuse_manager):
    """Commit a deferred context chunk for reuse only in commit_context_blocks."""
    request = LlmRequest(
        request_id=1,
        max_new_tokens=8,
        input_tokens=list(range(1000, 1000 + 3 * CHUNK)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    batch = _run_context_chunk(reuse_manager, request)
    with reuse_manager.deferred_context_commits() as commit_ends:
        reuse_manager.update_context_resources(batch)
    kv_cache = reuse_manager.kv_cache_map[request.py_request_id]
    assert commit_ends == {request.py_request_id: CHUNK}
    assert kv_cache.num_committed_tokens == 0
    reuse_manager.commit_context_blocks(batch, commit_ends)
    assert kv_cache.num_committed_tokens == CHUNK

    # Outside the scope, the next chunk is committed when its resources are updated.
    reuse_manager.update_context_resources(_run_context_chunk(reuse_manager, request))
    assert kv_cache.num_committed_tokens == 2 * CHUNK
    reuse_manager.free_resources(request)
