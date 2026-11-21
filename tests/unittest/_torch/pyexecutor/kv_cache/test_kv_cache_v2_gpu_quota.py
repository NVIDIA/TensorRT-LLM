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
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
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
