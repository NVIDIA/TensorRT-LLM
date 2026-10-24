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
"""SWA branch snapshots through a real cache manager and scheduler.

Three requests share a 192-token prefix and then diverge. Two settings drop the
SWA window at the fork: under per_request only the window at a prompt end
reaches the radix tree, and under SWA scratch reuse only the window at a chunk
end gets real pages. A sibling then reuses less than the shared prefix. With
block_reuse_config.enable_branch_snapshot, the second request ends a context
chunk at the fork and commits the window there, so the third reuses the shared
prefix.

Each prefill is driven in PyExecutor order: schedule, prepare_resources, advance
the context chunk, then update_context_resources. These tests allocate device
memory pools.
"""

import gc
from contextlib import contextmanager
from typing import Iterator

import pytest
import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

TOKENS_PER_BLOCK = 32
WINDOW = 64
MAX_SEQ_LEN = 512
FORK = 192
PROMPT_LEN = 256
# Below the prompt length, so a prefill takes budget-sized chunks of 128.
MAX_NUM_TOKENS = 128
# Layer 0 slides; layer 1 is full attention (a window of max_seq_len normalizes to None).
SWA_AND_FULL = [WINDOW, MAX_SEQ_LEN]


@contextmanager
def real_manager(
    *,
    enable_branch_snapshot: bool,
    policy: str = "per_request",
    swa_scratch_reuse: bool = False,
    max_attention_window: list[int] = SWA_AND_FULL,
    joint_kv_cache_reuse: bool = False,
) -> Iterator[KVCacheManagerV2]:
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            enable_block_reuse=True,
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
        joint_kv_cache_reuse=joint_kv_cache_reuse,
    )
    try:
        yield manager
    finally:
        manager.shutdown()
        del manager
        gc.collect()
        torch.cuda.empty_cache()


def make_scheduler(manager: KVCacheManagerV2) -> KVCacheV2Scheduler:
    # The executor selects FORCE_CHUNK when enable_branch_snapshot is set. The
    # flag-off case uses it too, so the two cases differ only in the manager.
    return KVCacheV2Scheduler(
        max_batch_size=4,
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        ctx_chunk_config=(ContextChunkingPolicy.FORCE_CHUNK, TOKENS_PER_BLOCK),
    )


def make_request(request_id: int) -> LlmRequest:
    shared = list(range(FORK))
    tail = [10000 * request_id + i for i in range(PROMPT_LEN - FORK)]
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=shared + tail,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def prefill(
    manager: KVCacheManagerV2, scheduler: KVCacheV2Scheduler, request: LlmRequest
) -> list[int]:
    """Run one request's context phase and return the start of each chunk."""
    chunk_starts = []
    while request.context_remaining_length > 0:
        output = scheduler.schedule_request([request], set())
        assert [r.py_request_id for r in output.context_requests] == [request.py_request_id]
        batch = ScheduledRequests()
        batch.append_context_request(request)
        manager.prepare_resources(batch)
        chunk_starts.append(request.context_current_position)
        request.move_to_next_context_chunk()
        manager.update_context_resources(batch)
    manager.free_resources(request)
    return chunk_starts


@pytest.mark.parametrize("enable_branch_snapshot", [False, True])
@pytest.mark.parametrize(
    ("policy", "swa_scratch_reuse", "r2_starts_off", "r2_starts_on"),
    [
        # Full attention matches the shared prefix, but no SWA window is left on
        # it, so R2 reuses nothing.
        ("per_request", False, [0, 128], [0, 128, FORK]),
        # R2 reuses up to R1's first chunk end, the last SWA window on the prefix.
        ("all_reusable", True, [128], [128, FORK]),
    ],
)
def test_branch_snapshot_lets_a_sibling_reuse_the_shared_prefix(
    policy: str,
    swa_scratch_reuse: bool,
    r2_starts_off: list[int],
    r2_starts_on: list[int],
    enable_branch_snapshot: bool,
) -> None:
    with real_manager(
        enable_branch_snapshot=enable_branch_snapshot,
        policy=policy,
        swa_scratch_reuse=swa_scratch_reuse,
    ) as manager:
        scheduler = make_scheduler(manager)
        r1, r2, r3 = (make_request(request_id) for request_id in (1, 2, 3))

        # R1 finds an empty tree and prefills in budget-sized chunks.
        assert prefill(manager, scheduler, r1) == [0, 128]
        assert r1.expect_snapshot_points == []

        r2_chunk_starts = prefill(manager, scheduler, r2)
        r3_chunk_starts = prefill(manager, scheduler, r3)
        if enable_branch_snapshot:
            # R2 ends a chunk at the fork; R3 then starts from it and records no
            # point of its own, because reuse already reached the fork.
            assert r2.expect_snapshot_points == [FORK]
            assert r2_chunk_starts == r2_starts_on
            assert r3_chunk_starts == [FORK]
            assert r3.expect_snapshot_points == []
        else:
            assert r2.expect_snapshot_points == []
            assert r2_chunk_starts == r2_starts_off
            assert r3_chunk_starts == r2_starts_off


@pytest.mark.parametrize(
    ("max_attention_window", "joint_kv_cache_reuse", "records"),
    [
        (SWA_AND_FULL, False, True),
        # Full attention alone keeps every block's pages, so there is no window to keep.
        ([MAX_SEQ_LEN, MAX_SEQ_LEN], False, False),
        # A paired draft pool caps the target lookup and hides the fork.
        (SWA_AND_FULL, True, False),
    ],
)
def test_branch_points_need_swa_layers_and_unpaired_reuse(
    max_attention_window: list[int], joint_kv_cache_reuse: bool, records: bool
) -> None:
    with real_manager(
        enable_branch_snapshot=True,
        max_attention_window=max_attention_window,
        joint_kv_cache_reuse=joint_kv_cache_reuse,
    ) as manager:
        assert manager._record_branch_snapshots is records
