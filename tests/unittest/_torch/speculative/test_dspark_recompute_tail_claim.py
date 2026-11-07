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
"""A DSpark context_recompute_tail caps a real V2 manager's reuse claim, paired or not."""

import gc
from contextlib import ExitStack, contextmanager
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
    DSparkDecodingConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

TOKENS_PER_BLOCK = 16
PROMPT_LEN = 128
SHARED_PREFIX = 96
MAX_SEQ_LEN = 256


@contextmanager
def real_manager(
    tail: int, *, is_draft: bool = False, joint: bool = False
) -> Iterator[KVCacheManagerV2]:
    """A V2 manager with block reuse under a standalone DSpark spec config."""
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            enable_block_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
        spec_config=DSparkDecodingConfig(
            max_draft_len=4, speculative_model="draft", context_recompute_tail=tail
        ),
        is_draft=is_draft,
        joint_kv_cache_reuse=joint,
    )
    try:
        yield manager
    finally:
        manager.shutdown()
        del manager
        gc.collect()
        torch.cuda.empty_cache()


def make_request(request_id: int) -> LlmRequest:
    """A prompt sharing its first SHARED_PREFIX tokens with every other request."""
    shared = list(range(SHARED_PREFIX))
    tail = [10000 * request_id + i for i in range(PROMPT_LEN - SHARED_PREFIX)]
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=shared + tail,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def prefill(
    managers: tuple[KVCacheManagerV2, ...], scheduler: KVCacheV2Scheduler, request: LlmRequest
) -> list[int]:
    """Run one request's context phase through every pool and return each chunk's start."""
    chunk_starts = []
    while request.context_remaining_length > 0:
        output = scheduler.schedule_request([request], set())
        assert [r.py_request_id for r in output.context_requests] == [request.py_request_id]
        batch = ScheduledRequests()
        batch.append_context_request(request)
        for manager in managers:
            manager.prepare_resources(batch)
        chunk_starts.append(request.context_current_position)
        request.move_to_next_context_chunk()
        for manager in managers:
            manager.update_context_resources(batch)
    for manager in managers:
        manager.free_resources(request)
    return chunk_starts


@pytest.mark.parametrize("paired", [False, True], ids=["unpaired", "paired"])
@pytest.mark.parametrize(
    "tail, second_start",
    [
        (0, SHARED_PREFIX),  # no recompute: the whole shared prefix is reused
        (16, SHARED_PREFIX),  # limit 128 - 16 = 112 is past the match
        (48, 80),  # limit 128 - 48 = 80 leaves the drafter its tail
        (-1, 0),  # a full re-prefill
    ],
)
def test_a_dspark_tail_caps_the_reuse_claim(tail: int, second_start: int, paired: bool) -> None:
    with ExitStack() as stack:
        target = stack.enter_context(real_manager(tail, joint=paired))
        draft = (
            stack.enter_context(real_manager(tail, is_draft=True, joint=True)) if paired else None
        )
        managers = (target, draft) if paired else (target,)
        scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=MAX_SEQ_LEN,
            kv_cache_manager=target,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            draft_kv_cache_manager=draft,
        )
        assert prefill(managers, scheduler, make_request(1)) == [0]
        assert prefill(managers, scheduler, make_request(2)) == [second_start]
