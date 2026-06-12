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

from dataclasses import dataclass, field

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DEFAULT_BEAM_INDEX
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import init_cuda_once

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@dataclass
class _ContextRequest:
    request_id: int
    tokens: list[int]
    context_remaining_length: int
    py_request_id: int = field(init=False)
    context_current_position: int = 0
    state: LlmRequestState = LlmRequestState.GENERATION_IN_PROGRESS
    is_finished_due_to_cancellation: bool = False
    prepopulated_prompt: tuple[int, int] | None = None

    lora_task_id: int | None = None
    cache_salt_id: int | None = None
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    is_disagg_generation_init_state: bool = False
    is_dummy_request: bool = False
    multimodal_hashes: None = None
    multimodal_positions: None = None
    multimodal_lengths: None = None

    def __post_init__(self) -> None:
        self.py_request_id = self.request_id

    @property
    def prompt_len(self) -> int:
        return len(self.tokens)

    @property
    def is_dummy(self) -> bool:
        return self.is_dummy_request

    @property
    def prepopulated_prompt_len(self) -> int:
        if self.prepopulated_prompt is None:
            return 0
        return self.prepopulated_prompt[0]

    def get_tokens(self, beam_id: int = DEFAULT_BEAM_INDEX) -> list[int]:
        assert beam_id == DEFAULT_BEAM_INDEX
        return self.tokens

    def set_prepopulated_prompt_len(self, length: int, tokens_per_block: int) -> None:
        self.prepopulated_prompt = (length, tokens_per_block)


@pytest.fixture
def manager() -> KVCacheManagerV2:
    init_cuda_once()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=True,
            max_gpu_total_bytes=8 << 20,
            max_util_for_resume=1.0,
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=4,
        max_seq_len=16,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=4096,
        enable_stats=False,
    )
    try:
        yield manager
    finally:
        manager.shutdown()


def _context_batch(*requests: _ContextRequest) -> ScheduledRequests:
    batch = ScheduledRequests()
    for request in requests:
        batch.append_context_request(request)
    return batch


def _free_if_active(manager: KVCacheManagerV2, request: _ContextRequest) -> None:
    if request.py_request_id in manager.kv_cache_map:
        manager.free_resources(request)


def test_context_update_does_not_commit_canceled_request(manager: KVCacheManagerV2) -> None:
    request = _ContextRequest(1, list(range(8)), context_remaining_length=8)
    reuse_request = _ContextRequest(2, list(range(8)), context_remaining_length=8)

    try:
        assert manager.prepare_context(request)
        assert manager.resize_context(request, num_tokens=8)
        request.context_current_position = 8
        request.context_remaining_length = 0
        request.state = LlmRequestState.GENERATION_COMPLETE
        request.is_finished_due_to_cancellation = True

        manager.update_context_resources(_context_batch(request))
        manager.free_resources(request)

        assert manager.prepare_context(reuse_request)
        assert reuse_request.prepopulated_prompt_len == 0
    finally:
        _free_if_active(manager, reuse_request)
        _free_if_active(manager, request)
