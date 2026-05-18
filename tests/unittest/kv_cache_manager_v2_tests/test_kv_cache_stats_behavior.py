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

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager as KVCacheManagerV1
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.bindings.internal.testing import simulate_prefill_completion_only_use_for_testing
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DEFAULT_BEAM_INDEX
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS_PER_BLOCK = 4
BYTES_PER_BLOCK = 2 << 20


@dataclass
class _StatsRequest:
    request_id: int
    tokens: list[int]
    context_remaining_length: int
    py_request_id: int = field(init=False)
    lora_task_id: int | None = None
    cache_salt_id: int | None = None
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    is_encoder_init_state: bool = False
    is_dummy_request: bool = False
    is_attention_dp_dummy: bool = False
    is_cuda_graph_dummy: bool = False
    is_disagg_generation_transmission_complete: bool = False
    context_phase_params: None = None
    py_draft_tokens: list[int] = field(default_factory=list)
    draft_tokens: list[int] = field(default_factory=list)
    context_current_position: int = 0
    context_chunk_size: int = 0
    prepopulated_prompt: tuple[int, int] | None = None
    kv_cache_perf_metric_calls: list[dict[str, int]] = field(default_factory=list)
    multimodal_hashes: None = None
    multimodal_positions: None = None
    multimodal_lengths: None = None

    def __post_init__(self) -> None:
        self.py_request_id = self.request_id
        self.context_chunk_size = self.context_remaining_length

    @property
    def prompt_len(self) -> int:
        return len(self.tokens)

    @property
    def is_dummy(self) -> bool:
        return self.is_attention_dp_dummy or self.is_cuda_graph_dummy or self.is_dummy_request

    def get_tokens(self, beam_id: int = DEFAULT_BEAM_INDEX) -> list[int]:
        assert beam_id == DEFAULT_BEAM_INDEX
        return self.tokens

    def set_prepopulated_prompt_len(self, length: int, tokens_per_block: int) -> None:
        self.prepopulated_prompt = (length, tokens_per_block)

    @property
    def prepopulated_prompt_len(self) -> int:
        if self.prepopulated_prompt is None:
            return 0
        return self.prepopulated_prompt[0]

    def update_kv_cache_perf_metrics(
        self,
        alloc_total_blocks: int,
        alloc_new_blocks: int,
        reused_blocks: int,
        missed_blocks: int,
    ) -> None:
        self.kv_cache_perf_metric_calls.append(
            {
                "alloc_total_blocks": alloc_total_blocks,
                "alloc_new_blocks": alloc_new_blocks,
                "reused_blocks": reused_blocks,
                "missed_blocks": missed_blocks,
            }
        )


def _create_manager(
    *,
    gpu_bytes: int,
    num_layers: int = 1,
    max_attention_window: list[int] | None = None,
    enable_block_reuse: bool = True,
    enable_stats: bool = True,
) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=True,
            max_gpu_total_bytes=gpu_bytes,
            max_util_for_resume=1.0,
            max_attention_window=max_attention_window,
        ),
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=16,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=4096,
        enable_stats=enable_stats,
    )


def _create_v1_manager(
    *,
    gpu_bytes: int,
    enable_block_reuse: bool = True,
) -> KVCacheManagerV1:
    max_gpu_blocks = gpu_bytes // BYTES_PER_BLOCK
    return KVCacheManagerV1(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=True,
            max_tokens=max_gpu_blocks * TOKENS_PER_BLOCK,
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=16,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
    )


def _create_llm_request(
    request_id: int,
    tokens: list[int],
) -> LlmRequest:
    sampling_params = SamplingParams()
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
        is_streaming=False,
    )


def _context_batch(*requests) -> ScheduledRequests:
    batch = ScheduledRequests()
    for request in requests:
        batch.append_context_request(request)
    return batch


def _generation_batch(request: _StatsRequest) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.append_generation_request(request)
    return batch


@pytest.fixture
def resource_guard():
    managers = []
    resources = []

    def register(manager, *requests):
        if manager not in managers:
            managers.append(manager)
        resources.extend((manager, request) for request in requests)
        return manager

    yield register

    for manager, request in reversed(resources):
        manager.free_resources(request)
    for manager in reversed(managers):
        manager.shutdown()


def _finish_context(manager: KVCacheManagerV2, request: _StatsRequest) -> None:
    request.context_current_position = request.prompt_len
    request.context_remaining_length = 0
    manager.update_context_resources(_context_batch(request))


def _commit_and_get_stats(manager: KVCacheManagerV2, batch: ScheduledRequests):
    manager.commit_scheduled_kv_cache_stats(batch)
    stats_report = manager.get_iteration_stats()
    assert stats_report is not None
    assert manager.max_seq_len in stats_report.by_window_size
    return stats_report.by_window_size[manager.max_seq_len]


def _assert_iteration_delta(
    stats,
    *,
    alloc_total: int = 0,
    alloc_new: int = 0,
    reused: int = 0,
    full_reused: int = 0,
    partial_reused: int = 0,
    missed: int = 0,
    gen_alloc: int = 0,
    intra_copy: int = 0,
    intra_copy_bytes: int = 0,
) -> None:
    assert stats.iter_alloc_total_blocks == alloc_total
    assert stats.iter_alloc_new_blocks == alloc_new
    assert stats.iter_reused_blocks == reused
    assert stats.iter_full_reused_blocks == full_reused
    assert stats.iter_partial_reused_blocks == partial_reused
    assert stats.iter_missed_blocks == missed
    assert stats.iter_gen_alloc_blocks == gen_alloc
    assert stats.iter_intra_device_copy_blocks == intra_copy
    assert stats.iter_intra_device_copy_bytes == intra_copy_bytes


def _metric_call(
    *,
    alloc_total: int = 0,
    alloc_new: int = 0,
    reused: int = 0,
    missed: int = 0,
) -> dict[str, int]:
    return {
        "alloc_total_blocks": alloc_total,
        "alloc_new_blocks": alloc_new,
        "reused_blocks": reused,
        "missed_blocks": missed,
    }


def _assert_request_stats(
    request: LlmRequest,
    *,
    alloc_total: int = 0,
    alloc_new: int = 0,
    reused: int = 0,
    missed: int = 0,
) -> None:
    assert request.alloc_total_blocks == alloc_total
    assert request.alloc_new_blocks == alloc_new
    assert request.reused_blocks == reused
    assert request.missed_blocks == missed


def _run_v1_context(manager: KVCacheManagerV1, request: LlmRequest):
    batch = _context_batch(request)
    manager.prepare_resources(batch)
    stats = manager.get_iteration_stats()[manager.max_seq_len]
    simulate_prefill_completion_only_use_for_testing(request)
    manager.update_resources(batch)
    return stats


def _run_v1_generation(manager: KVCacheManagerV1, request: LlmRequest):
    batch = _generation_batch(request)
    manager.prepare_resources(batch)
    return manager.get_iteration_stats()[manager.max_seq_len]


def _run_v2_context(manager: KVCacheManagerV2, request: LlmRequest):
    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=request.context_remaining_length)
    simulate_prefill_completion_only_use_for_testing(request)
    manager.update_context_resources(_context_batch(request))
    return _commit_and_get_stats(manager, _context_batch(request))


def _run_v2_generation(manager: KVCacheManagerV2, request: LlmRequest):
    assert manager.try_allocate_generation(request)
    return _commit_and_get_stats(manager, _generation_batch(request))


def test_stats_disabled_suppresses_v2_accounting(resource_guard) -> None:
    request = _StatsRequest(1, list(range(8)), context_remaining_length=8)
    manager = resource_guard(_create_manager(gpu_bytes=8 << 20, enable_stats=False), request)

    assert not manager.kv_cache_manager_py_config.enable_stats
    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=8)
    _finish_context(manager, request)

    manager.commit_scheduled_kv_cache_stats(_context_batch(request))
    assert manager.get_iteration_stats() is None

    kv_stats = manager.get_kv_cache_stats()
    assert kv_stats.alloc_total_blocks == 0
    assert kv_stats.alloc_new_blocks == 0
    assert kv_stats.reused_blocks == 0
    assert kv_stats.missed_blocks == 0
    assert request.kv_cache_perf_metric_calls == []


def test_context_and_generation_stats_are_reported(resource_guard) -> None:
    request = _StatsRequest(1, list(range(8)), context_remaining_length=8)
    manager = resource_guard(_create_manager(gpu_bytes=8 << 20), request)

    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=8)
    _finish_context(manager, request)

    context_stats = _commit_and_get_stats(manager, _context_batch(request))
    _assert_iteration_delta(context_stats, alloc_total=2, alloc_new=2, missed=2)
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=2, alloc_new=2, missed=2),
    ]

    assert manager.try_allocate_generation(request)
    generation_stats = _commit_and_get_stats(manager, _generation_batch(request))
    _assert_iteration_delta(generation_stats, alloc_total=1, alloc_new=1, gen_alloc=1)
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=2, alloc_new=2, missed=2),
        _metric_call(alloc_total=1, alloc_new=1),
    ]


def test_reverted_generation_allocation_does_not_report_stats(resource_guard) -> None:
    request = _StatsRequest(1, list(range(8)), context_remaining_length=8)
    manager = resource_guard(_create_manager(gpu_bytes=8 << 20), request)

    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=8)
    _finish_context(manager, request)
    _commit_and_get_stats(manager, _context_batch(request))

    assert manager.try_allocate_generation(request)
    manager.revert_allocate_generation(request)
    manager.commit_scheduled_kv_cache_stats(_generation_batch(request))
    stats_report = manager.get_iteration_stats()
    assert stats_report is not None
    _assert_iteration_delta(stats_report.by_window_size[manager.max_seq_len])


def test_reverted_context_allocation_does_not_report_pending_stats(resource_guard) -> None:
    request = _StatsRequest(1, list(range(8)), context_remaining_length=8)
    manager = resource_guard(_create_manager(gpu_bytes=8 << 20), request)

    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=4)
    request.context_current_position = 4
    request.context_remaining_length = 4
    manager.update_context_resources(_context_batch(request))
    first_chunk_stats = _commit_and_get_stats(manager, _context_batch(request))
    _assert_iteration_delta(first_chunk_stats, alloc_total=1, alloc_new=1, missed=1)
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
    ]

    request.is_first_context_chunk = False
    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=4)
    manager.revert_allocate_context(request)
    manager.commit_scheduled_kv_cache_stats(_context_batch(request))

    reverted_stats_report = manager.get_iteration_stats()
    assert reverted_stats_report is not None
    _assert_iteration_delta(reverted_stats_report.by_window_size[manager.max_seq_len])
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
    ]
    kv_stats = manager.get_kv_cache_stats()
    assert kv_stats.alloc_total_blocks == 1
    assert kv_stats.alloc_new_blocks == 1
    assert kv_stats.missed_blocks == 1

    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=4)
    _finish_context(manager, request)
    second_chunk_stats = _commit_and_get_stats(manager, _context_batch(request))
    _assert_iteration_delta(second_chunk_stats, alloc_total=1, alloc_new=1, missed=1)
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
    ]


def test_chunked_context_reports_generation_alloc_only_in_generation(resource_guard) -> None:
    request = _StatsRequest(1, list(range(8)), context_remaining_length=8)
    manager = resource_guard(_create_manager(gpu_bytes=8 << 20), request)

    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=4)
    request.context_current_position = 4
    request.context_remaining_length = 4
    manager.update_context_resources(_context_batch(request))
    first_chunk_stats = _commit_and_get_stats(manager, _context_batch(request))
    assert first_chunk_stats.iter_gen_alloc_blocks == 0
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
    ]

    request.is_first_context_chunk = False
    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=4)
    _finish_context(manager, request)
    second_chunk_stats = _commit_and_get_stats(manager, _context_batch(request))
    assert second_chunk_stats.iter_gen_alloc_blocks == 0
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
    ]

    assert manager.try_allocate_generation(request)
    generation_stats = _commit_and_get_stats(manager, _generation_batch(request))
    _assert_iteration_delta(generation_stats, alloc_total=1, alloc_new=1, gen_alloc=1)
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
        _metric_call(alloc_total=1, alloc_new=1, missed=1),
        _metric_call(alloc_total=1, alloc_new=1),
    ]


def test_v2_generation_alloc_updates_request_metrics_unlike_v1(resource_guard) -> None:
    v1_request = _create_llm_request(101, list(range(8)))
    v2_request = _create_llm_request(201, list(range(8)))
    v1_manager = resource_guard(_create_v1_manager(gpu_bytes=8 << 20), v1_request)
    v2_manager = resource_guard(_create_manager(gpu_bytes=8 << 20), v2_request)

    v1_context_stats = _run_v1_context(v1_manager, v1_request)
    v2_context_stats = _run_v2_context(v2_manager, v2_request)
    _assert_iteration_delta(v1_context_stats, alloc_total=2, alloc_new=2, missed=2)
    _assert_iteration_delta(v2_context_stats, alloc_total=2, alloc_new=2, missed=2)
    _assert_request_stats(v1_request, alloc_total=2, alloc_new=2, missed=2)
    _assert_request_stats(v2_request, alloc_total=2, alloc_new=2, missed=2)

    v1_generation_stats = _run_v1_generation(v1_manager, v1_request)
    v2_generation_stats = _run_v2_generation(v2_manager, v2_request)
    _assert_iteration_delta(v1_generation_stats, alloc_total=1, alloc_new=1, gen_alloc=1)
    _assert_iteration_delta(v2_generation_stats, alloc_total=1, alloc_new=1, gen_alloc=1)
    # V2 records generation allocation in request-level alloc_total/new.
    # Legacy V1 only reports it through iteration/global generation counters.
    _assert_request_stats(v2_request, alloc_total=3, alloc_new=3, missed=2)
    _assert_request_stats(v1_request, alloc_total=2, alloc_new=2, missed=2)


def test_v2_partial_prompt_reuse_classification_matches_v1(resource_guard) -> None:
    v1_warmup_request = _create_llm_request(101, list(range(12)))
    v2_warmup_request = _create_llm_request(201, list(range(12)))
    v1_reuse_request = _create_llm_request(102, list(range(10)))
    v2_reuse_request = _create_llm_request(202, list(range(10)))
    v1_manager = resource_guard(
        _create_v1_manager(gpu_bytes=8 << 20), v1_warmup_request, v1_reuse_request
    )
    v2_manager = resource_guard(
        _create_manager(gpu_bytes=8 << 20), v2_warmup_request, v2_reuse_request
    )

    _run_v1_context(v1_manager, v1_warmup_request)
    _run_v2_context(v2_manager, v2_warmup_request)
    v1_manager.free_resources(v1_warmup_request)
    v2_manager.free_resources(v2_warmup_request)

    v1_reuse_stats = _run_v1_context(v1_manager, v1_reuse_request)
    v2_reuse_stats = _run_v2_context(v2_manager, v2_reuse_request)
    _assert_iteration_delta(v1_reuse_stats, reused=3, full_reused=2, partial_reused=1)
    _assert_iteration_delta(
        v2_reuse_stats,
        alloc_total=1,
        alloc_new=1,
        reused=3,
        full_reused=2,
        partial_reused=1,
        intra_copy=1,
        intra_copy_bytes=BYTES_PER_BLOCK,
    )
    _assert_request_stats(v1_reuse_request, reused=3)
    # V2 copies the partially reused block into a private slot before writing to it.
    _assert_request_stats(v2_reuse_request, alloc_total=1, alloc_new=1, reused=3)


def test_block_reuse_disabled_records_generation_alloc(resource_guard) -> None:
    request = _StatsRequest(1, list(range(8)), context_remaining_length=8)
    manager = resource_guard(_create_manager(gpu_bytes=8 << 20, enable_block_reuse=False), request)

    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=8)
    _finish_context(manager, request)

    stats = _commit_and_get_stats(manager, _context_batch(request))
    _assert_iteration_delta(stats, alloc_total=2, alloc_new=2, missed=2)
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=2, alloc_new=2, missed=2),
    ]

    assert manager.try_allocate_generation(request)
    generation_stats = _commit_and_get_stats(manager, _generation_batch(request))
    _assert_iteration_delta(generation_stats, alloc_total=1, alloc_new=1, gen_alloc=1)
    assert request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=2, alloc_new=2, missed=2),
        _metric_call(alloc_total=1, alloc_new=1),
    ]


def test_v2_partial_leaf_reuse_counts_reuse_with_private_copy(resource_guard) -> None:
    warmup_request = _StatsRequest(201, list(range(9)), context_remaining_length=9)
    reuse_request = _StatsRequest(202, list(range(10)), context_remaining_length=10)
    manager = resource_guard(
        _create_manager(gpu_bytes=8 << 20),
        warmup_request,
        reuse_request,
    )

    assert manager.prepare_context(warmup_request)
    assert manager.resize_context(warmup_request, num_tokens=9)
    _finish_context(manager, warmup_request)
    _commit_and_get_stats(manager, _context_batch(warmup_request))
    manager.free_resources(warmup_request)

    assert manager.prepare_context(reuse_request)
    assert reuse_request.prepopulated_prompt == (9, TOKENS_PER_BLOCK)
    assert manager.resize_context(reuse_request, num_tokens=1)
    _finish_context(manager, reuse_request)

    reuse_stats = _commit_and_get_stats(manager, _context_batch(reuse_request))
    _assert_iteration_delta(
        reuse_stats,
        alloc_total=1,
        alloc_new=1,
        reused=3,
        full_reused=2,
        partial_reused=1,
        intra_copy=1,
        intra_copy_bytes=BYTES_PER_BLOCK,
    )
    assert reuse_request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, reused=3),
    ]


def test_swa_context_reuse_stats_skip_stale_prefix_blocks(resource_guard) -> None:
    warmup_request = _StatsRequest(1, list(range(16)), context_remaining_length=16)
    reuse_request = _StatsRequest(2, list(range(16)), context_remaining_length=16)
    manager = resource_guard(
        _create_manager(
            gpu_bytes=8 << 20,
            num_layers=1,
            max_attention_window=[8],
        ),
        warmup_request,
        reuse_request,
    )

    assert manager.prepare_context(warmup_request)
    assert manager.resize_context(warmup_request, num_tokens=16)
    _finish_context(manager, warmup_request)
    manager.commit_scheduled_kv_cache_stats(_context_batch(warmup_request))
    assert manager.get_iteration_stats() is not None
    manager.free_resources(warmup_request)

    assert manager.prepare_context(reuse_request)
    assert reuse_request.prepopulated_prompt == (15, TOKENS_PER_BLOCK)
    assert manager.resize_context(reuse_request, num_tokens=1)
    _finish_context(manager, reuse_request)
    manager.commit_scheduled_kv_cache_stats(_context_batch(reuse_request))

    stats_report = manager.get_iteration_stats()
    assert stats_report is not None
    swa_stats = stats_report.by_window_size[8]
    _assert_iteration_delta(
        swa_stats,
        alloc_total=1,
        alloc_new=1,
        reused=2,
        full_reused=1,
        partial_reused=1,
        intra_copy=1,
        intra_copy_bytes=BYTES_PER_BLOCK,
    )
    assert reuse_request.kv_cache_perf_metric_calls == [
        _metric_call(alloc_total=1, alloc_new=1, reused=2),
    ]


def test_pool_group_stats_are_reported(resource_guard) -> None:
    request = _StatsRequest(1, list(range(8)), context_remaining_length=8)
    manager = resource_guard(
        _create_manager(
            gpu_bytes=16 << 20,
            num_layers=2,
            max_attention_window=[16, 8],
        ),
        request,
    )

    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens=8)
    _finish_context(manager, request)
    manager.commit_scheduled_kv_cache_stats(_context_batch(request))

    stats_report = manager.get_iteration_stats()
    assert stats_report is not None
    assert set(stats_report.by_window_size) == {manager.max_seq_len, 8}
    assert set(stats_report.by_pool_group) == {0}

    pool_group = stats_report.by_pool_group[0]
    assert pool_group.pool_group_id == 0
    assert set(pool_group.window_sizes) == {manager.max_seq_len, 8}
    _assert_iteration_delta(pool_group.stats, alloc_total=4, alloc_new=4, missed=4)
    _assert_iteration_delta(
        stats_report.by_window_size[manager.max_seq_len],
        alloc_total=2,
        alloc_new=2,
        missed=2,
    )
    _assert_iteration_delta(
        stats_report.by_window_size[8],
        alloc_total=2,
        alloc_new=2,
        missed=2,
    )
