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
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    DEFAULT_BEAM_INDEX,
    BatchDesc,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import init_cuda_once

TOKENS_PER_BLOCK = 4
MAX_SEQ_LEN = 16


class _CacheTierInitError(Exception):
    pass


@dataclass
class _FakeManagerConfig:
    cache_tiers: list[object]
    layers: list[object] = field(default_factory=lambda: [None])


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int) -> None:
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens: list[int] | None = None
        self.stopped_committing = False

    def commit(self, tokens: list[int]) -> None:
        self.committed_tokens = tokens
        self.num_committed_tokens += len(tokens)

    def stop_committing(self) -> None:
        self.stopped_committing = True


def _make_cache_config_for_test(
    kv_cache_config: KvCacheConfig,
    *,
    is_draft: bool = False,
    max_batch_size: int = 1,
    max_seq_len: int = 1024,
    max_num_tokens: int | None = None,
    max_draft_len: int = 0,
    num_extra_kv_tokens: int = 0,
    draft_reuse_lookahead: int = 0,
) -> KVCacheManagerConfig:
    cache_manager = object.__new__(KVCacheManagerV2)
    cache_manager.kv_cache_type = CacheType.SELFKONLY
    cache_manager.dtype = DataType.HALF
    cache_manager.head_dim_per_layer = [128]
    cache_manager.enable_swa_scratch_reuse = False
    cache_manager.num_extra_kv_tokens = num_extra_kv_tokens
    cache_manager.enable_stats = False
    cache_manager.block_reuse_policy = BlockReusePolicy(kv_cache_config.block_reuse_config.policy)
    cache_manager.is_draft = is_draft
    cache_manager.num_local_layers = 1
    cache_manager.pp_layers = [0]
    cache_manager.max_attention_window_vec = [None]
    cache_manager.max_seq_len = max_seq_len
    cache_manager.max_batch_size = max_batch_size
    cache_manager.max_num_tokens = max_num_tokens
    cache_manager.max_draft_len = max_draft_len
    cache_manager._draft_reuse_lookahead = draft_reuse_lookahead
    cache_manager.get_layer_bytes_per_token = lambda **_: 128

    return cache_manager._build_base_config(
        kv_cache_config,
        tokens_per_block=128,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
    )


def _make_manager_for_cache_tier_test(
    kv_cache_config: KvCacheConfig,
    impl_side_effect: list[object],
    *,
    add_secondary_gpu_tier: bool = False,
) -> tuple[KVCacheManagerV2, Mock]:
    impl_constructor = Mock(side_effect=impl_side_effect)

    def build_base_config(
        self: KVCacheManagerV2,
        config: KvCacheConfig,
        *,
        tokens_per_block: int,
        cache_tiers: list[object],
    ) -> _FakeManagerConfig:
        del self, config, tokens_per_block
        return _FakeManagerConfig(cache_tiers=cache_tiers)

    def build_cache_config(
        self: KVCacheManagerV2, config: _FakeManagerConfig
    ) -> _FakeManagerConfig:
        del self
        if add_secondary_gpu_tier:
            return _FakeManagerConfig(
                cache_tiers=[
                    config.cache_tiers[0],
                    GpuCacheTierConfig(quota=1 << 20),
                    *config.cache_tiers[1:],
                ],
                layers=config.layers,
            )
        return config

    fake_impl = impl_side_effect[-1]
    assert not isinstance(fake_impl, BaseException)
    fake_impl.layer_grouping = [[0]]
    fake_impl.pool_group_descs = []
    fake_impl.get_layer_group_id.side_effect = lambda _: 0

    module = "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2"
    with (
        patch(f"{module}.CuError", _CacheTierInitError),
        patch(f"{module}.KVCacheManagerPy", impl_constructor),
        patch.object(KVCacheManagerV2, "_build_base_config", build_base_config),
        patch.object(KVCacheManagerV2, "_build_cache_config", build_cache_config),
        patch.object(KVCacheManagerV2, "get_num_available_tokens", return_value=MAX_SEQ_LEN),
        patch.object(KVCacheManagerV2, "_prepare_page_table_tensor"),
        patch.object(KVCacheManagerV2, "_log_kv_cache_pool_lifecycle_mapping"),
    ):
        manager = KVCacheManagerV2(
            kv_cache_config,
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=1,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.HALF,
            vocab_size=16,
            execution_stream=Mock(),
        )
    return manager, impl_constructor


@pytest.mark.parametrize(
    (
        "enable_block_reuse",
        "block_reuse_policy",
        "is_draft",
        "draft_reuse_lookahead",
        "commit_min_snapshot",
    ),
    [
        (True, "all_reusable", False, 0, False),
        (True, "per_request", False, 0, True),
        (False, "per_request", False, 0, False),
        (True, "per_request", True, 0, True),
        (True, "per_request", True, 3, False),
    ],
)
def test_commit_min_snapshot_follows_block_reuse_policy(
    enable_block_reuse: bool,
    block_reuse_policy: str,
    is_draft: bool,
    draft_reuse_lookahead: int,
    commit_min_snapshot: bool,
) -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            block_reuse_config=BlockReuseConfig(policy=block_reuse_policy),
            enable_partial_reuse=True,
        ),
        is_draft=is_draft,
        draft_reuse_lookahead=draft_reuse_lookahead,
    )

    assert config.commit_min_snapshot is commit_min_snapshot
    assert config.enable_partial_reuse


@pytest.mark.parametrize("enable_partial_reuse", [False, True])
def test_propagates_partial_reuse_config(enable_partial_reuse: bool) -> None:
    config = _make_cache_config_for_test(KvCacheConfig(enable_partial_reuse=enable_partial_reuse))

    assert config.enable_partial_reuse is enable_partial_reuse


def test_pool_ratio_overrides_constraints() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(pool_ratio=[1.0], avg_seq_len=256, host_cache_size=0),
        max_batch_size=3,
        max_num_tokens=2048,
    )

    assert config.initial_pool_ratio == pytest.approx([1.0])
    assert config.typical_step is None
    assert config.constraints == []


def test_default_uses_allocator_fallback() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(host_cache_size=0),
        max_batch_size=3,
        max_seq_len=1024,
        max_num_tokens=2048,
        max_draft_len=2,
    )

    assert config.initial_pool_ratio is None
    assert config.typical_step is None
    assert config.constraints == []


def test_avg_seq_len_builds_warmup_constraints() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(host_cache_size=0, avg_seq_len=1024),
        max_batch_size=3,
        max_seq_len=1024,
        max_num_tokens=2048,
        max_draft_len=2,
    )

    assert config.typical_step == BatchDesc(
        [KVCacheDesc(capacity=2048, history_length=0)]
        + [KVCacheDesc(capacity=1024, history_length=1021)] * 2
    )
    assert config.constraints == [
        BatchDesc(
            [
                KVCacheDesc(capacity=1024, history_length=1023),
                KVCacheDesc(capacity=3, history_length=0),
                KVCacheDesc(capacity=3, history_length=0),
            ]
        ),
        BatchDesc([KVCacheDesc(capacity=2048, history_length=0)]),
    ]


def test_avg_seq_len_updates_typical_step() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(avg_seq_len=256),
        max_batch_size=3,
        max_seq_len=1024,
        max_num_tokens=2048,
        max_draft_len=2,
    )

    assert config.typical_step == BatchDesc(
        [KVCacheDesc(capacity=2048, history_length=0)]
        + [KVCacheDesc(capacity=256, history_length=253)] * 2
    )


def test_avg_seq_len_must_not_exceed_max_seq_len() -> None:
    with pytest.raises(ValueError, match="avg_seq_len"):
        _make_cache_config_for_test(
            KvCacheConfig(avg_seq_len=2048),
            max_seq_len=1024,
        )


def test_disk_secondary_tier_enables_eviction(tmp_path) -> None:
    impl = Mock()
    manager, impl_constructor = _make_manager_for_cache_tier_test(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            disk_cache_size=16 << 20,
            disk_cache_path=str(tmp_path),
        ),
        [impl],
    )

    assert manager.can_evict
    assert impl_constructor.call_count == 1
    cache_tiers = impl_constructor.call_args.args[0].cache_tiers
    assert [type(tier) for tier in cache_tiers] == [
        GpuCacheTierConfig,
        DiskCacheTierConfig,
    ]


def test_disk_init_failure_does_not_use_host_fallback(tmp_path) -> None:
    with pytest.raises(_CacheTierInitError, match="disk tier init failed"):
        _make_manager_for_cache_tier_test(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=0,
                disk_cache_size=16 << 20,
                disk_cache_path=str(tmp_path),
            ),
            [_CacheTierInitError("disk tier init failed"), Mock()],
        )


@pytest.mark.parametrize(
    ("add_secondary_gpu_tier", "expected_can_evict"),
    [(False, False), (True, True)],
)
def test_host_init_fallback_recomputes_eviction_capability(
    add_secondary_gpu_tier: bool,
    expected_can_evict: bool,
) -> None:
    impl = Mock()
    manager, impl_constructor = _make_manager_for_cache_tier_test(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=16 << 20,
        ),
        [_CacheTierInitError("host tier init failed"), impl],
        add_secondary_gpu_tier=add_secondary_gpu_tier,
    )

    assert manager.can_evict is expected_can_evict
    assert impl_constructor.call_count == 2
    initial_tiers = impl_constructor.call_args_list[0].args[0].cache_tiers
    fallback_tiers = impl_constructor.call_args_list[1].args[0].cache_tiers
    assert any(isinstance(tier, HostCacheTierConfig) for tier in initial_tiers)
    assert all(isinstance(tier, GpuCacheTierConfig) for tier in fallback_tiers)
    assert len(fallback_tiers) == 1 + int(add_secondary_gpu_tier)


def test_host_init_fallback_drops_only_host_tier(tmp_path) -> None:
    impl = Mock()
    manager, impl_constructor = _make_manager_for_cache_tier_test(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=16 << 20,
            disk_cache_size=16 << 20,
            disk_cache_path=str(tmp_path),
        ),
        [_CacheTierInitError("host tier init failed"), impl],
    )

    assert manager.can_evict
    assert impl_constructor.call_count == 2
    initial_tiers = impl_constructor.call_args_list[0].args[0].cache_tiers
    fallback_tiers = impl_constructor.call_args_list[1].args[0].cache_tiers
    assert [type(tier) for tier in initial_tiers] == [
        GpuCacheTierConfig,
        HostCacheTierConfig,
        DiskCacheTierConfig,
    ]
    assert [type(tier) for tier in fallback_tiers] == [
        GpuCacheTierConfig,
        DiskCacheTierConfig,
    ]


def test_extra_tokens_are_in_context_capacity() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(avg_seq_len=264),
        max_batch_size=1,
        max_seq_len=264,
        max_num_tokens=256,
        max_draft_len=3,
        num_extra_kv_tokens=2,
    )

    assert config.typical_step == BatchDesc([KVCacheDesc(capacity=258, history_length=0)])
    assert config.constraints[1] == BatchDesc([KVCacheDesc(capacity=258, history_length=0)])


def test_real_llm_request_keeps_target_and_draft_reuse_views_in_sync() -> None:
    request = LlmRequest(
        request_id=39,
        max_new_tokens=16,
        input_tokens=[0] * 512,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.state = LlmRequestState.CONTEXT_INIT

    target_cache = SimpleNamespace(num_committed_tokens=128)
    draft_cache = SimpleNamespace(
        num_committed_tokens=64,
        capacity=64,
        is_active=True,
        resize=Mock(return_value=True),
    )
    target_manager = object.__new__(KVCacheManagerV2)
    target_manager.is_draft = False
    target_manager.enable_block_reuse = True
    target_manager.tokens_per_block = 64
    target_manager.kv_cache_map = {request.py_request_id: target_cache}
    draft_manager = object.__new__(KVCacheManagerV2)
    draft_manager.is_draft = True
    draft_manager.enable_block_reuse = True
    draft_manager.tokens_per_block = 64
    draft_manager.num_extra_kv_tokens = 0
    draft_manager.kv_cache_map = {request.py_request_id: draft_cache}
    draft_manager._resume_and_restore = Mock(return_value=True)

    target_manager.finalize_context_reuse(request, 128)
    draft_manager.finalize_context_reuse(request, 64)

    assert request.context_current_position == 128
    assert request.prepopulated_prompt_len == 128
    request.use_draft_model = True
    assert request.context_current_position == 64
    assert request.prepopulated_prompt_len == 64
    request.use_draft_model = False

    # Paired V2 admission chooses one common frontier, then copies the target
    # chunk size to the independent draft request fields before the forward.
    target_cache.num_committed_tokens = 64
    target_manager.finalize_context_reuse(request, 64)
    request.context_chunk_size = 128
    assert draft_manager.try_allocate_draft_context(request, 128)

    assert request.context_current_position == 64
    assert request.context_chunk_size == 128
    request.use_draft_model = True
    assert request.context_current_position == 64
    assert request.context_chunk_size == 128
    request.use_draft_model = False

    request.move_to_next_context_chunk()

    assert request.context_current_position == 192
    assert request.context_chunk_size == 0
    request.use_draft_model = True
    assert request.context_current_position == 192
    assert request.context_chunk_size == 0
    request.use_draft_model = False


def test_try_allocate_draft_context_reserves_scheduled_chunk() -> None:
    request = SimpleNamespace(
        py_request_id=39,
        prompt_len=61901,
        context_current_position=128,
        context_chunk_size=32000,
        prepopulated_prompt_len=128,
        use_draft_model=False,
        py_draft_tokens=[1, 2, 3],
        has_draft_tokens=True,
        num_draft_tokens=3,
    )
    kv_cache = Mock(capacity=0)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._resume_and_restore = Mock(return_value=True)

    assert manager.try_allocate_draft_context(request, 32000)

    manager._resume_and_restore.assert_called_once_with(request.py_request_id, kv_cache)
    kv_cache.resize.assert_called_once_with(32128)
    assert request.context_chunk_size == 32000
    assert not request.use_draft_model


def test_try_allocate_draft_context_keeps_existing_excess_capacity() -> None:
    request = SimpleNamespace(
        py_request_id=39,
        prompt_len=61901,
        context_current_position=128,
        context_chunk_size=32000,
        prepopulated_prompt_len=128,
        use_draft_model=False,
        py_draft_tokens=[1, 2, 3],
    )
    kv_cache = Mock(capacity=70000)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._resume_and_restore = Mock(return_value=True)

    assert manager.try_allocate_draft_context(request, 32000)

    kv_cache.resize.assert_called_once_with(70000)


def test_try_allocate_draft_context_reserves_exact_last_chunk_capacity() -> None:
    request = LlmRequest(
        request_id=39,
        max_new_tokens=16,
        input_tokens=[0] * 61901,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=[1, 2, 3],
    )
    request.state = LlmRequestState.CONTEXT_INIT
    request.context_current_position = 32000
    request.context_chunk_size = 29901
    request.use_draft_model = True
    request.context_current_position = 32000
    request.use_draft_model = False

    kv_cache = Mock(capacity=32000, is_active=True)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._resume_and_restore = Mock(return_value=True)

    assert request.is_last_context_chunk
    assert manager.try_allocate_draft_context(request, 29901 + 3)

    kv_cache.resize.assert_called_once_with(61904)
    request.use_draft_model = True
    assert request.context_current_position == 32000
    assert request.context_chunk_size == 29901
    request.use_draft_model = False


def test_draft_generation_reserve_is_preallocated_and_exactly_reverted() -> None:
    request = SimpleNamespace(
        py_request_id=38,
        py_draft_tokens=[1, 2, 3],
        use_draft_model=False,
    )
    kv_cache = Mock(capacity=100, is_active=True)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._kv_reserve_draft_tokens = 8
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._allocated_draft_lens = {}
    manager._generation_pre_resize_caps = {}

    assert manager.try_allocate_generation(request)

    kv_cache.resize.assert_called_once_with(109)
    assert manager._allocated_draft_lens[request.py_request_id] == 8
    assert manager._generation_pre_resize_caps[request.py_request_id] == 100

    manager.revert_allocate_generation(request)

    assert kv_cache.resize.call_args_list[-1].args == (100,)
    assert request.py_request_id not in manager._allocated_draft_lens
    assert request.py_request_id not in manager._generation_pre_resize_caps


def test_target_generation_reserves_dynamic_draft_width_before_normalization() -> None:
    request = SimpleNamespace(
        py_request_id=38,
        py_draft_tokens=[],
        use_draft_model=False,
        is_disagg_generation_transmission_complete=False,
    )
    kv_cache = Mock(capacity=100, is_active=True)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = False
    manager._kv_reserve_draft_tokens = 4
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._allocated_draft_lens = {}
    manager._generation_pre_resize_caps = {}

    assert manager.try_allocate_generation(request)

    kv_cache.resize.assert_called_once_with(105)
    assert manager._allocated_draft_lens[request.py_request_id] == 4


def test_generation_update_reclaims_exact_target_reserve() -> None:
    request = SimpleNamespace(
        py_request_id=38,
        py_rewind_len=2,
        py_num_accepted_draft_tokens=2,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        max_beam_num_tokens=103,
    )
    kv_cache = Mock(capacity=105, is_active=True)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = False
    manager._kv_reserve_draft_tokens = 4
    manager.kv_compression_manages_history = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._allocated_draft_lens = {request.py_request_id: 4}
    manager._generation_pre_resize_caps = {request.py_request_id: 100}
    batch = ScheduledRequests()
    batch.generation_requests.append(request)

    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2._update_kv_cache_draft_token_location"
    ):
        manager.update_resources(batch)

    kv_cache.resize.assert_called_once_with(103, 102)
    assert request.py_request_id not in manager._allocated_draft_lens
    assert request.py_request_id not in manager._generation_pre_resize_caps


@pytest.mark.parametrize("is_draft", [False, True])
def test_prepare_context_does_not_reapply_reuse_on_later_chunk(is_draft: bool) -> None:
    request = LlmRequest(
        request_id=39,
        max_new_tokens=4,
        input_tokens=[0] * 512,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.state = LlmRequestState.CONTEXT_INIT
    for use_draft_model in (False, True):
        request.use_draft_model = use_draft_model
        request.context_current_position = 64
        request.context_chunk_size = 128
    request.use_draft_model = False
    request.move_to_next_context_chunk()

    kv_cache = SimpleNamespace(num_committed_tokens=64, is_active=True)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = is_draft
    manager.enable_block_reuse = True
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._resume_and_restore = Mock(return_value=True)

    assert manager.prepare_context(request)

    request.use_draft_model = is_draft
    assert request.context_current_position == 192
    request.use_draft_model = False


def test_scheduler_preallocated_draft_generation_is_not_resized_twice() -> None:
    request = SimpleNamespace(
        py_request_id=38,
        py_draft_tokens=[1, 2, 3],
        use_draft_model=False,
    )
    kv_cache = Mock(capacity=109, is_active=True)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._kv_reserve_draft_tokens = 8
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._generation_pre_resize_caps = {request.py_request_id: 100}
    manager._resume_and_restore = Mock(return_value=True)
    batch = ScheduledRequests()
    batch.generation_requests.append(request)

    manager._prepare_draft_resources(batch)

    kv_cache.resize.assert_not_called()
    assert not request.use_draft_model


def test_disagg_draft_context_fallback_resizes_new_cache() -> None:
    request = SimpleNamespace(
        py_request_id=39,
        lora_task_id=None,
        cache_salt=None,
        is_dummy=False,
        is_disagg_generation_init_state=True,
        context_current_position=100,
        context_chunk_size=20,
        is_last_context_chunk=True,
        py_draft_tokens=[1, 2, 3],
        use_draft_model=False,
    )
    kv_cache = Mock(capacity=0)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_block_reuse = True
    manager.num_extra_kv_tokens = 1
    manager._scheduler_owns_draft_admission = True
    manager.kv_cache_map = {}
    manager._create_kv_cache = Mock(return_value=kv_cache)
    manager._resume_and_restore = Mock(return_value=True)
    batch = ScheduledRequests()
    batch.append_context_request(request)

    manager._prepare_draft_resources(batch)

    kv_cache.resize.assert_called_once_with(124)
    assert not request.use_draft_model


def test_draft_reuse_key_includes_shifted_lookahead() -> None:
    request = SimpleNamespace(
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
    )
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._draft_reuse_lookahead = 3
    manager.tokens_per_block = 4

    prompt_a = list(range(11))
    prompt_b = prompt_a.copy()
    prompt_b[4] = 99

    keys_a = manager._block_reuse_tokens(prompt_a, request, end=8)
    keys_b = manager._block_reuse_tokens(prompt_b, request, end=8)

    assert len(keys_a) == 8
    assert isinstance(keys_a[3], bytes)
    assert len(keys_a[3]) == 32
    assert keys_a[3] != keys_b[3]
    assert keys_a[7] == keys_b[7]


def test_draft_reuse_excludes_block_without_full_lookahead() -> None:
    request = SimpleNamespace(
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
    )
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._draft_reuse_lookahead = 3
    manager.tokens_per_block = 4

    assert manager._block_reuse_tokens(list(range(6)), request, end=6) == []


@pytest.mark.parametrize(
    ("compute_end", "expected_ceiling"),
    [(128, 64), (131, 128)],
)
def test_draft_commit_ceiling_stops_before_first_chunk_shifted_tail(
    compute_end: int, expected_ceiling: int
) -> None:
    request = SimpleNamespace(
        py_request_id=41,
        py_last_context_chunk=(0, compute_end),
    )
    kv_cache = SimpleNamespace(num_committed_tokens=0)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._draft_reuse_lookahead = 3
    manager.tokens_per_block = 64
    manager._draft_context_commit_ceilings = {}

    assert manager._draft_context_commit_limit(request, kv_cache, 3) == expected_ceiling

    request.py_last_context_chunk = (compute_end, compute_end + 128)
    assert manager._draft_context_commit_limit(request, kv_cache, 3) == expected_ceiling


@pytest.mark.parametrize("runtime_draft_len", [None, 2])
def test_draft_commit_ceiling_rejects_unproven_layer_depth(
    runtime_draft_len: int | None,
) -> None:
    request = SimpleNamespace(
        py_request_id=42,
        py_last_context_chunk=(0, 131),
    )
    kv_cache = SimpleNamespace(num_committed_tokens=64)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._draft_reuse_lookahead = 3
    manager.tokens_per_block = 64
    manager._draft_context_commit_ceilings = {}

    assert manager._draft_context_commit_limit(request, kv_cache, runtime_draft_len) == 64


def test_inactive_draft_cache_records_first_executed_chunk_ceiling() -> None:
    request = SimpleNamespace(
        py_request_id=43,
        py_last_context_chunk=(0, 128),
        is_last_context_chunk=False,
    )
    kv_cache = SimpleNamespace(is_active=False, num_committed_tokens=0)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._draft_reuse_lookahead = 3
    manager.tokens_per_block = 64
    manager._draft_context_commit_ceilings = {}
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    batch = ScheduledRequests()
    batch.append_context_request(request)

    manager.update_context_resources(batch, runtime_draft_len=3)

    assert manager._draft_context_commit_ceilings[request.py_request_id] == 64
    request.py_last_context_chunk = (128, 256)
    assert manager._draft_context_commit_limit(request, kv_cache, 3) == 64


def test_draft_commit_ceiling_keeps_native_history_at_executed_context() -> None:
    request = SimpleNamespace(
        py_request_id=44,
        py_last_context_chunk=(0, 131),
        is_dummy_request=False,
        is_last_context_chunk=False,
        context_current_position=131,
        context_remaining_length=125,
        get_tokens=lambda beam_id: list(range(256)),
        get_tokens_view=lambda beam_id: np.arange(256, dtype=np.int32),
    )
    kv_cache = Mock(is_active=True, num_committed_tokens=0)

    def commit(tokens: list[int]) -> None:
        kv_cache.num_committed_tokens += len(tokens)

    kv_cache.commit.side_effect = commit
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._draft_reuse_lookahead = 3
    manager.tokens_per_block = 64
    manager._draft_context_commit_ceilings = {}
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager.enable_block_reuse = True
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager.conversation_manager = None
    manager._block_reuse_tokens = lambda tokens, request, start, end: tokens[start:end]
    batch = ScheduledRequests()
    batch.append_context_request(request)

    manager.update_context_resources(batch, runtime_draft_len=3)

    assert kv_cache.resize.call_args_list == [call(None, 131)]
    assert len(kv_cache.commit.call_args.args[0]) == 128
    assert kv_cache.num_committed_tokens == 128

    request.py_last_context_chunk = (131, 256)
    request.context_current_position = 256
    request.context_remaining_length = 0
    manager.update_context_resources(batch, runtime_draft_len=3)

    assert kv_cache.resize.call_args_list == [call(None, 131), call(None, 256)]
    assert kv_cache.commit.call_count == 1
    assert kv_cache.num_committed_tokens == 128


def test_try_commit_blocks_commits_partial_block_at_context_end() -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        context_current_position=10,
        context_remaining_length=0,
        get_tokens=lambda beam_id: list(range(10)),
        # The C++ backend takes get_tokens_view on this path; it yields a contiguous
        # 1-D int32 view, so commit() sees an ndarray slice rather than a list.
        get_tokens_view=lambda beam_id: np.arange(10, dtype=np.int32),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    # list() so the assertion holds whichever token source the active backend used:
    # a plain list (Python backend) or an int32 ndarray slice (C++ backend).
    assert list(kv_cache.committed_tokens) == [4, 5, 6, 7, 8, 9]
    assert kv_cache.num_committed_tokens == 10
    assert kv_cache.stopped_committing


def test_try_commit_blocks_commits_draft_request_view() -> None:
    request = LlmRequest(
        request_id=1,
        max_new_tokens=4,
        input_tokens=list(range(10)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.state = LlmRequestState.CONTEXT_INIT
    request.use_draft_model = True
    request.context_current_position = request.prompt_len
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = True
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert list(kv_cache.committed_tokens) == [4, 5, 6, 7, 8, 9]
    assert kv_cache.num_committed_tokens == 10
    assert kv_cache.stopped_committing
    request.use_draft_model = False


@dataclass
class _ContextRequest:
    request_id: int
    tokens: list[int]
    context_remaining_length: int
    conversation_id: str
    py_request_id: int = field(init=False)
    py_conversation_params: ConversationParams | None = field(init=False)
    use_conversation_params: bool = True
    lora_task_id: int | None = None
    cache_salt: str | None = None
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    is_disagg_generation_init_state: bool = False
    is_dummy_request: bool = False
    context_current_position: int = 0
    prepopulated_prompt: tuple[int, int] | None = None
    multimodal_hashes: None = None
    multimodal_positions: None = None
    multimodal_lengths: None = None

    def __post_init__(self) -> None:
        self.py_request_id = self.request_id
        if not self.use_conversation_params:
            self.py_conversation_params = None
            return
        self.py_conversation_params = ConversationParams(conversation_id=self.conversation_id)

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

    def get_tokens_view(self, beam_id: int = DEFAULT_BEAM_INDEX) -> np.ndarray:
        """Mirror LlmRequest.get_tokens_view, which the C++ backend takes on the reuse path.

        The real binding returns a zero-copy contiguous 1-D int32 view of the token buffer;
        the dtype matters because it selects the C++ int32 ingest fast path.
        """
        assert beam_id == DEFAULT_BEAM_INDEX
        return np.asarray(self.tokens, dtype=np.int32)

    def set_prepopulated_prompt_len(self, length: int, tokens_per_block: int) -> None:
        self.prepopulated_prompt = (length, tokens_per_block)


@pytest.fixture
def max_num_turns() -> int:
    return 1


@pytest.fixture
def manager(max_num_turns: int) -> KVCacheManagerV2:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    init_cuda_once()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=True,
            max_gpu_total_bytes=16 << 20,
            max_attention_window=[MAX_SEQ_LEN, TOKENS_PER_BLOCK],
            max_util_for_resume=1.0,
            block_reuse_config=BlockReuseConfig(
                policy="per_conversation",
                max_num_turns=max_num_turns,
            ),
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
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


def _prepare_context_resources(
    manager: KVCacheManagerV2,
    *requests: _ContextRequest,
) -> ScheduledRequests:
    batch = _context_batch(*requests)
    manager.prepare_resources(batch)
    return batch


def _update_context_resources(
    manager: KVCacheManagerV2,
    batch: ScheduledRequests,
) -> None:
    manager.update_context_resources(batch)


def _free_if_active(
    manager: KVCacheManagerV2,
    request: _ContextRequest,
) -> None:
    manager.free_resources(request)


def _run_context(
    manager: KVCacheManagerV2,
    request: _ContextRequest,
) -> None:
    batch = _prepare_context_resources(manager, request)
    assert manager.prepare_context(request)
    request.context_remaining_length = request.prompt_len - request.context_current_position
    assert manager.resize_context(request, num_tokens=request.context_remaining_length)
    request.context_current_position = request.prompt_len
    request.context_remaining_length = 0
    _update_context_resources(manager, batch)


def test_per_conversation_policy_delays_commit_until_last_context_chunk(
    manager: KVCacheManagerV2,
) -> None:
    request = _ContextRequest(1, list(range(8)), 8, "conv-1")

    try:
        batch = _prepare_context_resources(manager, request)
        assert manager.prepare_context(request)
        assert manager.resize_context(request, num_tokens=4)
        request.context_current_position = 4
        request.context_remaining_length = 4
        _update_context_resources(manager, batch)

        kv_cache = manager.kv_cache_map[request.py_request_id]
        assert kv_cache.num_committed_tokens == 0
        assert kv_cache.history_length == 4

        request.is_first_context_chunk = False
        batch = _prepare_context_resources(manager, request)
        assert manager.prepare_context(request)
        assert manager.resize_context(request, num_tokens=4)
        request.context_current_position = 8
        request.context_remaining_length = 0
        _update_context_resources(manager, batch)

        assert kv_cache.num_committed_tokens == 8
        assert kv_cache.history_length == 8
    finally:
        _free_if_active(manager, request)


def test_per_conversation_policy_without_params_uses_per_request_commit(
    manager: KVCacheManagerV2,
) -> None:
    request = _ContextRequest(
        1,
        list(range(8)),
        8,
        "conv-1",
        use_conversation_params=False,
    )
    batch = _context_batch(request)

    try:
        assert manager.prepare_context(request)
        assert manager.resize_context(request, num_tokens=4)
        request.context_current_position = 4
        request.context_remaining_length = 4
        _update_context_resources(manager, batch)

        kv_cache = manager.kv_cache_map[request.py_request_id]
        assert kv_cache.num_committed_tokens == 0
        assert kv_cache.history_length == 4
    finally:
        if request.py_request_id in manager.kv_cache_map:
            manager.free_resources(request)


def test_per_conversation_policy_releases_cancelled_request(
    manager: KVCacheManagerV2,
) -> None:
    request_a = _ContextRequest(1, list(range(8)), 8, "conv-1")
    request_b = _ContextRequest(2, list(range(8)), 8, "conv-1")

    try:
        batch_a = _prepare_context_resources(manager, request_a)
        assert manager.prepare_context(request_a)
        assert manager.resize_context(request_a, num_tokens=4)
        request_a.context_current_position = 4
        request_a.context_remaining_length = 4
        _update_context_resources(manager, batch_a)
        _free_if_active(manager, request_a)

        batch_b = _prepare_context_resources(manager, request_b)
        with patch(
            "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.logger.warning"
        ) as mock_warning:
            assert manager.prepare_context(request_b)
            mock_warning.assert_not_called()
        assert manager.resize_context(request_b, num_tokens=request_b.prompt_len)
        request_b.context_current_position = request_b.prompt_len
        request_b.context_remaining_length = 0
        _update_context_resources(manager, batch_b)
    finally:
        _free_if_active(manager, request_b)
        _free_if_active(manager, request_a)


def test_per_conversation_policy_drops_previous_divergent_blocks(
    manager: KVCacheManagerV2,
) -> None:
    request_a = _ContextRequest(1, list(range(8)), 8, "conv-1")
    request_b = _ContextRequest(
        2,
        [*range(8), 100, 101, 102, 103],
        12,
        "conv-1",
    )
    request_old_prompt = _ContextRequest(3, list(range(8)), 8, "conv-2")
    try:
        _run_context(manager, request_a)
        _free_if_active(manager, request_a)

        _run_context(manager, request_b)
        assert request_b.prepopulated_prompt_len == 8
        _free_if_active(manager, request_b)

        assert manager.prepare_context(request_old_prompt)
        assert request_old_prompt.prepopulated_prompt_len == 0
    finally:
        _free_if_active(manager, request_old_prompt)
        _free_if_active(manager, request_b)
        _free_if_active(manager, request_a)


@pytest.mark.parametrize("max_num_turns", [2])
def test_per_conversation_policy_retains_configured_number_of_turns(
    manager: KVCacheManagerV2,
) -> None:
    request_a = _ContextRequest(1, list(range(8)), 8, "conv-1")
    request_b = _ContextRequest(2, list(range(100, 108)), 8, "conv-1")
    request_a_probe = _ContextRequest(3, list(range(8)), 8, "conv-2")
    request_c = _ContextRequest(4, list(range(200, 208)), 8, "conv-1")
    request_a_after_eviction = _ContextRequest(5, list(range(8)), 8, "conv-3")

    try:
        _run_context(manager, request_a)
        _free_if_active(manager, request_a)
        _run_context(manager, request_b)
        _free_if_active(manager, request_b)

        assert manager.prepare_context(request_a_probe)
        assert request_a_probe.prepopulated_prompt_len == request_a_probe.prompt_len - 1
        _free_if_active(manager, request_a_probe)

        _run_context(manager, request_c)
        _free_if_active(manager, request_c)

        assert manager.prepare_context(request_a_after_eviction)
        assert request_a_after_eviction.prepopulated_prompt_len == 0
    finally:
        _free_if_active(manager, request_a_after_eviction)
        _free_if_active(manager, request_c)
        _free_if_active(manager, request_a_probe)
        _free_if_active(manager, request_b)
        _free_if_active(manager, request_a)


def test_per_conversation_policy_ignores_overlapping_request(
    manager: KVCacheManagerV2,
) -> None:
    request_a = _ContextRequest(1, list(range(8)), 8, "conv-1")
    request_b = _ContextRequest(2, [0, 1, 2, 3, 100, 101, 102, 103], 8, "conv-1")
    request_old_prompt = _ContextRequest(3, list(range(8)), 8, "conv-2")
    conversation_params = request_b.py_conversation_params

    try:
        batch_a = _prepare_context_resources(manager, request_a)
        assert manager.prepare_context(request_a)
        assert manager.resize_context(request_a, num_tokens=4)
        request_a.context_current_position = 4
        request_a.context_remaining_length = 4
        _update_context_resources(manager, batch_a)

        batch_b = _prepare_context_resources(manager, request_b)
        with patch(
            "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.logger.warning"
        ) as mock_warning:
            assert manager.prepare_context(request_b)
            mock_warning.assert_called_once_with(
                "Conversation conv-1 already has current request 1. "
                "Request 2 will ignore conversation params."
            )
        assert request_b.py_conversation_params is conversation_params
        assert manager.resize_context(request_b, num_tokens=request_b.prompt_len)
        request_b.context_current_position = request_b.prompt_len
        request_b.context_remaining_length = 0
        _update_context_resources(manager, batch_b)
        _free_if_active(manager, request_b)

        request_a.is_first_context_chunk = False
        batch_a = _prepare_context_resources(manager, request_a)
        assert manager.prepare_context(request_a)
        assert manager.resize_context(request_a, num_tokens=4)
        request_a.context_current_position = 8
        request_a.context_remaining_length = 0
        _update_context_resources(manager, batch_a)
        _free_if_active(manager, request_a)

        assert manager.prepare_context(request_old_prompt)
        assert request_old_prompt.prepopulated_prompt_len == request_old_prompt.prompt_len - 1
    finally:
        _free_if_active(manager, request_old_prompt)
        _free_if_active(manager, request_b)
        _free_if_active(manager, request_a)


def test_iteration_stats_reports_physical_pool_groups_without_window_metadata() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_stats = True
    snapshot_delta = SimpleNamespace(
        iter_snapshot_lookups=2,
        iter_snapshot_hits=1,
        iter_snapshot_misses=1,
        iter_reused_tokens=32,
        iter_unreused_tokens=16,
        iter_aligned_snapshot_hits=1,
        iter_unaligned_snapshot_hits=0,
    )
    manager.impl = SimpleNamespace(
        cache_tier_list=[object()],
        get_and_reset_iteration_stats=lambda: {},
        get_and_reset_ssm_snapshot_iteration_stats=lambda: {3: snapshot_delta},
    )
    manager._stats_life_cycle_metadata = lambda: {3: (1, None, "ssm")}
    manager._storage_pool_groups_by_window = lambda: {}
    manager._get_and_reset_iteration_peak_block_stats = lambda _level: [None, None]
    manager._get_storage_statistics = lambda _level: [object(), object()]
    manager._build_pool_group_iteration_stats = lambda pool_group_id, *_args: pool_group_id

    stats = manager.get_iteration_stats()

    assert stats.by_pool_group == {0: 0, 1: 1}
    ssm_stats = stats.by_life_cycle[3]
    assert ssm_stats.kind == "ssm"
    assert ssm_stats.pool_group_id == 1
    assert ssm_stats.snapshot_stats.iter_snapshot_hit_rate == 0.5
    assert ssm_stats.snapshot_stats.iter_reused_tokens == 32


def test_disagg_role_mapper_kinds_default_to_indexed():
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role

    manager = object.__new__(KVCacheManagerV2)

    # K/V default to the TRTLLM head-major layout; the index-key side cache
    # defaults to REPLICATED (every shipped index-K — DSA V1, MiniMax M3 —
    # is TP-replicated). The INDEX_KEY entry is inert unless a subclass
    # registers such buffers.
    assert manager.get_disagg_role_mapper_kinds() == {
        Role.ALL: MapperKind.INDEXED,
        Role.INDEX_KEY: MapperKind.REPLICATED,
    }
