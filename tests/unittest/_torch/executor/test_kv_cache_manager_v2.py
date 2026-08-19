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
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
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
    cache_manager.get_layer_bytes_per_token = lambda **_: 128
    # Mirrors __init__: without helix the ledger block equals the physical
    # page (the helper re-enacts construction for partial instances).
    cache_manager._ledger_tokens_per_block = 128

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
    ("enable_block_reuse", "block_reuse_policy", "is_draft", "commit_min_snapshot"),
    [
        (True, "all_reusable", False, False),
        (True, "per_request", False, True),
        (False, "per_request", False, False),
        (True, "per_request", True, True),
    ],
)
def test_commit_min_snapshot_follows_block_reuse_policy(
    enable_block_reuse: bool,
    block_reuse_policy: str,
    is_draft: bool,
    commit_min_snapshot: bool,
) -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            block_reuse_config=BlockReuseConfig(policy=block_reuse_policy),
            enable_partial_reuse=True,
        ),
        is_draft=is_draft,
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


def test_prefill_constraint_registered_without_avg_seq_len() -> None:
    """The chunked-prefill constraint must not be gated behind avg_seq_len.

    Regression lock: when this constraint is missing, StorageManager falls back
    to a DECODE-shaped BatchDesc whose scratch range is provably empty, so SWA
    scratch reuse is inert for every model that does not set avg_seq_len, and
    the SWA pool is sized from swa_floor_blocks alone.
    """
    config = _make_cache_config_for_test(
        KvCacheConfig(host_cache_size=0),
        max_batch_size=3,
        max_seq_len=1024,
        max_num_tokens=2048,
        max_draft_len=2,
    )

    assert config.initial_pool_ratio is None
    # typical_step stays opt-in: it needs avg_seq_len, which is workload knowledge.
    assert config.typical_step is None
    assert config.constraints == [BatchDesc([KVCacheDesc(capacity=2048, history_length=0)])]


def test_prefill_constraint_includes_extra_kv_tokens() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(host_cache_size=0),
        max_batch_size=3,
        max_seq_len=1024,
        max_num_tokens=2048,
        num_extra_kv_tokens=4,
    )

    assert config.constraints == [BatchDesc([KVCacheDesc(capacity=2052, history_length=0)])]


def test_no_prefill_constraint_without_max_num_tokens() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(host_cache_size=0),
        max_batch_size=3,
        max_seq_len=1024,
        max_num_tokens=None,
    )

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
    return_perf_metrics: bool = False
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


def test_cold_pool_group_iteration_stats_sum_all_cold_levels() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._cold_pool_group_membership = lambda: ((0, frozenset({0, 1})),)
    life_cycle_metadata = {
        0: (0, 32, "attention"),
        1: (1, 64, "attention"),
    }
    secondary_stats_by_level = [
        [SimpleNamespace(total=7, available=2, evictable=1, slot_sizes=(4096,))],
        [SimpleNamespace(total=11, available=5, evictable=3, slot_sizes=(4096,))],
    ]
    secondary_peak_stats_by_level = [
        [SimpleNamespace(available=1, unavailable=6, evictable=2)],
        [SimpleNamespace(available=4, unavailable=7, evictable=3)],
    ]

    report = manager._build_cold_pool_group_iteration_stats(
        life_cycle_metadata,
        primary_stats=(),
        secondary_stats_by_level=secondary_stats_by_level,
        primary_peak_stats=(),
        secondary_peak_stats_by_level=secondary_peak_stats_by_level,
    )

    cold_group = report[0]
    assert cold_group.slot_size == (4096,)
    assert cold_group.window_sizes == (32, 64)
    assert cold_group.stats.secondary_max_num_blocks == 18
    assert cold_group.stats.secondary_free_num_blocks == 7
    assert cold_group.stats.secondary_used_num_blocks == 11
    assert cold_group.stats.secondary_evictable_num_blocks == 4
    assert cold_group.stats.secondary_peak_free_num_blocks == 5
    assert cold_group.stats.secondary_peak_used_num_blocks == 13
    assert cold_group.stats.secondary_peak_evictable_num_blocks == 5


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


# ---------------------------------------------------------------------------
# SWA scratch reuse: PER_LAYER flat page-index rotation.
#
# This is the arithmetic that addresses a scratch block on the FlashInfer path.
# It is the highest-risk code in the feature because it fails *silently*: a
# wrong index reads another layer's KV rather than raising, so an end-to-end run
# still exits 0 with plausible-looking output. The bug actually hit during
# Gemma4 bring-up (a layer_idx-less lookup yielding BAD_PAGE_INDEX) was found
# only by an illegal memory access on a B200, which is far too late and far too
# expensive a feedback loop for integer arithmetic.
#
# These tests pin the invariants the flat page table depends on, on a real
# Gemma4-12B-shaped configuration, with no GPU and no model.
# ---------------------------------------------------------------------------

# Gemma4-12B: 48 layers, 40 sliding (W=1024) / 8 full, K and V per layer.
GEMMA4_NUM_SWA_LAYERS = 40
GEMMA4_KV_FACTOR = 2
# One slot holds `scale` sub-pages: kv_factor per layer across the shared group.
GEMMA4_SCALE = GEMMA4_NUM_SWA_LAYERS * GEMMA4_KV_FACTOR
# Each scratch block advances by one K/V pair.
GEMMA4_SCRATCH_PAGES_PER_BLOCK = GEMMA4_KV_FACTOR


def _reference_flat_index(position, scratch_pages, scale, layer_offset, slot_ids, div_factor):
    """Independent restatement of the device kernel's arithmetic.

    Deliberately written as a scalar loop from the formula rather than by
    calling the implementation, so agreement is evidence rather than tautology.
    """
    total = position * scratch_pages
    slot = int(slot_ids[total // scale])
    sub = (total % scale + layer_offset) % scale
    return (slot * scale + sub) // div_factor


def _k_layer_offset(layer_idx):
    return layer_idx * GEMMA4_KV_FACTOR


class TestSwaScratchFlatIndexRotation:
    """Correctness of compute_scratch_flat_page_indices on a Gemma4 shape."""

    NUM_BLOCKS = 41  # a realistic prefill scratch range (~2340 tokens, W=1024, tpb=32)
    SLOT_IDS = tuple(range(7, 7 + 8))  # arbitrary non-contiguous-looking slot ids

    def _indices(self, layer_idx, div_factor=1, count=None):
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
            compute_scratch_flat_page_indices,
        )

        return compute_scratch_flat_page_indices(
            0,
            self.NUM_BLOCKS if count is None else count,
            GEMMA4_SCRATCH_PAGES_PER_BLOCK,
            GEMMA4_SCALE,
            _k_layer_offset(layer_idx),
            self.SLOT_IDS,
            div_factor,
        )

    def test_matches_device_kernel_formula_for_every_swa_layer(self):
        """Host rotation must equal the device kernel's, for all 40 SWA layers."""
        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            got = self._indices(layer_idx)
            expected = [
                _reference_flat_index(
                    pos,
                    GEMMA4_SCRATCH_PAGES_PER_BLOCK,
                    GEMMA4_SCALE,
                    _k_layer_offset(layer_idx),
                    self.SLOT_IDS,
                    1,
                )
                for pos in range(self.NUM_BLOCKS)
            ]
            assert got.tolist() == expected, f"layer {layer_idx} diverges from the kernel formula"

    def test_v_stays_exactly_one_subpage_after_k(self):
        """The precondition a flat page table cannot express any other way.

        A flat table carries one index per block plus a kv_factor axis, so it can
        only address V if V remains K+1 *after* the rotation. If this breaks,
        attention reads K as V and produces silently wrong output rather than an
        error. _validate_per_layer_kv_adjacency promises this; here it is checked
        against the arithmetic that has to honour it.
        """
        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            k_off = _k_layer_offset(layer_idx)
            k = self._indices(layer_idx)
            v = [
                _reference_flat_index(
                    pos, GEMMA4_SCRATCH_PAGES_PER_BLOCK, GEMMA4_SCALE, k_off + 1, self.SLOT_IDS, 1
                )
                for pos in range(self.NUM_BLOCKS)
            ]
            assert [b - a for a, b in zip(k.tolist(), v)] == [1] * self.NUM_BLOCKS, (
                f"layer {layer_idx}: V is not adjacent to K under the scratch rotation"
            )

    def test_no_two_swa_layers_alias_the_same_subpage(self):
        """Distinct layers must never resolve to the same page for a block.

        Aliasing is the failure mode that corrupts KV without any crash: two
        layers would read and write each other's cache. With scale == 40 layers
        x kv_factor, all 40 layers must land on 40 distinct K sub-pages.
        """
        per_layer = [
            self._indices(layer_idx).tolist() for layer_idx in range(GEMMA4_NUM_SWA_LAYERS)
        ]
        for position in range(self.NUM_BLOCKS):
            seen = {indices[position] for indices in per_layer}
            assert len(seen) == GEMMA4_NUM_SWA_LAYERS, (
                f"block position {position}: only {len(seen)} distinct pages for "
                f"{GEMMA4_NUM_SWA_LAYERS} layers -- layers alias each other's KV"
            )

    def test_indices_stay_inside_the_addressed_slots(self):
        """Every index must fall inside a slot the descriptor actually holds."""
        valid = {slot * GEMMA4_SCALE + sub for slot in self.SLOT_IDS for sub in range(GEMMA4_SCALE)}
        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            assert set(self._indices(layer_idx).tolist()) <= valid, (
                f"layer {layer_idx} produced an index outside the descriptor's slots"
            )

    def test_rotation_advances_with_block_position(self):
        """The rotation is the reason SHARED addressing cannot work.

        If a layer's sub-page were fixed across block positions it could be
        folded into a base pointer and none of the PER_LAYER machinery would be
        needed. Assert it genuinely moves, so this test fails if someone
        "simplifies" the rotation away.
        """
        idx = self._indices(layer_idx=3)
        sub_pages = [int(i) % GEMMA4_SCALE for i in idx.tolist()]
        assert len(set(sub_pages)) > 1, "sub-page did not rotate with block position"

    def test_kv_factor_division_preserves_pairing(self):
        """div_factor halves the index space; K must stay kv_factor-aligned.

        The flat table indexes block-granular entries, so the caller divides by
        kv_factor. That is only sound when K is kv_factor-aligned -- one of the
        conditions _validate_per_layer_kv_adjacency enforces.
        """
        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            raw = self._indices(layer_idx, div_factor=1).tolist()
            halved = self._indices(layer_idx, div_factor=GEMMA4_KV_FACTOR).tolist()
            assert all(r % GEMMA4_KV_FACTOR == 0 for r in raw), (
                f"layer {layer_idx}: K index is not kv_factor-aligned, so dividing by "
                "kv_factor would collapse K and V onto the same entry"
            )
            assert halved == [r // GEMMA4_KV_FACTOR for r in raw]

    def test_empty_range_is_empty(self):
        assert self._indices(layer_idx=0, count=0).tolist() == []


class TestSwaScratchSegmentClamping:
    """Range/segment clamping in apply_scratch_to_block_segment.

    The scratch range and a request's block count are computed independently, so
    they can fail to overlap. Getting the clamp wrong does not raise -- it shifts
    the wrong blocks by layer_offset and leaves them pointing at another layer's
    pages, which reads as plausible output. These cases are cheap to pin and
    impossible to notice at runtime.
    """

    SCALE = GEMMA4_SCALE
    SPB = GEMMA4_SCRATCH_PAGES_PER_BLOCK
    SLOT_IDS = tuple(range(7, 15))
    LAYER_OFFSET = 6  # layer 3, K

    def _apply(self, values, beg, end, div_factor=1):
        import numpy as np

        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
            apply_scratch_to_block_segment,
        )

        seg = np.asarray(values, dtype=np.int32).copy()
        apply_scratch_to_block_segment(
            seg,
            beg,
            end,
            self.SPB,
            self.SCALE,
            self.LAYER_OFFSET,
            self.SLOT_IDS,
            div_factor,
        )
        return seg.tolist()

    def test_range_entirely_before_segment_shifts_every_block(self):
        """beg < end <= 0: nothing is scratch, so every block just gains the offset.

        Regression: a naive ``seg[hi:]`` with a negative ``hi`` indexes from the
        end of the array and shifts only a suffix, silently leaving the leading
        blocks addressed as if the buffer were still SHARED-based.
        """
        values = [10, 11, 12, 13]
        got = self._apply(values, beg=-3, end=-1)
        assert got == [v + self.LAYER_OFFSET for v in values]

    def test_empty_range_shifts_every_block(self):
        values = [10, 11, 12, 13]
        assert self._apply(values, beg=2, end=2) == [v + self.LAYER_OFFSET for v in values]

    def test_range_entirely_after_segment_shifts_every_block(self):
        values = [10, 11, 12, 13]
        assert self._apply(values, beg=9, end=12) == [v + self.LAYER_OFFSET for v in values]

    def test_bad_page_index_is_never_shifted(self):
        """BAD_PAGE_INDEX must stay the sentinel; shifting it makes it a real page."""
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import BAD_PAGE_INDEX

        got = self._apply([BAD_PAGE_INDEX, 11, BAD_PAGE_INDEX], beg=5, end=5)
        assert got[0] == BAD_PAGE_INDEX and got[2] == BAD_PAGE_INDEX
        assert got[1] == 11 + self.LAYER_OFFSET

    def test_partial_overlap_splits_scratch_and_non_scratch(self):
        """Only blocks inside the range rotate; the rest are shifted."""
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
            compute_scratch_flat_page_indices,
        )

        values = [10, 11, 12, 13, 14]
        got = self._apply(values, beg=1, end=3)
        rotated = compute_scratch_flat_page_indices(
            0, 2, self.SPB, self.SCALE, self.LAYER_OFFSET, self.SLOT_IDS, 1
        ).tolist()
        assert got[0] == 10 + self.LAYER_OFFSET
        assert got[1:3] == rotated
        assert got[3:] == [13 + self.LAYER_OFFSET, 14 + self.LAYER_OFFSET]

    def test_range_clipped_to_segment_does_not_false_trip_slot_guard(self):
        """A range extending past the request's blocks is clipped, not rejected.

        Only the blocks actually addressed consume slots, so the bound must be
        checked on the clipped range. Checking [beg, end) instead would reject
        descriptors that are perfectly serviceable.
        """
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
            compute_scratch_flat_page_indices,
        )

        # 2 blocks addressed needs 1 slot; the unclipped range would demand 13.
        got = self._apply([10, 11], beg=0, end=500)
        assert (
            got
            == compute_scratch_flat_page_indices(
                0, 2, self.SPB, self.SCALE, self.LAYER_OFFSET, self.SLOT_IDS, 1
            ).tolist()
        )

    def test_insufficient_slots_raises_with_numbers(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="scratch slot"):
            # 400 blocks x 2 pages / scale 80 needs 10 slots; only 8 provided.
            self._apply(list(range(400)), beg=0, end=400)
