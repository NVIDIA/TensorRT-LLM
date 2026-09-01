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

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
    _KVCacheManagerInitStatus,
    _sync_kv_cache_manager_init_status,
    _update_kv_cache_draft_token_location,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import (
    BlockReuseConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
)
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
        self.published_keys: list[object] = []
        self.history_length = 0
        self.is_active = True
        self.enable_swa_scratch_reuse = True
        self.stopped_committing = False

    def commit(self, tokens: list[int]) -> None:
        self.committed_tokens = tokens
        self.published_keys.extend(tokens)
        self.num_committed_tokens += len(tokens)
        self.history_length = max(self.history_length, self.num_committed_tokens)

    def resize(self, capacity, history_length: int) -> bool:
        del capacity
        self.history_length = history_length
        return True

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
    max_attention_window_vec: list[int | None] | None = None,
    pp_layers: list[int] | None = None,
) -> KVCacheManagerConfig:
    if max_attention_window_vec is None:
        max_attention_window_vec = [None]
    if pp_layers is None:
        pp_layers = list(range(len(max_attention_window_vec)))
    assert len(max_attention_window_vec) == len(pp_layers)

    cache_manager = object.__new__(KVCacheManagerV2)
    cache_manager.kv_cache_type = CacheType.SELFKONLY
    cache_manager.dtype = DataType.HALF
    cache_manager.head_dim_per_layer = [128] * len(pp_layers)
    cache_manager.enable_swa_scratch_reuse = False
    cache_manager.num_extra_kv_tokens = num_extra_kv_tokens
    cache_manager.enable_stats = False
    cache_manager.block_reuse_policy = BlockReusePolicy(kv_cache_config.block_reuse_config.policy)
    cache_manager.is_draft = is_draft
    cache_manager.num_local_layers = len(pp_layers)
    cache_manager.pp_layers = pp_layers
    cache_manager.max_attention_window_vec = max_attention_window_vec
    cache_manager.max_seq_len = max_seq_len
    cache_manager.max_batch_size = max_batch_size
    cache_manager.max_num_tokens = max_num_tokens
    cache_manager.max_draft_len = max_draft_len
    cache_manager._can_publish_block_reuse = not is_draft
    cache_manager.enable_joint_kv_cache_reuse = False
    cache_manager.reuse_match_backoff = 0
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
    cold_page_codec_provider: object | None = None,
    spec_config=None,
    is_draft: bool = False,
    is_disagg: bool = False,
    joint_reuse: bool = False,
    mapping: Mapping | None = None,
) -> tuple[KVCacheManagerV2, Mock]:
    impl_constructor = Mock(side_effect=impl_side_effect)
    if mapping is None:
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

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

    fake_impl = next(
        (item for item in reversed(impl_side_effect) if not isinstance(item, BaseException)),
        None,
    )
    if fake_impl is not None:
        fake_impl.layer_grouping = [[0]]
        fake_impl.pool_group_descs = []
        fake_impl.get_layer_group_id.side_effect = lambda _: 0

    module = "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2"
    with (
        patch(f"{module}.CuError", _CacheTierInitError),
        patch(f"{module}.IndexMapper"),
        patch(f"{module}.KVCacheManagerPy", impl_constructor),
        patch.object(KVCacheManagerV2, "_build_base_config", build_base_config),
        patch.object(KVCacheManagerV2, "_build_cache_config", build_cache_config),
        patch.object(KVCacheManagerV2, "get_num_available_tokens", return_value=MAX_SEQ_LEN),
        patch.object(KVCacheManagerV2, "_prepare_page_table_tensor"),
        patch.object(KVCacheManagerV2, "_log_kv_cache_pool_lifecycle_mapping"),
        patch(f"{module}.get_pp_layers", return_value=([0], 1)),
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
            mapping=mapping,
            dtype=DataType.HALF,
            spec_config=spec_config,
            is_draft=is_draft,
            is_disagg=is_disagg,
            joint_kv_cache_reuse=joint_reuse,
            vocab_size=16,
            execution_stream=Mock(),
            cold_page_codec_provider=cold_page_codec_provider,
        )
    return manager, impl_constructor


def _multi_rank_host_fallback_consensus_worker() -> tuple[int, int, int, bool]:
    """Exercise the real world collective from an MPI worker."""
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    rank = mpi_rank()
    world_size = mpi_world_size()
    initial_impl = Mock()
    fallback_impl = Mock()
    impl_side_effect: list[object] = (
        [initial_impl, fallback_impl]
        if rank == 0
        else [_CacheTierInitError("rank-local host tier failure"), fallback_impl]
    )

    manager, impl_constructor = _make_manager_for_cache_tier_test(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=16 << 20,
        ),
        impl_side_effect,
        mapping=Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=world_size,
        ),
    )

    return (
        rank,
        impl_constructor.call_count,
        initial_impl.shutdown.call_count,
        any(
            isinstance(tier, HostCacheTierConfig)
            for tier in manager.kv_cache_manager_py_config.cache_tiers
        ),
    )


def test_base_config_uses_local_attention_window_order() -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(),
        max_attention_window_vec=[128, None],
        pp_layers=[3, 4],
    )

    assert [layer.sliding_window_size for layer in config.layers] == [
        128,
        None,
    ]


def test_draft_token_relocation_uses_local_cache_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    request = SimpleNamespace(
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_num_accepted_draft_tokens=1,
        py_num_accepted_draft_tokens_indices=[0],
    )
    batch = ScheduledRequests()
    batch.generation_requests = [request]

    accepted_offsets = object()
    accepted_indices = object()
    rewind_adjustments = object()

    def locate_accepted_draft_tokens(
        requests: list[object],
    ) -> tuple[object, object, object]:
        del requests
        return accepted_offsets, accepted_indices, rewind_adjustments

    monkeypatch.setattr(
        kv_cache_v2_module,
        "_locate_accepted_draft_tokens",
        locate_accepted_draft_tokens,
    )

    local_pool_pointers = object()
    local_block_offsets = object()
    cache_manager = SimpleNamespace(
        num_layers=8,
        num_local_layers=2,
        num_kv_heads_per_layer=[8, 8],
        head_dim=128,
        max_attention_window_vec=[None, None],
        max_seq_len=8192,
        max_total_draft_tokens=31,
        max_blocks_per_seq=256,
        tokens_per_block=32,
        kv_cache_pool_mapping=[[0, 0], [0, 1]],
        kv_cache_pool_pointers=[local_pool_pointers],
    )
    attention_metadata = SimpleNamespace(
        kv_lens_cuda=torch.tensor([128], dtype=torch.int32),
        kv_cache_block_offsets=[local_block_offsets],
        host_kv_cache_pool_pointers=object(),
        host_kv_cache_pool_mapping=object(),
    )
    update_op = Mock()
    monkeypatch.setattr(
        torch.ops.tensorrt_llm,
        "update_kv_cache_draft_token_location",
        update_op,
        raising=False,
    )

    _update_kv_cache_draft_token_location(
        cache_manager,
        batch,
        attention_metadata,
        kv_cache_dtype_byte_size=2,
    )

    update_op.assert_called_once()
    (
        actual_accepted_offsets,
        actual_accepted_indices,
        past_key_value_lengths,
        use_paged_kv_cache,
        layer_count,
        num_kv_heads,
        head_size_in_bytes,
        rewind_draft_token_count,
        max_kv_cache_len,
        actual_rewind_adjustments,
        past_key_value_list,
        pool_pointers,
        block_offsets,
        max_blocks_per_seq,
        tokens_per_block,
        stream,
    ) = update_op.call_args.args
    assert actual_accepted_offsets is accepted_offsets
    assert actual_accepted_indices is accepted_indices
    assert torch.equal(past_key_value_lengths, attention_metadata.kv_lens_cuda)
    assert use_paged_kv_cache is True
    assert layer_count == cache_manager.num_local_layers
    assert num_kv_heads == 8
    assert head_size_in_bytes == 256
    assert rewind_draft_token_count == cache_manager.max_total_draft_tokens
    assert max_kv_cache_len == cache_manager.max_seq_len
    assert actual_rewind_adjustments is rewind_adjustments
    assert past_key_value_list is None
    assert pool_pointers is local_pool_pointers
    assert block_offsets is local_block_offsets
    assert max_blocks_per_seq == cache_manager.max_blocks_per_seq
    assert tokens_per_block == cache_manager.tokens_per_block
    assert stream is None


@pytest.mark.parametrize(
    (
        "enable_block_reuse",
        "block_reuse_policy",
        "is_draft",
        "commit_min_snapshot",
    ),
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


@pytest.mark.parametrize(
    ("joint_reuse", "expected_core_backoff"),
    [(False, 1), (True, 1)],
    ids=["single_pool_trims_in_core", "paired_pools_trim_in_core"],
)
@pytest.mark.parametrize(
    "spec_config",
    [
        Eagle3DecodingConfig(
            max_draft_len=1,
            speculative_model="draft-model",
        ),
        MTPDecodingConfig(max_draft_len=1),
    ],
    ids=["eagle3_one_model", "mtp_eagle_one_model"],
)
def test_one_model_prompt_lookahead_configures_reuse_backoff(
    spec_config: Eagle3DecodingConfig | MTPDecodingConfig,
    joint_reuse: bool,
    expected_core_backoff: int,
) -> None:
    """Keep #18295's shift-by-one input aligned with both reuse protocols."""
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        max_gpu_total_bytes=16 << 20,
    )
    manager, _ = _make_manager_for_cache_tier_test(
        kv_cache_config,
        [Mock()],
        spec_config=spec_config,
        is_draft=joint_reuse,
        joint_reuse=joint_reuse,
    )

    # The public manager keeps the semantic span as lookup evidence. A claim
    # limit retains the following D tokens so the core can trim in the same
    # match, for both single and paired pools.
    assert manager.reuse_match_backoff == 1
    prompt = list(range(65))
    request = SimpleNamespace(
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
    )
    manager._reuse_token_source = lambda _: prompt
    assert list(manager._context_reuse_tokens(request)) == prompt
    assert list(manager._context_reuse_tokens(request, reuse_limit=64)) == prompt
    assert list(manager._context_reuse_tokens(request, reuse_limit=63)) == prompt[:64]

    # Both protocols bind the lookahead evidence and backoff in one core match.
    core_config = manager._build_base_config(
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
    )
    assert core_config.reuse_match_backoff == expected_core_backoff
    assert replace(core_config, commit_min_snapshot=True).reuse_match_backoff == 1


@pytest.mark.parametrize(
    ("fresh_cache", "expected_lookup_tokens"),
    [(True, 3), (False, None)],
    ids=["fresh_lookup", "resumed_cache"],
)
def test_prepare_context_cache_records_lookup_without_mutating_cursor(
    fresh_cache: bool, expected_lookup_tokens: int | None
) -> None:
    """Snapshot metadata survives the cursor-free cache preparation split."""
    request = SimpleNamespace(
        py_request_id=7,
        lora_task_id=3,
        cache_salt=11,
        is_dummy=False,
        return_perf_metrics=False,
        prompt_len=8,
        context_current_position=6,
        is_first_context_chunk=True,
        is_disagg_generation_init_state=False,
    )
    kv_cache = Mock(num_committed_tokens=2)
    manager = object.__new__(KVCacheManagerV2)
    manager.conversation_manager = None
    manager.enable_block_reuse = True
    manager._has_cp_helix = False
    manager.kv_cache_map = {} if fresh_cache else {request.py_request_id: kv_cache}
    manager._stream = SimpleNamespace(cuda_stream=Mock())
    manager._context_reuse_tokens = Mock(return_value=[10, 11, 12])
    manager._create_kv_cache = Mock(return_value=kv_cache)
    manager._record_branch_snapshot_point = Mock()
    manager._resume_and_restore = Mock(return_value=True)

    assert manager.prepare_context_cache(request, reuse_limit=2) == 2

    assert request.context_current_position == 6
    manager._record_branch_snapshot_point.assert_called_once_with(
        request, kv_cache, expected_lookup_tokens
    )
    if fresh_cache:
        manager._context_reuse_tokens.assert_called_once_with(request, 2)
    else:
        manager._context_reuse_tokens.assert_not_called()


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


def test_kv_cache_manager_init_status_sync_uses_world_max() -> None:
    mapping = SimpleNamespace(world_size=2)
    dist = Mock()
    dist.allreduce.return_value = int(_KVCacheManagerInitStatus.USE_NO_HOST)

    with patch.object(Distributed, "get", return_value=dist):
        status = _sync_kv_cache_manager_init_status(_KVCacheManagerInitStatus.KEEP_HOST, mapping)

    assert status == _KVCacheManagerInitStatus.USE_NO_HOST
    dist.allreduce.assert_called_once_with(
        int(_KVCacheManagerInitStatus.KEEP_HOST), op=ReduceOp.MAX
    )


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device (MPI) build required")
def test_world_ranks_converge_on_hostless_fallback() -> None:
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    session = MpiPoolSession(n_workers=2)
    try:
        results = session.submit_sync(_multi_rank_host_fallback_consensus_worker)
    finally:
        session.shutdown()

    assert sorted(results) == [(0, 2, 1, False), (1, 2, 0, False)]


def test_local_fallback_failure_is_shared_before_raising() -> None:
    module = "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2"

    with (
        patch(
            f"{module}._sync_kv_cache_manager_init_status",
            side_effect=[
                _KVCacheManagerInitStatus.USE_NO_HOST,
                _KVCacheManagerInitStatus.ABORT,
            ],
        ) as sync_status,
        pytest.raises(RuntimeError, match="fallback init failed"),
    ):
        _make_manager_for_cache_tier_test(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=16 << 20,
            ),
            [
                _CacheTierInitError("host tier init failed"),
                RuntimeError("fallback init failed"),
            ],
        )

    assert [call.args[0] for call in sync_status.call_args_list] == [
        _KVCacheManagerInitStatus.USE_NO_HOST,
        _KVCacheManagerInitStatus.ABORT,
    ]


def test_peer_fallback_failure_discards_local_candidate() -> None:
    initial_impl = Mock()
    fallback_impl = Mock()
    module = "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2"

    with (
        patch(
            f"{module}._sync_kv_cache_manager_init_status",
            side_effect=[
                _KVCacheManagerInitStatus.USE_NO_HOST,
                _KVCacheManagerInitStatus.ABORT,
            ],
        ),
        pytest.raises(RuntimeError, match="failed on another rank"),
    ):
        _make_manager_for_cache_tier_test(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=16 << 20,
            ),
            [initial_impl, fallback_impl],
        )

    initial_impl.shutdown.assert_called_once_with()
    fallback_impl.shutdown.assert_called_once_with()


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


@pytest.mark.cpu_only
def test_host_init_fallback_recreates_cold_codec_and_keeps_disk(tmp_path) -> None:
    impl = Mock()
    codecs = [object(), object()]
    codec_provider = Mock()
    codec_provider.create_cold_page_codec.side_effect = codecs
    manager, impl_constructor = _make_manager_for_cache_tier_test(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=16 << 20,
            disk_cache_size=16 << 20,
            disk_cache_path=str(tmp_path),
        ),
        [_CacheTierInitError("host tier init failed"), impl],
        cold_page_codec_provider=codec_provider,
    )

    assert manager.can_evict
    assert codec_provider.create_cold_page_codec.call_count == 2
    assert impl_constructor.call_count == 2
    assert impl_constructor.call_args_list[0].kwargs["cold_page_codec"] is codecs[0]
    assert impl_constructor.call_args_list[1].kwargs["cold_page_codec"] is codecs[1]
    fallback_tiers = impl_constructor.call_args_list[1].args[0].cache_tiers
    assert [type(tier) for tier in fallback_tiers] == [
        GpuCacheTierConfig,
        DiskCacheTierConfig,
    ]


@pytest.mark.cpu_only
def test_cold_codec_provider_receives_draft_role() -> None:
    impl = Mock()
    codec_provider = Mock()
    codec_provider.create_cold_page_codec.return_value = object()
    _make_manager_for_cache_tier_test(
        KvCacheConfig(max_gpu_total_bytes=16 << 20),
        [impl],
        cold_page_codec_provider=codec_provider,
        is_draft=True,
    )

    assert codec_provider.create_cold_page_codec.call_args.kwargs["is_draft"] is True


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
    manager._can_publish_block_reuse = True
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert list(kv_cache.committed_tokens) == list(range(4, 10))
    assert kv_cache.num_committed_tokens == 10
    assert kv_cache.stopped_committing


def test_generation_allocation_reserves_dynamic_width() -> None:
    request = SimpleNamespace(
        py_request_id=80,
        py_num_accepted_draft_tokens=2,
        py_rewind_len=2,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        max_beam_num_tokens=103,
    )
    kv_cache = Mock(is_active=True, capacity=100)

    def resize(capacity, history_length=None):
        if capacity is not None:
            kv_cache.capacity = capacity
        return True

    kv_cache.resize.side_effect = resize
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager._has_cp_helix = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._allocated_draft_lens = {}
    manager._kv_reserve_draft_tokens = 4
    manager._effective_draft_len = Mock(return_value=2)
    manager.kv_compression_manages_history = False

    assert manager.try_allocate_generation(request)
    assert kv_cache.resize.call_args_list[0].args == (105,)
    assert manager._allocated_draft_lens[request.py_request_id] == 4

    batch = ScheduledRequests()
    batch.generation_requests.append(request)
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2._update_kv_cache_draft_token_location"
    ):
        manager.update_resources(batch)

    assert kv_cache.resize.call_args_list[1].args == (103, 102)
    assert request.py_request_id not in manager._allocated_draft_lens


def _revert_context_request(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        py_ctx_pre_resize_cap=64,
        prompt_len=512,
        context_current_position=256,
        context_chunk_size=192,
        estimated_reusable_tokens=128,
        set_prepopulated_prompt_len=Mock(),
    )


def test_context_revert_drops_unshrinkable_cache_and_rewinds_progress() -> None:
    request = _revert_context_request(91)
    # history (256) past pre_cap (64) is the steady state for a sliding-window
    # request part-way through its prompt, so the cache cannot be shrunk back.
    kv_cache = Mock(is_active=True, capacity=128, history_length=256)
    manager = object.__new__(KVCacheManagerV2)
    manager.tokens_per_block = 64
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager.free_resources = Mock()

    assert manager.revert_allocate_context(request) is False

    manager.free_resources.assert_called_once_with(request)
    kv_cache.resize.assert_not_called()
    # The dropped pages are what the cursor described, so prefill restarts.
    request.set_prepopulated_prompt_len.assert_called_once_with(0, 64)
    assert request.context_current_position == 0
    assert request.context_chunk_size == 512
    assert request.estimated_reusable_tokens == 0
    assert request.py_ctx_pre_resize_cap is None


def test_context_revert_shrinks_in_place_when_history_fits() -> None:
    request = _revert_context_request(92)
    kv_cache = Mock(is_active=True, capacity=128, history_length=32)
    kv_cache.resize.return_value = True
    manager = object.__new__(KVCacheManagerV2)
    manager.tokens_per_block = 64
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager.free_resources = Mock()

    assert manager.revert_allocate_context(request) is True

    manager.free_resources.assert_not_called()
    kv_cache.resize.assert_called_once_with(64, 32)
    kv_cache.suspend.assert_called_once_with()
    # Shrinking keeps the prefix intact, so the cursor must not move.
    request.set_prepopulated_prompt_len.assert_not_called()
    assert request.context_current_position == 256
    assert request.context_chunk_size == 192


def test_draft_manager_keeps_shared_progress_across_context_and_generation() -> None:
    request = LlmRequest(
        request_id=39,
        max_new_tokens=4,
        input_tokens=[0] * 512,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=[1, 2, 3],
    )
    request.state = LlmRequestState.CONTEXT_INIT
    request.context_current_position = 64
    request.context_chunk_size = 128
    request.move_to_next_context_chunk()

    kv_cache = Mock(num_committed_tokens=64, is_active=True, capacity=192)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_block_reuse = True
    manager.enable_joint_kv_cache_reuse = True
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    observed_progress_views = []

    def resume_and_restore(request_id, current_cache):
        assert (request_id, current_cache) == (request.py_request_id, kv_cache)
        observed_progress_views.append(request.use_draft_model)
        return True

    manager._resume_and_restore = Mock(side_effect=resume_and_restore)

    assert manager.prepare_context(request)
    assert request.context_current_position == 192

    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    manager._allocated_draft_lens = {request.py_request_id: 3}
    manager._required_gen_capacity = Mock()
    batch = ScheduledRequests()
    batch.generation_requests.append(request)
    manager._prepare_draft_resources(batch)

    assert observed_progress_views == [False, False]
    assert not request.use_draft_model
    manager._required_gen_capacity.assert_not_called()


def _make_publishing_manager(policy: BlockReusePolicy) -> KVCacheManagerV2:
    """A manager wired just far enough to run the publish/history bookkeeping."""
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager._can_publish_block_reuse = True
    manager.block_reuse_policy = policy
    manager.conversation_manager = None
    manager.kv_cache_map = {}
    return manager


def _prefill(manager: KVCacheManagerV2, prompt: list[int], boundaries: list[int]):
    """Drive one request through *boundaries* and return its cache."""
    request = SimpleNamespace(
        py_request_id=7,
        is_dummy_request=False,
        prompt_len=len(prompt),
        context_current_position=0,
        context_remaining_length=len(prompt),
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
    )
    kv_cache = _FakeKVCache(num_committed_tokens=0)
    manager.kv_cache_map[request.py_request_id] = kv_cache
    manager._reuse_token_source = Mock(return_value=prompt)
    for end in boundaries:
        request.context_current_position = end
        request.context_remaining_length = len(prompt) - end
        manager.update_context_resources(SimpleNamespace(context_requests=[request]))
    return kv_cache


@pytest.mark.parametrize(
    "boundaries",
    [[12], [4, 12], [4, 9, 12], [7, 12], [11, 12]],
    ids=["single", "two", "three", "uneven", "tail_split"],
)
def test_chunking_does_not_change_what_a_prefill_publishes(boundaries) -> None:
    """Reuse must not depend on how a prompt happened to be chunked.

    Keys are built per commit, so a chunk boundary is the one place their
    indexing can drift. If it does, two identical prompts publish different
    keys depending on chunking, and a later request either misses a prefix it
    should hit or matches blocks built from different tokens.
    """
    prompt = list(range(12))

    # ALL_REUSABLE publishes at every boundary, exercising the incremental
    # (start > 0) half; deferred policies publish once and chunking cannot reach them.
    policy = BlockReusePolicy.ALL_REUSABLE
    chunked = _prefill(_make_publishing_manager(policy), prompt, boundaries)
    unchunked = _prefill(_make_publishing_manager(policy), prompt, [len(prompt)])

    assert chunked.published_keys == unchunked.published_keys


def test_context_publishes_the_whole_prompt_at_history_length() -> None:
    """Every computed prompt position is published under its raw-prompt key. The
    draft pool's tail is protected by the claim-time backoff, not by withholding.
    """
    prompt = list(range(12))

    manager = _make_publishing_manager(BlockReusePolicy.PER_REQUEST)
    kv_cache = _prefill(manager, prompt, [4, len(prompt)])

    assert kv_cache.history_length == len(prompt)
    assert kv_cache.num_committed_tokens == len(prompt)
    assert kv_cache.stopped_committing


def test_draft_pool_commits_every_chunk_it_computes() -> None:
    prompt = list(range(12))
    request = SimpleNamespace(
        py_request_id=41,
        is_dummy_request=False,
        prompt_len=len(prompt),
        context_current_position=4,
        context_remaining_length=8,
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
    )
    kv_cache = _FakeKVCache(num_committed_tokens=0)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = True
    manager._can_publish_block_reuse = True
    manager._reuse_token_source = Mock(return_value=prompt)
    manager.kv_cache_map = {request.py_request_id: kv_cache}

    manager.try_commit_blocks(request)
    assert kv_cache.num_committed_tokens == 4

    request.context_current_position = len(prompt)
    request.context_remaining_length = 0
    manager.try_commit_blocks(request)
    assert kv_cache.num_committed_tokens == len(prompt)
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
        self.context_current_position = length


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
        get_and_reset_iteration_suspend_resume_stats=lambda: (0, 0),
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
