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
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
    Role,
    _extend_swa_windows_for_reuse,
    _KVCacheManagerInitStatus,
    _sync_kv_cache_manager_init_status,
    _update_kv_cache_draft_token_location,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.bindings.internal.batch_manager import CacheType, LinearCacheType
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
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
    LayerId,
    SsmLayerConfig,
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

    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
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
        patch.object(KVCacheManagerV2, "_log_swa_scratch_summary"),
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
    ("joint_reuse", "is_draft", "expected_windows"),
    [
        # max_attention_window_vec is resolved per local layer, and the fixture
        # owns a single one, so the pattern's leading 5 is the only entry the
        # backoff widens to W+D=6.
        (False, False, [6]),
        (True, False, [6]),
        (True, True, [6]),
    ],
    ids=["single_kvcm", "joint_target", "joint_draft"],
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
    is_draft: bool,
    expected_windows: list[int | None],
) -> None:
    """Keep #18295's shift-by-one input aligned with both reuse protocols."""
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        max_gpu_total_bytes=16 << 20,
        max_attention_window=[5, MAX_SEQ_LEN, MAX_SEQ_LEN - 1],
    )
    manager, _ = _make_manager_for_cache_tier_test(
        kv_cache_config,
        [Mock()],
        spec_config=spec_config,
        is_draft=is_draft,
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
    assert manager.max_attention_window_vec == expected_windows
    assert kv_cache_config.max_attention_window == [
        5,
        MAX_SEQ_LEN,
        MAX_SEQ_LEN - 1,
    ]
    core_config = manager._build_base_config(
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
    )
    assert core_config.layers[0].sliding_window_size == expected_windows[0]
    assert core_config.reuse_match_backoff == 1
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


def test_extend_swa_windows_for_reuse_preserves_non_attention_windows() -> None:
    recurrent_states = LinearCacheType.RECURRENT_STATES.value

    assert _extend_swa_windows_for_reuse(
        [None, 0, recurrent_states, 5, MAX_SEQ_LEN - 1],
        reuse_match_backoff=1,
        max_seq_len=MAX_SEQ_LEN,
    ) == [None, 0, recurrent_states, 6, None]


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
    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"

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
    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"

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


def _attention_layer(layer_id: int, window: int | None) -> AttentionLayerConfig:
    return AttentionLayerConfig(
        layer_id=LayerId(layer_id),
        buffers=[BufferConfig(role=Role.KEY, size=256)],
        sliding_window_size=window,
    )


def _ssm_layer(layer_id: int) -> SsmLayerConfig:
    return SsmLayerConfig(
        layer_id=LayerId(layer_id),
        buffers=[BufferConfig(role="ssm_state", size=64)],
    )


def _run_swa_scratch_summary(
    layers: list[object],
    *,
    slots_without: list[int],
    slots_with: list[int],
    enabled: bool,
) -> tuple[list[str], list[str]]:
    """Drive ``_log_swa_scratch_summary`` and return its warning/debug lines.

    The method is diagnostics-only, but it runs unconditionally at the end of
    ``__init__``, so anything it cannot read aborts engine construction. Calling
    it unbound keeps the check on the layer walk and the saving arithmetic
    themselves, with no GPU and no real manager needed.
    """
    manager = Mock()
    manager.impl.get_layer_group_id.side_effect = lambda _: 0
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=layers,
        constraints=[BatchDesc([KVCacheDesc(capacity=TOKENS_PER_BLOCK, history_length=0)])],
    )
    manager.enable_swa_scratch_reuse = enabled
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.max_num_tokens = TOKENS_PER_BLOCK

    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
    with (
        patch(f"{module}._introspection") as introspection,
        patch(f"{module}.logger") as logger,
    ):
        introspection.swa_life_cycle_ids.return_value = [0]
        introspection.pool_group_index.return_value = 0
        introspection.compute_slots_for_batch.side_effect = [slots_without, slots_with]
        KVCacheManagerV2._log_swa_scratch_summary(manager)

    return (
        [str(call.args[0]) for call in logger.warning.call_args_list],
        [str(call.args[0]) for call in logger.debug.call_args_list],
    )


def test_swa_scratch_summary_skips_ssm_layers() -> None:
    """A hybrid SSM + SWA engine must survive the startup summary.

    ``SsmLayerConfig`` carries no ``sliding_window_size``. Reading it unguarded
    raised ``AttributeError`` out of ``__init__`` for any Mamba-hybrid model
    that also had a real attention window.
    """
    warnings, _ = _run_swa_scratch_summary(
        [_attention_layer(0, window=TOKENS_PER_BLOCK), _ssm_layer(1)],
        slots_without=[8],
        slots_with=[4],
        enabled=True,
    )

    assert warnings == []


def test_swa_scratch_summary_warns_when_a_real_saving_is_declined() -> None:
    warnings, _ = _run_swa_scratch_summary(
        [_attention_layer(0, window=TOKENS_PER_BLOCK)],
        slots_without=[8],
        slots_with=[4],
        enabled=False,
    )

    assert any("would cut windowed slots by up to 50%" in message for message in warnings)


def test_swa_scratch_summary_treats_a_slot_increase_as_no_saving() -> None:
    """A rise in slot count is not a saving and must not be reported as one.

    Counting it left ``best_saving_pct`` at 0, so the declined-saving warning
    advertised "up to 0%" while the inert-configuration branch never fired.
    """
    warnings, debugs = _run_swa_scratch_summary(
        [_attention_layer(0, window=TOKENS_PER_BLOCK)],
        slots_without=[4],
        slots_with=[8],
        enabled=False,
    )

    assert not any("would cut windowed slots" in message for message in warnings)
    assert any("scratch reuse is inert" in message for message in debugs)


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
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2."
        "_update_kv_cache_draft_token_location"
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
            "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.logger.warning"
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
            "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.logger.warning"
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
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

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
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
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


class TestSwaScratchSharedDecomposition:
    """The identity the FlashInfer per-layer derivation is built on.

    ``get_batch_cache_indices_flat_shared`` returns a table that is the same for
    every layer of a ``(pool, scale)`` group, and the backend reconstructs a
    layer's page indices on device with::

        index_L = base + valid * (((rot + layer_offset_L) % scale) // kv_factor)

    If that identity is off by anything, attention reads another layer's pages
    and produces plausible-but-wrong output rather than failing. These cases pin
    it against the per-layer builder that CI has been running all along, so a
    divergence is a test failure rather than an accuracy report.
    """

    NUM_BLOCKS = 41
    SLOT_IDS = tuple(range(7, 7 + 8))

    @staticmethod
    def _derive(base, rot, valid, layer_offset, div_factor):
        """The device-side per-layer step, restated in numpy."""
        import numpy as np

        sub = ((rot + layer_offset) % GEMMA4_SCALE) // div_factor
        return (np.asarray(base, dtype=np.int64) + np.asarray(valid, dtype=np.int64) * sub).tolist()

    def _shared_parts(self, count=None, div_factor=1):
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
            compute_scratch_shared_parts,
        )

        return compute_scratch_shared_parts(
            0,
            self.NUM_BLOCKS if count is None else count,
            GEMMA4_SCRATCH_PAGES_PER_BLOCK,
            GEMMA4_SCALE,
            self.SLOT_IDS,
            div_factor,
        )

    @pytest.mark.parametrize("div_factor", [1, GEMMA4_KV_FACTOR])
    def test_reconstructs_every_swa_layer(self, div_factor):
        """Reconstruction must match the device kernel formula for all 40 layers."""
        import numpy as np

        base, rot = self._shared_parts(div_factor=div_factor)
        valid = np.ones_like(base)
        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            layer_offset = _k_layer_offset(layer_idx)
            got = self._derive(base, rot, valid, layer_offset, div_factor)
            expected = [
                _reference_flat_index(
                    pos,
                    GEMMA4_SCRATCH_PAGES_PER_BLOCK,
                    GEMMA4_SCALE,
                    layer_offset,
                    self.SLOT_IDS,
                    div_factor,
                )
                for pos in range(self.NUM_BLOCKS)
            ]
            assert got == expected, f"layer {layer_idx} is not reconstructible"

    def test_shared_parts_do_not_depend_on_the_layer(self):
        """The whole point: one table serves the group.

        A regression that folded any layer term into ``base``/``rot`` would
        still reconstruct layer 0 correctly and quietly break the other 39.
        """
        first = self._shared_parts()
        for _ in range(3):
            other = self._shared_parts()
            assert other[0].tolist() == first[0].tolist()
            assert other[1].tolist() == first[1].tolist()

    def test_rotation_stays_inside_the_slot_for_every_layer(self):
        """``rot`` is a residue, so adding any layer offset cannot leave the slot."""
        base, rot = self._shared_parts()
        assert all(0 <= int(r) < GEMMA4_SCALE for r in rot)
        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            derived = self._derive(base, rot, [1] * len(base), _k_layer_offset(layer_idx), 1)
            for pos, index in enumerate(derived):
                slot = int(self.SLOT_IDS[(pos * GEMMA4_SCRATCH_PAGES_PER_BLOCK) // GEMMA4_SCALE])
                assert slot * GEMMA4_SCALE <= index < (slot + 1) * GEMMA4_SCALE

    def test_matches_the_per_layer_builder_on_a_mixed_segment(self):
        """Full-segment equivalence, scratch and non-scratch blocks together.

        This is the case the backend actually hits: a request whose block list
        straddles the scratch range, where the non-scratch blocks must gain
        exactly ``layer_offset // kv_factor`` and the scratch ones the rotation.
        """
        import numpy as np

        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
            BAD_PAGE_INDEX,
            apply_scratch_to_block_segment,
            apply_scratch_to_shared_segment,
        )

        div_factor = GEMMA4_KV_FACTOR
        # Blocks 0-1 committed, 2-5 scratch, 6 evicted out-of-window, 7 committed.
        raw = np.asarray(
            [
                10,
                11,
                BAD_PAGE_INDEX,
                BAD_PAGE_INDEX,
                BAD_PAGE_INDEX,
                BAD_PAGE_INDEX,
                BAD_PAGE_INDEX,
                12,
            ],
            dtype=np.int32,
        )
        beg, end = 2, 6

        def scaled(values):
            out = values.copy()
            addressable = out != BAD_PAGE_INDEX
            np.copyto(out, out * GEMMA4_SCALE // div_factor, where=addressable)
            return out

        shared_base = scaled(raw)
        shared_rot = np.zeros_like(shared_base)
        lo, hi = apply_scratch_to_shared_segment(
            shared_base,
            shared_rot,
            beg,
            end,
            GEMMA4_SCRATCH_PAGES_PER_BLOCK,
            GEMMA4_SCALE,
            self.SLOT_IDS,
            div_factor,
        )
        valid = (scaled(raw) != BAD_PAGE_INDEX).astype(np.int32)
        valid[lo:hi] = 1

        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            layer_offset = _k_layer_offset(layer_idx)
            reference = scaled(raw)
            apply_scratch_to_block_segment(
                reference,
                beg,
                end,
                GEMMA4_SCRATCH_PAGES_PER_BLOCK,
                GEMMA4_SCALE,
                layer_offset,
                self.SLOT_IDS,
                div_factor,
            )
            got = self._derive(shared_base, shared_rot, valid, layer_offset, div_factor)
            assert got == reference.tolist(), f"layer {layer_idx} diverges"
            # The sentinel must survive the derivation on the evicted block.
            assert got[6] == BAD_PAGE_INDEX

    def test_bad_page_index_survives_every_layer(self):
        """``valid`` is the only thing standing between a sentinel and a real page."""
        import numpy as np

        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BAD_PAGE_INDEX

        base = np.asarray([BAD_PAGE_INDEX, 80, BAD_PAGE_INDEX], dtype=np.int32)
        rot = np.zeros_like(base)
        valid = np.asarray([0, 1, 0], dtype=np.int32)
        for layer_idx in range(GEMMA4_NUM_SWA_LAYERS):
            got = self._derive(base, rot, valid, _k_layer_offset(layer_idx), 1)
            assert got[0] == BAD_PAGE_INDEX
            assert got[2] == BAD_PAGE_INDEX
            assert got[1] == 80 + _k_layer_offset(layer_idx)

    def test_empty_range_is_empty(self):
        base, rot = self._shared_parts(count=0)
        assert base.tolist() == [] and rot.tolist() == []


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

        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
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
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BAD_PAGE_INDEX

        got = self._apply([BAD_PAGE_INDEX, 11, BAD_PAGE_INDEX], beg=5, end=5)
        assert got[0] == BAD_PAGE_INDEX and got[2] == BAD_PAGE_INDEX
        assert got[1] == 11 + self.LAYER_OFFSET

    def test_partial_overlap_splits_scratch_and_non_scratch(self):
        """Only blocks inside the range rotate; the rest are shifted."""
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
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
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
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
