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

import array
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._torch.pyexecutor._util import _get_num_storage_pool_groups
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    HostMemoryLimits,
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
    _extend_swa_windows_for_reuse,
    _get_generation_kv_capacity,
    _get_kv_alloc_extra_tokens,
    _KVCacheManagerInitStatus,
    _raise_on_every_rank,
    _swa_branch_snapshot_point,
    _swa_endpoint_priority,
    _sync_kv_cache_manager_init_status,
    _update_kv_cache_draft_token_location,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.regions import RegionLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import (
    SPARSE_ARENA_ALIGNMENT,
    HostHistoryReservations,
    sparse_arena_bytes,
    sparse_arena_shapes,
    sparse_batch_shape,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.bindings.executor import KvCacheRetentionConfig
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.bindings.internal.batch_manager import CacheType, LinearCacheType
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import (
    BlockReuseConfig,
    CapacitySchedulerPolicy,
    DecodingBaseConfig,
    DFlashDecodingConfig,
    DraftTargetDecodingConfig,
    DSparkDecodingConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    PARDDecodingConfig,
    SparseKvCacheOffloadConfig,
)
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    CACHE_LEVEL1,
    DEFAULT_BEAM_INDEX,
    GPU_LEVEL,
    AttentionLayerConfig,
    AttnLifeCycle,
    Batch,
    BatchDesc,
    BufferConfig,
    BufferId,
    CacheLevel,
    CacheTier,
    CudaStream,
    DataRole,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    PageIndexMode,
    ReusePolicy,
    ReuseScope,
    SparseCacheConfig,
    SsmLayerConfig,
    _introspection,
)

TOKENS_PER_BLOCK = 4
MAX_SEQ_LEN = 16


@pytest.mark.cpu_only
def test_nvfp4_layer_spec_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="NVFP4 head_dim must be even"):
        KVCacheLayerSpec(
            global_layer_id=0,
            owner=KVCacheLayerOwner.TARGET,
            dtype=DataType.NVFP4,
            num_kv_heads=16,
            head_dim=1,
            buffer_roles=(Role.KEY, Role.KEY_BLOCK_SCALE),
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "prompt,rewind,ordinal,window,sinks,expected",
    [
        (8192, 1024, 95, 4096, 0, 0),
        (8192, 1024, 96, 4096, 0, 70),
        (8192, 1024, 255, 4096, 0, 70),
        (8192, 1024, 256, 4096, 0, 0),
        (8192, 1024, 0, 4096, 1, 70),
        (8192, 4096, 0, 4096, 0, 70),
        (8193, 1024, 96, 4096, 0, 70),
        (8193, 1024, 256, 4096, 0, 70),
        (32, 1024, 0, 4096, 0, 70),
        (8192, 1024, 0, None, 0, 35),
    ],
)
def test_swa_endpoint_priority(
    prompt: int, rewind: int, ordinal: int, window: int | None, sinks: int, expected: int
) -> None:
    priority = _swa_endpoint_priority(prompt, 32, rewind)
    assert priority(ordinal, AttnLifeCycle(window, sinks)) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rewind,is_dummy,is_draft,prompt,enabled",
    [
        (0, False, False, 8192, False),
        (1024, False, False, 8192, True),
        (4096, False, False, 8192, True),
        (1024, True, False, 8192, False),
        (1024, False, True, 8192, False),
        (1024, False, False, None, False),
    ],
)
def test_create_kv_cache_swa_endpoint_priority(
    rewind: int, is_dummy: bool, is_draft: bool, prompt: int | None, enabled: bool
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._swa_endpoint_rewind = rewind
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager.enable_block_reuse = True
    manager.is_draft = is_draft
    manager.tokens_per_block = 32
    manager.kv_cache_map = {}
    manager.index_mapper = Mock()
    manager.index_mapper.num_free_slots.return_value = 1
    manager.max_beam_width = 1
    manager.num_pools = 0
    manager.impl = Mock()
    manager.impl.create_kv_cache.return_value.beam_width = 1
    # Belongs to no Batch, so it still owns its page index buffers.
    manager.impl.create_kv_cache.return_value.batch_row = None

    manager._create_kv_cache(0, None, None, is_dummy=is_dummy, expected_prompt_length=prompt)

    kwargs = manager.impl.create_kv_cache.call_args.kwargs
    assert ("custom_priority_callback" in kwargs) is enabled
    if enabled:
        assert kwargs["custom_priority_callback"](255, AttnLifeCycle(4096, 0)) == 70


def _commit_released_prompt(core: KVCacheManager, prompt: list[int], priority_callback) -> None:
    """Run a prompt through a real cache and release it, leaving its pages reusable."""
    cache = core.create_kv_cache(None, prompt, custom_priority_callback=priority_callback)
    try:
        assert cache.resume(CudaStream(torch.cuda.current_stream().cuda_stream))
        # Grow capacity only: commit() advances the history itself. Advancing
        # the history first (resize(len, len)) marks out-of-window pages stale
        # before they are committed, so they would never become reusable.
        assert cache.resize(len(prompt))
        uncommitted = prompt[cache.num_committed_tokens :]
        if uncommitted:
            cache.commit(uncommitted)
    finally:
        cache.close()
    torch.cuda.synchronize()


@pytest.mark.parametrize("with_callback", [True, False], ids=["endpoint-priority", "lru-baseline"])
def test_swa_endpoint_priority_steers_real_eviction(with_callback: bool) -> None:
    """Priority-based eviction of real pages must follow the endpoint callback.

    A four-block SWA prompt is committed and released: with window 8 and rewind
    4 its block 0 lies before the protected endpoint span (priority 0) while
    blocks 1-3 overlap it (priority 70). A second request then reuses block 0,
    re-releasing it as the most recently used page. Bounded pressure (exactly
    one page beyond capacity) must evict:

    * with the callback: block 0, the lowest-priority page, despite being MRU;
    * without it (LRU baseline): block 1, the oldest page, an endpoint page.

    The baseline leg proves recency alone sacrifices the endpoint page here, so
    the endpoint-priority leg fails if the core ignores the callback or applies
    its priorities incorrectly.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    # The sibling tests reach the core through the pyexecutor manager, whose
    # constructor touches the torch CUDA runtime and thereby binds the primary
    # context to this thread. This test constructs the raw core first, so bind
    # the context explicitly or construction fails with CUDA error 201.
    torch.cuda.synchronize()
    window_size = 2 * TOKENS_PER_BLOCK
    rewind_tokens = TOKENS_PER_BLOCK
    prompt = list(range(4 * TOKENS_PER_BLOCK))
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=2 << 20)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=65536),
                        BufferConfig(role=Role.VALUE, size=65536),
                    ],
                    sliding_window_size=window_size,
                )
            ],
        )
    )

    def priority_callback(prompt_length: int):
        if not with_callback:
            return None
        return _swa_endpoint_priority(prompt_length, TOKENS_PER_BLOCK, rewind_tokens)

    pins = []
    try:
        _commit_released_prompt(core, prompt, priority_callback(len(prompt)))

        # Reuse block 0 so its page is re-released after blocks 1-3. The touch
        # request's one new page lies inside its own endpoint span, so it stays
        # priority 70 under the callback and never absorbs the pressure.
        touch_prompt = prompt[:TOKENS_PER_BLOCK] + [1000 + i for i in range(TOKENS_PER_BLOCK)]
        _commit_released_prompt(core, touch_prompt, priority_callback(len(touch_prompt)))

        swa_lc_id = _introspection.swa_life_cycle_ids(core)[0]
        _, pages = _introspection.reuse_match_pages(core, ReuseScope(), prompt, swa_lc_id)
        assert len(pages) == 4
        # The all-reusable premise: out-of-window pages stay reusable until evicted.
        assert all(page is not None for page in pages)

        # Bounded pressure: pin every free page with held (unevictable)
        # allocations, then demand one more page, forcing exactly one eviction.
        stream = CudaStream(torch.cuda.current_stream().cuda_stream)
        free_pages = core.get_storage_statistics()[0].free
        for _ in range(free_pages + 1):
            pin = core.create_kv_cache()
            pins.append(pin)
            assert pin.resume(stream)
            assert pin.resize(TOKENS_PER_BLOCK)

        _, pages = _introspection.reuse_match_pages(core, ReuseScope(), prompt, swa_lc_id)
        if with_callback:
            assert pages[0] is None, "the intermediate page must be evicted first"
            assert all(page is not None for page in pages[1:]), "endpoint pages must be retained"
        else:
            assert pages[0] is not None, "pure LRU keeps the recently touched page"
            assert pages[1] is None, "pure LRU evicts the oldest endpoint page"
            assert all(page is not None for page in pages[2:])
    finally:
        for pin in pins:
            pin.close()
        core.shutdown()


class _CacheTierInitError(Exception):
    pass


@dataclass
class _FakeManagerConfig:
    cache_tiers: list[object]
    layers: list[object] = field(
        default_factory=lambda: [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=128)],
            )
        ]
    )


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int) -> None:
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens: list[int] | None = None
        self.published_keys: list[object] = []
        self.history_length = 0
        self.is_active = True
        self.enable_swa_scratch_reuse = True
        self.stopped_committing = False
        self.commit_is_end = False

    def commit(self, tokens: list[int], is_end: bool = False) -> None:
        self.commit_is_end = is_end
        if is_end:
            self.stopped_committing = True
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
    dtype: DataType = DataType.HALF,
    kv_cache_type: CacheType = CacheType.SELFKONLY,
    sparse_kv_offload: SparseKvCacheOffloadConfig | None = None,
) -> KVCacheManagerConfig:
    if max_attention_window_vec is None:
        max_attention_window_vec = [None]
    if pp_layers is None:
        pp_layers = list(range(len(max_attention_window_vec)))
    assert len(max_attention_window_vec) == len(pp_layers)

    cache_manager = object.__new__(KVCacheManagerV2)
    cache_manager.layer_specs = None
    cache_manager._sparse_kv_offload = sparse_kv_offload
    cache_manager.max_beam_width = 1
    cache_manager.kv_cache_type = kv_cache_type
    cache_manager.dtype = dtype
    cache_manager.head_dim_per_layer = [128] * len(pp_layers)
    cache_manager.enable_swa_scratch_reuse = False
    cache_manager.enable_partial_reuse = kv_cache_v2_module._effective_partial_reuse(
        kv_cache_config.enable_partial_reuse,
        cache_manager._supports_partial_reuse,
        type(cache_manager).__name__,
    )
    cache_manager.num_extra_kv_tokens = num_extra_kv_tokens
    cache_manager.enable_stats = False
    cache_manager.block_reuse_policy = BlockReusePolicy(kv_cache_config.block_reuse_config.policy)
    cache_manager.is_draft = is_draft
    cache_manager.num_local_layers = len(pp_layers)
    cache_manager.pp_layers = pp_layers
    cache_manager.max_attention_window_vec = max_attention_window_vec
    cache_manager.max_seq_len = max_seq_len
    cache_manager.max_batch_size = max_batch_size
    cache_manager.max_cuda_graph_batch_size = None
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
    vocab_size: int | None = 16,
    dtype: DataType = DataType.HALF,
    layer_specs: list[KVCacheLayerSpec] | None = None,
    available_tokens: int = MAX_SEQ_LEN,
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
        fake_impl.buffer_grouping = [[BufferId(LayerId(0), Role.KEY)]]
        fake_impl.pool_group_descs = []
        fake_impl.get_layer_group_id.side_effect = lambda _: 0

    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
    with (
        patch(f"{module}.IndexMapper"),
        patch(f"{module}.KVCacheManagerPy", impl_constructor),
        patch.object(KVCacheManagerV2, "_build_base_config", build_base_config),
        patch.object(KVCacheManagerV2, "_build_cache_config", build_cache_config),
        patch.object(KVCacheManagerV2, "get_num_available_tokens", return_value=available_tokens),
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
            dtype=dtype,
            layer_specs=layer_specs,
            spec_config=spec_config,
            is_draft=is_draft,
            is_disagg=is_disagg,
            joint_kv_cache_reuse=joint_reuse,
            vocab_size=vocab_size,
            execution_stream=Mock(),
            cold_page_codec_provider=cold_page_codec_provider,
        )
    return manager, impl_constructor


@pytest.mark.cpu_only
@pytest.mark.parametrize("available_tokens", [TOKENS_PER_BLOCK, MAX_SEQ_LEN])
def test_page_index_capacity_covers_configured_sequence_length(available_tokens: int) -> None:
    manager, _ = _make_manager_for_cache_tier_test(
        KvCacheConfig(max_gpu_total_bytes=16 << 20, host_cache_size=0),
        [Mock()],
        available_tokens=available_tokens,
    )
    try:
        assert manager.max_seq_len == available_tokens
        required_blocks = (
            MAX_SEQ_LEN + manager._generation_kv_capacity_headroom + TOKENS_PER_BLOCK - 1
        ) // TOKENS_PER_BLOCK
        assert manager.max_blocks_per_seq >= required_blocks
        assert manager.max_blocks_per_seq % 4 == 0
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "enable_attention_dp,rank",
    [
        pytest.param(False, 0, id="rank-zero"),
        pytest.param(True, 1, id="attention-dp"),
    ],
)
@pytest.mark.parametrize("vocab_size", [131072, None], ids=["multimodal", "text-only"])
def test_buffered_events_receive_vocab_size(
    enable_attention_dp: bool, rank: int, vocab_size: int | None
) -> None:
    mapping = Mapping(world_size=2, rank=rank, tp_size=2, enable_attention_dp=enable_attention_dp)
    distributed = Mock()
    distributed.allreduce.side_effect = lambda value, op: value
    kv_cache_config = KvCacheConfig(
        max_gpu_total_bytes=16 << 20,
        host_cache_size=0,
        event_buffer_max_size=8,
    )

    # Exercise the real resource-manager constructor and selected event backend.
    with (
        patch.object(kv_cache_v2_module, "mpi_rank", return_value=rank),
        patch.object(Distributed, "get", return_value=distributed),
        patch.object(
            kv_cache_v2_module,
            "KVCacheEventManager",
            side_effect=kv_cache_v2_module.KVCacheEventManager,
        ) as event_constructor,
    ):
        manager, impl_constructor = _make_manager_for_cache_tier_test(
            kv_cache_config,
            [Mock()],
            mapping=mapping,
            vocab_size=vocab_size,
        )

    event_constructor.assert_called_once()
    assert event_constructor.call_args.args == (kv_cache_config.event_buffer_max_size,)
    assert event_constructor.call_args.kwargs["mm_token_id_offset"] == vocab_size
    if enable_attention_dp:
        assert event_constructor.call_args.kwargs["attention_dp_rank"] == rank
        assert event_constructor.call_args.kwargs["attention_dp_gather"] is distributed.allgather
    else:
        assert "attention_dp_rank" not in event_constructor.call_args.kwargs
    assert impl_constructor.call_args.kwargs["event_manager"] is manager.event_manager


def _multi_rank_host_failure_consensus_worker() -> tuple[int, int, bool, int]:
    """Exercise the real world collective from an MPI worker.

    Only rank 1 fails to build the host tier. Every rank must still raise: a
    rank that returned a working manager while another aborted would leave the
    job half-initialized and hang on the next collective.
    """
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    rank = mpi_rank()
    world_size = mpi_world_size()
    local_impl = Mock()
    impl_side_effect: list[object] = (
        [local_impl] if rank == 0 else [_CacheTierInitError("rank-local host tier failure")]
    )

    raised = False
    try:
        _make_manager_for_cache_tier_test(
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
    except Exception:
        raised = True

    # The rank that succeeded must also release what it built.
    return (rank, world_size, raised, local_impl.shutdown.call_count)


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


@pytest.mark.parametrize("kv_cache_type", [CacheType.SELF, CacheType.SELFKONLY])
@pytest.mark.parametrize(
    "kv_cache_dtype,dtype,has_block_scales",
    [
        pytest.param("auto", DataType.NVFP4, True, id="checkpoint-nvfp4"),
        pytest.param("nvfp4", DataType.NVFP4, True, id="explicit-nvfp4"),
        pytest.param("auto", DataType.HALF, False, id="unquantized"),
        pytest.param("nvfp4", DataType.FP8, False, id="fp8-fallback"),
    ],
)
def test_base_config_scale_buffers_follow_resolved_dtype(
    kv_cache_dtype: str, dtype: DataType, has_block_scales: bool, kv_cache_type: CacheType
) -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(dtype=kv_cache_dtype), dtype=dtype, kv_cache_type=kv_cache_type
    )

    expected_roles = {Role.KEY}
    if has_block_scales:
        expected_roles.add(Role.KEY_BLOCK_SCALE)
    if kv_cache_type == CacheType.SELF:
        expected_roles.add(Role.VALUE)
        if has_block_scales:
            expected_roles.add(Role.VALUE_BLOCK_SCALE)
    assert {buffer.role for buffer in config.layers[0].buffers} == expected_roles


@pytest.mark.parametrize(
    "num_kv_heads,head_dim",
    [
        pytest.param([9, 0, 2, 0, 3], [128] * 5, id="zero-heads"),
        pytest.param([9, None, 2, None, 3], [128] * 5, id="missing-heads"),
        pytest.param([9, 2, 2, 3, 3], [128, 0, 128, 0, 64], id="zero-head-dim"),
    ],
)
def test_zero_size_layers_are_removed_before_pool_allocation(
    num_kv_heads: list[int | None], head_dim: list[int]
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            max_attention_window=[MAX_SEQ_LEN, 8, MAX_SEQ_LEN, 12, TOKENS_PER_BLOCK],
            pool_ratio=[0.5, 0.5],
        ),
        CacheType.SELF,
        num_layers=4,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layer_mask=[False, True, True, True, True],
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=1,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=16,
    )
    try:
        config = manager.kv_cache_manager_py_config
        assert manager.num_layers == 5
        assert manager.num_local_layers == 2
        assert manager.pp_layers == [2, 4]
        assert manager.layer_offsets == {2: 0, 4: 1}
        assert manager.num_kv_heads_per_layer == [2, 3]
        assert manager.head_dim_per_layer == [head_dim[2], head_dim[4]]
        assert manager.max_attention_window_vec == [None, TOKENS_PER_BLOCK]
        assert [layer.layer_id for layer in config.layers] == [0, 1]
        assert [layer.sliding_window_size for layer in config.layers] == [
            None,
            TOKENS_PER_BLOCK,
        ]
        assert config.initial_pool_ratio == pytest.approx([0.5, 0.5])
        assert all(buffer.size > 0 for layer in config.layers for buffer in layer.buffers)
        assert set(manager.layer_to_pool_mapping_dict) == {0, 1}
        assert manager.num_pools == 2

        cache = manager._create_kv_cache(0, None, None)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(MAX_SEQ_LEN, 0)
        assert cache.capacity == MAX_SEQ_LEN
        for global_layer_id in manager.pp_layers:
            buffer = manager.get_buffers(global_layer_id)
            assert buffer.is_cuda
            assert buffer.shape[0] > 0
            assert buffer.shape[1:] == (
                2,
                TOKENS_PER_BLOCK,
                num_kv_heads[global_layer_id],
                head_dim[global_layer_id],
            )
            buffer[0].fill_(global_layer_id)
            assert torch.all(buffer[0] == global_layer_id)
    finally:
        manager.shutdown()


def _layer_api_snapshot(manager: KVCacheManagerV2) -> dict:
    """A manager's built cache config and layer-based API, with pool addresses as (pool, offset)."""
    impl = manager.impl
    config = manager.kv_cache_manager_py_config
    pools = {
        int(pool.base_address): (int(group.pool_group_index), int(pool.pool_index))
        for group in impl.pool_group_descs
        for pool in group.pools
    }

    def pool_offset(address: int):
        if address == 0:
            return None
        base = max(base for base in pools if base <= address)
        return pools[base], address - base

    return {
        "layers": [
            (
                type(layer).__name__,
                int(layer.layer_id),
                [
                    (str(buffer.role), buffer.size, buffer.coalescing_group, buffer.reuse_policy)
                    for buffer in layer.buffers
                ],
                getattr(layer, "sliding_window_size", None),
                layer.reuse_policy,
            )
            for layer in config.layers
        ],
        "commit_min_snapshot": config.commit_min_snapshot,
        "regions": [region.name for region in config.regions],
        "initial_pool_ratio": config.initial_pool_ratio,
        "pool_groups": [
            (int(group.num_slots), list(group.slot_desc.slot_size_list), len(group.pools))
            for group in impl.pool_group_descs
        ],
        "layer_grouping": [[int(layer_id) for layer_id in group] for group in impl.layer_grouping],
        "layer_group_ids": [
            int(impl.get_layer_group_id(LayerId(layer_id)))
            for layer_id in range(manager.num_local_layers)
        ],
        "layer_to_pool_mapping_dict": dict(manager.layer_to_pool_mapping_dict),
        "pool_pointers": [
            pool_offset(int(address))
            for address in manager.kv_cache_pool_pointers.flatten().tolist()
        ],
        "pool_mapping": manager.kv_cache_pool_mapping.tolist(),
        "index_scales": manager.index_scales.tolist(),
        "block_offsets_shape": tuple(manager.host_kv_cache_block_offsets.shape),
    }


@pytest.mark.parametrize("enable_block_reuse", [False, True])
def test_manager_without_regions_builds_as_before(monkeypatch, enable_block_reuse: bool) -> None:
    """Without regions, the region hooks change neither the built cache nor the layer-based API."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()

    def build() -> KVCacheManagerV2:
        return KVCacheManagerV2(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=0,
                enable_block_reuse=enable_block_reuse,
                max_attention_window=[MAX_SEQ_LEN, 8],
                pool_ratio=[0.5, 0.5],
            ),
            CacheType.SELF,
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.HALF,
            vocab_size=16,
        )

    manager = build()
    try:
        assert manager.region_layout == RegionLayout()
        assert manager.get_region_group_ids("recurrent_state") == ()
        with_hooks = _layer_api_snapshot(manager)
    finally:
        manager.shutdown()
    with monkeypatch.context() as patch:
        patch.setattr(KVCacheManagerV2, "_declare_regions", lambda self: None)
        patch.setattr(KVCacheManagerV2, "_lay_out_regions", lambda self, config: config)
        manager = build()
        try:
            without_hooks = _layer_api_snapshot(manager)
        finally:
            manager.shutdown()

    assert with_hooks == without_hooks
    assert len(with_hooks["layer_grouping"]) == 2


@pytest.mark.parametrize("kv_first", [False, True], ids=["key-first", "kv-first"])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_mixed_role_coalesced_pool_preserves_slot_stride(kv_first: bool, kv_layout: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    key_layer_id, kv_layer_id = 0, 1
    key_spec = KVCacheLayerSpec(
        global_layer_id=key_layer_id,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.HALF,
        num_kv_heads=2,
        head_dim=8,
        buffer_roles=(Role.KEY,),
    )
    kv_spec = replace(key_spec, global_layer_id=kv_layer_id, buffer_roles=(Role.KEY, Role.VALUE))
    layer_specs = [kv_spec, key_spec] if kv_first else [key_spec, kv_spec]
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=1,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        layer_specs=layer_specs,
    )
    try:
        key_local_id = manager.layer_offsets[key_layer_id]
        kv_local_id = manager.layer_offsets[kv_layer_id]
        pool_id = manager.layer_to_pool_mapping_dict[key_local_id]
        assert manager.layer_to_pool_mapping_dict[kv_local_id] == pool_id
        assert manager.num_pools == 1
        assert manager.get_layer_page_index_scale(key_layer_id) == 3
        assert manager.get_layer_page_index_scale(kv_layer_id) == 3

        key_buffer = manager.get_buffers(key_layer_id, kv_layout=kv_layout)
        kv_buffer = manager.get_buffers(kv_layer_id, kv_layout=kv_layout)
        page_stride = manager.impl.get_page_stride(kv_local_id, Role.KEY)
        key_base = manager.impl.get_mem_pool_base_address(
            kv_local_id, Role.KEY, PageIndexMode.SHARED
        )
        value_base = manager.impl.get_mem_pool_base_address(
            kv_local_id, Role.VALUE, PageIndexMode.SHARED
        )
        page_shape = (TOKENS_PER_BLOCK, 2, 8) if kv_layout == "NHD" else (2, TOKENS_PER_BLOCK, 8)
        assert kv_buffer.shape[1:] == (2, *page_shape)
        assert kv_buffer.stride(0) * kv_buffer.element_size() == 3 * page_stride
        # Slot 1 begins three physical pages later, even though this layer
        # exposes only two roles. Reversing the specs also exercises a base
        # pointer offset within the coalesced slot.
        assert kv_buffer[1, 0].data_ptr() == key_base + 3 * page_stride
        assert kv_buffer[1, 1].data_ptr() == value_base + 3 * page_stride
        assert key_buffer.stride(0) * key_buffer.element_size() == page_stride

        request_id = 7
        cache = manager._create_kv_cache(request_id, None, None)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(3 * TOKENS_PER_BLOCK, 0)
        raw_slots = [int(slot) for slot in cache.get_base_page_indices(pool_id)[: cache.num_blocks]]
        assert len(raw_slots) == 3
        assert any(slot % 2 == 1 for slot in raw_slots)
        for layer_id, buffer, expected in (
            (key_layer_id, key_buffer, [3 * slot for slot in raw_slots]),
            (kv_layer_id, kv_buffer, raw_slots),
        ):
            batch_indices = manager.get_batch_cache_indices([request_id], layer_idx=layer_id)
            flat_indices = manager.get_batch_cache_indices_flat(
                [request_id], [len(raw_slots)], layer_idx=layer_id
            ).tolist()
            assert batch_indices == [expected]
            assert flat_indices == expected
            local_id = manager.layer_offsets[layer_id]
            for indices in (batch_indices[0], flat_indices):
                for slot, index in zip(raw_slots, indices):
                    for role_index, role in enumerate(manager.get_layer_buffer_roles(layer_id)):
                        role_base = manager.impl.get_mem_pool_base_address(
                            local_id, role, PageIndexMode.SHARED
                        )
                        assert (
                            buffer[index, role_index].data_ptr()
                            == role_base + 3 * slot * page_stride
                        )

        # Give each role and slot a distinct value. Writing an odd K/V slot
        # must preserve the K-only page and both neighboring K/V slots.
        for ordinal, slot in enumerate(raw_slots):
            key_buffer[3 * slot, 0].fill_(10 + ordinal)
        for ordinal, slot in enumerate(raw_slots):
            kv_buffer[slot, 0].fill_(20 + ordinal)
            kv_buffer[slot, 1].fill_(30 + ordinal)
        for ordinal, slot in enumerate(raw_slots):
            assert torch.all(key_buffer[3 * slot, 0] == 10 + ordinal)
            assert torch.all(kv_buffer[slot, 0] == 20 + ordinal)
            assert torch.all(kv_buffer[slot, 1] == 30 + ordinal)
    finally:
        manager.shutdown()


def test_dflash_block_tables_with_fp8_target_and_bf16_draft() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    torch.cuda.init()
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=2,
        head_dim=8,
        buffer_roles=(Role.KEY, Role.VALUE),
    )
    # Equal byte sizes coalesce the FP8 target and BF16 draft pages, placing
    # the draft's K buffer at a nonzero offset within each physical slot.
    draft = replace(
        target,
        global_layer_id=1,
        owner=KVCacheLayerOwner.DRAFT,
        dtype=DataType.BF16,
        num_kv_heads=1,
    )
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=target.num_kv_heads,
        head_dim=target.head_dim,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_specs=[target, draft],
    )
    try:
        assert manager._use_per_layer_page_tables
        request_ids = [7, 9]
        for request_id, num_blocks in zip(request_ids, (2, 3)):
            cache = manager._create_kv_cache(request_id, None, None)
            assert cache.resume(manager._stream.cuda_stream)
            assert cache.resize(num_blocks * TOKENS_PER_BLOCK, 0)

        worker = object.__new__(DFlashWorker)
        worker._ctx_unified_cache = True
        pool = worker._managed_ctx_pool(
            manager, 1, draft.num_kv_heads, draft.head_dim, torch.bfloat16, 2
        )
        assert pool is not None
        assert worker._init_ctx_block_tables(
            manager, pool, len(request_ids), 1, draft.num_kv_heads, draft.head_dim, TOKENS_PER_BLOCK
        )
        assert worker._ctx_block_offset > 0

        block_offsets = torch.empty(
            (manager.num_attention_op_pools, len(request_ids), 2, manager.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        manager.copy_batch_block_offsets(
            block_offsets, request_ids, beam_width=1, num_contexts=0, num_seqs=len(request_ids)
        )
        metadata = SimpleNamespace(kv_cache_block_offsets=block_offsets)
        assert worker._refresh_ctx_block_tables(metadata, len(request_ids))

        expected = manager.get_batch_cache_indices(request_ids, layer_idx=draft.global_layer_id)
        for row, indices in enumerate(expected):
            assert len(indices) == (2, 3)[row]
            assert worker._ctx_block_tables[row, : len(indices)].tolist() == indices
    finally:
        manager.shutdown()


def test_nvfp4_scale_indices_with_mixed_bf16_storage() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    nvfp4 = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.NVFP4,
        num_kv_heads=4,
        head_dim=128,
        buffer_roles=(Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE),
    )
    bf16 = replace(
        nvfp4,
        global_layer_id=1,
        owner=KVCacheLayerOwner.DRAFT,
        dtype=DataType.BF16,
        num_kv_heads=1,
        buffer_roles=(Role.KEY, Role.VALUE),
    )
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=1,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        layer_specs=[nvfp4, bf16],
    )
    try:
        assert manager.num_pools == 1
        (group,) = manager.impl.pool_group_descs
        assert len(group.pools) == 3
        assert group.num_slots > 1
        slots = [1, int(group.num_slots) - 1]
        for offset, (data_role, scale_role) in enumerate(
            ((Role.KEY, Role.KEY_BLOCK_SCALE), (Role.VALUE, Role.VALUE_BLOCK_SCALE))
        ):
            data_converter = manager.impl.get_page_index_converter(0, data_role)
            scale_converter = manager.impl.get_page_index_converter(0, scale_role)
            pages = data_converter(slots, PageIndexMode.PER_LAYER)
            assert pages == [2 * slot + offset for slot in slots]
            assert scale_converter(slots, PageIndexMode.PER_LAYER) == pages

            scale_base = manager.impl.get_mem_pool_base_address(
                0, scale_role, PageIndexMode.PER_LAYER
            )
            scale_pool = next(pool for pool in group.pools if pool.base_address == scale_base)
            page_bytes = manager.impl.get_page_stride(0, scale_role)
            # Attention reuses data indices for scales, including at the final slot.
            for slot, page in zip(slots, pages):
                address = scale_base + page * page_bytes
                assert address == scale_base + slot * scale_pool.slot_bytes + offset * page_bytes
                assert address + page_bytes <= scale_base + group.num_slots * scale_pool.slot_bytes
    finally:
        manager.shutdown()


def test_mixed_nvfp4_bf16_pool_pointer_views() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    bf16 = torch.empty((2, 1, 32, 128), dtype=torch.bfloat16, device="cuda")
    nvfp4 = torch.empty((2, 1, 32, 64), dtype=torch.uint8, device="cuda")
    scales = torch.empty((2, 1, 32, 8), dtype=torch.float8_e4m3fn, device="cuda")
    pointers = torch.tensor(
        [
            [[nvfp4.data_ptr(), scales.data_ptr()], [0, 0]],
            # A nonzero scale pointer must be ignored for the BF16 layer.
            [[bf16.data_ptr(), scales.data_ptr()], [0, 0]],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    mapping = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device="cpu")
    block_offsets = torch.zeros((2, 1, 2, 1), dtype=torch.int32, device="cuda")
    for table, layer_id, quant_mode, expected_data, expected_scales in (
        (pointers, 1, 0, bf16, None),
        (pointers, 0, int(QuantMode.NVFP4_KV_CACHE), nvfp4, scales),
        (pointers[:, :, 0].contiguous(), 1, 0, bf16, None),
    ):
        data, blocks, block_scales = thop.build_trtllm_gen_kv_cache_metadata(
            host_kv_cache_pool_pointers=table,
            host_kv_cache_pool_mapping=mapping,
            kv_cache_block_offsets=block_offsets,
            layer_idx=layer_id,
            num_kv_heads=1,
            tokens_per_block=32,
            head_dim=128,
            kv_factor=2,
            total_num_blocks=2,
            kv_cache_quant_mode=quant_mode,
            batch_start=0,
            batch_size=1,
            dtype=torch.bfloat16,
        )
        assert data.data_ptr() == expected_data.data_ptr()
        assert data.shape == expected_data.shape
        assert data.dtype == expected_data.dtype
        assert blocks.data_ptr() == block_offsets[layer_id].data_ptr()
        if expected_scales is None:
            assert block_scales is None
        else:
            assert block_scales is not None
            assert block_scales.data_ptr() == expected_scales.data_ptr()
            assert block_scales.shape == expected_scales.shape


def test_zero_size_filter_preserves_ssm_and_nonempty_extra_buffers() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.pp_layers = [3, 4, 5]
    manager.num_layers = 6
    manager.num_local_layers = 3
    manager.layer_offsets = {3: 0, 4: 1, 5: 2}
    manager.num_kv_heads_per_layer = [0, 0, 2]
    manager.head_dim_per_layer = [128, 128, 64]
    manager.max_attention_window_vec = [64, None, 128]

    config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=0)],
                sliding_window_size=64,
            ),
            SsmLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(role=DataRole("ssm_state"), size=256)],
            ),
            AttentionLayerConfig(
                layer_id=LayerId(2),
                buffers=[
                    BufferConfig(role=Role.KEY, size=1024),
                    BufferConfig(role=Role.VALUE, size=1024),
                    BufferConfig(role=Role.INDEX_KEY, size=0),
                    BufferConfig(role=DataRole("extra_state"), size=32),
                ],
                sliding_window_size=128,
                num_sink_tokens=4,
            ),
        ],
        initial_pool_ratio=[0.25, 0.75],
        commit_min_snapshot=True,
    )

    filtered = manager._remove_zero_size_buffers(config)

    assert manager.num_layers == 6
    assert manager.pp_layers == [4, 5]
    assert manager.num_local_layers == 2
    assert manager.layer_offsets == {4: 0, 5: 1}
    assert manager.num_kv_heads_per_layer == [0, 2]
    assert manager.head_dim_per_layer == [128, 64]
    assert manager.max_attention_window_vec == [None, 128]
    assert [layer.layer_id for layer in filtered.layers] == [0, 1]
    ssm_layer, attention_layer = filtered.layers
    assert isinstance(ssm_layer, SsmLayerConfig)
    assert [(buffer.role, buffer.size) for buffer in ssm_layer.buffers] == [
        (DataRole("ssm_state"), 256)
    ]
    assert isinstance(attention_layer, AttentionLayerConfig)
    assert attention_layer.sliding_window_size == 128
    assert attention_layer.num_sink_tokens == 4
    assert [(buffer.role, buffer.size) for buffer in attention_layer.buffers] == [
        (Role.KEY, 1024),
        (Role.VALUE, 1024),
        (DataRole("extra_state"), 32),
    ]
    assert filtered.initial_pool_ratio == pytest.approx([0.25, 0.75])


def test_zero_size_filter_keeps_a_sparse_fetch_path() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.pp_layers = [0, 1]
    manager.num_layers = 2
    manager.num_local_layers = 2
    manager.layer_offsets = {0: 0, 1: 1}
    manager.num_kv_heads_per_layer = [1, 1]
    manager.head_dim_per_layer = [64, 64]
    manager.max_attention_window_vec = [None, None]
    cache = SparseCacheConfig(
        num_selected_blocks=1, max_cached_units=2, unit_tokens=TOKENS_PER_BLOCK
    )
    config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[
                    BufferConfig(role=Role.KEY, size=1024, is_sparse=True),
                    BufferConfig(role=Role.VALUE, size=256),
                    BufferConfig(role=Role.INDEX_KEY, size=0),
                ],
                sparse_cache=cache,
            ),
            # Its only sparse buffer is empty, so nothing is left to fetch.
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[
                    BufferConfig(role=Role.KEY, size=0, is_sparse=True),
                    BufferConfig(role=Role.VALUE, size=256),
                ],
                sparse_cache=cache,
            ),
        ],
    )

    kept, emptied = manager._remove_zero_size_buffers(config).layers

    assert kept.sparse_cache is not None
    assert kept.sparse_cache.max_cached_units == 2
    assert emptied.sparse_cache is None


def test_zero_size_filter_rejects_empty_local_cache() -> None:
    manager = object.__new__(KVCacheManagerV2)
    config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=0)],
            )
        ],
    )

    with pytest.raises(ValueError, match="at least one non-empty local cache layer"):
        manager._remove_zero_size_buffers(config)


def test_zero_size_filter_preserves_layer_and_buffer_reuse_policies() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.pp_layers = [0, 1, 2, 3]
    manager.num_layers = 4
    manager.num_local_layers = 4
    manager.layer_offsets = {0: 0, 1: 1, 2: 2, 3: 3}
    manager.num_kv_heads_per_layer = [2, 0, 0, 0]
    manager.head_dim_per_layer = [64, 64, 64, 64]
    manager.max_attention_window_vec = [None, None, None, None]

    config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=1024)],
                reuse_policy=ReusePolicy.PRIVATE,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1), buffers=[BufferConfig(role=Role.KEY, size=0)]
            ),
            SsmLayerConfig(
                layer_id=LayerId(2),
                buffers=[
                    BufferConfig(role=DataRole("ssm_state"), size=256),
                    BufferConfig(role=DataRole("conv_state"), size=0),
                    BufferConfig(role=DataRole("old_x"), size=64, reuse_policy=ReusePolicy.PRIVATE),
                ],
                reuse_policy=ReusePolicy.OPTIONAL,
            ),
            SsmLayerConfig(
                layer_id=LayerId(3),
                buffers=[BufferConfig(role=DataRole("ssm_state"), size=256)],
                reuse_policy=ReusePolicy.PRIVATE,
            ),
        ],
        commit_min_snapshot=True,
    )

    filtered = manager._remove_zero_size_buffers(config)

    assert manager.pp_layers == [0, 2, 3]
    assert [layer.layer_id for layer in filtered.layers] == [0, 1, 2]
    attention_layer, split_layer, private_layer = filtered.layers
    assert attention_layer.reuse_policy == ReusePolicy.PRIVATE
    assert isinstance(split_layer, SsmLayerConfig)
    assert split_layer.reuse_policy == ReusePolicy.OPTIONAL
    assert [(buffer.role, buffer.reuse_policy) for buffer in split_layer.buffers] == [
        (DataRole("ssm_state"), None),
        (DataRole("old_x"), ReusePolicy.PRIVATE),
    ]
    assert isinstance(private_layer, SsmLayerConfig)
    assert private_layer.reuse_policy == ReusePolicy.PRIVATE


def test_layer_group_helpers_describe_split_ssm_layers() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=4 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=4096)],
                sliding_window_size=8,
            ),
            *[
                SsmLayerConfig(
                    layer_id=LayerId(layer_id),
                    buffers=[
                        BufferConfig(role=DataRole("ssm_state"), size=4096),
                        BufferConfig(
                            role=DataRole("old_x"), size=2048, reuse_policy=ReusePolicy.PRIVATE
                        ),
                    ],
                )
                for layer_id in (1, 2)
            ],
        ],
        commit_min_snapshot=True,
    )
    impl = KVCacheManager(config)
    try:
        manager = object.__new__(KVCacheManagerV2)
        manager.max_seq_len = MAX_SEQ_LEN
        manager.tokens_per_block = TOKENS_PER_BLOCK
        manager.kv_cache_manager_py_config = config
        manager.impl = impl
        manager._layer_group_life_cycles = None

        attention = impl.get_layer_group_id(LayerId(0))
        required = impl.get_buffer_group_id(LayerId(1), DataRole("ssm_state"))
        private = impl.get_buffer_group_id(LayerId(1), DataRole("old_x"))
        assert len({attention, required, private}) == 3
        assert sorted({buffer.layer_id for buffer in impl.buffer_grouping[private]}) == [1, 2]

        assert manager._get_event_window_sizes_by_layer_group(attention_only=True) == {attention: 8}
        assert manager._get_event_window_sizes_by_layer_group() == {
            attention: 8,
            required: MAX_SEQ_LEN,
            private: MAX_SEQ_LEN,
        }
        life_cycles = manager._life_cycle_by_layer_group()
        assert life_cycles[attention].window_size == 8
        assert life_cycles[required].window_size is None
        assert life_cycles[private].window_size is None
        metadata = manager._stats_life_cycle_metadata()
        assert metadata[attention][1:] == (8, "attention")
        assert metadata[required][1:] == (None, "ssm")
        assert metadata[private][1:] == (None, "ssm")
        assert len({metadata[group][0] for group in (attention, required, private)}) == 3
    finally:
        impl.shutdown()


@pytest.mark.parametrize("sliding_window_size", [None, 8])
@pytest.mark.parametrize("attention_first", [False, True])
def test_event_window_sizes_filter_attention_without_backend_internals(
    sliding_window_size: int | None, attention_first: bool
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.max_seq_len = MAX_SEQ_LEN
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=128)],
                sliding_window_size=sliding_window_size,
            ),
            SsmLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(role=DataRole("ssm_state"), size=128)],
            ),
        ]
    )
    # Match the C++ binding surface: layer_grouping is public, while the
    # Python implementation's private _life_cycles registry is absent.
    manager.impl = SimpleNamespace(
        buffer_grouping=[
            [BufferId(LayerId(i), Role.KEY)] for i in ([0, 1] if attention_first else [1, 0])
        ]
    )

    attention_group_id = 0 if attention_first else 1
    window_size = MAX_SEQ_LEN if sliding_window_size is None else sliding_window_size
    # SSM must be excluded even when its window ties with full attention's window.
    assert manager._get_event_window_sizes_by_layer_group(attention_only=True) == {
        attention_group_id: window_size
    }
    assert manager._get_event_window_sizes_by_layer_group() == {
        attention_group_id: window_size,
        1 - attention_group_id: MAX_SEQ_LEN,
    }


@pytest.mark.parametrize(
    ("finished", "include_finished", "expect_relocation"),
    [(False, False, True), (True, False, False), (True, True, True)],
)
def test_draft_token_relocation_uses_local_cache_layout(
    monkeypatch: pytest.MonkeyPatch,
    finished: bool,
    include_finished: bool,
    expect_relocation: bool,
) -> None:
    request = SimpleNamespace(
        state=LlmRequestState.GENERATION_COMPLETE
        if finished
        else LlmRequestState.GENERATION_IN_PROGRESS,
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
        include_finished_requests=include_finished,
    )

    if not expect_relocation:
        update_op.assert_not_called()
        return

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
        cache_salt="11",
        kv_cache_retention_config=None,
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
    manager.kv_connector_manager = None
    manager.enable_block_reuse = True
    manager.is_estimating_kv_cache = False
    manager._has_cp_helix = False
    manager.kv_cache_map = {} if fresh_cache else {request.py_request_id: kv_cache}
    manager._on_allocation_stream = set()
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("offload", [False, True])
def test_pool_ratio_overrides_constraints(offload: bool) -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(pool_ratio=[1.0], avg_seq_len=256, host_cache_size=0),
        max_batch_size=3,
        max_num_tokens=2048,
        sparse_kv_offload=SparseKvCacheOffloadConfig() if offload else None,
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


def test_avg_seq_len_builds_context_warmup_constraint() -> None:
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
    assert config.constraints == [BatchDesc([KVCacheDesc(capacity=2048, history_length=0)])]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("avg_seq_len", "capacity"), [(1024, 897), (None, 1921)], ids=["avg_seq_len", "max_seq_len"]
)
def test_sparse_kv_offload_typical_step_models_only_decoding(
    avg_seq_len: int | None, capacity: int
) -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(host_cache_size=0, avg_seq_len=avg_seq_len),
        max_batch_size=3,
        max_seq_len=2048,
        max_num_tokens=2048,
        sparse_kv_offload=SparseKvCacheOffloadConfig(),
    )

    # Decode-only sizing samples a block boundary within avg_seq_len or max_seq_len,
    # with no context warmup batch. The native manager still derives the pool ratio.
    assert config.typical_step == BatchDesc(
        [KVCacheDesc(capacity=capacity, history_length=capacity - 1)] * 3
    )
    assert config.constraints == []
    assert config.initial_pool_ratio is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("max_seq_len", "full_history_pages"),
    [
        (1, 1),
        (127, 1),
        (128, 1),
        (129, 2),
        (255, 2),
        (256, 2),
        (257, 3),
        (32768, 256),
        (32769, 257),
        (33024, 258),
        (33025, 259),
    ],
)
@pytest.mark.parametrize("num_sink_tokens", [0, 129])
def test_sparse_kv_offload_pool_sizing_covers_window_boundaries(
    max_seq_len: int, full_history_pages: int, num_sink_tokens: int
) -> None:
    config = _make_cache_config_for_test(
        KvCacheConfig(host_cache_size=0),
        max_seq_len=max_seq_len,
        sparse_kv_offload=SparseKvCacheOffloadConfig(),
    )
    tokens_per_block = config.tokens_per_block
    generation = config.typical_step.kv_caches[0]
    assert 0 < generation.capacity <= max_seq_len
    assert generation.history_length == generation.capacity - 1
    assert (generation.capacity + tokens_per_block - 1) // tokens_per_block == full_history_pages

    def required_pages(life_cycle: AttnLifeCycle, capacity: int) -> int:
        stale = life_cycle.get_stale_range(capacity - 1, tokens_per_block)
        return (capacity + tokens_per_block - 1) // tokens_per_block - len(stale)

    for window in (None, 1, 8, 128, 129, 256, 65536):
        life_cycle = AttnLifeCycle.make(
            window, num_sink_tokens if window is not None else 0, tokens_per_block
        )
        # Native lifecycle accounting must cover every decode phase, including
        # windows shorter than a page, non-aligned windows and retained sinks.
        peak_pages = max(
            required_pages(life_cycle, capacity)
            for capacity in range(max(1, max_seq_len - tokens_per_block + 1), max_seq_len + 1)
        )
        assert required_pages(life_cycle, generation.capacity) == peak_pages, window
        if max_seq_len == 33024 and num_sink_tokens == 0 and window in (8, 128):
            assert peak_pages == 2


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
    dist.allreduce.return_value = int(_KVCacheManagerInitStatus.ABORT)

    with patch.object(Distributed, "get", return_value=dist):
        status = _sync_kv_cache_manager_init_status(_KVCacheManagerInitStatus.OK, mapping)

    assert status == _KVCacheManagerInitStatus.ABORT
    dist.allreduce.assert_called_once_with(int(_KVCacheManagerInitStatus.OK), op=ReduceOp.MAX)


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device (MPI) build required")
def test_world_ranks_converge_on_host_failure() -> None:
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    session = MpiPoolSession(n_workers=2)
    try:
        results = session.submit_sync(_multi_rank_host_failure_consensus_worker)
    finally:
        session.shutdown()

    # Rank 1 failed locally; rank 0 succeeded and must abort with it, after
    # shutting down the manager it had already built.
    assert sorted(results) == [(0, 2, True, 1), (1, 2, True, 0)]


def test_peer_init_failure_discards_local_candidate() -> None:
    """A rank whose own init succeeded still aborts, and releases what it built."""
    local_impl = Mock()
    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"

    with (
        patch(
            f"{module}._sync_kv_cache_manager_init_status",
            return_value=_KVCacheManagerInitStatus.ABORT,
        ),
        pytest.raises(RuntimeError, match="failed on another rank"),
    ):
        _make_manager_for_cache_tier_test(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=16 << 20,
            ),
            [local_impl],
        )

    local_impl.shutdown.assert_called_once_with()


@pytest.mark.cpu_only
@pytest.mark.parametrize("peer_failed", [False, True])
def test_agg_helix_gpu_only_init_syncs_status_across_ranks(peer_failed: bool) -> None:
    """Aggregated helix has no host tier but must still agree on init success.

    Without the sync, a rank that failed to allocate raises alone and its peers
    carry on into collectives that never complete.
    """
    local_impl = Mock()
    helix_mapping = Mapping(
        world_size=2,
        rank=0,
        cp_size=2,
        cp_config={"cp_type": CpType.HELIX},
    )
    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
    status = _KVCacheManagerInitStatus.ABORT if peer_failed else _KVCacheManagerInitStatus.OK
    dist = Mock()
    dist.allreduce.side_effect = lambda value, op: value

    with (
        patch.object(Distributed, "get", return_value=dist),
        patch(f"{module}._sync_kv_cache_manager_init_status", return_value=status) as sync,
    ):
        config = KvCacheConfig(
            max_gpu_total_bytes=16 << 20, host_cache_size=0, enable_block_reuse=False
        )
        if peer_failed:
            with pytest.raises(RuntimeError, match="failed on another rank"):
                _make_manager_for_cache_tier_test(config, [local_impl], mapping=helix_mapping)
            local_impl.shutdown.assert_called_once_with()
        else:
            _make_manager_for_cache_tier_test(config, [local_impl], mapping=helix_mapping)

    sync.assert_called_once_with(_KVCacheManagerInitStatus.OK, helix_mapping)


@pytest.mark.parametrize("add_secondary_gpu_tier", [False, True])
def test_host_init_failure_propagates_without_rebuilding(add_secondary_gpu_tier: bool) -> None:
    """A configured host tier is honoured or the manager does not come up.

    Dropping the tier and rebuilding would run the job on a cache other than the
    one asked for. The construction is attempted exactly once.
    """
    # A second impl is offered: if the host tier were dropped and construction
    # retried, that retry would succeed and nothing would be raised. The error
    # escaping is what shows the retry did not happen.
    with pytest.raises(_CacheTierInitError, match="host tier init failed"):
        _make_manager_for_cache_tier_test(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=16 << 20,
            ),
            [_CacheTierInitError("host tier init failed"), Mock()],
            add_secondary_gpu_tier=add_secondary_gpu_tier,
        )


def test_host_init_failure_propagates_with_disk_tier(tmp_path) -> None:
    """The disk tier is no reason to retry without the host tier either."""
    with pytest.raises(_CacheTierInitError, match="host tier init failed"):
        _make_manager_for_cache_tier_test(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=16 << 20,
                disk_cache_size=16 << 20,
                disk_cache_path=str(tmp_path),
            ),
            [_CacheTierInitError("host tier init failed"), Mock()],
        )


@pytest.mark.cpu_only
def test_cold_codec_provider_receives_draft_role() -> None:
    impl = Mock()
    codec_provider = Mock(mixed_precision_high_window=None)
    codec_provider.create_cold_page_codec.return_value = object()
    _make_manager_for_cache_tier_test(
        KvCacheConfig(max_gpu_total_bytes=16 << 20),
        [impl],
        cold_page_codec_provider=codec_provider,
        is_draft=True,
    )

    assert codec_provider.create_cold_page_codec.call_args.kwargs["is_draft"] is True


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "manager_dtype,layer_dtype",
    [
        (DataType.HALF, DataType.BF16),
        (DataType.BF16, DataType.HALF),
        (DataType.HALF, DataType.FP8),
    ],
)
def test_cold_codec_provider_receives_layer_spec_dtype(
    manager_dtype: DataType, layer_dtype: DataType
) -> None:
    codec_provider = Mock(mixed_precision_high_window=None)
    layer = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=layer_dtype,
        num_kv_heads=1,
        head_dim=1,
        buffer_roles=(Role.KEY,),
    )
    manager, _ = _make_manager_for_cache_tier_test(
        KvCacheConfig(max_gpu_total_bytes=16 << 20),
        [Mock()],
        cold_page_codec_provider=codec_provider,
        dtype=manager_dtype,
        layer_specs=[layer],
    )

    assert manager.dtype == manager_dtype
    codec_provider.create_cold_page_codec.assert_called_once()
    assert codec_provider.create_cold_page_codec.call_args.kwargs["runtime_dtype"] == layer_dtype


@pytest.mark.cpu_only
@pytest.mark.parametrize("manager_dtype", [DataType.HALF, DataType.BF16])
def test_cold_codec_provider_receives_legacy_manager_dtype(manager_dtype: DataType) -> None:
    codec_provider = Mock(mixed_precision_high_window=None)
    _make_manager_for_cache_tier_test(
        KvCacheConfig(max_gpu_total_bytes=16 << 20),
        [Mock()],
        cold_page_codec_provider=codec_provider,
        dtype=manager_dtype,
    )

    codec_provider.create_cold_page_codec.assert_called_once()
    assert codec_provider.create_cold_page_codec.call_args.kwargs["runtime_dtype"] == manager_dtype


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "layer_dtypes", [(DataType.BF16, DataType.HALF), (DataType.BF16, DataType.FP8)]
)
def test_cold_codec_provider_rejects_mixed_layer_spec_dtypes(
    layer_dtypes: tuple[DataType, DataType],
) -> None:
    codec_provider = Mock(mixed_precision_high_window=None)
    layer_specs = [
        KVCacheLayerSpec(
            global_layer_id=layer_id,
            owner=KVCacheLayerOwner.TARGET,
            dtype=layer_dtype,
            num_kv_heads=1,
            head_dim=1,
            buffer_roles=(Role.KEY,),
        )
        for layer_id, layer_dtype in enumerate(layer_dtypes)
    ]
    with pytest.raises(NotImplementedError, match="mixed-dtype KV-cache layer specs"):
        _make_manager_for_cache_tier_test(
            KvCacheConfig(max_gpu_total_bytes=16 << 20),
            [Mock()],
            cold_page_codec_provider=codec_provider,
            layer_specs=layer_specs,
        )

    codec_provider.create_cold_page_codec.assert_not_called()


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
    assert config.constraints == [BatchDesc([KVCacheDesc(capacity=258, history_length=0)])]


@pytest.mark.parametrize("guard", [None, "context_only", "compression", "connector"])
def test_try_commit_blocks_at_context_end(guard: str | None) -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        is_context_only_request=guard == "context_only",
        context_current_position=10,
        context_remaining_length=0,
        get_tokens_view=lambda beam_id: np.arange(10, dtype=np.int32),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = _make_publishing_manager(BlockReusePolicy.ALL_REUSABLE)
    manager.kv_compression_manages_history = guard == "compression"
    manager.kv_connector_manager = Mock() if guard == "connector" else None
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert list(kv_cache.committed_tokens) == [4, 5, 6, 7, 8, 9]
    assert kv_cache.num_committed_tokens == 10
    assert kv_cache.stopped_committing is (guard is not None)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("divergence", "num_lookup_tokens", "reused", "expected"),
    [
        # A sibling diverges at 34,400; the point aligns down to the 64-token block.
        (34400, 34999, 0, 34368),
        # The whole lookup matched, so there is no fork.
        (34999, 34999, 0, None),
        # Less than one shared block.
        (40, 999, 0, None),
        # Reuse already reached the fork.
        (34400, 34999, 34368, None),
    ],
)
def test_swa_branch_snapshot_point(
    divergence: int, num_lookup_tokens: int, reused: int, expected: int | None
) -> None:
    assert (
        _swa_branch_snapshot_point(divergence, num_lookup_tokens, reused, num_lookup_tokens + 1, 64)
        == expected
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("record", "is_dummy_request", "num_lookup_tokens", "points", "expected"),
    [
        (True, False, 999, [], [512]),
        # A later lookup that matches in full clears the earlier point.
        (True, False, 512, [256], []),
        # A resumed cache does no lookup and keeps the point.
        (True, False, None, [256], [256]),
        (False, False, 999, [], []),
        (True, True, 999, [], []),
    ],
)
def test_record_branch_snapshot_point(
    record: bool,
    is_dummy_request: bool,
    num_lookup_tokens: int | None,
    points: list[int],
    expected: list[int],
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._record_branch_snapshots = record
    manager.tokens_per_block = 64
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=is_dummy_request,
        prompt_len=1000,
        expect_snapshot_points=points,
    )
    kv_cache = Mock(num_committed_tokens=0)
    kv_cache._get_num_reusable_tokens_before_pruning.return_value = 512

    manager._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens)

    assert request.expect_snapshot_points == expected


def test_generation_allocation_reserves_dynamic_width() -> None:
    request = SimpleNamespace(
        py_request_id=80,
        py_beam_width=1,
        py_num_accepted_draft_tokens=2,
        py_rewind_len=2,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        max_beam_num_tokens=103,
    )
    kv_cache = Mock(is_active=True, capacity=100, beam_width=1)

    def resize(capacity, history_length=None):
        if capacity is not None:
            kv_cache.capacity = capacity
        return True

    kv_cache.resize.side_effect = resize
    manager = object.__new__(KVCacheManagerV2)
    manager.kv_cache_type = CacheType.SELFKONLY
    manager.max_beam_width = 1
    manager.is_draft = True
    manager._has_cp_helix = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._allocated_draft_lens = {}
    manager._inflight_draft_lens = {}
    manager._kv_reserve_draft_tokens = 4
    manager.enable_block_reuse = False
    manager._effective_draft_len = Mock(return_value=2)
    manager.kv_compression_manages_history = False
    # Fresh-page fill is off; the allocation path calls it unconditionally.
    manager._fresh_page_fill = None
    # No windowed pools; the window-crossing check reads them on every growth.
    manager._windowed_layer_group_sizes = {}

    assert manager.try_allocate_generation(request)
    assert kv_cache.resize.call_args_list[0].args == (105,)
    assert manager._allocated_draft_lens[request.py_request_id] == 4
    assert manager._inflight_draft_lens[request.py_request_id] == [4]

    batch = ScheduledRequests()
    batch.generation_requests.append(request)
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2."
        "_update_kv_cache_draft_token_location"
    ):
        manager.update_resources(batch)

    assert kv_cache.resize.call_args_list[1].args == (103, 102)
    assert request.py_request_id not in manager._allocated_draft_lens
    assert request.py_request_id not in manager._inflight_draft_lens


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
    manager.kv_connector_manager = None
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
    manager.kv_connector_manager = None
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

    kv_cache = Mock(num_committed_tokens=64, is_active=True, capacity=192, beam_width=1)
    manager = object.__new__(KVCacheManagerV2)
    manager.kv_connector_manager = None
    manager.kv_cache_type = CacheType.SELFKONLY
    manager.max_beam_width = 1
    manager.is_draft = True
    manager.enable_block_reuse = True
    manager.enable_joint_kv_cache_reuse = True
    manager._mirror_uses_draft_view = True
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


@pytest.mark.parametrize(
    (
        "uses_draft_view",
        "disagg_gen_init",
        "position",
        "mirror_capacity",
        "mirror_history",
        "expected_resize",
        "dflash",
    ),
    [
        # External drafter (target view), aggregated chunked prefill. The
        # prompt is 8192 + 2 * 1024 tokens, so the 1024-token chunk at 8192 is
        # not the last one and carries no draft tokens.
        pytest.param(False, False, 0, 0, 0, (1024, 0), False, id="first_chunk"),
        # A later chunk of a mirror that holds [0, 8192): the chunk start is
        # history, so a sliding window drops the pages below it.
        pytest.param(False, False, 8192, 8192, 4096, (9216, 8192), False, id="later_chunk"),
        # The history is already ahead of the chunk start; a resize never
        # lowers it.
        pytest.param(False, False, 8192, 8192, 8704, (9216, 8704), False, id="history_ahead"),
        # A mirror first seen past position 0 (a target prefix this worker did
        # not compute). A DFlash/DSpark pool takes the chunk start as history:
        # that drafter stores the chunk at its target position above the
        # prefix. Any other keeps its history and covers [0, chunk end).
        pytest.param(False, False, 8192, 0, 0, (9216, 8192), True, id="fresh_past_zero-dflash"),
        pytest.param(False, False, 8192, 0, 0, (9216, None), False, id="fresh_past_zero-pard"),
        # Target chunk progress sizes the mirror while its history is preserved.
        pytest.param(True, False, 8192, 8192, 0, (9216, None), False, id="draft_view"),
        # Disaggregated generation init: the mirror holds no drafter K/V for
        # the transferred prompt (10240 tokens), so the whole prompt is
        # history; the capacity stays absolute, as the drafter's positions.
        pytest.param(False, True, 0, 0, 0, (10243, 10240), False, id="disagg_gen_init-target_view"),
        pytest.param(True, True, 0, 0, 0, (10243, None), False, id="disagg_gen_init-draft_view"),
    ],
)
def test_unpaired_draft_mirror_context_resize(
    uses_draft_view: bool,
    disagg_gen_init: bool,
    position: int,
    mirror_capacity: int,
    mirror_history: int,
    expected_resize: tuple,
    dflash: bool,
) -> None:
    """Which request view an unpaired draft mirror reads, and the capacity and
    history its context resize asks for."""
    prompt, chunk, draft_len = 8192 + 2 * 1024, 1024, 3
    request = LlmRequest(
        request_id=41,
        max_new_tokens=4,
        input_tokens=[0] * prompt,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=[1] * draft_len,
    )
    if disagg_gen_init:
        request.state = LlmRequestState.DISAGG_GENERATION_INIT
        assert request.is_last_context_chunk
    else:
        request.state = LlmRequestState.CONTEXT_INIT
        request.context_current_position = position
        request.context_chunk_size = chunk
        assert not request.is_last_context_chunk

    kv_cache = Mock(is_active=True, capacity=mirror_capacity, history_length=mirror_history)
    views = []
    kv_cache.resize.side_effect = lambda *args: views.append(request.use_draft_model) or True
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_joint_kv_cache_reuse = False
    manager._mirror_uses_draft_view = uses_draft_view
    manager._is_dflash_draft = dflash
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._resume_and_restore = Mock(return_value=True)

    batch = ScheduledRequests()
    batch.context_requests_last_chunk.append(request)
    manager._prepare_draft_resources(batch)

    kv_cache.resize.assert_called_once_with(*expected_resize)
    assert views == [False]


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_unpaired_draft_mirror_that_cannot_grow_raises_with_the_pool_diagnostic(phase: str) -> None:
    """When a mirror finds no page to grow into, the iteration fails
    explicitly: the mirror left at its size has no storage for the K/V the
    drafter writes next, and there is no per-request way to skip drafting.
    The error carries the pool diagnostic."""
    position, chunk, draft_len = 8192, 1024, 3

    def make_request(request_id: int) -> LlmRequest:
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=4,
            input_tokens=[0] * (position + chunk),
            sampling_config=SamplingConfig(1),
            is_streaming=False,
            draft_tokens=[1] * draft_len,
        )
        if phase == "context":
            request.state = LlmRequestState.CONTEXT_INIT
            request.context_current_position = position
            request.context_chunk_size = chunk
        else:
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
        return request

    requests = [make_request(51), make_request(52)]
    caches = {
        51: Mock(is_active=True, capacity=position, history_length=position, beam_width=1),
        52: Mock(is_active=True, capacity=position, history_length=position, beam_width=1),
    }
    caches[51].resize.return_value = True
    caches[52].resize.return_value = False
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_joint_kv_cache_reuse = False
    manager._mirror_uses_draft_view = False
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = dict(caches)
    manager._allocated_draft_lens = set()
    manager._resume_and_restore = Mock(return_value=True)
    manager._required_gen_capacity = Mock(return_value=position + 1 + draft_len)
    manager._draft_pool_diagnostic = Mock(return_value=" [draft pool: full]")
    manager.kv_cache_type = CacheType.SELF
    manager.max_beam_width = 1

    batch = ScheduledRequests()
    if phase == "context":
        batch.context_requests_last_chunk.extend(requests)
        expected_resize = call(position + chunk + draft_len, position)
        expected_capacity = position + chunk + draft_len
    else:
        batch.generation_requests.extend(requests)
        expected_resize = call(position + 1 + draft_len)
        expected_capacity = position + 1 + draft_len
    with pytest.raises(RuntimeError) as excinfo:
        manager._prepare_draft_resources(batch)

    assert caches[51].resize.call_args_list == [expected_resize]
    assert caches[52].resize.call_args_list == [expected_resize]
    message = str(excinfo.value)
    assert f"Draft KV cache {phase} resize failed for request 52" in message
    assert f"could not resize to {expected_capacity} tokens" in message
    assert "[draft pool: full]" in message


def test_exhausted_pool_fails_explicitly_on_an_empty_fresh_mirror() -> None:
    """A fresh mirror whose first resize fails holds no page at all. The
    manager raises before the forward, with a diagnostic that shows the empty
    mirror and how full the pool was."""
    prompt, draft_len = 1024, 3
    request = LlmRequest(
        request_id=61,
        max_new_tokens=4,
        input_tokens=[0] * prompt,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=[1] * draft_len,
    )
    request.state = LlmRequestState.CONTEXT_INIT
    request.context_current_position = 0
    request.context_chunk_size = prompt

    # Two occupants with distinct fields, so a swapped field or a max in
    # place of the sum changes the message.
    occupant = Mock(is_active=True, capacity=1216, history_length=512, num_blocks=38)
    small = Mock(is_active=True, capacity=64, history_length=0, num_blocks=2)
    fresh = Mock(is_active=True, capacity=0, history_length=0, num_blocks=0)
    fresh.resize.return_value = False
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_joint_kv_cache_reuse = False
    manager._mirror_uses_draft_view = False
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = {60: occupant, 62: small, 61: fresh}
    manager.max_attention_window_vec = [2048]
    manager._gpu_max_tokens = 1216
    manager._resume_and_restore = Mock(return_value=True)

    batch = ScheduledRequests()
    batch.context_requests_last_chunk.append(request)
    with pytest.raises(RuntimeError) as excinfo:
        manager._prepare_draft_resources(batch)

    fresh.resize.assert_called_once_with(prompt + draft_len, 0)
    message = str(excinfo.value)
    assert (
        f"context resize failed for request 61: could not resize to {prompt + draft_len} tokens"
        in message
    )
    assert "3 live caches, capacity 1280 tokens in 40 logical blocks" in message
    assert "60: capacity=1216 history=512 logical_blocks=38" in message
    assert "61: capacity=0 history=0 logical_blocks=0" in message
    assert "windows=[2048]" in message
    assert "gpu_max_tokens=1216" in message


def test_windowed_draft_pool_backs_only_the_window_of_a_generation_dummy() -> None:
    """A CUDA-graph warmup dummy is a generation request at the longest length
    the target pool can hold. Its draft mirror carries the same history hint
    as the target cache, so a windowed draft pool sized for a few mirrors of
    window + chunk backs only the dummy's window; without the hint the mirror
    would need the dummy's whole length, which such a pool cannot hold."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    spec = DFlashDecodingConfig(max_draft_len=4, speculative_model="draft")
    # 32 KiB per token (8 layers, 8 KV heads of 128, fp16): a mirror of a
    # 256-token window plus a 512-token chunk is ~26 MiB; the pool holds four
    # of them (~104 MiB), while a 60k-token dummy held in full needs ~1.8 GiB.
    num_layers, num_kv_heads, head_dim = 8, 8, 128
    tokens_per_block, window, chunk, max_seq_len, batch_size = 32, 256, 512, 65536, 2
    token_num = 60_000
    draft_config = KvCacheConfig(
        enable_block_reuse=False, max_attention_window=[window] * num_layers
    )
    bytes_per_token = num_layers * 2 * num_kv_heads * head_dim * 2
    _, headroom = _get_generation_kv_capacity(spec, is_draft=True)
    tokens_per_mirror = window + chunk + 2 * headroom + tokens_per_block
    pages_per_mirror = -(-tokens_per_mirror // tokens_per_block)
    draft_budget = 2 * batch_size * pages_per_mirror * tokens_per_block * bytes_per_token
    assert token_num * bytes_per_token > 4 * draft_budget
    draft = KVCacheManagerV2(
        draft_config.model_copy(update={"max_gpu_total_bytes": draft_budget}),
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=chunk,
        mapping=Mapping(),
        dtype=DataType.HALF,
        spec_config=spec,
        is_draft=True,
    )
    # A small windowless target pool (512 B per token) that holds the whole
    # dummy, so the target side of add_dummy_requests cannot be what fails.
    target = KVCacheManagerV2(
        KvCacheConfig(enable_block_reuse=False, max_gpu_total_bytes=128 << 20),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=chunk,
        mapping=Mapping(),
        dtype=DataType.HALF,
        spec_config=spec,
    )
    stream = torch.cuda.current_stream().cuda_stream

    def live_pages(cache: object) -> int:
        return len(cache.get_aggregated_page_indices(0, valid_only=True))

    requests = None
    try:
        # The mirror resize without a history hint: the whole dummy on the
        # windowed pool, which is more than the pool has.
        probe = draft.impl.create_kv_cache()
        try:
            assert probe.resume(stream)
            assert not probe.resize(token_num + draft.num_extra_kv_tokens)
        finally:
            probe.close()
        requests = target.add_dummy_requests(
            [7],
            token_nums=[token_num],
            is_gen=True,
            max_num_draft_tokens=spec.max_draft_len,
            draft_kv_cache_manager=draft,
        )
        assert requests is not None, "the dummy's draft mirror did not fit the windowed pool"
        (request,) = requests
        target_cache = target.kv_cache_map[request.py_request_id]
        mirror = draft.kv_cache_map[request.py_request_id]
        assert live_pages(target_cache) >= token_num // tokens_per_block
        assert mirror.capacity == target_cache.capacity
        assert mirror.history_length == token_num - 1
        assert live_pages(mirror) <= pages_per_mirror
    finally:
        for request in requests or ():
            draft.free_resources(request)
            target.free_resources(request)
        draft.shutdown()
        target.shutdown()


def _small_spec_manager(
    spec: DecodingBaseConfig,
    is_draft: bool,
    *,
    window: int | None = None,
    num_kv_heads: int = 2,
    dtype: DataType = DataType.HALF,
    max_gpu_total_bytes: int = 256 << 20,
    max_seq_len: int = 16384,
    max_num_tokens: int = 8192,
    tokens_per_block: int = 32,
) -> KVCacheManagerV2:
    """A two-layer, head_dim 64 V2 manager for ``spec`` (32-token pages by
    default), optionally with a uniform sliding window and no block reuse."""
    num_layers = 2
    config = KvCacheConfig(
        enable_block_reuse=False,
        max_attention_window=None if window is None else [window] * num_layers,
        max_gpu_total_bytes=max_gpu_total_bytes,
    )
    return KVCacheManagerV2(
        config,
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=2,
        max_num_tokens=max_num_tokens,
        mapping=Mapping(),
        dtype=dtype,
        spec_config=spec,
        is_draft=is_draft,
    )


def _backed_ordinals(cache: object) -> set[int]:
    """Block ordinals of ``cache`` that have a page behind them."""
    return {
        ordinal
        for ordinal, page in enumerate(cache.get_base_page_indices(0))
        if page != BAD_PAGE_INDEX
    }


@pytest.mark.parametrize(
    ("make_spec", "external_drafter"),
    [
        pytest.param(
            lambda: DFlashDecodingConfig(max_draft_len=4, speculative_model="draft"),
            True,
            id="dflash",
        ),
        pytest.param(
            lambda: DSparkDecodingConfig(max_draft_len=4, speculative_model="draft"),
            True,
            id="dspark",
        ),
        pytest.param(
            lambda: PARDDecodingConfig(max_draft_len=4, speculative_model="draft"),
            True,
            id="pard",
        ),
        pytest.param(
            lambda: DraftTargetDecodingConfig(max_draft_len=4, speculative_model="draft"),
            True,
            id="draft_target_one_model",
        ),
        pytest.param(lambda: MTPDecodingConfig(num_nextn_predict_layers=1), False, id="mtp"),
        pytest.param(
            lambda: Eagle3DecodingConfig(max_draft_len=3, speculative_model="draft"),
            False,
            id="eagle3_one_model",
        ),
    ],
)
def test_only_an_external_drafters_draft_manager_reads_the_target_view(
    make_spec: Callable[[], DecodingBaseConfig], external_drafter: bool
) -> None:
    """The draft manager of a one-engine external drafter reads the target's
    position and chunk; every other manager keeps the draft request view."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    spec = make_spec()
    assert spec.spec_dec_mode.is_external_drafter() is external_drafter
    for is_draft in (False, True):
        manager = _small_spec_manager(spec, is_draft, max_gpu_total_bytes=4 << 20)
        try:
            assert manager._mirror_uses_draft_view is not (is_draft and external_drafter)
        finally:
            manager.shutdown()


@pytest.mark.parametrize(
    ("make_spec", "releases_below_window"),
    [
        # MTP/Eagle draft layers process the target's chunks through the draft
        # pool but the mirror reads the draft request view, which nothing
        # advances: it keeps the whole prompt backed.
        pytest.param(
            lambda: Eagle3DecodingConfig(max_draft_len=3, speculative_model="draft"),
            False,
            id="eagle3_one_model",
        ),
        # An external drafter follows the target's chunks and drops what falls
        # out of the window.
        pytest.param(
            lambda: DFlashDecodingConfig(max_draft_len=4, speculative_model="draft"),
            True,
            id="dflash",
        ),
    ],
)
def test_windowed_draft_mirror_backs_what_the_next_chunk_attends(
    make_spec: Callable[[], DecodingBaseConfig], releases_below_window: bool
) -> None:
    """Chunked prefill with block reuse off on a sliding-window draft pool:
    after the second chunk's resize, every page the chunk attends (its window
    back from the chunk start, through the chunk end) is backed."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    window, page, prompt, chunk = 256, 32, 8192, 512
    draft = _small_spec_manager(make_spec(), True, window=window)
    request = LlmRequest(
        request_id=101,
        max_new_tokens=8,
        input_tokens=[1] * prompt,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.state = LlmRequestState.CONTEXT_INIT
    try:
        request.context_chunk_size = chunk
        batch = ScheduledRequests()
        batch.context_requests_chunking.append(request)
        draft._prepare_draft_resources(batch)

        request.move_to_next_context_chunk()
        request.context_chunk_size = chunk
        assert request.context_current_position == chunk
        batch = ScheduledRequests()
        batch.context_requests_chunking.append(request)
        draft._prepare_draft_resources(batch)

        backed = _backed_ordinals(draft.kv_cache_map[request.py_request_id])
        attended = set(range((chunk - window) // page, 2 * chunk // page))
        assert attended <= backed
        assert (0 in backed) is not releases_below_window
        draft.free_resources(request)
    finally:
        draft.shutdown()


class _PoolBoundDFlashDraftModel:
    """Just enough of a DFlash draft model for the worker to bind the draft
    KV pool and run its generation-step context store."""

    block_size = 4
    config = SimpleNamespace(max_position_embeddings=1 << 17)
    dflash_attention_backend = "TRTLLM"
    hidden_norm = object()

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, hidden: int) -> None:
        self.fc = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16, device="cuda"))
        self._layers = (num_layers, num_kv_heads, head_dim)
        embed_tokens = torch.nn.Embedding(16, hidden).cuda()
        self.draft_model_full = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens))

    def _build_fused_kv_buffers(self) -> None:
        self._num_attn_layers, self._num_kv_heads, self._head_dim = self._layers
        self._num_heads = self._num_kv_heads

    def _get_attention_mask_args(self, layer_idx: int) -> tuple:
        return True, (-1, -1)

    def validate_block_attention_windows(self) -> frozenset:
        return frozenset()

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def precompute_context_kv(
        self, projected: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_layers, num_kv_heads, head_dim = self._layers
        k = torch.full(
            (projected.shape[0], num_layers, num_kv_heads, head_dim),
            3.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        return k, k.clone()


def _disagg_draft_fixture(
    spec: DFlashDecodingConfig,
    prompt: int,
    occupant_prompt: int,
    window: int,
    tokens_per_block: int = 32,
) -> tuple[KVCacheManagerV2, LlmRequest, LlmRequest]:
    """A sliding-window draft pool with room for 8192 tokens (two layers, one
    KV head of 64, bf16), an occupant that prefilled ``occupant_prompt``
    tokens on it, and a request in disaggregated generation init whose
    transferred prompt is ``prompt`` tokens."""
    num_layers, num_kv_heads, head_dim = 2, 1, 64
    bytes_per_token = num_layers * 2 * num_kv_heads * head_dim * 2
    draft = _small_spec_manager(
        spec,
        True,
        window=window,
        num_kv_heads=num_kv_heads,
        dtype=DataType.BF16,
        max_gpu_total_bytes=8192 * bytes_per_token,
        max_seq_len=1 << 17,
        max_num_tokens=occupant_prompt,
        tokens_per_block=tokens_per_block,
    )
    occupant = LlmRequest(
        request_id=1,
        max_new_tokens=8,
        input_tokens=[1] * occupant_prompt,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    occupant.state = LlmRequestState.CONTEXT_INIT
    occupant.context_chunk_size = occupant_prompt
    request = LlmRequest(
        request_id=2,
        max_new_tokens=64,
        input_tokens=[1] * prompt,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=[1] * spec.max_draft_len,
    )
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    return draft, occupant, request


def _disagg_generation_step(draft: KVCacheManagerV2, request: LlmRequest) -> None:
    """One executor generation step on the draft manager: grow the mirror,
    then the post-forward update_resources (nothing accepted, one token)."""
    batch = ScheduledRequests()
    batch.generation_requests.append(request)
    draft._prepare_draft_resources(batch)
    request.py_num_accepted_draft_tokens = 0
    request.py_rewind_len = 0
    draft.update_resources(batch)
    request.add_new_token(1, 0)


def test_gen_init_draft_mirror_survives_generation_updates() -> None:
    """A disaggregated generation worker's draft mirror for a prompt far
    larger than its sliding-window pool fits at generation init and stays
    consistent through generation steps, including update_resources, which
    advances the history in absolute positions: the history follows the
    request and the pages from the prompt end up to the capacity stay
    backed."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    spec = DFlashDecodingConfig(max_draft_len=3, speculative_model="draft")
    prompt, page = 65536, 32
    draft, occupant, request = _disagg_draft_fixture(spec, prompt, 512, window=256)
    try:
        for req in (occupant, request):
            batch = ScheduledRequests()
            batch.context_requests_last_chunk.append(req)
            draft._prepare_draft_resources(batch)
        cache = draft.kv_cache_map[request.py_request_id]
        # The context worker's first token arrives with the transferred KV.
        request.add_new_token(1, 0)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        for _ in range(3):
            _disagg_generation_step(draft, request)
            assert cache.history_length == request.max_beam_num_tokens - 2
            assert set(range(prompt // page, -(-cache.capacity // page))) <= _backed_ordinals(cache)
        draft.free_resources(request)
        draft.free_resources(occupant)
    finally:
        draft.shutdown()


@pytest.mark.parametrize("slot_loss", [False, True], ids=["first_step", "slot_loss"])
@pytest.mark.parametrize("reuse_addressable", [False, True], ids=["unpaired", "reuse_addressable"])
def test_disagg_generation_drafter_writes_stay_on_its_own_draft_pages(
    monkeypatch: pytest.MonkeyPatch, reuse_addressable: bool, slot_loss: bool
) -> None:
    """A disaggregated generation worker receives a prompt far larger than its
    sliding-window draft pool, while another request owns the pool's low
    pages. After generation steps (update_resources included), the drafter's
    step -- DFlashSpecMetadata.prepare() assigning the request a fresh slot,
    the context K/V store of the accepted tokens, the block decode's read and
    its block append -- must address only pages the request's own mirror
    holds. The slot is seeded on the page holding the request's absolute
    position and reads only what it stored, whether or not the pool joins
    block reuse, and again after the request loses its slot and comes back."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.dflash.validate_dflash_trtllm_gen_runtime",
        lambda **kwargs: None,
    )
    num_layers, num_kv_heads, head_dim, hidden = 2, 1, 64, 8
    window, prompt, occupant_prompt, max_draft_len = 256, 65536, 512, 3
    spec = DFlashDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model="draft",
        attention_backend="TRTLLM",
        mask_token_id=0,
    )
    draft, occupant, request = _disagg_draft_fixture(spec, prompt, occupant_prompt, window)
    offsets = torch.zeros(
        (getattr(draft, "num_attention_op_pools", draft.num_pools), 2, 2, draft.max_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    tokens_per_req = max_draft_len + 1
    try:
        for req in (occupant, request):
            batch = ScheduledRequests()
            batch.context_requests_last_chunk.append(req)
            draft._prepare_draft_resources(batch)
        request.add_new_token(1, 0)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        for _ in range(2):
            _disagg_generation_step(draft, request)

        worker = DFlashWorker(spec, Mapping())
        draft_model = _PoolBoundDFlashDraftModel(num_layers, num_kv_heads, head_dim, hidden)
        worker.set_draft_model(draft_model)
        attn_metadata = SimpleNamespace(
            max_seq_len=1 << 17,
            draft_kv_cache_block_offsets=offsets,
            num_contexts=0,
            num_seqs=1,
            num_ctx_tokens=0,
            _seq_lens_cuda=torch.zeros(1, dtype=torch.int32, device="cuda"),
            _seq_lens=torch.zeros(1, dtype=torch.int32),
        )
        worker._lazy_init_ctx_buffers(
            draft_model,
            SimpleNamespace(max_num_requests=2),
            attn_metadata,
            draft_kv_cache_manager=draft,
        )
        assert worker._ctx_block_tables is not None, "the drafter did not bind the draft pool"
        worker._ctx_reuse_addressable = reuse_addressable
        # V2 hands out per-layer views over one interleaved pool: a table entry
        # names layer 0's slot of a page, and layer l of that page is slot
        # entry + l of the same dense view, so view 0 sees every layer's write.
        assert worker._ctx_block_divisor == draft.kv_factor
        pool = worker._ctx_kv_buf[0]
        page = worker._ctx_page_size

        spec_metadata = DFlashSpecMetadata(
            max_num_requests=2, max_draft_len=max_draft_len, max_total_draft_tokens=max_draft_len
        )
        spec_metadata._dflash_worker = worker

        def prepare(request_ids: list[int]) -> None:
            """DFlashSpecMetadata.prepare() for a batch of generation rows at
            their current positions."""
            reqs = {request.py_request_id: request, occupant.py_request_id: occupant}
            spec_metadata.request_ids = request_ids
            spec_metadata.num_generations = len(request_ids)
            spec_metadata.seq_lens = [tokens_per_req] * len(request_ids)
            spec_metadata.host_position_ids = torch.tensor(
                [
                    reqs[rid].max_beam_num_tokens - 1 + offset
                    for rid in request_ids
                    for offset in range(tokens_per_req)
                ],
                dtype=torch.int32,
            )
            spec_metadata.prepare()

        prepare([request.py_request_id])
        if slot_loss:
            # The request misses a batch (its slot is evicted), runs a step
            # elsewhere in the meantime, and comes back on a fresh slot.
            prepare([])
            _disagg_generation_step(draft, request)
            prepare([request.py_request_id])
        batch = ScheduledRequests()
        batch.generation_requests.append(request)
        draft._prepare_draft_resources(batch)

        def own_slots(req: LlmRequest) -> set[int]:
            draft.copy_batch_block_offsets(offsets, [req.py_request_id], 1, 0, 1)
            worker._refresh_ctx_block_tables(attn_metadata, 1)
            row = worker._ctx_block_tables[0].tolist()
            return {
                row[ordinal] + layer
                for ordinal in _backed_ordinals(draft.kv_cache_map[req.py_request_id])
                for layer in range(num_layers)
            }

        occupant_slots = own_slots(occupant)
        request_slots = own_slots(request)
        assert occupant_slots and request_slots
        assert not occupant_slots & request_slots

        pool.fill_(7.0)
        before = pool.clone()
        captured = torch.randn(tokens_per_req, hidden, device="cuda")
        step_metadata = SimpleNamespace(
            hidden_size=hidden,
            runtime_draft_len=max_draft_len,
            captured_hidden_states=None,
            get_hidden_states=lambda num_tokens: captured[:num_tokens],
        )
        inputs = worker.prepare_1st_drafter_inputs(
            input_ids=torch.zeros(tokens_per_req, dtype=torch.long, device="cuda"),
            position_ids=torch.zeros(tokens_per_req, dtype=torch.long, device="cuda"),
            hidden_states=torch.zeros(tokens_per_req, hidden, device="cuda"),
            accepted_tokens=torch.ones((1, tokens_per_req), dtype=torch.long, device="cuda"),
            num_accepted_tokens=torch.tensor([tokens_per_req], dtype=torch.int32, device="cuda"),
            attn_metadata=attn_metadata,
            spec_metadata=step_metadata,
            draft_model=draft_model,
            total_target_tokens=tokens_per_req,
        )
        torch.cuda.synchronize()
        written = set((pool != before).flatten(1).any(dim=1).nonzero().flatten().tolist())
        assert written, "the drafter wrote nothing"
        assert not written & occupant_slots, "the drafter wrote another request's draft pages"
        assert written <= request_slots

        # The block decode reads num_ctx positions and appends the block right
        # after them, both through the table row it is handed.
        num_ctx = int(inputs["num_ctx_per_req"][0])
        read_row = inputs["ctx_page_table"][0].tolist()
        block = worker._compute_block_size
        addressed = {
            read_row[pos // page] + layer
            for pos in range(num_ctx + block)
            for layer in range(num_layers)
        }
        assert addressed <= request_slots, "the block decode addresses another request's pages"

        # The slot started empty on the page holding the request's absolute
        # position, every page below it hidden, and reads exactly the context
        # it stored this step.
        base = (request.max_beam_num_tokens - 1) // page
        slot = worker._req_to_slot[request.py_request_id]
        assert worker._ctx_seed_page_aligned
        assert worker._ctx_base_host[slot] == base
        assert worker._ctx_len_host[slot] == base * page
        assert num_ctx == tokens_per_req
        draft.free_resources(request)
        draft.free_resources(occupant)
    finally:
        draft.shutdown()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_slot_from_zero_worker_refuses_a_disagg_windowed_draft_pool(
    monkeypatch: pytest.MonkeyPatch, is_disagg: bool
) -> None:
    """A worker that numbers drafter slots from 0 (the Kimi K3 workers) binds a
    sliding-window draft pool, but not on a disaggregated worker, whose every
    generation slot starts past position 0."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.dflash.validate_dflash_trtllm_gen_runtime",
        lambda **kwargs: None,
    )
    spec = DFlashDecodingConfig(
        max_draft_len=3, speculative_model="draft", attention_backend="TRTLLM", mask_token_id=0
    )
    draft, _, _ = _disagg_draft_fixture(spec, 65536, 512, 256)
    draft.is_disagg = is_disagg
    try:
        worker = DFlashWorker(spec, Mapping())
        worker._ctx_reads_rebased_view = False
        draft_model = _PoolBoundDFlashDraftModel(2, 1, 64, 8)
        worker.set_draft_model(draft_model)
        num_pools = getattr(draft, "num_attention_op_pools", draft.num_pools)
        offsets = torch.zeros(
            (num_pools, 2, 2, draft.max_blocks_per_seq), dtype=torch.int32, device="cuda"
        )
        attn_metadata = SimpleNamespace(max_seq_len=1 << 17, draft_kv_cache_block_offsets=offsets)

        def bind() -> None:
            """Bind the drafter's ctx cache to the draft pool."""
            worker._lazy_init_ctx_buffers(
                draft_model,
                SimpleNamespace(max_num_requests=2),
                attn_metadata,
                draft_kv_cache_manager=draft,
            )

        if is_disagg:
            with pytest.raises(ValueError, match="numbers drafter slots from 0"):
                bind()
        else:
            bind()
            assert worker._ctx_block_tables is not None and worker._ctx_pool_windowed
    finally:
        draft.shutdown()


@pytest.mark.parametrize("lag", [2, 7, 31, 127])
def test_trailing_drafter_slot_reads_only_pages_its_request_holds(
    monkeypatch: pytest.MonkeyPatch, lag: int
) -> None:
    """A disaggregated generation worker seeds the drafter slot on the page
    holding the request's position, ``lag`` tokens behind it, while the
    request's sliding-window draft mirror advances its history with the
    request. Over many generation steps (one accepted token each), every
    block-table entry the drafter's window reads -- keys from the slot length
    + 1 - window, never below the slot's base -- must stay on a page the
    request's own mirror holds; a released entry resolves to another
    request's page."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.dflash.validate_dflash_trtllm_gen_runtime",
        lambda **kwargs: None,
    )
    num_layers, num_kv_heads, head_dim, hidden = 2, 1, 64, 8
    window, page, max_draft_len, steps = 256, 128, 3, 600
    spec = DFlashDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model="draft",
        attention_backend="TRTLLM",
        mask_token_id=0,
    )
    # The first token and two generation steps put the request at prompt + 2.
    prompt = 65536 + lag - 2
    draft, occupant, request = _disagg_draft_fixture(spec, prompt, 512, window, page)
    offsets = torch.zeros(
        (getattr(draft, "num_attention_op_pools", draft.num_pools), 2, 2, draft.max_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    try:
        for req in (occupant, request):
            batch = ScheduledRequests()
            batch.context_requests_last_chunk.append(req)
            draft._prepare_draft_resources(batch)
        request.add_new_token(1, 0)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        for _ in range(2):
            _disagg_generation_step(draft, request)

        worker = DFlashWorker(spec, Mapping())
        draft_model = _PoolBoundDFlashDraftModel(num_layers, num_kv_heads, head_dim, hidden)
        worker.set_draft_model(draft_model)
        attn_metadata = SimpleNamespace(
            max_seq_len=1 << 17,
            draft_kv_cache_block_offsets=offsets,
            num_contexts=0,
            num_seqs=1,
            num_ctx_tokens=0,
            _seq_lens_cuda=torch.zeros(1, dtype=torch.int32, device="cuda"),
            _seq_lens=torch.zeros(1, dtype=torch.int32),
        )
        worker._lazy_init_ctx_buffers(
            draft_model,
            SimpleNamespace(max_num_requests=2),
            attn_metadata,
            draft_kv_cache_manager=draft,
        )
        assert worker._ctx_page_size == page
        assert worker._ctx_seed_page_aligned
        spec_metadata = DFlashSpecMetadata(
            max_num_requests=2, max_draft_len=max_draft_len, max_total_draft_tokens=max_draft_len
        )
        spec_metadata._dflash_worker = worker
        spec_metadata.request_ids = [request.py_request_id]
        spec_metadata.num_generations = 1
        spec_metadata.seq_lens = [max_draft_len + 1]
        position = request.max_beam_num_tokens - 1
        spec_metadata.host_position_ids = torch.tensor(
            [position + offset for offset in range(max_draft_len + 1)], dtype=torch.int32
        )
        spec_metadata.prepare()
        slot = worker._req_to_slot[request.py_request_id]
        seed_len, base = int(worker._ctx_len_host[slot]), int(worker._ctx_base_host[slot])
        assert (base, position - seed_len) == (position // page, lag)

        cache = draft.kv_cache_map[request.py_request_id]
        unbacked_reads = []
        for step in range(steps):
            slot_len = seed_len + step + 1
            lowest = max(base * page, slot_len + 1 - window)
            read = {key // page for key in range(lowest, slot_len)}
            missing = read - _backed_ordinals(cache)
            if missing:
                unbacked_reads.append((step, cache.history_length, slot_len, sorted(missing)))
            _disagg_generation_step(draft, request)
        assert not unbacked_reads, (
            f"{len(unbacked_reads)}/{steps} steps read a released entry; first "
            f"(step, history, slot length, entries): {unbacked_reads[:3]}"
        )
        draft.free_resources(request)
        draft.free_resources(occupant)
    finally:
        draft.shutdown()


def _make_publishing_manager(policy: BlockReusePolicy) -> KVCacheManagerV2:
    """A manager wired just far enough to run the publish/history bookkeeping."""
    manager = object.__new__(KVCacheManagerV2)
    manager.kv_cache_type = CacheType.SELFKONLY
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.max_beam_width = 1
    manager._can_publish_block_reuse = True
    manager.block_reuse_policy = policy
    manager.kv_compression_manages_history = False
    manager.conversation_manager = None
    manager.kv_connector_manager = None
    manager.kv_cache_map = {}
    return manager


def _prefill(manager: KVCacheManagerV2, prompt: list[int], boundaries: list[int]):
    """Drive one request through *boundaries* and return its cache."""
    request = SimpleNamespace(
        py_request_id=7,
        is_dummy_request=False,
        prompt_len=len(prompt),
        is_context_only_request=False,
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
        is_context_only_request=False,
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
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager.kv_compression_manages_history = False
    manager.kv_connector_manager = None
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
    kv_cache_retention_config: KvCacheRetentionConfig | None = None
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    is_disagg_generation_init_state: bool = False
    is_generation_only_request: bool = False
    is_dummy_request: bool = False
    is_context_only_request: bool = False
    return_perf_metrics: bool = False
    context_current_position: int = 0
    py_served_prefix_end: int = 0
    py_num_connector_matched_tokens: int = 0
    py_connector_allocation_reported: bool = False
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


@pytest.mark.parametrize("config_cls", [DFlashDecodingConfig, PARDDecodingConfig])
def test_external_draft_estimated_quota_supports_allocation_and_resume(
    config_cls: type[DFlashDecodingConfig] | type[PARDDecodingConfig],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    spec = config_cls(max_draft_len=4, speculative_model="draft")
    model_config = SimpleNamespace(
        quant_config=None,
        pretrained_config=SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["full_attention"],
        ),
        get_num_attention_layers=lambda: 1,
    )
    batch_size, context_tokens, window = 64, 512, 504
    config = KvCacheConfig(enable_block_reuse=False, max_attention_window=[window])
    slope, fixed = KVCacheManagerV2.get_cache_size_per_token(
        model_config,
        Mapping(),
        tokens_per_block=32,
        max_seq_len=4096,
        max_batch_size=batch_size,
        max_num_tokens=context_tokens,
        kv_cache_config=config,
        spec_config=spec,
        is_draft=True,
    )
    config.max_gpu_total_bytes = slope * context_tokens + fixed
    manager = KVCacheManagerV2(
        config,
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        tokens_per_block=32,
        max_seq_len=4096,
        max_batch_size=batch_size,
        max_num_tokens=context_tokens,
        mapping=Mapping(),
        dtype=DataType.HALF,
        spec_config=spec,
        is_draft=True,
    )
    caches = []
    stream = torch.cuda.current_stream().cuda_stream
    try:
        for _ in range(batch_size):
            cache = manager.impl.create_kv_cache()
            caches.append(cache)
            assert cache.resume(stream)
        # One context request runs alongside a full generation batch minus one.
        assert caches[0].resize(context_tokens + spec.max_draft_len - 1)
        # Sweep a complete page so the workload exercises unaligned retention.
        for history in range(1024, 1056):
            capacity = history + spec.max_draft_len - 1 + spec.tokens_per_gen_step
            for cache in caches[1:]:
                assert cache.resize(capacity, history)
        # A quota that admits allocation must also allow requests to resume.
        for cache in caches:
            cache.suspend()
            assert cache.resume(stream)
    finally:
        for cache in caches:
            cache.close()
        manager.shutdown()


@pytest.mark.parametrize("draft_len", [0, 4])
def test_generation_dummy_uses_available_capacity(draft_len: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    spec_config = MTPDecodingConfig(max_draft_len=draft_len) if draft_len else None
    manager = KVCacheManagerV2(
        KvCacheConfig(enable_block_reuse=False, max_gpu_total_bytes=4 << 20),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=131072,
        max_batch_size=2,
        max_num_tokens=128,
        mapping=Mapping(),
        dtype=DataType.HALF,
        spec_config=spec_config,
    )
    try:
        requests = DummyRequestFactory(
            config=SimpleNamespace(
                spec_config=spec_config,
                max_draft_len=draft_len,
                max_draft_loop_tokens=draft_len,
                max_seq_len=manager.max_seq_len,
                max_beam_width=1,
                use_mrope=False,
            ),
            model=SimpleNamespace(
                model_config=SimpleNamespace(pretrained_config=SimpleNamespace())
            ),
            get_runtime_tokens_per_gen_step=lambda length: length + 1,
            release_request_state=None,
        )
        resources = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})
        batch = requests.create_cuda_graph_warmup_request(
            resources, batch_size=2, draft_len=draft_len
        )
        assert batch is not None
        assert len(batch.generation_requests) == 2
        longest_cache = manager.kv_cache_map[batch.generation_requests[0].py_request_id]
        assert longest_cache.capacity % manager.tokens_per_block == 0
        for request in batch.generation_requests:
            cache = manager.kv_cache_map[request.py_request_id]
            token_num = request.prompt_len + 1
            assert cache.history_length == request.prompt_len
            assert cache.capacity == token_num + manager.num_extra_kv_tokens + draft_len
            manager.free_resources(request)
        assert not manager.kv_cache_map
    finally:
        manager.shutdown()


def _adp_dummy_executor(
    cleanup: ExitStack,
    spec: DFlashDecodingConfig,
    *,
    joint_reuse: bool,
    transactional: bool,
) -> tuple[PyExecutor, dict[ResourceManagerType, KVCacheManagerV2]]:
    """An attention-DP executor over real target and draft KV pools.

    ``joint_reuse`` pairs the two pools (block reuse on both); without it the
    draft pool is an unpaired mirror of the target pool.
    """
    managers = {}
    for resource_type, is_draft in (
        (ResourceManagerType.KV_CACHE_MANAGER, False),
        (ResourceManagerType.DRAFT_KV_CACHE_MANAGER, True),
    ):
        manager = KVCacheManagerV2(
            KvCacheConfig(enable_block_reuse=joint_reuse, max_gpu_total_bytes=4 << 20),
            CacheType.SELF,
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            tokens_per_block=32,
            max_seq_len=128,
            max_batch_size=4,
            max_num_tokens=128,
            mapping=Mapping(),
            spec_config=spec,
            is_draft=is_draft,
            joint_kv_cache_reuse=joint_reuse,
        )
        cleanup.callback(manager.shutdown)
        managers[resource_type] = manager

    executor = object.__new__(PyExecutor)
    executor.resource_manager = ResourceManager(managers)
    executor.kv_cache_manager = managers[ResourceManagerType.KV_CACHE_MANAGER]
    executor.draft_kv_cache_manager = managers[ResourceManagerType.DRAFT_KV_CACHE_MANAGER]
    executor.scheduler = KVCacheV2Scheduler(
        max_batch_size=4,
        max_num_tokens=128,
        kv_cache_manager=executor.kv_cache_manager,
        draft_kv_cache_manager=executor.draft_kv_cache_manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )
    executor.enable_attention_dp = True
    # Two attention-DP ranks: a single-rank group skips dummy padding.
    executor.dist = SimpleNamespace(tp_size=2)
    executor.kv_cache_transceiver = object() if transactional else None
    executor.adp_router = SimpleNamespace(exclude_retiring_requests=False)
    executor.active_requests = []
    executor.expected_num_active_requests = 1
    executor.max_num_active_requests = 4
    executor.max_total_draft_tokens = spec.max_draft_len
    executor._adp_dummy_is_gen = True
    executor._pending_adp_dummy_request = None
    executor._enable_adp_dummy_fixes = True
    executor._enable_scheduler_aware_adp_dummy = False
    executor._enable_non_overlap_adp_forward_intent = False
    executor._benchmark_fill_phase_active = False
    executor._is_kv_manager_v2 = True
    executor.enable_joint_kv_cache_reuse = joint_reuse
    return executor, managers


@pytest.mark.parametrize("draft_len", [3, 8])
@pytest.mark.parametrize("path", ["plain", "transactional", "empty_batch"])
def test_adp_dummy_holds_paired_draft_kv(path: str, draft_len: int) -> None:
    """An idle attention-DP rank's dummy needs a cache in both paired pools.

    Under joint KV cache reuse the scheduler admits a generation request in
    the target and the draft pool together, so a dummy holding only a target
    cache is never scheduled and the fleet cannot run a forward. Each padding
    path must allocate the dummy in both pools, and a vetoed iteration must
    release both so the singleton dummy ID can be allocated again.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    spec = DFlashDecodingConfig(max_draft_len=draft_len, speculative_model="draft")
    with ExitStack() as cleanup:
        executor, managers = _adp_dummy_executor(
            cleanup, spec, joint_reuse=True, transactional=path == "transactional"
        )

        def assert_dummy_in_both_pools(dummy: LlmRequest) -> None:
            for manager in managers.values():
                assert manager.kv_cache_map[dummy.py_request_id].is_active

        def assert_dummy_released(dummy: LlmRequest) -> None:
            assert executor.active_requests == []
            for manager in managers.values():
                assert dummy.py_request_id not in manager.kv_cache_map
                assert dummy.py_request_id not in manager._allocated_draft_lens
                assert dummy.py_request_id not in manager._inflight_draft_lens

        def prepare_without_growing(scheduled: ScheduledRequests, dummy: LlmRequest) -> None:
            capacities = {
                key: manager.kv_cache_map[dummy.py_request_id].capacity
                for key, manager in managers.items()
            }
            executor.resource_manager.prepare_resources(scheduled)
            for key, manager in managers.items():
                assert manager.kv_cache_map[dummy.py_request_id].capacity == capacities[key]

        if path == "empty_batch":
            # The dummy joins the already scheduled batch directly; a vetoed
            # iteration rolls it back, and the next one pads again.
            for can_queue in (False, True):
                scheduled = ScheduledRequests()
                executor._pad_empty_attention_dp_batch(scheduled)
                (dummy,) = scheduled.generation_requests
                assert executor.active_requests == [dummy]
                assert_dummy_in_both_pools(dummy)
                executor._finalize_adp_dummy_allocation(can_queue)
                if not can_queue:
                    assert_dummy_released(dummy)
                else:
                    prepare_without_growing(scheduled, dummy)
            executor.resource_manager.free_resources(dummy)
            for manager in managers.values():
                assert dummy.py_request_id not in manager.kv_cache_map
            return

        # A rejected fleet vote must release both pools and permit the same
        # singleton dummy ID to be allocated and admitted on the next pass.
        for attempt in range(2 if path == "transactional" else 1):
            executor._pad_attention_dp_dummy_request()
            (dummy,) = executor.active_requests
            scheduled = executor.scheduler.schedule_request(executor.active_requests, set())
            assert scheduled.generation_requests == [dummy]
            assert scheduled.paused_requests == []
            assert_dummy_in_both_pools(dummy)
            if path == "transactional" and attempt == 0:
                executor._revert_gen_alloc(scheduled)
                executor._finalize_adp_dummy_allocation(False)
                assert_dummy_released(dummy)
            else:
                executor._finalize_adp_dummy_allocation(True)
                prepare_without_growing(scheduled, dummy)
                executor.resource_manager.free_resources(dummy)
                for manager in managers.values():
                    assert dummy.py_request_id not in manager.kv_cache_map


@pytest.mark.parametrize("path", ["plain", "transactional", "empty_batch"])
def test_adp_dummy_grows_unpaired_draft_mirror(path: str) -> None:
    """An unpaired draft pool sizes the attention-DP dummy's mirror in prepare_resources.

    The dummy is created in the target pool only. The scheduler admits the
    draft mirror while it schedules the dummy but leaves it empty, and the
    empty-batch path leaves no mirror at all, so the draft manager has to grow
    it by one generation step. An empty mirror cannot take the history of the
    previous iteration's finished dummy, which shares its ID and which the
    overlap loop applies once this iteration is prepared.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    spec = DFlashDecodingConfig(max_draft_len=8, speculative_model="draft")
    with ExitStack() as cleanup:
        executor, managers = _adp_dummy_executor(
            cleanup, spec, joint_reuse=False, transactional=path == "transactional"
        )
        draft = executor.draft_kv_cache_manager
        assert executor.scheduler._joint_draft_manager is None

        previous_batch = None
        for _ in range(2):
            batch = ScheduledRequests()
            if path == "empty_batch":
                executor._pad_empty_attention_dp_batch(batch)
                (dummy,) = batch.generation_requests
                assert dummy.py_request_id not in draft.kv_cache_map
            else:
                executor._pad_attention_dp_dummy_request()
                (dummy,) = executor.active_requests
                scheduled = executor.scheduler.schedule_request(executor.active_requests, set())
                assert scheduled.generation_requests == [dummy]
                batch.generation_requests.append(dummy)
                # Admitted by the scheduler, not sized.
                assert draft.kv_cache_map[dummy.py_request_id].capacity == 0
            executor._finalize_adp_dummy_allocation(True)

            executor.resource_manager.prepare_resources(batch)
            draft_cache = draft.kv_cache_map[dummy.py_request_id]
            assert draft_cache.capacity == draft._required_gen_capacity(dummy, 0)

            if previous_batch is not None:
                # _process_previous_batch: the deferred batch still holds the
                # finished dummy, whose ID now maps to this iteration's mirror.
                executor.resource_manager.update_resources(
                    previous_batch, skip_kv_cache_manager=True
                )
            # _terminate_attention_dp_dummy_requests.
            dummy.state = LlmRequestState.GENERATION_COMPLETE
            executor.resource_manager.free_resources(dummy)
            executor.active_requests.remove(dummy)
            previous_batch = batch

        for manager in managers.values():
            assert dummy.py_request_id not in manager.kv_cache_map


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "joint_reuse,is_dummy,mirror_capacity,grows",
    [
        # Paired pool: add_dummy_requests sized the dummy's mirror with the dummy.
        pytest.param(True, True, 24, False, id="paired-dummy-preallocated"),
        # Unpaired pool: the scheduler admitted the dummy's mirror, still empty.
        pytest.param(False, True, 0, True, id="unpaired-dummy-admitted-empty"),
        # Unpaired pool, empty-batch padding: the dummy has no mirror yet.
        pytest.param(False, True, None, True, id="unpaired-dummy-no-mirror"),
        # Real requests are a control: they grow by one step in either pool.
        pytest.param(True, False, 24, True, id="paired-request"),
        pytest.param(False, False, 24, True, id="unpaired-request"),
        pytest.param(False, False, None, True, id="unpaired-request-no-mirror"),
    ],
)
def test_draft_mirror_preserves_preallocated_dummy_capacity(
    joint_reuse: bool, is_dummy: bool, mirror_capacity: int | None, grows: bool
) -> None:
    """A paired pool's preallocated dummy fits one page; charging another step would need two.

    Every other mirror, an unpaired pool's dummy included, is sized here.
    """
    request = LlmRequest(
        request_id=100,
        max_new_tokens=1,
        input_tokens=[1] * 9,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=[1] * 8,
    )
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.is_dummy_request = is_dummy
    capacity = mirror_capacity or 0
    cache = Mock(capacity=capacity, is_active=True)
    # Dummies have only one 32-token page available. Real requests are a
    # control: unlike fixed-shape dummies they still need per-step growth.
    cache.resize.side_effect = lambda capacity: capacity <= (32 if is_dummy else 64)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_joint_kv_cache_reuse = joint_reuse
    manager._mirror_uses_draft_view = False
    manager._kv_reserve_draft_tokens = 8
    manager._allocated_draft_lens = {}
    manager.kv_cache_map = {} if mirror_capacity is None else {request.py_request_id: cache}
    manager._mirror_draft_kv_cache = Mock(return_value=cache)
    manager._resume_and_restore = Mock(return_value=True)
    manager._ensure_generation_beam_width = Mock(return_value=True)
    manager._mirror_resize_error = Mock(return_value=RuntimeError("draft pool exhausted"))
    batch = ScheduledRequests()
    batch.generation_requests.append(request)

    manager._prepare_draft_resources(batch)

    manager._resume_and_restore.assert_called_once_with(request.py_request_id, cache)
    if grows:
        # One generation step: the token itself and the eight draft slots.
        cache.resize.assert_called_once_with(capacity + 9)
    else:
        cache.resize.assert_not_called()
    # Recognizing a preallocated dummy must not fabricate scheduler rollback
    # bookkeeping; the dummy is freed as a whole after the iteration.
    assert manager._allocated_draft_lens == {}


def test_generation_dummy_does_not_reuse_real_request_prefix() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = KVCacheManagerV2(
        KvCacheConfig(enable_block_reuse=True, max_gpu_total_bytes=4 << 20),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=2048,
        max_batch_size=2,
        max_num_tokens=128,
        mapping=Mapping(),
        dtype=DataType.HALF,
    )
    tokens = [1] * 32
    seed = manager.impl.create_kv_cache(ReuseScope(), tokens)
    dummy = None
    try:
        assert seed.resume(manager._stream.cuda_stream)
        assert seed.resize(len(tokens))
        seed.commit(tokens, is_end=True)
        seed.close()

        dummy = manager.add_dummy_requests([1000], token_nums=[33], is_gen=True)
        assert dummy is not None
        assert manager.kv_cache_map[dummy[0].py_request_id].num_committed_tokens == 0
    finally:
        if dummy is not None:
            manager.free_resources(dummy[0])
        manager.shutdown()


@pytest.mark.parametrize(
    "global_prompt,local_prompt,rank,range_end,expected",
    [(256, 128, 0, 128, [35, 80]), (64, 0, 1, 128, [10]), (130, 66, 0, 65, [35, 80])],
)
def test_helix_retention_uses_rank_local_ranges(
    monkeypatch: pytest.MonkeyPatch,
    global_prompt: int,
    local_prompt: int,
    rank: int,
    range_end: int,
    expected: list[int],
) -> None:
    # Only cross-process coordination is replaced; allocation and policy run in native V2 on GPU.
    dist = SimpleNamespace(allreduce=lambda value, op: value, local_world_size=1)
    monkeypatch.setattr(Distributed, "get", lambda mapping: dist)
    manager = KVCacheManagerV2(
        KvCacheConfig(enable_block_reuse=False, max_gpu_total_bytes=4 << 20),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=64,
        max_seq_len=512,
        max_batch_size=1,
        max_num_tokens=256,
        mapping=Mapping(
            world_size=2, cp_size=2, tp_size=1, rank=rank, cp_config={"cp_type": "HELIX"}
        ),
        dtype=DataType.HALF,
        is_disagg=True,
    )
    request = _ContextRequest(
        1, list(range(local_prompt)), local_prompt, "retention", use_conversation_params=False
    )
    request.total_input_len_cp = global_prompt
    request.kv_cache_retention_config = KvCacheRetentionConfig(
        [KvCacheRetentionConfig.TokenRangeRetentionConfig(64, range_end, 80)],
        decode_retention_priority=10,
    )
    try:
        assert isinstance(manager.impl, KVCacheManager)
        assert manager.impl.init_config.tokens_per_block == 128
        assert manager.prepare_context_cache(request) is not None
        cache = manager.kv_cache_map[request.py_request_id]
        assert cache.resize(global_prompt)
        group = manager.impl.get_layer_group_id(0)
        assert cache.get_page_priorities(group) == expected
        assert cache.resize((len(expected) + 1) * 128)
        assert cache.get_page_priorities(group) == expected + [10]
    finally:
        manager.free_resources(request)
        manager.shutdown()


@pytest.mark.parametrize(
    "range_start,range_end,expected", [(0, 128, [80, 35]), (128, 256, [35, 80])]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_agg_helix_retention_uses_global_ranges(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    range_start: int,
    range_end: int,
    expected: list[int],
) -> None:
    # Aggregated helix holds the full prompt on every rank, so a range in prompt
    # tokens marks exactly the ledger blocks (two 64-token pages each) it covers.
    dist = SimpleNamespace(allreduce=lambda value, op: value, local_world_size=1)
    monkeypatch.setattr(Distributed, "get", lambda mapping: dist)
    manager = KVCacheManagerV2(
        KvCacheConfig(enable_block_reuse=False, max_gpu_total_bytes=4 << 20),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=64,
        max_seq_len=512,
        max_batch_size=1,
        max_num_tokens=256,
        mapping=Mapping(
            world_size=2, cp_size=2, tp_size=1, rank=rank, cp_config={"cp_type": "HELIX"}
        ),
        dtype=DataType.HALF,
        is_disagg=False,
    )
    prompt_len = 256
    request = _ContextRequest(
        1, list(range(prompt_len)), prompt_len, "retention", use_conversation_params=False
    )
    request.total_input_len_cp = prompt_len
    request.kv_cache_retention_config = KvCacheRetentionConfig(
        [KvCacheRetentionConfig.TokenRangeRetentionConfig(range_start, range_end, 80)],
        decode_retention_priority=10,
    )
    try:
        assert manager.prepare_context_cache(request) is not None
        cache = manager.kv_cache_map[request.py_request_id]
        assert cache.resize(prompt_len)
        group = manager.impl.get_layer_group_id(0)
        assert cache.get_page_priorities(group) == expected
        assert cache.resize((len(expected) + 1) * 128)
        assert cache.get_page_priorities(group) == expected + [10]
    finally:
        manager.free_resources(request)
        manager.shutdown()


@pytest.fixture
def max_num_turns() -> int:
    return 1


@pytest.fixture
def manager(max_num_turns: int) -> KVCacheManagerV2:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
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


def _try_run_context(manager: KVCacheManagerV2, request: _ContextRequest) -> bool:
    """Run a context request all the way through, reporting whether it fit."""
    batch = _prepare_context_resources(manager, request)
    if not manager.prepare_context(request):
        return False
    request.context_remaining_length = request.prompt_len - request.context_current_position
    if not manager.resize_context(request, num_tokens=request.context_remaining_length):
        return False
    request.context_current_position = request.prompt_len
    request.context_remaining_length = 0
    _update_context_resources(manager, batch)
    return True


def test_context_larger_than_kv_pool_returns_false_and_can_retry() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    max_seq_len = 8192
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            host_cache_size=0,
            max_gpu_total_bytes=4 << 20,
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        max_num_tokens=max_seq_len,
        mapping=Mapping(),
        dtype=DataType.HALF,
    )
    request = _ContextRequest(1, list(range(max_seq_len)), max_seq_len, "small-pool")
    try:
        assert 0 < manager.max_seq_len < max_seq_len
        assert manager.prepare_context(request)
        cache = manager.kv_cache_map[request.py_request_id]
        capacity, history = cache.capacity, cache.history_length

        assert not manager.resize_context(request, max_seq_len)
        assert (cache.capacity, cache.history_length) == (capacity, history)
        assert not cache.is_active

        assert manager.prepare_context(request)
        assert manager.resize_context(request, manager.tokens_per_block)
        assert cache.capacity == manager.tokens_per_block + manager.num_extra_kv_tokens
    finally:
        manager.free_resources(request)
        manager.shutdown()


def test_preempt_request_gives_a_full_pool_back_its_pages(
    manager: KVCacheManagerV2,
) -> None:
    """Preemption is the only way out of a full GPU-only pool.

    With GPU as the last cache level a suspended page stays HELD, which the
    eviction controller refuses to move, so suspension frees nothing. This
    covers the release itself: the pages returning, and the connector's
    matched-token count dropping.
    """
    victim = _ContextRequest(1, list(range(MAX_SEQ_LEN)), MAX_SEQ_LEN, "conv-1")
    victim.py_num_connector_matched_tokens = TOKENS_PER_BLOCK
    started: list[_ContextRequest] = [victim]

    try:
        assert _try_run_context(manager, victim)
        assert manager.is_request_active(victim.py_request_id)

        # The pool's capacity in requests follows from the fixture's layout and
        # windows, so find it rather than hard-code it. Distinct tokens per
        # request keep block reuse from hiding the pressure.
        blocked = None
        for request_id in range(2, 12):
            candidate = _ContextRequest(
                request_id,
                [request_id * 1000 + i for i in range(MAX_SEQ_LEN)],
                MAX_SEQ_LEN,
                f"conv-{request_id}",
            )
            started.append(candidate)
            if not _try_run_context(manager, candidate):
                blocked = candidate
                break
        assert blocked is not None, "the pool never filled, so there is nothing to preempt for"

        # Its own partial allocation goes back first, as the scheduler's retry
        # does. What is missing is the victim's pages.
        manager.free_resources(blocked)

        assert manager.preempt_request(victim) is True
        assert not manager.is_request_active(victim.py_request_id)
        assert victim.py_request_id not in manager.kv_cache_map
        # A replay recomputes from its own reuse match, so a stale count would
        # skip a prefix the pool no longer holds for it.
        assert victim.py_num_connector_matched_tokens == 0

        blocked.context_current_position = 0
        assert _try_run_context(manager, blocked)
    finally:
        for request in started:
            _free_if_active(manager, request)


def test_preemption_waits_for_a_connector_still_reading_the_victims_pages(
    manager: KVCacheManagerV2,
) -> None:
    """A connector save in flight defers the release rather than racing it.

    The pages are the source of those reads, so handing them to another request
    now would let it overwrite the bytes mid-transfer and publish them under a
    hash that says they are the victim's prefix.
    """
    victim = _ContextRequest(1, list(range(MAX_SEQ_LEN)), MAX_SEQ_LEN, "conv-1")
    victim.py_num_connector_matched_tokens = TOKENS_PER_BLOCK

    # Reports a save in flight for every request, which is the only answer the
    # deferral depends on; the rest is what free_resources reaches for.
    connector = SimpleNamespace(
        request_finished=lambda *_args: True,
        capacity_only=False,
        prefix_reservations_enabled=False,
        # Declining keeps the allocation hooks out of a preemption test.
        should_add_sequence=lambda _req: False,
        release_unstarted_prefix_loads=lambda _req: None,
        has_pending_load=lambda _req: False,
        release_prefix_reservation=lambda _req: None,
        reset_request_state=lambda _req: None,
    )
    manager.kv_connector_manager = connector
    try:
        assert _try_run_context(manager, victim)

        assert manager.preempt_request(victim) is False
        assert manager.has_pending_preemption()
        assert manager.is_preemption_pending(victim)
        # Nothing was given up yet: the victim still owns its cache.
        assert manager.is_request_active(victim.py_request_id)
        assert victim.py_num_connector_matched_tokens == TOKENS_PER_BLOCK

        assert manager.try_complete_preemption(victim) is True
        assert not manager.has_pending_preemption()
        assert not manager.is_preemption_pending(victim)
        assert not manager.is_request_active(victim.py_request_id)
        assert victim.py_request_id not in manager.kv_cache_map
        assert victim.py_num_connector_matched_tokens == 0

        # How the executor tells a preempted request apart from a finished one
        # in the connector's get_finished output, so a second call must not
        # claim this one again.
        assert manager.try_complete_preemption(victim) is False
    finally:
        manager.kv_connector_manager = None
        _free_if_active(manager, victim)


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
        probe_kv_cache = manager.kv_cache_map[request_a_probe.py_request_id]
        assert list(probe_kv_cache.cached_tokens_by_level)[0] == 7
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


def _make_spec_recompute_manager(
    policy: str,
    context_recompute_tail: int,
    manager_cls: type[KVCacheManagerV2] = KVCacheManagerV2,
    max_attention_window: list[int] | None = None,
) -> KVCacheManagerV2:
    """A real target manager with a DFlash spec config carrying the tail."""
    spec = DFlashDecodingConfig(
        max_draft_len=4,
        speculative_model="draft",
        context_recompute_tail=context_recompute_tail,
    )
    return manager_cls(
        KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=True,
            max_gpu_total_bytes=16 << 20,
            max_attention_window=max_attention_window or [MAX_SEQ_LEN],
            max_util_for_resume=1.0,
            block_reuse_config=BlockReuseConfig(policy=policy),
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=spec,
        vocab_size=4096,
        enable_stats=False,
    )


def _drive_context_chunks(
    manager: KVCacheManagerV2,
    batch: ScheduledRequests,
    request: _ContextRequest,
) -> None:
    """Finish *request* in single-block chunks, updating resources per chunk."""
    position = request.context_current_position
    while position < request.prompt_len:
        chunk = min(TOKENS_PER_BLOCK, request.prompt_len - position)
        assert manager.resize_context(request, num_tokens=chunk)
        position += chunk
        request.context_current_position = position
        request.context_remaining_length = request.prompt_len - position
        _update_context_resources(manager, batch)
        request.is_first_context_chunk = False


@pytest.mark.parametrize("policy", ["per_request", "per_conversation"])
def test_deferred_commit_policies_disable_the_spec_recompute_tail(policy: str) -> None:
    """The recompute tail is enforced by capping the fresh radix claim, and
    per_conversation reuse resumes the prior turn's cache without a claim to
    cap; the deferred-commit protocols are unvalidated with the tail. The
    manager disables the recompute tail for these policies at construction; a
    chunked cache-hit request must then run start to finish with its cursor
    held at the matched prefix.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = _make_spec_recompute_manager(policy, context_recompute_tail=-1)
    prompt = list(range(MAX_SEQ_LEN))
    request_a = _ContextRequest(1, prompt, MAX_SEQ_LEN, "conv-1", use_conversation_params=False)
    request_b = _ContextRequest(2, prompt, MAX_SEQ_LEN, "conv-2", use_conversation_params=False)
    try:
        assert manager._spec_recompute_tail == 0

        _run_context(manager, request_a)
        assert manager.kv_cache_map[request_a.py_request_id].num_committed_tokens > 0

        batch = _prepare_context_resources(manager, request_b)
        assert manager.prepare_context(request_b)
        matched = request_b.prepopulated_prompt_len
        assert matched > 0, "the second identical prompt must take a prefix hit"
        # No rewind: the cursor stays at the matched prefix, so no later chunk
        # can ask the history marker to decrease.
        assert request_b.context_current_position == matched

        _drive_context_chunks(manager, batch, request_b)
        assert manager.kv_cache_map[request_b.py_request_id].num_committed_tokens == MAX_SEQ_LEN
    finally:
        _free_if_active(manager, request_b)
        _free_if_active(manager, request_a)
        manager.shutdown()


def test_all_reusable_policy_recomputes_the_tail_across_chunks() -> None:
    """Under ALL_REUSABLE the recompute tail stays enabled: a cache-hit
    request's reuse claim is capped (here to a full re-prefill, which claims
    nothing), the cursor starts at the claim, and chunked prefill completes
    and commits the recomputed blocks.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = _make_spec_recompute_manager("all_reusable", context_recompute_tail=-1)
    prompt = list(range(MAX_SEQ_LEN))
    request_a = _ContextRequest(1, prompt, MAX_SEQ_LEN, "conv-1", use_conversation_params=False)
    request_b = _ContextRequest(2, prompt, MAX_SEQ_LEN, "conv-2", use_conversation_params=False)
    try:
        assert manager._spec_recompute_tail == -1

        _run_context(manager, request_a)

        batch = _prepare_context_resources(manager, request_b)
        assert manager.prepare_context(request_b)
        # A full re-prefill claims no reuse, so every prompt token passes the
        # target forward and every page it touches is freshly allocated.
        assert manager.kv_cache_map[request_b.py_request_id].num_committed_tokens == 0
        assert request_b.context_current_position == 0
        assert request_b.prepopulated_prompt_len == 0

        _drive_context_chunks(manager, batch, request_b)
        assert manager.kv_cache_map[request_b.py_request_id].num_committed_tokens == MAX_SEQ_LEN
    finally:
        _free_if_active(manager, request_b)
        _free_if_active(manager, request_a)
        manager.shutdown()


def test_windowed_layers_keep_the_recompute_tail_claim_capped() -> None:
    """Sliding-window regression for the claim cap: a match is claimed only up
    to the recompute boundary, so the pages the recomputed span needs (the
    window at the CLAIMED endpoint) are materialized by the core. The old
    rewind approach claimed the full match and walked the cursor back, where a
    windowed life cycle has no pages behind the out-of-window span.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    window = 2 * TOKENS_PER_BLOCK  # 8 < MAX_SEQ_LEN: a real sliding window
    manager = _make_spec_recompute_manager(
        "all_reusable",
        context_recompute_tail=6,
        max_attention_window=[window],
    )
    prompt = list(range(MAX_SEQ_LEN))
    request_a = _ContextRequest(1, prompt, MAX_SEQ_LEN, "conv-1", use_conversation_params=False)
    request_b = _ContextRequest(2, prompt, MAX_SEQ_LEN, "conv-2", use_conversation_params=False)
    try:
        assert manager._spec_recompute_tail == 6

        _run_context(manager, request_a)

        batch = _prepare_context_resources(manager, request_b)
        assert manager.prepare_context(request_b)
        committed = manager.kv_cache_map[request_b.py_request_id].num_committed_tokens
        # floor_block(16 - 6) = 8: the claim stops at the recompute boundary
        # (and the window at that endpoint is fully materialized), instead of
        # claiming the full match and rewinding below its live window.
        assert committed == 8
        assert request_b.context_current_position == committed
        assert request_b.prepopulated_prompt_len == committed

        _drive_context_chunks(manager, batch, request_b)
        assert manager.kv_cache_map[request_b.py_request_id].num_committed_tokens == MAX_SEQ_LEN
    finally:
        _free_if_active(manager, request_b)
        _free_if_active(manager, request_a)
        manager.shutdown()


@pytest.mark.parametrize("has_recurrent_state,expected_tail", [(False, 6), (True, -1)])
def test_recurrent_state_managers_allow_only_a_full_reprefill_tail(
    has_recurrent_state: bool, expected_tail: int
) -> None:
    """Managers with recurrent state coerce a positive tail to a full
    re-prefill; attention-only managers keep it. Conservative: a capped claim
    is equivalent to a shorter match, but the hybrid snapshot/commit protocol
    is unvalidated with a partial tail.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()

    class _RecurrentManager(KVCacheManagerV2):
        _has_recurrent_state = True

    manager_cls = _RecurrentManager if has_recurrent_state else KVCacheManagerV2
    manager = _make_spec_recompute_manager(
        "all_reusable", context_recompute_tail=6, manager_cls=manager_cls
    )
    try:
        assert manager._spec_recompute_tail == expected_tail
    finally:
        manager.shutdown()


def test_live_storage_stats_use_the_manager_api() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=8 << 20), HostCacheTierConfig(quota=8 << 20)],
            layers=[
                AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=4096)])
            ],
        )
    )
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = Mock(wraps=core, cache_tier_list=core.cache_tier_list)
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager._storage_pool_groups_by_window = lambda: {}
    manager._cold_pool_group_membership_cache = None
    cache = core.create_kv_cache()
    try:
        assert manager._cold_pool_group_membership() == ((0, frozenset({0})),)
        before = manager.get_kv_cache_stats()
        cache.resume(CudaStream(torch.cuda.Stream().cuda_stream))
        cache.resize(TOKENS_PER_BLOCK)
        after = manager.get_kv_cache_stats()
        manager.impl.get_life_cycle_pool_group_indices.assert_called_once_with(CacheLevel(1))
        assert manager.impl.get_storage_statistics.call_args_list == [
            call(CacheLevel(0)),
            call(CacheLevel(0)),
        ]
        assert before.used_num_blocks == 0
        assert after.used_num_blocks == 1
        assert after.free_num_blocks == before.free_num_blocks - 1
        assert after.max_num_blocks == before.max_num_blocks
    finally:
        cache.close()
        core.shutdown()


@pytest.mark.parametrize(
    "kv_connector_manager, expected_connector_cached_tokens",
    [(None, None), (SimpleNamespace(capacity_only=False), 9)],
    ids=["no_connector", "serving_connector"],
)
def test_iteration_stats_reports_physical_pool_groups_without_window_metadata(
    kv_connector_manager, expected_connector_cached_tokens
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_stats = True
    manager.is_draft = False
    manager.kv_connector_manager = kv_connector_manager
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
        get_and_reset_iteration_disk_prefetch_blocks=lambda: 7,
        get_and_reset_iteration_cached_tokens_by_level=lambda: [5, 2, 1],
        get_and_reset_iteration_connector_cached_tokens=lambda: 9,
        get_and_reset_iteration_reused_blocks_by_level=lambda: {},
        get_and_reset_iteration_connector_reused_blocks=lambda: {},
    )
    manager._stats_life_cycle_metadata = lambda: {3: (1, None, "ssm")}
    manager._storage_pool_groups_by_window = lambda: {}
    manager._get_and_reset_iteration_peak_block_stats_by_level = lambda num_levels: [
        [None, None] for _ in range(num_levels)
    ]
    manager._get_storage_statistics = lambda _level: [object(), object()]
    manager._build_pool_group_iteration_stats = lambda pool_group_id, *_args: pool_group_id

    stats = manager.get_iteration_stats()

    assert stats.by_pool_group == {0: 0, 1: 1}
    ssm_stats = stats.by_life_cycle[3]
    assert ssm_stats.kind == "ssm"
    assert ssm_stats.pool_group_id == 1
    assert ssm_stats.snapshot_stats.iter_snapshot_hit_rate == 0.5
    assert ssm_stats.snapshot_stats.iter_reused_tokens == 32
    assert stats.disk_prefetch_blocks == 7
    assert stats.cached_tokens_by_level == [5, 2, 1]
    assert stats.connector_cached_tokens == expected_connector_cached_tokens


def _make_admission_manager(
    *,
    is_estimating_kv_cache: bool = False,
    overwrites_whole_cached_prefix: bool = False,
) -> tuple[KVCacheManagerV2, SimpleNamespace]:
    """Partial manager whose kv_cache_map already holds a cache for request 1.

    Exercises the cached-token attribution branch of prepare_context_cache without a GPU. The
    attribution itself is staged inside the core; the manager only decides when to drop it.
    """
    manager = object.__new__(KVCacheManagerV2)
    manager.conversation_manager = None
    manager.kv_connector_manager = None
    manager.enable_block_reuse = True
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.is_draft = False
    manager.kv_cache_type = CacheType.SELF
    manager.is_estimating_kv_cache = is_estimating_kv_cache
    manager._spec_recompute_tail = 0
    manager._disagg_transfer_overwrites_whole_cached_prefix = lambda: overwrites_whole_cached_prefix
    manager._allocation_stream = None
    manager._on_allocation_stream = set()
    manager._allocation_ready = {}
    manager._resume_and_restore = lambda _req_id, _kv_cache: True
    kv_cache = SimpleNamespace(
        is_active=False,
        num_committed_tokens=7,
        enable_swa_scratch_reuse=True,
        drop_cached_token_attribution=Mock(),
        drop_partial_block_cached_token_attribution=Mock(),
    )
    manager.kv_cache_map = {1: kv_cache}
    return manager, kv_cache


def test_prepare_context_keeps_cached_token_attribution_staged_by_the_core() -> None:
    manager, kv_cache = _make_admission_manager()
    request = _ContextRequest(1, list(range(8)), 8, "conv-1")

    assert manager.prepare_context(request)

    kv_cache.drop_cached_token_attribution.assert_not_called()
    kv_cache.drop_partial_block_cached_token_attribution.assert_not_called()
    assert request.prepopulated_prompt_len == 7


def test_prepare_context_drops_cached_token_attribution_while_estimating() -> None:
    manager, kv_cache = _make_admission_manager(is_estimating_kv_cache=True)
    request = _ContextRequest(1, list(range(8)), 8, "conv-1")

    assert manager.prepare_context(request)

    kv_cache.drop_cached_token_attribution.assert_called_once_with()
    kv_cache.drop_partial_block_cached_token_attribution.assert_not_called()


def test_disagg_gen_init_drops_only_the_partial_block() -> None:
    manager, kv_cache = _make_admission_manager()
    request = _ContextRequest(1, list(range(8)), 8, "conv-1", is_disagg_generation_init_state=True)

    assert manager.prepare_context_cache(request) == kv_cache.num_committed_tokens

    kv_cache.drop_partial_block_cached_token_attribution.assert_called_once_with()
    kv_cache.drop_cached_token_attribution.assert_not_called()


@pytest.mark.parametrize("enable_stats", [True, False])
def test_disagg_partial_attribution_survives_admission_retry(enable_stats: bool) -> None:
    """A failed resume preserves the cache, so retrying must not exclude its tail again."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    stream = torch.cuda.Stream()
    core = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=8 << 20)],
            layers=[
                AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=4096)])
            ],
            enable_stats=enable_stats,
            enable_partial_reuse=True,
            max_util_for_resume=0.5,
        )
    )
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    seed = core.create_kv_cache(None, tokens)
    blocker = core.create_kv_cache()
    kv_cache = None
    try:
        assert seed.resume(CudaStream(stream.cuda_stream))
        assert seed.resize(len(tokens))
        seed.commit(tokens, is_end=True)
        seed.close()

        kv_cache = core.create_kv_cache(None, tokens[:-1])
        assert kv_cache.num_committed_tokens == len(tokens) - 1
        initial_attribution = list(kv_cache.cached_tokens_by_level)
        assert initial_attribution == [len(tokens) - 1]

        manager, _ = _make_admission_manager()
        manager.kv_cache_map = {1: kv_cache}
        manager._stream = stream
        # Use the real resume/utilization check. Output-index restoration is unrelated to admission.
        del manager._resume_and_restore
        manager._restore_page_index_bufs = Mock()
        request = _ContextRequest(
            1, tokens, len(tokens), "conv-1", is_disagg_generation_init_state=True
        )

        assert blocker.resume(CudaStream(stream.cuda_stream))
        total_slots = core.get_storage_statistics()[0].total
        assert blocker.resize((total_slots // 2 + 1) * TOKENS_PER_BLOCK)
        assert manager.prepare_context_cache(request) is None
        assert manager.kv_cache_map[1] is kv_cache
        assert not kv_cache.is_active
        manager._restore_page_index_bufs.assert_not_called()

        blocker.close()
        assert manager.prepare_context_cache(request) == len(tokens) - 1
        manager._restore_page_index_bufs.assert_called_once_with(1, kv_cache)
        assert list(kv_cache.cached_tokens_by_level) == initial_attribution

        kv_cache.commit_pending_stats()
        expected = [2 * TOKENS_PER_BLOCK] if enable_stats else []
        assert list(core.get_and_reset_iteration_cached_tokens_by_level()) == expected
        # Once attribution is committed, a later drop must not recreate it.
        kv_cache.drop_partial_block_cached_token_attribution()
        kv_cache.commit_pending_stats()
        assert list(core.get_and_reset_iteration_cached_tokens_by_level()) == []
    finally:
        if kv_cache is not None:
            kv_cache.close()
        blocker.close()
        seed.close()
        core.shutdown()


def test_disagg_gen_init_drops_everything_when_the_transfer_overwrites_the_prefix() -> None:
    manager, kv_cache = _make_admission_manager(overwrites_whole_cached_prefix=True)
    request = _ContextRequest(1, list(range(8)), 8, "conv-1", is_disagg_generation_init_state=True)

    assert manager.prepare_context_cache(request) == kv_cache.num_committed_tokens

    kv_cache.drop_cached_token_attribution.assert_called_once_with()
    kv_cache.drop_partial_block_cached_token_attribution.assert_not_called()


def test_disagg_gen_init_drops_everything_in_gen_only_benchmark_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
    manager, kv_cache = _make_admission_manager()
    request = _ContextRequest(1, list(range(8)), 8, "conv-1", is_disagg_generation_init_state=True)

    assert manager.prepare_context_cache(request) == kv_cache.num_committed_tokens

    kv_cache.drop_cached_token_attribution.assert_called_once_with()
    kv_cache.drop_partial_block_cached_token_attribution.assert_not_called()


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


def _index_mapper_capacity_for(
    *,
    max_batch_size: int,
    pp_size: int = 1,
    is_disagg: bool = False,
    num_reserved_index_slots: int = 1,
    disable_overlap_scheduler: bool = True,
    sparse: bool = False,
    guard_page: bool = False,
) -> tuple[int, int, int]:
    """Construct a manager and return the three sizes it derives.

    ``(IndexMapper capacity, page-table capacity, max_admissible_sequences)``.

    The first two must agree: ``host_kv_cache_block_offsets`` is indexed by the
    index the mapper hands out, so a page table sized below the mapper's capacity
    would be an out-of-bounds write. The third is the published admission bound
    that ``validate_seq_slot_pool_covers_admission`` checks the seat pool against
    at startup, and it is the capacity minus the reserved dummy slots.
    """
    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
    fake_impl = Mock()
    fake_impl.buffer_grouping = [[BufferId(LayerId(0), Role.KEY)]]
    fake_impl.pool_group_descs = []
    fake_impl.get_layer_group_id.side_effect = lambda _: 0

    def build_base_config(
        self: KVCacheManagerV2,
        config: KvCacheConfig,
        *,
        tokens_per_block: int,
        cache_tiers: list[object],
    ) -> KVCacheManagerConfig:
        del self, config
        return KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=cache_tiers,
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=128, is_sparse=sparse),
                        BufferConfig(role=Role.VALUE, size=128),
                    ],
                    sparse_cache=SparseCacheConfig(
                        num_selected_blocks=1, max_cached_units=2, unit_tokens=tokens_per_block
                    )
                    if sparse
                    else None,
                )
            ],
        )

    with (
        patch.dict(
            os.environ,
            {"TRTLLM_KV_GUARD_PAGE": "1" if guard_page else "", "TRTLLM_KV_FRESH_PAGE_FILL": ""},
        ),
        patch(f"{module}.IndexMapper") as index_mapper_cls,
        patch(f"{module}.KVCacheManagerPy", return_value=fake_impl) as create_impl,
        patch.object(KVCacheManagerV2, "_build_base_config", build_base_config),
        patch.object(KVCacheManagerV2, "_build_cache_config", lambda self, config: config),
        patch.object(KVCacheManagerV2, "get_num_available_tokens", return_value=MAX_SEQ_LEN),
        patch.object(KVCacheManagerV2, "_prepare_page_table_tensor") as page_table,
        patch.object(KVCacheManagerV2, "_log_kv_cache_pool_lifecycle_mapping"),
        patch.object(KVCacheManagerV2, "_reserve_guard_page"),
    ):
        manager = KVCacheManagerV2(
            # A quota must be set or __init__ asserts; the value is irrelevant here.
            KvCacheConfig(max_gpu_total_bytes=16 << 20, enable_block_reuse=False),
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=Mapping(
                world_size=pp_size,
                rank=0,
                tp_size=1,
                pp_size=pp_size,
            ),
            dtype=DataType.HALF,
            vocab_size=16,
            execution_stream=Mock(),
            is_disagg=is_disagg,
            num_reserved_index_slots=num_reserved_index_slots,
            disable_overlap_scheduler=disable_overlap_scheduler,
            sparse_kv_offload_config=SparseKvCacheOffloadConfig() if sparse else None,
        )
    index_mapper_cls.assert_called_once()
    page_table.assert_called_once()
    config = create_impl.call_args.args[0]
    # The sparse arenas are the Batch's and take their lease count from its rows, so
    # nothing encodes a request count as a constraint any more.
    assert not config.constraints
    return (
        index_mapper_cls.call_args.args[0],
        page_table.call_args.args[0],
        manager.max_admissible_sequences,
    )


# (max_batch_size, pp_size, disable_overlap_scheduler, is_disagg, reserved, expected)
#
# capacity == max_batch_size * pp_size
#             * (2 if is_disagg or (overlap on and pp_size == 1) else 1)
#             + reserved
_INDEX_MAPPER_CAPACITY_CASES = [
    # Overlap on, no PP: both cohorts are resident, so the mapper needs 2B.
    pytest.param(2, 1, False, False, 1, 5, id="overlap_on"),
    pytest.param(8, 1, False, False, 1, 17, id="overlap_on_b8"),
    pytest.param(2, 1, True, False, 1, 3, id="overlap_off"),
    pytest.param(2, 1, False, True, 1, 5, id="disagg_does_not_compound"),
    pytest.param(2, 1, True, True, 1, 5, id="disagg_only"),
    pytest.param(2, 4, True, False, 1, 9, id="pp4_plain"),
    pytest.param(2, 4, False, False, 1, 9, id="pp4_overlap_excluded"),
    # ... while the pre-existing disagg 2x under PP is left exactly as it was.
    pytest.param(2, 4, True, True, 1, 17, id="pp4_disagg_unchanged"),
    # Reserved slots are still added on top of the widened pool.
    pytest.param(2, 1, False, False, 5, 9, id="reserved_slots_still_added"),
    pytest.param(1, 1, True, False, 1, 2, id="single_live_request_and_padding"),
    pytest.param(2, 1, True, False, 0, 2, id="no_reserved_slots"),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "max_batch_size,pp_size,disable_overlap_scheduler,is_disagg,reserved,expected,sparse",
    [
        pytest.param(*case.values, sparse, id=f"{case.id}-{sparse}")
        for case in _INDEX_MAPPER_CAPACITY_CASES
        for sparse in (False, True)
        # Sparse offload rejects pipeline parallelism.
        if not (sparse and case.values[1] > 1)
    ],
)
def test_index_mapper_capacity_covers_the_overlapping_cohorts(
    max_batch_size: int,
    pp_size: int,
    disable_overlap_scheduler: bool,
    is_disagg: bool,
    reserved: int,
    expected: int,
    sparse: bool,
) -> None:
    capacity, page_table_capacity, max_admissible_sequences = _index_mapper_capacity_for(
        max_batch_size=max_batch_size,
        pp_size=pp_size,
        is_disagg=is_disagg,
        num_reserved_index_slots=reserved,
        disable_overlap_scheduler=disable_overlap_scheduler,
        sparse=sparse,
    )
    assert capacity == expected
    assert page_table_capacity == expected
    assert max_admissible_sequences == expected - reserved


@pytest.mark.cpu_only
def test_sparse_arena_also_reserves_guard_request() -> None:
    assert _index_mapper_capacity_for(
        max_batch_size=2, num_reserved_index_slots=1, sparse=True, guard_page=True
    ) == (4, 4, 2)


def _sparse_arena_config(
    max_batch_size: int,
    explicit_ratio: bool,
    dense_size: int = 4096,
    *,
    num_layers: int = 1,
    shared_selection: bool = False,
) -> KVCacheManagerConfig:
    return KVCacheManagerConfig(
        tokens_per_block=16,
        cache_tiers=[GpuCacheTierConfig(4 << 20), HostCacheTierConfig(4 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(layer),
                buffers=[
                    # Sparse pages of four 4-token units, beside the dense buffer
                    # that sizes the pool group's slots.
                    BufferConfig(role=Role.KEY, size=4 * 4096, is_sparse=True),
                    BufferConfig(role=Role.KEY_BLOCK_SCALE, size=4 * 128, is_sparse=True),
                    BufferConfig(role=Role.VALUE, size=dense_size),
                ],
                sparse_cache=SparseCacheConfig(
                    num_selected_blocks=2,
                    max_cached_units=4,
                    unit_tokens=4,
                    selection_id=0 if shared_selection else None,
                ),
            )
            for layer in range(num_layers)
        ],
        constraints=[]
        if explicit_ratio
        else [BatchDesc([KVCacheDesc(capacity=16, history_length=0)] * max_batch_size)],
        initial_pool_ratio=[1.0] if explicit_ratio else None,
    )


@pytest.mark.parametrize("explicit_ratio", [False, True])
def test_sparse_arena_leases_follow_the_batch_rows(explicit_ratio: bool) -> None:
    """The arena is the Batch's, so its rows alone decide how many leases exist."""
    torch.cuda.synchronize()
    config = _sparse_arena_config(2, explicit_ratio)
    core = KVCacheManager(config)
    stream = CudaStream(torch.cuda.current_stream().cuda_stream)
    try:
        (shape,) = sparse_arena_shapes(config.layers, tokens_per_block=config.tokens_per_block)
        for rows in (1, 2, 5):
            batch = Batch(
                core,
                stream,
                max_batch_size=rows,
                max_blocks_per_seq=1,
                index_tables=core.sparse_index_table_specs(),
            )
            try:
                assert batch.sparse_arena_bytes == shape.arena_bytes(rows)
            finally:
                del batch
    finally:
        core.shutdown()


@pytest.mark.parametrize("max_batch_size,reserved", [(1, 1), (3, 2)])
@pytest.mark.parametrize("explicit_ratio", [False, True])
def test_sparse_arena_retains_padding_with_full_live_batch(
    max_batch_size: int, reserved: int, explicit_ratio: bool
) -> None:
    torch.cuda.synchronize()
    config = _sparse_arena_config(max_batch_size, explicit_ratio)
    core = KVCacheManager(config)
    caches = []
    stream = CudaStream(torch.cuda.current_stream().cuda_stream)
    batch = Batch(
        core,
        stream,
        max_batch_size=max_batch_size,
        max_blocks_per_seq=1,
        index_tables=core.sparse_index_table_specs(),
    )
    try:
        # A lease follows membership, so padding caches outside the Batch hold none and a
        # resumed request receiving its history costs the arena nothing.
        lc_id = core.get_layer_group_id(0)
        for index in range(reserved + max_batch_size):
            cache = core.create_kv_cache()
            caches.append(cache)
            assert cache.resume(stream)
            assert cache.resize(1)
            assert _introspection.sparse_cache_base_cell(cache, lc_id) is None
            if index >= reserved:
                batch.add_kv_cache(cache)
                assert _introspection.sparse_cache_base_cell(cache, lc_id) is not None
        assert batch.max_batch_size == max_batch_size
        batch.publish()
        batch.wait_published()

        # The rows bound membership, not the arena: a free row always has a free lease,
        # so a full Batch refuses on its rows rather than running out of cells.
        extra = core.create_kv_cache()
        caches.append(extra)
        assert extra.resume(stream)
        assert extra.resize(1)
        with pytest.raises(Exception, match="full"):
            batch.add_kv_cache(extra)
        batch.remove_kv_cache(caches[-2])
        assert _introspection.sparse_cache_base_cell(caches[-2], lc_id) is None
        batch.add_kv_cache(extra)
        assert _introspection.sparse_cache_base_cell(extra, lc_id) is not None
    finally:
        for cache in caches:
            if batch.row_of(cache) is not None:
                batch.remove_kv_cache(cache)
            cache.close()
        core.shutdown()


def test_sparse_arena_shape_matches_native_layout() -> None:
    torch.cuda.synchronize()
    leases = 5
    config = _sparse_arena_config(2, False)
    core = KVCacheManager(config)
    try:
        (shape,) = sparse_arena_shapes(config.layers, tokens_per_block=config.tokens_per_block)
        layer = LayerId(0)
        units_per_page = core.get_sparse_cache_units_per_page(layer)
        assert shape.units_per_cell == units_per_page
        assert shape.cells_per_buffer == -(
            -core.get_sparse_cache_max_units(layer) // units_per_page
        )
        assert len(shape.cell_bytes) == core.get_sparse_cache_buffer_count(layer)
        # Pool regions are consecutive, each holding every lease's cells for one pool and
        # then a zeroed over-read tail of 128 selection units.
        regions = {
            -(
                -(
                    leases * shape.cells_per_buffer * cell_bytes
                    + 128 * (cell_bytes // shape.units_per_cell)
                )
                // SPARSE_ARENA_ALIGNMENT
            )
            * SPARSE_ARENA_ALIGNMENT
            for cell_bytes in shape.cell_bytes
        }
        batch = Batch(
            core,
            CudaStream(torch.cuda.current_stream().cuda_stream),
            max_batch_size=leases,
            max_blocks_per_seq=1,
            index_tables=core.sparse_index_table_specs(),
        )
        key_base = batch.get_sparse_cache_base_address(layer, Role.KEY)
        scale_base = batch.get_sparse_cache_base_address(layer, Role.KEY_BLOCK_SCALE)
        assert abs(scale_base - key_base) in regions

        # The sparse buffers' pools share one pool group with the dense buffer, and
        # their slots hold one page of every sparse buffer: a cell's worth each.
        sparse_buffers = {(0, str(Role.KEY)), (0, str(Role.KEY_BLOCK_SCALE))}
        (group,) = core.pool_group_descs
        coalesced_buffers = group.slot_desc.variants[0].coalesced_buffers
        sparse_pool_bytes = [
            pool.slot_bytes
            for pool in group.pools
            if any(
                (int(buffer_id.layer_id), str(buffer_id.role)) in sparse_buffers
                for buffer_id in coalesced_buffers[int(pool.pool_index)].buffer_ids
            )
        ]
        assert sorted(sparse_pool_bytes) == sorted(shape.cell_bytes)
        assert len(group.pools) == 3
    finally:
        core.shutdown()


# A dense buffer that sorts between the sparse ones leaves a pool index no sparse buffer
# uses; one that sorts last leaves none. The two exercise different arena layouts, and
# until both were checked a missing placeholder region and a phantom shared array cancelled.
@pytest.mark.parametrize("dense_size", [4096, 256], ids=["dense-between", "dense-last"])
def test_sparse_arena_bytes_match_the_native_allocation(dense_size: int) -> None:
    """Python predicts the arena's bytes; this is the only thing that checks the prediction.

    The reservation has to run before a manager exists, so the bytes are modelled in
    Python and compared against a figure Python also computed -- which cannot catch the
    model drifting from the allocation. Reading the allocation back closes that loop.
    """
    torch.cuda.synchronize()
    leases = 5
    config = _sparse_arena_config(2, False, dense_size)
    core = KVCacheManager(config)
    try:
        (shape,) = sparse_arena_shapes(config.layers, tokens_per_block=config.tokens_per_block)
        batch = Batch(
            core,
            CudaStream(torch.cuda.current_stream().cuda_stream),
            max_batch_size=leases,
            max_blocks_per_seq=1,
            index_tables=core.sparse_index_table_specs(),
        )
        assert batch.sparse_arena_bytes == shape.arena_bytes(leases)
        assert batch.sparse_arena_bytes > 0
    finally:
        core.shutdown()


@pytest.mark.parametrize("shared_selection", [False, True], ids=["apart", "shared"])
def test_sparse_arena_bytes_follow_the_selection_count(shared_selection: bool) -> None:
    """The metadata is sized by selections, so sharing one has to shrink the allocation.

    The same loop as above, closed over the dimension a selection group changes: the
    Python model counts metadata rows itself, and nothing else would catch it counting
    layers where the arena counts selections.
    """
    torch.cuda.synchronize()
    leases = 5
    config = _sparse_arena_config(2, False, num_layers=2, shared_selection=shared_selection)
    core = KVCacheManager(config)
    try:
        (shape,) = sparse_arena_shapes(config.layers, tokens_per_block=config.tokens_per_block)
        assert shape.num_layers == 2
        assert shape.num_selections == (1 if shared_selection else 2)
        batch = Batch(
            core,
            CudaStream(torch.cuda.current_stream().cuda_stream),
            max_batch_size=leases,
            max_blocks_per_seq=1,
            index_tables=core.sparse_index_table_specs(),
        )
        assert batch.sparse_arena_bytes == shape.arena_bytes(leases)
    finally:
        core.shutdown()


def _native_config_with_sparse_arena(
    *,
    arena_bytes: int,
    budget: int,
    sparse_layers: bool = False,
    batch_buffer_bytes: int = 0,
    mapping: Mapping | None = None,
    sparse_kv_offload_config: SparseKvCacheOffloadConfig | None = None,
    kv_cache_config: KvCacheConfig | None = None,
    host_history_budget: int = 0,
    host_pinnable_bytes: float = float("inf"),
) -> KVCacheManagerConfig:
    """Construct a manager that reserves ``arena_bytes`` for its arena and
    ``batch_buffer_bytes`` for its offload buffers, and plans
    ``host_history_budget`` bytes of host history out of ``host_pinnable_bytes``;
    return the config it gives the native manager."""
    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
    layers = (
        _sparse_arena_config(2, False).layers
        if sparse_layers
        else [
            AttentionLayerConfig(
                layer_id=LayerId(0), buffers=[BufferConfig(role=Role.KEY, size=128)]
            )
        ]
    )
    fake_impl = Mock()
    fake_impl.buffer_grouping = [[BufferId(LayerId(0), Role.KEY)]]
    fake_impl.pool_group_descs = []
    fake_impl.get_layer_group_id.side_effect = lambda _: 0

    def build_base_config(
        self: KVCacheManagerV2,
        config: KvCacheConfig,
        *,
        tokens_per_block: int,
        cache_tiers: list[object],
    ) -> KVCacheManagerConfig:
        del self, config
        return KVCacheManagerConfig(
            tokens_per_block=tokens_per_block, cache_tiers=cache_tiers, layers=layers
        )

    def plan_host_history_budget(
        self: KVCacheManagerV2, kv_cache_config: KvCacheConfig, **kwargs: object
    ) -> None:
        # Settles the budget only; the manager divides the device quota against it.
        self._host_history_budget = host_history_budget
        self._host_pinnable_bytes = host_pinnable_bytes

    with (
        patch.dict(os.environ, {"TRTLLM_KV_GUARD_PAGE": "", "TRTLLM_KV_FRESH_PAGE_FILL": ""}),
        patch(f"{module}.IndexMapper"),
        patch(f"{module}.KVCacheManagerPy", return_value=fake_impl) as create_impl,
        patch.object(KVCacheManagerV2, "_build_base_config", build_base_config),
        patch.object(KVCacheManagerV2, "_build_cache_config", lambda self, config: config),
        patch.object(KVCacheManagerV2, "get_num_available_tokens", return_value=MAX_SEQ_LEN),
        patch.object(KVCacheManagerV2, "_prepare_page_table_tensor"),
        patch.object(KVCacheManagerV2, "_log_kv_cache_pool_lifecycle_mapping"),
        patch.object(KVCacheManagerV2, "_reserve_guard_page"),
        patch.object(
            KVCacheManagerV2, "_get_sparse_arena_bytes", return_value=arena_bytes
        ) as get_arena_bytes,
        patch.object(
            KVCacheManagerV2,
            "_get_sparse_batch_buffer_bytes",
            return_value=(batch_buffer_bytes, 4096 if batch_buffer_bytes else 0),
        ) as get_batch_buffer_bytes,
        patch.object(KVCacheManagerV2, "_plan_host_history_budget", plan_host_history_budget),
    ):
        KVCacheManagerV2(
            kv_cache_config or KvCacheConfig(max_gpu_total_bytes=budget, enable_block_reuse=False),
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=2,
            mapping=mapping or Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.HALF,
            vocab_size=16,
            execution_stream=Mock(),
            is_disagg=True,
            sparse_kv_offload_config=sparse_kv_offload_config,
        )
    # A lease follows Batch membership, so the arena is sized by the rows rather than by
    # the 5 indices the mapper can hand out -- disagg doubling and padding included.
    get_arena_bytes.assert_called_once_with(ANY, tokens_per_block=TOKENS_PER_BLOCK, max_leases=2)
    get_batch_buffer_bytes.assert_called_once_with(ANY, tokens_per_block=TOKENS_PER_BLOCK)
    return create_impl.call_args.args[0]


def _device_quota_with_sparse_arena(**kwargs: object) -> int:
    """Return the device quota of ``_native_config_with_sparse_arena``."""
    return _native_config_with_sparse_arena(**kwargs).cache_tiers[0].quota


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("arena_bytes", "batch_buffer_bytes"), [(0, 0), (3 << 20, 0), (3 << 20, 1 << 20)]
)
def test_device_quota_leaves_room_for_the_offload_allocations(
    arena_bytes: int, batch_buffer_bytes: int
) -> None:
    quota = _device_quota_with_sparse_arena(
        arena_bytes=arena_bytes, budget=16 << 20, batch_buffer_bytes=batch_buffer_bytes
    )

    assert quota == (16 << 20) - arena_bytes - batch_buffer_bytes


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("arena_bytes", "batch_buffer_bytes"), [(16 << 20, 0), (12 << 20, 4 << 20)]
)
def test_offload_allocations_larger_than_the_budget_are_rejected(
    arena_bytes: int, batch_buffer_bytes: int
) -> None:
    with pytest.raises(ValueError, match="Sparse KV cache offload needs"):
        _device_quota_with_sparse_arena(
            arena_bytes=arena_bytes, budget=16 << 20, batch_buffer_bytes=batch_buffer_bytes
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_tokens", [4 * TOKENS_PER_BLOCK, 1 << 30])
def test_max_tokens_sizes_the_pools_beside_the_offload_allocations(max_tokens: int) -> None:
    # The arena and buffers come out of the device memory budget. max_tokens sizes the
    # pools alone, so a pool for a few tokens still fits beside a large arena.
    def quota(arena_bytes: int, batch_buffer_bytes: int) -> int:
        return _device_quota_with_sparse_arena(
            arena_bytes=arena_bytes,
            budget=16 << 20,
            batch_buffer_bytes=batch_buffer_bytes,
            kv_cache_config=KvCacheConfig(
                max_gpu_total_bytes=16 << 20, max_tokens=max_tokens, enable_block_reuse=False
            ),
        )

    pools = quota(0, 0)
    assert quota(12 << 20, 1 << 20) == min(pools, (16 << 20) - (13 << 20))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("arena_bytes", "error", "match"),
    [
        # Rank 1 fits its budget, but another rank does not.
        (0, RuntimeError, "Sizing sparse KV cache offload failed on another rank"),
        # Rank 1 itself does not fit, and raises its own error.
        (16 << 20, ValueError, "Sparse KV cache offload needs"),
    ],
)
def test_offload_sizing_fails_on_every_rank_before_the_token_sync(
    arena_bytes: int, error: type, match: str
) -> None:
    ok, abort = int(_KVCacheManagerInitStatus.OK), int(_KVCacheManagerInitStatus.ABORT)
    dist = Mock()
    # The config check passes everywhere; then sizing fails on some rank.
    dist.allreduce.side_effect = [ok, abort]

    with (
        patch.object(Distributed, "get", return_value=dist),
        pytest.raises(error, match=match),
    ):
        _device_quota_with_sparse_arena(
            arena_bytes=arena_bytes,
            budget=16 << 20,
            mapping=Mapping(world_size=2, rank=1, tp_size=2),
            sparse_kv_offload_config=SparseKvCacheOffloadConfig(),
        )

    # Every rank stops after agreeing; none is left in the token sync.
    this_rank = ok if arena_bytes == 0 else abort
    assert dist.allreduce.call_args_list == [
        call(ok, op=ReduceOp.MAX),
        call(this_rank, op=ReduceOp.MAX),
    ]


@pytest.mark.cpu_only
def test_host_history_budget_is_the_smallest_ranks() -> None:
    # Each rank sizes its budget from its own host memory. Ranks whose history pools
    # differed would suspend different requests, so every rank takes the smallest.
    local_budget, peer_budget = 8 << 30, 6 << 30
    dist = Mock()

    def allreduce(value, op):
        if op == ReduceOp.MIN and value == local_budget:
            return peer_budget
        return value

    dist.allreduce.side_effect = allreduce
    with patch.object(Distributed, "get", return_value=dist):
        config = _native_config_with_sparse_arena(
            arena_bytes=0,
            budget=16 << 20,
            mapping=Mapping(world_size=2, rank=1, tp_size=2),
            sparse_kv_offload_config=SparseKvCacheOffloadConfig(),
            kv_cache_config=KvCacheConfig(
                max_gpu_total_bytes=16 << 20, host_cache_size=16 << 20, enable_block_reuse=False
            ),
            host_history_budget=local_budget,
        )

    assert call(local_budget, op=ReduceOp.MIN) in dist.allreduce.call_args_list
    assert config.host_sparse_pool_quota == peer_budget


@pytest.mark.cpu_only
def test_automatic_host_tier_gets_what_offload_leaves(monkeypatch) -> None:
    # The 8 GiB of history and 4 KiB of staging leave 10 MiB of the memlock limit.
    monkeypatch.setattr(
        kv_cache_v2_module,
        "get_host_memory_limits",
        lambda mapping: HostMemoryLimits(1, float(400 << 30), float((8 << 30) + 4096 + (10 << 20))),
    )

    with patch.object(kv_cache_v2_module.logger, "warning") as warning:
        config = _native_config_with_sparse_arena(
            arena_bytes=0,
            budget=16 << 20,
            batch_buffer_bytes=1 << 20,
            sparse_kv_offload_config=SparseKvCacheOffloadConfig(),
            host_history_budget=8 << 30,
        )

    _, host_tier = config.cache_tiers
    assert isinstance(host_tier, HostCacheTierConfig)
    assert host_tier.quota == int((10 << 20) * 0.8)
    assert not any("cache tier" in c.args[0] for c in warning.call_args_list)


@pytest.mark.cpu_only
@pytest.mark.parametrize("host_history_size", [None, 8 << 30])
def test_automatic_host_tier_takes_what_the_history_pools_leave(
    monkeypatch, host_history_size: int | None
) -> None:
    monkeypatch.setattr(
        kv_cache_v2_module,
        "get_host_memory_limits",
        lambda mapping: HostMemoryLimits(1, float(400 << 30), float("inf")),
    )

    config = _native_config_with_sparse_arena(
        arena_bytes=0,
        budget=16 << 20,
        batch_buffer_bytes=1 << 20,
        sparse_kv_offload_config=SparseKvCacheOffloadConfig(host_history_size=host_history_size),
        host_history_budget=8 << 30,
        host_pinnable_bytes=float((8 << 30) + 4096 + (6 << 20)),
    )

    device_tier, host_tier = config.cache_tiers
    if host_history_size is None:
        # Sized automatically, the history pools come first: the tier gets the 6 MiB they
        # and the 4 KiB of staging leave of what automatic sizing may pin, not the device
        # quota.
        assert host_tier.quota == 6 << 20
    else:
        # An explicit size is the user's split; the tier is sized from available memory.
        assert host_tier.quota == device_tier.quota


def _host_tier_growth_manager(pinned: int, *, world_size: int = 1) -> KVCacheManagerV2:
    """A manager whose history pools pin ``pinned`` of their 7 GiB budget.

    Automatic sizing may pin 10 GiB beside 1 GiB of staging, the device quota is 6 GiB,
    and the host tier was sized at 2 GiB beside the whole budget.
    """
    manager = _offload_setup_manager([64], world_size=world_size, cold_tier=True)
    quotas = {GPU_LEVEL: 6 << 30, CACHE_LEVEL1: 2 << 30}
    manager.impl.get_quota = lambda level: quotas[level]
    manager.impl.resize = Mock(return_value=True)
    manager._get_host_backed_pool_bytes = lambda: pinned
    manager._host_pinnable_bytes = float(10 << 30)
    manager._sparse_batch_staging_bytes = 1 << 30
    manager._host_history_budget = 7 << 30
    manager._host_tier_follows_history = True
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("pinned", "grown_to"),
    [
        # Pinning 2 GiB leaves 7 GiB beside the pools and staging; the tier grows to the
        # 6 GiB device quota.
        (2 << 30, 6 << 30),
        # Pools that pin their whole budget leave the tier as it was sized.
        (7 << 30, None),
    ],
)
def test_automatic_host_tier_grows_into_unpinned_history(pinned: int, grown_to: int | None) -> None:
    manager = _host_tier_growth_manager(pinned)

    with patch.object(KVCacheManagerV2, "_log_sparse_kv_offload_capacity"):
        manager._setup_sparse_kv_offload(
            KvCacheConfig(), max_seq_len=1024, num_reserved_index_slots=0
        )

    if grown_to is None:
        manager.impl.resize.assert_not_called()
    else:
        manager.impl.resize.assert_called_once_with(CACHE_LEVEL1, grown_to)


@pytest.mark.cpu_only
def test_host_tier_keeps_its_size_when_a_rank_cannot_grow_it() -> None:
    manager = _host_tier_growth_manager(2 << 30, world_size=2)
    dist = Mock()
    # The ranks agree on the capacity and the 6 GiB target; another rank fails to grow.
    dist.allreduce.side_effect = [64, 6 << 30, 0]

    with (
        patch.object(kv_cache_v2_module.Distributed, "get", return_value=dist),
        patch.object(KVCacheManagerV2, "_log_sparse_kv_offload_capacity"),
        patch.object(kv_cache_v2_module.logger, "warning") as warning,
    ):
        manager._setup_sparse_kv_offload(
            KvCacheConfig(), max_seq_len=1024, num_reserved_index_slots=0
        )

    # Every rank goes back to the tier the ranks agreed on at construction.
    assert manager.impl.resize.call_args_list == [
        call(CACHE_LEVEL1, 6 << 30),
        call(CACHE_LEVEL1, 2 << 30),
    ]
    assert any("could not grow the host cache tier" in c.args[0] for c in warning.call_args_list)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("tiers", "reserve"),
    [
        # Room for one 1,000-token request: 8 pages of two layers' 4,096 + 512 bytes.
        ("automatic", 8 * 2 * (4096 + 512)),
        # An explicit host tier is set aside already, and a disk tier takes suspended
        # pages the host tier has no room for.
        ("explicit host", 0),
        ("disk", 0),
    ],
)
def test_host_tier_reserve_holds_one_max_seq_len_request(
    tmp_path, tiers: str, reserve: int
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.max_seq_len = 1000
    manager.num_extra_kv_tokens = 0
    manager._generation_kv_capacity_headroom = 1
    layer = AttentionLayerConfig(
        layer_id=LayerId(0),
        buffers=[BufferConfig(role=Role.KEY, size=4096), BufferConfig(role=Role.VALUE, size=512)],
    )
    config = SimpleNamespace(tokens_per_block=128, layers=[layer, layer])
    kv_cache_config = {
        "automatic": KvCacheConfig(),
        "explicit host": KvCacheConfig(host_cache_size=1 << 20),
        "disk": KvCacheConfig(disk_cache_size=1 << 20, disk_cache_path=str(tmp_path)),
    }[tiers]

    assert manager._get_host_tier_reserve_bytes(kv_cache_config, config) == reserve


@pytest.mark.cpu_only
@pytest.mark.parametrize(("max_seq_len", "history_pages"), [(128 << 10, 1024), (1 << 20, 8192)])
def test_host_tier_reserve_holds_only_the_pages_a_window_keeps(
    max_seq_len: int, history_pages: int
) -> None:
    # DeepSeek-V4-Flash at FP8 with fp4 indexer keys, in 128-token pages, one layer per
    # life cycle holding its buffers' bytes: the compressed history, which a request keeps
    # whole, its ratio-4 rows offloaded and its indexer keys and ratio-128 rows on the GPU;
    # the 128-token sliding window; and the 8-token compressor state.
    manager = object.__new__(KVCacheManagerV2)
    manager.max_seq_len = max_seq_len
    manager.num_extra_kv_tokens = 0
    manager._generation_kv_capacity_headroom = 1
    history = AttentionLayerConfig(
        layer_id=LayerId(0),
        buffers=[
            BufferConfig(role=DataRole("offloaded_rows"), size=344_064, is_sparse=True),
            BufferConfig(role=DataRole("resident_rows"), size=55_936),
        ],
    )
    sliding_window = AttentionLayerConfig(
        layer_id=LayerId(1),
        buffers=[BufferConfig(role=DataRole("swa"), size=13_303_808)],
        sliding_window_size=128,
    )
    compressor_state = AttentionLayerConfig(
        layer_id=LayerId(2),
        buffers=[BufferConfig(role=DataRole("compressor_state"), size=27_525_120)],
        sliding_window_size=8,
    )
    config = SimpleNamespace(
        tokens_per_block=128, layers=[history, sliding_window, compressor_state]
    )

    reserve = manager._get_host_tier_reserve_bytes(KvCacheConfig(), config)

    # Each window keeps the two pages its live tokens can straddle, so one request holds
    # 0.46 GiB at 128K tokens and 3.13 GiB at 1M, not the 39.3 GiB and 314.6 GiB that
    # max_seq_len pages of every life cycle would take.
    assert reserve == history_pages * (344_064 + 55_936) + 2 * 13_303_808 + 2 * 27_525_120


@pytest.mark.cpu_only
@pytest.mark.parametrize("disk_tier", [False, True])
def test_offload_refuses_to_leave_no_room_for_the_automatic_host_tier(
    monkeypatch, tmp_path, disk_tier: bool
) -> None:
    # The offloaded history takes all of the memlock limit.
    monkeypatch.setattr(
        kv_cache_v2_module,
        "get_host_memory_limits",
        lambda mapping: HostMemoryLimits(1, float(400 << 30), float(8 << 30)),
    )
    kwargs = dict(
        arena_bytes=0,
        budget=16 << 20,
        sparse_kv_offload_config=SparseKvCacheOffloadConfig(),
        host_history_budget=8 << 30,
    )

    if not disk_tier:
        # Falling back to the device quota would pin past the limit.
        with pytest.raises(ValueError, match="leaves none for the host cache tier"):
            _native_config_with_sparse_arena(**kwargs)
        return
    # A disk tier still gives evicted pages somewhere to go.
    config = _native_config_with_sparse_arena(
        **kwargs,
        kv_cache_config=KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            disk_cache_size=16 << 20,
            disk_cache_path=str(tmp_path),
            enable_block_reuse=False,
        ),
    )
    assert [type(tier) for tier in config.cache_tiers] == [
        GpuCacheTierConfig,
        DiskCacheTierConfig,
    ]


@pytest.mark.cpu_only
def test_offload_warns_when_it_runs_without_a_cold_tier() -> None:
    with patch.object(kv_cache_v2_module.logger, "warning") as warning:
        config = _native_config_with_sparse_arena(
            arena_bytes=0,
            budget=16 << 20,
            sparse_kv_offload_config=SparseKvCacheOffloadConfig(),
            kv_cache_config=KvCacheConfig(
                max_gpu_total_bytes=16 << 20, host_cache_size=0, enable_block_reuse=False
            ),
        )

    assert [type(tier) for tier in config.cache_tiers] == [GpuCacheTierConfig]
    assert any(
        "runs without a host or disk cache tier" in c.args[0] for c in warning.call_args_list
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("error", "agreed_status", "raised", "match"),
    [
        (None, _KVCacheManagerInitStatus.OK, None, None),
        (None, _KVCacheManagerInitStatus.ABORT, RuntimeError, "Checking failed on another rank"),
        (ValueError("no room"), _KVCacheManagerInitStatus.ABORT, ValueError, "no room"),
    ],
)
def test_raise_on_every_rank_agrees_across_ranks(
    error: Exception | None,
    agreed_status: _KVCacheManagerInitStatus,
    raised: type | None,
    match: str | None,
) -> None:
    mapping = SimpleNamespace(world_size=4, rank=2)
    dist = Mock()
    dist.allreduce.return_value = int(agreed_status)

    with patch.object(Distributed, "get", return_value=dist):
        if raised is None:
            _raise_on_every_rank(error, mapping, "Checking")
        else:
            with pytest.raises(raised, match=match):
                _raise_on_every_rank(error, mapping, "Checking")

    # Every rank offers its own outcome through the init-status sync, and the most
    # severe one is every rank's.
    local_status = _KVCacheManagerInitStatus.ABORT if error else _KVCacheManagerInitStatus.OK
    dist.allreduce.assert_called_once_with(int(local_status), op=ReduceOp.MAX)


@pytest.mark.cpu_only
def test_raise_on_every_rank_needs_no_collective_on_one_rank() -> None:
    mapping = SimpleNamespace(world_size=1, rank=0)

    with patch.object(Distributed, "get") as get:
        _raise_on_every_rank(None, mapping, "Checking")
        with pytest.raises(ValueError, match="no room"):
            _raise_on_every_rank(ValueError("no room"), mapping, "Checking")

    get.assert_not_called()


@pytest.mark.cpu_only
def test_sparse_arena_without_a_reservation_is_rejected() -> None:
    # Cached sparse buffers declared without sparse_kv_offload_config reserve no
    # device memory for their arena.
    with pytest.raises(RuntimeError, match="sparse_kv_offload_config"):
        _device_quota_with_sparse_arena(arena_bytes=0, budget=16 << 20, sparse_layers=True)


@pytest.mark.cpu_only
@pytest.mark.parametrize("offload", [False, True])
def test_sparse_arena_bytes_come_from_a_provisional_config(offload: bool) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.max_seq_len = MAX_SEQ_LEN
    manager.mapping = Mapping()
    if offload:
        manager._sparse_kv_offload = SparseKvCacheOffloadConfig()
    config = _sparse_arena_config(2, False)

    with patch.object(KVCacheManagerV2, "_build_manager_config", return_value=config) as build:
        arena_bytes = manager._get_sparse_arena_bytes(
            KvCacheConfig(), tokens_per_block=16, max_leases=5
        )

    if not offload:
        assert arena_bytes == 0
        build.assert_not_called()
        return
    assert arena_bytes == sparse_arena_bytes(config, 5) > 0
    # Layer configs do not depend on tier quotas, so any quota gives the same arena.
    (gpu_tier,) = build.call_args.kwargs["cache_tiers"]
    assert gpu_tier.quota == sys.maxsize


@pytest.mark.cpu_only
@pytest.mark.parametrize("offload", [False, True])
def test_sparse_batch_buffers_come_from_a_provisional_config(offload: bool) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.mapping = Mapping()
    manager.max_seq_len = MAX_SEQ_LEN
    manager.max_batch_size = 2
    manager._generation_kv_capacity_headroom = 1
    manager._ledger_tokens_per_block = TOKENS_PER_BLOCK
    if offload:
        manager._sparse_kv_offload = SparseKvCacheOffloadConfig()
    config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=sys.maxsize)],
        layers=_offload_layers(),
    )

    with (
        patch.object(KVCacheManagerV2, "_build_manager_config", return_value=config) as build,
        patch.object(
            KVCacheManagerV2, "get_sparse_offload_scratch_bytes", return_value=1000
        ) as get_scratch_bytes,
    ):
        device_bytes, staging_bytes = manager._get_sparse_batch_buffer_bytes(
            KvCacheConfig(), tokens_per_block=TOKENS_PER_BLOCK
        )

    if not offload:
        assert (device_bytes, staging_bytes) == (0, 0)
        build.assert_not_called()
        return
    # 17 tokens of 16 and the decode step's one span five pages, padded to eight.
    batch = sparse_batch_shape(
        config.layers, tokens_per_block=TOKENS_PER_BLOCK, max_batch_size=2, max_blocks_per_seq=8
    )
    assert batch.num_life_cycles == 2
    assert batch.selection_widths == (1,)
    # The Batch also converts the sparse buffer's indices into one table per row.
    assert batch.index_table_bytes == 4 * batch.index_table_entries * 2 * 8 > 0
    assert device_bytes == (
        batch.block_bytes
        + batch.mirror_bytes
        + batch.index_table_bytes
        + batch.fetch_scratch_bytes
        + 1000
    )
    assert staging_bytes == batch.staging_bytes == 4096
    get_scratch_bytes.assert_called_once_with(config)
    (gpu_tier,) = build.call_args.kwargs["cache_tiers"]
    assert gpu_tier.quota == sys.maxsize


@pytest.mark.cpu_only
def test_sparse_startup_rejects_ordinal_overflow_before_sizing() -> None:
    manager = _offload_manager()
    manager.max_seq_len = 262_143 * 4
    config = _sparse_arena_config(2, False)
    with (
        patch.object(KVCacheManagerV2, "_build_manager_config", return_value=config),
        patch.object(kv_cache_v2_module, "sparse_arena_bytes") as size_arena,
        pytest.raises(ValueError, match="max_seq_len=1048572"),
    ):
        manager._get_sparse_arena_bytes(KvCacheConfig(), tokens_per_block=16, max_leases=2)

    size_arena.assert_not_called()
    assert manager._offload_sizing_config is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("local_overflow", [False, True])
def test_sparse_ordinal_overflow_fails_startup_on_every_rank(local_overflow: bool) -> None:
    manager = _offload_manager()
    manager.max_seq_len = 1_000_000
    manager.mapping = Mapping(world_size=2, rank=0 if local_overflow else 1, tp_size=2)
    config = _sparse_arena_config(2, False)
    # Only the token-sparse rank exceeds the limit; the four-token units fit.
    layer = config.layers[0]
    layer.sparse_cache = SparseCacheConfig(
        num_selected_blocks=2, max_cached_units=4, unit_tokens=1 if local_overflow else 4
    )
    config.layers = [layer]
    dist = Mock()
    dist.allreduce.return_value = int(_KVCacheManagerInitStatus.ABORT)
    error_type = ValueError if local_overflow else RuntimeError
    message = "unit_tokens=1" if local_overflow else "sequence-length check failed on another rank"
    with (
        patch.object(Distributed, "get", return_value=dist),
        patch.object(KVCacheManagerV2, "_build_manager_config", return_value=config),
        patch.object(kv_cache_v2_module, "sparse_arena_bytes") as size_arena,
        pytest.raises(error_type, match=message),
    ):
        manager._get_sparse_arena_bytes(KvCacheConfig(), tokens_per_block=16, max_leases=2)

    status = _KVCacheManagerInitStatus.ABORT if local_overflow else _KVCacheManagerInitStatus.OK
    dist.allreduce.assert_called_once_with(int(status), op=ReduceOp.MAX)
    size_arena.assert_not_called()
    assert manager._offload_sizing_config is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("max_seq_len", "headroom", "max_blocks_per_seq"), [(16, 1, 8), (15, 1, 4), (16, 0, 4)]
)
def test_page_table_holds_the_decode_step_and_pads_to_four_pages(
    max_seq_len: int, headroom: int, max_blocks_per_seq: int
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._generation_kv_capacity_headroom = headroom
    manager._ledger_tokens_per_block = TOKENS_PER_BLOCK

    assert manager._get_max_blocks_per_seq(max_seq_len) == max_blocks_per_seq


def _gen_init_manager(host_capacity_blocks: int | None) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    if host_capacity_blocks is not None:
        manager._host_history_reservations = HostHistoryReservations(host_capacity_blocks)
    manager.num_extra_kv_tokens = 0
    manager.tokens_per_block = 128
    manager._on_allocation_stream = set()
    manager._allocation_ready = {}
    return manager


def _gen_init_request(request_id: int, prompt_len: int, max_new_tokens: int):
    return SimpleNamespace(
        py_request_id=request_id,
        prompt_len=prompt_len,
        orig_prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        py_beam_width=1,
        try_get_encoder_output_len=lambda: None,
    )


@pytest.mark.cpu_only
def test_gen_init_reserves_host_history_to_completion() -> None:
    manager = _gen_init_manager(host_capacity_blocks=10)
    reservations = manager._host_history_reservations

    with patch.object(
        KVCacheManagerV2, "_prepare_disagg_gen_init_cache", return_value=True
    ) as prepare:
        # 641 tokens through the last output token span six blocks.
        assert manager.prepare_disagg_gen_init(_gen_init_request(1, 512, 129))
        assert reservations.reserved_blocks == 6
        # Five more blocks do not fit beside it, and the cache is left untouched.
        assert not manager.prepare_disagg_gen_init(_gen_init_request(2, 512, 128))
        assert prepare.call_count == 1
        # Preparing a request again replaces its reservation.
        assert manager.prepare_disagg_gen_init(_gen_init_request(1, 256, 0))
        assert manager.prepare_disagg_gen_init(_gen_init_request(2, 512, 128))
        assert reservations.reserved_blocks == 7
        # A request larger than the whole pool is never admitted.
        assert not manager.prepare_disagg_gen_init(_gen_init_request(3, 128 * 11, 0))
    with patch.object(KVCacheManagerV2, "_prepare_disagg_gen_init_cache", return_value=False):
        # A request whose cache could not be prepared reserves nothing.
        assert not manager.prepare_disagg_gen_init(_gen_init_request(4, 128, 0))
    assert reservations.reserved_blocks == 7


@pytest.mark.cpu_only
def test_gen_init_without_sparse_offload_reserves_nothing() -> None:
    manager = _gen_init_manager(host_capacity_blocks=None)

    with patch.object(
        KVCacheManagerV2, "_prepare_disagg_gen_init_cache", return_value=True
    ) as prepare:
        assert manager.prepare_disagg_gen_init(_gen_init_request(1, 1 << 20, 0))

    prepare.assert_called_once()
    assert not manager.kv_block_budget_applies()
    assert manager.get_request_kv_block_budget(_gen_init_request(1, 1 << 20, 0)) is None


@pytest.mark.cpu_only
def test_request_larger_than_host_history_is_rejected_on_arrival() -> None:
    manager = _gen_init_manager(host_capacity_blocks=10)
    manager._host_history_capacity_blocks = 10
    executor = PyExecutor.__new__(PyExecutor)
    executor.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})

    assert manager.kv_block_budget_applies()
    # 1281 tokens through the last output token span 11 blocks, which no rank can
    # reserve, so the request fails instead of waiting for admission forever.
    with pytest.raises(ValueError, match="requires 11 KV cache blocks.*capacity of 10 blocks"):
        executor._validate_request_budget(_gen_init_request(1, 1024, 257))
    # Ten blocks fit, and admission reserves exactly what the check counted.
    request = _gen_init_request(2, 1024, 256)
    executor._validate_request_budget(request)
    with patch.object(KVCacheManagerV2, "_prepare_disagg_gen_init_cache", return_value=True):
        assert manager.prepare_disagg_gen_init(request)
    assert manager._host_history_reservations.reserved_blocks == 10


def _offload_setup_manager(
    local_slots: list[int],
    *,
    world_size: int = 1,
    attention_dp: bool = False,
    guard_page: bool = False,
    max_total_draft_tokens: int = 0,
    cold_tier: bool = False,
) -> KVCacheManagerV2:
    """A manager stand-in whose host-backed pool groups hold ``local_slots`` slots.

    Without ``cold_tier`` it has no cache tier below the GPU, so admission reserves
    history.
    """
    manager = object.__new__(KVCacheManagerV2)
    tiers = [CacheTier.GPU_MEM, CacheTier.HOST_MEM] if cold_tier else [CacheTier.GPU_MEM]
    manager.impl = SimpleNamespace(cache_tier_list=tiers)
    manager.mapping = SimpleNamespace(
        world_size=world_size, rank=0, enable_attention_dp=attention_dp
    )
    manager.is_estimating_kv_cache = False
    manager.num_extra_kv_tokens = 0
    manager.max_total_draft_tokens = max_total_draft_tokens
    manager.tokens_per_block = 128
    manager.kv_cache_map = {kv_cache_v2_module._GUARD_PAGE_REQUEST_ID: Mock()} if guard_page else {}
    manager.get_host_backed_pool_groups = lambda: [
        SimpleNamespace(num_slots=n) for n in local_slots
    ]
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("world_size", "local_slots", "peer_slots", "capacity_blocks"),
    [
        (1, [12, 10], None, 10),
        # Every rank judges arriving requests and admits them by the smallest pool,
        # including a rank whose layers hold no host history.
        (2, [12, 10], 8, 8),
        (2, [], 8, 8),
        (2, [], sys.maxsize, None),
    ],
)
def test_sparse_offload_setup_judges_requests_by_the_smallest_host_pool(
    world_size: int, local_slots: list[int], peer_slots: int | None, capacity_blocks: int | None
) -> None:
    manager = _offload_setup_manager(local_slots, world_size=world_size)
    dist = Mock()
    dist.allreduce.side_effect = lambda value, op: min(value, peer_slots)

    with (
        patch.object(kv_cache_v2_module.Distributed, "get", return_value=dist),
        patch.object(KVCacheManagerV2, "_log_sparse_kv_offload_capacity"),
        patch.object(kv_cache_v2_module.logger, "warning") as warning,
    ):
        manager._setup_sparse_kv_offload(
            KvCacheConfig(), max_seq_len=1024, num_reserved_index_slots=0
        )

    if world_size > 1:
        dist.allreduce.assert_called_once_with(
            min(local_slots, default=sys.maxsize), op=ReduceOp.MIN
        )
    else:
        dist.allreduce.assert_not_called()
    assert manager._host_history_capacity_blocks == capacity_blocks
    assert manager.kv_block_budget_applies() == (capacity_blocks is not None)
    if local_slots:
        # A rank that admitted against its own larger pool would serve requests
        # its peers left waiting, and wait for them in collectives alone.
        assert manager._host_history_reservations.capacity_blocks == capacity_blocks
    else:
        assert manager._host_history_reservations is None
    # Eight blocks of 128 tokens hold a max_seq_len request of 1024 tokens.
    warning.assert_not_called()


@pytest.mark.cpu_only
def test_sparse_offload_admits_by_current_need_with_a_cold_tier() -> None:
    # A suspended request's history can move to the host tier, so requests take
    # history slots as they decode and nothing is reserved up front. A request that
    # could never fit alone is still rejected on arrival.
    manager = _offload_setup_manager([12, 10], cold_tier=True)

    with patch.object(KVCacheManagerV2, "_log_sparse_kv_offload_capacity"):
        manager._setup_sparse_kv_offload(
            KvCacheConfig(), max_seq_len=1024, num_reserved_index_slots=0
        )

    assert manager._host_history_reservations is None
    assert manager._host_history_capacity_blocks == 10
    assert manager.get_request_kv_block_budget(_gen_init_request(1, 128 * 11, 0)) == (11, 10)
    manager._on_allocation_stream = set()
    manager._allocation_ready = {}
    with patch.object(
        KVCacheManagerV2, "_prepare_disagg_gen_init_cache", return_value=True
    ) as prepare:
        # Two requests of ten blocks each are admitted beside each other.
        assert manager.prepare_disagg_gen_init(_gen_init_request(1, 1024, 256))
        assert manager.prepare_disagg_gen_init(_gen_init_request(2, 1024, 256))
    assert prepare.call_count == 2


@pytest.mark.cpu_only
def test_sparse_offload_setup_warns_when_max_seq_len_outgrows_host_history() -> None:
    manager = _offload_setup_manager([8])

    with (
        patch.object(KVCacheManagerV2, "_log_sparse_kv_offload_capacity"),
        patch.object(kv_cache_v2_module.logger, "warning") as warning,
    ):
        manager._setup_sparse_kv_offload(
            KvCacheConfig(), max_seq_len=1025, num_reserved_index_slots=0
        )

    message = warning.call_args.args[0]
    assert "rejects requests whose prompt and max_tokens exceed 1024 tokens" in message
    assert "max_seq_len=1025 tokens needs 9" in message
    assert "kv_cache_offload_config.host_history_size" in message


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("num_reserved_index_slots", "attention_dp", "guard_page", "draft_tokens", "set_aside"),
    [
        # The CUDA-graph padding dummy.
        (1, False, False, 0, 1),
        # Plus the attention-DP dummy and the guard page.
        (1, True, False, 0, 2),
        (1, True, True, 0, 3),
        # A dummy with 64 draft tokens holds 1 + 2 x 64 tokens, two blocks of 128.
        (2, False, False, 64, 4),
    ],
)
def test_sparse_offload_setup_sets_dummy_history_aside(
    num_reserved_index_slots: int,
    attention_dp: bool,
    guard_page: bool,
    draft_tokens: int,
    set_aside: int,
) -> None:
    manager = _offload_setup_manager(
        [12, 10],
        attention_dp=attention_dp,
        guard_page=guard_page,
        max_total_draft_tokens=draft_tokens,
    )

    with patch.object(KVCacheManagerV2, "_log_sparse_kv_offload_capacity"):
        manager._setup_sparse_kv_offload(
            KvCacheConfig(), max_seq_len=128, num_reserved_index_slots=num_reserved_index_slots
        )

    # Dummy requests hold history pages without a reservation, so neither
    # admission nor the check on arrival may promise those pages to a request.
    assert manager._host_history_reservations.capacity_blocks == 10 - set_aside
    assert manager._host_history_capacity_blocks == 10 - set_aside
    whole_pool = _gen_init_request(1, 128 * 10, 0)
    assert manager.get_request_kv_block_budget(whole_pool) == (10, 10 - set_aside)


@pytest.mark.cpu_only
def test_free_resources_releases_host_history() -> None:
    manager = _gen_init_manager(host_capacity_blocks=10)
    manager._host_history_reservations.reserve(1, 6)
    manager.kv_connector_manager = None
    manager.conversation_manager = None
    manager._allocated_draft_lens = {}
    manager._inflight_draft_lens = {}
    manager._request_stats_enabled_ids = set()
    manager._fresh_pages_filled = {}
    manager._disagg_receive_ready = {}
    manager._pending_preemption = {}
    manager._on_allocation_stream = set()
    manager._allocation_ready = {}
    manager.kv_cache_map = {}
    manager.impl = Mock()

    manager.free_resources(SimpleNamespace(py_request_id=1, request_id=1))

    assert manager._host_history_reservations.reserved_blocks == 0


@pytest.mark.cpu_only
def test_host_memory_limits_cap_automatic_pinning() -> None:
    assert HostMemoryLimits(4, float(400 << 30), float("inf")).auto_pinnable_bytes == 50 << 30
    assert HostMemoryLimits(1, float("inf"), float(10 << 30)).auto_pinnable_bytes == 8 << 30
    assert HostMemoryLimits(1, float("inf"), float("inf")).auto_pinnable_bytes == float("inf")


@pytest.mark.cpu_only
def test_sparse_kv_offload_config_is_checked_against_this_rank(monkeypatch) -> None:
    limits = {"value": HostMemoryLimits(4, float(400 << 30), float("inf"))}
    monkeypatch.setattr(
        kv_cache_v2_module, "get_host_memory_limits", lambda mapping: limits["value"]
    )

    def check(offload_config, kv_cache_config, mapping):
        KVCacheManagerV2._check_sparse_kv_offload_config(
            offload_config, kv_cache_config, mapping, spec_config=None
        )

    kv_cache_config = KvCacheConfig(enable_block_reuse=False)

    check(SparseKvCacheOffloadConfig(host_history_size=50 << 30), kv_cache_config, Mapping())
    # A quarter of 400 GiB is this rank's share.
    with pytest.raises(ValueError, match="share of available host memory"):
        check(SparseKvCacheOffloadConfig(host_history_size=101 << 30), kv_cache_config, Mapping())
    # An explicit host cache tier is pinned beside the history.
    check(
        SparseKvCacheOffloadConfig(host_history_size=50 << 30),
        KvCacheConfig(enable_block_reuse=False, host_cache_size=50 << 30),
        Mapping(),
    )
    with pytest.raises(
        ValueError, match=r"plus kv_cache_config.host_cache_size \(51.00GiB\) exceeds this rank"
    ):
        check(
            SparseKvCacheOffloadConfig(host_history_size=50 << 30),
            KvCacheConfig(enable_block_reuse=False, host_cache_size=51 << 30),
            Mapping(),
        )
    limits["value"] = HostMemoryLimits(1, float("inf"), float(10 << 30))
    with pytest.raises(ValueError, match="RLIMIT_MEMLOCK"):
        check(SparseKvCacheOffloadConfig(host_history_size=50 << 30), kv_cache_config, Mapping())
    with pytest.raises(ValueError, match=r"plus kv_cache_config.host_cache_size .* RLIMIT_MEMLOCK"):
        check(
            SparseKvCacheOffloadConfig(host_history_size=5 << 30),
            KvCacheConfig(enable_block_reuse=False, host_cache_size=6 << 30),
            Mapping(),
        )
    with pytest.raises(ValueError, match="enable_kv_pool_rebalance"):
        check(
            SparseKvCacheOffloadConfig(),
            KvCacheConfig(enable_block_reuse=False, enable_kv_pool_rebalance=True),
            Mapping(),
        )


@pytest.mark.cpu_only
def test_sparse_kv_offload_rejects_speculative_decoding() -> None:
    # A sparse cache takes one input token per decode step, and the history set
    # aside for dummy requests counts a single CUDA-graph padding dummy.
    with pytest.raises(ValueError, match="does not support speculative decoding"):
        KVCacheManagerV2._check_sparse_kv_offload_config(
            SparseKvCacheOffloadConfig(),
            KvCacheConfig(enable_block_reuse=False),
            Mapping(),
            spec_config=MTPDecodingConfig(max_draft_len=1),
        )


@pytest.mark.cpu_only
def test_sparse_kv_offload_accepts_block_reuse() -> None:
    # Reuse is on by default. On a decode worker it lets the KV transfer skip a
    # prompt prefix the reuse tree already holds.
    KVCacheManagerV2._check_sparse_kv_offload_config(
        SparseKvCacheOffloadConfig(), KvCacheConfig(), Mapping(), spec_config=None
    )


@pytest.mark.cpu_only
def test_sparse_kv_offload_rejects_pipeline_parallelism() -> None:
    # The attention Batch holds one micro-batch's members; publishing the next one would
    # release the leases of the first while it is still in flight.
    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        KVCacheManagerV2._check_sparse_kv_offload_config(
            SparseKvCacheOffloadConfig(),
            KvCacheConfig(enable_block_reuse=False),
            Mapping(world_size=2, rank=0, pp_size=2),
            spec_config=None,
        )


@pytest.mark.cpu_only
def test_sparse_kv_offload_config_check_fails_on_every_rank(monkeypatch) -> None:
    # This rank's node has room, but rank 0's does not.
    monkeypatch.setattr(
        kv_cache_v2_module,
        "get_host_memory_limits",
        lambda mapping: HostMemoryLimits(4, float(400 << 30), float("inf")),
    )
    dist = Mock()
    dist.allreduce.return_value = int(_KVCacheManagerInitStatus.ABORT)

    with (
        patch.object(Distributed, "get", return_value=dist),
        pytest.raises(RuntimeError, match="config check failed on another rank"),
    ):
        KVCacheManagerV2._check_sparse_kv_offload_config(
            SparseKvCacheOffloadConfig(host_history_size=50 << 30),
            KvCacheConfig(enable_block_reuse=False),
            Mapping(world_size=2, rank=1, tp_size=2),
            spec_config=None,
        )

    dist.allreduce.assert_called_once_with(int(_KVCacheManagerInitStatus.OK), op=ReduceOp.MAX)


def _offload_layers() -> list[AttentionLayerConfig]:
    """A layer whose 1 KiB sparse pages sit beside 256-byte dense ones, and a windowed layer."""
    return [
        AttentionLayerConfig(
            layer_id=LayerId(0),
            buffers=[
                BufferConfig(role=Role.KEY, size=1024, is_sparse=True),
                BufferConfig(role=Role.VALUE, size=256),
            ],
            sparse_cache=SparseCacheConfig(
                num_selected_blocks=1, max_cached_units=2, unit_tokens=TOKENS_PER_BLOCK
            ),
        ),
        AttentionLayerConfig(
            layer_id=LayerId(1),
            buffers=[BufferConfig(role=Role.KEY, size=4096)],
            sliding_window_size=8,
        ),
    ]


def _offload_manager(host_history_size: int | None = None) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager._sparse_kv_offload = SparseKvCacheOffloadConfig(host_history_size=host_history_size)
    manager.max_seq_len = MAX_SEQ_LEN
    manager.max_batch_size = 2
    manager.mapping = Mapping()
    manager._generation_kv_capacity_headroom = 1
    return manager


def _base_config(layers: list, initial_pool_ratio: list[float] | None = None):
    config = KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=layers,
        initial_pool_ratio=initial_pool_ratio,
    )
    return (
        patch.object(KVCacheManagerV2, "_build_base_config", return_value=config),
        patch.object(KVCacheManagerV2, "_build_cache_config", lambda self, config: config),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("pinned", [False, True])
def test_offload_leaves_the_pool_ratio_to_the_manager_unless_pinned(pinned: bool) -> None:
    manager = _offload_manager()
    build_base, build_cache = _base_config(_offload_layers(), [0.25, 0.75] if pinned else None)

    with build_base, build_cache:
        config = manager._build_manager_config(
            KvCacheConfig(),
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        )

    # The manager splits the quota from the decode-only typical step itself, counting
    # host-backed pools in their own grain, so offload hands it no ratio of its own.
    assert config.initial_pool_ratio == ([0.25, 0.75] if pinned else None)


@pytest.mark.cpu_only
def test_host_backed_pools_come_from_the_native_pool_descriptions() -> None:
    manager = object.__new__(KVCacheManagerV2)

    def group(*media: CacheTier) -> SimpleNamespace:
        return SimpleNamespace(
            pools=[
                SimpleNamespace(pool_index=index, storage_medium=medium)
                for index, medium in enumerate(media)
            ]
        )

    mixed = group(CacheTier.HOST_MEM, CacheTier.GPU_MEM)
    dense = group(CacheTier.GPU_MEM)
    get_num_bytes = Mock(return_value=10 << 20)
    manager.impl = SimpleNamespace(
        pool_group_descs=[mixed, dense],
        sparse_backing_medium=lambda: CacheTier.HOST_MEM,
        get_num_bytes=get_num_bytes,
    )

    # A group keeps its dense pools on the GPU; each pool reports its own medium.
    assert manager.get_host_backed_pool_groups() == [mixed]
    # The native count includes how each pool rounds to the host grain.
    assert manager._get_host_backed_pool_bytes() == 10 << 20
    get_num_bytes.assert_called_once_with(GPU_LEVEL, CacheTier.HOST_MEM)

    manager.impl = SimpleNamespace(pool_group_descs=[dense], sparse_backing_medium=lambda: None)
    assert manager.get_host_backed_pool_groups() == []
    assert manager._get_host_backed_pool_bytes() == 0


@pytest.mark.cpu_only
def test_offload_hands_its_host_budget_to_the_manager(monkeypatch) -> None:
    """The device quota is not lowered to fit the host budget; the manager divides both.

    Lowering the quota to hold back host memory would shrink every pool group, device-only
    ones included, though only some of them spend host memory. The manager is given the
    budget instead and holds the host-backed pools inside it while the rest keep the device
    bytes.
    """

    module = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
    fake_impl = Mock()
    fake_impl.buffer_grouping = [[BufferId(LayerId(0), Role.KEY)]]
    fake_impl.pool_group_descs = []
    fake_impl.get_layer_group_id.side_effect = lambda _: 0

    def build_base_config(
        self: KVCacheManagerV2,
        config: KvCacheConfig,
        *,
        tokens_per_block: int,
        cache_tiers: list[object],
    ) -> KVCacheManagerConfig:
        del self, config
        return KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=cache_tiers,
            layers=_offload_layers()[:1],
        )

    # One rank on a node with 250 MiB of host memory available.
    monkeypatch.setattr(
        kv_cache_v2_module,
        "get_host_memory_limits",
        lambda mapping: HostMemoryLimits(1, float(250 << 20), float("inf")),
    )
    with (
        patch.dict(os.environ, {"TRTLLM_KV_GUARD_PAGE": "", "TRTLLM_KV_FRESH_PAGE_FILL": ""}),
        patch(f"{module}.IndexMapper"),
        patch(f"{module}.KVCacheManagerPy", return_value=fake_impl) as create_impl,
        patch(f"{module}.is_device_integrated", return_value=False),
        patch.object(KVCacheManagerV2, "_build_base_config", build_base_config),
        patch.object(KVCacheManagerV2, "_build_cache_config", lambda self, config: config),
        patch.object(KVCacheManagerV2, "get_num_available_tokens", return_value=MAX_SEQ_LEN),
        patch.object(KVCacheManagerV2, "_prepare_page_table_tensor"),
        patch.object(KVCacheManagerV2, "_log_kv_cache_pool_lifecycle_mapping"),
        patch.object(KVCacheManagerV2, "_reserve_guard_page"),
    ):
        KVCacheManagerV2(
            KvCacheConfig(max_gpu_total_bytes=64 << 20, enable_block_reuse=False),
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.HALF,
            vocab_size=16,
            execution_stream=Mock(),
            is_disagg=True,
            sparse_kv_offload_config=SparseKvCacheOffloadConfig(host_history_size=200 << 20),
        )

    config = create_impl.call_args.args[0]
    gpu_tier, host_tier = config.cache_tiers
    # The manager splits both from its typical step, host-backed pools included.
    assert config.initial_pool_ratio is None
    # Only the offload allocations come off the 64 MiB budget.
    assert gpu_tier.quota > (63 << 20)
    # The 200 MiB of host history less the sparse Batch's one-page staging copy, handed
    # over whole: the manager counts host bytes in their own grain.
    assert config.host_sparse_pool_quota == (200 << 20) - 4096
    # The automatic host tier still takes half of what the 200 MiB leaves of the 250 MiB.
    assert host_tier.quota == (50 << 20) // 2


@pytest.mark.parametrize(
    "setting, expected",
    [
        ("", None),
        ("1", 0.0),
        ("on", 0.0),
        ("true", 0.0),
        ("zero", 0.0),
        ("0", 0.0),
        ("-2.5", -2.5),
        ("garbage", None),
    ],
)
def test_kv_fill_value_parsing(setting, expected):
    """The knobs are off when unset and ignore anything they cannot parse."""
    assert kv_cache_v2_module._parse_kv_fill_value(setting, "TEST_VAR") == expected


def test_kv_fill_value_parsing_nan():
    value = kv_cache_v2_module._parse_kv_fill_value("nan", "TEST_VAR")
    assert value is not None and np.isnan(value)


def test_fill_kv_pages_float_buffer():
    buffer = torch.ones(4, 3, 2)
    kv_cache_v2_module._fill_kv_pages(buffer, [1, 3], 0.0)
    assert torch.equal(buffer[0], torch.ones(3, 2))
    assert torch.equal(buffer[1], torch.zeros(3, 2))
    assert torch.equal(buffer[2], torch.ones(3, 2))
    assert torch.equal(buffer[3], torch.zeros(3, 2))


def test_fill_kv_pages_packed_buffer():
    """A packed sub-byte pool is int8: zero stays zero, anything else poisons."""
    buffer = torch.ones(3, 4, dtype=torch.int8)
    kv_cache_v2_module._fill_kv_pages(buffer, [0], 0.0)
    kv_cache_v2_module._fill_kv_pages(buffer, [2], float("nan"))
    assert torch.equal(buffer[0], torch.zeros(4, dtype=torch.int8))
    assert torch.equal(buffer[1], torch.ones(4, dtype=torch.int8))
    assert torch.equal(buffer[2], torch.full((4,), 0x7F, dtype=torch.int8))


def test_fill_kv_pages_empty_is_a_noop():
    buffer = torch.ones(2, 2)
    assert kv_cache_v2_module._fill_kv_pages(buffer, [], 0.0) is True
    assert torch.equal(buffer, torch.ones(2, 2))


def test_fill_kv_pages_reports_applied():
    buffer = torch.ones(4, 2)
    assert kv_cache_v2_module._fill_kv_pages(buffer, [1], 0.0) is True


def test_fill_kv_pages_skips_dtype_without_index_fill():
    """FP8 has no ``index_fill_`` kernel; the fill is skipped, not fatal."""
    try:
        torch.zeros(2, dtype=torch.float8_e4m3fn).index_fill_(0, torch.tensor([0]), float("nan"))
    except NotImplementedError:
        pass
    else:
        pytest.skip("index_fill_ is supported for float8 on this platform")

    buffer = torch.zeros(4, 2, dtype=torch.float8_e4m3fn)
    assert kv_cache_v2_module._fill_kv_pages(buffer, [1], float("nan")) is False


def test_guard_page_accessors_report_no_guard_by_default():
    """With the guard page off, backends see ``None`` and keep page 0."""
    manager = SimpleNamespace(_guard_page_by_layer={})
    assert KVCacheManagerV2.guard_page_index(manager, 0) is None
    assert KVCacheManagerV2.guard_page_indices(manager) == frozenset()


def test_guard_page_accessors_report_reserved_pages():
    # Layers in one pool can carry different page-index scales, so one
    # reservation can surface as more than one index.
    manager = SimpleNamespace(_guard_page_by_layer={0: 7, 1: 14, 2: 7})
    assert KVCacheManagerV2.guard_page_index(manager, 1) == 14
    assert KVCacheManagerV2.guard_page_index(manager, 9) is None
    assert KVCacheManagerV2.guard_page_indices(manager) == frozenset({7, 14})


class _FreshFillKVCache:
    """Just enough of a KV cache for ``_fill_fresh_kv_pages``: a mutable page
    list standing in for the pool's base page indices."""

    def __init__(self, pages: list[int], num_committed_tokens: int = 0) -> None:
        self.pages = list(pages)
        self.num_committed_tokens = num_committed_tokens

    @property
    def num_blocks(self) -> int:
        return len(self.pages)

    def get_base_page_indices(self, pool_id: int) -> np.ndarray:
        del pool_id
        return np.asarray(self.pages, dtype=np.int32)


_FRESH_FILL = 3.0


def _make_fresh_fill_manager(
    buffer: torch.Tensor,
    kv_cache: _FreshFillKVCache,
    monkeypatch: pytest.MonkeyPatch,
    fill_value: float | None = _FRESH_FILL,
) -> KVCacheManagerV2:
    # These tests use CPU tensors and an explicit stream double; a device-wide wait is forbidden.
    monkeypatch.setattr(
        torch.cuda, "synchronize", Mock(side_effect=AssertionError("device-wide fill sync"))
    )
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=Mock()))
    manager = object.__new__(KVCacheManagerV2)
    manager._stream = SimpleNamespace(device=torch.device("cuda:0"))
    manager._fresh_page_fill = fill_value
    manager.layer_specs = None
    manager._fresh_pages_filled = {}
    manager._fresh_fill_announced = True
    manager.kv_cache_map = {1: kv_cache}
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.num_pools = 1
    manager.pp_layers = [0]
    manager.layer_offsets = {0: 0}
    manager.layer_to_pool_mapping_dict = {0: 0}
    manager.kv_factor = 1
    manager.get_buffers = lambda layer_idx: buffer
    manager.get_layer_page_index_scale = lambda layer_idx: 1
    return manager


def test_fresh_fill_covers_pages_on_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Newly handed-out pages get the pattern; pages the request keeps do not."""
    buffer = torch.zeros(8, 2)
    kv_cache = _FreshFillKVCache([2, 5])
    manager = _make_fresh_fill_manager(buffer, kv_cache, monkeypatch)
    other_buffer = torch.zeros_like(buffer)
    manager.pp_layers = [0, 1]
    manager.layer_offsets = {0: 0, 1: 1}
    manager.layer_to_pool_mapping_dict = {0: 0, 1: 0}
    manager.get_buffers = lambda layer_idx: (buffer, other_buffer)[layer_idx]
    order = Mock()
    order.attach_mock(torch.cuda.current_stream.return_value, "stream")
    fill = Mock(wraps=kv_cache_v2_module._fill_kv_pages)
    monkeypatch.setattr(kv_cache_v2_module, "_fill_kv_pages", fill)
    order.attach_mock(fill, "fill")

    manager._fill_fresh_kv_pages(1)
    torch.cuda.current_stream.assert_called_once_with(device=manager._stream.device)
    assert order.mock_calls == [
        call.stream.wait_stream(manager._stream),
        call.fill(buffer, [2, 5], _FRESH_FILL),
        call.fill(other_buffer, [2, 5], _FRESH_FILL),
        call.stream.synchronize(),
    ]
    assert torch.equal(buffer[2], torch.full((2,), _FRESH_FILL))
    assert torch.equal(buffer[5], torch.full((2,), _FRESH_FILL))
    assert torch.equal(buffer[0], torch.zeros(2))
    assert torch.equal(other_buffer, buffer)

    # The owner writes its KV into page 2; growing by one page must fill only
    # the new page, not rewrite what the owner already stored.
    buffer[2] = 1.5
    kv_cache.pages = [2, 5, 6]
    manager._fill_fresh_kv_pages(1)
    assert torch.equal(buffer[6], torch.full((2,), _FRESH_FILL))
    assert torch.equal(buffer[2], torch.full((2,), 1.5))
    assert order.mock_calls[-4:] == [
        call.stream.wait_stream(manager._stream),
        call.fill(buffer, [6], _FRESH_FILL),
        call.fill(other_buffer, [6], _FRESH_FILL),
        call.stream.synchronize(),
    ]


def test_fresh_fill_skips_committed_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pages holding the reused prefix carry a correct answer; writing the
    pattern over them would corrupt it rather than diagnose anything."""
    buffer = torch.zeros(8, 2)
    kv_cache = _FreshFillKVCache([2, 5], num_committed_tokens=TOKENS_PER_BLOCK)
    manager = _make_fresh_fill_manager(buffer, kv_cache, monkeypatch)

    manager._fill_fresh_kv_pages(1)
    assert torch.equal(buffer[2], torch.zeros(2))
    assert torch.equal(buffer[5], torch.full((2,), _FRESH_FILL))


def test_fresh_fill_disabled_or_unknown_request_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = torch.zeros(4, 2)
    kv_cache = _FreshFillKVCache([1])
    manager = _make_fresh_fill_manager(buffer, kv_cache, monkeypatch, fill_value=None)
    manager._fill_fresh_kv_pages(1)
    assert torch.equal(buffer, torch.zeros(4, 2))
    assert manager._fresh_pages_filled == {}

    manager._fresh_page_fill = _FRESH_FILL
    manager._fill_fresh_kv_pages(99)
    assert torch.equal(buffer, torch.zeros(4, 2))
    torch.cuda.current_stream.assert_not_called()


def test_fresh_fill_refills_page_reassigned_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shrink, reuse by another request, regrow: the page must be refilled.

    Tracking every page the request was ever given would skip the refill here
    and leave the other request's KV readable through this request's page
    table -- the exact leak the fill exists to catch.
    """
    buffer = torch.zeros(8, 2)
    kv_cache = _FreshFillKVCache([2, 5])
    manager = _make_fresh_fill_manager(buffer, kv_cache, monkeypatch)
    manager._fill_fresh_kv_pages(1)
    buffer[2] = 1.5  # the owner's own KV, must survive to the end

    # A resize releases page 5...
    kv_cache.pages = [2]
    manager._fill_fresh_kv_pages(1)
    # ...another request gets it and writes its KV...
    buffer[5] = 7.0
    # ...and it later comes back to this request.
    kv_cache.pages = [2, 5]
    manager._fill_fresh_kv_pages(1)

    assert torch.equal(buffer[5], torch.full((2,), _FRESH_FILL))
    assert torch.equal(buffer[2], torch.full((2,), 1.5))


def test_fresh_fill_preserves_generated_kv_across_slot_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """suspend -> physical-slot change -> resume must not overwrite generated KV.

    ``resume`` can onboard an already-generated block into a different physical
    page, copying the block's KV into the new slot. The block keeps its ordinal,
    so freshness has to be keyed off the ordinal, not the page id: keying off the
    page id sees the new slot as a fresh allocation and clobbers the restored KV
    (the reviewer's page 5 -> page 6 example). A genuinely new page grown after
    the resume must still be filled.
    """
    buffer = torch.zeros(8, 2)
    kv_cache = _FreshFillKVCache([2, 5])
    manager = _make_fresh_fill_manager(buffer, kv_cache, monkeypatch)

    manager._fill_fresh_kv_pages(1)
    # The owner generates into ordinal 1 (physical page 5).
    buffer[5] = 9.0

    # suspend + resume relocates ordinal 1 from page 5 to page 6, copying its KV.
    buffer[6] = 9.0
    kv_cache.pages = [2, 6]
    manager._fill_fresh_kv_pages(1)
    assert torch.equal(buffer[6], torch.full((2,), 9.0))  # restored KV survives
    assert torch.equal(buffer[5], torch.full((2,), 9.0))  # old slot untouched

    # A page grown after the resume is a real new allocation -> still filled.
    kv_cache.pages = [2, 6, 7]
    manager._fill_fresh_kv_pages(1)
    assert torch.equal(buffer[7], torch.full((2,), _FRESH_FILL))
    assert torch.equal(buffer[6], torch.full((2,), 9.0))


def test_fresh_fill_preserves_kv_across_block_onboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offload / onboard migration of several blocks must not be treated as fresh.

    A committed prefix plus two generated blocks all move to new physical pages
    when the cache is onboarded from a secondary tier. Every ordinal keeps its
    logical block, so none of the new pages is fresh and the copied KV stays.
    """
    buffer = torch.zeros(24, 2)
    kv_cache = _FreshFillKVCache([10, 11, 12], num_committed_tokens=TOKENS_PER_BLOCK)
    manager = _make_fresh_fill_manager(buffer, kv_cache, monkeypatch)

    manager._fill_fresh_kv_pages(1)
    assert torch.equal(buffer[10], torch.zeros(2))  # committed prefix, never filled
    assert torch.equal(buffer[11], torch.full((2,), _FRESH_FILL))
    # The owner generates into ordinals 1 and 2.
    buffer[11] = 4.0
    buffer[12] = 5.0

    # Onboard relocates every block; the KV is copied to the new slots.
    buffer[20] = 4.0
    buffer[21] = 4.0
    buffer[22] = 5.0
    kv_cache.pages = [20, 21, 22]
    manager._fill_fresh_kv_pages(1)
    assert torch.equal(buffer[21], torch.full((2,), 4.0))
    assert torch.equal(buffer[22], torch.full((2,), 5.0))


class _GuardKVCache:
    """Scriptable stand-in for the guard sequence's KV cache."""

    def __init__(self, resume_ok: bool = True, resize_ok: bool = True) -> None:
        self.resume_ok = resume_ok
        self.resize_ok = resize_ok
        self.closed = False
        self.committing_stopped = False

    def resume(self, stream: object) -> bool:
        del stream
        return self.resume_ok

    def stop_committing(self) -> None:
        self.committing_stopped = True

    def resize(self, num_tokens: int) -> bool:
        del num_tokens
        return self.resize_ok

    def close(self) -> None:
        self.closed = True


def _make_guard_manager(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache: _GuardKVCache | None,
    buffer: torch.Tensor | None,
    page: int,
) -> KVCacheManagerV2:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    manager = object.__new__(KVCacheManagerV2)
    manager._guard_page_fill = "nan"
    manager._guard_page_value = float("nan")
    manager._guard_page_by_layer = {}
    manager.num_extra_kv_tokens = 0
    manager.pp_layers = [0]
    manager._stream = SimpleNamespace(cuda_stream=None)
    manager.kv_cache_map = {}
    manager.index_mapper = Mock()
    manager.impl = Mock()

    def create_kv_cache(request_id: int, *args: object, **kwargs: object) -> object | None:
        del args, kwargs
        if kv_cache is not None:
            manager.kv_cache_map[request_id] = kv_cache
        return kv_cache

    manager._create_kv_cache = create_kv_cache
    manager.get_buffers = lambda layer_idx: buffer
    manager.get_batch_cache_indices = lambda request_ids, layer_idx: [[page]]
    return manager


def test_reserve_guard_page_fills_and_publishes_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = torch.zeros(8, 2)
    kv_cache = _GuardKVCache()
    manager = _make_guard_manager(monkeypatch, kv_cache, buffer, page=3)

    manager._reserve_guard_page()

    assert manager._guard_page_by_layer == {0: 3}
    assert manager.guard_page_indices() == frozenset({3})
    assert torch.isnan(buffer[3]).all()
    assert torch.equal(buffer[2], torch.zeros(2))
    # Never committed: the reuse tree must never hand this page to a request.
    assert kv_cache.committing_stopped
    assert not kv_cache.closed


@pytest.mark.parametrize("failure", ["create", "resume", "resize", "no_layer"])
def test_reserve_guard_page_failure_releases_the_reservation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Every bail-out path stands the diagnostic down and frees what it took,
    so the page and the index slot can be reassigned to real requests."""
    buffer = torch.zeros(8, 2)
    kv_cache = (
        None
        if failure == "create"
        else _GuardKVCache(
            resume_ok=failure != "resume",
            resize_ok=failure != "resize",
        )
    )
    page = BAD_PAGE_INDEX if failure == "no_layer" else 3
    manager = _make_guard_manager(monkeypatch, kv_cache, buffer, page=page)

    manager._reserve_guard_page()

    assert manager._guard_page_fill == ""
    assert manager._guard_page_by_layer == {}
    assert manager.guard_page_indices() == frozenset()
    assert manager.kv_cache_map == {}
    assert torch.equal(buffer, torch.zeros(8, 2))
    if kv_cache is not None:
        assert kv_cache.closed
        manager.index_mapper.remove_sequence.assert_called_once_with(
            kv_cache_v2_module._GUARD_PAGE_REQUEST_ID
        )
        # The dummy create marked the id stats-excluded; the bail-out must undo
        # it, matching what free_resources does on the normal teardown.
        manager.impl.clear_stats_excluded.assert_called_once_with(
            kv_cache_v2_module._GUARD_PAGE_REQUEST_ID
        )


def test_guard_page_request_id_clears_cuda_graph_draft_window() -> None:
    """The guard id must not collide with the CUDA-graph dummy request range.

    Spec-decode capture uses ``CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len``;
    at ``runtime_draft_len == 1`` the old guard id ``(1 << 64) - 2`` collided and
    aborted capture with the "already exists" assert in ``_create_kv_cache``. The
    guard now sits a full reserved window below the dummy, clear of any draft id.
    """
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

    guard = kv_cache_v2_module._GUARD_PAGE_REQUEST_ID
    window = kv_cache_v2_module._CUDA_GRAPH_DUMMY_RESERVED_WINDOW
    assert guard == CUDA_GRAPH_DUMMY_REQUEST_ID - window
    # The regression: the draft dummy at length 1 used to be the guard id.
    assert CUDA_GRAPH_DUMMY_REQUEST_ID - 1 != guard
    # Margin far above any supported max_draft_len, so the whole dummy range
    # [dummy - max_draft_len, dummy] stays clear of the guard.
    assert guard < CUDA_GRAPH_DUMMY_REQUEST_ID - 4096
    assert guard in kv_cache_v2_module._RESERVED_REQUEST_IDS


def test_warmup_zeroing_preserves_guard_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """``check_invalid_values_in_kv_cache(fill_with_zero=True)`` runs during
    warmup and scrubs the cache; it must leave the guard-page sentinel intact,
    otherwise the startup warning claims a guard that no longer exists."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    guard_page = 3
    buffer = torch.ones(6, 2)
    buffer[guard_page] = float("nan")  # the guard sentinel

    manager = object.__new__(KVCacheManagerV2)
    manager.layer_offsets = {0: 0}
    manager.layer_specs = None
    manager.layer_to_pool_mapping_dict = {0: 0}
    manager.get_buffers = lambda layer_idx, kv_layout="NHD": buffer
    manager._guard_page_by_layer = {0: guard_page}
    manager._guard_page_value = float("nan")

    manager.check_invalid_values_in_kv_cache(fill_with_zero=True)

    assert torch.isnan(buffer[guard_page]).all()  # sentinel survives the scrub
    assert torch.equal(buffer[0], torch.zeros(2))  # everything else is zeroed
    assert torch.equal(buffer[5], torch.zeros(2))


@pytest.mark.cpu_only
@pytest.mark.parametrize("offload", [False, True])
def test_warmup_scan_leaves_host_backed_history_alone(offload: bool) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.layer_offsets = {0: 0, 1: 1}
    manager.layer_to_pool_mapping_dict = {0: 0, 1: 1}
    manager.layer_specs = None
    manager.get_buffers = lambda layer_idx, kv_layout="NHD": torch.ones(2)
    # Layer 0's KEY pages are offloaded history; layer 1's are device memory.
    manager.impl = SimpleNamespace(
        is_sparse=lambda layer_id, role: layer_id == LayerId(0) and role == Role.KEY
    )
    if offload:
        manager._sparse_kv_offload = SparseKvCacheOffloadConfig()

    scanned = [layer_id for layer_id, _ in manager._iter_cache_buffers_for_invalid_check()]

    # Nothing reads host history before a transfer or a write-back fills it.
    assert scanned == ([1] if offload else [0, 1])


class _FakeLogger:
    """Stand-in for tensorrt_llm.logger with a settable level and captured messages."""

    def __init__(self, level: str = "debug") -> None:
        self.level = level
        self.messages: list[str] = []

    def is_debug_enabled(self) -> bool:
        return self.level in ("trace", "debug", "verbose")

    def debug(self, *msg) -> None:
        self.messages.append(" ".join(str(m) for m in msg))


class _FakeWindowedKVCache:
    """Matches the production contract: get_base_page_indices() is padded to
    max_blocks_per_seq with BAD_PAGE_INDEX, and num_blocks bounds the entries
    that belong to the sequence."""

    _MAX_BLOCKS_PER_SEQ = 256

    def __init__(self, page_indices: dict[int, list[int]]) -> None:
        self._page_indices = page_indices

    @property
    def num_blocks(self) -> int:
        return max(len(indices) for indices in self._page_indices.values())

    def get_base_page_indices(self, layer_group_id, beam_id=DEFAULT_BEAM_INDEX):
        indices = self._page_indices[int(layer_group_id)]
        return array.array(
            "i", indices + [BAD_PAGE_INDEX] * (self._MAX_BLOCKS_PER_SEQ - len(indices))
        )


def _make_window_crossing_manager() -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.max_seq_len = 4096
    manager.tokens_per_block = 64
    # Group 0 is windowed at 2048; group 1 spans the whole sequence, so it is
    # not windowed and must never produce a crossing line.
    manager._windowed_layer_group_sizes = {0: 2048}
    return manager


def test_window_crossing_logs_once_with_capacity_and_blocks(monkeypatch) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(kv_cache_v2_module, "logger", fake_logger)

    manager = _make_window_crossing_manager()
    kv_cache = _FakeWindowedKVCache({0: [7, 8, 9]})
    request = SimpleNamespace(py_request_id=42)

    manager._log_window_crossing(request, kv_cache, 2047, 2049, "generation")

    assert len(fake_logger.messages) == 1
    message = fake_logger.messages[0]
    assert "request 42" in message
    assert "generation" in message
    assert "layer_group_id=0" in message
    assert "window=2048" in message
    assert "capacity 2047 -> 2049 tokens" in message
    assert "num_blocks=3" in message
    assert "block_range=[7, 9]" in message

    # A further growth on the same request stays quiet: the window was already
    # crossed, so the once-per-sequence-per-pool contract holds without state.
    manager._log_window_crossing(request, kv_cache, 2049, 2050, "generation")
    assert len(fake_logger.messages) == 1


@pytest.mark.parametrize(
    "pre_capacity,new_capacity",
    [
        (100, 2048),  # grows up to the window but does not cross it
        (2048, 2048),  # no growth at all
        (2049, 3000),  # already past the window
    ],
)
def test_window_crossing_is_silent_without_a_crossing(
    monkeypatch, pre_capacity: int, new_capacity: int
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(kv_cache_v2_module, "logger", fake_logger)

    manager = _make_window_crossing_manager()
    manager._log_window_crossing(
        SimpleNamespace(py_request_id=1),
        _FakeWindowedKVCache({0: [1]}),
        pre_capacity,
        new_capacity,
        "context",
    )

    assert fake_logger.messages == []


@pytest.mark.parametrize(
    "pre_capacity,new_capacity",
    [
        (100, 101),  # an ordinary decode step far below the window
        (2049, 2050),  # an ordinary decode step past the window
        (2048, 2048),  # no growth at all
    ],
)
def test_window_crossing_does_not_query_the_logger_without_a_crossing(
    monkeypatch, pre_capacity: int, new_capacity: int
) -> None:
    """The per-step common case must not ask the logger anything.

    ``_log_window_crossing`` runs for every generation request on every
    step, and ``is_debug_enabled`` resolves the caller's module from the
    stack, so consulting it before the integer crossing test puts that cost
    on every request of every iteration.
    """
    fake_logger = _FakeLogger()
    calls: list[None] = []

    def counting_is_debug_enabled() -> bool:
        calls.append(None)
        return True

    fake_logger.is_debug_enabled = counting_is_debug_enabled
    monkeypatch.setattr(kv_cache_v2_module, "logger", fake_logger)

    manager = _make_window_crossing_manager()
    manager._log_window_crossing(
        SimpleNamespace(py_request_id=1),
        _FakeWindowedKVCache({0: [1]}),
        pre_capacity,
        new_capacity,
        "generation",
    )

    assert calls == []
    assert fake_logger.messages == []


@pytest.mark.parametrize(
    "level,expected",
    [("trace", 1), ("debug", 1), ("verbose", 1), ("info", 0), ("warning", 0), ("error", 0)],
)
def test_window_crossing_follows_the_logger_severity_order(monkeypatch, level, expected) -> None:
    """Every level at least as verbose as debug logs; the rest stay quiet.

    ``verbose`` and ``trace`` are valid ways to ask for debug output, so a
    guard that compares the level to the literal ``"debug"`` suppresses the
    line for users who did enable it.
    """
    fake_logger = _FakeLogger(level=level)
    monkeypatch.setattr(kv_cache_v2_module, "logger", fake_logger)

    manager = _make_window_crossing_manager()
    manager._log_window_crossing(
        SimpleNamespace(py_request_id=1), _FakeWindowedKVCache({0: [1]}), 2047, 2049, "context"
    )

    assert len(fake_logger.messages) == expected


@pytest.mark.parametrize(
    "global_level,module_level,expected",
    [
        # A module raised to debug logs even though the global level is info.
        ("info", 10, 1),
        # A module pinned to info stays quiet even though the global level is debug.
        ("debug", 20, 0),
    ],
)
def test_window_crossing_honours_per_module_log_levels(
    monkeypatch, global_level: str, module_level: int, expected: int
) -> None:
    """The real logger, not a stand-in: the gate must resolve this module.

    ``TLLM_LOG_LEVEL_BY_MODULE=debug:_torch`` is how a user turns this
    diagnostic on without drowning in every other subsystem's debug output,
    and the manager's own module (``_torch``) is what the override names.
    """
    from tensorrt_llm.logger import logger as real_logger

    assert kv_cache_v2_module.logger is real_logger, "the manager must log through the real logger"
    recorded: list[str] = []
    monkeypatch.setattr(real_logger, "_min_severity", global_level)
    monkeypatch.setattr(real_logger, "_module_levels", {"_torch": module_level})
    monkeypatch.setattr(real_logger, "debug", lambda *msg: recorded.append(" ".join(map(str, msg))))

    manager = _make_window_crossing_manager()
    manager._log_window_crossing(
        SimpleNamespace(py_request_id=1), _FakeWindowedKVCache({0: [1]}), 2047, 2049, "context"
    )

    assert len(recorded) == expected


def test_window_crossing_survives_unreadable_page_indices(monkeypatch) -> None:
    """Diagnostics must never turn a working run into a failing one."""
    fake_logger = _FakeLogger()
    monkeypatch.setattr(kv_cache_v2_module, "logger", fake_logger)

    class _BrokenKVCache:
        def get_base_page_indices(self, layer_group_id, beam_id=DEFAULT_BEAM_INDEX):
            raise KeyError("no base buffer")

    manager = _make_window_crossing_manager()
    manager._log_window_crossing(
        SimpleNamespace(py_request_id=1), _BrokenKVCache(), 2047, 2049, "context"
    )

    assert len(fake_logger.messages) == 1
    assert "unavailable" in fake_logger.messages[0]


@pytest.mark.parametrize(
    "target_dtype,target_layers,draft_windows",
    [
        (DataType.BF16, 2, [None]),
        (DataType.FP8, 2, [None]),
        (DataType.FP8, 3, [128]),
        (DataType.BF16, 2, [128]),
    ],
    ids=["bf16_same_window", "fp8_same_window", "fp8_windowed_draft", "bf16_windowed_draft"],
)
def test_storage_pool_group_count_matches_manager(
    target_dtype, target_layers, draft_windows
) -> None:
    """The estimator's pool-group count is what the manager creates."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=target_dtype,
        num_kv_heads=8,
        head_dim=128,
        buffer_roles=(Role.KEY, Role.VALUE),
    )
    layer_specs = [replace(target, global_layer_id=i) for i in range(target_layers)]
    layer_specs += [
        replace(
            target,
            global_layer_id=target_layers + i,
            owner=KVCacheLayerOwner.DRAFT,
            dtype=DataType.BF16,
            sliding_window_size=window,
        )
        for i, window in enumerate(draft_windows)
    ]
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_gpu_total_bytes=64 << 20,
        host_cache_size=0,
        enable_swa_scratch_reuse=False,
    )
    manager = KVCacheManagerV2(
        kv_cache_config,
        CacheType.SELF,
        num_layers=len(layer_specs),
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=1024,
        max_batch_size=2,
        max_num_tokens=128,
        mapping=Mapping(),
        dtype=target_dtype,
        layer_specs=layer_specs,
    )
    try:
        assert _get_num_storage_pool_groups(layer_specs, 32) == len(manager.impl.pool_group_descs)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("disagg", [False, True], ids=["aggregated", "disaggregated"])
@pytest.mark.parametrize(
    "spec_config_cls,shared_draft_block_width",
    [(DFlashDecodingConfig, None), (DSparkDecodingConfig, 7), (DSparkDecodingConfig, 8)],
    ids=["dflash", "dspark_shifted", "dspark_unshifted"],
)
def test_shared_draft_pages_cover_the_drafter_block_after_a_decode_resize(
    disagg: bool, spec_config_cls: type, shared_draft_block_width: int | None
) -> None:
    """The drafter appends its block after the context, so its pages must reach it.

    After a decode step accepts every draft token the context is ``prompt + K +
    1`` long and the drafter's block follows it. The prompt is chosen so that
    this furthest write is the first position of a third page.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    max_draft_len = 7
    block_width = shared_draft_block_width or max_draft_len + 1
    tokens_per_block = 32
    prompt_len = 2 * tokens_per_block - (max_draft_len + 1) - block_width + 1
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=8,
        head_dim=128,
        buffer_roles=(Role.KEY, Role.VALUE),
    )
    draft_layer = 2
    layer_specs = [
        target,
        replace(target, global_layer_id=1),
        replace(
            target, global_layer_id=draft_layer, owner=KVCacheLayerOwner.DRAFT, dtype=DataType.BF16
        ),
    ]
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=len(layer_specs),
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=tokens_per_block,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.FP8,
        layer_specs=layer_specs,
        spec_config=spec_config_cls(max_draft_len=max_draft_len, speculative_model="draft"),
        shared_draft_block_width=shared_draft_block_width,
        vocab_size=4096,
        enable_stats=False,
    )
    request = _ContextRequest(
        1,
        list(range(prompt_len)),
        prompt_len,
        "conversation",
        use_conversation_params=False,
        is_disagg_generation_init_state=disagg,
    )
    try:
        assert manager._shared_draft_block_width == block_width
        assert manager._generation_kv_capacity_headroom == max_draft_len + 1 + block_width
        if disagg:
            request.py_draft_tokens = []
            request.is_disagg_generation_transmission_complete = False
            request.context_phase_params = None
            assert manager.prepare_disagg_gen_init(request)
            request.is_disagg_generation_init_state = False
        else:
            _run_context(manager, request)
        generation = SimpleNamespace(
            py_request_id=request.py_request_id,
            py_beam_width=1,
            py_draft_tokens=[0] * max_draft_len,
            state=LlmRequestState.GENERATION_IN_PROGRESS,
            is_dummy_request=False,
            is_disagg_generation_transmission_complete=False,
            context_phase_params=None,
            max_beam_num_tokens=prompt_len + 1,
        )
        assert manager.try_allocate_generation(generation)

        furthest_write_end = prompt_len + max_draft_len + 1 + block_width
        pages = manager.get_batch_cache_indices([request.py_request_id], layer_idx=draft_layer)[0]
        assert len(pages) * tokens_per_block >= furthest_write_end
    finally:
        manager.free_resources(request)
        manager.shutdown()


def test_cost_headroom_counts_the_drafter_block_only_for_shared_layers() -> None:
    """The budget charges the allocation headroom, not the position-side count."""
    from tensorrt_llm._torch.speculative import get_num_extra_kv_tokens

    max_draft_len = 7
    spec_config = DFlashDecodingConfig(max_draft_len=max_draft_len, speculative_model="draft")
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.BF16,
        num_kv_heads=8,
        head_dim=128,
        buffer_roles=(Role.KEY, Role.VALUE),
    )
    draft = replace(target, global_layer_id=1, owner=KVCacheLayerOwner.DRAFT)

    _, own = _get_generation_kv_capacity(spec_config, is_draft=False, layer_specs=[target])
    _, shared = _get_generation_kv_capacity(
        spec_config, is_draft=False, layer_specs=[target, draft]
    )

    # The block is K + 1 = 8 wide against K - 1 = 6 extra positions.
    assert shared - own == (max_draft_len + 1) - (max_draft_len - 1)
    assert _get_kv_alloc_extra_tokens(spec_config, [target, draft]) == max_draft_len + 1
    assert get_num_extra_kv_tokens(spec_config) == max_draft_len - 1


def test_generation_token_slice_shares_token_view() -> None:
    token_view = np.arange(10, dtype=np.int32)
    request = SimpleNamespace(
        get_tokens_view=Mock(return_value=token_view),
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
    )
    manager = object.__new__(KVCacheManagerV2)

    source = manager._reuse_token_source(request)
    result = manager._augment_tokens_for_block_reuse(source, request, start=7, end=10)

    assert source is token_view
    np.testing.assert_array_equal(result, [7, 8, 9])
    assert np.shares_memory(result, token_view)
    request.get_tokens_view.assert_called_once_with(DEFAULT_BEAM_INDEX)


# Only a range that still overlaps the prompt can hold multimodal items to augment.
@pytest.mark.parametrize(("prompt_len", "augments"), [(10, False), (12, True)])
def test_try_commit_generation_blocks_commits_only_materialized_prefix(
    prompt_len: int, augments: bool
) -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        is_context_only_request=False,
        prompt_len=prompt_len,
        max_beam_num_tokens=15,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        get_tokens_view=lambda beam_id: np.arange(15, dtype=np.int32),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=10)
    manager = _make_publishing_manager(BlockReusePolicy.ALL_REUSABLE)
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = Mock(
        side_effect=lambda tokens, request, start, end: tokens[start:end]
    )

    manager.try_commit_generation_blocks(request)

    assert manager._augment_tokens_for_block_reuse.called is augments
    assert list(kv_cache.committed_tokens) == [10, 11, 12, 13]
    assert kv_cache.num_committed_tokens == 14
    assert not kv_cache.commit_is_end


def test_try_commit_generation_blocks_finalizes_without_new_tokens() -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        is_context_only_request=False,
        max_beam_num_tokens=11,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    kv_cache = _FakeKVCache(num_committed_tokens=10)
    manager = _make_publishing_manager(BlockReusePolicy.ALL_REUSABLE)
    manager.kv_cache_map = {request.py_request_id: kv_cache}

    manager.try_commit_generation_blocks(request)

    assert kv_cache.committed_tokens == []
    assert kv_cache.num_committed_tokens == 10
    assert kv_cache.commit_is_end
    assert kv_cache.stopped_committing


@pytest.mark.parametrize(
    ("manager_overrides", "request_overrides", "has_cache"),
    [
        ({"enable_block_reuse": False}, {}, True),
        ({"block_reuse_policy": BlockReusePolicy.PER_REQUEST}, {}, True),
        ({"kv_compression_manages_history": True}, {}, True),
        ({"kv_connector_manager": Mock()}, {}, True),
        ({"_sharing": Mock()}, {}, True),
        ({"is_draft": True}, {}, True),
        ({"max_beam_width": 2}, {}, True),
        ({}, {"is_dummy_request": True}, True),
        ({}, {"is_context_only_request": True}, True),
        ({}, {"state": LlmRequestState.CONTEXT_INIT}, True),
        ({}, {}, False),
    ],
)
def test_try_commit_generation_blocks_honors_guards(
    manager_overrides: dict[str, object],
    request_overrides: dict[str, object],
    has_cache: bool,
) -> None:
    request_fields = {
        "py_request_id": 1,
        "is_dummy_request": False,
        "is_context_only_request": False,
        "max_beam_num_tokens": 15,
        "state": LlmRequestState.GENERATION_IN_PROGRESS,
    }
    request_fields.update(request_overrides)
    request = SimpleNamespace(**request_fields)
    kv_cache = _FakeKVCache(num_committed_tokens=10)
    manager = _make_publishing_manager(BlockReusePolicy.ALL_REUSABLE)
    manager.kv_cache_map = {request.py_request_id: kv_cache} if has_cache else {}
    for name, value in manager_overrides.items():
        setattr(manager, name, value)

    manager.try_commit_generation_blocks(request)

    assert kv_cache.committed_tokens is None


@pytest.mark.parametrize("role", [Role.VALUE, Role.KEY_BLOCK_SCALE])
def test_default_attention_rejects_split_coupled_buffers(monkeypatch, role):
    original = KVCacheManagerV2._build_cache_config

    def build_config(self, config):
        config = original(self, config)
        layers = list(config.layers)
        layer = layers[0]
        buffers = [
            BufferConfig(
                role=buffer.role,
                size=buffer.size,
                reuse_policy=ReusePolicy.PRIVATE if buffer.role == role else None,
            )
            for buffer in layer.buffers
        ]
        if role == Role.KEY_BLOCK_SCALE:
            buffers.append(BufferConfig(role=role, size=128, reuse_policy=ReusePolicy.PRIVATE))
        layers[0] = AttentionLayerConfig(layer_id=layer.layer_id, buffers=buffers)
        return replace(config, layers=layers)

    monkeypatch.setattr(KVCacheManagerV2, "_build_cache_config", build_config)
    with pytest.raises(ValueError, match="requires one shared page-index table"):
        KVCacheManagerV2(
            KvCacheConfig(max_gpu_total_bytes=4 << 20, host_cache_size=0),
            CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=1,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.HALF,
            vocab_size=16,
        )
