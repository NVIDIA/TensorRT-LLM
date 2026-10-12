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

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
    _get_layer_spec_cache_size_components,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy, GpuCacheTierConfig


def _layer(
    global_layer_id: int,
    owner: KVCacheLayerOwner,
    *,
    dtype: DataType = DataType.HALF,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    window: int | None = None,
    sinks: int | None = None,
    reuse_policy: AttentionReusePolicy = AttentionReusePolicy.REQUIRED,
) -> KVCacheLayerSpec:
    roles = (Role.KEY, Role.VALUE)
    if dtype == DataType.NVFP4:
        roles += (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
    return KVCacheLayerSpec(
        global_layer_id=global_layer_id,
        owner=owner,
        dtype=dtype,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        buffer_roles=roles,
        sliding_window_size=window,
        num_sink_tokens=sinks,
        reuse_policy=reuse_policy,
    )


@pytest.mark.cpu_only
def test_layer_specs_describe_combined_cache_cost() -> None:
    layer_specs = (
        _layer(0, KVCacheLayerOwner.TARGET, window=64, sinks=4),
        _layer(
            1,
            KVCacheLayerOwner.DRAFT,
            reuse_policy=AttentionReusePolicy.OPTIONAL,
        ),
    )

    assert _get_layer_spec_cache_size_components(layer_specs) == ([1024, 1024], [64, None])


@pytest.mark.cpu_only
def test_layer_specs_reject_duplicate_global_layer_ids() -> None:
    with pytest.raises(ValueError, match="duplicate global layer IDs"):
        _get_layer_spec_cache_size_components(
            (
                _layer(0, KVCacheLayerOwner.TARGET),
                _layer(0, KVCacheLayerOwner.DRAFT),
            )
        )


@pytest.mark.cpu_only
def test_nvfp4_layer_spec_accounts_for_scale_buffers() -> None:
    layer = _layer(0, KVCacheLayerOwner.TARGET, dtype=DataType.NVFP4)

    assert layer.bytes_per_token(Role.KEY) == 128
    assert layer.bytes_per_token(Role.KEY_BLOCK_SCALE) == 16
    assert layer.bytes_per_token() == 288


@pytest.mark.cpu_only
def test_base_config_is_derived_from_explicit_layer_specs() -> None:
    layer_specs = (
        _layer(0, KVCacheLayerOwner.TARGET, window=64, sinks=4),
        _layer(
            1,
            KVCacheLayerOwner.DRAFT,
            reuse_policy=AttentionReusePolicy.OPTIONAL,
        ),
    )
    manager = object.__new__(KVCacheManagerV2)
    manager.layer_specs = layer_specs
    manager.enable_swa_scratch_reuse = False
    manager.num_extra_kv_tokens = 0
    manager.max_beam_width = 1
    manager.max_seq_len = 128
    manager.max_batch_size = 2
    manager.max_cuda_graph_batch_size = None
    manager.max_num_tokens = 64
    manager.enable_partial_reuse = True
    manager.reuse_match_backoff = 0
    manager.enable_stats = False
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager._ledger_tokens_per_block = 16

    config = manager._build_base_config(
        KvCacheConfig(),
        tokens_per_block=16,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
    )

    assert [layer.sliding_window_size for layer in config.layers] == [64, None]
    assert [layer.num_sink_tokens for layer in config.layers] == [4, None]
    assert [layer.reuse_policy for layer in config.layers] == [
        AttentionReusePolicy.REQUIRED,
        AttentionReusePolicy.OPTIONAL,
    ]
    assert [
        [(buffer.role, buffer.size) for buffer in layer.buffers] for layer in config.layers
    ] == [
        [(Role.KEY, 8192), (Role.VALUE, 8192)],
        [(Role.KEY, 8192), (Role.VALUE, 8192)],
    ]


@pytest.mark.cpu_only
def test_manager_reports_explicit_target_and_draft_layers() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.layer_specs = (
        _layer(10, KVCacheLayerOwner.TARGET),
        _layer(11, KVCacheLayerOwner.DRAFT),
    )
    manager.pp_layers = [10, 11]
    manager.layer_offsets = {10: 0, 11: 1}
    manager._legacy_draft_layer_ids = set()

    assert manager.target_layer_ids == (10,)
    assert manager.draft_layer_ids == (11,)
    assert manager.get_layer_dtype(11) == DataType.HALF
    assert manager.get_layer_buffer_roles(11) == (Role.KEY, Role.VALUE)


@pytest.mark.cpu_only
def test_layer_spec_rejects_nonstandard_buffer_roles() -> None:
    with pytest.raises(ValueError, match="unsupported standard buffer roles"):
        KVCacheLayerSpec(
            global_layer_id=0,
            owner=KVCacheLayerOwner.TARGET,
            dtype=DataType.HALF,
            num_kv_heads=4,
            head_dim=64,
            buffer_roles=(Role.KEY, Role.INDEX_KEY),
        )
