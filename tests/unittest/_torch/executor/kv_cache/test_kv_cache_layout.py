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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheLayerView,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.layout import (
    KVCacheLayoutPlan,
    LayerBufferSpec,
    LayerCacheOwner,
    LayerCacheSpec,
    LayerCacheTransferPolicy,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy, GpuCacheTierConfig


def _layer(
    local_layer_id: int,
    owner: LayerCacheOwner,
    *,
    dtype: DataType = DataType.HALF,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    window: int | None = None,
    sinks: int | None = None,
    reuse_policy: AttentionReusePolicy = AttentionReusePolicy.REQUIRED,
) -> LayerCacheSpec:
    return LayerCacheSpec(
        local_layer_id=local_layer_id,
        global_layer_id=local_layer_id,
        owner=owner,
        dtype=dtype,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        buffers=(
            LayerBufferSpec(role=Role.KEY, size=1024),
            LayerBufferSpec(role=Role.VALUE, size=1024),
        ),
        sliding_window_size=window,
        num_sink_tokens=sinks,
        reuse_policy=reuse_policy,
    )


@pytest.mark.cpu_only
def test_layout_plan_builds_exact_runtime_layer_configs() -> None:
    plan = KVCacheLayoutPlan(
        tokens_per_block=16,
        layers=(
            _layer(0, LayerCacheOwner.TARGET, window=64, sinks=4),
            _layer(
                1,
                LayerCacheOwner.DRAFT,
                reuse_policy=AttentionReusePolicy.OPTIONAL,
            ),
        ),
    )

    runtime_layers = plan.to_attention_layer_configs()

    assert plan.target_layer_ids == (0,)
    assert plan.draft_layer_ids == (1,)
    assert runtime_layers[0].sliding_window_size == 64
    assert runtime_layers[0].num_sink_tokens == 4
    assert runtime_layers[1].reuse_policy == AttentionReusePolicy.OPTIONAL
    assert [(buffer.role, buffer.size) for buffer in runtime_layers[1].buffers] == [
        (Role.KEY, 1024),
        (Role.VALUE, 1024),
    ]
    assert plan.cache_size_layer_components() == ([128, 128], [64, None])


@pytest.mark.cpu_only
def test_layout_plan_rejects_non_dense_local_layer_ids() -> None:
    with pytest.raises(ValueError, match="dense and ordered"):
        KVCacheLayoutPlan(
            tokens_per_block=16,
            layers=(
                _layer(0, LayerCacheOwner.TARGET),
                LayerCacheSpec(
                    local_layer_id=2,
                    global_layer_id=2,
                    owner=LayerCacheOwner.DRAFT,
                    dtype=DataType.HALF,
                    num_kv_heads=4,
                    head_dim=64,
                    buffers=(LayerBufferSpec(role=Role.KEY, size=1024),),
                ),
            ),
        )


@pytest.mark.cpu_only
def test_transfer_compatibility_reports_mixed_required_geometry() -> None:
    plan = KVCacheLayoutPlan(
        tokens_per_block=16,
        layers=(
            _layer(0, LayerCacheOwner.TARGET),
            _layer(
                1,
                LayerCacheOwner.DRAFT,
                dtype=DataType.FP8,
                num_kv_heads=2,
                head_dim=128,
            ),
        ),
    )

    assert plan.transfer_compatibility_errors() == (
        "required transfer layers use different dtypes",
        "required transfer layers use different KV-head counts",
        "required transfer layers use different head dimensions",
    )


@pytest.mark.cpu_only
def test_base_config_is_derived_from_explicit_layout_plan() -> None:
    plan = KVCacheLayoutPlan(
        tokens_per_block=16,
        layers=(
            _layer(0, LayerCacheOwner.TARGET, window=64, sinks=4),
            _layer(1, LayerCacheOwner.DRAFT),
        ),
    )
    manager = object.__new__(KVCacheManagerV2)
    manager._explicit_layout_plan = plan
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
    assert [
        [(buffer.role, buffer.size) for buffer in layer.buffers] for layer in config.layers
    ] == [
        [(Role.KEY, 1024), (Role.VALUE, 1024)],
        [(Role.KEY, 1024), (Role.VALUE, 1024)],
    ]


@pytest.mark.cpu_only
def test_draft_layer_view_does_not_own_lifecycle_or_transfer() -> None:
    plan = KVCacheLayoutPlan(
        tokens_per_block=16,
        layers=(
            _layer(0, LayerCacheOwner.TARGET),
            _layer(1, LayerCacheOwner.DRAFT),
        ),
    )
    manager = SimpleNamespace(kv_cache_layout_plan=plan)
    manager.get_layer_cache_spec = plan.get_layer
    manager.get_buffers = Mock(return_value="draft-buffer")
    view = KVCacheLayerView(manager, LayerCacheOwner.DRAFT)

    assert view.layer_ids == (1,)
    assert view.get_buffers(1) == "draft-buffer"
    view.free_resources(Mock())
    with pytest.raises(RuntimeError, match="owning unified manager"):
        view.report_batch_to_connector(Mock())
    with pytest.raises(ValueError, match="does not belong"):
        view.get_buffers(0)


@pytest.mark.cpu_only
def test_local_only_buffer_does_not_gate_transfer() -> None:
    layer = _layer(0, LayerCacheOwner.TARGET)
    draft = _layer(1, LayerCacheOwner.DRAFT)
    draft = LayerCacheSpec(
        local_layer_id=draft.local_layer_id,
        global_layer_id=draft.global_layer_id,
        owner=draft.owner,
        dtype=draft.dtype,
        num_kv_heads=draft.num_kv_heads,
        head_dim=draft.head_dim,
        buffers=draft.buffers
        + (
            LayerBufferSpec(
                role=Role.INDEX_KEY,
                size=256,
                transfer_policy=LayerCacheTransferPolicy.LOCAL_ONLY,
            ),
        ),
    )
    plan = KVCacheLayoutPlan(tokens_per_block=16, layers=(layer, draft))

    assert plan.transfer_compatibility_errors() == ()
