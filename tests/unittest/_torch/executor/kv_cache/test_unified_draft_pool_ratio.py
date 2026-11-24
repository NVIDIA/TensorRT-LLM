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
"""The explicit pool ratio of a KVCacheManagerV2 that owns a standalone draft cache.

The runtime takes ``initial_pool_ratio`` by layer group ID. The manager inserts
the draft's share at the draft layer group's ID and keeps every target ratio at
its own layer group's ID, whatever order the pool groups sort in.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    AttentionLayerConfig,
    AttentionReusePolicy,
    BufferConfig,
    CacheDomain,
    GpuCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    SsmLayerConfig,
    _introspection,
)

TARGET_RATIOS = [0.45, 0.4, 0.13, 0.02]
# Layer groups 1 and 2 hold every token; group 1 is the scarcer, with 0.4 of the
# target share for 8192 B per block. A 2048 B draft block matches its block
# count with d / 2048 == (1 - d) * 0.4 / 8192, so d = 1/11.
DRAFT_RATIO = 1 / 11
LAYER_GROUP_RATIOS = [ratio * (1 - DRAFT_RATIO) for ratio in TARGET_RATIOS] + [DRAFT_RATIO]


def k3_like_cache_config(pool_ratio):
    """A runtime config laid out like K3's context-server cache.

    Layer groups by ID: 0 recurrent state, 1 and 2 full-attention MLA (2 under
    another reuse policy), 3 a windowed high-precision tail, 4 the standalone
    draft. Pool groups sort by cache domain, reuse policy and slot sizes, an
    order unrelated to the layer group IDs.
    """
    return KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
        layers=[
            SsmLayerConfig(layer_id=0, buffers=[BufferConfig(role="ssm_state", size=1 << 20)]),
            AttentionLayerConfig(layer_id=1, buffers=[BufferConfig(role=Role.KEY, size=8192)]),
            AttentionLayerConfig(
                layer_id=2,
                buffers=[BufferConfig(role=Role.KEY, size=1024)],
                reuse_policy=AttentionReusePolicy.OPTIONAL,
            ),
            AttentionLayerConfig(
                layer_id=3,
                buffers=[BufferConfig(role=Role.KEY, size=4096)],
                sliding_window_size=65,
                num_sink_tokens=32,
            ),
            AttentionLayerConfig(
                layer_id=4,
                buffers=[BufferConfig(role=Role.KEY, size=2048)],
                cache_domain=CacheDomain.STANDALONE_DRAFT,
            ),
        ],
        initial_pool_ratio=pool_ratio,
        enable_partial_reuse=False,
        commit_min_snapshot=True,
    )


def configure(config):
    manager = object.__new__(KVCacheManagerV2)
    manager._standalone_draft_layer_ids = (61,)
    manager.max_seq_len = 1 << 20
    return manager._configure_unified_draft_pool_ratio(config)


@pytest.mark.cpu_only
def test_draft_share_goes_to_the_draft_layer_group():
    config = configure(k3_like_cache_config(TARGET_RATIOS))
    assert config.initial_pool_ratio == pytest.approx(LAYER_GROUP_RATIOS)


@pytest.mark.cpu_only
def test_pool_ratio_needs_one_entry_per_target_layer_group():
    with pytest.raises(ValueError, match="one entry per target layer group"):
        configure(k3_like_cache_config([0.25, 0.25, 0.5]))


def test_runtime_pools_get_their_layer_groups_ratios():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    config = configure(k3_like_cache_config(TARGET_RATIOS))
    manager = KVCacheManager(config)
    try:
        layer_groups = [manager.get_layer_group_id(layer.layer_id) for layer in config.layers]
        assert layer_groups == list(range(5))
        pool_groups = manager.get_life_cycle_pool_group_indices(GPU_LEVEL)
        assert sorted(pool_groups) == list(range(5))
        assert pool_groups != list(range(5)), "the layout must sort pool groups out of ID order"
        expected = [0.0] * len(pool_groups)
        for layer_group, pool_group in enumerate(pool_groups):
            expected[pool_group] = LAYER_GROUP_RATIOS[layer_group]
        assert _introspection.current_gpu_ratio(manager) == pytest.approx(expected, abs=0.01)
    finally:
        manager.shutdown()
