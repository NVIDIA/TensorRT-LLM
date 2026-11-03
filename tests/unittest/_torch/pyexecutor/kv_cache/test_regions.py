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
"""Laying persistent-state regions out on KV cache manager V2 layer configs."""

from dataclasses import dataclass

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.regions import (
    AllocationUnit,
    HandOff,
    Plane,
    Region,
    RegionLayer,
    RegionLayout,
    RegionLifeCycle,
    RegionProvider,
    StorageOwner,
    lay_out_regions,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    GpuCacheTierConfig,
    KVCacheManagerConfig,
    LayerId,
    ReusePolicy,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only

KEY = DataRole("key")
SSM = DataRole("ssm_state")
CONV = DataRole("conv_state")
RING = DataRole("ring")

SLOT_WITH_SNAPSHOTS = RegionLifeCycle(AllocationUnit.REQUEST, HandOff.DECLARED_POINTS)
FULL_HISTORY = RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.BLOCK_BOUNDARIES)
RING_OF_19 = RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.BLOCK_BOUNDARIES, window=19)


def _layer_summary(layer):
    """Every V2 layer-config field a region translation sets, in comparable form."""
    buffers = [
        (str(buffer.role), buffer.size, buffer.coalescing_group, buffer.reuse_policy)
        for buffer in layer.buffers
    ]
    if isinstance(layer, SsmLayerConfig):
        return ("ssm", int(layer.layer_id), buffers, layer.reuse_policy)
    return (
        "attention",
        int(layer.layer_id),
        buffers,
        layer.sliding_window_size,
        layer.num_sink_tokens,
        layer.reuse_policy,
    )


def _base_layers(num_layers):
    return [
        AttentionLayerConfig(layer_id=LayerId(i), buffers=[BufferConfig(role=KEY, size=64)])
        for i in range(num_layers)
    ]


def _hybrid_regions():
    """Recurrent state on layers 0 and 2; a latent and a ring on layers 1 and 3."""
    return [
        Region(
            "state",
            SLOT_WITH_SNAPSHOTS,
            tuple(RegionLayer(i, (Plane(SSM, 512), Plane(CONV, 96))) for i in (0, 2)),
        ),
        Region("latent", FULL_HISTORY, tuple(RegionLayer(i, (Plane(KEY, 128),)) for i in (1, 3))),
        Region("ring", RING_OF_19, tuple(RegionLayer(i, (Plane(RING, 76),)) for i in (1, 3))),
    ]


@pytest.mark.parametrize(
    "allocation, hand_off, window",
    [
        (AllocationUnit.REQUEST, HandOff.BLOCK_BOUNDARIES, None),
        (AllocationUnit.TOKEN_BLOCK, HandOff.DECLARED_POINTS, None),
        (AllocationUnit.REQUEST, HandOff.DECLARED_POINTS, 16),
        (AllocationUnit.TOKEN_BLOCK, HandOff.BLOCK_BOUNDARIES, 0),
    ],
)
def test_life_cycle_rejects_what_v2_storage_cannot_express(allocation, hand_off, window):
    with pytest.raises(ValueError):
        RegionLifeCycle(allocation, hand_off, window)


def test_declarations_reject_empty_and_repeated_storage():
    with pytest.raises(ValueError, match="positive size"):
        Plane(KEY, 0)
    with pytest.raises(ValueError, match="no planes"):
        RegionLayer(0, ())
    with pytest.raises(ValueError, match="repeats a plane role"):
        RegionLayer(0, (Plane(KEY, 8), Plane(KEY, 16)))
    with pytest.raises(ValueError, match="covers a model layer twice"):
        Region("r", FULL_HISTORY, (RegionLayer(0, (Plane(KEY, 8),)),) * 2)


def test_no_regions_keeps_the_manager_layers():
    base = _base_layers(3)

    layers, layout = lay_out_regions([], base)

    assert [_layer_summary(layer) for layer in layers] == [_layer_summary(layer) for layer in base]
    assert layout.owners == {}
    assert layout.owner(1) is None
    assert layout.model_layer_of(2) == 2


def test_regions_take_their_model_layers_and_append_shared_ones():
    layers, layout = lay_out_regions(_hybrid_regions(), _base_layers(4))

    assert [_layer_summary(layer) for layer in layers] == [
        (
            "ssm",
            0,
            [("ssm_state", 512, 0, None), ("conv_state", 96, 0, None)],
            ReusePolicy.REQUIRED,
        ),
        ("attention", 1, [("key", 128, 0, None)], None, None, ReusePolicy.REQUIRED),
        (
            "ssm",
            2,
            [("ssm_state", 512, 0, None), ("conv_state", 96, 0, None)],
            ReusePolicy.REQUIRED,
        ),
        ("attention", 3, [("key", 128, 0, None)], None, None, ReusePolicy.REQUIRED),
        # The ring shares model layers 1 and 3 with the latent, so it is stored
        # after the model layers, in its own declaration order.
        ("attention", 4, [("ring", 76, 0, None)], 19, None, ReusePolicy.REQUIRED),
        ("attention", 5, [("ring", 76, 0, None)], 19, None, ReusePolicy.REQUIRED),
    ]
    assert layout.owner(0) == StorageOwner("state", 0)
    assert layout.owner(3) == StorageOwner("latent", 3)
    assert layout.owner(5) == StorageOwner("ring", 3)
    assert layout.layer_ids("ring") == (LayerId(4), LayerId(5))
    assert layout.is_model_layer(3) and not layout.is_model_layer(4)
    assert layout.model_layer_of(4) == 1
    assert layout.region("ring").life_cycle.window == 19


def test_private_regions_never_reach_the_prefix_index():
    private_slot = RegionLifeCycle(AllocationUnit.REQUEST, HandOff.NEVER)
    private_span = RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.NEVER, window=8)
    regions = [
        Region("slot", private_slot, (RegionLayer(0, (Plane(SSM, 32),)),)),
        Region("span", private_span, (RegionLayer(1, (Plane(KEY, 32),)),)),
    ]

    layers, _ = lay_out_regions(regions, _base_layers(2))

    assert [layer.reuse_policy for layer in layers] == [ReusePolicy.PRIVATE] * 2


def test_layout_rejects_unknown_layers_and_duplicate_names():
    with pytest.raises(ValueError, match="model layer 4"):
        lay_out_regions(
            [Region("r", FULL_HISTORY, (RegionLayer(4, (Plane(KEY, 8),)),))], _base_layers(4)
        )
    region = Region("r", FULL_HISTORY, (RegionLayer(0, (Plane(KEY, 8),)),))
    with pytest.raises(ValueError, match="unique"):
        lay_out_regions([region, region], _base_layers(1))


def test_renumbering_follows_the_compacted_layer_list():
    regions = [
        Region("latent", FULL_HISTORY, (RegionLayer(1, (Plane(KEY, 128),)),)),
        Region("ring", RING_OF_19, (RegionLayer(1, (Plane(RING, 76),)),)),
    ]
    _, layout = lay_out_regions(regions, _base_layers(4))
    assert layout.owners == {1: StorageOwner("latent", 1), 4: StorageOwner("ring", 1)}

    # Model layer 3 had no buffers and was dropped; the ring moves down with it.
    renumbered = layout.renumbered([0, 1, 2, 4])

    assert renumbered.num_model_layers == 3
    assert renumbered.owners == {1: StorageOwner("latent", 1), 3: StorageOwner("ring", 1)}
    with pytest.raises(ValueError, match="dropped"):
        layout.renumbered([0, 2, 3, 4])


@dataclass
class _Provider:
    regions: list

    def declare_regions(self):
        return self.regions


def _manager_stub(providers):
    mgr = object.__new__(KVCacheManagerV2)
    mgr._region_providers = lambda: providers
    mgr.num_local_layers = 4
    return mgr


def _config(layers):
    return KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=layers,
    )


def test_manager_without_regions_keeps_its_config():
    mgr = _manager_stub([])
    config = _config(_base_layers(4))

    assert mgr._lay_out_regions(config) is config
    assert not config.commit_min_snapshot
    assert mgr.region_layout.owners == {}
    assert mgr.region_layout.num_model_layers == 4


def test_manager_stores_its_providers_regions():
    provider = _Provider(_hybrid_regions())
    assert isinstance(provider, RegionProvider)
    mgr = _manager_stub([provider])

    config = mgr._lay_out_regions(_config(_base_layers(4)))

    assert [int(layer.layer_id) for layer in config.layers] == [0, 1, 2, 3, 4, 5]
    assert isinstance(config.layers[2], SsmLayerConfig)
    # The runtime rejects per-request storage without minimum-snapshot commits.
    assert config.commit_min_snapshot
    assert config.layers[4].sliding_window_size == 19
    assert isinstance(mgr.region_layout, RegionLayout)
    assert mgr.region_layout.owner(4) == StorageOwner("ring", 1)


def test_dropping_an_empty_layer_keeps_region_ownership_consistent():
    regions = [
        Region("latent", FULL_HISTORY, (RegionLayer(1, (Plane(KEY, 128),)),)),
        Region("ring", RING_OF_19, (RegionLayer(1, (Plane(RING, 76),)),)),
    ]
    mgr = _manager_stub([_Provider(regions)])
    base = _base_layers(3)
    # Model layer 2 has no cache storage on this rank.
    base[2] = AttentionLayerConfig(layer_id=LayerId(2), buffers=[BufferConfig(role=KEY, size=0)])
    mgr.num_local_layers = 3
    mgr.pp_layers = [10, 11, 12]
    mgr.num_kv_heads_per_layer = [1, 1, 0]
    mgr.head_dim_per_layer = [64, 64, 64]
    mgr.max_attention_window_vec = [256, 256, 256]

    config = mgr._remove_zero_size_buffers(mgr._lay_out_regions(_config(base)))

    assert [int(layer.layer_id) for layer in config.layers] == [0, 1, 2]
    assert [str(layer.buffers[0].role) for layer in config.layers] == ["key", "key", "ring"]
    assert mgr.pp_layers == [10, 11]
    assert mgr.num_local_layers == 2
    assert mgr.region_layout.num_model_layers == 2
    assert mgr.region_layout.owner(2) == StorageOwner("ring", 1)
