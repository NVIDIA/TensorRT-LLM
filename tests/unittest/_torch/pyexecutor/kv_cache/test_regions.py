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
"""Laying persistent-state regions out as native KV cache manager V2 region configs."""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.regions import (
    AllocationUnit,
    HandOff,
    Plane,
    Region,
    RegionLayer,
    RegionLayout,
    RegionLifeCycle,
    lay_out_regions,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    GpuCacheTierConfig,
    KVCacheManagerConfig,
    LayerId,
    RegionAllocationUnit,
    ReusePolicy,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only

KEY = DataRole("key")
SSM = DataRole("ssm_state")
CONV = DataRole("conv_state")
HISTORY = DataRole("history")
COUNT = DataRole("count")

PRIVATE_SLOT = RegionLifeCycle(AllocationUnit.REQUEST, HandOff.NEVER)
SNAPSHOTTED_SLOT = RegionLifeCycle(AllocationUnit.REQUEST, HandOff.DECLARED_POINTS)
RING = RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.BLOCK_BOUNDARIES, window=19)


def _layers():
    """Attention on layer 0, recurrent state on layers 1 and 2."""
    return [
        AttentionLayerConfig(layer_id=LayerId(0), buffers=[BufferConfig(role=KEY, size=4096)]),
        *(
            SsmLayerConfig(
                layer_id=LayerId(layer),
                buffers=[BufferConfig(role=SSM, size=4096), BufferConfig(role=CONV, size=1024)],
            )
            for layer in (1, 2)
        ),
    ]


def _replay(name="replay", fixed_slot_count=5, layers=(1, 2)):
    return Region(
        name,
        PRIVATE_SLOT,
        tuple(
            RegionLayer(
                layer,
                (Plane(HISTORY, 512, coalescing_group=1),)
                + ((Plane(COUNT, 16),) if layer == layers[0] else ()),
            )
            for layer in layers
        ),
        fixed_slot_count=fixed_slot_count,
    )


def test_request_region_lays_out_as_a_native_region():
    """Each plane becomes a buffer of the native region on its layer, in declaration order, with the region's
    PRIVATE policy and fixed slot count; the layer configs are left alone."""
    configs, layout = lay_out_regions([_replay()], _layers())
    (config,) = configs
    assert config.name == "replay"
    assert config.allocation_unit == RegionAllocationUnit.REQUEST
    assert config.reuse_policy == ReusePolicy.PRIVATE
    assert config.fixed_slot_count == 5
    assert [
        (int(b.layer_id), str(b.buffer.role), b.buffer.size, b.buffer.coalescing_group)
        for b in config.buffers
    ] == [(1, "history", 512, 1), (1, "count", 16, 0), (2, "history", 512, 1)]
    assert all(b.buffer.reuse_policy is None for b in config.buffers)

    assert layout.regions == (_replay(),)
    assert layout.owner(1, COUNT).name == "replay"
    assert layout.owner(1, SSM) is None
    assert layout.layer_ids("replay") == (LayerId(1), LayerId(2))
    assert layout.roles("replay", 1) == (HISTORY, COUNT)
    assert layout.roles("replay", 2) == (HISTORY,)
    assert layout.region("replay").bytes_per_slot() == 512 + 16 + 512


def _ring(layers=(0,)):
    return Region(
        "ring", RING, tuple(RegionLayer(layer, (Plane(HISTORY, 2048),)) for layer in layers)
    )


def test_token_block_region_lays_out_as_a_native_region():
    """A token-block region handed off at block boundaries lowers to a REQUIRED token-block region with its
    window, which an attention layer with no buffers of its own hosts."""
    layers = [AttentionLayerConfig(layer_id=LayerId(0), buffers=[]), *_layers()[1:]]
    configs, layout = lay_out_regions([_ring(), _replay()], layers)
    ring, replay = configs
    assert ring.allocation_unit == RegionAllocationUnit.TOKEN_BLOCK
    assert ring.reuse_policy == ReusePolicy.REQUIRED
    assert ring.window_size == 19
    assert ring.fixed_slot_count is None
    assert [(int(b.layer_id), str(b.buffer.role), b.buffer.size) for b in ring.buffers] == [
        (0, "history", 2048)
    ]
    assert replay.window_size is None
    assert layout.owner(0, HISTORY).name == "ring"
    assert layout.layer_ids("ring") == (LayerId(0),)


def _native_config(layers, regions):
    return KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 24)],
        layers=layers,
        commit_min_snapshot=True,
        regions=regions,
    )


def test_native_config_places_each_region_kind_on_its_layer_kind():
    """A request region sits on SSM layers and a token-block region on attention layers; the native
    config rejects either on the other kind."""
    layers = [AttentionLayerConfig(layer_id=LayerId(0), buffers=[]), *_layers()[1:]]
    configs, _ = lay_out_regions([_ring(), _replay()], layers)
    assert [region.name for region in _native_config(layers, configs).regions] == [
        "ring",
        "replay",
    ]

    request_on_attention, _ = lay_out_regions([_replay(layers=(0,))], _layers())
    with pytest.raises(ValueError, match="SsmLayerConfig"):
        _native_config(_layers(), request_on_attention)
    token_block_on_ssm, _ = lay_out_regions([_ring(layers=(1,))], _layers())
    with pytest.raises(ValueError, match="AttentionLayerConfig"):
        _native_config(_layers(), token_block_on_ssm)


def _recurrent(layers=(1, 2)):
    """The recurrent state of ``layers`` as one region snapshotted at declared points."""
    return Region(
        "recurrent",
        SNAPSHOTTED_SLOT,
        tuple(RegionLayer(layer, (Plane(SSM, 4096), Plane(CONV, 1024))) for layer in layers),
    )


def _region_only_layers():
    """``_layers()`` with the recurrent layers' buffers left to regions."""
    return [
        _layers()[0],
        *(SsmLayerConfig(layer_id=LayerId(layer), buffers=[]) for layer in (1, 2)),
    ]


def test_snapshotted_region_lays_out_as_the_required_recurrent_state():
    """A region handed off at declared points lowers to a REQUIRED native region. It can hold the whole recurrent
    state of layers that keep no buffers of their own, beside a PRIVATE region, and the native config accepts both."""
    regions = [_recurrent(), _replay()]
    configs, layout = lay_out_regions(regions, _region_only_layers())
    assert [(config.name, config.reuse_policy) for config in configs] == [
        ("recurrent", ReusePolicy.REQUIRED),
        ("replay", ReusePolicy.PRIVATE),
    ]
    assert configs[0].fixed_slot_count is None
    assert [(int(b.layer_id), str(b.buffer.role)) for b in configs[0].buffers] == [
        (1, "ssm_state"),
        (1, "conv_state"),
        (2, "ssm_state"),
        (2, "conv_state"),
    ]
    assert layout.owner(2, SSM).name == "recurrent"
    assert layout.region("recurrent").bytes_per_slot() == 2 * (4096 + 1024)
    config = KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 24)],
        layers=_region_only_layers(),
        commit_min_snapshot=True,
        regions=configs,
    )
    assert [region.name for region in config.regions] == ["recurrent", "replay"]


def test_storing_finds_regions_by_their_declared_planes():
    """Before lay-out, when no buffer has an owner yet, the layout still answers which regions store a role."""
    layout = RegionLayout(regions=(_recurrent(), _replay()))
    assert not layout.owners
    assert layout.storing(SSM) == (_recurrent(),)
    assert layout.storing(HISTORY) == (_replay(),)
    assert layout.storing(KEY) == ()


def test_lay_out_rejects_what_it_cannot_store():
    with pytest.raises(ValueError, match="unique"):
        lay_out_regions([_replay(), _replay()], _layers())
    with pytest.raises(ValueError, match="does not have"):
        lay_out_regions([_replay(layers=(1, 7))], _layers())
    with pytest.raises(ValueError, match="already stores"):
        lay_out_regions(
            [Region("clash", PRIVATE_SLOT, (RegionLayer(1, (Plane(CONV, 64),)),))], _layers()
        )
    with pytest.raises(ValueError, match="already stores"):
        lay_out_regions([_replay(), _replay(name="other")], _layers())
    with pytest.raises(ValueError, match="already stores"):
        lay_out_regions([_recurrent()], _layers())
    private_blocks = Region(
        "private_blocks",
        RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.NEVER),
        (RegionLayer(0, (Plane(HISTORY, 64),)),),
    )
    with pytest.raises(ValueError, match="block-boundary"):
        lay_out_regions([private_blocks], _layers())


def test_region_declarations_are_checked():
    with pytest.raises(ValueError, match="hand off"):
        RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.DECLARED_POINTS)
    with pytest.raises(ValueError, match="positive size"):
        Plane(HISTORY, 0)
    with pytest.raises(ValueError, match="repeats a plane role"):
        RegionLayer(1, (Plane(HISTORY, 8), Plane(HISTORY, 8)))
    with pytest.raises(ValueError, match="twice"):
        Region(
            "twice",
            PRIVATE_SLOT,
            (RegionLayer(1, (Plane(HISTORY, 8),)), RegionLayer(1, (Plane(COUNT, 8),))),
        )
    for count in (0, -1):
        with pytest.raises(ValueError, match="fixed_slot_count"):
            _replay(fixed_slot_count=count)
    with pytest.raises(ValueError, match="fixed_slot_count"):
        Region("ring", RING, (RegionLayer(0, (Plane(HISTORY, 8),)),), fixed_slot_count=4)
    with pytest.raises(ValueError, match="has no window"):
        RegionLifeCycle(AllocationUnit.REQUEST, HandOff.NEVER, window=8)
    for window in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.BLOCK_BOUNDARIES, window=window)
    assert PRIVATE_SLOT.reuse_policy == ReusePolicy.PRIVATE
    assert SNAPSHOTTED_SLOT.reuse_policy == ReusePolicy.REQUIRED
    assert RING.reuse_policy == ReusePolicy.REQUIRED


def test_renumbered_follows_the_compacted_layers():
    """A manager that drops layer 0 renumbers the declared regions before laying them out."""
    layout = RegionLayout(regions=(_replay(),)).renumbered([1, 2])
    assert [layer.model_layer for layer in layout.region("replay").layers] == [0, 1]
    assert layout.region("replay").fixed_slot_count == 5
    with pytest.raises(ValueError, match="dropped"):
        RegionLayout(regions=(_replay(),)).renumbered([0, 2])
