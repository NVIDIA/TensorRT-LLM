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


def test_lowered_regions_validate_in_the_native_config():
    """The native config accepts the lowered region beside the layers, and rejects it on an attention layer."""
    configs, _ = lay_out_regions([_replay()], _layers())
    config = KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 24)],
        layers=_layers(),
        commit_min_snapshot=True,
        regions=configs,
    )
    assert [region.name for region in config.regions] == ["replay"]

    on_attention, _ = lay_out_regions([_replay(layers=(0,))], _layers())
    with pytest.raises(ValueError, match="SsmLayerConfig"):
        KVCacheManagerConfig(
            tokens_per_block=32,
            cache_tiers=[GpuCacheTierConfig(quota=1 << 24)],
            layers=_layers(),
            commit_min_snapshot=True,
            regions=on_attention,
        )


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
    snapshotted = Region(
        "snapshotted",
        RegionLifeCycle(AllocationUnit.REQUEST, HandOff.DECLARED_POINTS),
        (RegionLayer(1, (Plane(HISTORY, 64),)),),
    )
    with pytest.raises(ValueError, match="never handed off"):
        lay_out_regions([snapshotted], _layers())


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
    assert PRIVATE_SLOT.reuse_policy == ReusePolicy.PRIVATE


def test_renumbered_follows_the_compacted_layers():
    """A manager that drops layer 0 renumbers the declared regions before laying them out."""
    layout = RegionLayout(regions=(_replay(),)).renumbered([1, 2])
    assert [layer.model_layer for layer in layout.region("replay").layers] == [0, 1]
    assert layout.region("replay").fixed_slot_count == 5
    with pytest.raises(ValueError, match="dropped"):
        RegionLayout(regions=(_replay(),)).renumbered([0, 2])
