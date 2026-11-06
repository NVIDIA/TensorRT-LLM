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

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind, RoleLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.regions import (
    AllocationUnit,
    HandOff,
    Plane,
    Region,
    RegionLayer,
    RegionLayout,
    RegionLifeCycle,
    RegionProvider,
    RegionTransfer,
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
HISTORY = DataRole("history")

SLOT_WITH_SNAPSHOTS = RegionLifeCycle(AllocationUnit.REQUEST, HandOff.DECLARED_POINTS)
PRIVATE_SLOT = RegionLifeCycle(AllocationUnit.REQUEST, HandOff.NEVER)
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
    """Recurrent state on layers 0 and 2 with its private history; a latent on layers 1 and 3."""
    return [
        Region(
            "state",
            SLOT_WITH_SNAPSHOTS,
            tuple(RegionLayer(i, (Plane(SSM, 512), Plane(CONV, 96))) for i in (0, 2)),
        ),
        Region("latent", FULL_HISTORY, tuple(RegionLayer(i, (Plane(KEY, 128),)) for i in (1, 3))),
        Region(
            "history", PRIVATE_SLOT, tuple(RegionLayer(i, (Plane(HISTORY, 64),)) for i in (0, 2))
        ),
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


def test_attention_declarations_do_not_change_the_layout_yet():
    plain = Region("latent", FULL_HISTORY, (RegionLayer(0, (Plane(KEY, 128),)),))
    assert plain.layers[0].bytes_per_token == 0
    assert not plain.resumes_at_block_boundaries
    assert not plain.positioned_by_history
    declared = Region(
        "latent",
        FULL_HISTORY,
        (RegionLayer(0, (Plane(KEY, 128),), bytes_per_token=256),),
        resumes_at_block_boundaries=True,
        positioned_by_history=True,
    )

    plain_layers, plain_layout = lay_out_regions([plain], _base_layers(1))
    declared_layers, declared_layout = lay_out_regions([declared], _base_layers(1))

    assert [_layer_summary(layer) for layer in declared_layers] == [
        _layer_summary(layer) for layer in plain_layers
    ]
    assert declared_layout.owners == plain_layout.owners


def test_no_regions_keeps_the_manager_layers():
    base = _base_layers(3)

    layers, layout = lay_out_regions([], base)

    assert [_layer_summary(layer) for layer in layers] == [_layer_summary(layer) for layer in base]
    assert layout.owners == {}
    assert layout.owner(1, KEY) is None
    assert "state" not in layout


def test_regions_on_one_layer_become_buffers_of_that_layer():
    layers, layout = lay_out_regions(_hybrid_regions(), _base_layers(4))

    # The private history shares the recurrent-state layers: its buffers join
    # them and override the layer's policy, so no layer is added.
    ssm_buffers = [
        ("ssm_state", 512, 0, None),
        ("conv_state", 96, 0, None),
        ("history", 64, 0, ReusePolicy.PRIVATE),
    ]
    assert [_layer_summary(layer) for layer in layers] == [
        ("ssm", 0, ssm_buffers, ReusePolicy.REQUIRED),
        ("attention", 1, [("key", 128, 0, None)], None, None, ReusePolicy.REQUIRED),
        ("ssm", 2, ssm_buffers, ReusePolicy.REQUIRED),
        ("attention", 3, [("key", 128, 0, None)], None, None, ReusePolicy.REQUIRED),
    ]
    assert layout.owner(0, SSM).name == "state"
    assert layout.owner(2, HISTORY).name == "history"
    assert layout.owner(3, KEY).name == "latent"
    assert layout.owner(1, SSM) is None
    assert layout.layer_ids("history") == (LayerId(0), LayerId(2))
    assert layout.roles("state", 2) == (SSM, CONV)
    assert layout.roles("history", 2) == (HISTORY,)
    assert layout.roles("latent", 2) == ()
    assert "history" in layout and "ring" not in layout
    # Ownership lists every buffer layer by layer, in buffer order.
    assert list(layout.owners) == [
        (0, SSM),
        (0, CONV),
        (0, HISTORY),
        (1, KEY),
        (2, SSM),
        (2, CONV),
        (2, HISTORY),
        (3, KEY),
    ]


def test_first_region_on_a_layer_sets_its_policy():
    regions = [
        Region("history", PRIVATE_SLOT, (RegionLayer(0, (Plane(HISTORY, 64),)),)),
        Region("state", SLOT_WITH_SNAPSHOTS, (RegionLayer(0, (Plane(SSM, 512),)),)),
    ]

    layers, _ = lay_out_regions(regions, _base_layers(1))

    assert _layer_summary(layers[0]) == (
        "ssm",
        0,
        [("history", 64, 0, None), ("ssm_state", 512, 0, ReusePolicy.REQUIRED)],
        ReusePolicy.PRIVATE,
    )


def test_private_regions_never_reach_the_prefix_index():
    private_span = RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.NEVER, window=8)
    regions = [
        Region("slot", PRIVATE_SLOT, (RegionLayer(0, (Plane(SSM, 32),)),)),
        Region("span", private_span, (RegionLayer(1, (Plane(KEY, 32),)),)),
    ]

    layers, _ = lay_out_regions(regions, _base_layers(2))

    assert [layer.reuse_policy for layer in layers] == [ReusePolicy.PRIVATE] * 2


def test_layout_rejects_regions_one_layer_config_cannot_hold():
    state = Region("state", SLOT_WITH_SNAPSHOTS, (RegionLayer(0, (Plane(SSM, 32),)),))
    with pytest.raises(ValueError, match="both store"):
        lay_out_regions(
            [state, Region("other", PRIVATE_SLOT, (RegionLayer(0, (Plane(SSM, 16),)),))],
            _base_layers(1),
        )
    with pytest.raises(ValueError, match="mix"):
        lay_out_regions(
            [state, Region("latent", FULL_HISTORY, (RegionLayer(0, (Plane(KEY, 16),)),))],
            _base_layers(1),
        )
    # A window is a property of the V2 layer, so token-block regions sharing a
    # layer must agree on it.
    with pytest.raises(ValueError, match="windows"):
        lay_out_regions(
            [
                Region("latent", FULL_HISTORY, (RegionLayer(0, (Plane(KEY, 16),)),)),
                Region("ring", RING_OF_19, (RegionLayer(0, (Plane(RING, 16),)),)),
            ],
            _base_layers(1),
        )
    private_ring = RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.NEVER, window=19)
    layers, _ = lay_out_regions(
        [
            Region("ring", RING_OF_19, (RegionLayer(0, (Plane(RING, 16),)),)),
            Region("scratch", private_ring, (RegionLayer(0, (Plane(KEY, 16),)),)),
        ],
        _base_layers(1),
    )
    assert _layer_summary(layers[0]) == (
        "attention",
        0,
        [("ring", 16, 0, None), ("key", 16, 0, ReusePolicy.PRIVATE)],
        19,
        None,
        ReusePolicy.REQUIRED,
    )


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
        Region("state", SLOT_WITH_SNAPSHOTS, (RegionLayer(3, (Plane(SSM, 128),)),)),
        Region("history", PRIVATE_SLOT, (RegionLayer(3, (Plane(HISTORY, 76),)),)),
    ]
    _, layout = lay_out_regions(regions, _base_layers(4))
    assert layout.owners == {(3, SSM): "state", (3, HISTORY): "history"}

    # Model layer 1 had no buffers and was dropped; layer 3 moves down with it.
    renumbered = layout.renumbered([0, 2, 3])

    assert renumbered.owners == {(2, SSM): "state", (2, HISTORY): "history"}
    assert renumbered.regions == layout.regions
    with pytest.raises(ValueError, match="dropped"):
        layout.renumbered([0, 1, 2])


class _Cache:
    """Stand-in for the V2 runtime: the history buffers form their own group."""

    def get_buffer_group_id(self, layer_id, role):
        return 2 if role == HISTORY else int(layer_id) % 2


def test_layout_resolves_a_region_to_its_layer_groups():
    _, layout = lay_out_regions(_hybrid_regions(), _base_layers(4))

    assert layout.group_ids("history", _Cache()) == (2,)
    assert layout.group_ids("state", _Cache()) == (0,)
    assert layout.group_ids("latent", _Cache()) == (1,)
    assert layout.group_ids("missing", _Cache()) == ()


@dataclass
class _Provider:
    regions: list
    calls: int = 0

    def declare_regions(self):
        self.calls += 1
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
    assert mgr.region_layout.regions == ()


def test_manager_stores_its_providers_regions():
    provider = _Provider(_hybrid_regions())
    assert isinstance(provider, RegionProvider)
    mgr = _manager_stub([provider])

    config = mgr._lay_out_regions(_config(_base_layers(4)))

    assert [int(layer.layer_id) for layer in config.layers] == [0, 1, 2, 3]
    assert isinstance(config.layers[2], SsmLayerConfig)
    # The runtime rejects per-request storage without minimum-snapshot commits.
    assert config.commit_min_snapshot
    assert isinstance(mgr.region_layout, RegionLayout)
    assert mgr.region_layout.owner(2, HISTORY).name == "history"


def test_dropping_an_empty_layer_keeps_region_ownership_consistent():
    regions = [
        Region("latent", FULL_HISTORY, (RegionLayer(2, (Plane(KEY, 128),)),)),
        Region("ring", RING_OF_19, (RegionLayer(0, (Plane(RING, 76),)),)),
    ]
    mgr = _manager_stub([_Provider(regions)])
    base = _base_layers(3)
    # Model layer 1 has no cache storage on this rank.
    base[1] = AttentionLayerConfig(layer_id=LayerId(1), buffers=[BufferConfig(role=KEY, size=0)])
    mgr.num_local_layers = 3
    mgr.pp_layers = [10, 11, 12]
    mgr.num_kv_heads_per_layer = [1, 0, 1]
    mgr.head_dim_per_layer = [64, 64, 64]
    mgr.max_attention_window_vec = [256, 256, 256]

    config = mgr._remove_zero_size_buffers(mgr._lay_out_regions(_config(base)))

    assert [int(layer.layer_id) for layer in config.layers] == [0, 1]
    assert [str(layer.buffers[0].role) for layer in config.layers] == ["ring", "key"]
    assert mgr.pp_layers == [10, 12]
    assert mgr.num_local_layers == 2
    assert mgr.region_layout.owners == {(0, RING): "ring", (1, KEY): "latent"}


def test_regions_are_declared_before_the_layout():
    provider = _Provider(_hybrid_regions())
    mgr = _manager_stub([provider])

    # The manager costs its memory budget between declaring and laying out the
    # regions, so the declarations are readable before any buffer exists.
    mgr._declare_regions()
    assert [region.name for region in mgr.region_layout.regions] == ["state", "latent", "history"]
    assert mgr.region_layout.owners == {}

    mgr._lay_out_regions(_config(_base_layers(4)))

    assert provider.calls == 1
    assert mgr.region_layout.owner(0, CONV).name == "state"


def test_manager_resolves_a_region_to_its_layer_groups():
    mgr = _manager_stub([_Provider(_hybrid_regions())])
    mgr._lay_out_regions(_config(_base_layers(4)))
    mgr.impl = _Cache()

    assert mgr.get_region_group_ids("history") == (2,)
    assert mgr.get_region_group_ids("missing") == ()


def test_transfer_declarations_reach_the_disagg_hooks():
    ring_layout = RoleLayout(section_bytes=(16, 16))
    regions = [
        Region(
            "state",
            SLOT_WITH_SNAPSHOTS,
            (RegionLayer(0, (Plane(SSM, 32), Plane(HISTORY, 8))),),
            transfer=RegionTransfer(rebuilt_roles=frozenset({HISTORY})),
        ),
        Region(
            "ring",
            RING_OF_19,
            (RegionLayer(1, (Plane(RING, 32),)),),
            transfer=RegionTransfer(
                mapper_kinds={RING: MapperKind.REPLICATED}, role_layouts={RING: ring_layout}
            ),
        ),
    ]
    with_regions = _manager_stub([_Provider(regions)])
    with_regions._declare_regions()
    without_regions = _manager_stub([])

    assert with_regions.get_disagg_ignored_roles() == {HISTORY}
    assert with_regions.get_disagg_role_layouts() == {RING: ring_layout}
    assert with_regions.get_disagg_role_mapper_kinds() == {
        Role.ALL: MapperKind.INDEXED,
        Role.INDEX_KEY: MapperKind.REPLICATED,
        RING: MapperKind.REPLICATED,
    }
    assert without_regions.get_disagg_ignored_roles() == frozenset()
    assert without_regions.get_disagg_role_layouts() == {}
    assert without_regions.get_disagg_role_mapper_kinds() == {
        Role.ALL: MapperKind.INDEXED,
        Role.INDEX_KEY: MapperKind.REPLICATED,
    }
