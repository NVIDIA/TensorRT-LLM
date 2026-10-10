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
"""``_layout``: ``layout_id`` over hand-built layouts (CPU), ``derive_layout`` over a stand-in
manager scripting TP ranks, pipeline stages and recurrent groups (CPU), and over real
``KVCacheManagerV2`` instances (GPU): what each description comes from and what moves the id. Part
names are checked in ``test_names.py``."""

import dataclasses
import enum
import hashlib
import json
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind, RoleLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import (
    LAYOUT_ID_VERSION,
    BufferDesc,
    BufferGeometry,
    BufferMapper,
    LayerGroupDesc,
    Layout,
    ShardDesc,
    _Heads,
    canonical_bytes,
    canonical_order,
    derive_layout,
    global_layer_ids,
    layout_document,
    layout_id,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

# -- hand-built layouts --------------------------------------------------------------------------

# Device pool group 0 has two pools (a 100-byte and a 40-byte slot); pool group 1 is a single-pool
# group for a second layer group.


def make_layout(
    *,
    tokens_per_block=32,
    windows=(None, 64),
    layers=((0, 2), (1,)),
    shard=ShardDesc(),
    group_ids=(0, 1),
    slot_bytes=(100, 40),
) -> Layout:
    g0, g1 = group_ids
    pool_groups = {g0: tuple(slot_bytes), g1: (64,)}
    layer_groups = (
        LayerGroupDesc(
            kind="attention",
            window=windows[0],
            sink_blocks=0,
            pool_group=g0,
            layers=layers[0],
            shard=shard,
        ),
        LayerGroupDesc(
            kind="attention",
            window=windows[1],
            sink_blocks=0,
            pool_group=g1,
            layers=layers[1],
            shard=shard,
        ),
    )
    buffers = []
    for layer in layers[0]:
        buffers += [
            BufferDesc(layer, "key", (g0, 0), 50 * (layer // 2), 50),
            BufferDesc(layer, "scale", (g0, 1), 20 * (layer // 2), 20),
        ]
    for layer in layers[1]:
        buffers.append(BufferDesc(layer, "key", (g1, 0), 0, 64))
    return Layout(tokens_per_block, pool_groups, layer_groups, tuple(buffers))


def _replace_buffers(layout, **change):
    return dataclasses.replace(
        layout,
        buffers=tuple(
            dataclasses.replace(b, **change) if b.role == "scale" else b for b in layout.buffers
        ),
    )


def _first_group(layout, **change):
    first = dataclasses.replace(layout.layer_groups[0], **change)
    return dataclasses.replace(layout, layer_groups=(first,) + layout.layer_groups[1:])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(lambda: make_layout(group_ids=(7, 3)), id="pool_group_numbering"),
        pytest.param(
            lambda: dataclasses.replace(
                make_layout(), layer_groups=make_layout().layer_groups[::-1]
            ),
            id="layer_group_order",
        ),
        pytest.param(lambda: _first_group(make_layout(), sink_blocks=3), id="sinks"),
        pytest.param(lambda: make_layout(windows=(None, 128)), id="window"),
        pytest.param(lambda: make_layout(windows=(64, 64)), id="full_to_window"),
        pytest.param(lambda: make_layout(shard=ShardDesc(2, 1)), id="shard"),
        pytest.param(
            lambda: dataclasses.replace(make_layout(), buffers=make_layout().buffers[::-1]),
            id="buffer_order",
        ),
    ],
)
def test_layout_id_leaves_out_what_does_not_decide_how_bytes_read(variant):
    """Local numbering and order, sinks and windows (which blocks exist, not how their bytes read),
    and the shard position, which rides in the name so every rank computes the same ``layout_id``.
    Buffers are described by global layer, so their listing order does not matter either."""
    assert layout_id(variant()) == layout_id(make_layout())


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "change",
    [
        lambda: make_layout(tokens_per_block=64),
        lambda: make_layout(layers=((0, 1), (2,))),
        lambda: make_layout(slot_bytes=(104, 40)),
        lambda: _replace_buffers(make_layout(), mapper=BufferMapper.REPLICATED),
        lambda: _replace_buffers(make_layout(), geometry=BufferGeometry(bytes_per_head=10)),
        lambda: _replace_buffers(make_layout(), geometry=BufferGeometry(section_bytes=(15, 5))),
        lambda: _replace_buffers(make_layout(), transfer=False),
        lambda: _replace_buffers(make_layout(), expansion=2),
        lambda: _replace_buffers(make_layout(), role="scale2"),
        lambda: _first_group(make_layout(), kind="state"),
        lambda: dataclasses.replace(
            make_layout(),
            buffers=tuple(
                dataclasses.replace(b, offset=b.offset + 10) if b.role == "scale" else b
                for b in make_layout().buffers
            ),
        ),
    ],
    ids=[
        "tokens_per_block",
        "layer_membership",
        "slot_width",
        "mapper",
        "head_geometry",
        "sections",
        "transfer",
        "expansion",
        "role",
        "kind",
        "position_in_the_transfer_bytes",
    ],
)
def test_layout_id_changes_with_every_input_that_decides_how_bytes_read(change):
    assert layout_id(change()) != layout_id(make_layout())


@pytest.mark.cpu_only
def test_canonical_order_sorts_groups_by_smallest_layer():
    layout = make_layout(layers=((3, 5), (1, 4)))
    assert canonical_order([d.layers for d in layout.layer_groups]) == [1, 0]
    assert canonical_order([d.layers for d in layout.layer_groups[::-1]]) == [0, 1]
    # Ties and empty groups keep the local order; empty groups go last.
    assert canonical_order([(), (2,), (2,), (0, 9)]) == [3, 1, 2, 0]
    assert canonical_order([(9,), (2, 11), (5,)]) == [2, 0, 1]


@pytest.mark.cpu_only
def test_layout_id_is_the_hash_of_the_documented_version_2_encoding():
    """The canonical encoding, spelled out here byte for byte so a change to it cannot pass
    unnoticed."""
    layout = make_layout(layers=((0,), (1,)))
    expected = (
        '{"buffers":['
        '{"layer":0,"offset":0,"role":"key","size":50},'
        '{"layer":0,"offset":100,"role":"scale","size":20},'
        '{"layer":1,"offset":0,"role":"key","size":64}],'
        '"groups":[{"layers":[0],"slot_bytes":[100,40]},{"layers":[1],"slot_bytes":[64]}],'
        '"tokens_per_block":32,"v":2}'
    ).encode()
    assert LAYOUT_ID_VERSION == 2
    assert canonical_bytes(layout_document(layout)) == expected
    assert layout_id(layout) == hashlib.sha256(expected).digest()


@pytest.mark.cpu_only
def test_named_keys_at_their_defaults_are_left_out_and_others_are_written():
    layout = make_layout(layers=((0,), (1,)))
    plain = layout_document(layout)["buffers"][1]
    assert set(plain) == {"layer", "role", "offset", "size"}
    changed = layout_document(
        _replace_buffers(
            layout,
            mapper=BufferMapper.SECTIONED,
            geometry=BufferGeometry(section_bytes=(15, 5)),
            transfer=False,
            expansion=4,
        )
    )["buffers"][1]
    assert changed["mapper"] == "SECTIONED"
    assert changed["geometry"] == {"section_bytes": [15, 5]}
    assert changed["transfer"] is False
    assert changed["expansion"] == 4
    assert layout_document(_first_group(layout, kind="state"))["groups"][0]["kind"] == "state"
    assert "kind" not in layout_document(layout)["groups"][0]


@pytest.mark.cpu_only
def test_enums_encode_by_name_and_the_hnd_alias_is_indexed():
    hnd = _replace_buffers(make_layout(), mapper=MapperKind.HND)
    indexed = _replace_buffers(make_layout(), mapper=BufferMapper.INDEXED)
    assert layout_id(hnd) == layout_id(indexed) == layout_id(make_layout())

    class Color(enum.IntEnum):
        RED = 3

    assert json.loads(canonical_bytes({"c": Color.RED, "n": None, "l": [None, 1]})) == {
        "c": "RED",
        "l": [None, 1],
    }


@pytest.mark.cpu_only
def test_floats_encode_as_their_binary64_bits():
    assert canonical_bytes({"x": 1.0}) == b'{"x":"f64:3ff0000000000000"}'
    assert canonical_bytes({"x": -0.0}) != canonical_bytes({"x": 0.0})


@pytest.mark.cpu_only
def test_a_document_refuses_what_the_layout_does_not_hold():
    layout = make_layout()
    with pytest.raises(ValueError, match="maps to pool group 9"):
        layout_document(_first_group(layout, pool_group=9))
    outside = layout.buffers + (BufferDesc(7, "key", (9, 0), 0, 8),)
    with pytest.raises(ValueError, match="not in the layout"):
        layout_document(dataclasses.replace(layout, buffers=outside))
    twice = layout.buffers + (layout.buffers[0],)
    with pytest.raises(ValueError, match="appears twice"):
        layout_document(dataclasses.replace(layout, buffers=twice))


# -- geometry and buffers ------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_a_geometry_that_does_not_fit_its_buffer_is_refused():
    with pytest.raises(ValueError, match="sum"):
        BufferDesc(0, "conv", (0, 0), 0, 30, geometry=BufferGeometry(section_bytes=(10, 10)))
    with pytest.raises(ValueError, match="heads"):
        BufferDesc(0, "k", (0, 0), 0, 30, geometry=BufferGeometry(bytes_per_head=8))
    with pytest.raises(ValueError, match="heads"):
        BufferDesc(0, "k", (0, 0), 0, 32, geometry=BufferGeometry(bytes_per_head=8, num_heads=2))


@pytest.mark.cpu_only
def test_an_expansion_must_divide_the_block():
    with pytest.raises(ValueError, match="positive"):
        BufferDesc(0, "k", (0, 0), 0, 32, expansion=0)
    with pytest.raises(ValueError, match="does not divide"):
        _replace_buffers(make_layout(), expansion=3)


@pytest.mark.cpu_only
@pytest.mark.parametrize("shard", [(2, 2), (0, 0), (2, -1)])
def test_a_shard_is_a_share_of_its_count(shard):
    with pytest.raises(ValueError, match="shard"):
        ShardDesc(*shard)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sections",
    [(np.int64(4), np.int64(4)), np.array([4, 4]), [4, 4], (4, 4)],
    ids=["numpy-ints", "numpy-array", "list", "tuple"],
)
def test_a_validated_geometry_holds_its_sections_as_a_tuple_of_ints(sections):
    checked = BufferGeometry(section_bytes=sections, bytes_per_head=2).validate()
    assert checked == BufferGeometry(section_bytes=(4, 4), bytes_per_head=2)
    assert type(checked.section_bytes) is tuple
    assert all(type(s) is int for s in checked.section_bytes)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "fields", [{"section_bytes": (4, 0)}, {"bytes_per_head": 0}, {"num_heads": -1}]
)
def test_validation_refuses_sizes_that_are_not_positive(fields):
    with pytest.raises(ValueError, match="positive"):
        BufferGeometry(**fields).validate()


# -- a stand-in manager --------------------------------------------------------------------------

TPB = 4
HEAD_BYTES = TPB * 8 * 2  # 8 fp16 values per token and head


def coalesced(size, members):
    return SimpleNamespace(
        single_buffer_size=size,
        buffer_ids=[SimpleNamespace(layer_id=layer, role=role) for layer, role in members],
    )


class Manager:
    """One layer group per entry of ``groups``: ``(layer config class, [internal layer ids],
    [(role, size, override)])``; group ``i`` draws from pool group ``pool_group_of[i]`` (its own by
    default), one pool per role, and has window ``windows[i]`` and ``sinks[i]`` sink tokens."""

    def __init__(
        self,
        groups,
        *,
        tp_size=1,
        tp_rank=0,
        attention_dp=False,
        num_kv_heads=4,
        pp_layers: Optional[Sequence[int]] = None,
        mappers: Optional[Dict[str, MapperKind]] = None,
        layouts: Optional[Dict[str, RoleLayout]] = None,
        ignored=(),
        pool_group_of: Optional[Sequence[int]] = None,
        windows: Optional[Sequence[Optional[int]]] = None,
        sinks: Optional[Sequence[Optional[int]]] = None,
    ):
        heads_per_rank = -(-num_kv_heads // (1 if attention_dp else tp_size))
        internal = sorted({lid for _, lids, _ in groups for lid in lids})
        pool_group_of = list(pool_group_of if pool_group_of is not None else range(len(groups)))
        windows = list(windows or [None] * len(groups))
        sinks = list(sinks or [None] * len(groups))
        self.tokens_per_block = TPB
        self.mapping = SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, enable_attention_dp=attention_dp
        )
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_per_layer = [heads_per_rank] * len(internal)
        self.pp_layers = list(pp_layers if pp_layers is not None else internal)
        self.mappers = {"all": MapperKind.INDEXED, **(mappers or {})}
        self.layouts = layouts or {}
        self.ignored = frozenset(ignored)
        layers: List[object] = []
        descs = {}
        for lg, (cls, lids, roles) in enumerate(groups):
            for lid in lids:
                buffers = [
                    BufferConfig(role=role, size=size, tokens_per_block_override=override)
                    for role, size, override in roles
                ]
                if cls is SsmLayerConfig:
                    layers.append(SsmLayerConfig(layer_id=lid, buffers=buffers))
                    continue
                layers.append(
                    cls(
                        layer_id=lid,
                        buffers=buffers,
                        sliding_window_size=windows[lg],
                        num_sink_tokens=sinks[lg],
                    )
                )
            pools = [coalesced(size, [(lid, role) for lid in lids]) for role, size, _ in roles]
            variant = SimpleNamespace(layer_group_id=lg, coalesced_buffers=pools)
            g = pool_group_of[lg]
            if g in descs:
                descs[g].slot_desc.variants.append(variant)
                continue
            descs[g] = SimpleNamespace(
                pool_group_index=g,
                num_slots=8,
                pools=[
                    SimpleNamespace(
                        pool_index=i,
                        base_address=0x1000 * (g + 1) + 0x100 * i,
                        slot_bytes=c.single_buffer_size * len(lids),
                    )
                    for i, c in enumerate(pools)
                ],
                slot_desc=SimpleNamespace(variants=[variant]),
            )
        self.impl = SimpleNamespace(
            layer_grouping=[list(lids) for _, lids, _ in groups],
            pool_group_descs=list(descs.values()),
            get_life_cycle_pool_group_indices=lambda: list(pool_group_of),
            init_config=SimpleNamespace(layers=layers),
        )

    def get_disagg_role_mapper_kinds(self):
        return self.mappers

    def get_disagg_role_layouts(self):
        return self.layouts

    def get_disagg_ignored_roles(self):
        return self.ignored


def kv_group(lids=(0, 1), heads_per_rank=1, extra=()):
    size = HEAD_BYTES * heads_per_rank
    return (AttentionLayerConfig, list(lids), [("key", size, None), ("value", size, None), *extra])


def attention(tp_size, tp_rank, num_kv_heads, **kwargs):
    heads = -(-num_kv_heads // tp_size)
    return Manager(
        [kv_group(heads_per_rank=heads)],
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads=num_kv_heads,
        **kwargs,
    )


def recurrent_state():
    return Manager(
        [(SsmLayerConfig, [0, 1], [("conv_state", 96, None), ("ssm_state", 128, None)])],
        tp_size=2,
        tp_rank=1,
        mappers={"conv_state": MapperKind.SECTIONED},
        layouts={
            "conv_state": RoleLayout(section_bytes=(32, 32, 32)),
            "ssm_state": RoleLayout(bytes_per_head=32),
        },
    )


TOKEN_MAJOR_SHARES = "head-major and token-major K/V share a layout_id"


def check_token_major_kv_has_its_own_layout_id():
    """A manager declaring token-major K/V (NHD) and one declaring the default head-major K/V (HND)
    lay a block's bytes out differently, so they never share a ``layout_id``."""
    head_major = derive_layout(attention(1, 0, 4)).layout
    token_major = derive_layout(attention(1, 0, 4, mappers={"all": MapperKind.NHD})).layout
    assert layout_id(head_major) != layout_id(token_major), TOKEN_MAJOR_SHARES
    assert {b.mapper for b in token_major.buffers} == {BufferMapper.NHD}
    assert {b.mapper for b in head_major.buffers} == {BufferMapper.INDEXED}
    hnd = derive_layout(attention(1, 0, 4, mappers={"all": MapperKind.HND})).layout
    assert layout_id(hnd) == layout_id(head_major), "HND is the default head-major layout"


@pytest.mark.cpu_only
def test_token_major_kv_has_its_own_layout_id():
    check_token_major_kv_has_its_own_layout_id()


# (TP size, KV heads, attention DP) -> the share each rank holds, rank by rank.
KV_HEAD_SHARES = {
    # DeepSeek-V4 at TP8 without attention DP: one KV head, repeated on every rank, so every rank
    # describes the group as whole and all of them compute one namespace.
    "single_kv_head": ((8, 1, False), [ShardDesc()] * 8),
    "head_sharded": ((4, 8, False), [ShardDesc(4, r) for r in range(4)]),
    # Two KV heads over four ranks: ranks 0 and 1 hold head 0, ranks 2 and 3 head 1, so there are
    # two shares, not four.
    "repeated_heads": ((4, 2, False), [ShardDesc(2, r // 2) for r in range(4)]),
    "attention_dp_holds_whole_heads": ((4, 8, True), [ShardDesc()] * 4),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", list(KV_HEAD_SHARES))
def test_kv_heads_name_the_share_each_rank_holds(case):
    """The byte layout is the same on every rank; the share is part of the object name."""
    (tp_size, num_kv_heads, attention_dp), shares = KV_HEAD_SHARES[case]
    geometries = [
        derive_layout(attention(tp_size, r, num_kv_heads, attention_dp=attention_dp))
        for r in range(tp_size)
    ]
    assert [geo.layout.layer_groups[0].shard for geo in geometries] == shares
    assert [geo.shards for geo in geometries] == [((s.count, s.index),) for s in shares]
    assert len({layout_id(geo.layout) for geo in geometries}) == 1


def heads_on(rank, tp_size, total, per_rank):
    """``_Heads`` of a rank that holds ``per_rank`` of ``total`` KV heads over ``tp_size`` ranks."""
    mapping = SimpleNamespace(tp_size=tp_size, tp_rank=rank, enable_attention_dp=False)
    manager = SimpleNamespace(
        mapping=mapping, num_kv_heads_per_layer=[per_rank], num_kv_heads=total, pp_layers=[0]
    )
    return _Heads(manager)


# (KV heads T, TP size, heads per rank) -> each rank's share. Rank r holds its heads from
# r * T // tp; where any rank's range starts inside a share, every rank is a share of its own.
UNALIGNED_HEADS = {
    (6, 4, 2): [ShardDesc(4, r) for r in range(4)],
    (12, 8, 2): [ShardDesc(8, r) for r in range(8)],
    (8, 4, 2): [ShardDesc(4, r) for r in range(4)],
    (2, 4, 1): [ShardDesc(2, 0), ShardDesc(2, 0), ShardDesc(2, 1), ShardDesc(2, 1)],
    (1, 4, 1): [ShardDesc(1, 0)] * 4,
}


@pytest.mark.cpu_only
def test_unaligned_head_ranges_give_every_rank_its_own_share():
    for (total, tp_size, per_rank), shares in UNALIGNED_HEADS.items():
        got = [heads_on(r, tp_size, total, per_rank).attention_shard(0) for r in range(tp_size)]
        assert got == shares, f"{total} heads over {tp_size} ranks: {got}"


@pytest.mark.cpu_only
def test_a_group_of_replicated_buffers_is_whole_and_one_head_sharded_buffer_makes_it_a_share():
    index = ("index_key", 64, None)
    replicated_only = Manager(
        [(AttentionLayerConfig, [0], [index])],
        tp_size=2,
        tp_rank=1,
        num_kv_heads=8,
        mappers={"index_key": MapperKind.REPLICATED},
    )
    assert derive_layout(replicated_only).layout.layer_groups[0].shard == ShardDesc()
    mixed = Manager(
        [kv_group(heads_per_rank=4, extra=[index])],
        tp_size=2,
        tp_rank=1,
        num_kv_heads=8,
        mappers={"index_key": MapperKind.REPLICATED},
    )
    assert derive_layout(mixed).layout.layer_groups[0].shard == ShardDesc(2, 1)


@pytest.mark.cpu_only
def test_local_only_buffers_do_not_decide_the_identity():
    scratch = ("scratch", 32, None)
    manager = Manager(
        [(AttentionLayerConfig, [0], [("index_key", 64, None), scratch])],
        tp_size=2,
        tp_rank=1,
        mappers={"index_key": MapperKind.REPLICATED},
        ignored=("scratch",),
    )
    layout = derive_layout(manager).layout
    assert layout.layer_groups[0].shard == ShardDesc()
    assert {b.role: b.transfer for b in layout.buffers} == {"index_key": True, "scratch": False}


@pytest.mark.cpu_only
def test_groups_of_one_manager_are_judged_one_by_one():
    manager = Manager(
        [
            kv_group(lids=(0,), heads_per_rank=4),
            (AttentionLayerConfig, [1], [("index_key", 64, None)]),
        ],
        tp_size=2,
        tp_rank=0,
        num_kv_heads=8,
        mappers={"index_key": MapperKind.REPLICATED},
    )
    groups = derive_layout(manager).layout.layer_groups
    assert [g.shard for g in groups] == [ShardDesc(2, 0), ShardDesc()]


def mixed_shares(tp_size, tp_rank, heads):
    """``heads`` KV heads in layers 0 and 1 over ``tp_size`` ranks, one head per rank in each, one
    group."""
    manager = Manager([kv_group(lids=(0, 1))], tp_size=tp_size, tp_rank=tp_rank, num_kv_heads=8)
    manager.num_kv_heads = list(heads)
    manager.num_kv_heads_per_layer = [1, 1]
    return manager


# name: (ranks, heads per layer). The group's first buffer is layer 0's. At TP8 the narrowest
# share equals the fallback, and the first buffer's too with eight heads first; at TP6 none does.
MIXED_SHARES = {
    "eight_first": (8, (8, 4)),
    "four_first": (8, (4, 8)),
    "three_first": (6, (3, 2)),
    "two_first": (6, (2, 3)),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", list(MIXED_SHARES))
def test_a_group_whose_buffers_hold_different_shares_is_each_rank_alone(case):
    """The two layers' buffers hold shares of different counts, so the group falls back to each
    rank's own share of the ranks, and no two ranks name a block alike."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._identity import Identity

    tp_size, heads = MIXED_SHARES[case]
    keys = np.arange(3 * 32, dtype=np.uint8).reshape(3, 32)
    names = set()
    for rank in range(tp_size):
        geometry = derive_layout(mixed_shares(tp_size, rank, heads))
        assert geometry.layout.layer_groups[0].shard == ShardDesc(tp_size, rank), (
            f"rank {rank}: the group's share is not the rank's own"
        )
        identity = Identity(b"scope", layout_id(geometry.layout), geometry.layers, geometry.shards)
        names |= {row.tobytes() for row in identity.names(0, keys)}
    assert len(names) == tp_size * len(keys), "two ranks name a block alike"


@pytest.mark.cpu_only
def test_recurrent_state_is_described_and_sharded_by_rank():
    geometry = derive_layout(recurrent_state())
    layout = geometry.layout
    (group,) = layout.layer_groups
    assert group.kind == "state" and group.shard == ShardDesc(2, 1)
    assert geometry.recurrent == (True,) and geometry.windows == (None,)
    by_role = {b.role: b for b in layout.buffers}
    assert by_role["conv_state"].mapper is BufferMapper.SECTIONED
    assert by_role["conv_state"].geometry == BufferGeometry(section_bytes=(32, 32, 32))
    assert by_role["ssm_state"].geometry == BufferGeometry(bytes_per_head=32, num_heads=4)


@pytest.mark.cpu_only
def test_head_geometry_is_described_per_buffer():
    layout = derive_layout(attention(2, 0, num_kv_heads=8)).layout
    for b in layout.buffers:
        assert b.mapper is BufferMapper.INDEXED
        assert b.geometry == BufferGeometry(bytes_per_head=HEAD_BYTES, num_heads=4)


@pytest.mark.cpu_only
def test_layers_are_global_ids_stable_across_pipeline_stages():
    """Stage 2 of a pipeline holds model layers 10 and 11 as internal layers 0 and 1."""
    manager = Manager([kv_group()], pp_layers=[10, 11])
    geometry = derive_layout(manager)
    assert geometry.layout.layer_groups[0].layers == (10, 11)
    assert geometry.layers == ((10, 11),)
    assert {b.layer for b in geometry.layout.buffers} == {10, 11}
    assert global_layer_ids(manager, [1, 0]) == [11, 10]
    # A manager without the map is refused, not named by its stage-local ids, which another
    # stage reuses for other layers.
    del manager.pp_layers
    with pytest.raises(AttributeError, match="pp_layers"):
        derive_layout(manager)


@pytest.mark.cpu_only
def test_virtual_layers_are_numbered_by_model_layer_and_attention_type():
    """Two attention types over three enum members: model layer ``m`` of type ``t`` is
    ``3 * m + t``, whichever types this stage holds."""

    class AttentionType(enum.Enum):
        SLIDING = 0
        COMPRESSED = 1
        INDEXER = 2

    manager = Manager([kv_group(lids=(0, 1, 2))])
    manager._layer_attn_to_layer_id = {
        (4, AttentionType.SLIDING): 0,
        (4, AttentionType.COMPRESSED): 1,
        (5, AttentionType.SLIDING): 2,
    }
    assert global_layer_ids(manager, [0, 1, 2]) == [12, 13, 15]
    assert derive_layout(manager).layers == ((12, 13, 15),)


@pytest.mark.cpu_only
def test_declared_global_layer_ids_name_extra_internal_layers():
    """A manager whose internal layers outnumber its model layers (FP4 MLA keeps its tail on
    extra ones) declares the ids per layer group; they are used as declared, in group order."""
    manager = Manager(
        [kv_group(lids=(1, 0)), (AttentionLayerConfig, [2, 3], [("tail", 64, None)])],
        pp_layers=[0, 1],
        mappers={"tail": MapperKind.REPLICATED},
    )
    declared = {0: [2, 0], 1: [1, 3]}
    manager.get_disagg_global_layer_ids = lambda lg: declared[lg]
    assert global_layer_ids(manager, [0, 1, 2, 3]) == [0, 2, 1, 3]
    layout = derive_layout(manager).layout
    assert [g.layers for g in layout.layer_groups] == [(0, 2), (1, 3)]
    assert {(b.layer, b.role) for b in layout.buffers if b.role == "tail"} == {
        (1, "tail"),
        (3, "tail"),
    }
    manager.get_disagg_global_layer_ids = lambda lg: declared[lg][:1]
    with pytest.raises(ValueError, match="declared global ids"):
        global_layer_ids(manager, [0])


@pytest.mark.cpu_only
def test_a_buffer_with_its_own_tokens_per_block_is_expanded():
    manager = Manager(
        [kv_group(extra=[("index_key", 64, 2)])], mappers={"index_key": MapperKind.REPLICATED}
    )
    layout = derive_layout(manager).layout
    assert {b.role: b.expansion for b in layout.buffers} == {"key": 1, "value": 1, "index_key": 2}
    plain = derive_layout(
        Manager(
            [kv_group(extra=[("index_key", 64, None)])],
            mappers={"index_key": MapperKind.REPLICATED},
        )
    ).layout
    assert layout_id(layout) != layout_id(plain)


@pytest.mark.cpu_only
def test_the_manager_layout_holds_per_group_and_per_pool_group_facts():
    """Two attention groups share pool group 3, each with its own window and sinks; a state group
    has pool group 1 and neither window nor sinks."""
    manager = Manager(
        [
            kv_group(lids=(0,)),
            kv_group(lids=(1,)),
            (SsmLayerConfig, [2], [("conv_state", 96, None)]),
        ],
        pool_group_of=[3, 3, 1],
        windows=[None, 8, 16],
        sinks=[None, 5, 4],
    )
    geometry = derive_layout(manager)
    assert geometry.num_layer_groups == 3
    assert geometry.tokens_per_block == TPB
    assert geometry.pool_group_of == (3, 3, 1)
    assert [d.pool_group for d in geometry.layout.layer_groups] == [3, 3, 1]
    assert geometry.pool_groups == (1, 3)
    assert geometry.windows == (None, 8, None)
    assert geometry.sink_blocks == (0, 2, 0)  # 5 sink tokens take two 4-token blocks
    assert geometry.recurrent == (False, False, True)
    assert geometry.layers == ((0,), (1,), (2,))
    assert geometry.shards == ((1, 0),) * 3
    assert dict(geometry.page_bytes) == {3: 2 * HEAD_BYTES, 1: 96}
    assert dict(geometry.layout.pool_groups) == {3: (HEAD_BYTES, HEAD_BYTES), 1: (96,)}
    pools = {
        g: [(p.group, p.index, p.base, p.slot_bytes, p.num_slots) for p in ps]
        for g, ps in geometry.device_pools.items()
    }
    assert pools == {
        3: [(3, 0, 0x4000, HEAD_BYTES, 8), (3, 1, 0x4100, HEAD_BYTES, 8)],
        1: [(1, 0, 0x2000, 96, 8)],
    }


@pytest.mark.cpu_only
def test_a_state_group_is_told_by_its_config_class_not_its_name():
    """Only the runtime's state layer config makes a group a state group, whatever another layer
    config class is named."""

    class SsmNamedAttentionConfig(SimpleNamespace):
        """An attention layer config whose class name reads like the state one."""

    _, lids, roles = kv_group(lids=(0,))
    manager = Manager(
        [(SsmNamedAttentionConfig, lids, roles), (SsmLayerConfig, [1], [("conv_state", 96, None)])]
    )
    assert derive_layout(manager).recurrent == (False, True)


@pytest.mark.cpu_only
def test_a_manager_whose_parts_disagree_is_refused():
    no_config = Manager([kv_group()])
    no_config.impl.init_config.layers = []
    with pytest.raises(ValueError, match="has no layer config"):
        derive_layout(no_config)
    miscounted = Manager([kv_group()])
    miscounted.impl.get_life_cycle_pool_group_indices = lambda: [0, 0]
    with pytest.raises(ValueError, match="pool-group indices"):
        derive_layout(miscounted)
    poolless = Manager([kv_group()])
    poolless.impl.get_life_cycle_pool_group_indices = lambda: [5]
    with pytest.raises(ValueError, match="which has no pools"):
        derive_layout(poolless)
    with pytest.raises(ValueError, match="does not divide"):
        derive_layout(Manager([kv_group(extra=[("index_key", 64, 3)])]))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "build,expected",
    [
        (
            lambda: Manager(
                [kv_group(heads_per_rank=4, extra=[("index_key", 64, 2)])],
                tp_size=2,
                tp_rank=1,
                num_kv_heads=8,
                mappers={"index_key": MapperKind.REPLICATED},
            ),
            "069d747522257b726c6b28d917cb2c260350e26c55550bb687bea7350a76f51c",
        ),
        (
            recurrent_state,
            "3eba9af0efca5fe1e8701398e86b43ada86f07ba249f483be435925ffeaab6fa",
        ),
    ],
    ids=["head_sharded_with_expanded_replicated_role", "recurrent_state"],
)
def test_the_derived_layout_id_is_pinned(build, expected):
    """The derivation and the encoding together: a change here moves every name built on it."""
    assert layout_id(derive_layout(build()).layout).hex() == expected


# -- real managers -------------------------------------------------------------------------------

REAL_TPB = 32
WINDOW = 64


def native_pool_groups(mgr):
    """``{g: [(base, slot bytes)...]}`` and slot counts, read straight off the manager."""
    pools, counts = {}, {}
    for pg in mgr.impl.pool_group_descs:
        g = int(pg.pool_group_index)
        pools[g] = [
            (int(p.base_address), int(p.slot_bytes))
            for p in sorted(pg.pools, key=lambda p: int(p.pool_index))
        ]
        counts[g] = int(pg.num_slots)
    return pools, counts


def check_pool_groups(mgr, geometry):
    layout = geometry.layout
    pools, counts = native_pool_groups(mgr)
    assert set(layout.pool_groups) == set(pools)
    assert geometry.pool_groups == tuple(sorted(pools))
    for g, native in pools.items():
        widths = tuple(w for _, w in native)
        assert layout.pool_groups[g] == widths
        assert geometry.page_bytes[g] == sum(widths)
        # Slot i at base + i * the layout's slot width: a pool is its slots back to back.
        assert [(p.base, p.slot_bytes * p.num_slots) for p in geometry.device_pools[g]] == [
            (base, width * counts[g]) for base, width in native
        ]


def check_buffers(mgr, layout):
    """Every (layer, role) of the manager appears once, inside its pool's slot."""
    seen = {}
    for b in layout.buffers:
        assert (b.layer, b.role) not in seen
        seen[(b.layer, b.role)] = b
        width = layout.pool_groups[b.pool[0]][b.pool[1]]
        assert 0 <= b.offset and b.offset + b.size <= width
    internal = sorted({int(bid.layer_id) for bid in mgr.impl.all_buffer_ids})
    global_of = dict(zip(internal, global_layer_ids(mgr, internal)))
    native = {(global_of[int(bid.layer_id)], str(bid.role)) for bid in mgr.impl.all_buffer_ids}
    assert set(seen) == native


@gpu
def test_a_full_attention_cache_is_one_layer_group_over_one_pool_group(real_manager):
    with real_manager() as mgr:
        geometry = derive_layout(mgr)
        layout = geometry.layout

        assert layout.tokens_per_block == geometry.tokens_per_block == REAL_TPB
        (group,) = layout.layer_groups
        assert group.kind == "attention"
        assert group.window is None
        assert group.layers == (0, 1)
        g = int(mgr.impl.get_life_cycle_pool_group_indices()[0])
        assert group.pool_group == g
        assert geometry.pool_group_of == (g,)
        assert geometry.windows == (None,)
        assert geometry.sink_blocks == (0,)
        assert geometry.recurrent == (False,)
        assert geometry.shards == ((1, 0),)
        check_pool_groups(mgr, geometry)
        check_buffers(mgr, layout)
        # K and V are head-sharded (INDEXED): four heads of ``tokens_per_block * head_dim`` fp16
        # values on this rank.
        for b in layout.buffers:
            assert b.mapper is BufferMapper.INDEXED
            assert b.transfer and b.expansion == 1
            assert b.geometry == BufferGeometry(bytes_per_head=REAL_TPB * 64 * 2, num_heads=4)
        assert group.shard == ShardDesc()


@gpu
def test_two_windows_are_two_layer_groups_with_their_own_windows(real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        geometry = derive_layout(mgr)
        layout = geometry.layout

        assert len(layout.layer_groups) == 2
        native = [int(g) for g in mgr.impl.get_life_cycle_pool_group_indices()]
        assert list(geometry.pool_group_of) == native
        by_layer = {}
        for lg, group in enumerate(layout.layer_groups):
            assert group.pool_group == native[lg]
            assert geometry.windows[lg] == group.window
            for layer in group.layers:
                by_layer[layer] = group.window
        # Layer 0 slides; layer 1's window reaches max_seq_len and normalizes to full attention.
        assert mgr.max_seq_len <= 256
        assert by_layer == {0: WINDOW, 1: None}
        check_pool_groups(mgr, geometry)
        check_buffers(mgr, layout)


@gpu
def test_layout_id_ignores_slot_counts_and_addresses(real_manager):
    with real_manager(max_tokens=2048) as small:
        small_id = layout_id(derive_layout(small).layout)
        small_pools = native_pool_groups(small)
    with real_manager(max_tokens=8192) as large:
        large_geometry = derive_layout(large)
        large_pools = native_pool_groups(large)
        assert large_pools[1] != small_pools[1], "the two caches must differ in slot counts"
        assert layout_id(large_geometry.layout) == small_id


@gpu
def test_layout_id_leaves_the_window_out(real_manager):
    """One window for every layer: the same single layer group, only its window differs. The
    window says which blocks exist, not how a block's bytes read, so the id stays."""
    with real_manager(windows=[WINDOW]) as mgr:
        before = derive_layout(mgr).layout
    with real_manager(windows=[2 * WINDOW]) as mgr:
        after = derive_layout(mgr).layout
    assert [g.window for g in before.layer_groups] != [g.window for g in after.layer_groups]
    assert layout_id(after) == layout_id(before)


@gpu
def test_layout_id_follows_the_tokens_per_block(real_manager):
    with real_manager() as mgr:
        before = derive_layout(mgr).layout
    with real_manager(tokens_per_block=64) as mgr:
        after = derive_layout(mgr).layout
    assert len(after.layer_groups) == len(before.layer_groups)
    assert layout_id(after) != layout_id(before)


@gpu
def test_layout_id_leaves_the_element_type_out(real_manager):
    """FP16 and BF16 caches differ in no size or offset, so they share a ``layout_id``; what the
    bytes mean is for the caller's scope to say."""
    from tensorrt_llm.bindings import DataType

    with real_manager(dtype=DataType.HALF) as mgr:
        fp16, fp16_type = derive_layout(mgr).layout, mgr.dtype
    with real_manager(dtype=DataType.BF16) as mgr:
        bf16, bf16_type = derive_layout(mgr).layout, mgr.dtype
    assert fp16_type != bf16_type, "the two caches hold the same element type: proves nothing"
    assert layout_id(bf16) == layout_id(fp16)


def fp4_mla(kit, monkeypatch, backend):
    """A three-layer FP4 MLA manager on ``backend``, shut down when the block exits."""
    from tensorrt_llm._torch.attention.backends.fp4_mla import (
        FP4_MLA_ATTENTION_BACKEND_ENV,
        FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
    )
    from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
    from tensorrt_llm._torch.kimi_k3_cache_policy import KIMI_K3_BF16_KV_LAYERS_ENV
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.delenv(KIMI_K3_BF16_KV_LAYERS_ENV, raising=False)
    return kit.managed(
        lambda: Fp4MlaKVCacheManagerV2(
            KvCacheConfig(
                max_tokens=512, dtype="nvfp4", enable_block_reuse=True, host_cache_size=0
            ),
            CacheType.SELFKONLY,
            num_layers=3,
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=128,
            max_seq_len=512,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            dtype=DataType.NVFP4,
            max_num_tokens=512,
            pretrained_config=SimpleNamespace(kv_lora_rank=512),
        )
    )


@gpu
@skip_pre_hopper
def test_fp4_mla_buffers_carry_the_declared_global_layer_ids(kit, monkeypatch):
    """The tail lives on internal layers past the model's; in the layout its buffers take the ids
    the manager declares (``2 * layer + 1``), next to the cache's ``2 * layer``."""
    with fp4_mla(kit, monkeypatch, "triton") as mgr:
        buffers = {(b.layer, b.role) for b in derive_layout(mgr).layout.buffers}
        assert {layer for layer, role in buffers if role == "key"} == {0, 2, 4}
        assert {layer for layer, role in buffers if role == "mla_hp_tail"} == {1, 3, 5}


M3_TPB = 128
M3_KV_HEADS = 2
M3_HEAD_DIM = 128
M3_INDEX_DIM = 128
M3_SPARSE_LAYER = 3
M3_KV_DTYPES = {"nvfp4": "NVFP4", "fp8": "FP8"}


def minimax_m3(kit, kv_dtype, max_tokens=1024):
    """A four-layer MiniMax-M3 manager: layers 0 to 2 dense, layer 3 sparse with a bf16 index-K.
    With NVFP4 KV only the sparse layer is NVFP4 (packed K/V and block scales); dense layers FP8.
    Shut down when the block exits."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3KVCacheManagerV2
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    return kit.managed(
        lambda: MiniMaxM3KVCacheManagerV2(
            kv_cache_config=KvCacheConfig(
                max_tokens=max_tokens, dtype=kv_dtype, enable_block_reuse=True, host_cache_size=0
            ),
            kv_cache_type=CacheType.SELF,
            num_layers=4,
            num_kv_heads=M3_KV_HEADS,
            head_dim=M3_HEAD_DIM,
            tokens_per_block=M3_TPB,
            max_seq_len=512,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            dtype=getattr(DataType, M3_KV_DTYPES[kv_dtype]),
            vocab_size=32000,
            max_num_tokens=512,
            sparse_layer_ids=[M3_SPARSE_LAYER],
            disable_index_value_layer_ids=[M3_SPARSE_LAYER],
            sparse_index_dim=M3_INDEX_DIM,
            sparse_attention_config=SimpleNamespace(implementation="msa", indexer_kv_dtype="bf16"),
        )
    )


def check_minimax_m3_nvfp4(mgr, layout):
    """Every buffer once, with the size, mapper and heads its storage has: FP8 K/V on dense layers,
    half-width K/V plus one scale byte per 16 elements on the sparse layer, replicated index-K."""
    check_buffers(mgr, layout)
    fp8 = (M3_TPB * M3_KV_HEADS * M3_HEAD_DIM, BufferMapper.INDEXED, M3_KV_HEADS)
    packed = (fp8[0] // 2, BufferMapper.INDEXED, M3_KV_HEADS)
    scale = (fp8[0] // 16, BufferMapper.INDEXED, M3_KV_HEADS)
    expected = {(layer, role): fp8 for layer in range(4) for role in ("key", "value")}
    expected.update({(M3_SPARSE_LAYER, role): packed for role in ("key", "value")})
    expected.update(
        {(M3_SPARSE_LAYER, role): scale for role in ("key_block_scale", "value_block_scale")}
    )
    expected[(M3_SPARSE_LAYER, "index_key")] = (
        M3_TPB * M3_INDEX_DIM * 2,
        BufferMapper.REPLICATED,
        None,
    )
    described = {
        (b.layer, b.role): (b.size, b.mapper, b.geometry.num_heads) for b in layout.buffers
    }
    assert described == expected
    assert all(b.transfer and b.expansion == 1 for b in layout.buffers)
    assert sorted(layer for group in layout.layer_groups for layer in group.layers) == [0, 1, 2, 3]
    assert all(group.shard == ShardDesc() for group in layout.layer_groups)


def minimax_m3_nvfp4_lies(layout):
    """Layouts that each misstate one NVFP4 fact: a lost scale, a scale on a dense layer, a scale
    as wide as its data and a head-sharded index-K."""
    scale = next(b for b in layout.buffers if b.role == "key_block_scale")
    index = next(b for b in layout.buffers if b.role == "index_key")

    def buffers(bufs):
        return dataclasses.replace(layout, buffers=tuple(bufs))

    def swap(old, new):
        return buffers(new if b is old else b for b in layout.buffers)

    return {
        "scale dropped": buffers(b for b in layout.buffers if b is not scale),
        "scale on a dense layer": buffers((*layout.buffers, dataclasses.replace(scale, layer=0))),
        "scale as wide as its data": swap(
            scale, dataclasses.replace(scale, size=scale.size * 8, geometry=BufferGeometry())
        ),
        "index-K head-sharded": swap(index, dataclasses.replace(index, mapper=BufferMapper.NHD)),
    }


@gpu
@skip_pre_hopper
def test_minimax_m3_nvfp4_describes_its_packed_kv_and_block_scales(kit):
    with minimax_m3(kit, "nvfp4") as mgr:
        geometry = derive_layout(mgr)
        check_pool_groups(mgr, geometry)
        check_minimax_m3_nvfp4(mgr, geometry.layout)
        for name, lie in minimax_m3_nvfp4_lies(geometry.layout).items():
            with pytest.raises(AssertionError):
                check_minimax_m3_nvfp4(mgr, lie)
                pytest.fail(f"the {name} lie went unnoticed")


def check_minimax_m3_ids(builds, id_of):
    """``id_of(layout, slot_counts)`` is one id for every NVFP4 build and another for FP8."""
    ids = {}
    for kv_dtype, layout, counts in builds:
        ids.setdefault(kv_dtype, set()).add(id_of(layout, counts))
    assert len(ids["nvfp4"]) == 1, "the NVFP4 builds disagree"
    assert ids["nvfp4"] != ids["fp8"], "NVFP4 and FP8 share an id"


@gpu
@skip_pre_hopper
def test_minimax_m3_nvfp4_and_fp8_caches_have_their_own_layout_ids(kit):
    """NVFP4 builds that differ only in slot count agree; an FP8 cache of the same model reads its
    bytes differently."""
    builds = []
    for kv_dtype, max_tokens in (("nvfp4", 1024), ("fp8", 1024), ("nvfp4", 16384)):
        with minimax_m3(kit, kv_dtype, max_tokens) as mgr:
            builds.append((kv_dtype, derive_layout(mgr).layout, native_pool_groups(mgr)[1]))
    assert builds[0][2] != builds[2][2], "the two NVFP4 caches must differ in slot counts"
    check_minimax_m3_ids(builds, lambda layout, counts: layout_id(layout))
    lies = {
        "one id for every cache": (
            lambda layout, counts: bytes(32),
            "NVFP4 and FP8 share an id",
        ),
        "an id over slot counts": (
            lambda layout, counts: layout_id(layout) + repr(sorted(counts.items())).encode(),
            "the NVFP4 builds disagree",
        ),
    }
    for name, (lie, caught_by) in lies.items():
        with pytest.raises(AssertionError, match=caught_by):
            check_minimax_m3_ids(builds, lie)
            pytest.fail(f"{name} went unnoticed")


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize("fp8_ds_mla", [False, True], ids=["fp8", "fp8_ds_mla"])
def test_the_deepseek_v4_layout_covers_every_virtual_layer_and_pool(
    deepseek_v4_manager, fp8_ds_mla
):
    with deepseek_v4_manager(fp8_ds_mla=fp8_ds_mla) as mgr:
        assert mgr.use_fp8_ds_mla == fp8_ds_mla
        geometry = derive_layout(mgr)
        layout = geometry.layout

        life_cycles = mgr._life_cycle_by_layer_group()
        assert len(layout.layer_groups) == len(life_cycles) >= 2
        assert list(geometry.windows) == [lc.window_size for lc in life_cycles]
        assert None in geometry.windows and any(w is not None for w in geometry.windows)

        # Several pools per group: a page is their slots back to back.
        native = {int(pg.pool_group_index): pg for pg in mgr.impl.pool_group_descs}
        assert max(len(pg.pools) for pg in native.values()) >= 2
        for g, pg in native.items():
            widths = tuple(
                int(p.slot_bytes) for p in sorted(pg.pools, key=lambda p: int(p.pool_index))
            )
            assert layout.pool_groups[g] == widths
            assert geometry.page_bytes[g] == sum(widths)

        # Buffers and layer groups name layers by global layer id: for virtual layers, model layer
        # times the number of attention types plus a type the internal layer holds.
        internal = sorted(int(i) for group in mgr.impl.layer_grouping for i in group)
        global_of = dict(zip(internal, global_layer_ids(mgr, internal)))
        assert len(set(global_of.values())) == len(internal)
        virtual = mgr._layer_attn_to_layer_id
        num_types = max(t.value for t in type(next(iter(virtual))[1])) + 1
        for (model_layer, attn_type), layer_id in virtual.items():
            assert global_of[layer_id] // num_types == model_layer
            held = {t.value for (m, t), lid in virtual.items() if lid == layer_id}
            assert global_of[layer_id] % num_types in held
        described = {(b.layer, b.role) for b in layout.buffers}
        assert described == {
            (global_of[int(b.layer_id)], str(b.role)) for b in mgr.impl.all_buffer_ids
        }
        layers = sorted(layer for group in layout.layer_groups for layer in group.layers)
        assert layers == sorted(global_of.values())
        # One KV head: every layer group is the same content on every rank.
        assert all(group.shard == ShardDesc() for group in layout.layer_groups)
