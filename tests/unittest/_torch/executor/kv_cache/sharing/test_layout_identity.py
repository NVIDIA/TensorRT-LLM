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
manager scripting TP ranks, pipeline stages and recurrent groups (CPU) and real ``KVCacheManagerV2``
instances (GPU): each description's source and what moves the id. Part names: ``test_names.py``."""

import dataclasses
import enum
import hashlib
import inspect
import json
import subprocess
import sys
import types
import weakref
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm._torch.pyexecutor.kv_cache import page_layout, page_table, sharing
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    Axis,
    BufferDesc,
    LayerGroupDesc,
    MapperKind,
    PageLayout,
    RankTopology,
    RoleLayout,
    ShardDesc,
    Tensor,
    head_major,
    replicated,
    sectioned,
    token_major,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import attention_shard
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    MemoryKind,
    MemoryRegion,
    StagingOptions,
    _layout,
    _lender,
    _manager,
    attach_lender,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._identity import Identity
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import (
    LAYOUT_ID_VERSION,
    canonical_bytes,
    canonical_order,
    derive_layout,
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
    **change,
) -> PageLayout:
    g0, g1 = group_ids
    pool_groups = {g0: tuple(slot_bytes), g1: (64,)}
    layer_groups = (
        LayerGroupDesc(
            kind="attention",
            kept_window=windows[0],
            num_sink_blocks=0,
            pool_group=g0,
            layers=layers[0],
            shard=shard,
        ),
        LayerGroupDesc(
            kind="attention",
            kept_window=windows[1],
            num_sink_blocks=0,
            pool_group=g1,
            layers=layers[1],
            shard=shard,
        ),
    )
    buffers = []
    for layer in layers[0]:
        buffers += [
            BufferDesc(layer, "key", g0, 0, 50 * (layer // 2), 50),
            BufferDesc(layer, "scale", g0, 1, 20 * (layer // 2), 20),
        ]
    for layer in layers[1]:
        buffers.append(BufferDesc(layer, "key", g1, 0, 0, 64))
    layout = PageLayout(tokens_per_block, pool_groups, layer_groups, tuple(buffers))
    return dataclasses.replace(layout, **change)


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
        pytest.param(lambda: _first_group(make_layout(), num_sink_blocks=3), id="sinks"),
        pytest.param(lambda: make_layout(windows=(None, 128)), id="window"),
        pytest.param(lambda: make_layout(windows=(64, 64)), id="full_to_window"),
        pytest.param(lambda: make_layout(shard=ShardDesc(2, 1)), id="shard"),
        pytest.param(lambda: _first_group(make_layout(), model_layers=(0, 0)), id="model_layers"),
        pytest.param(
            lambda: dataclasses.replace(make_layout(), buffers=make_layout().buffers[::-1]),
            id="buffer_order",
        ),
    ],
)
def test_layout_id_leaves_out_what_does_not_decide_how_bytes_read(variant):
    """Local numbering and order, sinks and windows (which blocks exist, not how their bytes read),
    the shard position, which rides in the name so every rank computes the same ``layout_id``, and
    the model layer each layer belongs to. Buffers are described by global layer, so their listing
    order does not matter either."""
    assert layout_id(variant()) == layout_id(make_layout())


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "change",
    [
        lambda: make_layout(tokens_per_block=64),
        lambda: make_layout(layers=((0, 1), (2,))),
        lambda: make_layout(slot_bytes=(104, 40)),
        lambda: _replace_buffers(make_layout(), tensors=head_major(20, 2, 4)),
        lambda: _replace_buffers(make_layout(), tensors=head_major(20, 4, 8)),
        lambda: _replace_buffers(make_layout(), tensors=token_major(20, 2, 2, 4)),
        lambda: _replace_buffers(make_layout(), tensors=sectioned((15, 5), 2)),
        lambda: _replace_buffers(make_layout(), transferred=False),
        lambda: _replace_buffers(make_layout(), expansion=2),
        lambda: _replace_buffers(make_layout(), role="scale2"),
        lambda: _first_group(make_layout(), kind="state"),
        lambda: dataclasses.replace(
            make_layout(),
            buffers=tuple(
                dataclasses.replace(b, slot_offset=b.slot_offset + 10) if b.role == "scale" else b
                for b in make_layout().buffers
            ),
        ),
    ],
    ids=[
        "tokens_per_block",
        "layer_membership",
        "slot_width",
        "split_by_head",
        "head_size",
        "token_major",
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
def test_layout_id_is_the_hash_of_the_documented_version_4_encoding():
    """The canonical encoding, spelled out here byte for byte so a change to it cannot pass
    unnoticed."""
    layout = make_layout(layers=((0,), (1,)))
    expected = (
        '{"buffers":['
        '{"layer":0,"offset":0,"role":"key","size":50},'
        '{"layer":0,"offset":100,"role":"scale","size":20},'
        '{"layer":1,"offset":0,"role":"key","size":64}],'
        '"groups":[{"layers":[0],"slot_bytes":[100,40]},{"layers":[1],"slot_bytes":[64]}],'
        '"tokens_per_block":32,"v":4}'
    ).encode()
    assert LAYOUT_ID_VERSION == 4
    assert canonical_bytes(layout_document(layout)) == expected
    assert layout_id(layout) == hashlib.sha256(expected).digest()


@pytest.mark.cpu_only
def test_named_keys_at_their_defaults_are_left_out_and_others_are_written():
    layout = make_layout(layers=((0,), (1,)))
    plain = layout_document(layout)["buffers"][1]
    assert set(plain) == {"layer", "role", "offset", "size"}
    changed = layout_document(
        _replace_buffers(layout, tensors=sectioned((15, 5), 2), transferred=False, expansion=4)
    )["buffers"][1]
    assert changed["tensors"] == [
        {
            "axes": [{"name": "byte", "extent": 30, "stride": 1, "split": ["tp"], "held": 15}],
            "name": "s0",
        },
        {
            "axes": [{"name": "byte", "extent": 10, "stride": 1, "split": ["tp"], "held": 5}],
            "name": "s1",
            "offset": 15,
        },
    ]
    assert changed["transfer"] is False
    assert changed["expansion"] == 4
    assert layout_document(_first_group(layout, kind="state"))["groups"][0]["kind"] == "state"
    assert "kind" not in layout_document(layout)["groups"][0]


@pytest.mark.cpu_only
def test_enums_encode_by_name_and_a_whole_buffer_is_its_default_tensor():
    whole = _replace_buffers(make_layout(), tensors=replicated(20))
    assert whole == make_layout() and layout_id(whole) == layout_id(make_layout())

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
    outside = layout.buffers + (BufferDesc(7, "key", 9, 0, 0, 8),)
    with pytest.raises(ValueError, match="not in the layout"):
        layout_document(dataclasses.replace(layout, buffers=outside))
    twice = layout.buffers + (layout.buffers[0],)
    with pytest.raises(ValueError, match="appears twice"):
        layout_document(dataclasses.replace(layout, buffers=twice))


# -- geometry and buffers ------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_tensors_that_do_not_fit_their_buffer_are_refused():
    with pytest.raises(ValueError, match="reaches byte 40"):
        BufferDesc(0, "conv", 0, 0, 0, 30, tensors=sectioned((20, 20), 2))
    with pytest.raises(ValueError, match="reaches byte 32"):
        BufferDesc(0, "k", 0, 0, 0, 30, tensors=head_major(32, 4, 8))
    with pytest.raises(ValueError, match="whole number"):
        head_major(30, 8, 8)
    with pytest.raises(ValueError, match="repeat"):
        BufferDesc(0, "k", 0, 0, 0, 30, tensors=replicated(30) + replicated(30))


@pytest.mark.cpu_only
def test_an_axis_says_how_its_split_holds():
    with pytest.raises(ValueError, match="says how many indices"):
        Axis("head", 8, 4, ("tp",))
    with pytest.raises(ValueError, match="says how many indices"):
        Axis("head", 8, 4, held=2)
    with pytest.raises(ValueError, match="not split"):
        Axis("row", 8, 4, cyclic=True)
    with pytest.raises(ValueError, match="splits over"):
        Axis("head", 8, 4, ("pp",), 2)
    with pytest.raises(ValueError, match="splits two axes"):
        Tensor((Axis("a", 4, 4, ("tp",), 2), Axis("b", 4, 1, ("tp",), 2)))


@pytest.mark.cpu_only
def test_an_expansion_must_divide_the_block():
    with pytest.raises(ValueError, match="positive"):
        BufferDesc(0, "k", 0, 0, 0, 32, expansion=0)
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
    checked = RoleLayout(section_bytes=sections, bytes_per_head=2)
    assert checked == RoleLayout(section_bytes=(4, 4), bytes_per_head=2)
    assert type(checked.section_bytes) is tuple
    assert all(type(s) is int for s in checked.section_bytes)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "fields", [{"section_bytes": (4, 0)}, {"bytes_per_head": 0}, {"section_bytes": (8, -4)}]
)
def test_validation_refuses_sizes_that_are_not_positive(fields):
    with pytest.raises(ValueError, match="positive"):
        RoleLayout(**fields)


def described(b):
    """(kind, heads on this rank, bytes per head) a buffer's tensors amount to: "REPLICATED" alike
    on every rank, "SECTIONED" split by byte, else "NHD" or "INDEXED" by where its head axis lies;
    heads and bytes are None without a head axis."""
    if not b.split:
        return "REPLICATED", None, None
    (tensor, *rest) = b.tensors
    axes = {a.name: a for a in tensor.axes}
    if rest or "head" not in axes:
        return "SECTIONED", None, None
    kind = "NHD" if tensor.axes[0].name == "token" else "INDEXED"
    return kind, axes["head"].held, b.size // axes["head"].held


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
    default), one pool per role, and has window ``windows[i]`` and ``sinks[i]`` sink tokens. It
    builds its page table with ``KVCacheManagerV2``'s methods, from the state that manager keeps."""

    get_page_table = KVCacheManagerV2.get_page_table
    _page_head_split = KVCacheManagerV2._page_head_split
    _page_rebuilds_untransferred = KVCacheManagerV2._page_rebuilds_untransferred
    _page_layer_configs = KVCacheManagerV2._page_layer_configs
    _page_layer_ids = KVCacheManagerV2._page_layer_ids
    _page_kv_heads = KVCacheManagerV2._page_kv_heads
    _page_dtype = KVCacheManagerV2._page_dtype
    _page_quant_mode = KVCacheManagerV2._page_quant_mode

    def _virtual_layers(self):
        """The model layers and attention types a test names in ``_layer_attn_to_layer_id``."""
        return page_table.virtual_layers(getattr(self, "_layer_attn_to_layer_id", None))

    _model_has_attention_window = False
    _model_has_sparse_buffers = False
    _model_attention_window_pattern = ()
    _has_cp_helix = False
    _helix_cp_rank = 0
    _helix_cp_size = 1
    dtype = None

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
        self.num_layers = len(internal)
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
        self.windows = windows
        self.kv_cache_manager_py_config = SimpleNamespace(layers=layers)
        self.impl = SimpleNamespace(
            layer_grouping=[list(lids) for _, lids, _ in groups],
            pool_group_descs=list(descs.values()),
            get_life_cycle_pool_group_indices=lambda: list(pool_group_of),
            init_config=self.kv_cache_manager_py_config,
        )

    def _attention_read_window(self, layer_group):
        """Attention reads the whole window it keeps."""
        return self.windows[layer_group]

    def _page_buffer_tensors(self, layer_id, role, size, tokens, state):
        """Each role spread as ``mappers`` names it ("all" for the roles it does not list):
        replicated, sectioned by ``layouts``' section bytes, or split by head, head-major (INDEXED)
        or token-major (NHD), ``layouts``' bytes per head or the layer's KV heads apart."""
        kind = self.mappers.get(role, self.mappers["all"])
        layout = self.layouts.get(role, RoleLayout())
        if kind == MapperKind.REPLICATED:
            return replicated(size)
        if kind == MapperKind.SECTIONED:
            split = ("tp", "cp") if state else ("tp",)
            ranks, _ = page_layout.grid_place(page_table.rank_topology(self.mapping), split)
            return sectioned(layout.section_bytes, ranks, split)
        return self._page_head_split(
            layer_id,
            size,
            tokens,
            state,
            head_bytes=layout.bytes_per_head,
            token_major_=kind == MapperKind.NHD,
        )

    def _page_untransferred_roles(self):
        return frozenset(self.ignored)


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
    head_major = derive_layout(attention(1, 0, 4)).page_layout
    token_major = derive_layout(attention(1, 0, 4, mappers={"all": MapperKind.NHD})).page_layout
    assert layout_id(head_major) != layout_id(token_major), TOKEN_MAJOR_SHARES
    assert {described(b)[0] for b in token_major.buffers} == {"NHD"}
    assert {described(b)[0] for b in head_major.buffers} == {"INDEXED"}
    hnd = derive_layout(attention(1, 0, 4, mappers={"all": MapperKind.HND})).page_layout
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
    "unaligned_heads": ((4, 6, False), [ShardDesc(4, r) for r in range(4)]),
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
    assert [geo.page_layout.layer_groups[0].shard for geo in geometries] == shares
    assert [geo.shards for geo in geometries] == [((s.count, s.index),) for s in shares]
    assert len({layout_id(geo.page_layout) for geo in geometries}) == 1
    keys = np.full((1, 32), ord("k"), dtype=np.uint8)
    names = [
        Identity(b"head-shares", layout_id(geo.page_layout), geo.layers, geo.shards)
        .object_keys(0, keys)[0]
        .tobytes()
        for geo in geometries
    ]
    for rank, share in enumerate(shares):
        for other, other_share in enumerate(shares):
            assert (names[rank] == names[other]) == (share == other_share)


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
        got = [attention_shard(tp_size, r, per_rank, total) for r in range(tp_size)]
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
    assert derive_layout(replicated_only).page_layout.layer_groups[0].shard == ShardDesc()
    mixed = Manager(
        [kv_group(heads_per_rank=4, extra=[index])],
        tp_size=2,
        tp_rank=1,
        num_kv_heads=8,
        mappers={"index_key": MapperKind.REPLICATED},
    )
    assert derive_layout(mixed).page_layout.layer_groups[0].shard == ShardDesc(2, 1)


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
    layout = derive_layout(manager).page_layout
    assert layout.layer_groups[0].shard == ShardDesc()
    assert {b.role: b.transferred for b in layout.buffers} == {"index_key": True, "scratch": False}


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
    groups = derive_layout(manager).page_layout.layer_groups
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
        assert geometry.page_layout.layer_groups[0].shard == ShardDesc(tp_size, rank), (
            f"rank {rank}: the group's share is not the rank's own"
        )
        identity = Identity(
            b"scope", layout_id(geometry.page_layout), geometry.layers, geometry.shards
        )
        names |= {row.tobytes() for row in identity.object_keys(0, keys)}
    assert len(names) == tp_size * len(keys), "two ranks name a block alike"


@pytest.mark.cpu_only
def test_recurrent_state_is_described_and_sharded_by_rank():
    geometry = derive_layout(recurrent_state())
    layout = geometry.page_layout
    (group,) = layout.layer_groups
    assert group.kind == "state" and group.shard == ShardDesc(2, 1)
    assert geometry.recurrent == (True,) and geometry.windows == (None,)
    by_role = {b.role: b for b in layout.buffers}
    assert by_role["conv_state"].tensors == sectioned((32, 32, 32), 2, ("tp", "cp"))
    assert by_role["ssm_state"].tensors == head_major(128, 4, 8, ("tp", "cp"))


def tensor_ranks_per_group(topology):
    """Tensor ranks in the attention data-parallel group of a rank, by its topology."""
    return topology.tp_size // topology.dp_size if topology.attention_dp else topology.tp_size


def state_grid(topology):
    """``(ranks, place)`` of the grid recurrent state splits over, by a rank's topology: tensor
    ranks, none under attention data parallelism, tensor times context ranks (context-minor) under
    Helix."""
    if topology.attention_dp:
        return 1, 0
    if topology.cp_size > 1:
        return (
            topology.tp_size * topology.cp_size,
            topology.tp_rank * topology.cp_size + topology.cp_rank,
        )
    return topology.tp_size, topology.tp_rank


def attention_and_state(tp_size, tp_rank, attention_dp, num_kv_heads, cp_size=1, cp_rank=0):
    """A rank of ``num_kv_heads`` KV heads in layers 0 and 1 and recurrent state in layers 2 and
    3; under Helix, context rank ``cp_rank`` of ``cp_size``."""
    heads = -(-num_kv_heads // (1 if attention_dp else tp_size))
    manager = Manager(
        [
            kv_group(lids=(0, 1), heads_per_rank=heads),
            (SsmLayerConfig, [2, 3], [("conv_state", 96, None), ("ssm_state", 128, None)]),
        ],
        tp_size=tp_size,
        tp_rank=tp_rank,
        attention_dp=attention_dp,
        num_kv_heads=num_kv_heads,
        mappers={"conv_state": MapperKind.SECTIONED},
        layouts={
            "conv_state": RoleLayout(section_bytes=(32, 32, 32)),
            "ssm_state": RoleLayout(bytes_per_head=32),
        },
    )
    if cp_size > 1:
        manager._has_cp_helix = True
        manager._helix_cp_rank, manager._helix_cp_size = cp_rank, cp_size
        manager.mapping.cp_size, manager.mapping.cp_rank = cp_size, cp_rank
    return manager


# (TP size, attention DP, Helix context ranks, KV heads) of every instance the sweep builds.
SHARD_SWEEP = [
    (tp_size, attention_dp, cp_size, num_kv_heads)
    for tp_size in (1, 2, 4, 8)
    for attention_dp in ((False, True) if tp_size > 1 else (False,))
    for cp_size in ((1,) if attention_dp else (1, 2, 4))
    for num_kv_heads in (1, 2, 4, 6, 8, 12, 16)
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tp_size, attention_dp, cp_size, num_kv_heads",
    SHARD_SWEEP,
    ids=[f"tp{t}{'_adp' if a else ''}_cp{c}_heads{h}" for t, a, c, h in SHARD_SWEEP],
)
def test_published_shards_are_the_shares_the_topology_gives(
    tp_size, attention_dp, cp_size, num_kv_heads
):
    """Every rank's published shards name what its topology gives it.

    Attention: a rank at place ``p`` of an attention data-parallel group of ``n`` tensor ranks
    holds the heads from ``p * T // n``; its share's first head, ``index * T // count``, is that
    head, and two ranks have equal shares exactly when they hold the same heads. State: the share
    is the rank's place in the grid recurrent state splits over.
    """
    layouts = [
        derive_layout(
            attention_and_state(tp_size, tp_rank, attention_dp, num_kv_heads, cp_size, cp_rank)
        ).page_layout
        for tp_rank in range(tp_size)
        for cp_rank in range(cp_size)
    ]
    first_heads = []
    for layout in layouts:
        attention, state = layout.layer_groups
        topology = layout.topology
        ranks = tensor_ranks_per_group(topology)
        first = (topology.tp_rank % ranks) * num_kv_heads // ranks
        first_heads.append(first)
        share = attention.shard
        assert share.index * num_kv_heads // share.count == first, (topology, share)
        count, index = state_grid(topology)
        assert state.shard == (ShardDesc(count, index) if count > 1 else ShardDesc()), (
            topology,
            state.shard,
        )
    for one, first_one in zip(layouts, first_heads):
        for other, first_other in zip(layouts, first_heads):
            same_share = one.layer_groups[0].shard == other.layer_groups[0].shard
            assert same_share == (first_one == first_other), (one.topology, other.topology)


# -- which ranks of an instance hold the same bytes ---------------------------------------------


def replica_instance(tp=1, pp=1, cp=1, attention_dp=False, groups_of=None):
    """Every rank's layout by flat rank, ``pp * (tp * cp) + tp * cp + cp``.

    ``groups_of(pp_rank, tp_rank, cp_rank)`` gives each rank's ``(kind, layers, shard)`` groups.
    """
    layouts = []
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            for cp_rank in range(cp):
                groups = tuple(
                    LayerGroupDesc(kind, None, 0, g, tuple(layers), shard)
                    for g, (kind, layers, shard) in enumerate(groups_of(pp_rank, tp_rank, cp_rank))
                )
                layouts.append(
                    PageLayout(
                        tokens_per_page=4,
                        pool_slot_bytes={g: (64,) for g in range(len(groups))},
                        layer_groups=groups,
                        buffers=(),
                        topology=RankTopology(
                            tp_size=tp,
                            tp_rank=tp_rank,
                            pp_size=pp,
                            pp_rank=pp_rank,
                            cp_size=cp,
                            cp_rank=cp_rank,
                            dp_size=tp if attention_dp else 1,
                            dp_rank=tp_rank if attention_dp else 0,
                            attention_dp=attention_dp,
                        ),
                    )
                )
    return layouts


def two_heads_over_four_ranks():
    """Ranks 0 and 1 hold KV head 0, ranks 2 and 3 KV head 1."""
    return replica_instance(
        tp=4, groups_of=lambda p, t, c: [("attention", (0, 1), ShardDesc(2, t // 2))]
    )


@pytest.mark.cpu_only
def test_ranks_of_equal_shards_are_replicas_and_replicated_buffers_are_on_every_rank():
    ranks = two_heads_over_four_ranks()
    assert [page_layout.replica_ranks(ranks, r, 0) for r in range(4)] == [
        (0, 1),
        (0, 1),
        (2, 3),
        (2, 3),
    ]
    assert page_layout.replica_ranks(ranks, 3, 0, replicated=True) == (0, 1, 2, 3)


@pytest.mark.cpu_only
def test_replicas_are_listed_by_ascending_flat_rank():
    """The lowest rank holding a share, which publishes it to a store, comes first."""
    ranks = two_heads_over_four_ranks()
    assert [page_layout.replica_ranks(ranks, r, 0)[0] for r in range(4)] == [0, 0, 2, 2]
    assert page_layout.replica_ranks(ranks, 2, 0, replicated=True)[0] == 0
    assert page_layout.replica_ranks(ranks, 3, 0) == (2, 3)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tp_size,heads,expected",
    [
        (8, (2, 4), [(0, 1), (0, 1), (2, 3), (2, 3), (4, 5), (4, 5), (6, 7), (6, 7)]),
        (6, (2, 3), [(0, 1), (0, 1), (2,), (3,), (4, 5), (4, 5)]),
        (8, (4, 8), [(rank,) for rank in range(8)]),
    ],
)
def test_mixed_head_replication_matches_every_buffer(tp_size, heads, expected):
    ranks = [
        derive_layout(mixed_shares(tp_size, rank, heads)).page_layout for rank in range(tp_size)
    ]
    assert [page_layout.replica_ranks(ranks, rank, 0) for rank in range(tp_size)] == expected


@pytest.mark.cpu_only
def test_each_attention_data_parallel_rank_holds_its_own_requests():
    ranks = replica_instance(
        tp=4, attention_dp=True, groups_of=lambda p, t, c: [("attention", (0, 1), ShardDesc())]
    )
    for r in range(4):
        assert page_layout.replica_ranks(ranks, r, 0) == (r,)
        assert page_layout.replica_ranks(ranks, r, 0, replicated=True) == (r,)


@pytest.mark.cpu_only
def test_pipeline_stages_hold_other_layers_and_groups_of_another_kind_other_bytes():
    def groups_of(pp_rank, tp_rank, cp_rank):
        if pp_rank == 0:
            return [("attention", (0, 1), ShardDesc()), ("state", (2,), ShardDesc())]
        return [("attention", (3, 4), ShardDesc()), ("attention", (2,), ShardDesc())]

    ranks = replica_instance(tp=2, pp=2, groups_of=groups_of)
    assert page_layout.replica_ranks(ranks, 1, 0) == (0, 1)
    assert page_layout.replica_ranks(ranks, 3, 0) == (2, 3)
    # Layer 2 is state on stage 0 and attention on stage 1: no rank of the other stage matches.
    assert page_layout.replica_ranks(ranks, 0, 1) == (0, 1)
    assert page_layout.replica_ranks(ranks, 2, 1) == (2, 3)


@pytest.mark.cpu_only
def test_helix_ranks_hold_their_own_attention_pages_and_split_state_by_context_rank():
    def groups_of(pp_rank, tp_rank, cp_rank):
        return [("attention", (0,), ShardDesc()), ("state", (1,), ShardDesc(2, cp_rank))]

    ranks = replica_instance(cp=2, groups_of=groups_of)
    assert [page_layout.replica_ranks(ranks, r, 0) for r in range(2)] == [(0,), (1,)]
    assert [page_layout.replica_ranks(ranks, r, 1) for r in range(2)] == [(0,), (1,)]
    # Replicated state is the same on both context ranks.
    assert page_layout.replica_ranks(ranks, 1, 1, replicated=True) == (0, 1)


@pytest.mark.cpu_only
def test_replicas_are_every_rank_of_the_instance_holding_the_bytes():
    """Which of them a caller may choose from is the caller's to filter."""
    parameters = inspect.signature(page_layout.replica_ranks).parameters
    assert list(parameters) == ["ranks", "rank", "group", "replicated"]
    assert parameters["replicated"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["replicated"].default is False


@pytest.mark.cpu_only
def test_head_geometry_is_described_per_buffer():
    layout = derive_layout(attention(2, 0, num_kv_heads=8)).page_layout
    for b in layout.buffers:
        assert b.tensors == head_major(b.size, 4, 8)


@pytest.mark.cpu_only
def test_layer_configs_are_keyed_by_their_layer_id_or_position():
    layers = [
        SimpleNamespace(layer_id="7"),
        SimpleNamespace(layer_id=4),
        SimpleNamespace(),
        SimpleNamespace(layer_id=0),
    ]
    actual = page_table.layer_configs_by_id(SimpleNamespace(layers=layers))
    assert list(actual) == [7, 4, 2, 0]
    assert all(actual[layer_id] is layer for layer_id, layer in zip([7, 4, 2, 0], layers))


@pytest.mark.cpu_only
def test_layers_are_global_ids_stable_across_pipeline_stages():
    """Stage 2 of a pipeline holds model layers 10 and 11 as internal layers 0 and 1."""
    manager = Manager([kv_group()], pp_layers=[10, 11])
    geometry = derive_layout(manager)
    assert geometry.page_layout.layer_groups[0].layers == (10, 11)
    assert geometry.layers == ((10, 11),)
    assert {b.layer for b in geometry.page_layout.buffers} == {10, 11}
    assert manager._page_layer_ids() == {0: (10, 10), 1: (11, 11)}
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
    assert manager._page_layer_ids() == {0: (12, 4), 1: (13, 4), 2: (15, 5)}
    assert derive_layout(manager).layers == ((12, 13, 15),)


@pytest.mark.cpu_only
def test_an_internal_layer_several_attention_types_share_takes_the_first_s_global_id():
    """An internal layer several model layers share is named by its first entry, as native does."""

    class AttentionType(enum.Enum):
        SLIDING = 0
        COMPRESSED = 1
        INDEXER = 2

    manager = Manager([kv_group(lids=(0, 1, 2))])
    manager._layer_attn_to_layer_id = {
        (4, AttentionType.SLIDING): 0,
        (4, AttentionType.COMPRESSED): 1,
        (5, AttentionType.SLIDING): 2,
        (5, AttentionType.COMPRESSED): 1,  # shares model layer 4's compressed cache
    }
    assert manager._page_layer_ids() == {0: (12, 4), 1: (13, 4), 2: (15, 5)}
    assert derive_layout(manager).layers == ((12, 13, 15),)


@pytest.mark.cpu_only
def test_the_layout_carries_the_rank_heads_element_type_and_read_windows():
    """The layout carries native's rank info, KV heads, element type and read windows."""
    manager = Manager(
        [kv_group(lids=(0, 1), heads_per_rank=2), kv_group(lids=(2,), heads_per_rank=2)],
        tp_size=4,
        tp_rank=3,
        attention_dp=False,
        num_kv_heads=8,
        pp_layers=[10, 11, 12],
        windows=[None, 64],
    )
    manager.mapping.pp_size, manager.mapping.pp_rank = 2, 1
    manager.dtype = SimpleNamespace(name="FP8")
    layout = derive_layout(manager).page_layout
    assert layout.topology == RankTopology(tp_size=4, tp_rank=3, pp_size=2, pp_rank=1)
    assert dict(layout.total_kv_heads_by_layer) == {10: 8, 11: 8, 12: 8}
    assert (layout.dtype, layout.quant_mode) == ("FP8", "fp8")
    assert [(g.kept_window, g.read_window) for g in layout.layer_groups] == [(None, None), (64, 64)]
    adp = Manager([kv_group(heads_per_rank=8)], tp_size=4, tp_rank=3, attention_dp=True)
    topology = derive_layout(adp).page_layout.topology
    assert (topology.dp_size, topology.dp_rank, topology.attention_dp) == (4, 3, True)
    state = derive_layout(recurrent_state()).page_layout
    assert (dict(state.total_kv_heads_by_layer), state.dtype, state.quant_mode) == ({}, None, None)


@pytest.mark.cpu_only
def test_a_declared_read_window_narrows_the_kept_one():
    """A declared read window narrows the kept one; one the kept window does not hold is refused."""
    manager = Manager([kv_group(lids=(0,)), kv_group(lids=(1,))], windows=[None, 72])
    manager._attention_read_window = lambda lg: None if lg == 0 else 64
    geometry = derive_layout(manager)
    groups = geometry.page_layout.layer_groups
    assert [(g.kept_window, g.read_window) for g in groups] == [(None, None), (72, 64)]
    assert geometry.windows == (None, 72) and geometry.read_windows == (None, 64)
    for liar in (lambda lg: None if lg == 0 else 80, lambda lg: 64):
        manager._attention_read_window = liar
        with pytest.raises(ValueError, match="does not hold"):
            derive_layout(manager)


@pytest.mark.cpu_only
def test_a_manager_naming_its_own_layers_names_extra_internal_layers():
    """A manager whose internal layers outnumber its model layers (FP4 MLA keeps its tail on
    extra ones) names them itself (``_page_layer_ids``); a layer it leaves unnamed is refused."""
    manager = Manager(
        [kv_group(lids=(1, 0)), (AttentionLayerConfig, [2, 3], [("tail", 64, None)])],
        pp_layers=[0, 1],
        mappers={"tail": MapperKind.REPLICATED},
    )
    named = {1: (2, 1), 0: (0, 0), 2: (1, 0), 3: (3, 1)}
    manager._page_layer_ids = lambda: named
    layout = derive_layout(manager).page_layout
    assert [g.layers for g in layout.layer_groups] == [(0, 2), (1, 3)]
    assert {(b.layer, b.role) for b in layout.buffers if b.role == "tail"} == {
        (1, "tail"),
        (3, "tail"),
    }
    manager._page_layer_ids = lambda: {k: v for k, v in named.items() if k != 3}
    with pytest.raises(ValueError, match=r"names no global layer for internal layers \[3\]"):
        derive_layout(manager)


@pytest.mark.cpu_only
def test_layer_groups_publish_each_layer_s_model_layer():
    """A manager numbering its layers by model layer publishes them as their own model layers:
    a pipeline stage's layers, attention and recurrent state alike."""
    manager = Manager(
        [
            kv_group(lids=(1, 0)),
            (SsmLayerConfig, [2, 3], [("conv_state", 96, None), ("ssm_state", 128, None)]),
        ],
        pp_layers=[7, 6, 8, 9],
        mappers={"conv_state": MapperKind.SECTIONED},
        layouts={"conv_state": RoleLayout(section_bytes=(32, 32, 32))},
    )
    layout = derive_layout(manager).page_layout
    assert [(g.kind, g.layers, g.model_layers) for g in layout.layer_groups] == [
        ("attention", (6, 7), (6, 7)),
        ("state", (8, 9), (8, 9)),
    ]


@pytest.mark.cpu_only
def test_virtual_layers_publish_the_model_layer_they_belong_to():
    """Layer ``3 * m + t`` of model layer ``m`` and attention type ``t`` is model layer ``m``,
    listed beside its layer id; an internal layer two model layers share is its first's."""

    class AttentionType(enum.Enum):
        SLIDING = 0
        COMPRESSED = 1
        INDEXER = 2

    manager = Manager([kv_group(lids=(2, 0)), kv_group(lids=(1, 3))])
    manager._layer_attn_to_layer_id = {
        (4, AttentionType.SLIDING): 0,
        (4, AttentionType.COMPRESSED): 1,
        (5, AttentionType.SLIDING): 2,
        (6, AttentionType.COMPRESSED): 3,
        (6, AttentionType.INDEXER): 1,  # shares model layer 4's compressed cache
    }
    layout = derive_layout(manager).page_layout
    assert [(g.layers, g.model_layers) for g in layout.layer_groups] == [
        ((12, 15), (4, 5)),
        ((13, 19), (4, 6)),
    ]
    assert {lid: ids[1] for lid, ids in manager._page_layer_ids().items()} == {
        0: 4,
        1: 4,
        2: 5,
        3: 6,
    }


@pytest.mark.cpu_only
def test_layers_a_manager_names_publish_the_model_layers_named_beside_them():
    """A manager naming its layers (FP4 MLA: ``2 * m`` for model layer ``m``'s cache and
    ``2 * m + 1`` for its tail) names each one's model layer beside it."""
    manager = Manager(
        [kv_group(lids=(1, 0)), (AttentionLayerConfig, [2, 3], [("tail", 64, None)])],
        pp_layers=[0, 1],
        mappers={"tail": MapperKind.REPLICATED},
    )
    manager._page_layer_ids = lambda: {1: (2, 1), 0: (0, 0), 2: (1, 0), 3: (3, 1)}
    layout = derive_layout(manager).page_layout
    assert [(g.layers, g.model_layers) for g in layout.layer_groups] == [
        ((0, 2), (0, 1)),
        ((1, 3), (0, 1)),
    ]


@pytest.mark.cpu_only
def test_a_layer_group_names_one_model_layer_per_layer():
    group = LayerGroupDesc("attention", None, 0, 0, [12, 13, 15], model_layers=[4, 4, 5])
    assert group.model_layers == (4, 4, 5) and type(group.model_layers) is tuple
    plain = LayerGroupDesc("attention", None, 0, 0, [3, 1])
    assert plain.model_layers == (3, 1) == plain.layers
    assert group != LayerGroupDesc("attention", None, 0, 0, (12, 13, 15))
    assert hash(group) != hash(LayerGroupDesc("attention", None, 0, 0, (12, 13, 15)))
    for bad in ([4, 4], [4, 4, 5, 6], [4, -1, 5], [4, True, 5], [4, 4.0, 5]):
        with pytest.raises(ValueError, match="model layer"):
            LayerGroupDesc("attention", None, 0, 0, (12, 13, 15), model_layers=bad)


@pytest.mark.cpu_only
def test_a_buffer_with_its_own_tokens_per_block_is_expanded():
    manager = Manager(
        [kv_group(extra=[("index_key", 64, 2)])], mappers={"index_key": MapperKind.REPLICATED}
    )
    layout = derive_layout(manager).page_layout
    assert {b.role: b.expansion for b in layout.buffers} == {"key": 1, "value": 1, "index_key": 2}
    plain = derive_layout(
        Manager(
            [kv_group(extra=[("index_key", 64, None)])],
            mappers={"index_key": MapperKind.REPLICATED},
        )
    ).page_layout
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
    assert [d.pool_group for d in geometry.page_layout.layer_groups] == [3, 3, 1]
    assert geometry.pool_groups == (1, 3)
    assert geometry.windows == (None, 8, None)
    assert geometry.sink_blocks == (0, 2, 0)  # 5 sink tokens take two 4-token blocks
    assert geometry.recurrent == (False, False, True)
    assert geometry.layers == ((0,), (1,), (2,))
    assert geometry.shards == ((1, 0),) * 3
    assert dict(geometry.page_bytes) == {3: 2 * HEAD_BYTES, 1: 96}
    assert dict(geometry.page_layout.pool_slot_bytes) == {3: (HEAD_BYTES, HEAD_BYTES), 1: (96,)}
    pools = {
        g: [(p.group, p.index, p.base, p.slot_bytes, p.num_slots) for p in ps]
        for g, ps in geometry.device_pools.items()
    }
    assert pools == {
        3: [(3, 0, 0x4000, HEAD_BYTES, 8), (3, 1, 0x4100, HEAD_BYTES, 8)],
        1: [(1, 0, 0x2000, 96, 8)],
    }


@pytest.mark.cpu_only
def test_a_helix_rank_s_layout_counts_the_tokens_of_its_own_page():
    """Under Helix the layout counts one page's tokens; the lender's block is ``cp_size`` pages."""
    manager = attention(1, 0, 4)
    manager._has_cp_helix, manager._helix_cp_rank, manager._helix_cp_size = True, 1, 2
    manager.mapping.cp_size, manager.mapping.cp_rank = 2, 1
    manager.impl.init_config.tokens_per_block = 2 * TPB  # the runtime's block: both ranks' pages
    geometry = derive_layout(manager)
    layout = geometry.page_layout
    assert (layout.topology.cp_size, layout.topology.cp_rank) == (2, 1), layout.topology
    assert layout.tokens_per_page == manager.tokens_per_block == TPB, layout.tokens_per_page
    assert geometry.cp == (1, 2) and geometry.tokens_per_block == 2 * TPB, geometry


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
            "fd26ca16e4400324b2590f51fdff89ce93369776249ffdeb08e10ecaf07c3652",
        ),
        (
            recurrent_state,
            "ddbb463ba7a5ec34551df4ac7de306d27cd28d4bbf984e9645004ec26f005747",
        ),
    ],
    ids=["head_sharded_with_expanded_replicated_role", "recurrent_state"],
)
def test_the_derived_layout_id_is_pinned(build, expected):
    """The derivation and the encoding together: a change here moves every name built on it."""
    assert layout_id(derive_layout(build()).page_layout).hex() == expected


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
    layout = geometry.page_layout
    pools, counts = native_pool_groups(mgr)
    assert set(layout.pool_slot_bytes) == set(pools)
    assert geometry.pool_groups == tuple(sorted(pools))
    for g, native in pools.items():
        widths = tuple(w for _, w in native)
        assert layout.pool_slot_bytes[g] == widths
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
        width = layout.pool_slot_bytes[b.pool_group][b.pool_index]
        assert 0 <= b.slot_offset and b.slot_offset + b.size <= width
    global_of = {lid: ids[0] for lid, ids in mgr._page_layer_ids().items()}
    native = {(global_of[int(bid.layer_id)], str(bid.role)) for bid in mgr.impl.all_buffer_ids}
    assert set(seen) == native


@gpu
def test_a_full_attention_cache_is_one_layer_group_over_one_pool_group(real_manager):
    with real_manager() as mgr:
        geometry = derive_layout(mgr)
        layout = geometry.page_layout

        assert layout.tokens_per_page == geometry.tokens_per_block == REAL_TPB
        (group,) = layout.layer_groups
        assert group.kind == "attention"
        assert group.kept_window is None
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
            assert described(b) == ("INDEXED", 4, REAL_TPB * 64 * 2)
            assert b.transferred and b.expansion == 1
        assert group.shard == ShardDesc()


@gpu
def test_two_windows_are_two_layer_groups_with_their_own_windows(real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        geometry = derive_layout(mgr)
        layout = geometry.page_layout

        assert len(layout.layer_groups) == 2
        native = [int(g) for g in mgr.impl.get_life_cycle_pool_group_indices()]
        assert list(geometry.pool_group_of) == native
        by_layer = {}
        for lg, group in enumerate(layout.layer_groups):
            assert group.pool_group == native[lg]
            assert geometry.windows[lg] == group.kept_window
            for layer in group.layers:
                by_layer[layer] = group.kept_window
        # Layer 0 slides; layer 1's window reaches max_seq_len and normalizes to full attention.
        assert mgr.max_seq_len <= 256
        assert by_layer == {0: WINDOW, 1: None}
        check_pool_groups(mgr, geometry)
        check_buffers(mgr, layout)


@gpu
def test_layout_id_ignores_slot_counts_and_addresses(real_manager):
    with real_manager(max_tokens=2048) as small:
        small_id = layout_id(derive_layout(small).page_layout)
        small_pools = native_pool_groups(small)
    with real_manager(max_tokens=8192) as large:
        large_geometry = derive_layout(large)
        large_pools = native_pool_groups(large)
        assert large_pools[1] != small_pools[1], "the two caches must differ in slot counts"
        assert layout_id(large_geometry.page_layout) == small_id


@gpu
def test_layout_id_leaves_the_window_out(real_manager):
    """One window for every layer: the same single layer group, only its window differs. The
    window says which blocks exist, not how a block's bytes read, so the id stays."""
    with real_manager(windows=[WINDOW]) as mgr:
        before = derive_layout(mgr).page_layout
    with real_manager(windows=[2 * WINDOW]) as mgr:
        after = derive_layout(mgr).page_layout
    assert [g.kept_window for g in before.layer_groups] != [
        g.kept_window for g in after.layer_groups
    ]
    assert layout_id(after) == layout_id(before)


@gpu
def test_layout_id_follows_the_tokens_per_block(real_manager):
    with real_manager() as mgr:
        before = derive_layout(mgr).page_layout
    with real_manager(tokens_per_block=64) as mgr:
        after = derive_layout(mgr).page_layout
    assert len(after.layer_groups) == len(before.layer_groups)
    assert layout_id(after) != layout_id(before)


@gpu
def test_layout_id_leaves_the_element_type_out(real_manager):
    """FP16 and BF16 caches differ in no size or offset, so they share a ``layout_id``; what the
    bytes mean is for the caller's scope to say."""
    from tensorrt_llm.bindings import DataType

    with real_manager(dtype=DataType.HALF) as mgr:
        fp16, fp16_type = derive_layout(mgr).page_layout, mgr.dtype
    with real_manager(dtype=DataType.BF16) as mgr:
        bf16, bf16_type = derive_layout(mgr).page_layout, mgr.dtype
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
        # A cache with more layers than the model reads the windows it keeps.
        mgr._model_attention_windows = dict.fromkeys(mgr.pp_layers, 1)
        derived = derive_layout(mgr)
        buffers = {(b.layer, b.role) for b in derived.page_layout.buffers}
        assert {layer for layer, role in buffers if role == "key"} == {0, 2, 4}
        assert {layer for layer, role in buffers if role == "mla_hp_tail"} == {1, 3, 5}
        assert any(window is not None for window in derived.windows), derived.windows
        assert derived.read_windows == derived.windows, (derived.windows, derived.read_windows)


@gpu
@skip_pre_hopper
def test_fp4_mla_publishes_each_layer_s_model_layer(kit, monkeypatch):
    """The cache and the tail of model layer ``m``, layers ``2 * m`` and ``2 * m + 1``, are both
    model layer ``m``."""
    with fp4_mla(kit, monkeypatch, "triton") as mgr:
        mgr._model_attention_windows = dict.fromkeys(mgr.pp_layers, 1)
        groups = derive_layout(mgr).page_layout.layer_groups
        published = {layer: m for g in groups for layer, m in zip(g.layers, g.model_layers)}
        assert published == {layer: layer // 2 for layer in range(6)}, published


@gpu
@skip_pre_hopper
def test_fp4_mla_lending_requires_the_sidecars_decode_reads(kit, monkeypatch):
    """A lender fetch must deliver V sidecars because it has no native receive rebuild."""
    with fp4_mla(kit, monkeypatch, "triton") as mgr:
        mgr._model_attention_windows = dict.fromkeys(mgr.pp_layers, 1)
        layout = derive_layout(mgr).page_layout
        required = [b for b in layout.buffers if b.role in ("mla_v_scale", "mla_v_packed")]
        assert "mla_v_scale" in {b.role for b in required}
        assert all(b.transferred for b in required)


@gpu
@skip_pre_blackwell
def test_retained_precision_twins_describe_their_original_layer_s_head_stride(kit):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    with kit.managed(
        lambda: KVCacheManagerV2(
            kv_cache_config=KvCacheConfig(
                dtype="mixed_nvfp4_bfloat16",
                mixed_precision_first_n=1,
                mixed_precision_last_n=65,
                enable_block_reuse=True,
                pool_ratio=None,
                max_gpu_total_bytes=128 << 20,
            ),
            kv_cache_type=CacheType.SELF,
            num_layers=2,
            num_kv_heads=[2, 4],
            head_dim=64,
            tokens_per_block=128,
            max_seq_len=1024,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            dtype=DataType.BF16,
        )
    ) as mgr:
        layout = derive_layout(mgr).page_layout
        for original, internal in mgr.high_layers.items():
            global_layer = mgr._page_layer_ids()[internal][0]
            actual = mgr.get_buffers(original, "HND", mixed_precision_pool="high")
            head_bytes = actual.stride(-3) * actual.element_size()
            buffers = [b for b in layout.buffers if b.layer == global_layer]
            assert buffers and all(described(b)[2] == head_bytes for b in buffers)


@pytest.mark.cpu_only
def test_fp4_mla_installs_its_model_layers_beside_its_layer_ids():
    """A manager the FP4 MLA policy installs its methods on names each layer and its model layer,
    and lends the roles native's transfer leaves to the receiver's rebuild, as an FP4 MLA manager
    does."""
    from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
        Fp4MlaV2CacheLayoutPolicy,
    )

    # Model layers 4 and 7: their caches are internal layers 0 and 1, their tails 2 and 3.
    manager = SimpleNamespace(
        _fp4_mla_hp_pool_size=0,
        impl=SimpleNamespace(layer_grouping=[[0, 1], [2, 3]]),
        _cache_manager_layer_ids=[0, 1],
        _hp_manager_layer_ids=[2, 3],
        _bf16_mla_manager_layer_ids=[],
        _fp4_mla_compact_to_local=[0, 1],
        pp_layers=[4, 7],
    )
    Fp4MlaV2CacheLayoutPolicy.install_manager_capabilities(manager)
    assert manager._page_layer_ids() == {0: (8, 4), 1: (14, 7), 2: (9, 4), 3: (15, 7)}
    # Native's transfer skips the V sidecars, which the receiver rebuilds; the page table says so,
    # and a lender, which invokes no rebuild, lends them too.
    assert manager._page_untransferred_roles()
    assert manager._page_rebuilds_untransferred()


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
    fp8 = (M3_TPB * M3_KV_HEADS * M3_HEAD_DIM, "INDEXED", M3_KV_HEADS)
    packed = (fp8[0] // 2, "INDEXED", M3_KV_HEADS)
    scale = (fp8[0] // 16, "INDEXED", M3_KV_HEADS)
    expected = {(layer, role): fp8 for layer in range(4) for role in ("key", "value")}
    expected.update({(M3_SPARSE_LAYER, role): packed for role in ("key", "value")})
    expected.update(
        {(M3_SPARSE_LAYER, role): scale for role in ("key_block_scale", "value_block_scale")}
    )
    expected[(M3_SPARSE_LAYER, "index_key")] = (M3_TPB * M3_INDEX_DIM * 2, "REPLICATED", None)
    assert {(b.layer, b.role): (b.size, *described(b)[:2]) for b in layout.buffers} == expected
    assert all(b.transferred and b.expansion == 1 for b in layout.buffers)
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
            scale,
            dataclasses.replace(
                scale, size=scale.size * 8, tensors=head_major(scale.size * 8, 1, M3_KV_HEADS)
            ),
        ),
        "index-K head-sharded": swap(
            index, dataclasses.replace(index, tensors=token_major(index.size, M3_TPB, 1, 1))
        ),
    }


@gpu
@skip_pre_hopper
def test_minimax_m3_nvfp4_describes_its_packed_kv_and_block_scales(kit):
    with minimax_m3(kit, "nvfp4") as mgr:
        geometry = derive_layout(mgr)
        check_pool_groups(mgr, geometry)
        check_minimax_m3_nvfp4(mgr, geometry.page_layout)
        for name, lie in minimax_m3_nvfp4_lies(geometry.page_layout).items():
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
            builds.append((kv_dtype, derive_layout(mgr).page_layout, native_pool_groups(mgr)[1]))
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
@pytest.mark.parametrize("draft_len", [1, 3])
@pytest.mark.parametrize("swa_only", [False, True], ids=["compressed", "swa_only"])
def test_a_drafting_deepseek_v4_manager_uses_its_full_kept_windows(
    deepseek_v4_manager, draft_len, swa_only
):
    """Without a backend read declaration, lending requires the draft reserves too."""
    from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig

    spec_config = NGramDecodingConfig(max_draft_len=draft_len, max_matching_ngram_size=2)
    with deepseek_v4_manager(
        spec_config=spec_config, compress_ratios=(1,) * 4 if swa_only else None
    ) as mgr:
        assert mgr.max_draft_len == draft_len, mgr.max_draft_len
        groups = derive_layout(mgr).page_layout.layer_groups
        windows = [(g.kept_window, g.read_window) for g in groups]
        assert any(w is not None for w, _ in windows)
        assert any(w is None for w, _ in windows) == (not swa_only)
        for window, read in windows:
            assert read == window, windows


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize("fp8_ds_mla", [False, True], ids=["fp8", "fp8_ds_mla"])
def test_the_deepseek_v4_layout_covers_every_virtual_layer_and_pool(
    deepseek_v4_manager, fp8_ds_mla
):
    with deepseek_v4_manager(fp8_ds_mla=fp8_ds_mla) as mgr:
        assert mgr.use_fp8_ds_mla == fp8_ds_mla
        geometry = derive_layout(mgr)
        layout = geometry.page_layout

        life_cycles = mgr._life_cycle_by_layer_group()
        assert len(layout.layer_groups) == len(life_cycles) >= 2
        assert list(geometry.windows) == [lc.window_size for lc in life_cycles]
        assert None in geometry.windows and any(w is not None for w in geometry.windows)
        assert geometry.read_windows == geometry.windows, "no draft keeps a margin"

        # Several pools per group: a page is their slots back to back.
        native = {int(pg.pool_group_index): pg for pg in mgr.impl.pool_group_descs}
        assert max(len(pg.pools) for pg in native.values()) >= 2
        for g, pg in native.items():
            widths = tuple(
                int(p.slot_bytes) for p in sorted(pg.pools, key=lambda p: int(p.pool_index))
            )
            assert layout.pool_slot_bytes[g] == widths
            assert geometry.page_bytes[g] == sum(widths)

        # Buffers and layer groups name layers by global layer id: for virtual layers, model layer
        # times the number of attention types plus a type the internal layer holds.
        internal = sorted(int(i) for group in mgr.impl.layer_grouping for i in group)
        global_of = {lid: ids[0] for lid, ids in mgr._page_layer_ids().items()}
        assert sorted(global_of) == internal and len(set(global_of.values())) == len(internal)
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


@gpu
@skip_pre_blackwell
def test_deepseek_v4_publishes_each_virtual_layer_s_model_layer(deepseek_v4_manager):
    """Each cache of model layer ``m``, one per attention type, is model layer ``m``; so is the
    last model layer's, as a draft layer after the target model's would be."""
    with deepseek_v4_manager(compress_ratios=(1, 4, 128, 4, 4)) as mgr:
        groups = derive_layout(mgr).page_layout.layer_groups
        virtual = mgr._layer_attn_to_layer_id
        num_types = max(t.value for t in type(next(iter(virtual))[1])) + 1
        published = {layer: m for g in groups for layer, m in zip(g.layers, g.model_layers)}
        assert published == {layer: layer // num_types for layer in published}, published
        assert set(published.values()) == set(range(5)) == {m for m, _ in virtual}
        assert any(layer != m for layer, m in published.items())


@gpu
@skip_pre_blackwell
def test_csa2_publishes_each_virtual_layer_s_model_layer(kit, monkeypatch):
    """CSA2 numbers each model layer's SWA, global and compressor caches apart; each is that
    model layer."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    config = KvCacheConfig(
        max_gpu_total_bytes=128 << 20, host_cache_size=0, enable_block_reuse=False, dtype="fp8"
    )
    with kit.managed(
        lambda: CSA2CacheManager(
            config,
            CacheType.SELFKONLY,
            num_layers=4,
            tokens_per_block=128,
            max_seq_len=2048,
            max_batch_size=2,
            max_num_tokens=2048,
            mapping=Mapping(),
            dtype=DataType.BF16,
            vocab_size=8192,
            layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3), window_size=256, index_topk=4),
        )
    ) as mgr:
        groups = derive_layout(mgr).page_layout.layer_groups
        virtual = mgr._layer_attn_to_layer_id
        num_types = max(t.value for t in type(next(iter(virtual))[1])) + 1
        published = {layer: m for g in groups for layer, m in zip(g.layers, g.model_layers)}
        assert published == {layer: layer // num_types for layer in published}, published
        assert set(published.values()) == {m for m, _ in virtual}
        assert any(layer != m for layer, m in published.items())


@gpu
def test_a_hybrid_stage_publishes_its_model_layers_for_state_and_attention(gdn_manager):
    """On pipeline stage 1 of 2, a gated delta-net and attention manager holds model layers 4 to
    7 as its internal layers 0 to 3; both kinds of group publish the model layers."""
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=2, rank=1, pp_size=2)
    with gdn_manager(mapping=mapping, pattern="MMM*MMM*") as mgr:
        assert list(mgr.pp_layers) == [4, 5, 6, 7], mgr.pp_layers
        groups = derive_layout(mgr).page_layout.layer_groups
        assert {g.kind for g in groups} == {"state", "attention"}
        for g in groups:
            assert g.model_layers == g.layers, g
        assert sorted(layer for g in groups for layer in g.layers) == [4, 5, 6, 7]


@pytest.mark.cpu_only
def test_private_reuse_policy_stays_internal_and_disables_named_rows():
    """A PRIVATE layer group lends no block by name, and the public layout does not show it."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import SlotQueue
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy

    manager = Manager([kv_group(lids=(0,)), kv_group(lids=(1,))])
    manager.impl.init_config.layers[1].reuse_policy = AttentionReusePolicy.PRIVATE
    internal = derive_layout(manager)
    assert internal.named_groups == (True, False)
    assert internal.prefix_required == (True, False)
    assert all(not hasattr(group, "reuse_policy") for group in internal.page_layout.layer_groups)
    ref = weakref.ref(manager)
    lender = _lender._ManagerLender(ref, internal, None, (), SlotQueue({}), ref)
    pages = list(range(4))
    kv = SimpleNamespace(
        num_committed_tokens=4 * TPB,
        reuse_status=[SimpleNamespace(complete=True, endpoint=4 * TPB)] * 2,
        num_blocks=len(pages),
        get_base_page_indices=lambda lg: pages,
        get_aggregated_page_indices=lambda lg, valid_only=False: pages,
    )

    def blocks(entries):
        return dict(zip(entries.layer_group_indices, map(list, entries.block_ordinals)))

    every = lender._block_entries_of_range(manager, kv, 0, 4 * TPB)
    assert blocks(every) == {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    assert blocks(lender._committed_entries(kv, every)) == {0: [0, 1, 2, 3], 1: []}
    named = lender._block_entries_of_range(manager, kv, 0, 4 * TPB, named=True)
    assert blocks(named) == {0: [0, 1, 2, 3], 1: []}


@pytest.mark.cpu_only
def test_a_layout_names_the_groups_lent_by_name_or_builds_no_lender():
    """Which groups lend by name has no default: no layout and no lender builds without it."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import SlotQueue

    manager = Manager([kv_group(lids=(0,)), kv_group(lids=(1,))])
    internal = derive_layout(manager)
    fields = {f.name: getattr(internal, f.name) for f in dataclasses.fields(internal)}
    del fields["named_groups"]
    with pytest.raises(TypeError, match="named_groups"):
        _layout.ManagerLayout(**fields)
    unnamed = SimpleNamespace(**fields, num_layer_groups=2, pool_groups=internal.pool_groups)
    ref = weakref.ref(manager)
    with pytest.raises(AttributeError, match="named_groups"):
        _lender._ManagerLender(ref, unnamed, None, (), SlotQueue({}), ref)


@gpu
@pytest.mark.parametrize("explicit", [True, False], ids=["model_window", "backend_fallback"])
def test_mtp_read_windows_follow_the_model_or_the_actual_backend_fallback(
    kit, real_manager, explicit
):
    from tensorrt_llm._torch.models.modeling_exaone_moe import get_exaone_attention_window
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_windows=[256, 0],
        sliding_window=256,
        mtp_layer_types=None,
    )
    windows = {
        i: get_exaone_attention_window(config, i, i >= 2) if explicit else None for i in range(3)
    }
    assert windows == {0: 256 if explicit else None, 1: None, 2: None}
    with real_manager(
        num_layers=2,
        max_seq_len=2048,
        max_tokens=4096,
        tokens_per_block=128,
        windows=[256, 2048, 2048],
        spec_config=MTPDecodingConfig(max_draft_len=1),
        attention_read_windows=windows,
        block_reuse_config={"policy": "per_request"},
    ) as mgr:
        lender = attach_lender(mgr, scope=b"mtp-window", staging=StagingOptions(511))
        groups = lender.page_layout.layer_groups
        swa = next(i for i, g in enumerate(groups) if g.kept_window is not None)
        assert (groups[swa].kept_window, groups[swa].read_window) == (257, 256 if explicit else 257)
        req = kit.make_request(
            91, list(range(511)), llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        )
        assert mgr.prepare_disagg_gen_init(req)
        event = mgr.take_disagg_receive_ready(req)
        lease = lender.lend_write(req, 0, 511, **{**kit.DEVICE_ALL, "ready_event": event})
        mgr._stream.synchronize()
        view = lease.poll()
        assert view is not None, lease.failure
        kit.fill_sentinel(mgr, req)
        mgr._stream.synchronize()
        masks = [
            np.isin(g.page_ordinals, (2, 3)) | (g.layer_group_index != swa) for g in view.groups
        ]
        (bounds,) = lender.return_writes([(lease, masks, True)])
        assert bounds == lender.poll_resume_bounds(req), bounds
        if explicit:
            assert bounds == (511, 511)
        else:
            assert bounds.max_position < bounds.min_position


@gpu
def test_a_configured_window_below_the_model_s_is_the_window_attention_reads(real_manager):
    """A configured window below the model's is kept, and is then the window attention reads."""
    with real_manager(
        num_layers=2,
        max_seq_len=2048,
        max_tokens=4096,
        tokens_per_block=128,
        windows=[256, 2048],
        attention_read_windows={0: 4096, 1: None},
        block_reuse_config={"policy": "per_request"},
    ) as mgr:
        lender = attach_lender(mgr, scope=b"configured-window", staging=StagingOptions(511))
        (group,) = [g for g in lender.page_layout.layer_groups if g.kept_window is not None]
        assert group.kept_window < 4096 and group.read_window == group.kept_window, group


@gpu
@skip_pre_blackwell
@pytest.mark.parametrize("draft_len", [0, 8])
def test_csa2_lender_uses_kept_swa_and_compressor_windows(kit, monkeypatch, draft_len):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    spec = DSparkDecodingConfig(max_draft_len=draft_len) if draft_len else None
    config = KvCacheConfig(
        max_gpu_total_bytes=128 << 20, host_cache_size=0, enable_block_reuse=False, dtype="fp8"
    )
    with kit.managed(
        lambda: CSA2CacheManager(
            config,
            CacheType.SELFKONLY,
            num_layers=4,
            tokens_per_block=128,
            max_seq_len=2048,
            max_batch_size=2,
            max_num_tokens=2048,
            mapping=Mapping(),
            dtype=DataType.BF16,
            vocab_size=8192,
            layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3), window_size=256, index_topk=4),
            spec_config=spec,
        )
    ) as mgr:
        public = derive_layout(mgr).page_layout
        margin = mgr.max_draft_len + mgr.reuse_match_backoff
        seen = set()
        for desc in public.layer_groups:
            roles = {buffer.role for buffer in public.buffers if buffer.layer in desc.layers}
            seen.update(roles)
            if roles & {"csa2_global", "csa2_index"}:
                assert desc.read_window is None
            else:
                assert desc.read_window == desc.kept_window
                if "csa2_swa" in roles:
                    assert desc.read_window == mgr.layout.window_size + margin
                else:
                    assert desc.read_window == 2 + margin
        kinds = ("swa", "global", "index", "compressor_kv", "compressor_score")
        assert seen == {f"csa2_{kind}" for kind in kinds}


@gpu
@skip_pre_blackwell
def test_csa2_lender_requires_the_full_bounded_replay_window(kit, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    read_window, replay_window = 256, 512
    spec = DSparkDecodingConfig(max_draft_len=8)
    spec.__dict__["draft_is_embedded_in_target"] = True
    with kit.managed(
        lambda: CSA2CacheManager(
            KvCacheConfig(
                max_gpu_total_bytes=128 << 20,
                host_cache_size=0,
                enable_block_reuse=False,
                dtype="fp8",
            ),
            CacheType.SELFKONLY,
            num_layers=4,
            tokens_per_block=128,
            max_seq_len=2048,
            max_batch_size=2,
            max_num_tokens=2048,
            mapping=Mapping(),
            dtype=DataType.BF16,
            vocab_size=8192,
            layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3), window_size=read_window, index_topk=4),
            spec_config=spec,
            pretrained_config=SimpleNamespace(sliding_window=replay_window),
        )
    ) as mgr:
        assert mgr._decoder_replay_window == replay_window
        layout = derive_layout(mgr).page_layout
        swa = [
            group
            for group in layout.layer_groups
            if any(
                buffer.role == "csa2_swa" and buffer.layer in group.layers
                for buffer in layout.buffers
            )
        ]
        margin = mgr.max_draft_len + mgr.reuse_match_backoff
        assert all(group.read_window == group.kept_window for group in swa)
        assert {group.kept_window for group in swa} == {
            read_window + margin,
            replay_window + margin,
        }


@pytest.mark.cpu_only
def test_csa2_decoder_replay_reads_rebuilt_rows_inside_its_kept_margin():
    """A replay query can read rebuilt SWA rows outside the final query's read window."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
        plan_decoder_replay,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        gather_rows,
        row_bytes,
        store_rows,
    )

    prompt, read_window, replay_window, tokens_per_block = 768, 256, 512, 128
    seq_lens = torch.tensor([prompt], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        mapping=None,
        num_tokens=prompt,
        num_contexts=1,
        num_seqs=1,
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens,
        prompt_lens=[prompt],
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
    )
    request = SimpleNamespace(prompt_len=prompt, py_ced_replay=None)
    plan = plan_decoder_replay(metadata, replay_window, [request], private_decoder=True)
    floor = prompt - replay_window
    assert plan is not None and plan.swa_floors == [floor]
    assert plan.rows.tolist() == list(range(floor, prompt))

    positions = plan.rows
    requests = torch.zeros(len(positions), dtype=torch.int64)
    floors = torch.tensor(plan.swa_floors, dtype=torch.int64)
    pages = torch.arange(prompt // tokens_per_block, dtype=torch.int32).unsqueeze(0)
    reads = torch.empty((len(positions), read_window), dtype=torch.int64)
    writes = torch.empty(len(positions), dtype=torch.int64)
    visible = torch.empty(len(positions), dtype=torch.int64)
    CSA2TrtllmMetadata._refresh_swa_tensor_outputs(
        positions,
        requests,
        floors,
        tokens_per_block,
        read_window,
        [(pages, reads, writes, visible, 0)],
        read_floors=floors,
    )
    assert writes.tolist() == list(range(floor, prompt))
    assert reads[0, :-1].eq(-1).all() and reads[0, -1] == floor
    assert reads[read_window, 0] == floor + 1

    pool = torch.empty((prompt, row_bytes(512, "swa")), dtype=torch.uint8)
    old_rows = torch.full((prompt, 512), -1, dtype=torch.bfloat16)
    rebuilt_rows = torch.ones((len(positions), 512), dtype=torch.bfloat16)
    store_rows(pool, torch.arange(prompt), old_rows, "swa")
    store_rows(pool, writes, rebuilt_rows, "swa")
    assert gather_rows(pool, torch.tensor([floor - 1]), 512, "swa").eq(-1).all()
    assert gather_rows(pool, reads[0], 512, "swa")[:-1].eq(0).all()
    assert gather_rows(pool, reads[read_window], 512, "swa").eq(1).all()


@pytest.mark.cpu_only
@pytest.mark.parametrize("policy", ["OPTIONAL", "PRIVATE"])
@pytest.mark.parametrize("window", [None, 64], ids=["full_attention", "windowed"])
def test_attach_requires_a_window_for_every_non_required_group(monkeypatch, policy, window):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy

    manager = Manager([kv_group(lids=(0,)), kv_group(lids=(1,))], windows=[None, window])
    manager.mapping.cp_size = 1
    manager.impl.init_config.layers[1].reuse_policy = getattr(AttentionReusePolicy, policy)
    monkeypatch.setattr(_manager, "require_v2", lambda owner: None)
    if window is None:
        with pytest.raises(ValueError, match="non-REQUIRED.*window"):
            _lender._checked_layout(manager)
    else:
        assert _lender._checked_layout(manager).windows == (None, 64)


@pytest.mark.cpu_only
@pytest.mark.parametrize("attention_dp", [False, True], ids=["tp", "attention_dp"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_a_state_group_splits_over_the_grid_its_kind_sets(tp_size, attention_dp):
    """State groups split over the TP ranks, and over none under attention data parallelism."""
    grid = 1 if attention_dp else tp_size
    for rank in range(tp_size):
        manager = Manager(
            [(SsmLayerConfig, [0, 1], [("ssm_state", 128, None)])],
            tp_size=tp_size,
            tp_rank=rank,
            attention_dp=attention_dp,
            layouts={"ssm_state": RoleLayout(bytes_per_head=32)},
        )
        (group,) = derive_layout(manager).page_layout.layer_groups
        assert group.shard == (ShardDesc(grid, rank) if grid > 1 else ShardDesc()), (rank, group)


@gpu
def test_a_v2_manager_whose_tensors_overrun_a_buffer_attaches_no_lender(real_manager, monkeypatch):
    """A V2 manager declaring a tensor that reaches past its buffer fails to attach and keeps no
    lender."""
    with real_manager() as mgr:
        monkeypatch.setattr(
            mgr,
            "_page_buffer_tensors",
            lambda layer, role, size, tokens, state: replicated(size + 1),
        )
        with pytest.raises(ValueError, match="reaches byte"):
            attach_lender(mgr, scope=b"overrun", staging=StagingOptions(64))
        assert mgr._sharing is None


@gpu
def test_a_manager_whose_regions_overlap_attaches_no_lender(real_manager, monkeypatch):
    """Two device regions that overlap fail the attach, which keeps no lender."""
    regions_of = _lender._device_regions

    def overlapping(*args):
        regions = regions_of(*args)
        first = regions[0]
        return [*regions, dataclasses.replace(first, address=first.address + first.nbytes - 1)]

    with real_manager() as mgr:
        monkeypatch.setattr(_lender, "_device_regions", overlapping)
        with pytest.raises(ValueError, match="overlap in DEVICE memory"):
            attach_lender(mgr, scope=b"overlap", staging=StagingOptions(64))
        assert mgr._sharing is None


# -- the public layout types ---------------------------------------------------------------------

LAYOUT_TYPES = (
    PageLayout,
    LayerGroupDesc,
    BufferDesc,
    RankTopology,
    ShardDesc,
    RoleLayout,
    Tensor,
    Axis,
)
# Each layout type's public names beyond its fields and ``unchecked``.
LAYOUT_TYPE_EXTRAS = {BufferDesc: {"split"}, Tensor: {"split"}}


def pinned_fields(cls):
    """``name`` or ``name=default`` per field, in order; a factory default shows as ``{}``."""
    out = []
    for f in dataclasses.fields(cls):
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        out.append(f.name if default is dataclasses.MISSING else f"{f.name}={default!r}")
    return " ".join(out)


@pytest.mark.cpu_only
def test_the_layout_types_keep_their_module_fields_and_members():
    from tensorrt_llm._torch.disaggregation.resource import page as native

    assert native.MapperKind is MapperKind and native.RoleLayout is RoleLayout
    assert sorted(page_layout.__all__) == sorted(
        [
            *(t.__name__ for t in (*LAYOUT_TYPES, MapperKind)),
            "GRID_DIMENSIONS",
            "Progression",
            "grid_place",
            "head_major",
            "held",
            "layout_from_dict",
            "layout_to_dict",
            "replica_ranks",
            "replicated",
            "sectioned",
            "token_major",
        ]
    )
    assert not hasattr(sharing, "PageLayout") and "PageLayout" not in sharing.__all__
    assert _layout.ManagerLayout is page_table.ManagerLayout
    assert _layout.DevicePool is page_table.DevicePool
    assert pinned_fields(PageLayout) == (
        "tokens_per_page pool_slot_bytes layer_groups buffers topology=RankTopology(tp_size=1, "
        "tp_rank=0, pp_size=1, pp_rank=0, cp_size=1, cp_rank=0, dp_size=1, dp_rank=0, "
        "attention_dp=False) total_kv_heads_by_layer={} dtype=None quant_mode=None "
        "device_region_index={} staging_region_index={} rebuilds_untransferred=False"
    )
    assert pinned_fields(LayerGroupDesc) == (
        "kind kept_window num_sink_blocks pool_group layers shard=ShardDesc(count=1, index=0) "
        "read_window=None model_layers=None"
    )
    assert pinned_fields(BufferDesc) == (
        "layer role pool_group pool_index slot_offset size tensors=() transferred=True expansion=1"
    )
    assert pinned_fields(Tensor) == "axes offset=0 name=''"
    assert pinned_fields(Axis) == "name extent stride split=() held=None cyclic=False"
    assert pinned_fields(RankTopology) == (
        "tp_size=1 tp_rank=0 pp_size=1 pp_rank=0 cp_size=1 cp_rank=0 dp_size=1 dp_rank=0 "
        "attention_dp=False"
    )
    assert pinned_fields(ShardDesc) == "count=1 index=0"
    assert pinned_fields(RoleLayout) == "section_bytes=None bytes_per_head=None"
    assert [m.name for m in MapperKind] == ["INDEXED", "REPLICATED", "NHD", "SECTIONED"]
    assert set(MapperKind.__members__) == {"INDEXED", "HND", "REPLICATED", "NHD", "SECTIONED"}
    assert MapperKind.HND is MapperKind.INDEXED
    assert {name for name in vars(MapperKind) if not name.startswith("_")} == set(
        MapperKind.__members__
    )
    layout = every_field_set()
    group, buffer = layout.layer_groups[0], layout.buffers[0]
    tensor, axis = buffer.tensors[0], buffer.tensors[0].axes[0]
    for value in (layout, group, buffer, layout.topology, group.shard, tensor, axis):
        cls = type(value)
        assert cls.__dataclass_params__.frozen, cls
        named = {f.name for f in dataclasses.fields(cls)}
        beyond = {name for obj in (cls, value) for name in dir(obj) if not name.startswith("_")}
        assert beyond - named == {"unchecked", *LAYOUT_TYPE_EXTRAS.get(cls, ())}, cls


@pytest.mark.cpu_only
@pytest.mark.parametrize("unchecked", [False, True], ids=["checked", "unchecked"])
def test_a_layout_holds_its_own_read_only_copies_and_compares_by_value(unchecked):
    pools, heads, regions = {0: (100, 40), 1: (64,)}, {0: 8}, {(0, 0): 2}
    groups, buffers = list(make_layout().layer_groups), list(make_layout().buffers)
    build = PageLayout.unchecked if unchecked else PageLayout
    layout = build(
        32, pools, groups, buffers, total_kv_heads_by_layer=heads, device_region_index=regions
    )
    pools[5], heads[9], regions[(9, 9)] = (1,), 1, 9
    groups.pop()
    buffers.clear()
    same = make_layout(total_kv_heads_by_layer={0: 8}, device_region_index={(0, 0): 2})
    assert layout == same and hash(layout) == hash(same), "the caller's values changed the layout"
    assert len({layout, same, dataclasses.replace(same, staging_region_index={0: 1})}) == 2
    assert type(layout.layer_groups) is tuple and type(layout.buffers) is tuple
    for mapping in (
        layout.pool_slot_bytes,
        layout.total_kv_heads_by_layer,
        layout.device_region_index,
        layout.staging_region_index,
    ):
        assert isinstance(mapping, types.MappingProxyType)
        with pytest.raises(TypeError):
            mapping[7] = 0


@pytest.mark.cpu_only
def test_a_read_window_defaults_to_the_kept_window_and_lies_within_it():
    assert LayerGroupDesc("attention", None, 0, 0, (0,)).read_window is None
    assert LayerGroupDesc("attention", 133, 0, 0, (0,)).read_window == 133
    assert LayerGroupDesc("attention", 133, 0, 0, (0,), read_window=128).read_window == 128
    shorter = dataclasses.replace(LayerGroupDesc("attention", None, 0, 0, (0,)), kept_window=64)
    assert shorter.read_window == 64, "a replaced window left the read window behind"
    for window, read in ((64, 65), (64, 0), (64, -1), (None, 8)):
        with pytest.raises(ValueError, match=r"read.window"):
            LayerGroupDesc("attention", window, 0, 0, (0,), read_window=read)
    kept = LayerGroupDesc("attention", 133, 0, 0, (0,))
    with pytest.raises(ValueError, match="read window of 133 tokens"):
        dataclasses.replace(kept, kept_window=64)
    assert dataclasses.replace(kept, kept_window=64, read_window=None).read_window == 64


@pytest.mark.cpu_only
@pytest.mark.parametrize("axis", ["tp", "pp", "cp", "dp"])
def test_a_topology_rank_lies_within_its_size(axis):
    assert getattr(RankTopology(**{f"{axis}_size": 4, f"{axis}_rank": 3}), f"{axis}_rank") == 3
    for size, rank in ((4, 4), (4, -1), (0, 0)):
        with pytest.raises(ValueError, match=axis):
            RankTopology(**{f"{axis}_size": size, f"{axis}_rank": rank})


@pytest.mark.cpu_only
def test_the_quantization_is_one_the_layout_names():
    import typing

    literal, none = typing.get_args(typing.get_type_hints(PageLayout)["quant_mode"])
    assert none is type(None) and typing.get_args(literal) == ("fp8", "nvfp4", "fp8_ds_mla")
    for mode in (None, *typing.get_args(literal)):
        assert make_layout(quant_mode=mode).quant_mode == mode
    for mode in ("FP8", "int8", ""):
        with pytest.raises(ValueError, match="quant_mode"):
            make_layout(quant_mode=mode)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mapping,expected",
    [
        (
            dict(tp_size=4, tp_rank=3, pp_size=2, pp_rank=1, cp_size=1, cp_rank=0, dp_size=1),
            RankTopology(tp_size=4, tp_rank=3, pp_size=2, pp_rank=1),
        ),
        (
            dict(tp_size=4, tp_rank=3, enable_attention_dp=True, dp_size=4),
            RankTopology(tp_size=4, tp_rank=3, dp_size=4, dp_rank=3, attention_dp=True),
        ),
        (dict(tp_size=1, tp_rank=0, cp_size=2, cp_rank=1), RankTopology(cp_size=2, cp_rank=1)),
    ],
    ids=["tensor_and_pipeline", "attention_data_parallel", "helix"],
)
def test_the_topology_is_what_native_rank_info_reads_from_the_mapping(mapping, expected):
    """As native's ``RankInfo``: under attention DP, ``dp_size`` and ``dp_rank`` are the TP ones."""
    assert page_table.rank_topology(SimpleNamespace(**mapping)) == expected
    assert page_table.rank_topology(None) == RankTopology()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "dtype,quant",
    [("BF16", None), ("HALF", None), ("FP8", "fp8"), ("NVFP4", "nvfp4"), (None, None)],
)
def test_the_quantization_follows_the_element_type_the_manager_declares(dtype, quant):
    manager = attention(1, 0, 4)
    if dtype is not None:
        manager.dtype = SimpleNamespace(name=dtype)
    assert (manager._page_dtype(), manager._page_quant_mode()) == (dtype, quant)


@pytest.mark.cpu_only
@pytest.mark.parametrize("ds_mla,quant", [(True, "fp8_ds_mla"), (False, "fp8")])
def test_deepseek_v4_names_its_fp8_pages_with_their_scales(ds_mla, quant):
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )

    manager = DeepseekV4CacheManager.__new__(DeepseekV4CacheManager)
    manager.dtype, manager.use_fp8_ds_mla = SimpleNamespace(name="FP8"), ds_mla
    assert manager._page_quant_mode() == quant


@pytest.mark.cpu_only
def test_the_layout_id_leaves_out_the_topology_heads_element_type_and_regions():
    layout = make_layout()
    shorter = dataclasses.replace(layout.layer_groups[1], read_window=59)
    rich = make_layout(
        topology=RankTopology(tp_size=2, tp_rank=1, pp_size=2, pp_rank=1),
        total_kv_heads_by_layer={0: 8},
        dtype="FP8",
        quant_mode="fp8",
        device_region_index={(0, 0): 4},
        staging_region_index={0: 0},
        layer_groups=(layout.layer_groups[0], shorter),
    )
    assert layout_id(rich) == layout_id(layout)


@pytest.mark.cpu_only
@pytest.mark.parametrize("staged", [True, False], ids=["staging", "in_place_alone"])
def test_a_lender_s_layout_gives_each_pool_its_region_positions(staged):
    """Staging regions come first, one per pool group, then one device region per pool."""
    manager = Manager([kv_group(lids=(0,)), kv_group(lids=(1,))])
    geometry = derive_layout(manager)
    ident = Identity(b"scope", layout_id(geometry.page_layout), geometry.layers, geometry.shards)
    staging = [
        MemoryRegion(f"s{g}", 4096 * (g + 1), 4096, 1024, 4, (g,), MemoryKind.HOST) for g in (0, 1)
    ]
    regions = tuple((staging if staged else []) + _lender._device_regions(manager, geometry, ident))
    ref = weakref.ref(manager)
    lender = _lender._ManagerLender(ref, geometry, ident, regions, None, ref)
    public = lender.page_layout
    first = 2 if staged else 0
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert dict(public.device_region_index) == {pool: first + i for i, pool in enumerate(order)}
    bases = {(p.group, p.index): p.base for pools in geometry.device_pools.values() for p in pools}
    for pool, position in public.device_region_index.items():
        region = regions[position]
        assert region.memory_kind is MemoryKind.DEVICE and region.address == bases[pool], pool
    assert dict(public.staging_region_index) == ({0: 0, 1: 1} if staged else {})
    assert dataclasses.replace(public, device_region_index={}, staging_region_index={}) == (
        geometry.page_layout
    )
    assert lender.page_layout is public, "the layout is built once"


@pytest.mark.cpu_only
@pytest.mark.parametrize("bad", [True, False, 1.5, "2", -1, 0])
def test_layout_values_are_refused_rather_than_truncated(bad):
    """Indices refuse booleans, non-integers and negatives; sizes refuse zero as well."""
    buffer = dict(layer=0, role="kv", pool_group=0, pool_index=0, slot_offset=0, size=16)
    names = ["size", "expansion"]
    if isinstance(bad, bool) or bad != 0:
        names += ["layer", "pool_group", "pool_index", "slot_offset"]
        with pytest.raises(ValueError, match="model layer"):
            LayerGroupDesc("attention", None, 0, 0, (0,), model_layers=(bad,))
    for name in names:
        with pytest.raises(ValueError, match=name):
            BufferDesc(**{**buffer, name: bad})
    for construct in (
        lambda: make_layout(tokens_per_page=bad),
        lambda: make_layout(pool_slot_bytes={0: (bad,)}),
        lambda: make_layout(total_kv_heads_by_layer={0: bad}),
        lambda: RoleLayout(bytes_per_head=bad),
        lambda: RoleLayout(section_bytes=(bad,)),
        lambda: ShardDesc(count=bad),
        lambda: RankTopology(tp_size=bad),
        lambda: LayerGroupDesc("attention", bad, 0, 0, (0,)),
    ):
        with pytest.raises(ValueError):
            construct()


def every_field_set():
    """A layout with each of its own fields away from the default, and parts of every kind."""
    return make_layout(
        layer_groups=(
            LayerGroupDesc(
                "attention", 128, 1, 0, (0, 2), ShardDesc(2, 1), read_window=96, model_layers=(0, 1)
            ),
            LayerGroupDesc("state", None, 0, 1, (1,)),
        ),
        buffers=(
            BufferDesc(0, "key", 0, 0, 0, 64, token_major(64, 4, 2, 8)),
            BufferDesc(2, "scale", 0, 1, 8, 8, transferred=False),
            BufferDesc(1, "conv", 1, 0, 0, 48, sectioned((16, 32), 2), expansion=2),
            BufferDesc(
                1,
                "rows",
                1,
                0,
                0,
                48,
                (
                    Tensor(
                        (Axis("row", 12, 8, ("tp",), cyclic=True), Axis("byte", 8, 1)),
                        offset=8,
                        name="r",
                    ),
                ),
            ),
        ),
        topology=RankTopology(4, 3, 2, 1, 2, 1, 4, 3, True),
        total_kv_heads_by_layer={0: 8, 2: 8},
        dtype="FP8",
        quant_mode="fp8",
        device_region_index={(0, 0): 2, (0, 1): 3, (1, 0): 4},
        staging_region_index={0: 0, 1: 1},
    )


def rebuilt(value):
    """``value`` rebuilt through the unchecked constructors from plain values."""
    if dataclasses.is_dataclass(value):
        parts = {f.name: rebuilt(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return type(value).unchecked(**parts)
    if isinstance(value, types.MappingProxyType):
        return {key: rebuilt(item) for key, item in value.items()}
    return tuple(map(rebuilt, value)) if isinstance(value, tuple) else value


def typed(value):
    """``value`` as nested ``(type name, parts)``, equal only for equal values of equal types."""
    name = type(value).__name__
    if dataclasses.is_dataclass(value):
        return name, tuple(typed(getattr(value, f.name)) for f in dataclasses.fields(value))
    if isinstance(value, types.MappingProxyType):
        return name, tuple(sorted((typed(key), typed(item)) for key, item in value.items()))
    return name, tuple(map(typed, value)) if isinstance(value, tuple) else value


@pytest.mark.cpu_only
def test_each_unchecked_constructor_takes_the_constructor_s_fields_and_defaults():
    for cls in LAYOUT_TYPES:
        parameters = list(inspect.signature(cls.unchecked).parameters.values())
        assert [p.name for p in parameters] == [f.name for f in dataclasses.fields(cls)], cls
        for p, f in zip(parameters, dataclasses.fields(cls)):
            if f.default is not dataclasses.MISSING:
                assert p.default == f.default, (cls, p.name, p.default)
            elif f.default_factory is not dataclasses.MISSING:
                assert p.default == f.default_factory(), (cls, p.name, p.default)
            else:
                assert p.default is inspect.Parameter.empty, (cls, p.name, p.default)


@pytest.mark.cpu_only
@pytest.mark.parametrize("build", [make_layout, every_field_set], ids=["defaults", "every_field"])
def test_an_unchecked_rebuild_equals_the_checked_layout(build):
    """Rebuilt from plain values without checks, a layout and each of its parts come back alike."""
    layout = build()
    again = rebuilt(layout)
    assert again == layout and hash(again) == hash(layout) and typed(again) == typed(layout)
    tensors = [t for b in layout.buffers for t in b.tensors]
    parts = (
        *layout.layer_groups,
        *layout.buffers,
        layout.topology,
        RoleLayout((16, 32), 16),
        *tensors,
        *(a for t in tensors for a in t.axes),
    )
    for value in (layout, ShardDesc(2, 1), *parts):
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        unchecked = type(value).unchecked(**fields)
        assert typed(unchecked) == typed(value) == typed(type(value)(**fields)), value
    assert LayerGroupDesc.unchecked("attention", 64, 0, 0, (0,)).read_window == 64
    assert LayerGroupDesc.unchecked("attention", 64, 0, 0, (0, 2)).model_layers == (0, 2)


@pytest.mark.cpu_only
def test_unchecked_constructors_keep_values_the_constructor_refuses_or_converts():
    """Values the constructor refuses or converts stay as given; a list makes them unhashable."""
    group = dict(kind="attention", kept_window=64, num_sink_blocks=0, pool_group=0, layers=(0,))
    odd = BufferDesc(0, "kv", 0, 0, 0, 8, expansion=3)
    refused = [
        (BufferDesc, dict(layer=-1, role="kv", pool_group=0, pool_index=0, slot_offset=0, size=8)),
        (LayerGroupDesc, dict(group, read_window=128)),
        (LayerGroupDesc, dict(group, model_layers=(0, 1))),
        (RankTopology, dict(tp_size=2, tp_rank=2)),
        (ShardDesc, dict(count=1, index=1)),
        (RoleLayout, dict(bytes_per_head=0)),
        (PageLayout, dict(tokens_per_page=32, pool_slot_bytes={}, layer_groups=(), buffers=(odd,))),
    ]
    for cls, values in refused:
        with pytest.raises(ValueError):
            cls(**values)
        kept = cls.unchecked(**values)
        assert all(getattr(kept, name) == value for name, value in values.items()), cls
        with pytest.raises(ValueError):
            dataclasses.replace(kept)
    listed = LayerGroupDesc.unchecked("attention", None, 0, 0, [0, 2])
    assert listed.layers == [0, 2]
    assert LayerGroupDesc("attention", None, 0, 0, [0, 2]).layers == (0, 2)
    with pytest.raises(TypeError, match="unhashable"):
        hash(listed)


@pytest.mark.cpu_only
def test_page_layout_imports_with_only_the_standard_library_available():
    script = """
import importlib.util
import sys
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("_page_layout_leaf", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.PageLayout(32, {}, (), ()).tokens_per_page == 32
unexpected = {
    name.split(".")[0] for name in set(sys.modules) - before
    if name != spec.name and name.split(".")[0] not in sys.stdlib_module_names
}
assert not unexpected, unexpected
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, page_layout.__file__],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
