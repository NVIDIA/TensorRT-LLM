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

"""Golden tests for region matching (``match_regions``) and ``PeerRegistrar.transfer_groups``.

Locks the behavior of disagg region matching across the representative scenarios:

  * single KV pool, full layer overlap (basic MHA)
  * MLA (kv_factor=1, KEY-only regions)
  * KV + block-scale pools coexisting in one layer group
  * KV + REPLICATED indexer pools coexisting in one layer group
  * PP partial layer overlap (regions match by global layer id)
  * peer splitting the same layers over two layer groups, or over two pools of one layer group

Regions match by ``(spec, layer, role)``; matched regions must declare the same layout.
"""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.cache_specs import (
    BufferDecl,
    LayerGroup,
    PageTable,
    Pool,
    Region,
    match_regions,
    validate_peer_compatible,
)
from tensorrt_llm._torch.disaggregation.cache_specs.correspondence import check_page_table
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import AttentionInfo, RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1

pytestmark = pytest.mark.cpu_only

BUF = 128

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _hnd(heads=2):
    return (BufferDecl(BUF, (("head", heads),), "head"),)


def _nhd(heads=2):
    return (BufferDecl(BUF, (("token", 16), ("head", heads)), "head"),)


def _replicated():
    return (BufferDecl(BUF),)


def _regions(pool, layer_roles, buffers=None):
    """One region per ``(global_layer_id, role)``, packed back to back in ``pool``."""
    buffers = buffers if buffers is not None else _hnd()
    return [
        Region(pool, layer, role, i * BUF, buffers, local_layer=layer)
        for i, (layer, role) in enumerate(layer_roles)
    ]


def _kv(pool, layers, buffers=None):
    """Key + value regions of each global layer."""
    return _regions(pool, [(g, r) for g in layers for r in ("key", "value")], buffers)


def _key_only(pool, layers):
    """KEY-only regions (MLA)."""
    return _regions(pool, [(g, "key") for g in layers], _hnd(heads=1))


def _indexer(pool, layers, role="indexer_k"):
    """Replicated indexer regions: one per layer."""
    return _regions(pool, [(g, role) for g in layers], _replicated())


def _layer_group(regions):
    return LayerGroup(
        spec="attention",
        kind=CacheKind.PAGED,
        regions=regions,
        params={"window": None},
    )


def _page_table(layer_groups, pool_specs=None):
    """A PageTable over ``pool_specs`` = ``[(slot_bytes, num_slots, base_addr), ...]``.

    Defaults to one 1024-byte/64-slot pool per pool index the regions name.
    """
    if pool_specs is None:
        num_pools = 1 + max(r.pool for s in layer_groups for r in s.regions)
        pool_specs = [(1024, 64, 0x1000 * (p + 1)) for p in range(num_pools)]
    return PageTable(
        pools=[Pool(base_address=base, slot_bytes=sb, num_slots=ns) for sb, ns, base in pool_specs],
        layer_groups=layer_groups,
    )


def _rank_info(
    name="self",
    rank=0,
    layer_num_per_pp=None,
    page_table=None,
    is_mla=False,
    kv_heads=2,
):
    if layer_num_per_pp is None:
        layer_num_per_pp = [2]
    return RankInfo(
        instance_name=name,
        instance_rank=rank,
        tp_size=1,
        tp_rank=0,
        pp_size=len(layer_num_per_pp),
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        device_id=0,
        layer_num_per_pp=layer_num_per_pp,
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=kv_heads,
            tokens_per_block=16,
            dims_per_head=8,
            element_bytes=2,
            enable_attention_dp=False,
            is_mla=is_mla,
        ),
        aux_meta=None,
        page_table=page_table,
        sender_endpoints=[],
    )


def _registrar(self_pt, **kwargs):
    self_ri = _rank_info(name="self", page_table=self_pt, **kwargs)
    return PeerRegistrar(self_ri, KVRegionExtractorV1(self_pt))


def _identity(num_regions, layer_group=0):
    return {(layer_group, i): (layer_group, i) for i in range(num_regions)}


def _group_shape(groups):
    return [
        (g.self_layer_group, g.peer_layer_group, g.self_pool, g.peer_pool, g.regions)
        for g in groups
    ]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_kv_only_full_overlap():
    """Single layer group, single KV pool on each side. Identity mapping."""
    self_pt = _page_table([_layer_group(_kv(0, [10, 11]))])
    peer_pt = _page_table([_layer_group(_kv(0, [10, 11]))])

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    assert match_regions(self_pt, peer_pt) == _identity(4)
    groups = reg.transfer_groups(peer_ri)
    assert _group_shape(groups) == [(0, 0, 0, 0, (0, 1, 2, 3))]
    # Same layout on both sides: the whole slot moves as one run.
    assert groups[0].plan.runs == [(0, 0, 4 * BUF)]


def test_mla_kv_only_full_overlap():
    """MLA: KEY-only regions, kv_factor=1."""
    self_pt = _page_table([_layer_group(_key_only(0, [0, 1]))])
    peer_pt = _page_table([_layer_group(_key_only(0, [0, 1]))])

    reg = _registrar(self_pt, is_mla=True, kv_heads=1)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt, is_mla=True, kv_heads=1)

    assert match_regions(self_pt, peer_pt) == _identity(2)
    assert _group_shape(reg.transfer_groups(peer_ri)) == [(0, 0, 0, 0, (0, 1))]


def test_kv_and_block_scale_in_same_layer_group():
    """KV pool (idx=0) + block-scale pool (idx=1) match by role within the layer group."""

    def _layer_group_with_scales():
        scales = _regions(
            1, [(g, r) for g in (100, 101) for r in ("key_block_scale", "value_block_scale")]
        )
        return _layer_group(_kv(0, [100, 101]) + scales)

    self_pt = _page_table([_layer_group_with_scales()], [(1024, 64, 0x1000), (512, 64, 0x2000)])
    peer_pt = _page_table([_layer_group_with_scales()], [(1024, 64, 0x3000), (512, 64, 0x4000)])

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    assert match_regions(self_pt, peer_pt) == _identity(8)
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, 0, 0, (0, 1, 2, 3)),
        (0, 0, 1, 1, (4, 5, 6, 7)),
    ]


def test_kv_and_indexer_in_same_layer_group():
    """KV pool + replicated indexer pool match independently by role."""
    self_pt = _page_table(
        [_layer_group(_kv(0, [0, 1]) + _indexer(1, [0, 1]))],
        [(1024, 64, 0x1000), (512, 64, 0x2000)],
    )
    peer_pt = _page_table(
        [_layer_group(_kv(0, [0, 1]) + _indexer(1, [0, 1]))],
        [(1024, 64, 0x3000), (512, 64, 0x4000)],
    )

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    assert match_regions(self_pt, peer_pt) == _identity(6)
    groups = reg.transfer_groups(peer_ri)
    assert _group_shape(groups) == [(0, 0, 0, 0, (0, 1, 2, 3)), (0, 0, 1, 1, (4, 5))]
    assert all(reg.owns(g, peer_ri) for g in groups)


def test_minimax_split_kv_and_replicated_index_pools_match():
    """MiniMax M3 Triton shape: NHD KV pool + REPLICATED index-key pool.

    Both sides present the same (layer, role) regions with the same layouts, so the
    NHD and replicated regions pair 1:1 with their declarations intact.
    """

    def _pt(base):
        return _page_table(
            [_layer_group(_kv(0, [0, 1], _nhd()) + _indexer(1, [0, 1], role="index_key"))],
            [(512, 64, base), (256, 64, base + 0x1000)],
        )

    self_pt, peer_pt = _pt(0x1000), _pt(0x3000)
    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    mapping = match_regions(self_pt, peer_pt)
    assert mapping == _identity(6)
    for (gi, ri), (pgi, pri) in mapping.items():
        assert (
            self_pt.layer_groups[gi].regions[ri].buffers
            == peer_pt.layer_groups[pgi].regions[pri].buffers
        )
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, 0, 0, (0, 1, 2, 3)),
        (0, 0, 1, 1, (4, 5)),
    ]


def test_mismatched_region_layouts_raise():
    """Layout disagreement on one (layer, role) must fail loudly.

    Same region but a different declaration on the two sides is a version or
    configuration inconsistency.
    """
    self_pt = _page_table([_layer_group(_regions(0, [(0, "index_key")], _replicated()))])
    peer_pt = _page_table([_layer_group(_regions(0, [(0, "index_key")], _nhd()))])

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    with pytest.raises(ValueError, match="layout differs"):
        reg.transfer_groups(peer_ri)
    with pytest.raises(ValueError, match="layout differs"):
        validate_peer_compatible(reg.self_rank_info, peer_ri, self_pt, peer_pt)


def test_pp_partial_layer_overlap():
    """Self covers global layers {10,11}, peer covers {11,12}. Only layer 11 matches."""
    self_pt = _page_table([_layer_group(_kv(0, [10, 11]))])
    peer_pt = _page_table([_layer_group(_kv(0, [11, 12]))])

    reg = _registrar(self_pt, layer_num_per_pp=[1, 1])
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt, layer_num_per_pp=[1, 1])

    # Self layer 11 (regions 2, 3) lands on the peer's first layer (regions 0, 1).
    assert match_regions(self_pt, peer_pt) == {(0, 2): (0, 0), (0, 3): (0, 1)}
    groups = reg.transfer_groups(peer_ri)
    assert _group_shape(groups) == [(0, 0, 0, 0, (2, 3))]
    assert groups[0].plan.runs == [(2 * BUF, 0, 2 * BUF)]


def test_regions_spanning_multiple_peer_layer_groups_form_one_transfer_group_each():
    """A local layer group whose layers the peer holds in two layer groups maps into both.

    Each peer layer group indexes its own per-request units, so the regions split into one
    transfer group per peer layer group instead of being merged.
    """
    self_pt = _page_table([_layer_group(_kv(0, [10, 11]))])
    peer_pt = _page_table([_layer_group(_kv(0, [10])), _layer_group(_kv(0, [11]))])

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    assert match_regions(self_pt, peer_pt) == {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (1, 0),
        (0, 3): (1, 1),
    }
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, 0, 0, (0, 1)),
        (0, 1, 0, 0, (2, 3)),
    ]


def test_buffer_count_mismatch_raises():
    """Matched regions must declare the same number of buffers on both sides."""
    sections = (BufferDecl(64), BufferDecl(64))
    self_pt = _page_table([_layer_group(_regions(0, [(0, "key")], _replicated()))])
    peer_pt = _page_table([_layer_group(_regions(0, [(0, "key")], sections))])

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    with pytest.raises(ValueError, match="buffer count"):
        validate_peer_compatible(reg.self_rank_info, peer_ri, self_pt, peer_pt)
    with pytest.raises(ValueError, match="buffer count"):
        reg.transfer_groups(peer_ri)


def test_duplicate_region_across_spaces_raises():
    """A spec holds each (layer, role) once.

    The same global layer id in two layer groups of one spec would silently corrupt matching.
    """
    dup_pt = _page_table([_layer_group(_kv(0, [10])), _layer_group(_kv(0, [10]))])
    ok_pt = _page_table([_layer_group(_kv(0, [10]))])

    # A peer table is checked when matching against it; a local one when it is built.
    with pytest.raises(ValueError, match="appears in layer groups 0 and 1"):
        match_regions(ok_pt, dup_pt)
    with pytest.raises(ValueError, match="appears in layer groups 0 and 1"):
        check_page_table(dup_pt)


def test_two_pools_distinct_roles_in_same_layer_group():
    """Regions of distinct roles in two pools of one layer group match by role, not by pool.

    The peer stores the two roles in the opposite pool order; each local region still finds
    the peer region with the same (layer, role).
    """
    self_pt = _page_table(
        [_layer_group(_kv(0, [10, 11]) + _indexer(1, [10, 11]))],
        [(1024, 64, 0x1000), (512, 64, 0x2000)],
    )
    peer_pt = _page_table(
        [_layer_group(_indexer(0, [10, 11]) + _kv(1, [10, 11]))],
        [(512, 64, 0x3000), (1024, 64, 0x4000)],
    )

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    assert match_regions(self_pt, peer_pt) == {
        (0, 0): (0, 2),
        (0, 1): (0, 3),
        (0, 2): (0, 4),
        (0, 3): (0, 5),
        (0, 4): (0, 0),
        (0, 5): (0, 1),
    }
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, 0, 1, (0, 1, 2, 3)),
        (0, 0, 1, 0, (4, 5)),
    ]


def test_same_role_regions_matched_by_layer():
    """Same role in two peer pools with different layers: each region finds its own layer.

    A peer region whose layer self does not hold is never matched even if the role matches.
    """
    self_pt = _page_table([_layer_group(_kv(0, [10, 11]))])
    # Peer pool 0 holds only layer 12; pool 1 holds layers 10 and 11.
    peer_pt = _page_table(
        [_layer_group(_kv(0, [12]) + _kv(1, [10, 11]))], [(1024, 64, 0x3000), (1024, 64, 0x4000)]
    )

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt, layer_num_per_pp=[3])

    assert match_regions(self_pt, peer_pt) == {
        (0, 0): (0, 2),
        (0, 1): (0, 3),
        (0, 2): (0, 4),
        (0, 3): (0, 5),
    }
    assert _group_shape(reg.transfer_groups(peer_ri)) == [(0, 0, 0, 1, (0, 1, 2, 3))]


@pytest.mark.parametrize(
    ("self_buffers", "peer_buffers"),
    [
        pytest.param(_replicated(), _hnd(), id="replicated-vs-head"),
        pytest.param(_hnd(), _replicated(), id="head-vs-replicated"),
    ],
)
def test_mixed_layouts_are_rejected(self_buffers, peer_buffers):
    """Region layouts must be the same on both peers."""
    self_pt = _page_table(
        [_layer_group(_regions(0, [(10, "indexer_k"), (11, "indexer_k")], self_buffers))]
    )
    peer_pt = _page_table(
        [_layer_group(_regions(0, [(10, "indexer_k"), (11, "indexer_k")], peer_buffers))]
    )
    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    with pytest.raises(ValueError, match="layout differs"):
        validate_peer_compatible(reg.self_rank_info, peer_ri, self_pt, peer_pt)
    with pytest.raises(ValueError, match="layout differs"):
        reg.transfer_groups(peer_ri)


def test_region_offsets_drive_extracted_pointers():
    """Matched regions' runs are relative to a unit's slot base from ``extract``."""
    self_pt = _page_table([_layer_group(_kv(0, [10, 11]))], [(1024, 64, 0x1000)])
    peer_pt = _page_table([_layer_group(_kv(0, [11]))], [(512, 64, 0x9000)])

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)
    (group,) = reg.transfer_groups(peer_ri)

    ids = np.array([3, -1, 5], dtype=np.int64)
    src = KVRegionExtractorV1(self_pt).extract(ids, group.self_pool)
    dst = KVRegionExtractorV1(peer_pt).extract(ids, group.peer_pool)
    (pair,) = group.plan.map(src, dst)
    np.testing.assert_array_equal(
        pair.src.memory.ptrs, [0x1000 + 3 * 1024 + 2 * BUF, 0x1000 + 5 * 1024 + 2 * BUF]
    )
    np.testing.assert_array_equal(pair.dst.memory.ptrs, [0x9000 + 3 * 512, 0x9000 + 5 * 512])
    assert pair.src.memory.bytes_per_region == 2 * BUF
