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

"""Golden tests for buffer matching (``match_buffers``) and ``PeerRegistrar.transfer_groups``.

Locks the behavior of disagg buffer matching across the representative scenarios:

  * single KV pool, full layer overlap (basic MHA)
  * MLA (kv_factor=1, KEY-only buffers)
  * KV + block-scale pools coexisting in one layer group
  * KV + REPLICATED indexer pools coexisting in one layer group
  * PP partial layer overlap (buffers match by global layer id)
  * peer splitting the same layers over two layer groups, or over two pools of one layer group

Buffers match by ``(layer, role)``; matched buffers must declare the same segment layout.
"""

import numpy as np
import pytest
from pages_test_utils import Segment, buf, group, head_major, pages, pool, replicated, token_major

from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.page_mapping import match_buffers, validate_peer_compatible

pytestmark = pytest.mark.cpu_only

BUF = 128

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _hnd(heads=2):
    return head_major(BUF, heads)


def _nhd(heads=2):
    return token_major(BUF, 16, heads)


def _replicated():
    return replicated(BUF)


def _buffers(pool_index, layer_roles, segments=None):
    """One buffer per ``(global_layer_id, role)``, back to back in flat pool ``pool_index``."""
    segments = segments if segments is not None else _hnd()
    size = sum(s.nbytes for s in segments)
    return [
        buf(pool_index, layer, role, i * size, segments)
        for i, (layer, role) in enumerate(layer_roles)
    ]


def _kv(pool_index, layers, segments=None):
    """Key + value buffers of each global layer."""
    return _buffers(pool_index, [(g, r) for g in layers for r in ("key", "value")], segments)


def _key_only(pool_index, layers):
    """KEY-only buffers (MLA)."""
    return _buffers(pool_index, [(g, "key") for g in layers], _hnd(heads=1))


def _indexer(pool_index, layers, role="indexer_k"):
    """Replicated indexer buffers: one per layer."""
    return _buffers(pool_index, [(g, role) for g in layers], _replicated())


def _layer_group(buffers):
    return group("attention", buffers)


def _pages(layer_groups, pool_specs=None):
    """Pages over ``pool_specs`` = ``[(slot_bytes, num_slots, base_addr), ...]`` (flat pools).

    Defaults to one 1024-byte/64-slot pool per flat pool index the buffers name.
    """
    if pool_specs is None:
        num_pools = 1 + max(b.pool for g in layer_groups for b in g.buffers)
        pool_specs = [(1024, 64, 0x1000 * (p + 1)) for p in range(num_pools)]
    return pages([pool(base, sb, ns) for sb, ns, base in pool_specs], layer_groups)


def _rank_info(
    name="self",
    rank=0,
    layer_num_per_pp=None,
    pages=None,
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
        aux_meta=None,
        pages=pages,
        sender_endpoints=[],
    )


def _registrar(self_pages, **kwargs):
    self_ri = _rank_info(name="self", pages=self_pages, **kwargs)
    return PeerRegistrar(self_ri)


def _identity(num_buffers, layer_group=0):
    return {(layer_group, i): (layer_group, i) for i in range(num_buffers)}


def _group_shape(groups):
    return [
        (g.self_layer_group, g.peer_layer_group, g.self_pool, g.peer_pool, g.buffers)
        for g in groups
    ]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_kv_only_full_overlap():
    """Single layer group, single KV pool on each side. Identity mapping."""
    self_pages = _pages([_layer_group(_kv(0, [10, 11]))])
    peer_pages = _pages([_layer_group(_kv(0, [10, 11]))])

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    assert match_buffers(self_pages, peer_pages) == _identity(4)
    groups = reg.transfer_groups(peer_ri)
    assert _group_shape(groups) == [(0, 0, (0, 0), (0, 0), (0, 1, 2, 3))]
    # Same layout on both sides: the whole slot moves as one run.
    assert groups[0].plan.runs == [(0, 0, 4 * BUF)]


def test_mla_kv_only_full_overlap():
    """MLA: KEY-only buffers, kv_factor=1."""
    self_pages = _pages([_layer_group(_key_only(0, [0, 1]))])
    peer_pages = _pages([_layer_group(_key_only(0, [0, 1]))])

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    assert match_buffers(self_pages, peer_pages) == _identity(2)
    assert _group_shape(reg.transfer_groups(peer_ri)) == [(0, 0, (0, 0), (0, 0), (0, 1))]


def test_kv_and_block_scale_in_same_layer_group():
    """KV pool (idx=0) + block-scale pool (idx=1) match by role within the layer group."""

    def _layer_group_with_scales():
        scales = _buffers(
            1, [(g, r) for g in (100, 101) for r in ("key_block_scale", "value_block_scale")]
        )
        return _layer_group(_kv(0, [100, 101]) + scales)

    self_pages = _pages([_layer_group_with_scales()], [(1024, 64, 0x1000), (512, 64, 0x2000)])
    peer_pages = _pages([_layer_group_with_scales()], [(1024, 64, 0x3000), (512, 64, 0x4000)])

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    assert match_buffers(self_pages, peer_pages) == _identity(8)
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, (0, 0), (0, 0), (0, 1, 2, 3)),
        (0, 0, (0, 1), (0, 1), (4, 5, 6, 7)),
    ]


def test_kv_and_indexer_in_same_layer_group():
    """KV pool + replicated indexer pool match independently by role."""
    self_pages = _pages(
        [_layer_group(_kv(0, [0, 1]) + _indexer(1, [0, 1]))],
        [(1024, 64, 0x1000), (512, 64, 0x2000)],
    )
    peer_pages = _pages(
        [_layer_group(_kv(0, [0, 1]) + _indexer(1, [0, 1]))],
        [(1024, 64, 0x3000), (512, 64, 0x4000)],
    )

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    assert match_buffers(self_pages, peer_pages) == _identity(6)
    groups = reg.transfer_groups(peer_ri)
    assert _group_shape(groups) == [
        (0, 0, (0, 0), (0, 0), (0, 1, 2, 3)),
        (0, 0, (0, 1), (0, 1), (4, 5)),
    ]
    assert all(reg.owns(g, peer_ri) for g in groups)


def test_minimax_split_kv_and_replicated_index_pools_match():
    """MiniMax M3 Triton shape: NHD KV pool + REPLICATED index-key pool.

    Both sides present the same (layer, role) buffers with the same layouts, so the
    NHD and replicated buffers pair 1:1 with their segments intact.
    """

    def _make(base):
        return _pages(
            [_layer_group(_kv(0, [0, 1], _nhd()) + _indexer(1, [0, 1], role="index_key"))],
            [(512, 64, base), (256, 64, base + 0x1000)],
        )

    self_pages, peer_pages = _make(0x1000), _make(0x3000)
    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    mapping = match_buffers(self_pages, peer_pages)
    assert mapping == _identity(6)
    for (gi, ri), (pgi, pri) in mapping.items():
        assert self_pages.buffers(gi)[ri].segments == peer_pages.buffers(pgi)[pri].segments
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, (0, 0), (0, 0), (0, 1, 2, 3)),
        (0, 0, (0, 1), (0, 1), (4, 5)),
    ]


def test_mismatched_buffer_layouts_raise():
    """Layout disagreement on one (layer, role) must fail loudly.

    Same buffer but a different layout on the two sides is a version or
    configuration inconsistency.
    """
    self_pages = _pages([_layer_group(_buffers(0, [(0, "index_key")], _replicated()))])
    peer_pages = _pages([_layer_group(_buffers(0, [(0, "index_key")], _nhd()))])

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    with pytest.raises(ValueError, match="layout differs"):
        reg.transfer_groups(peer_ri)
    with pytest.raises(ValueError, match="layout differs"):
        validate_peer_compatible(reg.self_rank_info, peer_ri, self_pages, peer_pages)


def test_pp_partial_layer_overlap():
    """Self covers global layers {10,11}, peer covers {11,12}. Only layer 11 matches."""
    self_pages = _pages([_layer_group(_kv(0, [10, 11]))])
    peer_pages = _pages([_layer_group(_kv(0, [11, 12]))])

    reg = _registrar(self_pages, layer_num_per_pp=[1, 1])
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages, layer_num_per_pp=[1, 1])

    # Self layer 11 (buffers 2, 3) lands on the peer's first layer (buffers 0, 1).
    assert match_buffers(self_pages, peer_pages) == {(0, 2): (0, 0), (0, 3): (0, 1)}
    groups = reg.transfer_groups(peer_ri)
    assert _group_shape(groups) == [(0, 0, (0, 0), (0, 0), (2, 3))]
    assert groups[0].plan.runs == [(2 * BUF, 0, 2 * BUF)]


def test_buffers_spanning_multiple_peer_layer_groups_form_one_transfer_group_each():
    """A local layer group whose layers the peer holds in two layer groups maps into both.

    Each peer layer group indexes its own per-request units, so the buffers split into one
    transfer group per peer layer group instead of being merged.
    """
    self_pages = _pages([_layer_group(_kv(0, [10, 11]))])
    peer_pages = _pages([_layer_group(_kv(0, [10])), _layer_group(_kv(0, [11]))])

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    assert match_buffers(self_pages, peer_pages) == {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (1, 0),
        (0, 3): (1, 1),
    }
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, (0, 0), (0, 0), (0, 1)),
        (0, 1, (0, 0), (0, 0), (2, 3)),
    ]


def test_segment_count_mismatch_raises():
    """Matched buffers must declare the same number of segments on both sides."""
    sections = (Segment(64), Segment(64))
    self_pages = _pages([_layer_group(_buffers(0, [(0, "key")], _replicated()))])
    peer_pages = _pages([_layer_group(_buffers(0, [(0, "key")], sections))])

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    with pytest.raises(ValueError, match="segment count"):
        validate_peer_compatible(reg.self_rank_info, peer_ri, self_pages, peer_pages)
    with pytest.raises(ValueError, match="segment count"):
        reg.transfer_groups(peer_ri)


def test_duplicate_buffer_across_layer_groups_raises():
    """A rank holds each (layer, role) once.

    The same global layer id in two layer groups would silently corrupt matching; the peer's
    pages are checked when matching against them.
    """
    dup_pages = _pages([_layer_group(_kv(0, [10])), _layer_group(_kv(1, [10]))])
    ok_pages = _pages([_layer_group(_kv(0, [10]))])

    with pytest.raises(ValueError, match="appears in layer groups 0 and 1"):
        match_buffers(ok_pages, dup_pages)


def test_two_pools_distinct_roles_in_same_layer_group():
    """Buffers of distinct roles in two pools of one layer group match by role, not by pool.

    The peer stores the two roles in the opposite pool order; each local buffer still finds
    the peer buffer with the same (layer, role).
    """
    self_pages = _pages(
        [_layer_group(_kv(0, [10, 11]) + _indexer(1, [10, 11]))],
        [(1024, 64, 0x1000), (512, 64, 0x2000)],
    )
    peer_pages = _pages(
        [_layer_group(_indexer(0, [10, 11]) + _kv(1, [10, 11]))],
        [(512, 64, 0x3000), (1024, 64, 0x4000)],
    )

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    assert match_buffers(self_pages, peer_pages) == {
        (0, 0): (0, 2),
        (0, 1): (0, 3),
        (0, 2): (0, 4),
        (0, 3): (0, 5),
        (0, 4): (0, 0),
        (0, 5): (0, 1),
    }
    assert _group_shape(reg.transfer_groups(peer_ri)) == [
        (0, 0, (0, 0), (0, 1), (0, 1, 2, 3)),
        (0, 0, (0, 1), (0, 0), (4, 5)),
    ]


def test_same_role_buffers_matched_by_layer():
    """Same role in two peer pools with different layers: each buffer finds its own layer.

    A peer buffer whose layer self does not hold is never matched even if the role matches.
    """
    self_pages = _pages([_layer_group(_kv(0, [10, 11]))])
    # Peer pool 0 holds only layer 12; pool 1 holds layers 10 and 11.
    peer_pages = _pages(
        [_layer_group(_kv(0, [12]) + _kv(1, [10, 11]))], [(1024, 64, 0x3000), (1024, 64, 0x4000)]
    )

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages, layer_num_per_pp=[3])

    assert match_buffers(self_pages, peer_pages) == {
        (0, 0): (0, 2),
        (0, 1): (0, 3),
        (0, 2): (0, 4),
        (0, 3): (0, 5),
    }
    assert _group_shape(reg.transfer_groups(peer_ri)) == [(0, 0, (0, 0), (0, 1), (0, 1, 2, 3))]


@pytest.mark.parametrize(
    ("self_segments", "peer_segments"),
    [
        pytest.param(_replicated(), _hnd(), id="replicated-vs-head"),
        pytest.param(_hnd(), _replicated(), id="head-vs-replicated"),
    ],
)
def test_mixed_layouts_are_rejected(self_segments, peer_segments):
    """Buffer layouts must be the same on both peers."""
    self_pages = _pages(
        [_layer_group(_buffers(0, [(10, "indexer_k"), (11, "indexer_k")], self_segments))]
    )
    peer_pages = _pages(
        [_layer_group(_buffers(0, [(10, "indexer_k"), (11, "indexer_k")], peer_segments))]
    )
    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)

    with pytest.raises(ValueError, match="layout differs"):
        validate_peer_compatible(reg.self_rank_info, peer_ri, self_pages, peer_pages)
    with pytest.raises(ValueError, match="layout differs"):
        reg.transfer_groups(peer_ri)


def test_buffer_offsets_drive_extracted_pointers():
    """Matched buffers' runs are relative to a unit's slot base from ``RankPages.slots``."""
    self_pages = _pages([_layer_group(_kv(0, [10, 11]))], [(1024, 64, 0x1000)])
    peer_pages = _pages([_layer_group(_kv(0, [11]))], [(512, 64, 0x9000)])

    reg = _registrar(self_pages)
    peer_ri = _rank_info(name="peer", rank=1, pages=peer_pages)
    (group,) = reg.transfer_groups(peer_ri)

    ids = np.array([3, -1, 5], dtype=np.int64)
    src = self_pages.slots(group.self_pool, ids)
    dst = peer_pages.slots(group.peer_pool, ids)
    (pair,) = group.plan.map(src, dst)
    np.testing.assert_array_equal(
        pair.src.memory.ptrs, [0x1000 + 3 * 1024 + 2 * BUF, 0x1000 + 5 * 1024 + 2 * BUF]
    )
    np.testing.assert_array_equal(pair.dst.memory.ptrs, [0x9000 + 3 * 512, 0x9000 + 5 * 512])
    assert pair.src.memory.bytes_per_region == 2 * BUF
