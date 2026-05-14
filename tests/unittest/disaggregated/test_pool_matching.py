"""Golden tests for ``PeerRegistrar.get_pool_mapping``.

Locks the current (pre-refactor) behavior of disagg pool matching across the
representative scenarios that the refactor must keep working:

  * single KV pool, full layer overlap (basic MHA)
  * MLA (kv_factor=1, KEY-only pool)
  * KV + block-scale pools coexisting in one LG
  * KV + FLAT pools coexisting in one LG (FLAT has empty buffer_entries)
  * PP partial layer overlap (Step-1 LG match by global_layer_id)
  * Two pools with same role but different layer sets within a peer LG
    (DSv4 virtual-layer scenario, exercises ``best_overlap``)

The refactor (PoolRole -> PoolMatchKey + TransferLayout) must keep these
results stable. If a test needs updating, the change should be deliberate and
called out in the refactor commit message.
"""

import numpy as np

from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _entries(layer_role_pairs, per_buf_size=128):
    """Build a structured numpy array of buffer entries.

    ``layer_role_pairs`` is a list of (local_layer_id, role_label_str). The
    role label only feeds ``pool_role`` — buffer_entries itself only carries
    (lid, offset, size).
    """
    rows = [
        (lid, i * per_buf_size, per_buf_size) for i, (lid, _role) in enumerate(layer_role_pairs)
    ]
    if not rows:
        return np.array([], dtype=BUFFER_ENTRY_DTYPE)
    return np.array(rows, dtype=BUFFER_ENTRY_DTYPE)


def _pool_view(pool_idx, layer_role_pairs, *, pool_role=None, mapper_kind=MapperKind.INDEXED):
    """A PoolView with buffer entries.

    ``pool_role`` defaults to the set of distinct role labels appearing in
    ``layer_role_pairs`` (matching the V1 builder convention).
    """
    if pool_role is None:
        pool_role = frozenset(role for _, role in layer_role_pairs)
    return PoolView(
        pool_idx=pool_idx,
        buffer_entries=_entries(layer_role_pairs),
        pool_role=pool_role,
        mapper_kind=mapper_kind,
    )


def _empty_pool_view(pool_idx, *, pool_role=frozenset({"indexer_k"})):
    """Pool view with no buffer entries — FLAT pool convention."""
    return PoolView(
        pool_idx=pool_idx,
        buffer_entries=np.array([], dtype=BUFFER_ENTRY_DTYPE),
        pool_role=pool_role,
        mapper_kind=MapperKind.FLAT,
    )


def _attn_lg(pool_group_idx, local_global_pairs, pool_views, sliding_window=None):
    return AttentionLayerGroup(
        pool_group_idx=pool_group_idx,
        kv_head_num_per_rank=2,
        sliding_window_size=sliding_window,
        local_layers=[
            LocalLayer(local_layer_id=lid, global_layer_id=gid) for lid, gid in local_global_pairs
        ],
        pool_views=pool_views,
    )


def _page_table(layer_groups, pool_specs=None):
    """Build a KVCachePageTable.

    pool_specs: ``{pool_group_idx: [(slot_bytes, num_slots, base_addr), ...]}``.
    Defaults to one 1024-byte/64-slot pool per group.
    """
    num_pgs = max((lg.pool_group_idx for lg in layer_groups), default=-1) + 1
    if pool_specs is None:
        pool_specs = {pg: [(1024, 64, 0x1000 * (pg + 1))] for pg in range(num_pgs)}

    pool_groups = [
        PhysicalPoolGroup(
            pools=[
                PhysicalPool(base_address=base, slot_bytes=sb, num_slots=ns)
                for sb, ns, base in pool_specs.get(pg, [(1024, 64, 0x1000 * (pg + 1))])
            ]
        )
        for pg in range(num_pgs)
    ]

    return KVCachePageTable(
        tokens_per_block=16,
        layer_groups=layer_groups,
        pool_groups=pool_groups,
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
        server_endpoint="",
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


def _kv_pool_view(pool_idx, local_layer_ids):
    """Convenience: KV pool view (key+value for each given local layer)."""
    return _pool_view(
        pool_idx,
        [(lid, role) for lid in local_layer_ids for role in ("key", "value")],
    )


def _key_only_pool_view(pool_idx, local_layer_ids):
    """Convenience: KEY-only pool view (MLA)."""
    return _pool_view(pool_idx, [(lid, "key") for lid in local_layer_ids])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_kv_only_full_overlap():
    """Single LG, single KV pool on each side. Identity mapping."""
    self_lg = _attn_lg(0, [(0, 10), (1, 11)], [_kv_pool_view(0, [0, 1])])
    peer_lg = _attn_lg(0, [(0, 10), (1, 11)], [_kv_pool_view(0, [0, 1])])

    reg = _registrar(_page_table([self_lg]))
    peer_ri = _rank_info(name="peer", rank=1, page_table=_page_table([peer_lg]))

    mapping = reg.get_pool_mapping(peer_ri)
    assert mapping == {(0, 0): (0, 0)}


def test_mla_kv_only_full_overlap():
    """MLA: KEY-only pool, kv_factor=1."""
    self_lg = _attn_lg(0, [(0, 0), (1, 1)], [_key_only_pool_view(0, [0, 1])])
    peer_lg = _attn_lg(0, [(0, 0), (1, 1)], [_key_only_pool_view(0, [0, 1])])

    reg = _registrar(_page_table([self_lg]), is_mla=True)
    peer_ri = _rank_info(name="peer", rank=1, page_table=_page_table([peer_lg]), is_mla=True)

    mapping = reg.get_pool_mapping(peer_ri)
    assert mapping == {(0, 0): (0, 0)}


def test_kv_and_block_scale_in_same_lg():
    """KV pool (idx=0) + block-scale pool (idx=1) match by role within the LG."""

    def _bq_pool(pool_idx, lids):
        return _pool_view(
            pool_idx,
            [(lid, role) for lid in lids for role in ("key_block_scale", "value_block_scale")],
        )

    self_lg = _attn_lg(0, [(0, 100), (1, 101)], [_kv_pool_view(0, [0, 1]), _bq_pool(1, [0, 1])])
    peer_lg = _attn_lg(0, [(0, 100), (1, 101)], [_kv_pool_view(0, [0, 1]), _bq_pool(1, [0, 1])])

    self_pt = _page_table([self_lg], pool_specs={0: [(1024, 64, 0x1000), (256, 64, 0x2000)]})
    peer_pt = _page_table([peer_lg], pool_specs={0: [(1024, 64, 0x3000), (256, 64, 0x4000)]})

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    mapping = reg.get_pool_mapping(peer_ri)
    assert mapping == {(0, 0): (0, 0), (0, 1): (0, 1)}


def test_kv_and_indexer_in_same_lg():
    """KV pool + FLAT pool. FLAT has empty buffer_entries; matches by role."""
    self_lg = _attn_lg(
        0,
        [(0, 0), (1, 1)],
        [_kv_pool_view(0, [0, 1]), _empty_pool_view(1)],
    )
    peer_lg = _attn_lg(
        0,
        [(0, 0), (1, 1)],
        [_kv_pool_view(0, [0, 1]), _empty_pool_view(1)],
    )

    self_pt = _page_table([self_lg], pool_specs={0: [(1024, 64, 0x1000), (512, 64, 0x2000)]})
    peer_pt = _page_table([peer_lg], pool_specs={0: [(1024, 64, 0x3000), (512, 64, 0x4000)]})

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    mapping = reg.get_pool_mapping(peer_ri)
    assert mapping == {(0, 0): (0, 0), (0, 1): (0, 1)}


def test_pp_partial_layer_overlap():
    """Self covers global layers {10,11}, peer covers {11,12}. Match via overlap on layer 11."""
    self_lg = _attn_lg(0, [(0, 10), (1, 11)], [_kv_pool_view(0, [0, 1])])
    peer_lg = _attn_lg(0, [(0, 11), (1, 12)], [_kv_pool_view(0, [0, 1])])

    reg = _registrar(_page_table([self_lg]), layer_num_per_pp=[1, 1])
    peer_ri = _rank_info(
        name="peer", rank=1, page_table=_page_table([peer_lg]), layer_num_per_pp=[1, 1]
    )

    mapping = reg.get_pool_mapping(peer_ri)
    assert mapping == {(0, 0): (0, 0)}


def test_two_pools_distinct_roles_in_same_lg():
    """Two pools with distinct pool_role in one LG match by role, not by layer overlap.

    Within one LG, pool_role normally identifies a pool uniquely (builder
    invariant). Even when the pools' layer sets are different sizes / not
    equal, the matching is purely role-based: each self pool finds the peer
    pool with the same pool_role.
    """
    self_kv = _kv_pool_view(0, [0, 1])
    self_indexer = _empty_pool_view(1)
    self_lg = _attn_lg(0, [(0, 10), (1, 11)], [self_kv, self_indexer])

    peer_kv = _kv_pool_view(0, [0, 1])
    peer_indexer = _empty_pool_view(1)
    peer_lg = _attn_lg(0, [(0, 10), (1, 11)], [peer_kv, peer_indexer])

    self_pt = _page_table([self_lg], pool_specs={0: [(1024, 64, 0x1000), (512, 64, 0x2000)]})
    peer_pt = _page_table([peer_lg], pool_specs={0: [(1024, 64, 0x3000), (512, 64, 0x4000)]})

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt)

    mapping = reg.get_pool_mapping(peer_ri)
    assert mapping == {(0, 0): (0, 0), (0, 1): (0, 1)}


def test_same_role_pools_disambiguated_by_layer_overlap():
    """Defensive: same pool_role + different layer sets is resolved by overlap.

    If a peer LG holds two pools with the same pool_role but different layer
    sets, matching picks the one whose layer set actually overlaps self. A
    peer pool with zero overlap is never matched even if the role matches.
    """
    self_lg = _attn_lg(0, [(0, 10), (1, 11)], [_kv_pool_view(0, [0, 1])])
    peer_lg = _attn_lg(
        0,
        [(0, 10), (1, 11), (2, 12)],
        [
            # First peer pool covers layers {12} only — same role, zero overlap with self.
            _kv_pool_view(0, [2]),
            # Second peer pool covers layers {10, 11} — same role, full overlap.
            _kv_pool_view(1, [0, 1]),
        ],
    )

    self_pt = _page_table([self_lg])
    peer_pt = _page_table([peer_lg], pool_specs={0: [(1024, 64, 0x3000), (1024, 64, 0x4000)]})

    reg = _registrar(self_pt)
    peer_ri = _rank_info(name="peer", rank=1, page_table=peer_pt, layer_num_per_pp=[3])

    mapping = reg.get_pool_mapping(peer_ri)
    assert mapping == {(0, 0): (0, 1)}
