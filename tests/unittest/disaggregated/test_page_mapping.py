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
"""page_mapping: a rank's pages, the unit rules, and how two ranks' pages correspond.

The resharding matrix checks every receiving rank byte by byte against an independent model of TP /
GQA / helix / attention-DP / PP sharding (``check_resharding``), for the built-in layouts and for a
cache with a new layout that only its manager's hooks describe.
"""

import dataclasses
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from pages_test_utils import (
    Segment,
    buf,
    group,
    head_major,
    pages,
    pool,
    replicated,
    sectioned,
    token_major,
)

from tensorrt_llm._torch.disaggregation import page_mapping as pm
from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.page_mapping import (
    ChunkContext,
    LocalUnitsContext,
    PairContext,
    RankPages,
)
from tensorrt_llm._torch.disaggregation.page_mapping import topology as topo
from tensorrt_llm._torch.disaggregation.page_mapping import units as pm_units
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_builder import (
    DevicePool,
    LayerGroupTraits,
    build_page_layout,
    default_layer_group_traits,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    BufferDesc,
    RankTopology,
    grid_position,
)

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# Oracle: an independent model of sharding the transfer plan is checked against
#
# On every receiving rank, every byte of every buffer the senders also hold must be written exactly
# once, by the elected sender, from the source byte that holds the same
# ``(layer, role, segment, row, global unit, offset in unit)``. A rank's local unit ``u`` of a
# split axis is global unit ``rank * G // grid + u`` (``G`` the model's true extent: heads,
# channels, ...; ranks beyond ``G`` duplicate units, as GQA heads do).
# ---------------------------------------------------------------------------


GlobalUnits = Callable[[BufferDesc, int, Segment], int]
"""The model's true extent of segment ``index`` of a buffer along its split axis."""

ByteId = Tuple[int, str, int, int, int, int]
"""``(global layer id, role, segment index, row, global unit, offset in unit)``."""

Address = Tuple[Tuple[int, int], int]
"""``((pool group, pool index), offset within a unit's slot)``."""


def check_pages_round_trip(rank_pages: RankPages) -> RankPages:
    restored = RankPages.from_dict(rank_pages.to_dict())
    if restored.layout != rank_pages.layout or restored.pools != rank_pages.pools:
        raise AssertionError("pages do not survive to_dict/from_dict")
    return restored


def _byte_ids(
    rank_pages: RankPages, g: int, ri: RankInfo, global_units: GlobalUnits
) -> Dict[Address, ByteId]:
    """Slot address -> identity of every byte layer group ``g`` transfers on one rank."""
    grid = pm.shard_grid(rank_pages.group(g).grid, ri)
    ids: Dict[Address, ByteId] = {}
    for buffer in rank_pages.buffers(g):
        pool_key = (buffer.pool_group, buffer.pool_index)
        offset = buffer.slot_offset
        for index, segment in enumerate(buffer.segments):
            if segment.split_axis is None:
                for b in range(segment.nbytes):
                    ids[(pool_key, offset + b)] = (buffer.layer, buffer.role, index, 0, 0, b)
            else:
                unit_bytes = segment.nbytes // (segment.rows * segment.split_extent)
                first = (grid.rank * global_units(buffer, index, segment)) // grid.size
                for b in range(segment.nbytes):
                    row, rest = divmod(b, segment.split_extent * unit_bytes)
                    unit, within = divmod(rest, unit_bytes)
                    ids[(pool_key, offset + b)] = (
                        buffer.layer,
                        buffer.role,
                        index,
                        row,
                        first + unit,
                        within,
                    )
            offset += segment.nbytes
    return ids


def check_resharding(
    senders: Sequence[RankInfo],
    receivers: Sequence[RankInfo],
    global_units: GlobalUnits,
    *,
    sender_dp_rank: int = 0,
) -> int:
    """Check every receiver gets each byte exactly once from the right source byte.

    ``senders`` / ``receivers`` are ``RankInfo`` objects with ``pages`` set, one per rank of each
    instance. Senders are the ranks a receiver addresses (``rank_overlap``), as at runtime, and
    each of them must address the receiver in turn. Returns the number of bytes checked.
    """
    by_rank = {ri.instance_rank: ri for ri in senders}
    checked = 0
    for dst_ri in receivers:
        dst_pages = dst_ri.pages
        writes: Counter = Counter()
        expected: Dict[Tuple[int, Address], ByteId] = {}
        overlap = pm.rank_overlap(dst_ri, senders[0], sender_dp_rank)
        for flat in overlap.ranks:
            src_ri = by_rank[flat]
            src_pages = src_ri.pages
            # Pairing is symmetric: the sender addresses this receiver too.
            back = pm.rank_overlap(src_ri, dst_ri, dst_ri.dp_rank).ranks
            if dst_ri.instance_rank not in back:
                raise AssertionError(
                    f"receiver rank {dst_ri.instance_rank} pairs with sender {flat}, "
                    f"which pairs with {back}"
                )
            src_keys = {
                (b.layer, b.role)
                for g in range(src_pages.num_layer_groups)
                for b in src_pages.buffers(g)
            }
            for tg in pm.plan_transfers(src_ri, dst_ri, src_pages, dst_pages):
                if not pm.owns_transfer(tg, src_ri, dst_ri):
                    continue
                src_ids = _byte_ids(src_pages, tg.self_layer_group, src_ri, global_units)
                dst_ids = _byte_ids(dst_pages, tg.peer_layer_group, dst_ri, global_units)
                for s, d, n in tg.plan.runs:
                    for b in range(n):
                        src_id = src_ids[(tg.self_pool, s + b)]
                        dst_id = dst_ids[(tg.peer_pool, d + b)]
                        if src_id != dst_id:
                            raise AssertionError(
                                f"rank {flat} -> {dst_ri.instance_rank} layer group "
                                f"{tg.peer_layer_group}: byte {d + b} of pool {tg.peer_pool} gets "
                                f"{src_id}, expected {dst_id}"
                            )
                        writes[(tg.peer_layer_group, (tg.peer_pool, d + b))] += 1
            for g in range(dst_pages.num_layer_groups):
                held = _byte_ids(dst_pages, g, dst_ri, global_units)
                for address, byte_id in held.items():
                    if (byte_id[0], byte_id[1]) in src_keys:
                        expected[(g, address)] = byte_id
        missing = [k for k in expected if writes[k] == 0]
        repeated = [k for k, count in writes.items() if count > 1]
        if missing or repeated:
            raise AssertionError(
                f"receiver rank {dst_ri.instance_rank}: {len(missing)} bytes never written "
                f"(first {missing[:3]}), {len(repeated)} written more than once "
                f"(first {repeated[:3]})"
            )
        checked += len(expected)
    return checked


def all_ranks(make: Callable[..., RankInfo], tp: int, pp: int = 1, cp: int = 1) -> List[RankInfo]:
    """``make(pp_rank=, tp_rank=, cp_rank=)`` for every rank of one instance, CP-minor."""
    return [
        make(pp_rank=p, tp_rank=t, cp_rank=c)
        for p in range(pp)
        for t in range(tp)
        for c in range(cp)
    ]


# ---------------------------------------------------------------------------
# Synthetic ranks: a model, a parallel layout, and the pages one rank describes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    pattern: str = "AAAA"  # per model layer: A = attention, M = recurrent
    kv_heads: int = 8
    head_dim: int = 8
    element_bytes: float = 2
    tokens_per_block: int = 4
    mla: bool = False
    nhd: bool = False
    index_key: Optional[int] = None  # replicated bytes per layer and block
    index_coalesced: bool = False  # index-key shares the K/V slot, interleaved per layer
    conv_sections: Tuple[int, ...] = (64, 32, 32)  # global bytes per layer
    ssm_heads: int = 8
    ssm_bytes_per_head: int = 16
    ple: Optional[int] = None  # replicated recurrent side state, bytes per layer
    interleaved_state: bool = False  # recurrent roles share one slot-major pool

    def global_units(self, buffer: BufferDesc, index: int, segment: Segment) -> int:
        """The true extent of a split axis, for the conformance oracle."""
        if buffer.role == "conv_state":
            return self.conv_sections[index]
        if buffer.role == "ssm_state":
            return self.ssm_heads
        return 1 if self.mla else self.kv_heads


@dataclass(frozen=True)
class Par:
    tp: int = 1
    pp: int = 1
    cp: int = 1
    adp: bool = False


def _stage_layers(model: Model, par: Par, pp_rank: int) -> List[int]:
    n = len(model.pattern)
    sizes = [n // par.pp + (1 if i < n % par.pp else 0) for i in range(par.pp)]
    start = sum(sizes[:pp_rank])
    return list(range(start, start + sizes[pp_rank]))


def _attention_group(model: Model, layers, kv: int, pools: list, base: int) -> group:
    size = int(model.tokens_per_block * kv * model.head_dim * model.element_bytes)
    kv_segments = (
        token_major(size, model.tokens_per_block, kv) if model.nhd else head_major(size, kv)
    )
    roles = ["key"] if model.mla else ["key", "value"]
    buffers = []
    kv_pool = len(pools)
    if model.index_key and model.index_coalesced:
        stride = len(roles) * size + model.index_key
        pools.append(pool(base, slot_bytes=stride * len(layers), num_slots=64))
        for i, layer in enumerate(layers):
            for r, role in enumerate(roles):
                buffers.append(buf(kv_pool, layer, role, i * stride + r * size, kv_segments))
            buffers.append(
                buf(
                    kv_pool,
                    layer,
                    "index_key",
                    i * stride + len(roles) * size,
                    replicated(model.index_key),
                )
            )
    else:
        pools.append(pool(base, slot_bytes=len(roles) * size * len(layers), num_slots=64))
        for i, layer in enumerate(layers):
            for r, role in enumerate(roles):
                buffers.append(buf(kv_pool, layer, role, (i * len(roles) + r) * size, kv_segments))
        if model.index_key:
            index_pool = len(pools)
            pools.append(
                pool(base + 0x100000, slot_bytes=model.index_key * len(layers), num_slots=64)
            )
            for i, layer in enumerate(layers):
                buffers.append(
                    buf(
                        index_pool,
                        layer,
                        "index_key",
                        i * model.index_key,
                        replicated(model.index_key),
                    )
                )
    return group("attention", buffers)


def _state_group(model: Model, layers, grid_size: int, pools: list, base: int) -> group:
    heads = model.ssm_heads // grid_size
    roles = [
        ("conv_state", sectioned(s // grid_size for s in model.conv_sections)),
        ("ssm_state", head_major(heads * model.ssm_bytes_per_head, heads)),
    ]
    if model.ple:
        roles.append(("ple", replicated(model.ple)))
    n = len(layers)
    first_pool = len(pools)
    buffers = []
    if model.interleaved_state:
        stride = sum(sum(s.nbytes for s in segments) for _, segments in roles)
        pools.append(pool(base, slot_bytes=stride * n))
        for i, layer in enumerate(layers):
            offset = i * stride
            for role, segments in roles:
                buffers.append(buf(first_pool, layer, role, offset, segments))
                offset += sum(s.nbytes for s in segments)
    else:
        for k, (role, segments) in enumerate(roles):
            size = sum(s.nbytes for s in segments)
            pools.append(pool(base + 0x100000 * k, slot_bytes=size * n))
            for i, layer in enumerate(layers):
                buffers.append(buf(first_pool + k, layer, role, i * size, segments))
    return group("state", buffers)


def make_rank(model: Model, par: Par, *, name="ctx", pp_rank=0, tp_rank=0, cp_rank=0) -> RankInfo:
    layers = _stage_layers(model, par, pp_rank)
    attn = [i for i in layers if model.pattern[i] == "A"]
    state = [i for i in layers if model.pattern[i] == "M"]
    kv = 1 if model.mla else (model.kv_heads if par.adp else max(1, model.kv_heads // par.tp))
    sizes = [len(_stage_layers(model, par, p)) for p in range(par.pp)]
    ri = RankInfo(
        instance_name=name,
        instance_rank=pp_rank * par.tp * par.cp + tp_rank * par.cp + cp_rank,
        tp_size=par.tp,
        tp_rank=tp_rank,
        pp_size=par.pp,
        pp_rank=pp_rank,
        layer_num_per_pp=sizes,
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        dp_size=par.tp if par.adp else 1,
        dp_rank=tp_rank if par.adp else 0,
        cp_size=par.cp,
        cp_rank=cp_rank,
    )
    topology = RankTopology(
        tp_size=par.tp,
        tp_rank=tp_rank,
        pp_size=par.pp,
        pp_rank=pp_rank,
        cp_size=par.cp,
        cp_rank=cp_rank,
        dp_size=par.tp if par.adp else 1,
        dp_rank=tp_rank if par.adp else 0,
        attention_dp=par.adp,
    )
    base = 0x10_0000_0000 if name == "ctx" else 0x20_0000_0000
    pools, groups = [], []
    if attn:
        groups.append(_attention_group(model, attn, kv, pools, base))
    if state:
        state_grid_size, _ = grid_position(topology, ("tp", "cp"))
        groups.append(_state_group(model, state, state_grid_size, pools, base + 0x4000000))
    ri.pages = pages(pools, groups, tokens_per_page=model.tokens_per_block, topology=topology)
    return ri


def instance(model: Model, par: Par, name: str) -> List[RankInfo]:
    return all_ranks(lambda **at: make_rank(model, par, name=name, **at), par.tp, par.pp, par.cp)


def _with_buffers(rank_pages: RankPages, buffers) -> RankPages:
    return RankPages(
        dataclasses.replace(rank_pages.layout, buffers=tuple(buffers)), rank_pages.pools
    )


# ---------------------------------------------------------------------------
# RankPages: the layout as peers read it, round trip
# ---------------------------------------------------------------------------


def test_pages_round_trip_and_dedup_segments_on_the_wire():
    for model in (Model(), Model(pattern="AMAM", index_key=8, ple=12), Model(mla=True, kv_heads=1)):
        check_pages_round_trip(make_rank(model, Par(tp=2)).pages)
    rank_pages = make_rank(Model(pattern="AAAAAAAA"), Par()).pages
    # Eight layers x K/V share one segment layout.
    assert len(rank_pages.to_dict()["segments"]) == 1


def test_rank_info_carries_its_pages():
    ri = make_rank(Model(pattern="AMAM", index_key=8, ple=12), Par(tp=2), tp_rank=1)
    restored = RankInfo.from_bytes(ri.to_bytes())
    assert restored.pages.layout == ri.pages.layout and restored.pages.pools == ri.pages.pools


def test_buffers_are_listed_per_group_in_slot_order_without_untransferred_ones():
    rank_pages = pages(
        [pool(0x1000, 96), pool(0x9000, 32)],
        [
            group(
                "attention",
                [
                    buf(1, 0, "index_key", 0, replicated(32)),
                    buf(0, 0, "value", 32, head_major(32, 2)),
                    buf(0, 0, "padding", 64, replicated(32), transferred=False),
                    buf(0, 0, "key", 0, head_major(32, 2)),
                ],
            )
        ],
    )
    assert [(b.pool_index, b.slot_offset, b.role) for b in rank_pages.buffers(0)] == [
        (0, 0, "key"),
        (0, 32, "value"),
        (1, 0, "index_key"),
    ]
    assert len(rank_pages.layout.buffers) == 4
    assert rank_pages.kind(0) == CacheKind.PAGED
    assert rank_pages.pool((0, 1)) == DevicePool(0, 1, 0x9000, 32, 16)
    slots = rank_pages.slots((0, 0), np.array([2, -1, 0]))
    assert slots.memory.ptrs.tolist() == [0x1000 + 2 * 96, 0x1000]
    assert slots.memory.bytes_per_region == 96


def test_segment_and_buffer_declarations():
    nhd = Segment(64, (("token", 4), ("head", 2)), "head")
    assert (nhd.rows, nhd.split_extent, nhd.unit_bytes) == (4, 2, 8)
    whole = BufferDesc(3, "index_key", 0, 0, 128, 24)
    assert whole.segments == (Segment(24),) and pm.replicated(whole)
    assert not pm.replicated(BufferDesc(3, "key", 0, 0, 0, 16, head_major(16, 2)))
    with pytest.raises(ValueError, match="is not one of the axes"):
        Segment(8, (("head", 2),), "token")
    with pytest.raises(ValueError, match="not a whole number"):
        Segment(10, (("head", 4),), "head")
    with pytest.raises(ValueError, match="do not make up"):
        BufferDesc(0, "key", 0, 0, 0, 32, head_major(16, 2))


# ---------------------------------------------------------------------------
# Describing a manager: build_page_layout over its storage and hooks
# ---------------------------------------------------------------------------


class _StorageManager:
    """A cache manager as ``build_page_layout`` reads it, with no runtime behind it.

    ``pool_groups``: per pool group, ``(num_slots, [(base, slot_bytes)], variants)`` with variants
    ``[(life_cycle, [(single_buffer_size, [(layer, role)])])]``. Internal layer ``i`` is global
    layer ``10 + i``. The hooks are the base class's: K/V head-major, index-K replicated, any other
    role undeclared.
    """

    tokens_per_block = 4
    mapping = None

    def __init__(self, *, life_cycles, pool_groups, windows=None, kv_heads=(2, None)):
        windows = windows or {}
        layers = sorted({i for ids in life_cycles for i in ids})
        self.pp_layers = [10 + i for i in layers]
        self.num_kv_heads_per_layer = [kv_heads[0]] * len(layers)
        self._kv_heads = kv_heads
        self._windows = [windows.get(ids[0]) for ids in life_cycles]
        self.kv_cache_manager_py_config = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_id=i, buffers=[], window_size=windows.get(i), num_sink_tokens=0
                )
                for i in layers
            ]
        )
        pool_group_of = {}
        descs = []
        for index, (num_slots, pool_specs, variants) in enumerate(pool_groups):
            for life_cycle, _ in variants:
                pool_group_of[life_cycle] = index
            descs.append(
                SimpleNamespace(
                    pool_group_index=index,
                    num_slots=num_slots,
                    pools=[
                        SimpleNamespace(pool_index=j, base_address=b, slot_bytes=n)
                        for j, (b, n) in enumerate(pool_specs)
                    ],
                    slot_desc=SimpleNamespace(
                        variants=[
                            SimpleNamespace(
                                layer_group_id=life_cycle,
                                coalesced_buffers=[
                                    SimpleNamespace(
                                        single_buffer_size=size,
                                        buffer_ids=[
                                            SimpleNamespace(layer_id=layer, role=role)
                                            for layer, role in ids
                                        ],
                                    )
                                    for size, ids in coalesced
                                ],
                            )
                            for life_cycle, coalesced in variants
                        ]
                    ),
                )
            )
        self.impl = SimpleNamespace(
            layer_grouping=life_cycles,
            pool_group_descs=descs,
            get_life_cycle_pool_group_indices=lambda: [
                pool_group_of[lc] for lc in range(len(life_cycles))
            ],
        )

    def get_buffer_layout(self, layer_id, role, size, tokens):
        if role == "index_key":
            return replicated(size)
        if role in ("key", "value"):
            return head_major(size, self.get_kv_heads(layer_id)[0])
        raise ValueError(f"{type(self).__name__} declares no layout for role {role!r}")

    def get_untransferred_roles(self):
        return frozenset()

    def get_layer_ids(self):
        return {i: (layer, layer) for i, layer in enumerate(self.pp_layers)}

    def get_kv_heads(self, layer_id):
        return self._kv_heads

    def get_layer_group_traits(self, layer_group):
        return default_layer_group_traits(self, layer_group)

    def _attention_read_window(self, layer_group):
        return self._windows[layer_group]

    def get_page_layout(self):
        return build_page_layout(self)


def _mapping(tp: int, tp_rank: int, cp: int = 1, cp_rank: int = 0) -> SimpleNamespace:
    """The parts of a ``Mapping`` ``build_page_layout`` reads to place a rank."""
    return SimpleNamespace(
        tp_size=tp,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        cp_size=cp,
        cp_rank=cp_rank,
        dp_size=1,
        enable_attention_dp=False,
    )


class _PaddedManager(_StorageManager):
    """Declares a ``padding`` role that keeps its place in the slot but never travels."""

    def get_buffer_layout(self, layer_id, role, size, tokens):
        if role == "padding":
            return replicated(size)
        return super().get_buffer_layout(layer_id, role, size, tokens)

    def get_untransferred_roles(self):
        return frozenset({"padding"})


def _life_cycles_manager(cls):
    # Two life cycles (full / SWA) share one pool group; index-K is coalesced into the K/V pool and
    # a padding role occupies slot bytes.
    return cls(
        life_cycles=[[0, 2], [1]],
        windows={1: 64},
        pool_groups=[
            (
                32,
                [(0x1000, 4096)],
                [
                    (0, [(64, [(0, "key"), (0, "value"), (0, "index_key"), (2, "key")])]),
                    (1, [(64, [(1, "key"), (1, "padding"), (1, "value")])]),
                ],
            )
        ],
    )


def test_manager_life_cycles_become_layer_groups_of_buffers():
    rank_pages = RankPages.of_manager(_life_cycles_manager(_PaddedManager))
    assert [rank_pages.kind(g) for g in range(2)] == [CacheKind.PAGED, CacheKind.PAGED]
    assert rank_pages.group(0).kept_window is None
    assert (rank_pages.group(1).kept_window, rank_pages.group(1).read_window) == (64, 64)
    assert rank_pages.group(0).layers == (10, 12) and rank_pages.group(1).layers == (11,)
    assert [(b.layer, b.role, b.slot_offset) for b in rank_pages.buffers(0)] == [
        (10, "key", 0),
        (10, "value", 64),
        (10, "index_key", 128),
        (12, "key", 192),
    ]
    # The padding buffer keeps its slot offset but is not transferred.
    assert [(b.role, b.slot_offset) for b in rank_pages.buffers(1)] == [("key", 0), ("value", 128)]
    (padding,) = [b for b in rank_pages.layout.buffers if b.role == "padding"]
    assert not padding.transferred and padding.slot_offset == 64
    assert rank_pages.buffers(0)[0].segments == head_major(64, 2)
    assert rank_pages.buffers(0)[2].segments == replicated(64)
    assert rank_pages.pools == (DevicePool(0, 0, 0x1000, 4096, 32),)
    check_pages_round_trip(rank_pages)


def test_an_undeclared_role_is_an_error():
    with pytest.raises(ValueError, match="declares no layout for role 'padding'"):
        RankPages.of_manager(_life_cycles_manager(_StorageManager))


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_shard_span_duplicates_units_on_the_finer_grid():
    # 2 global heads over TP=4: ranks 0, 1 hold head 0 and ranks 2, 3 head 1.
    grid = [pm.ShardGrid(4, r) for r in range(4)]
    spans = [topo.shard_span(1, g, virtual_units=2) for g in grid]
    assert spans == [(0, 1), (0, 1), (1, 2), (1, 2)]
    # Two ranks hold the head one TP=1 peer needs: exactly one of them is elected.
    factor = topo.replication_factor(1, grid[0], 2, pm.ShardGrid(1, 0))
    assert factor == 2
    assert [topo.elects_sender(g, 0, factor) for g in grid] == [True, False, True, False]


def test_state_grid_follows_attention_dp_and_helix():
    model, par = Model(pattern="AMAM"), Par(tp=2, cp=2)
    helix = make_rank(model, par, tp_rank=1, cp_rank=1)
    assert [helix.pages.group(g).grid for g in range(2)] == [("tp",), ("tp", "cp")]
    assert pm.shard_grid(("tp", "cp"), helix) == pm.ShardGrid(4, 3)
    assert pm.shard_grid(("tp",), helix) == pm.ShardGrid(2, 1)
    adp = make_rank(model, Par(tp=4, adp=True), tp_rank=3)
    assert pm.shard_grid(("tp", "cp"), adp) == pm.ShardGrid(1, 0)
    assert pm.shard_grid(("tp",), adp) == pm.ShardGrid(1, 0)
    assert adp.tp_size_per_dp_group == 1 and helix.tp_size_per_dp_group == 2
    for grid in (("pp",), (), ("cp",), ("cp", "tp"), ("tp", "tp")):
        with pytest.raises(ValueError, match="grid must be a leading part"):
            pages(
                [pool(0x1000, 32)],
                [group("attention", [buf(0, 0, "x", 0, replicated(32))], grid=grid)],
            )


# ---------------------------------------------------------------------------
# Resharding conformance: every receiver byte exactly once, from the right source byte
# ---------------------------------------------------------------------------

_MODELS = {
    "gqa": Model(),
    "gqa_dup": Model(kv_heads=2),
    "nhd": Model(kv_heads=4, nhd=True),
    "mla_index_key": Model(mla=True, kv_heads=1, head_dim=16, index_key=24),
    "coalesced_index_key": Model(kv_heads=4, nhd=True, index_key=8, index_coalesced=True),
    "nvfp4": Model(element_bytes=0.5, head_dim=16),
    "hybrid": Model(pattern="AMAM", kv_heads=4),
    "hybrid_ple_interleaved": Model(pattern="AMAMAM", ple=12, interleaved_state=True),
    "recurrent_only": Model(pattern="MMMM"),
}

_LAYOUTS = {
    "tp1_tp1": (Par(tp=1), Par(tp=1)),
    "tp1_tp2": (Par(tp=1), Par(tp=2)),
    "tp4_tp2": (Par(tp=4), Par(tp=2)),
    "tp2_tp8": (Par(tp=2), Par(tp=8)),
    "tp8_tp2": (Par(tp=8), Par(tp=2)),
    "tp4_adp4": (Par(tp=4), Par(tp=4, adp=True)),
    "adp2_tp4": (Par(tp=2, adp=True), Par(tp=4)),
    "pp2_tp2_to_tp1_pp2": (Par(tp=2, pp=2), Par(tp=1, pp=2)),
    "pp4_to_tp2_pp2": (Par(pp=4), Par(tp=2, pp=2)),
    "helix_tp2_to_tp2_cp2": (Par(tp=2), Par(tp=2, cp=2)),
    "helix_tp4_to_tp2_cp2": (Par(tp=4), Par(tp=2, cp=2)),
}


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_resharding_conformance(model_name, layout):
    model = _MODELS[model_name]
    ctx_par, gen_par = _LAYOUTS[layout]
    checked = check_resharding(
        instance(model, ctx_par, "ctx"), instance(model, gen_par, "gen"), model.global_units
    )
    assert checked > 0


def test_oracle_catches_a_wrong_mapping(monkeypatch):
    """The oracle is not vacuous: it catches a correspondence that ignores rank shards."""
    from tensorrt_llm._torch.disaggregation.page_mapping import correspondence

    monkeypatch.setattr(correspondence, "shard_span", lambda units, grid, virtual: (0, units))
    model = Model(kv_heads=4)
    with pytest.raises(AssertionError):
        check_resharding(
            instance(model, Par(tp=2), "ctx"), instance(model, Par(tp=1), "gen"), model.global_units
        )


def test_helix_replicated_state_reaches_every_receiver():
    """Replicated side state elects its sender on the state grid, the grid pairing uses.

    Under helix the attention grid (TP) and the state grid (TP x CP) differ; an election on the
    attention grid would leave some receivers with no sender for the side state at all.
    """
    model = Model(pattern="AMAM", kv_heads=4, ple=12)
    ctx, gen = instance(model, Par(tp=4), "ctx"), instance(model, Par(tp=2, cp=2), "gen")
    for dst in gen:
        senders = [
            src
            for src in ctx
            for tg in pm.plan_transfers(src, dst, src.pages, dst.pages)
            if any(src.pages.buffers(tg.self_layer_group)[i].role == "ple" for i in tg.buffers)
            and pm.owns_transfer(tg, src, dst)
        ]
        assert len(senders) == 1, dst.instance_rank


def test_duplicate_head_holders_elect_one_sender():
    # 2 KV heads over 4 context ranks: ranks 0, 1 hold head 0 and ranks 2, 3 head 1.
    model = Model(kv_heads=2)
    ctx, (gen,) = instance(model, Par(tp=4), "ctx"), instance(model, Par(tp=1), "gen")
    owners = []
    for src in ctx:
        groups = pm.plan_transfers(src, gen, src.pages, gen.pages)
        assert groups and all(tg.owner_factor == 2 for tg in groups)
        if all(pm.owns_transfer(tg, src, gen) for tg in groups):
            owners.append(src.tp_rank)
    assert owners == [0, 2]
    # A receiver of another DP rank rotates the election onto the other duplicate.
    other_dp = dataclasses.replace(gen, dp_rank=1)
    rotated = [
        src.tp_rank
        for src in ctx
        if all(
            pm.owns_transfer(tg, src, other_dp)
            for tg in pm.plan_transfers(src, gen, src.pages, gen.pages)
        )
    ]
    assert rotated == [1, 3]


# ---------------------------------------------------------------------------
# Matching, transfer groups and peer validation
# ---------------------------------------------------------------------------


def test_buffers_match_by_layer_and_role():
    ctx = make_rank(Model(pattern="AMAM", index_key=8), Par(pp=2), pp_rank=1)
    gen = make_rank(Model(pattern="AMAM", index_key=8), Par(), name="gen")
    mapping = pm.match_buffers(ctx.pages, gen.pages)
    for (g, i), (pg, pi) in mapping.items():
        mine = ctx.pages.buffers(g)[i]
        theirs = gen.pages.buffers(pg)[pi]
        assert (mine.layer, mine.role) == (theirs.layer, theirs.role)
    # Every buffer of the PP stage (layers 2, 3) has a counterpart; the peer's layers 0, 1 do not.
    assert len(mapping) == sum(len(ctx.pages.buffers(g)) for g in range(ctx.pages.num_layer_groups))
    assert pm.match_buffers(ctx.pages, None) == {}

    dup = _with_buffers(gen.pages, gen.pages.layout.buffers + (gen.pages.layout.buffers[0],))
    with pytest.raises(ValueError, match="appears in layer groups 0 and 0"):
        pm.match_buffers(ctx.pages, dup)


def test_a_buffer_paged_on_one_side_and_state_on_the_other_is_rejected():
    paged = pages([pool(0x1000, 32)], [group("attention", [buf(0, 0, "x", 0, replicated(32))])])
    state = pages([pool(0x2000, 32)], [group("state", [buf(0, 0, "x", 0, replicated(32))])])
    with pytest.raises(ValueError, match="lies in layer groups that split differently"):
        pm.match_buffers(paged, state)
    # A state group whose units are blocks, split over TP alone, still splits differently.
    blocks = pages(
        [pool(0x2000, 32)],
        [group("state", [buf(0, 0, "x", 0, replicated(32))], grid=("tp",), unit="block")],
    )
    with pytest.raises(ValueError, match="split differently"):
        pm.match_buffers(paged, blocks)
    assert pm.match_buffers(state, state) == {(0, 0): (0, 0)}


def test_transfer_groups_merge_contiguous_buffers_and_skip_excluded_ones():
    ctx = make_rank(Model(), Par(), name="ctx")
    gen = make_rank(Model(), Par(), name="gen")
    (tg,) = pm.plan_transfers(ctx, gen, ctx.pages, gen.pages)
    # Equal layouts: every layer's K and V are contiguous on both sides, one run moves the slot.
    slot = ctx.pages.pool((0, 0)).slot_bytes
    assert tg.plan.runs == [(0, 0, slot)] and tg.owner_factor == 1
    assert (tg.self_pool, tg.peer_pool, tg.grid) == ((0, 0), (0, 0), ("tp",))
    excluded = frozenset({(0, 0), (0, 1)})  # layer 0's K and V
    (tg,) = pm.plan_transfers(ctx, gen, ctx.pages, gen.pages, excluded)
    assert tg.plan.runs == [(slot // 4, slot // 4, slot * 3 // 4)]
    assert tg.buffers == tuple(range(2, 8))


def test_head_split_buffer_runs_take_the_peer_shard():
    # TP=2 sender rank 1 holds heads 4..7 of 8; a TP=1 receiver holds all eight.
    ctx = make_rank(Model(), Par(tp=2), tp_rank=1)
    gen = make_rank(Model(), Par(), name="gen")
    src, dst = ctx.pages.buffers(0)[0], gen.pages.buffers(0)[0]
    head_bytes = src.size // 4
    runs = pm.buffer_runs(src, dst, ("tp",), ctx, gen)
    assert runs == [(src.slot_offset, dst.slot_offset + 4 * head_bytes, 4 * head_bytes)]


def test_excluded_buffers_are_the_ones_the_manager_names():
    ri = make_rank(Model(pattern="AMAM", ple=12), Par())
    manager = SimpleNamespace(
        get_untransferred_buffers=lambda request: (
            frozenset({(1, "ple"), (0, "value")}) if request.rebuilds else frozenset()
        )
    )
    excluded = pm.excluded_buffers(manager, ri.pages, SimpleNamespace(rebuilds=True))
    named = {(ri.pages.buffers(g)[i].layer, ri.pages.buffers(g)[i].role) for g, i in excluded}
    assert named == {(1, "ple"), (0, "value")}
    assert pm.excluded_buffers(manager, ri.pages, SimpleNamespace(rebuilds=False)) == frozenset()
    assert pm.excluded_buffers(manager, None, SimpleNamespace(rebuilds=True)) == frozenset()
    # Excluded buffers do not move.
    gen = make_rank(Model(pattern="AMAM", ple=12), Par(), name="gen")
    moved = {
        (
            ri.pages.buffers(tg.self_layer_group)[i].layer,
            ri.pages.buffers(tg.self_layer_group)[i].role,
        )
        for tg in pm.plan_transfers(ri, gen, ri.pages, gen.pages, excluded)
        for i in tg.buffers
    }
    assert moved and not moved & named


def test_peer_layout_mismatches_are_rejected():
    ctx = make_rank(Model(pattern="MMMM"), Par(tp=2))
    gen = make_rank(Model(pattern="MMMM", ssm_heads=4), Par(tp=1), name="gen")
    with pytest.raises(ValueError, match="TP-aggregated"):
        pm.validate_peer_compatible(ctx, gen, ctx.pages, gen.pages)

    nhd = make_rank(Model(nhd=True), Par(), name="gen")
    hnd = make_rank(Model(), Par())
    with pytest.raises(ValueError, match="layer=0, role='key'\\) segment 0: layout differs"):
        pm.validate_peer_compatible(hnd, nhd, hnd.pages, nhd.pages)

    with_index = make_rank(Model(index_key=8), Par(), name="gen")
    with pytest.raises(ValueError, match="replicated roles differ on overlapping layers"):
        pm.validate_peer_compatible(hnd, with_index, hnd.pages, with_index.pages)

    split_key = _with_buffers(
        hnd.pages,
        [
            dataclasses.replace(b, segments=head_major(b.size // 2, 4) * 2)
            if b.role == "key"
            else b
            for b in hnd.pages.layout.buffers
        ],
    )
    with pytest.raises(ValueError, match="segment count differs"):
        pm.validate_peer_compatible(hnd, hnd, hnd.pages, split_key)

    pm.validate_peer_compatible(hnd, hnd, hnd.pages, hnd.pages)
    pm.validate_peer_compatible(hnd, hnd, None, hnd.pages)


# ---------------------------------------------------------------------------
# Unit rules: selection, chunking, pairing
# ---------------------------------------------------------------------------


def _pair_ctx(self_ri, peer_ri, g, dst_start_token=None):
    return PairContext(
        self_ri=self_ri,
        peer_ri=peer_ri,
        group=self_ri.pages.group(g),
        dst_start_token=dst_start_token,
    )


def test_local_units_list_prompt_blocks_or_the_state_slot():
    ri = make_rank(Model(pattern="AMAM"), Par())
    adapter = SimpleNamespace(
        tokens_per_block=4,
        get_block_ordinals=lambda request, g: np.array([7, 8, 9, 10, 11]),
        get_state_slot=lambda request, g: request.slot[g],
    )

    def units(g, request, transfer_len, cached_tokens=0):
        ctx = LocalUnitsContext(request, g, ri.pages, adapter, transfer_len, cached_tokens)
        return pm.local_units(ctx).tolist()

    request = SimpleNamespace(slot={1: 3})
    # The speculative tail past the prompt's blocks is left out.
    assert units(0, request, 12) == [7, 8, 9]
    # Fewer blocks allocated than the prompt spans: the rest is padded.
    assert units(0, request, 26) == [7, 8, 9, 10, 11, -1, -1]
    # The receiver's cached prefix is masked.
    assert units(0, request, 12, cached_tokens=4) == [-1, 8, 9]
    assert units(1, request, 12) == [3]
    assert units(1, SimpleNamespace(slot={1: -1}), 12) == []


def test_paged_units_pair_by_ordinal_and_state_by_slot():
    ri = make_rank(Model(pattern="AMAM"), Par())
    paged, state = 0, 1  # layer groups of the attention and the recurrent layers
    src, dst = pm.pair_units(
        np.array([10, 11, -1, 13]), np.array([-1, 21, 22, 23]), _pair_ctx(ri, ri, paged)
    )
    assert src.tolist() == [11, 13] and dst.tolist() == [21, 23]
    src, dst = pm.pair_units(
        np.array([10, 11, 12, 13]), np.array([20, 21, 22, 23]), _pair_ctx(ri, ri, paged, 8)
    )
    assert src.tolist() == [12, 13] and dst.tolist() == [22, 23]
    src, dst = pm.pair_units(np.array([5]), np.array([7]), _pair_ctx(ri, ri, state))
    assert src.tolist() == [5] and dst.tolist() == [7]
    empty = np.array([], dtype=np.int64)
    assert pm.pair_units(np.array([5]), empty, _pair_ctx(ri, ri, state))[0].size == 0
    with pytest.raises(ValueError, match="differ in length"):
        pm.pair_units(np.array([1, 2]), np.array([1]), _pair_ctx(ri, ri, paged))


def test_helix_receiver_takes_its_strided_blocks():
    ctx = make_rank(Model(), Par(tp=1))
    gen = make_rank(Model(), Par(tp=1, cp=2), name="gen", cp_rank=1)
    src, dst = pm.pair_units(np.array([10, 11, 12, 13]), np.array([21, 23]), _pair_ctx(ctx, gen, 0))
    assert src.tolist() == [11, 13] and dst.tolist() == [21, 23]


def test_chunk_units_defer_windows_and_whole_request_groups():
    rank_pages = pages(
        [pool(0x1000, 32), pool(0x2000, 32), pool(0x3000, 32)],
        [
            group("attention", [buf(0, 0, "key", 0, head_major(32, 2))]),
            group("attention", [buf(1, 1, "key", 0, head_major(32, 2))], window=8),
            group("state", [buf(2, 2, "ssm_state", 0, head_major(32, 2))]),
        ],
    )
    units = np.arange(6)

    def chunk(g, values, is_last):
        return pm.chunk_units(rank_pages, g, values, ChunkContext(2, 4, is_last, 24)).tolist()

    assert chunk(0, units, False) == [-1, -1, 2, 3, -1, -1]
    assert chunk(1, units, False) == [-1] * 6
    assert chunk(1, units, True) == units.tolist()
    assert chunk(2, np.array([5]), False) == [-1]
    assert chunk(2, np.array([5]), True) == [5]
    assert pm.has_whole_request_groups(rank_pages)
    assert not pm.has_whole_request_groups(make_rank(Model(), Par()).pages)
    assert not pm.has_whole_request_groups(None)


def test_transfer_bytes_count_blocks_or_one_state_per_request():
    assert pm_units.transfer_bytes(CacheKind.PAGED, 10, np.array([3, -1, 4])) == 20
    assert pm_units.transfer_bytes(CacheKind.STATE, 10, np.array([9])) == 10
    assert pm_units.transfer_bytes(CacheKind.STATE, 10, np.array([-1])) == 0
    assert pm_units.sends_incrementally(CacheKind.PAGED)
    assert not pm_units.sends_incrementally(CacheKind.STATE)


# ---------------------------------------------------------------------------
# Whole-pages questions: bytes, bounce sizing, memory registration
# ---------------------------------------------------------------------------


def test_chunk_bytes_count_buffers_not_slots():
    rank_pages = make_rank(Model(pattern="AMAM", kv_heads=4, ple=12), Par()).pages
    attn_bytes = sum(b.size for b in rank_pages.buffers(0))
    state_bytes = sum(b.size for b in rank_pages.buffers(1))
    units = [np.array([3, -1, 4]), np.array([9])]
    assert pm.chunk_transfer_bytes(rank_pages, units) == 2 * attn_bytes + state_bytes
    ple = {(1, i) for i, b in enumerate(rank_pages.buffers(1)) if b.role == "ple"}
    assert pm.chunk_transfer_bytes(rank_pages, units, ple) == 2 * attn_bytes + state_bytes - 2 * 12
    assert pm.chunk_transfer_bytes(None, units) == 0
    assert pm.receiver_extra_bytes(rank_pages, rank_pages, units) == state_bytes
    assert pm.receiver_extra_bytes(None, rank_pages, units) is None
    # Only the state buffers the sender also holds land in the receiver's slot.
    attention_only = make_rank(Model(pattern="AAAA"), Par()).pages
    assert pm.receiver_extra_bytes(attention_only, rank_pages, units) == 0
    (attention_pool,) = rank_pages.group_pools(0)
    assert pm.bounce_unit_bytes_per_layer_group(rank_pages) == [attention_pool.slot_bytes, None]


def test_bounce_bytes_cover_every_pool_of_a_paged_group():
    rank_pages = make_rank(Model(index_key=8), Par()).pages
    assert pm.bounce_unit_bytes_per_layer_group(rank_pages) == [
        rank_pages.pool((0, 0)).slot_bytes + rank_pages.pool((0, 1)).slot_bytes
    ]


def test_replicated_buffers_are_detected():
    assert pm.has_replicated_buffers(make_rank(Model(index_key=8), Par()).pages)
    assert pm.has_replicated_buffers(make_rank(Model(pattern="AMAM", ple=12), Par()).pages)
    assert not pm.has_replicated_buffers(make_rank(Model(), Par()).pages)
    assert not pm.has_replicated_buffers(None)


def test_memory_registration_dedups_shared_state_allocation():
    rank_pages = make_rank(Model(pattern="MMMM", ple=12, interleaved_state=True), Par()).pages
    descs = pm.memory_registration_descs(rank_pages, device_id=2)
    assert len(descs) == 1
    (p,) = rank_pages.pools
    assert descs[0] == (p.base, p.num_slots * p.slot_bytes, 2, "kv_cache_memory_pool0")

    # Two state pools carved from one allocation register once; paged pools always register.
    shared = pages(
        [pool(0x1000, 64, num_slots=4), pool(0x1000 + 128, 32, num_slots=4)],
        [
            group(
                "state",
                [
                    buf(0, 0, "conv_state", 0, sectioned([64])),
                    buf(1, 0, "ssm_state", 0, head_major(32, 2)),
                ],
            )
        ],
    )
    regions = pm_units.registration_regions(shared, 0)
    assert [(r.base_address, r.size, r.within_known) for r in regions] == [
        (0x1000, 256, True),
        (0x1080, 128, True),
    ]
    assert [d[:2] for d in pm.memory_registration_descs(shared, 0)] == [(0x1000, 256)]
    paged = pages(
        [pool(0x1000, 64, num_slots=4), pool(0x1000 + 128, 32, num_slots=4)],
        [
            group(
                "attention",
                [
                    buf(0, 0, "key", 0, head_major(64, 2)),
                    buf(1, 0, "index_key", 0, replicated(32)),
                ],
            )
        ],
    )
    assert not any(r.within_known for r in pm_units.registration_regions(paged, 0))
    assert len(pm.memory_registration_descs(paged, 0)) == 2


# ---------------------------------------------------------------------------
# The sender path end to end: Sender._build_kv_write_meta over real pages
# ---------------------------------------------------------------------------


def test_sender_write_meta_covers_paged_blocks_and_state_slot():
    from tensorrt_llm import DisaggregatedParams
    from tensorrt_llm._torch.disaggregation.base import TokenRange
    from tensorrt_llm._torch.disaggregation.native.extent import Chunk
    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    from tensorrt_llm._torch.disaggregation.native.transfer import KVSendTask, RecvReqInfo, Sender

    model = Model(pattern="AMAM", kv_heads=4, ple=12)
    ctx = make_rank(model, Par(tp=2))
    gen = make_rank(model, Par(tp=1), name="gen")
    registrar = PeerRegistrar(ctx)
    registrar.register(gen.instance_name, gen.instance_rank, gen)
    sender = Sender.__new__(Sender)
    sender._registrar = registrar

    task = KVSendTask(
        Chunk(
            block_ids_per_layer_groups=[np.array([10, -1, 12]), np.array([3])],
            kind_per_layer_group=[ctx.pages.kind(g) for g in range(ctx.pages.num_layer_groups)],
            token_range=TokenRange(0, 12),
            is_last=True,
        ),
        DisaggregatedParams(disagg_request_id=7),
        slice_id=0,
        prompt_len=12,
    )
    req = RecvReqInfo(
        sender_req_id=7,
        instance_name=gen.instance_name,
        instance_rank=gen.instance_rank,
        block_ids_per_layer_groups=[np.array([20, 21, 22]), np.array([5])],
        unique_rid=7,
    )
    meta = sender._build_kv_write_meta(task, req)

    expected = []
    for tg in registrar.transfer_groups(gen):
        if not pm.owns_transfer(tg, ctx, gen):
            continue
        src_pool = ctx.pages.pool(tg.self_pool)
        dst_pool = gen.pages.pool(tg.peer_pool)
        pairs = [(10, 20), (12, 22)] if tg.self_layer_group == 0 else [(3, 5)]
        for s_unit, d_unit in pairs:
            for s, d, n in tg.plan.runs:
                expected.append(
                    (
                        src_pool.base + s_unit * src_pool.slot_bytes + s,
                        dst_pool.base + d_unit * dst_pool.slot_bytes + d,
                        n,
                    )
                )
    got = sorted(zip(meta.src_ptrs.tolist(), meta.dst_ptrs.tolist(), meta.sizes.tolist()))
    assert got == sorted(expected)
    assert meta.expected_transfers == 1  # this ctx rank addresses the single TP=1 receiver
    # TP rank 0 of 2 is the elected sender of the replicated side state, and it sends the state
    # once, into the receiver's slot 5.
    (ple,) = [b for b in gen.pages.buffers(1) if b.role == "ple"][:1]
    ple_pool = gen.pages.pool((ple.pool_group, ple.pool_index))
    slot5 = ple_pool.base + 5 * ple_pool.slot_bytes
    assert sum(n for _, d, n in got if slot5 <= d < slot5 + ple_pool.slot_bytes) == 2 * 12


# ---------------------------------------------------------------------------
# A new cache is its manager's hooks: a layout no built-in manager writes
# ---------------------------------------------------------------------------


class ChannelCacheManager(_StorageManager):
    """A cache with a ``latent`` role whose channels split across TP inside each token.

    Per layer it holds a head-major ``key`` and a token-major ``latent`` (``[token, channel]``,
    channels split), interleaved in one pool, plus a ``scratch`` role in a pool of its own that
    never travels. It describes itself only through the hooks ``build_page_layout`` calls;
    nothing in ``page_mapping`` or the pages builder knows about it.
    """

    LAYERS = 3
    HEADS = 4
    HEAD_BYTES = 16  # one head of one block
    CHANNELS = 8
    CHANNEL_BYTES = 2
    SCRATCH_BYTES = 16

    def __init__(self, tp: int, tp_rank: int, base: int):
        self.tp = tp
        self.mapping = _mapping(tp, tp_rank)
        self.local_heads = self.HEADS // tp
        self.local_channels = self.CHANNELS // tp
        key = self.local_heads * self.HEAD_BYTES
        latent = self.tokens_per_block * self.local_channels * self.CHANNEL_BYTES
        assert key == latent, "key and latent share one coalesced buffer"
        n = self.LAYERS
        roles = [(layer, role) for layer in range(n) for role in ("key", "latent")]
        super().__init__(
            life_cycles=[list(range(n))],
            kv_heads=(self.local_heads, self.HEADS),
            pool_groups=[
                (
                    8,
                    [(base, key * 2 * n), (base + 0x100000, self.SCRATCH_BYTES * n)],
                    [
                        (
                            0,
                            [
                                (key, roles),
                                (self.SCRATCH_BYTES, [(layer, "scratch") for layer in range(n)]),
                            ],
                        )
                    ],
                )
            ],
        )

    def get_buffer_layout(self, layer_id, role, size, tokens):
        if role == "latent":
            axes = (("token", tokens), ("channel", self.local_channels))
            return (Segment(size, axes, "channel"),)
        if role == "scratch":
            return replicated(size)
        return super().get_buffer_layout(layer_id, role, size, tokens)

    def get_untransferred_roles(self):
        return frozenset({"scratch"})

    @classmethod
    def global_units(cls, buffer: BufferDesc, index: int, segment: Segment) -> int:
        return cls.CHANNELS if buffer.role == "latent" else cls.HEADS


def _channel_rank(tp: int, name: str, tp_rank: int) -> RankInfo:
    base = 0x30_0000_0000 if name == "ctx" else 0x40_0000_0000
    manager = ChannelCacheManager(tp, tp_rank, base + tp_rank * 0x1000_0000)
    return RankInfo(
        instance_name=name,
        instance_rank=tp_rank,
        tp_size=tp,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[ChannelCacheManager.LAYERS],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        pages=RankPages.of_manager(manager),
    )


def test_new_cache_layout_comes_from_its_manager_hooks():
    rank_pages = _channel_rank(2, "ctx", 0).pages
    assert rank_pages.kind(0) == CacheKind.PAGED
    assert [(b.layer, b.role, b.slot_offset) for b in rank_pages.buffers(0)] == [
        (10, "key", 0),
        (10, "latent", 32),
        (11, "key", 64),
        (11, "latent", 96),
        (12, "key", 128),
        (12, "latent", 160),
    ]
    latent = rank_pages.buffers(0)[1]
    (segment,) = latent.segments
    assert (segment.rows, segment.split_extent, segment.unit_bytes) == (4, 4, 2)
    assert {b.role for b in rank_pages.layout.buffers if not b.transferred} == {"scratch"}
    # The scratch pool holds nothing that travels: it is neither registered nor bounced.
    assert [p.index for p in rank_pages.group_pools(0)] == [0]
    assert pm.bounce_unit_bytes_per_layer_group(rank_pages) == [rank_pages.pool((0, 0)).slot_bytes]
    check_pages_round_trip(rank_pages)


@pytest.mark.parametrize("ctx_tp,gen_tp", [(1, 1), (1, 2), (2, 1), (2, 4), (4, 2), (1, 4)])
def test_new_cache_reshards_through_its_hooks_alone(ctx_tp, gen_tp):
    ctx = [_channel_rank(ctx_tp, "ctx", r) for r in range(ctx_tp)]
    gen = [_channel_rank(gen_tp, "gen", r) for r in range(gen_tp)]
    for dst in gen:
        pm.validate_peer_compatible(dst, ctx[0], dst.pages, ctx[0].pages)
    assert check_resharding(ctx, gen, ChannelCacheManager.global_units) > 0
    assert pm.memory_registration_descs(ctx[0].pages, 0)


def test_new_cache_rejects_a_peer_writing_the_role_another_way():
    class HeadMajorLatent(ChannelCacheManager):
        def get_buffer_layout(self, layer_id, role, size, tokens):
            if role == "latent":
                return head_major(size, self.local_channels)
            return super().get_buffer_layout(layer_id, role, size, tokens)

    ctx = _channel_rank(2, "ctx", 0)
    peer = RankPages.of_manager(HeadMajorLatent(2, 0, 0x50_0000_0000))
    with pytest.raises(ValueError, match="role='latent'\\) segment 0: layout differs"):
        pm.validate_peer_compatible(ctx, ctx, ctx.pages, peer)


# ---------------------------------------------------------------------------
# Layer groups that split over another grid or hold other units, declared by their hooks alone
# ---------------------------------------------------------------------------


class _GridCacheManager(_StorageManager):
    """One layer group of ``LAYERS`` layers, one pool per role, declared by its hooks.

    Its buffers split over the ranks ``GRID`` names, as ``get_layer_group_traits`` says.
    """

    LAYERS = 2
    KIND = "attention"
    GRID: Tuple[str, ...] = ("tp", "cp")
    UNIT = "block"

    def __init__(self, tp: int, tp_rank: int, cp: int, cp_rank: int, base: int):
        self.mapping = _mapping(tp, tp_rank, cp, cp_rank)
        self.grid_size, _ = grid_position(
            RankTopology(tp_size=tp, tp_rank=tp_rank, cp_size=cp, cp_rank=cp_rank), self.GRID
        )
        n = self.LAYERS
        sizes = self.role_bytes()
        super().__init__(
            life_cycles=[list(range(n))],
            kv_heads=self.kv_heads(),
            pool_groups=[
                (
                    8,
                    [(base + 0x100000 * k, size * n) for k, size in enumerate(sizes.values())],
                    [
                        (
                            0,
                            [
                                (size, [(layer, role) for layer in range(n)])
                                for role, size in sizes.items()
                            ],
                        )
                    ],
                )
            ],
        )

    def role_bytes(self) -> Dict[str, int]:
        raise NotImplementedError

    def kv_heads(self) -> Tuple[int, Optional[int]]:
        return (0, None)

    def get_layer_group_traits(self, layer_group):
        return LayerGroupTraits(self.KIND, self.GRID, self.UNIT, False)


class TokenSplitCacheManager(_GridCacheManager):
    """K/V whose tokens split over TP x CP.

    Each rank of the grid holds a contiguous run of every block's tokens, all heads of each
    (``[token, head]``, tokens split). Every rank holds every block (``blocks_by_cp`` False).
    """

    tokens_per_block = 8
    HEADS = 2
    TOKEN_HEAD_BYTES = 4  # one head of one token

    def role_bytes(self):
        size = self.tokens_per_block // self.grid_size * self.HEADS * self.TOKEN_HEAD_BYTES
        return {"key": size, "value": size}

    def kv_heads(self):
        return (self.HEADS, self.HEADS)

    def get_buffer_layout(self, layer_id, role, size, tokens):
        axes = (("token", tokens // self.grid_size), ("head", self.HEADS))
        return (Segment(size, axes, "token"),)

    @classmethod
    def global_units(cls, buffer: BufferDesc, index: int, segment: Segment) -> int:
        return cls.tokens_per_block


class BlockStateCacheManager(_GridCacheManager):
    """Recurrent state snapshotted once per block.

    A "state" layer group whose units are block ordinals, its conv sections and SSM heads split
    over the grid of ``GRID``.
    """

    KIND = "state"
    HEADS = 8
    HEAD_BYTES = 16
    CONV_SECTIONS = (32, 16, 16)

    def role_bytes(self):
        return {
            "conv_state": sum(self.CONV_SECTIONS) // self.grid_size,
            "ssm_state": self.HEADS // self.grid_size * self.HEAD_BYTES,
        }

    def get_buffer_layout(self, layer_id, role, size, tokens):
        if role == "conv_state":
            return sectioned(s // self.grid_size for s in self.CONV_SECTIONS)
        if role == "ssm_state":
            return head_major(size, self.HEADS // self.grid_size)
        return super().get_buffer_layout(layer_id, role, size, tokens)

    @classmethod
    def global_units(cls, buffer: BufferDesc, index: int, segment: Segment) -> int:
        return cls.CONV_SECTIONS[index] if buffer.role == "conv_state" else cls.HEADS


class BlockStateOverTpManager(BlockStateCacheManager):
    GRID = ("tp",)


def _grid_instance(manager_cls, tp: int, cp: int, name: str) -> List[RankInfo]:
    base = 0x60_0000_0000 if name == "ctx" else 0x70_0000_0000

    def make(pp_rank, tp_rank, cp_rank):
        flat = tp_rank * cp + cp_rank
        manager = manager_cls(tp, tp_rank, cp, cp_rank, base + flat * 0x1000_0000)
        return RankInfo(
            instance_name=name,
            instance_rank=flat,
            tp_size=tp,
            tp_rank=tp_rank,
            pp_size=1,
            pp_rank=pp_rank,
            layer_num_per_pp=[manager_cls.LAYERS],
            sender_endpoints=[],
            self_endpoint="",
            transfer_engine_info=b"",
            cp_size=cp,
            cp_rank=cp_rank,
            pages=RankPages.of_manager(manager),
        )

    return all_ranks(make, tp, cp=cp)


def _check_grid_resharding(manager_cls, ctx_tp, ctx_cp, gen_tp, gen_cp) -> None:
    ctx = _grid_instance(manager_cls, ctx_tp, ctx_cp, "ctx")
    gen = _grid_instance(manager_cls, gen_tp, gen_cp, "gen")
    for dst in gen:
        assert pm.check_peer_compatible(dst, ctx[0])
        pm.validate_peer_compatible(dst, ctx[0], dst.pages, ctx[0].pages)
    assert check_resharding(ctx, gen, manager_cls.global_units) > 0


_GRID_LAYOUTS = [
    (1, 1, 1, 2),
    (1, 1, 2, 2),
    (2, 1, 1, 2),
    (2, 1, 2, 2),
    (4, 1, 1, 2),
    (1, 1, 1, 4),
    (2, 1, 4, 1),
    (4, 1, 2, 1),
    (1, 2, 1, 1),
    (1, 2, 2, 1),
]


@pytest.mark.parametrize("ctx_tp,ctx_cp,gen_tp,gen_cp", _GRID_LAYOUTS)
def test_token_split_cache_reshards_over_tp_and_cp(ctx_tp, ctx_cp, gen_tp, gen_cp):
    _check_grid_resharding(TokenSplitCacheManager, ctx_tp, ctx_cp, gen_tp, gen_cp)


def test_token_split_cache_declares_its_grid():
    (rank,) = [
        r for r in _grid_instance(TokenSplitCacheManager, 2, 2, "gen") if r.instance_rank == 3
    ]
    desc = rank.pages.group(0)
    assert (desc.kind, desc.grid, desc.unit, desc.blocks_by_cp) == (
        "attention",
        ("tp", "cp"),
        "block",
        False,
    )
    (segment,) = rank.pages.buffers(0)[0].segments
    assert (segment.rows, segment.split_extent, segment.unit_bytes) == (1, 2, 8)
    assert pm.shard_grid(desc.grid, rank) == pm.ShardGrid(4, 3)
    check_pages_round_trip(rank.pages)
    # Every context-parallel rank holds every block: a helix receiver pairs all of them.
    (ctx,) = _grid_instance(TokenSplitCacheManager, 1, 1, "ctx")
    src, dst = pm.pair_units(np.arange(4), np.arange(10, 14), _pair_ctx(ctx, rank, 0))
    assert src.tolist() == [0, 1, 2, 3] and dst.tolist() == [10, 11, 12, 13]


@pytest.mark.parametrize(
    "manager_cls,ctx_tp,ctx_cp,gen_tp,gen_cp",
    [(BlockStateCacheManager, *layout) for layout in _GRID_LAYOUTS if 4 not in layout[2:]]
    + [
        (BlockStateOverTpManager, 1, 1, 1, 2),
        (BlockStateOverTpManager, 2, 1, 1, 2),
        (BlockStateOverTpManager, 2, 1, 4, 2),
        (BlockStateOverTpManager, 4, 1, 2, 1),
        # Context-parallel senders outside the grid hold the same bytes; one is elected.
        (BlockStateOverTpManager, 1, 2, 2, 1),
        (BlockStateOverTpManager, 2, 2, 1, 1),
    ],
)
def test_block_state_snapshots_reshard_by_ordinal(manager_cls, ctx_tp, ctx_cp, gen_tp, gen_cp):
    _check_grid_resharding(manager_cls, ctx_tp, ctx_cp, gen_tp, gen_cp)


def test_block_state_snapshots_are_paged_units():
    (ctx,) = _grid_instance(BlockStateCacheManager, 1, 1, "ctx")
    desc = ctx.pages.group(0)
    assert (desc.kind, desc.grid, desc.unit, desc.blocks_by_cp) == (
        "state",
        ("tp", "cp"),
        "block",
        False,
    )
    assert ctx.pages.kind(0) == CacheKind.PAGED
    check_pages_round_trip(ctx.pages)
    # Units are block ordinals: they pair ordinal by ordinal and move with every chunk.
    gen = _grid_instance(BlockStateCacheManager, 1, 2, "gen")[1]
    src, dst = pm.pair_units(np.array([5, -1, 7]), np.array([15, 16, 17]), _pair_ctx(ctx, gen, 0))
    assert src.tolist() == [5, 7] and dst.tolist() == [15, 17]
    chunk = pm.chunk_units(ctx.pages, 0, np.arange(6), ChunkContext(2, 4, False, 24))
    assert chunk.tolist() == [-1, -1, 2, 3, -1, -1]
    assert not pm.has_whole_request_groups(ctx.pages)


@pytest.mark.parametrize("manager_cls", [TokenSplitCacheManager, BlockStateCacheManager])
def test_oracle_catches_a_wrong_mapping_of_a_tp_cp_grid(monkeypatch, manager_cls):
    from tensorrt_llm._torch.disaggregation.page_mapping import correspondence

    monkeypatch.setattr(correspondence, "shard_span", lambda units, grid, virtual: (0, units))
    with pytest.raises(AssertionError):
        _check_grid_resharding(manager_cls, 2, 1, 1, 2)
