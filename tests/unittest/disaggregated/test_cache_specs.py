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
"""cache_specs: the region page table, specs claiming managers, the correspondence.

The resharding matrix checks every receiving rank byte by byte against an independent model of TP /
GQA / helix sharding (``check_resharding``), for the built-in layouts and for a test-only
spec that is added without touching anything outside its own file.
"""

import enum
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation import cache_specs as cs
from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.cache_specs.page_table import (
    BufferDecl,
    LayerGroup,
    PageTable,
    Pool,
    Region,
)
from tensorrt_llm._torch.disaggregation.cache_specs.registry import register, unregister
from tensorrt_llm._torch.disaggregation.cache_specs.spec import ChunkContext, PairContext
from tensorrt_llm._torch.disaggregation.cache_specs.specs._base import StandardSpec
from tensorrt_llm._torch.disaggregation.native.rank_info import AttentionInfo, RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1

pytestmark = pytest.mark.cpu_only

A, S = cs.ATTENTION_TP, cs.STATE_TP


# ---------------------------------------------------------------------------
# Oracle: an independent model of sharding the transfer plan is checked against
#
# On every receiving rank, every byte of every region the senders also hold must be written exactly
# once, by the elected sender, from the source byte that holds the same
# ``(layer, role, buffer, row, global unit, offset in unit)``. A rank's local unit ``u`` of a split
# axis is global unit ``rank * G // grid + u`` (``G`` the model's true extent: heads, channels, ...;
# ranks beyond ``G`` duplicate units, as GQA heads do).
# ---------------------------------------------------------------------------


GlobalUnits = Callable[[Region, int, BufferDecl], int]
"""The model's true extent of buffer ``index`` of a region along its split axis."""

ByteId = Tuple[int, str, int, int, int, int]
"""``(global layer id, role, buffer index, row, global unit, offset in unit)``."""

Address = Tuple[int, int]
"""``(pool, offset within a unit's slot)``."""


def check_page_table_round_trip(page_table: PageTable) -> None:
    restored = PageTable.from_dict(page_table.to_dict())
    if restored != page_table:
        raise AssertionError("page table does not survive to_dict/from_dict")
    cs.specs_in(restored)


def _byte_ids(region: Region, ri, global_units: GlobalUnits) -> Dict[Address, ByteId]:
    """Slot address -> identity of every byte the region holds on one rank."""
    ids: Dict[Address, ByteId] = {}
    offset = region.offset
    for index, decl in enumerate(region.buffers):
        grid = cs.shard_grid(decl.domain, ri)
        if decl.split_axis is None:
            for b in range(decl.nbytes):
                ids[(region.pool, offset + b)] = (region.layer, region.role, index, 0, 0, b)
        else:
            unit_bytes = decl.unit_bytes("oracle")
            first = (grid.rank * global_units(region, index, decl)) // grid.size
            for b in range(decl.nbytes):
                row, rest = divmod(b, decl.units * unit_bytes)
                unit, within = divmod(rest, unit_bytes)
                ids[(region.pool, offset + b)] = (
                    region.layer,
                    region.role,
                    index,
                    row,
                    first + unit,
                    within,
                )
        offset += decl.nbytes
    return ids


def _layer_group_byte_ids(
    page_table: PageTable, layer_group: int, ri, global_units
) -> Dict[Address, ByteId]:
    ids: Dict[Address, ByteId] = {}
    for region in page_table.layer_groups[layer_group].regions:
        ids.update(_byte_ids(region, ri, global_units))
    return ids


def check_resharding(
    senders: Sequence,
    receivers: Sequence,
    global_units: GlobalUnits,
    *,
    sender_dp_rank: int = 0,
) -> int:
    """Check every receiver gets each byte exactly once from the right source byte.

    ``senders`` / ``receivers`` are ``RankInfo`` objects with ``page_table`` set, one per rank of
    each instance. Senders are the ranks a receiver addresses (``rank_overlap``), as at runtime.
    Returns the number of bytes checked.
    """
    by_rank = {ri.instance_rank: ri for ri in senders}
    checked = 0
    for dst_ri in receivers:
        dst_pt = dst_ri.page_table
        writes: Counter = Counter()
        expected: Dict[Tuple[int, Address], ByteId] = {}
        overlap = cs.rank_overlap(dst_ri, senders[0], sender_dp_rank)
        for flat in overlap.ranks:
            src_ri = by_rank[flat]
            src_pt = src_ri.page_table
            src_layers = {
                (layer_group.spec, region.key)
                for layer_group in src_pt.layer_groups
                for region in layer_group.regions
            }
            for group in cs.plan_transfers(src_ri, dst_ri, src_pt, dst_pt):
                if not cs.owns_transfer(group, src_ri, dst_ri):
                    continue
                src_ids = _layer_group_byte_ids(
                    src_pt, group.self_layer_group, src_ri, global_units
                )
                dst_ids = _layer_group_byte_ids(
                    dst_pt, group.peer_layer_group, dst_ri, global_units
                )
                for s, d, n in group.plan.runs:
                    for b in range(n):
                        src_id = src_ids[(group.self_pool, s + b)]
                        dst_id = dst_ids[(group.peer_pool, d + b)]
                        if src_id != dst_id:
                            raise AssertionError(
                                f"rank {flat} -> {dst_ri.instance_rank} layer group {group.peer_layer_group}: "
                                f"byte {d + b} of pool {group.peer_pool} gets {src_id}, expected "
                                f"{dst_id}"
                            )
                        writes[(group.peer_layer_group, (group.peer_pool, d + b))] += 1
            for group_index, layer_group in enumerate(dst_pt.layer_groups):
                for region in layer_group.regions:
                    if (layer_group.spec, region.key) not in src_layers:
                        continue
                    for address, byte_id in _byte_ids(region, dst_ri, global_units).items():
                        expected[(group_index, address)] = byte_id
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


def all_ranks(make_rank: Callable[..., object], tp: int, pp: int = 1, cp: int = 1) -> List:
    """``make_rank(pp_rank=, tp_rank=, cp_rank=)`` for every rank of one instance, CP-minor."""
    return [
        make_rank(pp_rank=p, tp_rank=t, cp_rank=c)
        for p in range(pp)
        for t in range(tp)
        for c in range(cp)
    ]


# ---------------------------------------------------------------------------
# Synthetic ranks: a model, a parallel layout, and the page table one rank describes
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

    def global_units(self, region: Region, index: int, decl: BufferDecl) -> int:
        """The true extent of a split axis, for the conformance oracle."""
        if region.role == "conv_state":
            return self.conv_sections[index]
        if region.role == "ssm_state":
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


def _attention_layer_group(model, layers, kv, pools, base) -> LayerGroup:
    buf = int(model.tokens_per_block * kv * model.head_dim * model.element_bytes)
    axes = (("token", model.tokens_per_block), ("head", kv)) if model.nhd else (("head", kv),)
    kv_decl = (BufferDecl(buf, A, axes, "head"),)
    roles = ["key"] if model.mla else ["key", "value"]
    idx = (BufferDecl(model.index_key, A),) if model.index_key else None
    regions = []
    if idx and model.index_coalesced:
        stride = len(roles) * buf + model.index_key
        pools.append(Pool(base, slot_bytes=stride * len(layers), num_slots=64))
        for i, gid in enumerate(layers):
            for r, role in enumerate(roles):
                regions.append(Region(0, gid, role, i * stride + r * buf, kv_decl, i))
            regions.append(Region(0, gid, "index_key", i * stride + len(roles) * buf, idx, i))
    else:
        pools.append(Pool(base, slot_bytes=len(roles) * buf * len(layers), num_slots=64))
        for i, gid in enumerate(layers):
            for r, role in enumerate(roles):
                regions.append(Region(0, gid, role, (i * len(roles) + r) * buf, kv_decl, i))
        if idx:
            pools.append(
                Pool(base + 0x100000, slot_bytes=model.index_key * len(layers), num_slots=64)
            )
            for i, gid in enumerate(layers):
                regions.append(Region(1, gid, "index_key", i * model.index_key, idx, i))
    params = {"tokens_per_block": model.tokens_per_block, "window": None}
    return LayerGroup("attention", CacheKind.PAGED, regions, params)


def _state_layer_group(model, layers, grid_size, pools, base) -> LayerGroup:
    sections = [s // grid_size for s in model.conv_sections]
    heads = model.ssm_heads // grid_size
    roles = [
        (
            "conv_state",
            tuple(BufferDecl(s, S, (("byte", s),), "byte") for s in sections),
        ),
        (
            "ssm_state",
            (BufferDecl(heads * model.ssm_bytes_per_head, S, (("head", heads),), "head"),),
        ),
    ]
    if model.ple:
        roles.append(("ple", (BufferDecl(model.ple, S),)))
    n = len(layers)
    first_pool = len(pools)
    regions = []
    if model.interleaved_state:
        stride = sum(sum(b.nbytes for b in decls) for _, decls in roles)
        pools.append(Pool(base, slot_bytes=stride * n, num_slots=16))
        for i, gid in enumerate(layers):
            offset = i * stride
            for role, decls in roles:
                regions.append(Region(first_pool, gid, role, offset, decls, 100 + i))
                offset += sum(b.nbytes for b in decls)
    else:
        for k, (role, decls) in enumerate(roles):
            size = sum(b.nbytes for b in decls)
            pools.append(Pool(base + 0x100000 * k, slot_bytes=size * n, num_slots=16))
            for i, gid in enumerate(layers):
                regions.append(Region(first_pool + k, gid, role, i * size, decls, 100 + i))
    return LayerGroup("attention", CacheKind.STATE, regions, {})


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
        attention=AttentionInfo(
            kv_heads_per_rank=kv if attn else 0,
            tokens_per_block=model.tokens_per_block,
            dims_per_head=model.head_dim,
            element_bytes=model.element_bytes,
            enable_attention_dp=par.adp,
            is_mla=model.mla,
        ),
    )
    base = 0x10_0000_0000 if name == "ctx" else 0x20_0000_0000
    pools, layer_groups = [], []
    if attn:
        layer_groups.append(_attention_layer_group(model, attn, kv, pools, base))
    if state:
        layer_groups.append(
            _state_layer_group(model, state, cs.state_grid(ri).size, pools, base + 0x4000000)
        )
    # A hybrid manager is described by the mamba spec, attention life cycles included.
    for layer_group in layer_groups:
        layer_group.spec = "mamba" if "M" in model.pattern else "attention"
    ri.page_table = PageTable(pools=pools, layer_groups=layer_groups)
    return ri


def instance(model: Model, par: Par, name: str) -> List[RankInfo]:
    return all_ranks(lambda **at: make_rank(model, par, name=name, **at), par.tp, par.pp, par.cp)


# ---------------------------------------------------------------------------
# The page table: spec-free shape, round trip
# ---------------------------------------------------------------------------


def test_page_table_round_trips_and_dedups_layouts_on_the_wire():
    for model in (Model(), Model(pattern="AMAM", index_key=8, ple=12), Model(mla=True, kv_heads=1)):
        check_page_table_round_trip(make_rank(model, Par(tp=2)).page_table)
    table = make_rank(Model(pattern="AAAAAAAA"), Par()).page_table
    # Eight layers x K/V share one declaration.
    assert len(table.to_dict()["layer_groups"][0]["layouts"]) == 1


def test_region_and_buffer_declarations():
    nhd = BufferDecl(64, A, (("token", 4), ("head", 2)), "head")
    assert (nhd.rows, nhd.units, nhd.unit_bytes("t")) == (4, 2, 8)
    replicated = Region(0, 3, "index_key", 128, (BufferDecl(24, A),))
    assert replicated.replicated and replicated.key == (3, "index_key") and replicated.nbytes == 24
    with pytest.raises(ValueError, match="split axis"):
        BufferDecl(8, A, (("head", 2),), "token")
    with pytest.raises(ValueError, match="not byte-aligned"):
        BufferDecl(10, A, (("head", 4),), "head").unit_bytes("t")
    with pytest.raises(ValueError, match="declares no buffers"):
        Region(0, 0, "key", 0, ())


# ---------------------------------------------------------------------------
# Registry: specs claim managers by class, layer groups name their spec
# ---------------------------------------------------------------------------


def test_builtin_variants_are_registered_specs():
    names = {spec.name for spec in cs.registered_specs()}
    assert {
        "attention",
        "minimax_m3",
        "fp4_mla",
        "deepseek_v4",
        "csa2",
        "mamba",
        "qsa",
    } <= names
    layer_group = LayerGroup("csa2", CacheKind.PAGED)
    assert cs.spec_for(layer_group) is cs.spec_named("csa2")


class _BaseManager:
    pass


class _ChildManager(_BaseManager):
    pass


class _GrandChildManager(_ChildManager):
    pass


@pytest.fixture
def claiming_specs():
    @register
    class BaseKind(StandardSpec):
        name = "test_base"
        managers = (f"{__name__}._BaseManager",)

    @register
    class ChildKind(StandardSpec):
        name = "test_child"
        managers = (f"{__name__}._ChildManager",)

    yield BaseKind, ChildKind
    unregister(BaseKind)
    unregister(ChildKind)


def test_claim_picks_the_spec_nearest_in_the_mro(claiming_specs):
    assert cs.claim(_BaseManager()).name == "test_base"
    assert cs.claim(_ChildManager()).name == "test_child"
    assert cs.claim(_GrandChildManager()).name == "test_child"
    # A manager no spec names by class is described as plain paged attention.
    assert cs.claim(SimpleNamespace()).name == "attention"


def test_two_specs_cannot_claim_one_manager_class(claiming_specs):
    class Duplicate(StandardSpec):
        name = "test_duplicate"
        managers = (f"{__name__}._ChildManager",)

    with pytest.raises(ValueError, match="already claimed"):
        register(Duplicate)


def test_unknown_spec_fails_loudly():
    table = make_rank(Model(), Par()).page_table
    alien = PageTable.from_dict(table.to_dict())
    alien.layer_groups[0].spec = "alien"
    for call in (
        lambda: cs.match_regions(table, alien),
        lambda: cs.specs_in(alien),
        lambda: cs.validate_peer_compatible(None, None, table, alien),
    ):
        with pytest.raises(ValueError, match="unknown cache spec 'alien'"):
            call()


# ---------------------------------------------------------------------------
# Describing a KVCacheManagerV2
# ---------------------------------------------------------------------------


def _buffer(layer, role):
    return SimpleNamespace(layer_id=layer, role=role)


def _v2_manager(*, life_cycles, pool_groups, ignored=(), windows=None):
    """A KVCacheManagerV2 as the page-table builder reads it.

    ``pool_groups``: per pool group, ``(num_slots, [(base, slot_bytes)], variants)`` with variants
    ``[(life_cycle, [(single_buffer_size, [(layer, role)])])]``.
    """
    layers = sorted({i for ids in life_cycles for i in ids})
    windows = windows or {}
    return SimpleNamespace(
        pp_layers=[10 + i for i in layers],
        num_kv_heads_per_layer=[2] * len(layers),
        get_disagg_ignored_roles=lambda: frozenset(ignored),
        impl=SimpleNamespace(
            layer_grouping=life_cycles,
            init_config=SimpleNamespace(
                tokens_per_block=4,
                layers=[SimpleNamespace(window_size=windows.get(i)) for i in layers],
            ),
            pool_group_descs=[
                SimpleNamespace(
                    num_slots=num_slots,
                    pools=[SimpleNamespace(base_address=b, slot_bytes=n) for b, n in pools],
                    slot_desc=SimpleNamespace(
                        variants=[
                            SimpleNamespace(
                                layer_group_id=lc,
                                coalesced_buffers=[
                                    SimpleNamespace(
                                        single_buffer_size=size,
                                        buffer_ids=[_buffer(lid, role) for lid, role in ids],
                                    )
                                    for size, ids in coalesced
                                ],
                            )
                            for lc, coalesced in variants
                        ]
                    ),
                )
                for num_slots, pools, variants in pool_groups
            ],
        ),
    )


def test_v2_life_cycles_become_spaces_of_regions():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    key, value, index = str(Role.KEY), str(Role.VALUE), str(Role.INDEX_KEY)
    # Two life cycles (full / SWA) share one pool group; index-K is coalesced into the K/V pool and
    # a padding role occupies slot bytes without being transferred.
    manager = _v2_manager(
        life_cycles=[[0, 2], [1]],
        windows={1: 64},
        ignored={"padding"},
        pool_groups=[
            (
                32,
                [(0x1000, 4096)],
                [
                    (0, [(64, [(0, Role.KEY), (0, Role.VALUE), (0, index), (2, Role.KEY)])]),
                    (1, [(64, [(1, Role.KEY), (1, "padding"), (1, Role.VALUE)])]),
                ],
            )
        ],
    )
    table = cs.build_page_table(manager)
    assert [s.spec for s in table.layer_groups] == ["attention", "attention"]
    assert [s.kind for s in table.layer_groups] == [CacheKind.PAGED, CacheKind.PAGED]
    assert table.layer_groups[0].params == {"tokens_per_block": 4, "window": None}
    assert table.layer_groups[1].params == {"tokens_per_block": 4, "window": 64}
    assert [(r.pool, r.layer, r.role, r.offset) for r in table.layer_groups[0].regions] == [
        (0, 10, key, 0),
        (0, 10, value, 64),
        (0, 10, index, 128),
        (0, 12, key, 192),
    ]
    # The padding buffer keeps its slot offset but has no region.
    assert [(r.role, r.offset) for r in table.layer_groups[1].regions] == [(key, 0), (value, 128)]
    kv_decl = BufferDecl(64, A, (("head", 2),), "head")
    assert table.layer_groups[0].regions[0].buffers == (kv_decl,)
    assert table.layer_groups[0].regions[2].buffers == (BufferDecl(64, A),)
    assert table.pools == [Pool(0x1000, 4096, 32)]


def test_v1_managers_are_rejected():
    with pytest.raises(ValueError, match="not a KVCacheManagerV2"):
        cs.build_page_table(SimpleNamespace(tokens_per_block=4))


def _role_buffers(spec, manager, unit, role, nbytes=64):
    return cs.spec_named(spec).role_buffers(
        manager, unit, role, nbytes, heads=2, tokens_per_block=4, context="test"
    )


def test_specs_own_their_role_layouts():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    paged, key = CacheKind.PAGED, str(Role.KEY)
    hnd = (BufferDecl(64, A, (("head", 2),), "head"),)
    nhd = (BufferDecl(64, A, (("token", 4), ("head", 2)), "head"),)
    replicated = (BufferDecl(64, A),)
    plain = SimpleNamespace()
    assert _role_buffers("attention", plain, paged, key) == hnd
    assert _role_buffers("attention", plain, paged, str(Role.INDEX_KEY)) == replicated
    assert _role_buffers("csa2", plain, paged, key) == replicated
    assert _role_buffers("fp4_mla", plain, paged, key) == nhd
    assert _role_buffers("fp4_mla", plain, paged, str(Role.MLA_HP_TAIL)) == replicated
    for layout, expected in (("HND", hnd), ("NHD", nhd)):
        minimax = SimpleNamespace(_main_kv_layout_name=lambda layout=layout: layout)
        assert _role_buffers("minimax_m3", minimax, paged, key) == expected
    # A state role with no declared geometry is held whole by every rank of the state grid.
    assert _role_buffers("mamba", plain, CacheKind.STATE, "kda_extra") == (BufferDecl(64, S),)


def test_mamba_describes_attention_and_state_life_cycles():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    conv, ssm = "conv_state", "ssm_state"
    manager = _v2_manager(
        life_cycles=[[0], [1, 2]],
        pool_groups=[
            (8, [(0x1000, 128)], [(0, [(64, [(0, Role.KEY), (0, Role.VALUE)])])]),
            (
                4,
                [(0x9000, 64), (0xA000, 256)],
                [(1, [(32, [(1, conv), (2, conv)]), (128, [(1, ssm), (2, ssm)])])],
            ),
        ],
    )
    manager._is_local_mamba_layer = lambda lid: lid != 0
    manager.local_num_mamba_layers = 2
    # Conv sections [x | B | C] of 4, 2, 2 channels x 2 taps x 2 bytes; 2 x 4 x 4-byte SSM heads.
    manager.conv_section_dims = (4, 2, 2)
    manager.conv_state_shape = (8, 2)
    manager.conv_state_dtype = SimpleNamespace(itemsize=2)
    manager.ssm_state_shape = (4, 2, 4)
    manager.ssm_state_dtype = SimpleNamespace(itemsize=4)
    manager._request_id_to_state_index = {7: 3}
    table = cs.spec_named("mamba").describe(manager)
    attention, state = table.layer_groups
    assert (attention.spec, attention.kind) == ("mamba", CacheKind.PAGED)
    assert (state.spec, state.kind, state.params) == ("mamba", CacheKind.STATE, {})
    conv_region = state.regions[0]
    assert (conv_region.pool, conv_region.role) == (1, conv)
    assert [b.nbytes for b in conv_region.buffers] == [16, 8, 8]
    assert state.regions[2].buffers == (BufferDecl(128, S, (("head", 4),), "head"),)
    spec = cs.spec_for(state)
    assert not spec.sends_incrementally(state) and spec.sends_incrementally(attention)
    ctx = cs.LocalUnitsContext(SimpleNamespace(py_request_id=7), 1, state, manager, None, 12, 0)
    assert spec.local_units(ctx).tolist() == [3]

    manager.conv_section_dims = (4, 2, 1)
    with pytest.raises(ValueError, match="do not sum to the role's 32 bytes"):
        cs.spec_named("mamba").describe(manager)


class _AttnType(enum.Enum):
    SWA = 0
    COMPRESS = 1


class _CacheRole(enum.Enum):
    SWA = "swa"
    GLOBAL = "global"


def _virtual_manager():
    """Two internal layers (SWA, compressed) per model layer 4 and 5 on this PP stage."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    manager = _v2_manager(
        life_cycles=[[0, 2], [1, 3]],
        pool_groups=[
            (
                8,
                [(0x1000, 256)],
                [
                    (0, [(64, [(0, Role.KEY), (2, Role.KEY)])]),
                    (1, [(64, [(1, Role.KEY), (3, Role.KEY)])]),
                ],
            )
        ],
    )
    manager.layer_offsets = {4: 0, 5: 1}
    manager.impl.init_config.layers = [SimpleNamespace(window_size=128)]
    manager._layer_attn_to_layer_id = {
        (4, _AttnType.SWA): 0,
        (4, _AttnType.COMPRESS): 1,
        (5, _AttnType.SWA): 2,
        (5, _AttnType.COMPRESS): 3,
    }
    manager._physical_roles = {
        (0, Role.KEY): (4, 1, _CacheRole.SWA),
        (2, Role.KEY): (5, 1, _CacheRole.SWA),
        (1, Role.KEY): (4, 4, _CacheRole.GLOBAL),
        (3, Role.KEY): (5, 4, _CacheRole.GLOBAL),
    }
    return manager


@pytest.mark.parametrize("spec", ["deepseek_v4", "csa2"])
def test_virtual_layers_get_pp_independent_global_ids(spec):
    table = cs.spec_named(spec).describe(_virtual_manager())
    swa, compressed = table.layer_groups
    # model_layer * num_attention_types + attention_type
    assert [r.layer for r in swa.regions] == [8, 10]
    assert [r.layer for r in compressed.regions] == [9, 11]
    # Internal layer 1 is past the per-layer config: its window resolves through model layer 4.
    assert swa.params["window"] == 128 and compressed.params["window"] == 128


def test_csa2_remote_tail_rules():
    manager = _virtual_manager()
    table = cs.spec_named("csa2").describe(manager)
    tail = SimpleNamespace(
        prompt_len=40,
        py_csa2_remote_tail_start=24,
        py_csa2_remote_tail_split=5,
        py_csa2_remote_tail_mode="source",
    )
    plain = SimpleNamespace(prompt_len=40)
    assert cs.transfer_length(table, tail) == 24 and cs.transfer_length(table, plain) == 40
    assert not cs.sends_first_tokens(table, tail) and cs.sends_first_tokens(table, plain)
    # Decoder SWA state of model layer 5 (>= split) is rebuilt by the generation side.
    assert cs.excluded_regions(manager, table, tail) == {(0, 1)}
    assert cs.excluded_regions(manager, table, plain) == frozenset()


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_shard_span_duplicates_units_on_the_finer_grid():
    # 2 global heads over TP=4: ranks 0, 1 hold head 0 and ranks 2, 3 head 1.
    grid = [cs.ShardGrid(4, r) for r in range(4)]
    spans = [cs.topology.shard_span(1, g, virtual_units=2) for g in grid]
    assert spans == [(0, 1), (0, 1), (1, 2), (1, 2)]
    # Two ranks hold the head one TP=1 peer needs: exactly one of them is elected.
    factor = cs.topology.replication_factor(1, grid[0], 2, cs.ShardGrid(1, 0))
    assert factor == 2
    assert [cs.topology.elects_sender(g, 0, factor) for g in grid] == [True, False, True, False]


def test_state_grid_follows_attention_dp_and_helix():
    model, par = Model(pattern="MMMM"), Par(tp=2, cp=2)
    assert cs.state_grid(make_rank(model, par, tp_rank=1, cp_rank=1)) == cs.ShardGrid(4, 3)
    assert cs.state_grid(make_rank(model, Par(tp=4, adp=True), tp_rank=3)) == cs.ShardGrid(1, 0)
    assert cs.attention_grid(make_rank(model, Par(tp=4, adp=True), tp_rank=3)) == cs.ShardGrid(1, 0)


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
            for group in cs.plan_transfers(src, dst, src.page_table, dst.page_table)
            if any(
                src.page_table.layer_groups[group.self_layer_group].regions[r].role == "ple"
                for r in group.regions
            )
            and cs.owns_transfer(group, src, dst)
        ]
        assert len(senders) == 1, dst.instance_rank


# ---------------------------------------------------------------------------
# Matching, transfer groups and peer validation
# ---------------------------------------------------------------------------


def test_regions_match_by_spec_layer_and_role():
    ctx = make_rank(Model(pattern="AMAM", index_key=8), Par(pp=2), pp_rank=1)
    gen = make_rank(Model(pattern="AMAM", index_key=8), Par(), name="gen")
    mapping = cs.match_regions(ctx.page_table, gen.page_table)
    for (gi, ri), (pgi, pri) in mapping.items():
        mine = ctx.page_table.layer_groups[gi].regions[ri]
        theirs = gen.page_table.layer_groups[pgi].regions[pri]
        assert mine.key == theirs.key
    # Every region of the PP stage (layers 2, 3) has a counterpart; the peer's layers 0, 1 do not.
    assert len(mapping) == sum(len(s.regions) for s in ctx.page_table.layer_groups)

    dup = PageTable.from_dict(gen.page_table.to_dict())
    dup.layer_groups[0].regions.append(dup.layer_groups[0].regions[0])
    with pytest.raises(ValueError, match="holds each \\(layer, role\\) once"):
        cs.match_regions(ctx.page_table, dup)


def test_transfer_groups_merge_contiguous_regions_and_skip_excluded_ones():
    ctx = make_rank(Model(), Par(), name="ctx")
    gen = make_rank(Model(), Par(), name="gen")
    (group,) = cs.plan_transfers(ctx, gen, ctx.page_table, gen.page_table)
    # Equal layouts: every layer's K and V are contiguous on both sides, one run moves the slot.
    slot = ctx.page_table.pools[0].slot_bytes
    assert group.plan.runs == [(0, 0, slot)] and group.owner_factor == 1
    excluded = frozenset({(0, 0), (0, 1)})  # layer 0's K and V
    (group,) = cs.plan_transfers(ctx, gen, ctx.page_table, gen.page_table, excluded)
    assert group.plan.runs == [(slot // 4, slot // 4, slot * 3 // 4)]
    assert group.regions == tuple(range(2, 8))


def test_peer_layout_mismatches_are_rejected():
    ctx = make_rank(Model(pattern="MMMM"), Par(tp=2))
    gen = make_rank(Model(pattern="MMMM", ssm_heads=4), Par(tp=1), name="gen")
    with pytest.raises(ValueError, match="TP-aggregated"):
        cs.validate_peer_compatible(ctx, gen, ctx.page_table, gen.page_table)

    nhd = make_rank(Model(nhd=True), Par(), name="gen")
    hnd = make_rank(Model(), Par())
    with pytest.raises(ValueError, match="layout differs"):
        cs.validate_peer_compatible(hnd, nhd, hnd.page_table, nhd.page_table)

    with_index = make_rank(Model(index_key=8), Par(), name="gen")
    with pytest.raises(ValueError, match="replicated roles differ"):
        cs.validate_peer_compatible(hnd, with_index, hnd.page_table, with_index.page_table)

    other_spec = make_rank(Model(), Par(), name="gen")
    for layer_group in other_spec.page_table.layer_groups:
        layer_group.spec = "minimax_m3"
    with pytest.raises(ValueError, match="cache specs differ"):
        cs.validate_peer_compatible(hnd, other_spec, hnd.page_table, other_spec.page_table)


# ---------------------------------------------------------------------------
# Selection and whole-table questions
# ---------------------------------------------------------------------------


def _pair_ctx(self_ri, peer_ri, layer_group, dst_start_token=None):
    return PairContext(
        self_ri=self_ri, peer_ri=peer_ri, layer_group=layer_group, dst_start_token=dst_start_token
    )


def test_paged_units_pair_by_ordinal_and_state_by_slot():
    ri = make_rank(Model(pattern="AMAM"), Par())
    paged, state = ri.page_table.layer_groups
    spec = cs.spec_for(paged)
    src, dst = spec.pair_units(
        np.array([10, 11, -1, 13]), np.array([-1, 21, 22, 23]), _pair_ctx(ri, ri, paged)
    )
    assert src.tolist() == [11, 13] and dst.tolist() == [21, 23]
    src, dst = spec.pair_units(
        np.array([10, 11, 12, 13]), np.array([20, 21, 22, 23]), _pair_ctx(ri, ri, paged, 8)
    )
    assert src.tolist() == [12, 13] and dst.tolist() == [22, 23]
    src, dst = spec.pair_units(np.array([5]), np.array([7]), _pair_ctx(ri, ri, state))
    assert src.tolist() == [5] and dst.tolist() == [7]
    empty = np.array([], dtype=np.int64)
    assert spec.pair_units(np.array([5]), empty, _pair_ctx(ri, ri, state))[0].size == 0


def test_helix_receiver_takes_its_strided_blocks():
    ctx = make_rank(Model(), Par(tp=1))
    gen = make_rank(Model(), Par(tp=1, cp=2), name="gen", cp_rank=1)
    layer_group = ctx.page_table.layer_groups[0]
    src, dst = cs.spec_for(layer_group).pair_units(
        np.array([10, 11, 12, 13]), np.array([21, 23]), _pair_ctx(ctx, gen, layer_group)
    )
    assert src.tolist() == [11, 13] and dst.tolist() == [21, 23]


def test_chunk_units_defer_windows_and_whole_request_spaces():
    spec = cs.spec_named("attention")
    units = np.arange(6)
    full = LayerGroup("attention", CacheKind.PAGED, params={"tokens_per_block": 4, "window": None})
    swa = LayerGroup("attention", CacheKind.PAGED, params={"tokens_per_block": 4, "window": 8})
    state = LayerGroup("attention", CacheKind.STATE)

    def chunk(layer_group, values, is_last):
        return spec.chunk_units(
            layer_group, values, ChunkContext(2, 4, is_last, prompt_len=24)
        ).tolist()

    assert chunk(full, units, False) == [-1, -1, 2, 3, -1, -1]
    assert chunk(swa, units, False) == [-1] * 6
    assert chunk(swa, units, True) == units.tolist()
    assert chunk(state, np.array([5]), False) == [-1]
    assert chunk(state, np.array([5]), True) == [5]
    table = make_rank(Model(pattern="AMAM"), Par()).page_table
    assert cs.whole_request_specs(table) == ["mamba"]


def test_chunk_bytes_count_regions_not_slots():
    table = make_rank(Model(pattern="AMAM", kv_heads=4, ple=12), Par()).page_table
    attn_bytes = sum(r.nbytes for r in table.layer_groups[0].regions)
    state_bytes = sum(r.nbytes for r in table.layer_groups[1].regions)
    units = [np.array([3, -1, 4]), np.array([9])]
    assert cs.chunk_transfer_bytes(table, units) == 2 * attn_bytes + state_bytes
    ple_regions = {(1, i) for i, r in enumerate(table.layer_groups[1].regions) if r.role == "ple"}
    assert cs.chunk_transfer_bytes(table, units, ple_regions) == (
        2 * attn_bytes + state_bytes - 2 * 12
    )
    assert cs.receiver_extra_bytes(table, table, units) == state_bytes
    assert cs.bounce_unit_bytes_per_layer_group(table) == [table.pools[0].slot_bytes, None]
    assert cs.resume_after_receive(table, SimpleNamespace(prompt_len=37)) == 37


def test_memory_registration_dedups_shared_state_allocation():
    table = make_rank(Model(pattern="MMMM", ple=12, interleaved_state=True), Par()).page_table
    descs = cs.memory_registration_descs(table, device_id=2)
    assert len(descs) == 1
    pool = table.pools[0]
    assert descs[0][:3] == (pool.base_address, pool.num_slots * pool.slot_stride_bytes, 2)


# ---------------------------------------------------------------------------
# The sender path end to end: Sender._build_kv_write_meta over real page tables
# ---------------------------------------------------------------------------


def test_sender_write_meta_covers_paged_blocks_and_state_slot():
    from tensorrt_llm import DisaggregatedParams
    from tensorrt_llm._torch.disaggregation.base import Chunk, TokenRange
    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    from tensorrt_llm._torch.disaggregation.native.transfer import KVSendTask, RecvReqInfo, Sender

    model = Model(pattern="AMAM", kv_heads=4, ple=12)
    ctx = make_rank(model, Par(tp=2))
    gen = make_rank(model, Par(tp=1), name="gen")
    registrar = PeerRegistrar(ctx, KVRegionExtractorV1(ctx.page_table))
    registrar.register(gen.instance_name, gen.instance_rank, gen)
    sender = Sender.__new__(Sender)
    sender._registrar = registrar

    task = KVSendTask(
        Chunk(
            block_ids_per_layer_groups=[np.array([10, -1, 12]), np.array([3])],
            kind_per_layer_group=[s.kind for s in ctx.page_table.layer_groups],
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
    for group in registrar.transfer_groups(gen):
        if not cs.owns_transfer(group, ctx, gen):
            continue
        src_pool = ctx.page_table.pools[group.self_pool]
        dst_pool = gen.page_table.pools[group.peer_pool]
        pairs = [(10, 20), (12, 22)] if group.self_layer_group == 0 else [(3, 5)]
        for s_unit, d_unit in pairs:
            for s, d, n in group.plan.runs:
                expected.append(
                    (
                        src_pool.base_address + s_unit * src_pool.slot_stride_bytes + s,
                        dst_pool.base_address + d_unit * dst_pool.slot_stride_bytes + d,
                        n,
                    )
                )
    got = sorted(zip(meta.src_ptrs.tolist(), meta.dst_ptrs.tolist(), meta.sizes.tolist()))
    assert got == sorted(expected)
    assert meta.expected_transfers == 1  # this ctx rank addresses the single TP=1 receiver
    # TP rank 0 of 2 is the elected sender of the replicated side state, and it sends the state
    # once, into the receiver's slot 5.
    ple_pool = gen.page_table.pools[
        next(r.pool for r in gen.page_table.layer_groups[1].regions if r.role == "ple")
    ]
    slot5 = ple_pool.base_address + 5 * ple_pool.slot_stride_bytes
    assert sum(n for _, d, n in got if slot5 <= d < slot5 + ple_pool.slot_bytes) == 2 * 12


# ---------------------------------------------------------------------------
# A new spec is one file: a linear-attention state added in this test alone
# ---------------------------------------------------------------------------


class LinearStateSpec(StandardSpec):
    """Gated linear attention: q/k/v conv channels plus a [head, d_k, d_v] state, one slot each.

    It names its own axes; nothing outside this declaration (page table, correspondence, other
    specs, managers) knows about it. Selection reuses the state-unit rules.
    """

    name = "linear_state"


@dataclass(frozen=True)
class LinearModel:
    layers: int = 3
    sections: Tuple[int, ...] = (32, 32, 64)  # q | k | v channels' bytes, global
    heads: int = 8
    d_k: int = 4
    d_v: int = 8  # one head's state is d_k x d_v bytes (1-byte elements)

    def global_units(self, region: Region, index: int, decl: BufferDecl) -> int:
        return self.heads if region.role == "recurrent" else self.sections[index]


def _linear_rank(model: LinearModel, par: Par, name: str, tp_rank: int) -> RankInfo:
    grid = par.tp
    sections = [s // grid for s in model.sections]
    heads = model.heads // grid
    conv = tuple(BufferDecl(s, S, (("channel", s),), "channel") for s in sections)
    dims = (("head", heads), ("d_k", model.d_k), ("d_v", model.d_v))
    state = (BufferDecl(heads * model.d_k * model.d_v, S, dims, "head"),)
    conv_bytes, state_bytes = sum(sections), state[0].nbytes
    base = 0x30_0000_0000 if name == "ctx" else 0x40_0000_0000
    n = model.layers
    regions = [Region(0, i, "qkv_conv", i * conv_bytes, conv, i) for i in range(n)]
    regions += [Region(1, i, "recurrent", i * state_bytes, state, i) for i in range(n)]
    table = PageTable(
        pools=[
            Pool(base, slot_bytes=conv_bytes * n, num_slots=8),
            Pool(base + 0x100000, slot_bytes=state_bytes * n, num_slots=8),
        ],
        layer_groups=[LayerGroup("linear_state", CacheKind.STATE, regions)],
    )
    return RankInfo(
        instance_name=name,
        instance_rank=tp_rank,
        tp_size=par.tp,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[n],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        page_table=table,
    )


@pytest.fixture
def linear_state_spec():
    register(LinearStateSpec)
    yield
    unregister(LinearStateSpec)


@pytest.mark.parametrize("ctx_tp,gen_tp", [(1, 1), (1, 4), (4, 1), (2, 4), (4, 2)])
def test_new_spec_reshards_with_no_change_outside_its_file(linear_state_spec, ctx_tp, gen_tp):
    model = LinearModel()
    ctx = [_linear_rank(model, Par(tp=ctx_tp), "ctx", r) for r in range(ctx_tp)]
    gen = [_linear_rank(model, Par(tp=gen_tp), "gen", r) for r in range(gen_tp)]
    check_page_table_round_trip(ctx[0].page_table)
    for dst in gen:
        cs.validate_peer_compatible(dst, ctx[0], dst.page_table, ctx[0].page_table)
    assert check_resharding(ctx, gen, model.global_units) > 0
    assert cs.memory_registration_descs(ctx[0].page_table, 0)


def test_new_spec_is_unknown_until_registered():
    table = _linear_rank(LinearModel(), Par(tp=1), "ctx", 0).page_table
    with pytest.raises(ValueError, match="unknown cache spec 'linear_state'"):
        cs.match_regions(table, table)
