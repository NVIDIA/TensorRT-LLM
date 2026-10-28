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
"""``layout_mapping.map_buffers`` against the native transfer, byte for byte.

For the same cache managers, every context rank writes the same bytes to the same places of every
generation rank whether native's mappers place them, from the ``get_disagg_*`` declarations, or
the layout mapping does, from the managers' page tables. The managers are the real classes over a
stand-in for the runtime storage, so both sides read the declarations those classes make.
"""

import itertools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaV2CacheLayoutPolicy
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
    _CSA2LayerKind,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    COMPRESS_BLOCK_SCALE_ROLE,
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3KVCacheManagerV2
from tensorrt_llm._torch.disaggregation.native.extent import Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import RecvReqInfo, Sender
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import (
    PLE_CONV_STATE,
    Qwen4ExpHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import (
    layer_group_of,
    map_buffers,
    merge_runs,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManagerV2,
    MambaRole,
)
from tensorrt_llm.runtime import kv_cache_manager_v2 as rt

TOKENS_PER_BLOCK = 4
HEAD_DIM = 8
ELEMENT_BYTES = 2
PAGES = 4
SLOTS = 16
POOL_SPACING = 1 << 24
# A storage-only buffer DeepSeek-V4 keeps for alignment, which no transfer moves.
DSV4_PADDING_ROLE = rt.DataRole("deepseek_v4_test_padding")


@dataclass(frozen=True)
class Model:
    """A cache to transfer: which manager keeps it and what its layers hold.

    Attributes:
        name: Test id.
        kind: "attention", "minimax" (index-K beside K/V, head- or token-major), "mamba",
            "qwen4exp" (Mamba with replicated PLE state), "fp4_mla", "csa2" or "dsv4".
        layers: Model layers.
        heads: KV heads over all tensor ranks.
        mla: One latent per token (K only, one head).
        index: Index-K "separate" (its own size) or "coalesced" (the size of a TP2 K/V page,
            sharing their pool where the sizes meet).
        nhd: K/V token-major.
        vswa: Every other layer windowed.
        shared_pool_group: Windowed and full layers draw slots from one pool group.
    """

    name: str
    kind: str = "attention"
    layers: int = 4
    heads: int = 8
    mla: bool = False
    index: Optional[str] = None
    nhd: bool = False
    vswa: bool = False
    shared_pool_group: bool = False


@dataclass(frozen=True)
class Topology:
    """An instance's parallel sizes; ``adp`` for attention data parallelism."""

    tp: int = 1
    pp: int = 1
    cp: int = 1
    adp: bool = False

    def __str__(self) -> str:
        parts = [f"tp{self.tp}", f"pp{self.pp}"] + ([f"cp{self.cp}"] if self.cp > 1 else [])
        return "_".join(parts + (["adp"] if self.adp else []))

    def ranks(self):
        """(pp, tp, cp) of each rank, in rank order (context-parallel minor)."""
        return itertools.product(range(self.pp), range(self.tp), range(self.cp))


MODELS = [
    Model("mha"),
    Model("gqa", heads=2),
    Model("mla", mla=True),
    Model("vswa", layers=6, vswa=True),
    Model("vswa_shared_pool_group", layers=6, vswa=True, shared_pool_group=True),
    Model("pp_uneven", layers=5),
    Model("index_k_separate", kind="minimax", index="separate"),
    Model("index_k_coalesced", kind="minimax", index="coalesced"),
    Model("nhd_index_k_coalesced", kind="minimax", index="coalesced", nhd=True),
    Model("nhd_gqa", kind="minimax", heads=2, nhd=True),
    Model("mamba", kind="mamba", layers=6),
    Model("mamba_gqa", kind="mamba", layers=4, heads=2),
    Model("qwen4exp_ple", kind="qwen4exp", layers=6),
    Model("fp4_mla", kind="fp4_mla"),
    Model("csa2", kind="csa2"),
    Model("dsv4", kind="dsv4"),
]
CONTEXT_TOPOLOGIES = [
    Topology(),
    Topology(tp=2),
    Topology(tp=4),
    Topology(pp=2),
    Topology(tp=2, pp=2),
    Topology(tp=2, adp=True),
    Topology(tp=4, adp=True),
]
# Helix: a generation side splitting context ranks receives from a context side that does not.
GENERATION_TOPOLOGIES = CONTEXT_TOPOLOGIES + [Topology(cp=2), Topology(tp=2, cp=2)]


class _KeptWindow:
    """Reads each layer group's window as the window it keeps.

    That is what a manager declaring no narrower read window answers; the runtime life cycles the
    base manager reads it from are not built here.
    """

    def _attention_read_window(self, layer_group_id: int) -> Optional[int]:
        first = int(self.impl.layer_grouping[layer_group_id][0])
        return self.kv_cache_manager_py_config.layers[first].window_size


class _Attention(_KeptWindow, KVCacheManagerV2):
    pass


class _MiniMaxM3(_KeptWindow, MiniMaxM3KVCacheManagerV2):
    pass


class _Mamba(_KeptWindow, MambaHybridCacheManagerV2):
    pass


class _Qwen4Exp(_KeptWindow, Qwen4ExpHybridCacheManager):
    pass


class _Csa2(_KeptWindow, CSA2CacheManager):
    pass


class _DeepseekV4(_KeptWindow, DeepseekV4CacheManager):
    pass


def _split_layers(layers: int, pp: int) -> list[int]:
    return [layers // pp + (1 if i < layers % pp else 0) for i in range(pp)]


def _attention_layer(layer_id: int, buffers: list, window: Optional[int] = None):
    return rt.AttentionLayerConfig(
        layer_id=layer_id,
        buffers=[rt.BufferConfig(role=role, size=size) for role, size in buffers],
        sliding_window_size=window,
    )


def _state_layer(layer_id: int, buffers: list):
    return rt.SsmLayerConfig(
        layer_id=layer_id, buffers=[rt.BufferConfig(role=role, size=size) for role, size in buffers]
    )


def _storage(configs: dict, groups: list[list[int]], share: bool, base: int) -> SimpleNamespace:
    """The runtime storage a manager reads, for layer ``configs`` by internal id in ``groups``.

    Per pool group, one variant per layer group drawing from it, whose pool ``i`` holds the
    ``i``-th distinct buffer size, its buffers back to back. Attention groups share one pool group
    when ``share``; every other group has its own.
    """
    state = {g for g, ids in enumerate(groups) if isinstance(configs[ids[0]], rt.SsmLayerConfig)}
    attention = [g for g in range(len(groups)) if g not in state]
    plan = [[g] for g in range(len(groups)) if not (share and g in attention)]
    if share and attention:
        plan.insert(0, attention)
    descs, address = [], base
    for index, members in enumerate(plan):
        variants = []
        for g in members:
            by_size: dict[int, list] = {}
            for layer_id in groups[g]:
                for buffer in configs[layer_id].buffers:
                    by_size.setdefault(int(buffer.size), []).append(
                        SimpleNamespace(layer_id=layer_id, role=buffer.role)
                    )
            variants.append(
                SimpleNamespace(
                    layer_group_id=g,
                    coalesced_buffers=[
                        SimpleNamespace(single_buffer_size=size, buffer_ids=ids)
                        for size, ids in by_size.items()
                    ],
                )
            )
        pools = []
        for i in range(max(len(v.coalesced_buffers) for v in variants)):
            slot_bytes = max(
                v.coalesced_buffers[i].single_buffer_size * len(v.coalesced_buffers[i].buffer_ids)
                for v in variants
                if i < len(v.coalesced_buffers)
            )
            pools.append(SimpleNamespace(pool_index=i, base_address=address, slot_bytes=slot_bytes))
            address += POOL_SPACING
        descs.append(
            SimpleNamespace(
                pool_group_index=index,
                pools=pools,
                num_slots=SLOTS,
                slot_desc=SimpleNamespace(variants=variants),
            )
        )
    pool_group_of = [next(i for i, m in enumerate(plan) if g in m) for g in range(len(groups))]
    return SimpleNamespace(
        layer_grouping=[list(ids) for ids in groups],
        pool_group_descs=descs,
        init_config=SimpleNamespace(
            tokens_per_block=TOKENS_PER_BLOCK, layers=[configs[i] for i in sorted(configs)]
        ),
        get_life_cycle_pool_group_indices=lambda: pool_group_of,
    )


def _wire(manager, topology, pp, tp, cp, layers, configs, groups, base, heads, model) -> None:
    """Set what the page-table builders read of a manager on rank (pp, tp, cp).

    Its layers keep ``heads`` KV heads each: one count per layer, or one for all.
    """
    manager.impl = _storage(configs, groups, model.shared_pool_group, base)
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=[configs[i] for i in sorted(configs)]
    )
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.pp_layers = layers
    manager.layer_offsets = {layer: i for i, layer in enumerate(layers)}
    manager.num_kv_heads = 1 if _single_latent(model) else model.heads
    manager.num_kv_heads_per_layer = heads if isinstance(heads, list) else [heads] * len(layers)
    manager.mapping = SimpleNamespace(
        tp_size=topology.tp,
        tp_rank=tp,
        pp_size=topology.pp,
        pp_rank=pp,
        cp_size=topology.cp,
        cp_rank=cp,
        dp_size=1,
        enable_attention_dp=topology.adp,
    )
    manager._helix_cp_rank, manager._helix_cp_size = cp, topology.cp
    manager.dtype = None


def _single_latent(model: Model) -> bool:
    """Whether the model caches one latent per token (one KV head) rather than K/V by head."""
    return model.mla or model.kind in ("fp4_mla", "csa2", "dsv4")


def _heads_per_rank(model: Model, topology: Topology) -> int:
    if _single_latent(model):
        return 1
    return model.heads if topology.adp else max(1, model.heads // topology.tp)


def _kv_buffers(model: Model, heads: int) -> list:
    page = TOKENS_PER_BLOCK * heads * HEAD_DIM * ELEMENT_BYTES
    return [(Role.KEY, page)] + ([] if model.mla else [(Role.VALUE, page)])


def _build_attention(model, topology, pp, tp, cp, layers, base):
    heads = _heads_per_rank(model, topology)
    if model.kind == "minimax":
        manager = _MiniMaxM3.__new__(_MiniMaxM3)
        manager._main_kv_mapper_kind = MapperKind.NHD if model.nhd else MapperKind.HND
    else:
        manager = _Attention.__new__(_Attention)
    configs, full, windowed = {}, [], []
    for i, layer in enumerate(layers):
        buffers = _kv_buffers(model, heads)
        if model.index == "separate":
            buffers.append((Role.INDEX_KEY, TOKENS_PER_BLOCK * 10))
        elif model.index == "coalesced":
            buffers.append(
                (Role.INDEX_KEY, TOKENS_PER_BLOCK * (model.heads // 2) * HEAD_DIM * ELEMENT_BYTES)
            )
        window = 2 * TOKENS_PER_BLOCK if model.vswa and layer % 2 else None
        configs[i] = _attention_layer(i, buffers, window)
        (windowed if window else full).append(i)
    groups = [ids for ids in (full, windowed) if ids]
    _wire(manager, topology, pp, tp, cp, layers, configs, groups, base, heads, model)
    return manager, heads


def _build_mamba(model, topology, pp, tp, cp, layers, base):
    """Attention and Mamba layers interleaved (A M A M M A).

    Convolution sections and SSM heads split over tensor times context ranks, over none under
    attention data parallelism.
    """
    heads = _heads_per_rank(model, topology)
    cls = _Qwen4Exp if model.kind == "qwen4exp" else _Mamba
    manager = cls.__new__(cls)
    grid = 1 if topology.adp else topology.tp * topology.cp
    manager.conv_section_dims = [64 // grid, 32 // grid, 32 // grid]
    manager.conv_state_shape = (sum(manager.conv_section_dims), 3)
    manager.conv_state_dtype = SimpleNamespace(itemsize=2)
    manager.ssm_state_shape = (8 // grid, 4, 4)
    manager.ssm_state_dtype = SimpleNamespace(itemsize=4)
    conv_bytes = sum(manager.conv_section_dims) * 3 * 2
    ssm_bytes = (8 // grid) * 4 * 4 * 4
    pattern = "AMAMMA"[: model.layers]
    configs, attention, mamba = {}, [], []
    for i, layer in enumerate(layers):
        if pattern[layer] == "M":
            buffers = [(MambaRole.CONV_STATE, conv_bytes), (MambaRole.SSM_STATE, ssm_bytes)]
            if model.kind == "qwen4exp":
                buffers.append((PLE_CONV_STATE, 48))
            configs[i] = _state_layer(i, buffers)
            mamba.append(i)
        else:
            configs[i] = _attention_layer(i, _kv_buffers(model, heads))
            attention.append(i)
    manager._mamba_layer_mask = {layer: pattern[layer] == "M" for layer in range(model.layers)}
    manager.local_num_mamba_layers = len(mamba)
    per_layer = [heads if pattern[layer] == "A" else 0 for layer in layers]
    groups = [ids for ids in (attention, mamba) if ids]
    _wire(manager, topology, pp, tp, cp, layers, configs, groups, base, per_layer, model)
    return manager, heads if attention else 0


def _build_fp4_mla(model, topology, pp, tp, cp, layers, base):
    """Per model layer, an FP4 latent cache and a high-precision tail in a layer of its own.

    The latent cache's V sidecars never move.
    """
    manager = _Attention.__new__(_Attention)
    manager._fp4_mla_hp_pool_size = 0
    Fp4MlaV2CacheLayoutPolicy.install_manager_capabilities(manager)
    n = len(layers)
    configs = {}
    for i in range(n):
        configs[i] = _attention_layer(
            i,
            [
                (Role.KEY, 576 * TOKENS_PER_BLOCK),
                (Role.MLA_V_SCALE, 8 * TOKENS_PER_BLOCK),
                (Role.MLA_V_PACKED, 32 * TOKENS_PER_BLOCK),
            ],
        )
        configs[n + i] = _attention_layer(n + i, [(Role.MLA_HP_TAIL, 128 * TOKENS_PER_BLOCK)])
    manager._cache_manager_layer_ids = list(range(n))
    manager._hp_manager_layer_ids = list(range(n, 2 * n))
    manager._bf16_mla_manager_layer_ids = []
    manager._fp4_mla_compact_to_local = list(range(n))
    groups = [list(range(n)), list(range(n, 2 * n))]
    _wire(manager, topology, pp, tp, cp, layers, configs, groups, base, 1, model)
    return manager, 1


def _build_csa2(model, topology, pp, tp, cp, layers, base):
    """Per model layer a windowed SWA cache and a global cache with its index, all replicated."""
    manager = _Csa2.__new__(_Csa2)
    manager.bounded_replay_on_generation = False
    page = TOKENS_PER_BLOCK * HEAD_DIM * ELEMENT_BYTES
    configs, swa, main = {}, [], []
    manager._layer_attn_to_layer_id = {}
    for i, layer in enumerate(layers):
        configs[2 * i] = _attention_layer(
            2 * i, [(CSA2CacheRole.SWA.role, page)], window=2 * TOKENS_PER_BLOCK
        )
        configs[2 * i + 1] = _attention_layer(
            2 * i + 1,
            [(CSA2CacheRole.GLOBAL.role, 2 * page), (CSA2CacheRole.INDEX.role, TOKENS_PER_BLOCK)],
        )
        manager._layer_attn_to_layer_id[layer, _CSA2LayerKind.SWA] = 2 * i
        manager._layer_attn_to_layer_id[layer, _CSA2LayerKind.GLOBAL] = 2 * i + 1
        swa.append(2 * i)
        main.append(2 * i + 1)
    _wire(manager, topology, pp, tp, cp, layers, configs, [swa, main], base, 1, model)
    return manager, 1


def _build_dsv4(model, topology, pp, tp, cp, layers, base):
    """Per model layer, a windowed SWA cache and a compressed cache, MLA latents of one KV head.

    The compressed cache keeps its block scales and a storage-only padding buffer beside it.
    """
    manager = _DeepseekV4.__new__(_DeepseekV4)
    manager._disagg_ignored_roles = frozenset((DSV4_PADDING_ROLE,))
    manager.use_fp8_ds_mla = False
    page = TOKENS_PER_BLOCK * HEAD_DIM * ELEMENT_BYTES
    configs, swa, main = {}, [], []
    manager._layer_attn_to_layer_id = {}
    for i, layer in enumerate(layers):
        configs[2 * i] = _attention_layer(
            2 * i, [(DeepseekV4AttentionType.SWA.role, page)], window=2 * TOKENS_PER_BLOCK
        )
        configs[2 * i + 1] = _attention_layer(
            2 * i + 1,
            [
                (DeepseekV4AttentionType.COMPRESS.role, 2 * page),
                (COMPRESS_BLOCK_SCALE_ROLE, TOKENS_PER_BLOCK * 2),
                (DSV4_PADDING_ROLE, TOKENS_PER_BLOCK * 6),
            ],
        )
        manager._layer_attn_to_layer_id[layer, DeepseekV4AttentionType.SWA] = 2 * i
        manager._layer_attn_to_layer_id[layer, DeepseekV4AttentionType.COMPRESS] = 2 * i + 1
        swa.append(2 * i)
        main.append(2 * i + 1)
    _wire(manager, topology, pp, tp, cp, layers, configs, [swa, main], base, 1, model)
    return manager, 1


_BUILDERS = {
    "attention": _build_attention,
    "minimax": _build_attention,
    "mamba": _build_mamba,
    "qwen4exp": _build_mamba,
    "fp4_mla": _build_fp4_mla,
    "csa2": _build_csa2,
    "dsv4": _build_dsv4,
}


@dataclass
class _Rank:
    info: RankInfo
    manager: KVCacheManagerV2
    tables: list[np.ndarray]


def _make_rank(model, topology, pp, tp, cp, name, index, base, side) -> _Rank:
    split = _split_layers(model.layers, topology.pp)
    start = sum(split[:pp])
    layers = list(range(start, start + split[pp]))
    manager, heads = _BUILDERS[model.kind](model, topology, pp, tp, cp, layers, base)
    info = RankInfo(
        instance_name=name,
        instance_rank=index,
        tp_size=topology.tp,
        tp_rank=tp,
        pp_size=topology.pp,
        pp_rank=pp,
        layer_num_per_pp=split,
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        dp_size=topology.tp if topology.adp else 1,
        dp_rank=tp if topology.adp else 0,
        cp_size=topology.cp,
        cp_rank=cp,
        attention=AttentionInfo(
            kv_heads_per_rank=heads,
            tokens_per_block=TOKENS_PER_BLOCK,
            dims_per_head=HEAD_DIM,
            element_bytes=ELEMENT_BYTES,
            enable_attention_dp=topology.adp,
            is_mla=_single_latent(model),
        ),
        page_table=build_page_table_from_manager(manager),
    )
    # Each rank's request holds every page of the prompt, a Helix rank every cp-th of them; a
    # state layer group holds one slot. Slots are shuffled and never repeat, as layer groups
    # sharing a pool group draw distinct slots from it.
    slots = iter(np.random.RandomState(1000 * side + index).permutation(SLOTS).tolist())
    tables = []
    for group in info.page_table.layer_groups:
        count = 1 if group.kind.name == "STATE" else len(range(cp, PAGES, topology.cp))
        tables.append(np.array([next(slots) for _ in range(count)], dtype=np.int64))
    return _Rank(info, manager, tables)


def _native_writes(src: _Rank, dst: _Rank):
    """What native's sender writes into the destination, as canonical (src, dst, size) runs.

    The write plan is the one native's ``Sender`` builds from a registrar alone. None when the two
    ranks do not pair.
    """
    registrar = PeerRegistrar(src.info, KVRegionExtractorV1(src.info.page_table))
    registrar.register(dst.info.instance_name, dst.info.instance_rank, dst.info)
    if dst.info.instance_rank not in registrar.get_peer_overlap(dst.info, dst.info.dp_rank).ranks:
        return None
    prompt = PAGES * TOKENS_PER_BLOCK
    chunk = Chunk(
        block_ids_per_layer_groups=src.tables,
        kind_per_layer_group=[group.kind for group in src.info.page_table.layer_groups],
        token_range=TokenRange(start=0, end=prompt),
        is_last=True,
    )
    task = SimpleNamespace(
        _chunk=chunk, _prompt_len=prompt, _perf_timer=None, _unique_rid=0, slice_id=0
    )
    request = RecvReqInfo(
        sender_req_id=0,
        instance_name=dst.info.instance_name,
        instance_rank=dst.info.instance_rank,
        block_ids_per_layer_groups=dst.tables,
        unique_rid=0,
    )
    write = Sender._build_kv_write_meta(SimpleNamespace(_registrar=registrar), task, request)
    return merge_runs(
        np.asarray(write.src_ptrs, dtype=np.int64),
        np.asarray(write.dst_ptrs, dtype=np.int64),
        np.asarray(write.sizes, dtype=np.int64),
    )


def _paired_pages(src: _Rank, src_group: int, dst: _Rank, dst_group: int, state: bool):
    """(source slot, destination slot) of each page both requests hold.

    The state slots, or the pages by ordinal, a Helix receiver's ordinal ``i`` being the sender's
    ``cp_rank + i * cp``.
    """
    src_table, dst_table = src.tables[src_group], dst.tables[dst_group]
    if state:
        return [(int(src_table[0]), int(dst_table[0]))]
    if dst.info.cp_size > 1 and src.info.cp_size == 1:
        src_table = src_table[dst.info.cp_rank :: dst.info.cp_size]
    return [(int(s), int(d)) for s, d in zip(src_table, dst_table) if s >= 0 and d >= 0]


def _layout_writes(src: _Rank, dst: _Rank, leave_out=lambda runs: False):
    """What ``map_buffers`` has the source write into the destination, as canonical runs.

    The runs are (src, dst, size) at the slots of the pages both requests hold, without the
    buffers ``leave_out`` picks.
    """
    src_table, dst_table = src.manager.get_page_table(), dst.manager.get_page_table()
    src_layout, dst_layout = src_table.page_layout, dst_table.page_layout
    src_groups, dst_groups = layer_group_of(src_layout), layer_group_of(dst_layout)
    src_parts, dst_parts, size_parts = [], [], []
    for runs in map_buffers(src_layout, dst_layout):
        if leave_out(runs):
            continue
        src_group = src_groups[runs.src.pool_group, runs.src.layer]
        dst_group = dst_groups[runs.dst.pool_group, runs.dst.layer]
        state = src_layout.layer_groups[src_group].kind == "state"
        src_pool = src_table.device_pools[runs.src.pool_group][runs.src.pool_index]
        dst_pool = dst_table.device_pools[runs.dst.pool_group][runs.dst.pool_index]
        for src_slot, dst_slot in _paired_pages(src, src_group, dst, dst_group, state):
            src_parts.append(src_pool.base + src_slot * src_pool.slot_bytes + runs.src_offsets)
            dst_parts.append(dst_pool.base + dst_slot * dst_pool.slot_bytes + runs.dst_offsets)
            size_parts.append(runs.sizes)
    if not src_parts:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    return merge_runs(
        np.concatenate(src_parts), np.concatenate(dst_parts), np.concatenate(size_parts)
    )


def _native_gap(src: _Rank, dst: _Rank):
    """The buffers native may leave out of what the source writes into the destination, or None.

    Native pairs a Helix receiver's recurrent state with the sender holding the same share of the
    tensor-times-context grid, but elects the sender of replicated recurrent state (PLE) among
    tensor ranks alone, so a receiver paired with a sender that is not elected gets that state
    from no rank. The layout mapping elects among the ranks it pairs, for every buffer alike.
    """
    if dst.info.cp_size == 1 or src.info.cp_size > 1:
        return None
    return lambda runs: runs.src.role == str(PLE_CONV_STATE)


def _same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def _first_difference(native, layout) -> str:
    rows_native = set(zip(*(a.tolist() for a in native)))
    rows_layout = set(zip(*(a.tolist() for a in layout)))
    only_native = sorted(rows_native - rows_layout)[:3]
    only_layout = sorted(rows_layout - rows_native)[:3]
    return (
        f"{len(rows_native)} native runs ({sum(r[2] for r in rows_native)} bytes), "
        f"{len(rows_layout)} layout runs ({sum(r[2] for r in rows_layout)} bytes); "
        f"native only {only_native}, layout only {only_layout}"
    )


@pytest.mark.parametrize("context", CONTEXT_TOPOLOGIES, ids=str)
@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.name)
def test_layout_mapping_writes_what_native_writes(model: Model, context: Topology) -> None:
    differences = []
    for generation in GENERATION_TOPOLOGIES:
        ctx = [
            _make_rank(model, context, *place, "ctx", i, (1 + i) << 32, 0)
            for i, place in enumerate(context.ranks())
        ]
        gen = [
            _make_rank(model, generation, *place, "gen", i, (100 + i) << 32, 1)
            for i, place in enumerate(generation.ranks())
        ]
        for src, dst in itertools.product(ctx, gen):
            pair = (
                f"{context}->{generation} rank {src.info.instance_rank}->{dst.info.instance_rank}"
            )
            native = _native_writes(src, dst)
            layout = _layout_writes(src, dst)
            if native is None:
                if layout[0].size:
                    differences.append(f"{pair}: native does not pair them; layout writes")
                continue
            if _same(native, layout):
                continue
            gap = _native_gap(src, dst)
            if gap is not None and _same(native, _layout_writes(src, dst, leave_out=gap)):
                continue
            differences.append(f"{pair}: {_first_difference(native, layout)}")
    assert not differences, "\n".join(differences)
