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
"""The layout of a manager's blocks as the lender needs it, and ``layout_id``, the digest of how a
block's bytes are laid out. Read from the manager's public layout surface and declarations."""

from __future__ import annotations

import enum
import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from ..page_layout import (
    BufferDesc,
    LayerGroupDesc,
    MapperKind,
    PageLayout,
    RankTopology,
    RoleLayout,
    ShardDesc,
)
from . import _manager

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig

    from ..kv_cache_manager_v2 import KVCacheManagerV2

# A new field left out at its default keeps the version; changing the encoding or the meaning of
# an existing field bumps it.
LAYOUT_ID_VERSION = 3

# The fallback entry of the manager's per-role declarations (``Role.ALL``).
_ALL_ROLE = "all"


# Enums encode as the member name, floats as "f64:" plus the 16 hex digits of their big-endian
# IEEE-754 binary64 bits; a None value drops an object's key and is null in an array.
def _canonical(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.name
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return "f64:" + struct.pack(">d", float(value)).hex()
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical objects have string keys, got {key!r}")
            if item is not None:
                out[key] = _canonical(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TypeError(f"{type(value).__name__} has no canonical encoding")


def canonical_bytes(document: Mapping) -> bytes:
    """``document`` as ASCII JSON, keys sorted, no whitespace, values encoded by ``_canonical``."""
    return json.dumps(
        _canonical(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_order(layers: Sequence[Sequence[int]]) -> list[int]:
    """Each local layer group's canonical index: groups ordered by their smallest global layer id,
    which pipeline parallelism does not change; ties keep the local order."""
    order = sorted(range(len(layers)), key=lambda g: (min(layers[g], default=1 << 62), g))
    canonical = [0] * len(layers)
    for index, local in enumerate(order):
        canonical[local] = index
    return canonical


def _geometry_document(buffer: BufferDesc) -> dict:
    geometry = buffer.geometry
    indexed = buffer.mapper not in (MapperKind.REPLICATED, MapperKind.SECTIONED)
    return {
        "section_bytes": None if geometry.section_bytes is None else list(geometry.section_bytes),
        "bytes_per_head": geometry.bytes_per_head,
        "num_heads": buffer.size // geometry.bytes_per_head
        if indexed and geometry.bytes_per_head
        else None,
    }


def _buffer_document(layout: PageLayout, b: BufferDesc) -> dict:
    # A buffer's position in the page: the slot widths of the pools before it in its group, plus
    # its offset inside the slot.
    widths = layout.pool_slot_bytes.get(b.pool_group)
    if widths is None:
        raise ValueError(
            f"buffer ({b.layer}, {b.role}) lies in pool {(b.pool_group, b.pool_index)}, not in the layout"
        )
    doc = {
        "layer": int(b.layer),
        "role": b.role,
        "offset": sum(widths[: b.pool_index]) + int(b.slot_offset),
        "size": int(b.size),
    }
    if b.mapper is not MapperKind.INDEXED:
        doc["mapper"] = b.mapper.name
    geometry = {k: v for k, v in _geometry_document(b).items() if v is not None}
    if geometry:
        doc["geometry"] = geometry
    if not b.transferred:
        doc["transfer"] = False
    if int(b.expansion) != 1:
        doc["expansion"] = int(b.expansion)
    return doc


def layout_document(layout: PageLayout) -> dict:
    """Hash canonical global layer groups, slot widths, kinds and buffer geometry, but no addresses,
    windows, sinks, shards, topology, head counts, dtypes or local region numbering."""
    canonical = canonical_order([desc.layers for desc in layout.layer_groups])
    groups = [None] * len(layout.layer_groups)
    for local, desc in enumerate(layout.layer_groups):
        widths = layout.pool_slot_bytes.get(desc.pool_group)
        if widths is None:
            raise ValueError(
                f"layer group {local} maps to pool group {desc.pool_group}, which is not in the "
                "layout"
            )
        entry = {
            "layers": sorted(int(layer) for layer in desc.layers),
            "slot_bytes": [int(w) for w in widths],
        }
        if desc.kind != "attention":
            entry["kind"] = desc.kind
        groups[canonical[local]] = entry
    buffers = sorted(
        (_buffer_document(layout, b) for b in layout.buffers),
        key=lambda d: (d["layer"], d["role"]),
    )
    for before, after in zip(buffers, buffers[1:]):
        if (before["layer"], before["role"]) == (after["layer"], after["role"]):
            raise ValueError(f"buffer ({before['layer']}, {before['role']}) appears twice")
    return {
        "v": LAYOUT_ID_VERSION,
        "tokens_per_block": int(layout.tokens_per_page),
        "groups": groups,
        "buffers": buffers,
    }


def layout_id(layout: PageLayout) -> bytes:
    """SHA-256 of ``canonical_bytes(layout_document(layout))``."""
    return hashlib.sha256(canonical_bytes(layout_document(layout))).digest()


@dataclass(frozen=True)
class DevicePool:
    """One device pool of pool group ``group``: slot ``s`` lives at ``base + s * slot_bytes``."""

    group: int
    index: int
    base: int
    slot_bytes: int
    num_slots: int


@dataclass(frozen=True, eq=False)
class ManagerLayout:
    """Plain values without a manager reference; ``cp`` is (rank, size), (0, 1) without Helix.
    ``tokens_per_block`` spans ``size`` physical pages of ``tokens_per_block // size`` tokens."""

    page_layout: PageLayout
    tokens_per_block: int
    pool_group_of: tuple[int, ...]
    windows: tuple[int | None, ...]
    sink_blocks: tuple[int, ...]
    recurrent: tuple[bool, ...]
    sparse: tuple[bool, ...]
    layers: tuple[tuple[int, ...], ...]
    shards: tuple[tuple[int, int], ...]
    page_bytes: Mapping[int, int]
    device_pools: Mapping[int, tuple[DevicePool, ...]]
    cp: tuple[int, int] = (0, 1)
    # Per layer group, the window its attention reads, at most the window kept; ``()``: the same.
    read_windows: tuple[int | None, ...] = ()
    # Per layer group, whether it lends by name: none of its layers kept out of prefix reuse.
    named_groups: tuple[bool, ...] = field(kw_only=True)
    # Whether every windowed layer group's stale range is the runtime's window formula of its
    # window and sink blocks at each history, as the manager answered for every history probed.
    stale_ranges_by_formula: bool = field(default=False, kw_only=True)

    @property
    def num_layer_groups(self) -> int:
        """The manager's layer groups."""
        return len(self.pool_group_of)

    @property
    def pool_groups(self) -> tuple[int, ...]:
        """The device pool groups, sorted; staging regions follow this order."""
        return tuple(sorted(self.device_pools))


def _topology(mapping: object) -> RankTopology:
    """The rank's place as native's ``RankInfo`` reads it from the mapping: under attention data
    parallelism each tensor-parallel rank is a data-parallel rank of its own."""
    if mapping is None:
        return RankTopology()
    adp = bool(getattr(mapping, "enable_attention_dp", False))
    tp_size, tp_rank = int(getattr(mapping, "tp_size", 1)), int(getattr(mapping, "tp_rank", 0))
    return RankTopology(
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=int(getattr(mapping, "pp_size", 1)),
        pp_rank=int(getattr(mapping, "pp_rank", 0)),
        cp_size=int(getattr(mapping, "cp_size", 1)),
        cp_rank=int(getattr(mapping, "cp_rank", 0)),
        dp_size=tp_size if adp else int(getattr(mapping, "dp_size", 1)),
        dp_rank=tp_rank if adp else 0,
        attention_dp=adp,
    )


def _sparse(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime has sparse buffers (``is_sparse``), whose read-only
    pages a cache can lock in host memory."""
    return any(getattr(b, "is_sparse", False) for b in getattr(layer_config, "buffers", ()) or ())


def _expansions(
    layer_config: AttentionLayerConfig | SsmLayerConfig, tokens_per_block: int
) -> dict[str, int]:
    """Role -> sub-pages per block, from the layer's ``tokens_per_block_override``."""
    out = {}
    for buffer in getattr(layer_config, "buffers", ()) or ():
        override = getattr(buffer, "tokens_per_block_override", None)
        if override is not None:
            if tokens_per_block % int(override):
                raise ValueError(
                    f"buffer {buffer.role}: {override} tokens per block does not divide "
                    f"{tokens_per_block}"
                )
            out[str(buffer.role)] = tokens_per_block // int(override)
    return out


class _Heads:
    """Head counts of attention buffers: per rank from the manager, in total from its inputs; and
    the grid of ranks recurrent state splits over."""

    def __init__(self, manager: KVCacheManagerV2) -> None:
        mapping = getattr(manager, "mapping", None)
        dp = bool(getattr(mapping, "enable_attention_dp", False))
        self.tp_size = 1 if mapping is None or dp else max(int(getattr(mapping, "tp_size", 1)), 1)
        self.tp_rank = (
            0 if self.tp_size == 1 else int(getattr(mapping, "tp_rank", 0)) % self.tp_size
        )
        cp_rank, cp_size = (0, 1) if dp else _manager.helix(manager)
        self.state_size = self.tp_size * cp_size
        self.state_rank = self.tp_rank * cp_size + cp_rank
        self._per_rank = list(getattr(manager, "num_kv_heads_per_layer", ()) or ())
        self._total = getattr(manager, "num_kv_heads", None)
        self._pp_layers = manager.pp_layers
        virtual = _manager.virtual_layers(manager)
        self._virtual = virtual is not None
        local_of_model = {int(model): local for local, model in enumerate(self._pp_layers)}
        self._head_index = (
            {
                internal: local_of_model[model]
                for internal, (model, _) in virtual[0].items()
                if model in local_of_model
            }
            if virtual is not None
            else {}
        )

    def per_rank(self, internal: int) -> int:
        if not self._per_rank:
            return 0
        index = self._head_index.get(internal, internal)
        if index >= len(self._per_rank):
            index = 0
        return int(self._per_rank[index] or 0)

    def total(self, internal: int) -> int | None:
        if isinstance(self._total, int):
            return int(self._total)
        if self._total is None or self._virtual:
            return None
        if internal >= len(self._pp_layers):
            return None  # an extra internal layer (FP4 MLA's tail) is no model layer
        value = self._total[int(self._pp_layers[internal])]
        return None if value is None else int(value)

    def attention_shard(self, internal: int) -> ShardDesc:
        """The share of a head-sharded attention buffer this rank holds: ``T`` heads over ``tp``
        ranks, rank ``r`` holding ``ceil(T / tp)`` from ``r * T // tp``."""
        if self.tp_size == 1:
            return ShardDesc()
        heads, total = self.per_rank(internal), self.total(internal)
        if heads <= 0 or total is None or total <= 0:
            return ShardDesc(self.tp_size, self.tp_rank)
        if any(rank * total // self.tp_size % heads for rank in range(self.tp_size)):
            # Some rank's heads straddle two shares, so each rank's buffer is a share of its own.
            return ShardDesc(self.tp_size, self.tp_rank)
        # total // heads distinct shares; fewer heads than ranks are repeated, so a single head is
        # one share that every rank holds.
        count = max(total // heads, 1)
        index = min((self.tp_rank * total // self.tp_size) // heads, count - 1)
        return ShardDesc(count, index)

    def state_shard(self) -> ShardDesc:
        """The share of recurrent state this rank holds: over the tensor-parallel ranks, over none
        under attention data parallelism, and over tensor times context ranks, context-minor, under
        Helix."""
        if self.state_size == 1:
            return ShardDesc()
        return ShardDesc(self.state_size, self.state_rank)


def _buffer_geometry(
    declared: object, mapper: MapperKind, size: int, heads: int, state: bool
) -> RoleLayout:
    """A buffer's head geometry: declared sections or head size, completed per buffer.
    ``ValueError`` if the declared or completed geometry has a size that is not positive."""
    declared = RoleLayout(
        getattr(declared, "section_bytes", None), getattr(declared, "bytes_per_head", None)
    )
    if mapper in (MapperKind.REPLICATED, MapperKind.SECTIONED):
        return declared
    bytes_per_head = declared.bytes_per_head
    if bytes_per_head is None and not state and heads > 0 and size % heads == 0:
        bytes_per_head = size // heads
    if bytes_per_head is None or size % bytes_per_head:
        return declared
    return RoleLayout(declared.section_bytes, bytes_per_head)


# A layer group is whole when every transferred buffer holds the same bytes on every rank
# (replicated, or one KV head shared by all ranks); else it is the share this rank holds.
def _group_shard(shards: Sequence[ShardDesc], fallback: ShardDesc) -> ShardDesc:
    parts = {s for s in shards if s.count != 1}
    if not parts:
        return ShardDesc()
    return parts.pop() if len(parts) == 1 else fallback


def derive_layout(manager: KVCacheManagerV2) -> ManagerLayout:
    """The layout of a ``KVCacheManagerV2`` and its device pools. ``ValueError`` if its layer
    groups, layer configs and declarations do not agree."""
    tokens_per_block = int(manager.tokens_per_block)
    layer_cfg = _manager.layer_configs(manager)
    grouping = _manager.layer_group_layer_ids(manager)
    pg_of_lg = _manager.layer_group_pool_indices(manager)
    if len(pg_of_lg) != len(grouping):
        raise ValueError(f"{len(grouping)} layer groups but {len(pg_of_lg)} pool-group indices")
    internal = sorted({lid for group in grouping for lid in group})
    global_of = dict(zip(internal, _manager.global_layer_ids(manager, internal)))
    model_of = dict(zip(internal, _manager.model_layer_ids(manager, internal)))

    state_groups = set()
    for lg, layer_ids in enumerate(grouping):
        if not layer_ids or layer_ids[0] not in layer_cfg:
            raise ValueError(f"layer group {lg} has no layer config (layers {layer_ids})")
        if _manager.is_recurrent_layer(layer_cfg[layer_ids[0]]):
            state_groups.add(lg)

    # TODO: the base manager declares head-major K/V whichever backend writes its pages, so a
    # token-major backend's manager gets a head-major layout_id unless it declares its own.
    mappers = {
        role: MapperKind(int(kind)) for role, kind in _manager.role_mapper_kinds(manager).items()
    }
    if _ALL_ROLE not in mappers:
        raise ValueError(
            "the manager declares no mapper kind for Role.ALL, the fallback of the roles it does "
            "not list"
        )
    default_mapper = mappers[_ALL_ROLE]
    role_layouts = _manager.role_layouts(manager)
    ignored = _manager.ignored_roles(manager)
    heads = _Heads(manager)
    expansions = {lid: _expansions(cfg, tokens_per_block) for lid, cfg in layer_cfg.items()}

    device_pools: dict[int, tuple[DevicePool, ...]] = {}
    buffers: list[BufferDesc] = []
    shards: dict[int, list[ShardDesc]] = {lg: [] for lg in range(len(grouping))}
    for pg in _manager.pool_group_descriptors(manager):
        g = int(pg.pool_group_index)
        pools = tuple(
            DevicePool(
                g, int(p.pool_index), int(p.base_address), int(p.slot_bytes), int(pg.num_slots)
            )
            for p in pg.pools
        )
        device_pools[g] = tuple(sorted(pools, key=lambda p: p.index))
        # One variant per layer group drawing from this pool group; pool ``i`` of a slot holds the
        # ``i``-th coalesced buffer, whose members sit back to back.
        for variant in pg.slot_desc.variants:
            lg = int(variant.layer_group_id)
            state = lg in state_groups
            for pool_index, coalesced in enumerate(variant.coalesced_buffers):
                size = int(coalesced.single_buffer_size)
                for j, buffer_id in enumerate(coalesced.buffer_ids):
                    lid = int(buffer_id.layer_id)
                    role = str(buffer_id.role)
                    mapper = mappers.get(role, default_mapper)
                    transfer = role not in ignored
                    buffers.append(
                        BufferDesc(
                            layer=global_of.get(lid, lid),
                            role=role,
                            pool_group=g,
                            pool_index=pool_index,
                            slot_offset=j * size,
                            size=size,
                            mapper=mapper,
                            geometry=_buffer_geometry(
                                role_layouts.get(role), mapper, size, heads.per_rank(lid), state
                            ),
                            transferred=transfer,
                            expansion=expansions.get(lid, {}).get(role, 1),
                        )
                    )
                    if not transfer or mapper is MapperKind.REPLICATED:
                        continue
                    shards[lg].append(heads.state_shard() if state else heads.attention_shard(lid))

    layer_groups: list[LayerGroupDesc] = []
    total_kv_heads: dict[int, int] = {}
    for lg, layer_ids in enumerate(grouping):
        first = layer_cfg[layer_ids[0]]
        is_state = lg in state_groups
        window = read = None
        sink_tokens = 0
        if not is_state:
            window = getattr(first, "window_size", None)
            window = None if window is None else int(window)
            read = _manager.read_window(manager, lg, window)
            read = None if read is None else int(read)
            if (read is None) != (window is None) or (read is not None and not 0 < read <= window):
                raise ValueError(
                    f"layer group {lg} reads a window of {read} tokens, which its kept window of "
                    f"{window} tokens does not hold"
                )
            sink_tokens = int(getattr(first, "num_sink_tokens", None) or 0)
            for lid in layer_ids:
                total = heads.total(lid)
                if total:
                    total_kv_heads[global_of[lid]] = total
        # Layer groups with equal slot sizes may share a pool group, each drawing its own slots.
        g = pg_of_lg[lg]
        if g not in device_pools:
            raise ValueError(f"layer group {lg} maps to pool group {g}, which has no pools")
        named = sorted((global_of[lid], model_of[lid]) for lid in layer_ids)
        layer_groups.append(
            LayerGroupDesc(
                kind="state" if is_state else "attention",
                kept_window=window,
                num_sink_blocks=-(-sink_tokens // tokens_per_block),
                pool_group=g,
                layers=tuple(layer for layer, _ in named),
                shard=_group_shard(shards[lg], ShardDesc(heads.tp_size, heads.tp_rank)),
                read_window=read,
                model_layers=tuple(model_layer for _, model_layer in named),
            )
        )

    page_layout = PageLayout(
        tokens_per_page=tokens_per_block,
        pool_slot_bytes={
            g: tuple(p.slot_bytes for p in pools) for g, pools in device_pools.items()
        },
        layer_groups=tuple(layer_groups),
        buffers=tuple(buffers),
        topology=_topology(getattr(manager, "mapping", None)),
        total_kv_heads_by_layer=total_kv_heads,
        dtype=_manager.kv_dtype(manager),
        quant_mode=_manager.kv_quant_mode(manager),
    )
    cp_rank, cp_size = _manager.helix(manager)
    return ManagerLayout(
        page_layout=page_layout,
        tokens_per_block=tokens_per_block * cp_size,
        pool_group_of=tuple(pg_of_lg),
        windows=tuple(desc.kept_window for desc in layer_groups),
        sink_blocks=tuple(desc.num_sink_blocks for desc in layer_groups),
        recurrent=tuple(desc.kind == "state" for desc in layer_groups),
        sparse=tuple(_sparse(layer_cfg[layer_ids[0]]) for layer_ids in grouping),
        layers=tuple(desc.layers for desc in layer_groups),
        shards=tuple((desc.shard.count, desc.shard.index) for desc in layer_groups),
        page_bytes={g: sum(p.slot_bytes for p in pools) for g, pools in device_pools.items()},
        device_pools=device_pools,
        cp=(cp_rank, cp_size),
        read_windows=tuple(desc.read_window for desc in layer_groups),
        named_groups=tuple(
            not any(_manager.is_private_layer(layer_cfg[lid]) for lid in group)
            for group in grouping
        ),
    )
