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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np

from . import _manager

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig

    from ..kv_cache_manager_v2 import KVCacheManagerV2

# A new field left out at its default keeps the version; changing the encoding or the meaning of
# an existing field bumps it.
LAYOUT_ID_VERSION = 2

# The fallback entry of the manager's per-role declarations (``Role.ALL``).
_ALL_ROLE = "all"


class BufferMapper(enum.IntEnum):
    """How a buffer's bytes split across tensor-parallel ranks; the values of the mapper kinds
    managers declare per role, read by value."""

    INDEXED = 0  # head-major K/V, the default: a head range is one contiguous range per buffer
    REPLICATED = 1  # bytes identical on every TP rank
    NHD = 2  # token-major [token, head, dim] K/V: a head range is a slice inside every token
    SECTIONED = 3  # [Sec0|Sec1|...], each section sharded independently (recurrent conv state)


@dataclass(frozen=True)
class BufferGeometry:
    """Resharding geometry of one buffer; all ``None`` means no head axis. ``section_bytes`` sum to
    the buffer's size; ``bytes_per_head * num_heads`` equals it."""

    section_bytes: tuple[int, ...] | None = None
    bytes_per_head: int | None = None
    num_heads: int | None = None

    def validate(self) -> BufferGeometry:
        """A copy with ``section_bytes`` as ints; ``ValueError`` unless every size given is
        positive."""
        sections = self.section_bytes
        if sections is not None:
            sections = tuple(int(s) for s in sections)
            if any(s <= 0 for s in sections):
                raise ValueError(f"section sizes must be positive, got {sections}")
        for name in ("bytes_per_head", "num_heads"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        return BufferGeometry(sections, self.bytes_per_head, self.num_heads)

    def check_size(self, size: int) -> None:
        """``ValueError`` unless this geometry describes a buffer of ``size`` bytes."""
        if self.section_bytes is not None and sum(self.section_bytes) != size:
            raise ValueError(f"sections {self.section_bytes} do not sum to {size} bytes")
        if self.bytes_per_head is not None and size % self.bytes_per_head:
            raise ValueError(
                f"{size} bytes is not a whole number of {self.bytes_per_head}-byte heads"
            )
        if (
            self.bytes_per_head is not None
            and self.num_heads is not None
            and self.bytes_per_head * self.num_heads != size
        ):
            raise ValueError(
                f"{self.num_heads} heads of {self.bytes_per_head} bytes is not {size} bytes"
            )


@dataclass(frozen=True)
class ShardDesc:
    """Which share (``index`` of ``count``) of a layer group's content this rank holds; ``(1, 0)``
    is the whole content, which every rank names alike. ``ValueError`` for an index outside the
    count."""

    count: int = 1
    index: int = 0

    def __post_init__(self) -> None:
        if self.count < 1 or not 0 <= self.index < self.count:
            raise ValueError(f"shard {self.index} of {self.count}")


@dataclass(frozen=True)
class LayerGroupDesc:
    """One layer group: its kind, window and sink blocks (which blocks exist), its device pool
    group, its global layer ids and the share of its content this rank holds."""

    kind: Literal["attention", "state"]
    window: int | None
    sink_blocks: int
    pool_group: int
    layers: tuple[int, ...]
    shard: ShardDesc = ShardDesc()


@dataclass(frozen=True)
class BufferDesc:
    """One buffer of one layer (global id) in a page: ``size`` bytes at ``offset`` in the slot of
    ``pool`` = (pool group, pool index), how its bytes split across ranks, whether peers transfer
    it, and its sub-pages per block. ``ValueError`` for a non-positive size or expansion."""

    layer: int
    role: str
    pool: tuple[int, int]
    offset: int
    size: int
    mapper: BufferMapper = BufferMapper.INDEXED
    geometry: BufferGeometry = BufferGeometry()
    # False for local-only buffers peers skip; they still ride along wherever whole slots are
    # copied.
    transfer: bool = True
    # The buffer's own tokens per block is tokens_per_block / expansion; a consumer reading inside
    # the buffer that does not support the value must reject the layout.
    expansion: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "mapper", BufferMapper(self.mapper))
        if not isinstance(self.geometry, BufferGeometry):
            raise ValueError(f"geometry must be a buffer geometry, got {self.geometry!r}")
        if self.size <= 0 or self.offset < 0:
            raise ValueError(f"buffer at offset {self.offset} of {self.size} bytes")
        if int(self.expansion) < 1:
            raise ValueError(f"expansion must be a positive integer, got {self.expansion}")
        self.geometry.check_size(self.size)


@dataclass(frozen=True)
class Layout:
    """How pages are built, without addresses or slot counts: ``pool_groups[g]`` holds the slot
    widths of device pool group ``g`` in pool order. ``ValueError`` for an expansion that does not
    divide ``tokens_per_block``."""

    tokens_per_block: int
    pool_groups: Mapping[int, tuple[int, ...]]
    layer_groups: tuple[LayerGroupDesc, ...]
    buffers: tuple[BufferDesc, ...]

    def __post_init__(self) -> None:
        pool_groups = {
            int(g): tuple(int(w) for w in widths) for g, widths in dict(self.pool_groups).items()
        }
        object.__setattr__(self, "pool_groups", pool_groups)
        for b in self.buffers:
            if self.tokens_per_block % int(b.expansion):
                raise ValueError(
                    f"buffer ({b.layer}, {b.role}): expansion {b.expansion} does not divide "
                    f"{self.tokens_per_block} tokens per block"
                )


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
    """``document`` as ASCII JSON, keys sorted, no whitespace; enums by name, floats as ``f64:`` and
    their big-endian bits in hex, ``None`` values dropped from objects."""
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


def _geometry_document(geometry: BufferGeometry) -> dict:
    return {
        "section_bytes": None if geometry.section_bytes is None else list(geometry.section_bytes),
        "bytes_per_head": geometry.bytes_per_head,
        "num_heads": geometry.num_heads,
    }


def _buffer_document(layout: Layout, b: BufferDesc) -> dict:
    # A buffer's position in the page: the slot widths of the pools before it in its group, plus
    # its offset inside the slot.
    widths = layout.pool_groups.get(b.pool[0])
    if widths is None:
        raise ValueError(f"buffer ({b.layer}, {b.role}) lies in pool {b.pool}, not in the layout")
    doc = {
        "layer": int(b.layer),
        "role": b.role,
        "offset": sum(widths[: b.pool[1]]) + int(b.offset),
        "size": int(b.size),
    }
    # Fields at their defaults are left out, so a new defaulted field keeps existing ids.
    if b.mapper is not BufferMapper.INDEXED:
        doc["mapper"] = b.mapper.name
    geometry = {k: v for k, v in _geometry_document(b.geometry).items() if v is not None}
    if geometry:
        doc["geometry"] = geometry
    if not b.transfer:
        doc["transfer"] = False
    if int(b.expansion) != 1:
        doc["expansion"] = int(b.expansion)
    return doc


def layout_document(layout: Layout) -> dict:
    """The document ``layout_id`` hashes: tokens per block, groups in canonical order (layers, slot
    widths, kind), buffers by (layer, role). Addresses, windows, sinks, shards, local numbering and
    element types stay out."""
    canonical = canonical_order([desc.layers for desc in layout.layer_groups])
    groups = [None] * len(layout.layer_groups)
    for local, desc in enumerate(layout.layer_groups):
        widths = layout.pool_groups.get(desc.pool_group)
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
        "tokens_per_block": int(layout.tokens_per_block),
        "groups": groups,
        "buffers": buffers,
    }


def layout_id(layout: Layout) -> bytes:
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
    """What the lender reads of one manager, as plain values that hold no manager: the address-free
    ``layout``, facts per layer group (tuples indexed by the local layer group) and per device pool
    group (mappings keyed by the pool group)."""

    layout: Layout
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

    @property
    def num_layer_groups(self) -> int:
        """The manager's layer groups."""
        return len(self.pool_group_of)

    @property
    def pool_groups(self) -> tuple[int, ...]:
        """The device pool groups, sorted; staging parts follow this order."""
        return tuple(sorted(self.device_pools))


def global_layer_ids(manager: KVCacheManagerV2, internal_ids: Sequence[int]) -> list[int]:
    """The global id of each internal layer id, the same under any pipeline split: the declared id;
    else for virtual layers ``model_layer * number_of_attention_types + attention_type``; else the
    model layer."""
    # One declared id per internal layer of each layer group, in order (FP4 MLA's tail layers).
    declared = getattr(manager, "get_disagg_global_layer_ids", None)
    if declared is not None:
        table: dict[int, int] = {}
        for lg, layer_ids in enumerate(manager.impl.layer_grouping):
            ids = [int(gid) for gid in declared(lg)]
            if len(ids) != len(layer_ids):
                raise ValueError(
                    f"layer group {lg}: {len(ids)} declared global ids for {len(layer_ids)} layers"
                )
            table.update(zip((int(lid) for lid in layer_ids), ids))
        return [table[int(lid)] for lid in internal_ids]
    virtual = _manager.virtual_layers(manager)
    if virtual is None:
        return [int(manager.pp_layers[int(lid)]) for lid in internal_ids]
    # The number of types counts every member of the enum, so stages holding different attention
    # types agree.
    inverse, num_types = virtual
    return [inverse[int(lid)][0] * num_types + inverse[int(lid)][1] for lid in internal_ids]


def _layer_configs(manager: KVCacheManagerV2) -> Mapping[int, object]:
    """Internal layer id -> layer config, looked up by ``layer_id``, not by list position."""
    config = getattr(manager, "kv_cache_manager_py_config", None)
    if config is None:
        config = manager.impl.init_config
    out: dict[int, object] = {}
    for position, layer in enumerate(config.layers):
        out[int(getattr(layer, "layer_id", position))] = layer
    return out


def _sparse(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime has sparse buffers (``is_sparse``), whose read-only
    pages a cache can lock in host memory."""
    return any(getattr(b, "is_sparse", False) for b in getattr(layer_config, "buffers", ()) or ())


def _declared(manager: KVCacheManagerV2, getter: str) -> dict[str, object]:
    method = getattr(manager, getter, None)
    if method is None:
        return {}
    return {str(role): value for role, value in dict(method()).items()}


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
    """Head counts of attention buffers: per rank from the manager, in total from its inputs."""

    def __init__(self, manager: KVCacheManagerV2) -> None:
        mapping = getattr(manager, "mapping", None)
        dp = bool(getattr(mapping, "enable_attention_dp", False))
        self.tp_size = 1 if mapping is None or dp else max(int(getattr(mapping, "tp_size", 1)), 1)
        self.tp_rank = (
            0 if self.tp_size == 1 else int(getattr(mapping, "tp_rank", 0)) % self.tp_size
        )
        self._per_rank = list(getattr(manager, "num_kv_heads_per_layer", ()) or ())
        self._total = getattr(manager, "num_kv_heads", None)
        self._virtual = _manager.virtual_layers(manager) is not None
        self._pp_layers = manager.pp_layers

    def per_rank(self, internal: int) -> int:
        if not self._per_rank:
            return 0
        index = internal if internal < len(self._per_rank) else 0
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
        ranks, rank ``r`` holds ``ceil(T / tp)`` from ``r * T // tp``. Where any rank's range starts
        inside a share, each rank's buffer is a share of its own."""
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
        return ShardDesc(self.tp_size, self.tp_rank) if self.tp_size > 1 else ShardDesc()


def _buffer_geometry(
    declared: object, mapper: BufferMapper, size: int, heads: int, state: bool
) -> BufferGeometry:
    """A buffer's head geometry: declared sections or head size, completed per buffer.
    ``ValueError`` if the declared or completed geometry has a size that is not positive."""
    declared = BufferGeometry(
        getattr(declared, "section_bytes", None), getattr(declared, "bytes_per_head", None)
    ).validate()
    if mapper in (BufferMapper.REPLICATED, BufferMapper.SECTIONED):
        return declared
    bytes_per_head = declared.bytes_per_head
    if bytes_per_head is None and not state and heads > 0 and size % heads == 0:
        bytes_per_head = size // heads
    if bytes_per_head is None or size % bytes_per_head:
        return declared
    return BufferGeometry(declared.section_bytes, bytes_per_head, size // bytes_per_head).validate()


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
    impl = manager.impl
    tokens_per_block = int(manager.tokens_per_block)
    layer_cfg = _layer_configs(manager)
    grouping = [[int(lid) for lid in group] for group in impl.layer_grouping]
    pg_of_lg = [int(x) for x in impl.get_life_cycle_pool_group_indices()]
    if len(pg_of_lg) != len(grouping):
        raise ValueError(f"{len(grouping)} layer groups but {len(pg_of_lg)} pool-group indices")
    internal = sorted({lid for group in grouping for lid in group})
    global_of = dict(zip(internal, global_layer_ids(manager, internal)))

    state_groups = set()
    for lg, layer_ids in enumerate(grouping):
        if not layer_ids or layer_ids[0] not in layer_cfg:
            raise ValueError(f"layer group {lg} has no layer config (layers {layer_ids})")
        if _manager.holds_state(layer_cfg[layer_ids[0]]):
            state_groups.add(lg)

    # TODO: the base manager declares head-major K/V whichever attention backend writes its pages,
    # so a manager paired with a token-major backend gets the head-major layout_id unless it
    # declares its layout itself.
    mappers = {
        role: BufferMapper(int(kind))
        for role, kind in _declared(manager, "get_disagg_role_mapper_kinds").items()
    }
    default_mapper = mappers.get(_ALL_ROLE, BufferMapper.INDEXED)
    role_layouts = _declared(manager, "get_disagg_role_layouts")
    ignored_getter = getattr(manager, "get_disagg_ignored_roles", None)
    ignored = frozenset(str(r) for r in (ignored_getter() if ignored_getter else ()))
    heads = _Heads(manager)
    expansions = {lid: _expansions(cfg, tokens_per_block) for lid, cfg in layer_cfg.items()}

    device_pools: dict[int, tuple[DevicePool, ...]] = {}
    buffers: list[BufferDesc] = []
    shards: dict[int, list[ShardDesc]] = {lg: [] for lg in range(len(grouping))}
    for pg in impl.pool_group_descs:
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
                            pool=(g, pool_index),
                            offset=j * size,
                            size=size,
                            mapper=mapper,
                            geometry=_buffer_geometry(
                                role_layouts.get(role), mapper, size, heads.per_rank(lid), state
                            ),
                            transfer=transfer,
                            expansion=expansions.get(lid, {}).get(role, 1),
                        )
                    )
                    if not transfer or mapper is BufferMapper.REPLICATED:
                        continue
                    shards[lg].append(heads.state_shard() if state else heads.attention_shard(lid))

    layer_groups: list[LayerGroupDesc] = []
    for lg, layer_ids in enumerate(grouping):
        first = layer_cfg[layer_ids[0]]
        is_state = lg in state_groups
        window = None
        sink_tokens = 0
        if not is_state:
            window = getattr(first, "window_size", None)
            window = None if window is None else int(window)
            sink_tokens = int(getattr(first, "num_sink_tokens", None) or 0)
        # Layer groups with equal slot sizes may share a pool group, each drawing its own slots.
        g = pg_of_lg[lg]
        if g not in device_pools:
            raise ValueError(f"layer group {lg} maps to pool group {g}, which has no pools")
        layer_groups.append(
            LayerGroupDesc(
                kind="state" if is_state else "attention",
                window=window,
                sink_blocks=-(-sink_tokens // tokens_per_block),
                pool_group=g,
                layers=tuple(sorted(global_of[lid] for lid in layer_ids)),
                shard=_group_shard(shards[lg], ShardDesc(heads.tp_size, heads.tp_rank)),
            )
        )

    layout = Layout(
        tokens_per_block=tokens_per_block,
        pool_groups={g: tuple(p.slot_bytes for p in pools) for g, pools in device_pools.items()},
        layer_groups=tuple(layer_groups),
        buffers=tuple(buffers),
    )
    return ManagerLayout(
        layout=layout,
        tokens_per_block=tokens_per_block,
        pool_group_of=tuple(pg_of_lg),
        windows=tuple(desc.window for desc in layer_groups),
        sink_blocks=tuple(desc.sink_blocks for desc in layer_groups),
        recurrent=tuple(desc.kind == "state" for desc in layer_groups),
        sparse=tuple(_sparse(layer_cfg[layer_ids[0]]) for layer_ids in grouping),
        layers=tuple(desc.layers for desc in layer_groups),
        shards=tuple((desc.shard.count, desc.shard.index) for desc in layer_groups),
        page_bytes={g: sum(p.slot_bytes for p in pools) for g, pools in device_pools.items()},
        device_pools=device_pools,
    )
