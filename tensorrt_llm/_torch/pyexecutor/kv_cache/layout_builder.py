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
"""How a ``KVCacheManagerV2`` builds its ``PageLayout``.

The storage says where every buffer sits: in which layer group, pool and slot offset. The manager
says what each layer group is (``get_layer_group_traits``: its kind, the grid of ranks its buffers
spread over, and whether a request holds blocks or a slot), how each buffer's bytes split
(``get_buffer_layout``), which roles never travel (``get_untransferred_roles``), the global and model
layer of each internal layer (``get_layer_ids``), its KV heads (``get_kv_heads``) and the window
its attention reads (``_attention_read_window``). ``build_page_layout`` puts the two together. It
imports no torch, so it runs on duck-typed managers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .page_layout import (
    KIND_DEFAULTS,
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    RankTopology,
    ShardDesc,
    grid_position,
    held_span,
)

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig

    from .kv_cache_manager_v2 import KVCacheManagerV2

__all__ = [
    "DevicePool",
    "LayerGroupTraits",
    "build_page_layout",
    "default_layer_group_traits",
    "default_kv_heads",
    "default_layer_ids",
    "device_pools",
]


def build_page_layout(manager: KVCacheManagerV2) -> PageLayout:
    """The page layout of ``manager``. ``ValueError`` if its storage and declarations disagree."""
    tokens_per_block = int(manager.tokens_per_block)
    configs = layer_configs(manager)
    groups = [[int(layer) for layer in group] for group in manager.impl.layer_grouping]
    pool_group_of = [int(g) for g in manager.impl.get_life_cycle_pool_group_indices()]
    if len(pool_group_of) != len(groups):
        raise ValueError(f"{len(groups)} layer groups but {len(pool_group_of)} pool-group indices")
    for lg, layers in enumerate(groups):
        if not layers or layers[0] not in configs:
            raise ValueError(f"layer group {lg} has no layer config (layers {layers})")
    layer_ids = _layer_ids(manager, groups)
    traits = [manager.get_layer_group_traits(lg) for lg in range(len(groups))]
    untransferred = frozenset(str(role) for role in manager.get_untransferred_roles())

    stored = _stored_buffers(manager)
    buffers = []
    for b in stored:
        expansion = _expansion(configs[b.layer], b.role, tokens_per_block)
        try:
            segments = manager.get_buffer_layout(
                b.layer, b.role, b.size, tokens_per_block // expansion
            )
        except ValueError as error:
            raise ValueError(f"buffer ({b.layer}, {b.role}): {error}") from error
        buffers.append(
            BufferDesc(
                layer=layer_ids[b.layer][0],
                role=b.role,
                pool_group=b.pool_group,
                pool_index=b.pool_index,
                slot_offset=b.slot_offset,
                size=b.size,
                segments=segments,
                transferred=b.role not in untransferred,
                expansion=expansion,
            )
        )

    slot_bytes = _slot_bytes(manager)
    topology = _topology(manager)
    layer_groups = []
    total_kv_heads = {}
    for lg, layers in enumerate(groups):
        if pool_group_of[lg] not in slot_bytes:
            raise ValueError(f"layer group {lg} maps to pool group {pool_group_of[lg]}, no pools")
        group_buffers = [(s, d) for s, d in zip(stored, buffers) if s.layer_group == lg]
        attention = traits[lg].kind == "attention"
        kept, read, sink_tokens = _windows(manager, lg, configs) if attention else (None, None, 0)
        if attention:
            for layer in layers:
                total = manager.get_kv_heads(layer)[1]
                if total:
                    total_kv_heads[layer_ids[layer][0]] = total
        named = sorted(layer_ids[layer] for layer in layers)
        layer_groups.append(
            LayerGroupDesc(
                kind=traits[lg].kind,
                kept_window=kept,
                num_sink_blocks=-(-sink_tokens // tokens_per_block),
                pool_group=pool_group_of[lg],
                layers=tuple(global_layer for global_layer, _ in named),
                shard=_group_shard(
                    manager, grid_position(topology, traits[lg].grid), attention, group_buffers
                ),
                read_window=read,
                model_layers=tuple(model_layer for _, model_layer in named),
                grid=traits[lg].grid,
                unit=traits[lg].unit,
                blocks_by_cp=traits[lg].blocks_by_cp,
            )
        )

    return PageLayout(
        tokens_per_page=tokens_per_block,
        pool_slot_bytes=slot_bytes,
        layer_groups=tuple(layer_groups),
        buffers=tuple(buffers),
        topology=topology,
        total_kv_heads_by_layer=total_kv_heads,
        dtype=kv_dtype(manager),
        quant_mode=kv_quant_mode(manager),
    )


# -- the storage --------------------------------------------------------------------------------


@dataclass(frozen=True)
class _StoredBuffer:
    """Where the storage keeps one buffer: an internal layer's role in a slot of one pool."""

    layer_group: int
    layer: int
    role: str
    pool_group: int
    pool_index: int
    slot_offset: int
    size: int


def _stored_buffers(manager: KVCacheManagerV2) -> list[_StoredBuffer]:
    """Every buffer of the storage. Each layer group draws its slots from one pool group, as one
    variant of the group's slot; pool ``i`` of a slot holds the variant's ``i``-th coalesced
    buffer, whose members sit back to back."""
    stored = []
    for pool_group in manager.impl.pool_group_descs:
        for variant in pool_group.slot_desc.variants:
            for pool_index, coalesced in enumerate(variant.coalesced_buffers):
                size = int(coalesced.single_buffer_size)
                for j, buffer_id in enumerate(coalesced.buffer_ids):
                    stored.append(
                        _StoredBuffer(
                            layer_group=int(variant.layer_group_id),
                            layer=int(buffer_id.layer_id),
                            role=str(buffer_id.role),
                            pool_group=int(pool_group.pool_group_index),
                            pool_index=pool_index,
                            slot_offset=j * size,
                            size=size,
                        )
                    )
    return stored


def _slot_bytes(manager: KVCacheManagerV2) -> dict[int, tuple[int, ...]]:
    """Pool group -> the slot width of each of its pools, in pool order."""
    return {
        int(pool_group.pool_group_index): tuple(
            int(pool.slot_bytes) for pool in sorted(pool_group.pools, key=lambda p: p.pool_index)
        )
        for pool_group in manager.impl.pool_group_descs
    }


@dataclass(frozen=True)
class DevicePool:
    """One device pool of pool group ``group``: slot ``s`` lives at ``base + s * slot_bytes``."""

    group: int
    index: int
    base: int
    slot_bytes: int
    num_slots: int


def device_pools(manager: KVCacheManagerV2) -> dict[int, tuple[DevicePool, ...]]:
    """Pool group -> where each of its pools lies in device memory, in pool order."""
    return {
        int(pool_group.pool_group_index): tuple(
            DevicePool(
                group=int(pool_group.pool_group_index),
                index=int(pool.pool_index),
                base=int(pool.base_address),
                slot_bytes=int(pool.slot_bytes),
                num_slots=int(pool_group.num_slots),
            )
            for pool in sorted(pool_group.pools, key=lambda p: p.pool_index)
        )
        for pool_group in manager.impl.pool_group_descs
    }


def _expansion(
    config: AttentionLayerConfig | SsmLayerConfig, role: str, tokens_per_block: int
) -> int:
    """Sub-pages per block of a role's buffer: ``tokens_per_block`` over the layer config's
    ``tokens_per_block_override`` for the role, 1 without one."""
    for buffer in config.buffers:
        override = buffer.tokens_per_block_override
        if str(buffer.role) == role and override is not None:
            if tokens_per_block % int(override):
                raise ValueError(
                    f"buffer {role}: {override} tokens per block does not divide {tokens_per_block}"
                )
            return tokens_per_block // int(override)
    return 1


def layer_configs(manager: KVCacheManagerV2) -> dict[int, AttentionLayerConfig | SsmLayerConfig]:
    """Internal layer id -> layer config, preferring the manager's config to the runtime's. A layer
    without an explicit id uses its position in the config."""
    config = getattr(manager, "kv_cache_manager_py_config", None) or manager.impl.init_config
    return {
        int(getattr(layer, "layer_id", position)): layer
        for position, layer in enumerate(config.layers)
    }


def is_recurrent_layer(config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime describes a recurrent-state layer."""
    from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime

    return isinstance(config, runtime.SsmLayerConfig)


# -- the manager's facts --------------------------------------------------------------------------


def _layer_ids(manager: KVCacheManagerV2, groups: list[list[int]]) -> dict[int, tuple[int, int]]:
    """The manager's layer ids, checked to name every internal layer and no global layer twice."""
    layer_ids = {
        int(layer): (int(global_layer), int(model_layer))
        for layer, (global_layer, model_layer) in manager.get_layer_ids().items()
    }
    missing = sorted({layer for layers in groups for layer in layers} - layer_ids.keys())
    if missing:
        raise ValueError(f"the manager declares no global layer for internal layers {missing}")
    global_layers = [global_layer for global_layer, _ in layer_ids.values()]
    if len(set(global_layers)) != len(global_layers):
        raise ValueError("two internal layers share a global layer id")
    return layer_ids


def _windows(
    manager: KVCacheManagerV2, lg: int, configs: dict[int, AttentionLayerConfig]
) -> tuple[int | None, int | None, int]:
    """An attention layer group's (kept window, read window, sink tokens); a window of None is full
    attention. ``ValueError`` if the read window is not within the kept one."""
    first = configs[manager.impl.layer_grouping[lg][0]]
    kept = first.window_size
    read = manager._attention_read_window(lg)
    if (read is None) != (kept is None) or (read is not None and not 0 < read <= kept):
        raise ValueError(
            f"layer group {lg} reads a window of {read} tokens, which its kept window of {kept} "
            "tokens does not hold"
        )
    return kept, read, int(first.num_sink_tokens or 0)


@dataclass(frozen=True)
class LayerGroupTraits:
    """What a layer group is: ``kind`` ("attention" or "state", a label for readers), the parallel
    dimensions its split buffers spread over (``grid``), whether a request holds blocks by ordinal
    or one slot (``unit``), and whether, under Helix, each context-parallel rank holds its own
    blocks (``blocks_by_cp``); see ``LayerGroupDesc``."""

    kind: str
    grid: tuple[str, ...]
    unit: str
    blocks_by_cp: bool

    @classmethod
    def of_kind(cls, kind: str) -> "LayerGroupTraits":
        return cls(kind, *KIND_DEFAULTS[kind])


def default_layer_group_traits(manager: KVCacheManagerV2, layer_group: int) -> LayerGroupTraits:
    """A recurrent-state layer group (its layers' configs are ``SsmLayerConfig``) holds one slot
    per request and splits over tensor times context ranks; any other is attention, in blocks,
    split over the tensor-parallel ranks."""
    first = manager.impl.layer_grouping[layer_group][0]
    config = layer_configs(manager)[int(first)]
    return LayerGroupTraits.of_kind("state" if is_recurrent_layer(config) else "attention")


def default_layer_ids(manager: KVCacheManagerV2) -> dict[int, tuple[int, int]]:
    """Internal layer id -> (global layer id, model layer id), the same under any pipeline split:
    the model layer itself, or ``model_layer * num_types + type`` for a manager with several
    attention types per model layer."""
    internal = sorted({int(layer) for group in manager.impl.layer_grouping for layer in group})
    virtual = virtual_layers(manager)
    if virtual is None:
        return {layer: (int(manager.pp_layers[layer]),) * 2 for layer in internal}
    types_of, num_types = virtual
    return {
        layer: (types_of[layer][0] * num_types + types_of[layer][1], types_of[layer][0])
        for layer in internal
    }


def default_kv_heads(manager: KVCacheManagerV2, layer: int) -> tuple[int, int | None]:
    """(KV heads on this rank, KV heads over all tensor-parallel ranks or None if unknown) of an
    internal layer. A virtual layer counts the heads of its model layer; an internal layer past the
    local layers counts those of the first."""
    virtual = virtual_layers(manager)
    local = layer
    if virtual is not None and layer in virtual[0]:
        model_layer = virtual[0][layer][0]
        if model_layer in manager.pp_layers:
            local = list(manager.pp_layers).index(model_layer)
    per_layer = list(getattr(manager, "num_kv_heads_per_layer", None) or ())
    per_rank = int(per_layer[local if local < len(per_layer) else 0] or 0) if per_layer else 0
    total = getattr(manager, "num_kv_heads", None)
    if isinstance(total, int):
        return per_rank, total
    if total is None or virtual is not None or layer >= len(manager.pp_layers):
        return per_rank, None
    value = total[int(manager.pp_layers[layer])]
    return per_rank, None if value is None else int(value)


def virtual_layers(manager: KVCacheManagerV2) -> tuple[dict[int, tuple[int, int]], int] | None:
    """(internal id -> (model layer, attention type), type count), or None without virtual layers.
    An internal id shared by several types takes its first: under DeepSeek-V4.1 long-range layers
    sharing a compressed cache resolve to the layer that allocated it, which comes first in layer
    order on every pipeline stage."""
    virtual = getattr(manager, "_layer_attn_to_layer_id", None)
    if not virtual:
        return None
    types_of: dict[int, tuple[int, int]] = {}
    for (model_layer, attn_type), layer in virtual.items():
        types_of.setdefault(int(layer), (int(model_layer), int(attn_type.value)))
    # The whole enum counts, so stages holding different attention types agree.
    return types_of, max(int(member.value) for member in type(next(iter(virtual))[1])) + 1


def helix(manager: KVCacheManagerV2) -> tuple[int, int]:
    """Helix ``(rank, size)``, or ``(0, 1)``: block ``b`` spans ``size`` pages of the manager's
    ``tokens_per_block`` tokens, rank ``j``'s from ``(b * size + j) * tokens_per_block``."""
    if not getattr(manager, "_has_cp_helix", False):
        return 0, 1
    return int(manager._helix_cp_rank), int(manager._helix_cp_size)


def kv_dtype(manager: KVCacheManagerV2) -> str | None:
    """The element type of the manager's KV data by name, such as ``"BF16"`` or ``"NVFP4"``;
    ``None`` for a manager declaring none."""
    dtype = getattr(manager, "dtype", None)
    return None if dtype is None else str(getattr(dtype, "name", dtype))


def kv_quant_mode(manager: KVCacheManagerV2) -> str | None:
    """How the manager quantizes its KV data: ``None`` unquantized, else ``"fp8"``, ``"nvfp4"``, or
    ``"fp8_ds_mla"``, DeepSeek-V4's FP8 pages carrying their scales."""
    if getattr(manager, "use_fp8_ds_mla", False):
        return "fp8_ds_mla"
    return {"FP8": "fp8", "NVFP4": "nvfp4"}.get(kv_dtype(manager))


# -- where a rank stands --------------------------------------------------------------------------


def _topology(manager: KVCacheManagerV2) -> RankTopology:
    """The rank's place as native's ``RankInfo`` reads it from the mapping: under attention data
    parallelism each tensor-parallel rank is a data-parallel rank of its own."""
    mapping = getattr(manager, "mapping", None)
    if mapping is None:
        return RankTopology()
    adp = bool(mapping.enable_attention_dp)
    return RankTopology(
        tp_size=mapping.tp_size,
        tp_rank=mapping.tp_rank,
        pp_size=mapping.pp_size,
        pp_rank=mapping.pp_rank,
        cp_size=mapping.cp_size,
        cp_rank=mapping.cp_rank,
        dp_size=mapping.tp_size if adp else mapping.dp_size,
        dp_rank=mapping.tp_rank if adp else 0,
        attention_dp=adp,
    )


def _head_share(size: int, rank: int, per_rank: int, total: int | None) -> ShardDesc:
    """The share of a head-split KV buffer this rank holds: ``total`` heads over a grid of ``size``
    ranks, rank ``r`` holding ``held_span(per_rank, size, r, total)``. Ranks holding the same heads
    hold the same share; when some rank's heads straddle two shares, or the heads are unknown,
    every rank's buffer is a share of its own."""
    if per_rank <= 0 or not total:
        return ShardDesc(size, rank)
    starts = [held_span(per_rank, size, r, total)[0] for r in range(size)]
    if any(start % per_rank for start in starts):
        return ShardDesc(size, rank)
    count = max(total // per_rank, 1)
    return ShardDesc(count, min(starts[rank] // per_rank, count - 1))


def _group_shard(
    manager: KVCacheManagerV2,
    position: tuple[int, int],
    attention: bool,
    buffers: list[tuple[_StoredBuffer, BufferDesc]],
) -> ShardDesc:
    """The share of a layer group's content this rank holds, at ``position`` (size, rank) in the
    group's grid: the share of its split, transferred buffers -- for attention K/V the share of the
    KV heads -- the whole content when every rank holds them alike, and this rank's own share when
    they disagree."""
    size, rank = position
    if size == 1:
        return ShardDesc()
    shares = set()
    for stored, desc in buffers:
        if not desc.transferred or all(seg.split_axis is None for seg in desc.segments):
            continue
        share = (
            _head_share(size, rank, *manager.get_kv_heads(stored.layer))
            if attention
            else ShardDesc(size, rank)
        )
        if share.count != 1:
            shares.add(share)
    if not shares:
        return ShardDesc()
    return shares.pop() if len(shares) == 1 else ShardDesc(size, rank)
