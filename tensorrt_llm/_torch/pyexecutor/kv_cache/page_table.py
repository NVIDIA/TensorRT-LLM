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
"""A ``KVCacheManagerV2``'s page table as plain values: how its pages are laid out (``PageLayout``),
where its device pools lie and what each layer group is; and what a layer config or a rank's place
says about it. The manager builds the table itself (``KVCacheManagerV2.get_page_table``); readers
of the table, such as the lender, never look into the manager."""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from .page_layout import PageLayout, RankTopology, layout_from_dict, layout_to_dict

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig


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
    """A manager's page table as plain values, without a manager reference; ``cp`` is (rank, size),
    (0, 1) without Helix. ``tokens_per_block`` spans ``size`` physical pages of
    ``tokens_per_block // size`` tokens."""

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
    # Per layer group, whether every layer must cover a reused prefix; ``()``: every group's.
    prefix_required: tuple[bool, ...] = field(default=(), kw_only=True)
    # Per layer group, its (internal layer id, global layer id) pairs in storage order.
    local_layers: tuple[tuple[tuple[int, int], ...], ...] = field(default=(), kw_only=True)
    # Per layer group, the KV heads on this rank of its first layer.
    kv_heads: tuple[int, ...] = field(default=(), kw_only=True)
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


def page_table_to_dict(table: ManagerLayout) -> dict:
    """``table`` as plain values (dicts, lists, ints, strings) for a peer, which
    ``page_table_from_dict`` reads back; its page layout as ``layout_to_dict`` writes it."""
    data: dict[str, Any] = {}
    for f in dataclasses.fields(table):
        value = getattr(table, f.name)
        if f.name == "page_layout":
            value = layout_to_dict(value)
        elif f.name == "device_pools":
            value = {int(g): [list(dataclasses.astuple(p)) for p in ps] for g, ps in value.items()}
        elif f.name == "page_bytes":
            value = {int(g): int(n) for g, n in value.items()}
        else:
            value = _lists(value)
        data[f.name] = value
    return data


def page_table_from_dict(data: Mapping) -> ManagerLayout:
    """The page table ``page_table_to_dict`` wrote."""
    values: dict[str, Any] = {}
    for f in dataclasses.fields(ManagerLayout):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "page_layout":
            value = layout_from_dict(value)
        elif f.name == "device_pools":
            value = {int(g): tuple(DevicePool(*p) for p in ps) for g, ps in value.items()}
        elif f.name == "page_bytes":
            value = {int(g): int(n) for g, n in value.items()}
        else:
            value = _tuples(value)
        values[f.name] = value
    return ManagerLayout(**values)


def _lists(value: Any) -> Any:
    return [_lists(v) for v in value] if isinstance(value, (tuple, list)) else value


def _tuples(value: Any) -> Any:
    return tuple(_tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


def virtual_layers(
    attn_to_layer_id: Mapping[tuple[int, enum.Enum], int] | None,
) -> tuple[dict[int, tuple[int, int]], int] | None:
    """(internal id -> (model layer, attention type), type count) of a manager keeping several
    caches per model layer, from its (model layer, attention type) -> internal id map; None for an
    empty map. An internal id shared by several types takes its first: under DeepSeek-V4.1
    long-range layers sharing a compressed cache resolve to the layer that allocated it, which
    comes first on every pipeline stage. Every member of the attention-type enum counts, so stages
    holding different attention types agree."""
    if not attn_to_layer_id:
        return None
    types_of: dict[int, tuple[int, int]] = {}
    for (model_layer, attn_type), layer in attn_to_layer_id.items():
        types_of.setdefault(int(layer), (int(model_layer), int(attn_type.value)))
    kinds = type(next(iter(attn_to_layer_id))[1])
    return types_of, max(int(member.value) for member in kinds) + 1


# -- what a layer config says ---------------------------------------------------------------------


def layer_configs_by_id(config: object) -> dict[int, AttentionLayerConfig | SsmLayerConfig]:
    """Internal layer id -> layer config of a runtime config; a layer without an explicit id uses
    its position in the config."""
    return {
        int(getattr(layer, "layer_id", position)): layer
        for position, layer in enumerate(config.layers)
    }


def is_recurrent_layer(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime describes a recurrent-state layer."""
    from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime

    return isinstance(layer_config, runtime.SsmLayerConfig)


def is_private_layer(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether the native lifecycle excludes this layer from prefix reuse."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy

    return getattr(layer_config, "reuse_policy", None) == AttentionReusePolicy.PRIVATE


def is_non_required_layer(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether prefix coverage is optional or private for this layer."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy

    return (
        getattr(layer_config, "reuse_policy", AttentionReusePolicy.REQUIRED)
        != AttentionReusePolicy.REQUIRED
    )


def has_sparse_buffers(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime has sparse buffers (``is_sparse``), whose read-only
    pages a cache can lock in host memory."""
    return any(getattr(b, "is_sparse", False) for b in getattr(layer_config, "buffers", ()) or ())


def buffer_expansions(
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


# -- where a rank stands and what it holds --------------------------------------------------------


def rank_topology(mapping: object) -> RankTopology:
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
