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

import dataclasses
import enum
import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from ..layout_builder import DevicePool, device_pools
from ..page_layout import KIND_DEFAULTS, BufferDesc, PageLayout, Segment
from . import _manager

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig

    from ..kv_cache_manager_v2 import KVCacheManagerV2

# A new field left out at its default keeps the version; changing the encoding or the meaning of
# an existing field bumps it.
LAYOUT_ID_VERSION = 4


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


def _segments_document(buffer: BufferDesc) -> list[dict] | None:
    """The buffer's segments, or None for one segment of the whole buffer that every rank holds
    alike."""
    if buffer.segments == (Segment(buffer.size),):
        return None
    return [
        {
            "bytes": int(segment.nbytes),
            "axes": [[name, int(extent)] for name, extent in segment.axes],
            "split": segment.split_axis,
        }
        for segment in buffer.segments
    ]


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
    segments = _segments_document(b)
    if segments is not None:
        doc["segments"] = segments
    if not b.transferred:
        doc["transfer"] = False
    if int(b.expansion) != 1:
        doc["expansion"] = int(b.expansion)
    return doc


def layout_document(layout: PageLayout) -> dict:
    """Hash canonical global layer groups, slot widths, kinds, how each group splits and buffer
    geometry, but no addresses, windows, sinks, shards, topology, head counts, dtypes or local
    region numbering."""
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
        # How the group splits, only where it departs from its kind's defaults.
        for name, value in zip(("grid", "unit", "blocks_by_cp"), KIND_DEFAULTS[desc.kind]):
            if getattr(desc, name) != value:
                entry[name] = list(getattr(desc, name)) if name == "grid" else getattr(desc, name)
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


def _sparse(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime has sparse buffers (``is_sparse``), whose read-only
    pages a cache can lock in host memory."""
    return any(getattr(b, "is_sparse", False) for b in getattr(layer_config, "buffers", ()) or ())


def derive_layout(manager: KVCacheManagerV2) -> ManagerLayout:
    """The layout of a ``KVCacheManagerV2`` as the lender needs it: the manager's page layout and
    its device pools. ``ValueError`` if its layer groups, layer configs and declarations do not
    agree."""
    page_layout = manager.get_page_layout()
    if callable(getattr(manager, "rebuild_disagg_imported_cache", None)):
        # The manager rebuilds the roles it omits only after an import; lending invokes no
        # rebuild, so every role travels.
        page_layout = dataclasses.replace(
            page_layout,
            buffers=tuple(
                dataclasses.replace(buffer, transferred=True) for buffer in page_layout.buffers
            ),
        )
    layer_cfg = _manager.layer_configs(manager)
    grouping = _manager.layer_group_layer_ids(manager)
    pools = device_pools(manager)
    groups = page_layout.layer_groups
    cp_rank, cp_size = _manager.helix(manager)
    return ManagerLayout(
        page_layout=page_layout,
        tokens_per_block=page_layout.tokens_per_page * cp_size,
        pool_group_of=tuple(desc.pool_group for desc in groups),
        windows=tuple(desc.kept_window for desc in groups),
        sink_blocks=tuple(desc.num_sink_blocks for desc in groups),
        recurrent=tuple(desc.kind == "state" for desc in groups),
        sparse=tuple(_sparse(layer_cfg[layer_ids[0]]) for layer_ids in grouping),
        layers=tuple(desc.layers for desc in groups),
        shards=tuple((desc.shard.count, desc.shard.index) for desc in groups),
        page_bytes={g: sum(p.slot_bytes for p in group) for g, group in pools.items()},
        device_pools=pools,
        cp=(cp_rank, cp_size),
        read_windows=tuple(desc.read_window for desc in groups),
        named_groups=tuple(
            not any(_manager.is_private_layer(layer_cfg[lid]) for lid in group)
            for group in grouping
        ),
    )
