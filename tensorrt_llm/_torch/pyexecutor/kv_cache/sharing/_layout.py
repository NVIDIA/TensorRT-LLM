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
"""The layout of a manager's blocks as the lender needs it, the page table the manager builds, and
``layout_id``, the digest of how a block's bytes are laid out."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import struct
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from ..page_layout import BufferDesc, PageLayout, replicated
from ..page_table import DevicePool, ManagerLayout  # noqa: F401  DevicePool: the lender's pools

if TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2

# A new field left out at its default keeps the version; changing the encoding or the meaning of
# an existing field bumps it.
LAYOUT_ID_VERSION = 5


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


def _tensors_document(buffer: BufferDesc) -> list[dict] | None:
    """The buffer's tensors, or None for the whole buffer alike on every rank."""
    if buffer.tensors == replicated(buffer.size):
        return None
    out = []
    for tensor in buffer.tensors:
        axes = []
        for axis in tensor.axes:
            doc = {"name": axis.name, "extent": int(axis.extent), "stride": int(axis.stride)}
            if axis.split:
                doc["split"] = list(axis.split)
            if axis.held is not None:
                doc["held"] = int(axis.held)
            if axis.cyclic:
                doc["cyclic"] = True
            axes.append(doc)
        doc = {"axes": axes}
        if tensor.name:
            doc["name"] = tensor.name
        if tensor.offset:
            doc["offset"] = int(tensor.offset)
        out.append(doc)
    return out


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
    tensors = _tensors_document(b)
    if tensors is not None:
        doc["tensors"] = tensors
    if not b.transferred:
        doc["transfer"] = False
    if int(b.expansion) != 1:
        doc["expansion"] = int(b.expansion)
    return doc


def layout_document(layout: PageLayout) -> dict:
    """Hash canonical global layer groups, the model layer of each global layer where it is not
    the global layer itself, slot widths, kinds and each buffer's tensors, but no addresses,
    windows, sinks, shards, topology, total head counts, dtypes or local region numbering. Two
    pipeline stages naming their layers alike but holding other model layers differ."""
    canonical = canonical_order([desc.layers for desc in layout.layer_groups])
    groups = [None] * len(layout.layer_groups)
    for local, desc in enumerate(layout.layer_groups):
        widths = layout.pool_slot_bytes.get(desc.pool_group)
        if widths is None:
            raise ValueError(
                f"layer group {local} maps to pool group {desc.pool_group}, which is not in the "
                "layout"
            )
        named = sorted(zip(desc.layers, desc.model_layers or desc.layers))
        entry = {
            "layers": [int(layer) for layer, _ in named],
            "slot_bytes": [int(w) for w in widths],
        }
        if any(layer != model for layer, model in named):
            entry["model_layers"] = [int(model) for _, model in named]
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


def derive_layout(manager: KVCacheManagerV2) -> ManagerLayout:
    """The page table the manager builds (``KVCacheManagerV2.get_page_table``), as a lender
    reads it: buffers the manager rebuilds after a native import travel too, since lending invokes
    no rebuild."""
    table = manager.get_page_table()
    layout = table.page_layout
    if not layout.rebuilds_untransferred:
        return table
    buffers = tuple(dataclasses.replace(b, transferred=True) for b in layout.buffers)
    return dataclasses.replace(table, page_layout=dataclasses.replace(layout, buffers=buffers))
