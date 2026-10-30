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
"""Object keys and region labels instances exchange; equal keys mean interchangeable bytes given
scope, manager and ``attach_lender``'s obligations. Format changes bump ``KEY_FORMAT_VERSION``."""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

from ..cache_layout.page_layout import ShardDesc
from ._layout import canonical_order

KEY_FORMAT_VERSION = 2
REUSE_KEY_BYTES = 32
_NAMESPACE_BYTES = 16
_GROUP_BYTES = 6  # the layer group's canonical index, shard count and shard index, u16 each
OBJECT_KEY_BYTES = _NAMESPACE_BYTES + REUSE_KEY_BYTES + _GROUP_BYTES
_REGION_LAYOUT_HEX = 16


def _u16(value: int) -> bytes:
    return int(value).to_bytes(2, "big")


def namespace(scope: bytes, layout_id: bytes) -> bytes:
    """The 16 bytes every object key starts with, hashed from length-prefixed ``scope`` and
    ``layout_id``; ``ValueError`` for an empty ``layout_id`` or a ``scope`` over 65535 bytes."""
    if not layout_id:
        raise ValueError("a namespace without layout_id would mix incompatible bytes")
    if len(scope) > 0xFFFF:
        raise ValueError(f"scope is {len(scope)} bytes; at most 65535 fit its length prefix")
    digest = hashlib.sha256(b"trtllm-kv-object" + _u16(KEY_FORMAT_VERSION))
    digest.update(_u16(len(scope)) + scope)
    digest.update(_u16(len(layout_id)) + layout_id)
    return digest.digest()[:_NAMESPACE_BYTES]


class Identity:
    """Object keys and region labels for one manager. ``layers[g]`` holds layer group ``g``'s global
    layer ids and ``shards[g]`` the ``(count, index)`` share of its content this rank holds."""

    namespace: bytes
    layout_id: bytes
    canonical: list[int]

    def __init__(
        self,
        scope: bytes,
        layout_id: bytes,
        layers: Sequence[Sequence[int]],
        shards: Sequence[tuple[int, int]],
    ) -> None:
        if len(shards) != len(layers):
            raise ValueError(f"{len(layers)} layer groups but {len(shards)} shards")
        self.layout_id = bytes(layout_id)
        self.namespace = namespace(bytes(scope), self.layout_id)
        self.canonical = canonical_order(layers)
        self._groups = []
        for index, (count, share) in zip(self.canonical, shards):
            ShardDesc(count, share)  # ValueError for a share outside the count
            suffix = _u16(index) + _u16(count) + _u16(share)
            self._groups.append(suffix)

    def object_keys(self, layer_group_index: int, reuse_keys: np.ndarray) -> np.ndarray:
        """``uint8 (n, OBJECT_KEY_BYTES)``: the namespace, each row's 32-byte reuse key, then
        the group's canonical index, shard count and shard index as big-endian ``u16``."""
        if not 0 <= layer_group_index < len(self._groups):
            raise ValueError(f"unknown layer group {layer_group_index}")
        reuse_keys = np.ascontiguousarray(reuse_keys, dtype=np.uint8).reshape(-1, REUSE_KEY_BYTES)
        suffix = self._groups[layer_group_index]
        payload = b"".join(piece for key in reuse_keys for piece in (self.namespace, key, suffix))
        return np.frombuffer(payload, dtype=np.uint8).reshape(-1, OBJECT_KEY_BYTES)

    def region_label(self, layer_group_indices: Iterable[int]) -> str:
        """The layout-derived label for a staging region serving ``layer_group_indices``:
        equal on instances laid out alike, independent of addresses, capacity and scope."""
        groups = sorted({self.canonical[int(g)] for g in layer_group_indices})
        return f"{self.layout_id.hex()[:_REGION_LAYOUT_HEX]}:lg{'+'.join(str(g) for g in groups)}"

    def device_region_label(self, layer_group_indices: Iterable[int], pool_index: int) -> str:
        """The label of pool ``pool_index``'s device region serving ``layer_group_indices``: equal
        on instances laid out alike, distinct from every staging label and from other pools'."""
        return f"{self.region_label(layer_group_indices)}:pool{int(pool_index)}"
