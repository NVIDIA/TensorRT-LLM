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
"""Names of the rows and staging parts instances exchange; equal names mean interchangeable bytes.
The format is a contract between instances: changing it bumps ``KEY_FORMAT_VERSION``."""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ._layout import canonical_order

KEY_FORMAT_VERSION = 2
KEY_BYTES = 32
_NAMESPACE_BYTES = 16
_GROUP_BYTES = 6  # the layer group's canonical index, shard count and shard index, u16 each
NAME_BYTES = _NAMESPACE_BYTES + KEY_BYTES + _GROUP_BYTES
_PART_LAYOUT_HEX = 16


def _u16(value: int) -> bytes:
    return int(value).to_bytes(2, "big")


def namespace(scope: bytes, layout_id: bytes) -> bytes:
    """The 16 bytes every name starts with, from the caller's ``scope`` and the block layout's
    ``layout_id`` (each length-prefixed). ``ValueError`` for an empty ``layout_id`` or a ``scope``
    longer than 65535 bytes."""
    if not layout_id:
        raise ValueError("a namespace without layout_id would mix incompatible bytes")
    if len(scope) > 0xFFFF:
        raise ValueError(f"scope is {len(scope)} bytes; at most 65535 fit its length prefix")
    digest = hashlib.sha256(b"trtllm-kv-object" + _u16(KEY_FORMAT_VERSION))
    digest.update(_u16(len(scope)) + scope)
    digest.update(_u16(len(layout_id)) + layout_id)
    return digest.digest()[:_NAMESPACE_BYTES]


class Identity:
    """Names of one manager's rows and staging parts. ``layers[g]`` holds layer group ``g``'s global
    layer ids and ``shards[g]`` the ``(count, index)`` share of its content this rank holds."""

    namespace: bytes
    layout_id: bytes
    canonical: List[int]

    def __init__(
        self,
        scope: bytes,
        layout_id: bytes,
        layers: Sequence[Sequence[int]],
        shards: Sequence[Tuple[int, int]],
    ) -> None:
        if len(shards) != len(layers):
            raise ValueError(f"{len(layers)} layer groups but {len(shards)} shards")
        self.layout_id = bytes(layout_id)
        self.namespace = namespace(bytes(scope), self.layout_id)
        self.canonical = canonical_order(layers)
        self._namespace = np.frombuffer(self.namespace, dtype=np.uint8)
        self._groups = []
        for index, (count, share) in zip(self.canonical, shards):
            if count < 1 or not 0 <= share < count:
                raise ValueError(f"shard {share} of {count}")
            suffix = _u16(index) + _u16(count) + _u16(share)
            self._groups.append(np.frombuffer(suffix, dtype=np.uint8))

    def names(self, layer_group: int, keys: np.ndarray) -> np.ndarray:
        """``uint8 (n, NAME_BYTES)``: the namespace, each row's 32-byte block key, then the layer
        group's canonical index, shard count and shard index as big-endian ``u16``."""
        if not 0 <= layer_group < len(self._groups):
            raise ValueError(f"unknown layer group {layer_group}")
        keys = np.ascontiguousarray(keys, dtype=np.uint8).reshape(-1, KEY_BYTES)
        out = np.empty((keys.shape[0], NAME_BYTES), dtype=np.uint8)
        out[:, :_NAMESPACE_BYTES] = self._namespace
        out[:, _NAMESPACE_BYTES : _NAMESPACE_BYTES + KEY_BYTES] = keys
        out[:, _NAMESPACE_BYTES + KEY_BYTES :] = self._groups[layer_group]
        return out

    def part_name(self, layer_groups: Iterable[int]) -> str:
        """The name of the staging part serving the local ``layer_groups``: equal on instances laid
        out alike."""
        groups = sorted({self.canonical[int(g)] for g in layer_groups})
        return f"{self.layout_id.hex()[:_PART_LAYOUT_HEX]}:lg{'+'.join(str(g) for g in groups)}"
