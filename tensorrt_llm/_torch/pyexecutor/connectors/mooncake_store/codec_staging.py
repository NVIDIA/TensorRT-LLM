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
"""Pinned staging for pages stored in the cache's cold-page encoding.

With a `KvCachePageCodec`, a stored page is not the GPU page's bytes but the
codec's encoding of them, so it cannot be read or written in place. Each page
passes through an arena of registered pinned memory instead: encoded from its
GPU slot into the arena before a put, decoded back out of it after a get.

The codec addresses a batch as ``base + slot * page_bytes(layer group)``, so a
batch is laid out one layer group at a time, each group's pages packed at that
group's encoded size from an aligned base. The store sees one buffer per page.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "ENCODED_GROUP_ALIGNMENT",
    "EncodedBatch",
    "EncodedGroup",
    "EncodedStagingArena",
    "plan_encoded_batches",
]

#: Alignment of each layer group's region in a batch. Codecs write in wide
#: vectors (NVFP4 moves 16 bytes at a time), and 256 keeps every region on a
#: boundary any of them could ask for.
ENCODED_GROUP_ALIGNMENT = 256


def _align(value: int, alignment: int = ENCODED_GROUP_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class EncodedGroup:
    """One layer group's pages in a batch: what the codec is called with."""

    layer_group_id: int
    #: Address of the group's first slot in the arena.
    base: int
    #: Arena slot of each page, counted from `base` in units of the encoded size.
    slots: List[int] = field(default_factory=list)
    #: GPU page slot of each page, matching `slots`.
    pages: List[int] = field(default_factory=list)


@dataclass
class EncodedBatch:
    """Pages moved by one store call, in the order the caller listed them."""

    keys: List[str] = field(default_factory=list)
    #: Arena address and encoded size of each page, matching `keys`.
    addresses: List[int] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    groups: List[EncodedGroup] = field(default_factory=list)


def plan_encoded_batches(
    items: Sequence[Tuple[str, int, int]],
    page_bytes: Callable[[int], int],
    arena_base: int,
    arena_bytes: int,
    max_pages: int,
) -> Iterator[EncodedBatch]:
    """Split pages into batches that each fit the arena, and lay each one out.

    Args:
        items: `(key, layer group, GPU page slot)` per page, in transfer order.
        page_bytes: Encoded page size of a layer group.
        arena_base: First byte of the arena.
        arena_bytes: Arena capacity.
        max_pages: Most pages per batch, the store call's batch size.

    Yields:
        Batches whose pages keep their order from `items`, the last one possibly
        short. A page larger than the whole arena raises, since no batch could
        ever hold it.
    """
    if max_pages <= 0:
        raise ValueError(f"max_pages must be > 0, got {max_pages}")
    pending: List[Tuple[str, int, int]] = []
    group_bytes: Dict[int, int] = {}

    def footprint(groups: Dict[int, int]) -> int:
        return sum(_align(size) for size in groups.values())

    for key, layer_group_id, page in items:
        size = page_bytes(layer_group_id)
        if _align(size) > arena_bytes:
            raise ValueError(
                f"an encoded page of layer group {layer_group_id} is {size} B, more "
                f"than the {arena_bytes} B staging arena"
            )
        trial = dict(group_bytes)
        trial[layer_group_id] = trial.get(layer_group_id, 0) + size
        if pending and (len(pending) >= max_pages or footprint(trial) > arena_bytes):
            yield _lay_out(pending, page_bytes, arena_base)
            pending, trial = [], {layer_group_id: size}
        pending.append((key, layer_group_id, page))
        group_bytes = trial
    if pending:
        yield _lay_out(pending, page_bytes, arena_base)


def _lay_out(
    pending: Sequence[Tuple[str, int, int]],
    page_bytes: Callable[[int], int],
    arena_base: int,
) -> EncodedBatch:
    groups: Dict[int, EncodedGroup] = {}
    counts: Dict[int, int] = {}
    for _, layer_group_id, _ in pending:
        counts[layer_group_id] = counts.get(layer_group_id, 0) + 1
    cursor = arena_base
    for layer_group_id, count in counts.items():
        groups[layer_group_id] = EncodedGroup(layer_group_id=layer_group_id, base=cursor)
        cursor += _align(count * page_bytes(layer_group_id))
    batch = EncodedBatch(groups=list(groups.values()))
    for key, layer_group_id, page in pending:
        group = groups[layer_group_id]
        slot = len(group.slots)
        size = page_bytes(layer_group_id)
        group.slots.append(slot)
        group.pages.append(page)
        batch.keys.append(key)
        batch.addresses.append(group.base + slot * size)
        batch.sizes.append(size)
    return batch


class EncodedStagingArena:
    """A registered pinned buffer encoded pages pass through, one per direction.

    Loads run on the executor thread and saves on the connector's background
    thread, so each direction owns an arena and neither takes a lock.
    """

    def __init__(self, *, capacity_bytes: int, store, label: str):
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be > 0, got {capacity_bytes}")
        self._store = store
        self._label = label
        # Pinned so the store can register it and the codec reach it from the
        # GPU; both read and write it while no Python object refers to it.
        pin = torch.cuda.is_available()
        self._buffer = torch.empty(int(capacity_bytes), dtype=torch.uint8, pin_memory=pin)
        self._base = int(self._buffer.data_ptr())
        status = store.register_buffer(self._base, self._buffer.numel())
        if status != 0:
            raise RuntimeError(
                f"MooncakeDistributedStore.register_buffer failed with status {status} "
                f"for the {label} encoded-page staging arena at "
                f"[{self._base:#x}, {self._base + self._buffer.numel():#x}). Host memory "
                "registration failing points at the pool or the fabric."
            )
        logger.info(
            f"mooncake-store {label} encoded staging: {self._buffer.numel() / 1024**2:.1f} MiB "
            f"pinned={pin}"
        )

    @property
    def base(self) -> int:
        """Address of the arena's first byte."""
        return self._base

    @property
    def capacity(self) -> int:
        """Arena size in bytes."""
        return 0 if self._buffer is None else int(self._buffer.numel())

    def close(self) -> None:
        """Hand the registration back before the buffer is freed. Idempotent."""
        if self._buffer is None:
            return
        status = self._store.unregister_buffer(self._base)
        if status != 0:
            raise RuntimeError(
                f"MooncakeDistributedStore.unregister_buffer failed with status {status} "
                f"for the {self._label} encoded-page staging arena at {self._base:#x}. "
                "The buffer is kept alive rather than freed."
            )
        self._buffer = None
