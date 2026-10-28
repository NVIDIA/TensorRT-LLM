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
"""A byte oracle for page layouts.

Each rank's page holds, at every byte of every tensor, the id of the global element there. After
every source rank sends what ``layout_mapping.map_buffers`` gives it, every byte a destination rank
holds of every buffer the sources transfer must have arrived exactly once, holding its own global
element, and only from source ranks the destination pairs with (``layout_mapping.fan``).

Use it on hand-written layouts, or on the page layouts real managers build
(``manager.get_page_table().page_layout``) to check a new cache's declarations across the
topologies it is served with.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import fan, map_buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    GRID_DIMENSIONS,
    BufferDesc,
    PageLayout,
    RankTopology,
    Tensor,
    grid_place,
    held,
)

Pool = Tuple[int, int]  # (pool group, pool index)


class Elements:
    """Global element ids, allotted per (layer, role, tensor name) in the order first seen."""

    def __init__(self) -> None:
        self._bases: Dict[tuple, int] = {}
        self._next = 0

    def _base(self, buffer: BufferDesc, tensor: Tensor) -> int:
        key = (buffer.layer, buffer.role, tensor.name)
        if key not in self._bases:
            self._bases[key] = self._next
            self._next += int(np.prod([a.extent for a in tensor.axes]))
        return self._bases[key]

    def of(
        self, page: PageLayout, buffer: BufferDesc, tensor: Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The elements of the tensor's local piece.

        Returns:
            Each element's byte offset within the pool slot, and its global element id.
        """
        offsets = np.array([buffer.slot_offset + tensor.offset], dtype=np.int64)
        ids = np.zeros(1, dtype=np.int64)
        for axis in tensor.axes:
            h = held(axis, page.topology)
            local = np.arange(h.count, dtype=np.int64)
            offsets = (offsets[:, None] + local[None, :] * axis.stride).ravel()
            ids = (ids[:, None] * axis.extent + (h.start + h.step * local)[None, :]).ravel()
        return offsets, ids + self._base(buffer, tensor)

    def memory(self, page: PageLayout) -> Dict[Pool, np.ndarray]:
        """One pool slot of each pool, each byte holding its element id (-1 where none lies)."""
        mem = {
            (g, i): np.full(width, -1, dtype=np.int64)
            for g, widths in page.pool_slot_bytes.items()
            for i, width in enumerate(widths)
        }
        for buffer in page.buffers:
            for tensor in buffer.tensors:
                offsets, ids = self.of(page, buffer, tensor)
                mem[buffer.pool_group, buffer.pool_index][offsets] = ids
        return mem


def paired(src: RankTopology, dst: RankTopology) -> bool:
    """Whether the source rank's share overlaps the destination's along every dimension."""
    for dim in GRID_DIMENSIONS:
        src_size, src_rank = grid_place(src, (dim,))
        dst_size, dst_rank = grid_place(dst, (dim,))
        if src_rank not in fan(src_size, dst_size, dst_rank):
            return False
    return True


def assert_layouts_reshard(
    sources: Sequence[PageLayout], destinations: Sequence[PageLayout]
) -> None:
    """Check the byte oracle for one request.

    Args:
        sources: The page layouts of every source rank serving the request: one attention
            data-parallel group, all pipeline stages and context ranks.
        destinations: Page layouts of destination ranks, each checked against all sources.
    """
    elements = Elements()
    source_memory = [elements.memory(page) for page in sources]
    sent = {(b.layer, b.role) for page in sources for b in page.buffers if b.transferred}
    for dst in destinations:
        got = {
            (g, i): np.full(width, -1, dtype=np.int64)
            for g, widths in dst.pool_slot_bytes.items()
            for i, width in enumerate(widths)
        }
        writes = {pool: np.zeros(arr.size, dtype=np.int64) for pool, arr in got.items()}
        for src, memory in zip(sources, source_memory):
            maps = map_buffers(src, dst)
            if maps:
                assert paired(src.topology, dst.topology), (src.topology, dst.topology)
            for runs in maps:
                src_pool = (runs.src.pool_group, runs.src.pool_index)
                dst_pool = (runs.dst.pool_group, runs.dst.pool_index)
                for s, d, n in zip(runs.src_offsets, runs.dst_offsets, runs.sizes):
                    got[dst_pool][d : d + n] = memory[src_pool][s : s + n]
                    writes[dst_pool][d : d + n] += 1
        for buffer in dst.buffers:
            if not buffer.transferred or (buffer.layer, buffer.role) not in sent:
                continue
            pool = (buffer.pool_group, buffer.pool_index)
            for tensor in buffer.tensors:
                offsets, ids = elements.of(dst, buffer, tensor)
                where = (
                    f"{dst.topology} buffer ({buffer.layer}, {buffer.role}) tensor {tensor.name!r}"
                )
                miscount = np.flatnonzero(writes[pool][offsets] != 1)
                assert not miscount.size, (
                    f"{where} byte {offsets[miscount[0]]}: "
                    f"{writes[pool][offsets[miscount[0]]]} writes"
                )
                wrong = np.flatnonzero(got[pool][offsets] != ids)
                assert not wrong.size, (
                    f"{where} byte {offsets[wrong[0]]}: element {got[pool][offsets[wrong[0]]]}, "
                    f"not {ids[wrong[0]]}"
                )
