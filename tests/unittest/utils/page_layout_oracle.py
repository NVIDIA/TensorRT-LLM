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
element, and only from source ranks the destination pairs with (``layout_mapping.fan``); no run
reaches past a pool slot, and no other byte of the destination's slots is written. A buffer the
destination transfers of a model layer the sources hold, of a role they transfer none of for it,
is one the destination rebuilds per request (``PageLayout.request_rebuilt_roles``).

Use it on hand-written layouts, or on the page layouts real managers build
(``adapter_for(manager).page_table().page_layout``) to check a new cache's declarations across the
topologies it is served with. It checks that the ranks' declarations agree with one another; that
they describe what the kernels write is for tests running the kernels to show.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.layout_mapping import (
    fan,
    layer_group_of,
    map_buffers,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.page_layout import (
    GRID_DIMENSIONS,
    BufferDesc,
    PageLayout,
    RankTopology,
    Tensor,
    grid_place,
    held,
)

# (pool group, pool index, layer group): layer groups sharing a pool group draw slots of their
# own, so one page of each holds that group's buffers alone.
Slot = Tuple[int, int, int]


def _slot(groups: Dict[Tuple[int, int], int], buffer: BufferDesc) -> Slot:
    return buffer.pool_group, buffer.pool_index, groups[buffer.pool_group, buffer.layer]


def _slots(page: PageLayout, groups: Dict[Tuple[int, int], int]) -> Dict[Slot, np.ndarray]:
    """An empty slot of each pool for each layer group drawing from it, -1 at every byte."""
    return {
        _slot(groups, b): np.full(page.pool_slot_bytes[b.pool_group][b.pool_index], -1, np.int64)
        for b in page.buffers
    }


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

    def memory(self, page: PageLayout) -> Dict[Slot, np.ndarray]:
        """One slot of each pool per layer group drawing from it, holding element ids.

        Each byte holds the id of the element there, -1 where none lies.
        """
        groups = layer_group_of(page)
        mem = _slots(page, groups)
        for buffer in page.buffers:
            for tensor in buffer.tensors:
                offsets, ids = self.of(page, buffer, tensor)
                mem[_slot(groups, buffer)][offsets] = ids
        return mem


def _model_layer(page: PageLayout, groups: Dict[Tuple[int, int], int], buffer: BufferDesc) -> int:
    group = page.layer_groups[groups[buffer.pool_group, buffer.layer]]
    return group.model_layers[group.layers.index(buffer.layer)]


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
    source_groups = [layer_group_of(page) for page in sources]
    sent = {(b.layer, b.role) for page in sources for b in page.buffers if b.transferred}
    sent_models = {
        (_model_layer(page, groups, b), b.role)
        for page, groups in zip(sources, source_groups)
        for b in page.buffers
        if b.transferred
    }
    held_models = {model for model, _ in sent_models}
    for dst in destinations:
        dst_groups = layer_group_of(dst)
        for buffer in dst.buffers:
            model = _model_layer(dst, dst_groups, buffer)
            if (
                buffer.transferred
                and model in held_models
                and (model, buffer.role) not in sent_models
            ):
                assert buffer.role in dst.request_rebuilt_roles, (
                    f"{dst.topology} buffer ({buffer.layer}, {buffer.role}) of model layer {model}: "
                    "no source sends it, and the destination does not rebuild it"
                )
        got = _slots(dst, dst_groups)
        writes = {pool: np.zeros(arr.size, dtype=np.int64) for pool, arr in got.items()}
        for src, memory, groups in zip(sources, source_memory, source_groups):
            maps = map_buffers(src, dst)
            if maps:
                assert paired(src.topology, dst.topology), (src.topology, dst.topology)
            for runs in maps:
                src_pool = _slot(groups, runs.src)
                dst_pool = _slot(dst_groups, runs.dst)
                for s, d, n in zip(runs.src_offsets, runs.dst_offsets, runs.sizes):
                    assert n > 0 and 0 <= s and s + n <= memory[src_pool].size, (src_pool, s, n)
                    assert 0 <= d and d + n <= got[dst_pool].size, (dst.topology, dst_pool, d, n)
                    got[dst_pool][d : d + n] = memory[src_pool][s : s + n]
                    writes[dst_pool][d : d + n] += 1
        expected = {pool: np.zeros(arr.size, dtype=bool) for pool, arr in got.items()}
        for buffer in dst.buffers:
            if not buffer.transferred or (buffer.layer, buffer.role) not in sent:
                continue
            pool = _slot(dst_groups, buffer)
            for tensor in buffer.tensors:
                offsets, ids = elements.of(dst, buffer, tensor)
                expected[pool][offsets] = True
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
        for pool, arr in writes.items():
            stray = np.flatnonzero((arr != 0) & ~expected[pool])
            assert not stray.size, (
                f"{dst.topology} pool {pool} byte {stray[0]}: written, but no transferred tensor "
                "holds it"
            )
