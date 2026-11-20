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
"""Work distribution of the attention kernel inside a launch: the static tile scheduler over (Q tile pair, head,
batch) and the split-KV scheduler that walks the KV pieces of the units of the last partial wave (launch.py plans the
two launches). A launch with valid prefix lengths distributes the live units, those with a query row below the length
of their batch element: its main launch walks them through LiveTileScheduler, in the order of the static scheduler,
and its split launch runs the KV pieces that hold a valid key."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import Int32
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo

from ..valid_len import valid_len


def thread_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class TileSchedParams:
    def __init__(self, problem_shape_mhb, n_valid):
        self.problem_shape_mhb = problem_shape_mhb
        self.n_valid = n_valid

    def _objs(self):
        """The fields in the order the constructor takes them."""
        return [self.problem_shape_mhb, self.n_valid]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in self._objs():
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(self._objs(), self._values_pos):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return type(self)(*obj_list)


class TileScheduler:
    def __init__(self, params, current_work_linear_idx, grid_shape):
        self._params = params
        self._grid_shape = grid_shape
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mhb = cute.make_layout(params.problem_shape_mhb)
        self._num_blocks = self._block_count(params)
        self.num_persistent_sm = cute.size(grid_shape)

    def _block_count(self, params):
        """The work items of the launch."""
        return params.n_valid

    def get_current_work(self):
        is_valid = self._current_work_linear_idx < self._num_blocks
        blk_coord = self._problem_shape_mhb.get_hier_coord(self._current_work_linear_idx)
        # (m_block, 0, ((h_r=0, h), b))
        cur_tile_coord = (blk_coord[0], 0, (blk_coord[1], blk_coord[2]))
        return WorkTileInfo(cur_tile_coord, is_valid)

    def advance_to_next_work(self):
        self._current_work_linear_idx += self.num_persistent_sm

    def __extract_mlir_values__(self):
        return extract_mlir_values(self._current_work_linear_idx)

    def __new_from_mlir_values__(self, values):
        """This scheduler at another work position."""
        assert len(values) == 1
        idx = new_from_mlir_values(self._current_work_linear_idx, values)
        return type(self)(self._params, idx, self._grid_shape)


class LiveTileScheduler(TileScheduler):
    """The scheduler of a main launch with valid prefix lengths: the CTAs share the live units of the launch as the
    static scheduler shares all its units, the CTA at position c of the grid taking the live units c, c + grid, ...
    in the order (batch, head, Q block pair). Batch element b of n valid tokens has the live Q block pairs
    m < ceil(n / 256), rounded up to a multiple of `group` (2 where CTA pairs share K/V: both units of a pair are
    then live together), in every head; the launch holds the first n_valid units of the static order, so the last
    head it reaches may hold fewer. lens: (the global address of the lengths, the padded sequence length, group).

    The state is the live index of the current unit with the batch element that holds it and the live index of
    that element's first unit; start() and advance_to_next_work() move the element forward to the current unit.
    The work tile carries the valid length of its batch element after the coordinates of the unit."""

    def __init__(self, params, current_work_linear_idx, grid_shape, lens, batch=None, base=None):
        super().__init__(params, current_work_linear_idx, grid_shape)
        self._lens = lens
        # the lengths are read as data that no thread writes (valid_len); own_loads() turns that off
        self._invariant = True
        self._batch = Int32(0) if batch is None else batch
        self._base = Int32(0) if base is None else base

    def _valid_len(self, b):
        """The valid length of batch element b (the last element for a b past it)."""
        lens_addr, seqlen, _ = self._lens
        return valid_len(
            lens_addr,
            cutlass.min(b, self._params.problem_shape_mhb[2] - 1),
            seqlen,
            self._invariant,
        )

    def _live(self, b):
        """(live Q block pairs per head, heads the launch holds whole, live pairs of the head it holds in part) of
        batch element b."""
        m_all, h_all, _ = self._params.problem_shape_mhb
        group = self._lens[2]
        n = self._valid_len(b)
        m_live = cutlass.min((n + (256 * group - 1)) // (256 * group) * group, m_all)
        # the static order cuts the launch after n_valid units: whole heads up to row_cut, m_cut pairs of that head
        row_cut = self._params.n_valid // m_all
        m_cut = self._params.n_valid - row_cut * m_all
        heads = cutlass.max(cutlass.min(row_cut - b * h_all, h_all), Int32(0))
        m_part = cutlass.select_(heads < h_all, cutlass.min(m_live, m_cut), Int32(0))
        m_part = cutlass.select_(row_cut >= b * h_all, m_part, Int32(0))
        return m_live, heads, m_part

    @cute.jit
    def _locate(self):
        """Moves the batch element forward until it holds the current live index (or past the last element)."""
        b, base = self._batch, self._base
        n_batch = self._params.problem_shape_mhb[2]
        m_live, heads, m_part = self._live(b)
        while (b < n_batch) & (self._current_work_linear_idx >= base + heads * m_live + m_part):
            base = base + heads * m_live + m_part
            b = b + 1
            m_live, heads, m_part = self._live(b)
        self._batch, self._base = b, base

    def start(self):
        """The scheduler at the first work of the CTA."""
        self._locate()
        return self

    def get_current_work(self):
        is_valid = self._batch < self._params.problem_shape_mhb[2]
        m_live, heads, _ = self._live(self._batch)
        local = self._current_work_linear_idx - self._base
        h = local // cutlass.max(m_live, Int32(1))
        m = local - h * m_live
        # past the whole heads: the head the launch holds in part
        part = h >= heads
        h = cutlass.select_(part, heads, h)
        m = cutlass.select_(part, local - heads * m_live, m)
        # (m_block, 0, ((h_r=0, h), b), valid length of b)
        cur_tile_coord = (m, 0, (h, self._batch), self._valid_len(self._batch))
        return WorkTileInfo(cur_tile_coord, is_valid)

    def advance_to_next_work(self):
        self._current_work_linear_idx += self.num_persistent_sm
        self._locate()

    def __extract_mlir_values__(self):
        return extract_mlir_values([self._current_work_linear_idx, self._batch, self._base])

    def __new_from_mlir_values__(self, values):
        """This scheduler at another work position."""
        assert len(values) == 3
        idx, batch, base = new_from_mlir_values(
            [self._current_work_linear_idx, self._batch, self._base], values
        )
        new = type(self)(self._params, idx, self._grid_shape, self._lens, batch, base)
        new._invariant = self._invariant
        return new

    def own_loads(self):
        """This scheduler at the same work, reading every length with a load of its own."""
        new = self.__new_from_mlir_values__(self.__extract_mlir_values__())
        new._invariant = False
        return new


class SplitTileSchedParams(TileSchedParams):
    """Work items: n_full whole (Q-block pair, head, batch) units, then the remaining (tail) units each split into
    n_parts contiguous KV ranges, so the last wave keeps every SM busy; the split parts are merged in the epilogue."""

    def __init__(self, problem_shape_mhb, n_full, n_parts, n_kv):
        self.problem_shape_mhb = problem_shape_mhb
        self.n_full = n_full
        self.n_parts = n_parts
        self.n_kv = n_kv

    def _objs(self):
        return [self.problem_shape_mhb, self.n_full, self.n_parts, self.n_kv]


class SplitTileScheduler(TileScheduler):
    """lens: None, or (the global address of the valid prefix lengths, the padded sequence length): a unit of n valid
    tokens then has the KV tiles below ceil(n / 128), its live pieces are those that hold one of them (they come
    first, and take the place of the unit's piece count in the work tile), its last live piece ends at the last of
    them, and the work of a piece that is not live, or of a unit without a query row below n, is not valid. The
    work tile carries n after the coordinates."""

    def __init__(self, params, current_work_linear_idx, grid_shape, lens=None):
        super().__init__(params, current_work_linear_idx, grid_shape)
        self._lens = lens

    def _block_count(self, params):
        """The whole units, then n_parts KV pieces of each remaining unit."""
        num_units = cute.size(self._problem_shape_mhb)
        return params.n_full + (num_units - params.n_full) * params.n_parts

    def get_current_work(self):
        idx = Int32(self._current_work_linear_idx)
        is_valid = idx < self._num_blocks
        n_full = Int32(self._params.n_full)
        n_parts = Int32(self._params.n_parts)
        n_kv = Int32(self._params.n_kv)
        is_split = idx >= n_full
        rel = cutlass.select_(is_split, idx - n_full, Int32(0))
        r = rel // n_parts
        part = cutlass.select_(is_split, rel - r * n_parts, Int32(0))
        nparts = cutlass.select_(is_split, n_parts, Int32(1))
        unit = cutlass.select_(is_split, n_full + r, idx)
        kv_begin = (part * n_kv) // nparts
        kv_end = ((part + 1) * n_kv) // nparts
        slot_base = r * n_parts
        blk_coord = self._problem_shape_mhb.get_hier_coord(unit)
        if self._lens is not None:
            lens_addr, seqlen = self._lens
            n = valid_len(
                lens_addr, cutlass.min(blk_coord[2], self._params.problem_shape_mhb[2] - 1), seqlen
            )
            n_kv_live = (n + 127) // 128
            is_valid = is_valid & (blk_coord[0] * 256 < n) & (kv_begin < n_kv_live)
            kv_end = cutlass.min(kv_end, n_kv_live)
            # the pieces with kv_begin below n_kv_live
            nparts = (n_kv_live * nparts + n_kv - 1) // n_kv
            cur_tile_coord = (
                blk_coord[0],
                0,
                (blk_coord[1], blk_coord[2]),
                kv_begin,
                kv_end - kv_begin,
                part,
                nparts,
                slot_base,
                n,
            )
            return WorkTileInfo(cur_tile_coord, is_valid)
        # (m_block, 0, (h, b), kv_begin, n_local_kv, part, nparts, slot_base): the last three for the epilogue
        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
            kv_begin,
            kv_end - kv_begin,
            part,
            nparts,
            slot_base,
        )
        return WorkTileInfo(cur_tile_coord, is_valid)

    def __new_from_mlir_values__(self, values):
        """This scheduler at another work position."""
        assert len(values) == 1
        idx = new_from_mlir_values(self._current_work_linear_idx, values)
        return type(self)(self._params, idx, self._grid_shape, self._lens)
