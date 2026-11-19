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
two launches)."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import Int32
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo


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
