# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GMEM coordinate resources for scale factors A and B.

These resources publish the coordinate dictionaries consumed by SmemSf
resources. They do not allocate memory; lifetime is tied to the task resource
instance and the only persistent state is cached tile metadata such as
``tile_expert_idx``. The SMEM resources own the actual staging buffers and
pipeline waits.

Typical use:
  ``coords = gmem_sfb.compute_sfb_coords_loop(); smem_sfb.load_sfb_tile(*coords)``

Invariants and special cases:
  - coordinate producers refresh the expert id for the current token tile.
  - no-route weight SFs fold experts into the outer row-tile coordinate.
  - no-route activation SFs use expanded token rows and no expert coordinate.
  - ``has_cast_a`` makes SFA address the per-expert MXFP4 A scale tile.
  - ``is_swap_ab`` changes which tile coordinate names the activation token
    dimension; SFB routed activations use the N tile in swapAB mode.
  - routed SFB may publish a tile-row coordinate while the route map supplies
    the final row address inside SmemSfLdgstsResource.

Known limitations: only the generated layouts represented by BatchedGemmConfig
are supported. Invalid dtype/routing combinations are rejected during config
validation, not by these coordinate resources.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
)

from .batched_gemm_config import (
    BatchedGemmConfig,
    BatchMode,
)

Constexpr = cutlass.Constexpr


@dataclass(kw_only=True)
class GmemSfAResource(MemoryResource):
    """GMEM coordinate source for scale-factor A.

    Args:
        cfg: Compile-time BatchedGemmConfig.
        tile_idx_view: Optional array view mapping token tiles to experts.
        problem_m_tiles: Number of M tiles per expert for generated R128c4 SFA.

    Returns ``coord_sfa_k`` and ``coord_sfa_mn``. Normal block-scaled A uses
    ``coord_sfa_mn = tile_coord_m``. The generated R128c4 SFA TensorMap:
    swapAB weight A collapses expert selection into the
    outer row-tile coordinate, so it uses
    ``expert_idx * problem_m_tiles + tile_coord_m``. Non-swapAB activation A
    uses expanded token rows.
    """

    cfg: Constexpr[BatchedGemmConfig]
    tile_idx_view: Any = None
    problem_m_tiles: Any = None
    tile_idx_ptr: Any = None
    tile_expert_idx: Any = None
    coord_sfa_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_sfa_mn: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self) -> None:
        self.coord_sfa_k = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="K tile coordinate for SFA."
        )
        self.coord_sfa_mn = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="M/N tile coordinate for SFA."
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_coords_state(self, stage_info: StageInfo) -> None:
        del stage_info
        self.tile_expert_idx = Int32(0)

    @cute.jit
    def _sfa_mn(self, stage_info: StageInfo) -> Int32:
        """Compute the SFA M/N coordinate and cache the per-tile expert index."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        expert_idx = Int32(0)
        if cutlass.const_expr(self.cfg.is_swap_ab):
            if cutlass.const_expr(self.tile_idx_view is not None):
                if cutlass.const_expr(self.cfg.is_swap_ab):
                    token_tile = tile_coord_n
                else:
                    token_tile = tile_coord_m
                expert_idx = self.tile_idx_view.load(idx=token_tile, vector_size=1)[0]
        self.tile_expert_idx = expert_idx

        if cutlass.const_expr(self.cfg.is_swap_ab):
            problem_m_tiles = self.problem_m_tiles
            if cutlass.const_expr(problem_m_tiles is None):
                problem_m_tiles = Int32(1)
            coord_sfa_mn = expert_idx * problem_m_tiles + tile_coord_m
        else:
            coord_sfa_mn = tile_coord_m
        return coord_sfa_mn

    # Head vs loop differ only in the K coordinate: the head call uses its
    # compile-time ``prefetch_idx`` (prologue prefetch ordinal); body stages use the
    # running ``loop_offset``.
    @consumer_work(returns=(coord_sfa_k, coord_sfa_mn))
    @cute.jit
    def compute_sfa_coords_head(
        self, stage_info: StageInfo, *, prefetch_idx: cutlass.Constexpr[int]
    ) -> tuple[Int32, Int32]:
        coord_sfa_mn = self._sfa_mn(stage_info)
        return Int32(prefetch_idx), coord_sfa_mn

    @consumer_work(returns=(coord_sfa_k, coord_sfa_mn))
    @cute.jit
    def compute_sfa_coords_loop(self, stage_info: StageInfo) -> tuple[Int32, Int32]:
        coord_sfa_mn = self._sfa_mn(stage_info)
        return stage_info.loop_offset, coord_sfa_mn


@dataclass(kw_only=True)
class GmemSfBResource(MemoryResource):
    """GMEM coordinate source for scale-factor B.

    Args:
        cfg: Compile-time BatchedGemmConfig.
        tile_idx_view: Optional array view mapping token tiles to experts.
        problem_n_tiles: Number of N tiles per expert for generated R128c4 SFB.

    Returns ``coord_sfb_k`` and ``coord_sfb_mn``. The head call uses
    ``prefetch_idx - 1`` for ``coord_sfb_k`` to match the generated prologue
    schedule; body stages use ``loop_offset``. The generated R128c4 SFB
    TensorMap collapses expert selection into the outer row-tile coordinate
    for non-swapAB weight B. SwapAB activation B uses expanded token rows;
    routed SFB widens ``coord_sfb_mn`` to a row offset.
    """

    cfg: Constexpr[BatchedGemmConfig]
    tile_idx_view: Any = None
    problem_n_tiles: Any = None
    tile_idx_ptr: Any = None
    tile_idx_arr: Any = None
    tile_expert_idx: Any = None
    tile_sfb_mn: Any = None
    coord_sfb_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_sfb_mn: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self) -> None:
        self.coord_sfb_k = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="K tile coordinate for SFB."
        )
        self.coord_sfb_mn = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="M/N tile coordinate for SFB."
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_coords_state(self, stage_info: StageInfo) -> None:
        del stage_info
        self.tile_expert_idx = Int32(0)
        self.tile_sfb_mn = Int32(0)
        if cutlass.const_expr(self.tile_idx_ptr is not None):
            self.tile_idx_arr = cutlass.Array(
                cutlass.inttoptr(self.tile_idx_ptr, mem_space=1, dtype=cutlass.Int32),
                dtype=cutlass.Int32,
            )

    @cute.jit
    def _sfb_mn(self, stage_info: StageInfo) -> Int32:
        """Compute the SFB M/N coordinate and cache the per-tile expert index."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        self.tile_expert_idx = Int32(0)
        if cutlass.const_expr(not self.cfg.is_swap_ab):
            if cutlass.const_expr(self.tile_idx_view is not None):
                self.tile_expert_idx = self.tile_idx_view.load(
                    idx=tile_coord_m, vector_size=1
                )[0]
            elif cutlass.const_expr(self.tile_idx_ptr is not None):
                if cutlass.const_expr(self.cfg.batch_mode == int(BatchMode.BATCH_N)):
                    self.tile_expert_idx = (
                        self.tile_idx_arr.subview(tile_coord_m)
                    ).load()
                else:
                    self.tile_expert_idx = (
                        self.tile_idx_arr.subview(tile_coord_n)
                    ).load()
        if cutlass.const_expr(self.cfg.has_routed_sfs and self.cfg.is_swap_ab):
            coord_sfb_mn = tile_coord_n * Int32(self.cfg.tile_n)
        elif cutlass.const_expr(not self.cfg.is_swap_ab):
            problem_n_tiles = self.problem_n_tiles
            if cutlass.const_expr(problem_n_tiles is None):
                problem_n_tiles = Int32(1)
            coord_sfb_mn = self.tile_expert_idx * problem_n_tiles + tile_coord_n
        else:
            coord_sfb_mn = tile_coord_n
        return coord_sfb_mn

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_tile_state(self, stage_info: StageInfo) -> None:
        """Cache per-work-tile SFB metadata before the K loop."""
        self.tile_sfb_mn = self._sfb_mn(stage_info)

    # Head vs loop differ only in the K coordinate: the head call uses
    # ``prefetch_idx - 1`` to match the generated prologue schedule; body stages use
    # the running ``loop_offset``.
    @consumer_work(returns=(coord_sfb_k, coord_sfb_mn))
    @cute.jit
    def compute_sfb_coords_head(
        self, stage_info: StageInfo, *, prefetch_idx: cutlass.Constexpr[int]
    ) -> tuple[Int32, Int32]:
        del stage_info
        return Int32(prefetch_idx - 1), self.tile_sfb_mn

    @consumer_work(returns=(coord_sfb_k, coord_sfb_mn))
    @cute.jit
    def compute_sfb_coords_loop(self, stage_info: StageInfo) -> tuple[Int32, Int32]:
        return stage_info.loop_offset, self.tile_sfb_mn
