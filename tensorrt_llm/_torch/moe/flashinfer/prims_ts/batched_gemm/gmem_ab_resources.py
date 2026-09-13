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

"""GMEM coordinate resources for BatchedGemm operands A and B.

The resources publish per-stage coordinate dictionaries consumed by SMEM load
resources. ``tile_idx_view`` maps each token tile to an expert and
``mn_limit_view`` publishes the TRT-LLM Gen absolute end-row limit for that
token tile. The head schedule stage derives the local valid row count and
caches the metadata on the resource so all K-loop stages for the same work
tile use identical values.

Coordinate contract:
  - ``route_act == NONE``: activations are stacked in L=0 and weights are
    addressed by ``expert_idx``.
  - ``route_act == LDGSTS``: the GMEM A/B coordinates remain stacked for the
    activation operand; SmemGatherResource performs the route-map gather.
  - ``route_act == TMA``: TMA gather4 routes activation rows, so the
    activation operand keeps ``coord_l == 0`` and the expert id is still
    returned for consumers that need tile metadata.
  - ``is_swap_ab`` swaps the operand roles: A becomes per-expert weights
    (``coord_a_l = expert_idx``) and B becomes stacked activations
    (``coord_b_l = 0``).
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
)


def nonnegative_div(value, divisor: int):
    """Divide an index by a positive constant without signed correction code."""

    if divisor > 0 and divisor & (divisor - 1) == 0:
        return value >> Int32(divisor.bit_length() - 1)
    return value // Int32(divisor)


def nonnegative_mod(value, divisor: int):
    """Modulo an index by a positive constant without signed correction code."""

    if divisor > 0 and divisor & (divisor - 1) == 0:
        return value & Int32(divisor - 1)
    return value % Int32(divisor)


Constexpr = cutlass.Constexpr


@dataclass(kw_only=True)
class GmemAResource(MemoryResource):
    """GMEM coordinate source for operand A.

    Returns ``coord_a_k``, ``coord_a_mn``, ``coord_a_l``, ``expert_idx``, and
    ``mn_limit``. Non-swapAB A is the activation operand: ``NONE`` and
    ``LDGSTS`` use stacked activations with ``coord_a_l = 0``; ``TMA`` also uses
    ``coord_a_l = 0`` because gather4 computes routed row addresses. In swapAB
    mode A is the weight operand and uses ``coord_a_l = expert_idx``.
    """

    cfg: Constexpr[BatchedGemmConfig]
    tile_idx_view: Any = None  # maps tile coord → expert idx
    mn_limit_view: Any = None  # maps tile coord → absolute token end-row limit
    rows_per_expert: Any = None  # for TMA route: tokens per expert
    tile_expert_idx: Any = None
    tile_mn_limit: Any = None
    coord_a_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_a_mn: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_a_l: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    expert_idx: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    mn_limit: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self) -> None:
        self.coord_a_k = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="K coordinate for operand A."
        )
        self.coord_a_mn = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="M/N coordinate for operand A."
        )
        self.coord_a_l = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="L coordinate for operand A."
        )
        self.expert_idx = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="Expert index for this tile."
        )
        self.mn_limit = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="Local valid token rows."
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_coords_state(self, stage_info: StageInfo) -> None:
        del stage_info
        self.tile_expert_idx = Int32(0)
        self.tile_mn_limit = Int32(0)

    @cute.jit
    def _load_tile_metadata(self, stage_info: StageInfo) -> None:
        """Cache the per-tile expert index and token limit (head phase only)."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        if cutlass.const_expr(self.cfg.is_swap_ab):
            token_tile = tile_coord_n
            token_rows = self.cfg.tile_n
        else:
            token_tile = tile_coord_m
            token_rows = self.cfg.tile_m
        self.tile_expert_idx = self.tile_idx_view.load(idx=token_tile, vector_size=1)[0]
        self.tile_mn_limit = self._local_tile_limit(
            self.mn_limit_view.load(idx=token_tile, vector_size=1)[0],
            token_tile,
            token_rows,
        )

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @cute.jit
    def _a_coords(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Int32, Int32]:
        """Build the A coordinates from the cached per-tile metadata."""
        tile_coord_m, _, _ = stage_info.work_tile.tile_idx
        coord_k = stage_info.loop_offset * Int32(self.cfg.tile_k)
        expert_idx = self.tile_expert_idx

        coord_mn = tile_coord_m * Int32(self.cfg.tile_m)
        if cutlass.const_expr(self.cfg.is_swap_ab):
            # SwapAB: A=weights (per-expert), coord_a_l = expert_idx
            coord_l = expert_idx
        elif cutlass.const_expr(self.cfg.has_tma_route):
            # TMA gather4: coord_a_l unused (gather4 addresses rows in 2D)
            coord_l = Int32(0)
        else:
            # Non-swapAB: A=activations (stacked), coord_a_l = 0
            coord_l = Int32(0)

        return coord_k, coord_mn, coord_l, expert_idx, self.tile_mn_limit

    # Head and loop are distinct work methods so the schedule never needs to
    # pass a stage tag: the head call caches per-tile metadata, the loop call
    # reuses it.
    @consumer_work(returns=(coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit))
    @cute.jit
    def compute_a_coords_head(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Int32, Int32]:
        self._load_tile_metadata(stage_info)
        return self._a_coords(stage_info)

    @consumer_work(returns=(coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit))
    @cute.jit
    def compute_a_coords_loop(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Int32, Int32]:
        return self._a_coords(stage_info)


@dataclass(kw_only=True)
class GmemBResource(MemoryResource):
    """GMEM coordinate source for operand B.

    Returns ``coord_b_k``, ``coord_b_mn``, ``coord_b_l``, and ``mn_limit``.
    Non-swapAB B is the per-expert weight operand and uses
    ``coord_b_l = expert_idx``. In swapAB mode B is stacked activations and uses
    ``coord_b_l = 0``; any LDGSTS activation routing is handled by the matching
    SmemGatherResource rather than by changing the GMEM L coordinate.
    """

    cfg: Constexpr[BatchedGemmConfig]
    tile_idx_view: Any = None  # make_array_view of tile_idx tensor
    mn_limit_view: Any = None  # maps tile coord → absolute token end-row limit
    tile_expert_idx: Any = None
    tile_mn_limit: Any = None
    coord_b_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_b_mn: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    coord_b_l: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    mn_limit: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self) -> None:
        self.coord_b_k = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="K coordinate for operand B."
        )
        self.coord_b_mn = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="M/N coordinate for operand B."
        )
        self.coord_b_l = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="L coordinate for operand B."
        )
        self.mn_limit = TaskLocalVariable(
            dtype=Int32, default=Int32(0), docs="Local valid token rows."
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_coords_state(self, stage_info: StageInfo) -> None:
        del stage_info
        self.tile_expert_idx = Int32(0)
        self.tile_mn_limit = Int32(0)

    @cute.jit
    def _load_tile_metadata(self, stage_info: StageInfo) -> None:
        """Cache the per-tile expert index and token limit (head phase only)."""
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        if cutlass.const_expr(self.cfg.is_swap_ab):
            token_tile = tile_coord_n
            token_rows = self.cfg.tile_n
        else:
            token_tile = tile_coord_m
            token_rows = self.cfg.tile_m
        self.tile_expert_idx = self.tile_idx_view.load(idx=token_tile, vector_size=1)[0]
        self.tile_mn_limit = self._local_tile_limit(
            self.mn_limit_view.load(idx=token_tile, vector_size=1)[0],
            token_tile,
            token_rows,
        )

    @cute.jit
    def _local_tile_limit(self, raw_limit, token_tile, tile_rows):
        """Convert TRT-LLM Gen absolute end-row limit to a local row count."""
        local_limit = raw_limit - token_tile * Int32(tile_rows)
        if local_limit < Int32(0):
            local_limit = Int32(0)
        if local_limit > Int32(tile_rows):
            local_limit = Int32(tile_rows)
        return local_limit

    @cute.jit
    def _b_coords(self, stage_info: StageInfo) -> tuple[Int32, Int32, Int32, Int32]:
        """Build the B coordinates from the cached per-tile metadata."""
        _, tile_coord_n, _ = stage_info.work_tile.tile_idx
        coord_k = stage_info.loop_offset * Int32(self.cfg.tile_k)
        expert_idx = self.tile_expert_idx

        coord_mn = tile_coord_n * Int32(self.cfg.tile_n)
        if cutlass.const_expr(self.cfg.is_swap_ab):
            # SwapAB: B=activations (stacked), coord_b_l = 0
            coord_l = Int32(0)
        else:
            # Non-swapAB: B=weights (per-expert), coord_b_l = expert_idx
            coord_l = expert_idx

        return coord_k, coord_mn, coord_l, self.tile_mn_limit

    # Head and loop are distinct work methods so the schedule never needs to
    # pass a stage tag: the head call caches per-tile metadata, the loop call
    # reuses it.
    @consumer_work(returns=(coord_b_k, coord_b_mn, coord_b_l, mn_limit))
    @cute.jit
    def compute_b_coords_head(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Int32]:
        self._load_tile_metadata(stage_info)
        return self._b_coords(stage_info)

    @consumer_work(returns=(coord_b_k, coord_b_mn, coord_b_l, mn_limit))
    @cute.jit
    def compute_b_coords_prefetch(
        self,
        stage_info: StageInfo,
        *,
        prefetch_idx: cutlass.Constexpr[int],
    ) -> tuple[Int32, Int32, Int32, Int32]:
        """Build a prologue coordinate without relying on a loop offset."""
        _, tile_coord_n, _ = stage_info.work_tile.tile_idx
        coord_k = Int32(prefetch_idx * self.cfg.tile_k)
        coord_mn = tile_coord_n * Int32(self.cfg.tile_n)
        coord_l = Int32(0) if self.cfg.is_swap_ab else self.tile_expert_idx
        return coord_k, coord_mn, coord_l, self.tile_mn_limit

    @consumer_work(returns=(coord_b_k, coord_b_mn, coord_b_l, mn_limit))
    @cute.jit
    def compute_b_coords_loop(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32, Int32]:
        return self._b_coords(stage_info)
